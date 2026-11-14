"""Quantization overlay: see what a model loses when its weights are stored in 8, 4 or fewer bits.

Like the LoRA overlay it sits on the WeightStore, so it works with every model plugin and nothing is
re-saved: each weight matrix is rounded when the engine asks for it.

    scale  = largest |w| in the row / (2^(bits-1) - 1)        one scale per output row
    w_q    = round(w / scale) · scale                         only 2^bits different values per row

This is plain round-to-nearest ("RTN") with a symmetric per-row scale: the baseline every cleverer scheme
(GPTQ, AWQ, k-quants) is measured against. Only the 2-D weight matrices inside blocks are rounded; embeddings,
norms and biases stay as they are, which is also what most real schemes do.
"""
from __future__ import annotations

import torch

from .weights import _LAYER_RE

MEASURE_LIMIT = 8 * 2**30          # measure the rounding error of every matrix only for models up to this size


def quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    wf = w.float()
    if bits <= 1:                                       # 1 bit: sign × the row's mean size
        return (wf.sign() * wf.abs().mean(dim=-1, keepdim=True)).to(w.dtype)
    qmax = 2 ** (bits - 1) - 1
    scale = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    return (torch.round(wf / scale).clamp(-qmax - 1, qmax) * scale).to(w.dtype)


class QuantOverlay:
    def __init__(self, bits: int = 4, layers: str = "all", layer_bits: dict[int, int] | None = None):
        if bits is not None and not 1 <= int(bits) <= 16:
            raise ValueError("bits must be between 1 and 16.")
        self.bits = int(bits) if bits is not None else 4
        self.layers_spec = str(layers or "all").strip()
        self.layer_bits: dict[int, int] = {int(k): int(v) for k, v in (layer_bits or {}).items() if v and 1 <= int(v) <= 16}
        self._allowed: set[int] | None = None
        if self.layers_spec.lower() not in ("all", "*", ""):
            from .flow import parse_index_spec
            self._allowed = set(parse_index_spec(self.layers_spec, 10_000))
        self.layers: list[dict] = []
        self.n_weights = 0
        self.n_params = 0
        self.measured = False

    def bits_for_layer(self, layer: int | None) -> int | None:
        if layer is None:
            return None
        if layer in self.layer_bits:
            return self.layer_bits[layer]
        if self._allowed is not None:
            return self.bits if layer in self._allowed else None
        return self.bits

    @staticmethod
    def wants(name: str, ndim: int) -> bool:
        return ndim == 2 and name.endswith(".weight") and "norm" not in name and bool(_LAYER_RE.search(name))

    def apply(self, name: str, w: torch.Tensor) -> torch.Tensor:
        if not self.wants(name, w.ndim):
            return w
        m = _LAYER_RE.search(name)
        layer = int(m.group(1)) if m else None
        b = self.bits_for_layer(layer)
        return quantize(w, b) if b is not None else w

    def bind(self, stores: dict) -> dict:
        """Attach to the model's weight stores and (for models that fit) measure the rounding error per block."""
        self.layers, self.n_weights, self.n_params = [], 0, 0
        per_layer: dict[int, list[float]] = {}
        total = sum(getattr(s, "total_bytes", 0) for s in stores.values() if s is not None)
        self.measured = total <= MEASURE_LIMIT
        for store in stores.values():
            if store is None:
                continue
            store._quant = self
            for name in list(store._all_names()):
                shape = store.shape(name)
                if not shape or not self.wants(name, len(shape)):
                    continue
                layer = store._layer_of(name)
                b = self.bits_for_layer(layer)
                if b is None:
                    continue
                self.n_weights += 1
                self.n_params += shape[0] * shape[1]
                if self.measured:
                    w = store.compact(name).float()
                    d = per_layer.setdefault(layer, [0.0, 0.0, b])
                    d[0] += float((quantize(w, b) - w).pow(2).sum())
                    d[1] += float(w.pow(2).sum())
        for layer in sorted(k for k in per_layer if k is not None):
            e, b, lbits = per_layer[layer]
            self.layers.append({"layer": layer, "bits": lbits, "relative_error": round((e / max(b, 1e-12)) ** 0.5, 5)})
        mx = max((x["relative_error"] for x in self.layers), default=0.0) or 1.0
        for x in self.layers:
            x["heat"] = round(x["relative_error"] / mx, 4)
        return self.summary()

    def summary(self) -> dict:
        return {"active": True, "bits": self.bits, "layers_spec": self.layers_spec,
                "layer_bits": self.layer_bits, "n_weights": self.n_weights, "n_params": self.n_params,
                "size_mb_16bit": round(self.n_params * 16 / 8 / 2**20, 1),
                "size_mb": round(self.n_params * self.bits / 8 / 2**20, 1),
                "measured": self.measured, "layers": self.layers,
                "mean_error": round(sum(x["relative_error"] for x in self.layers) / max(1, len(self.layers)), 5)}


def iter_compare(loaded, plan, bits_list: list[int], prefix=None, sources=None):
    """Run the same flow at full precision and at each bit width. Yields the table as it fills in.

    For every setting: the text it generates on its own, and — forced along the full-precision text, so the
    comparison is token for token — how far its next-token distributions drift (KL divergence) and how often
    its top choice still agrees."""
    from .llm import Stepper
    ad = loaded.adapter
    before = loaded.quant.bits if loaded.quant is not None else None
    rows, base = [], None

    def stepper():
        st = Stepper(ad, plan, prefix=prefix)
        st.ctx.sources = sources or {}
        return st

    try:
        for n, bits in enumerate([None] + [b for b in bits_list if b]):
            loaded.set_quant(bits, measure=False)
            st = stepper()
            while st.step_token():
                pass
            gen = st.seq[st.n_in:]
            row = {"bits": bits, "label": "full precision" if bits is None else f"{bits}-bit",
                   "text": ad.tok.decode(gen, skip_special=True), "tokens": [ad.tok.piece(i) for i in gen]}
            if bits is None:
                base = {"gen": gen, "logp": torch.log_softmax(torch.cat(st.logits_rows), -1), "n_in": st.n_in}
                row.update(kl=0.0, agree=1.0, same_tokens=len(gen))
            else:
                forced = stepper()
                forced.step_token()                                     # the prompt (and prefill)
                while forced.queue:
                    forced.step_token()
                for tid in base["gen"][:-1] if base["gen"] and base["gen"][-1] in forced.eos else base["gen"]:
                    forced.force(tid)
                    forced.step_token()
                logp = torch.log_softmax(torch.cat(forced.logits_rows), -1)
                m = min(logp.shape[0], base["logp"].shape[0])
                a, b = base["logp"][base["n_in"] - 1:m], logp[base["n_in"] - 1:m]     # rows that predict generated tokens
                kl = (a.exp() * (a - b)).sum(-1)
                same = 0
                for x, y in zip(gen, base["gen"]):
                    if x != y:
                        break
                    same += 1
                row.update(kl=round(float(kl.mean()), 4) if len(kl) else 0.0, kl_max=round(float(kl.max()), 4) if len(kl) else 0.0,
                           agree=round(float((a.argmax(-1) == b.argmax(-1)).float().mean()), 4) if len(kl) else 1.0,
                           same_tokens=same)
            rows.append(row)
            yield {"rows": list(rows), "n_base": len(base["gen"]), "_progress": (n + 1) / (1 + len(bits_list))}
    finally:
        loaded.set_quant(before, measure=False)
