"""Ways to adapt a model, beyond LoRA — so they can be compared on the same examples.

All of them freeze almost everything and train a small part. What differs is *which* part:

    lora       a low-rank add-on  W + (α/r)·B·A  on chosen matrices in chosen blocks          (train.LoRAJob)
    full       the chosen matrices themselves, in chosen blocks (plain fine-tuning of a few)   most numbers
    ia3        one scale per output row of the chosen matrices: W ← diag(s)·W                  fewest numbers
    norms      only the two RMSNorm weight vectors of the chosen blocks
    adapter    a new *thin layer* after chosen blocks: x + up(act(down(x))), r numbers wide    new weights
    new_block  a whole *new block* inserted after block i, then trained. It starts as
                 duplicate  a copy of block i                (the model first gets worse, training repairs it)
                 average    the average of block i and i+1
                 identity   a copy of block i whose outputs are zeroed (changes nothing until trained)
                 random     small random numbers

The result is saved under data/loras/<name>/ (adaptation.safetensors + adaptation.json) and attached like a
LoRA: changed matrices show up as ΔW per block in the LoRA / Fine-tune tab, new weights as "added parameters",
and a new block appears as a new box in the flow, at the place it was inserted.
"""
from __future__ import annotations

import json
import math
import random
import time

import torch

from .train import LoRAJob, forward_train, next_token_loss

METHODS = ("lora", "full", "ia3", "norms", "adapter", "new_block")
INITS = ("duplicate", "average", "identity", "random")
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")


# --------------------------------------------------------------------------------------
# Attaching what an adaptation adds (used by the app's registry and by the Python API)
# --------------------------------------------------------------------------------------

def apply_extras(adapter, overlay) -> dict | None:
    """New blocks and adapter layers have weights the base model's files don't contain. Keep them in memory on
    the weight store; a new block becomes a new layer number (the flow runs it where it was inserted)."""
    from .weights import read_json
    folder = overlay.path if overlay.path.is_dir() else overlay.path.parent
    meta = read_json(folder / "adaptation.json")
    if not meta or getattr(adapter, "family", "") != "causal_lm":
        return meta or None
    m, cfg = adapter.model, adapter.cfg
    base = getattr(cfg, "base_layers", cfg.n_layers)
    blocks = meta.get("extra_blocks") or []
    if blocks and int(blocks[0]["layer"]) != base:
        raise ValueError(f"This adaptation adds a block to a model with {blocks[0]['layer']} blocks; the loaded model "
                         f"has {base}.")
    m.W._extra = {k: t.float() for k, t in overlay.raw_tensors.items()
                  if k.startswith("model.layers.") and not m.W.has(k)}
    cfg.base_layers, cfg.n_layers = base, base + len(blocks)
    adapter.extra_blocks = blocks
    return meta


def clear_extras(adapter):
    if adapter is None or getattr(adapter, "family", "") != "causal_lm" or getattr(adapter, "model", None) is None:
        return
    adapter.model.W._extra = {}
    if hasattr(adapter.cfg, "base_layers"):
        adapter.cfg.n_layers = adapter.cfg.base_layers
    adapter.extra_blocks = []


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

class AdaptJob(LoRAJob):
    kind = "lora"            # shares the LoRA slot in the Train tab: one adaptation is trained at a time

    def __init__(self, loaded, tasks, out_dir, name, examples, method: str = "full", after: int = 0,
                 init: str = "duplicate", **kw):
        super().__init__(loaded, tasks, out_dir, name, examples, **kw)
        if method not in METHODS or method == "lora":
            raise ValueError(f"method must be one of {', '.join(METHODS)}")
        if init not in INITS:
            raise ValueError(f"init must be one of {', '.join(INITS)}")
        self.method, self.after, self.init = method, int(after), init
        cfg = loaded.adapter.cfg
        if method == "new_block":
            if getattr(cfg, "base_layers", cfg.n_layers) != cfg.n_layers:
                raise ValueError("A trained block is already attached. Detach it (LoRA / Fine-tune tab) before adding "
                                 "another: blocks are added to the base model one at a time.")
            if not 0 <= self.after < cfg.n_layers:
                raise ValueError(f"Insert after a block between 0 and {cfg.n_layers - 1}.")

    def extra(self):
        return {**super().extra(), "method": self.method, "after": self.after, "init": self.init}

    def build(self, m, cfg):
        """The trainable numbers P, how each weight is formed from them, and what the forward pass needs."""
        from .flow import parse_index_spec
        layers = parse_index_spec(self.layers, cfg.n_layers)
        P: dict[str, torch.Tensor] = {}
        meta = {"method": self.method}
        base = lambda name: (m.norm_w(name) if name.endswith("norm.weight") else m.W(name)).float()     # noqa: E731
        schedule, adapters = list(range(cfg.n_layers)), []

        def part(t):
            return "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"

        if self.method in ("full", "ia3"):
            for l in layers:
                for t in self.targets:
                    name = f"model.layers.{l}.{part(t)}.{t}.weight"
                    shape = m.W.shape(name)
                    if shape:
                        P[name] = (torch.zeros(shape) if self.method == "full" else torch.ones(shape[0])).to(m.device).requires_grad_()
            if self.method == "full":
                W = lambda name: base(name) + P[name] if name in P else base(name)                        # noqa: E731
            else:
                W = lambda name: base(name) * P[name][:, None] if name in P else base(name)               # noqa: E731
            meta.update(layers=layers, targets=list(self.targets))
        elif self.method == "norms":
            for l in layers:
                for n in NORMS:
                    name = f"model.layers.{l}.{n}"
                    P[name] = torch.zeros(m.W.shape(name)).to(m.device).requires_grad_()
            W = lambda name: base(name) + P[name] if name in P else base(name)                            # noqa: E731
            meta.update(layers=layers)
        elif self.method == "adapter":
            d = cfg.hidden
            for l in layers:
                pre = f"model.layers.{l}.adapter."
                P[pre + "down.weight"] = (torch.randn(self.rank, d) / math.sqrt(d)).to(m.device).requires_grad_()
                P[pre + "up.weight"] = torch.zeros(d, self.rank, device=m.device, requires_grad=True)   # starts as "add nothing"
            adapters = layers
            W = lambda name: P[name] if name in P else base(name)                                         # noqa: E731
            meta.update(adapters=layers, width=self.rank)
        else:                                               # new_block
            i, n = self.after, cfg.n_layers
            j = min(i + 1, n - 1)
            for name in [k for k in m.W._all_names() if k.startswith(f"model.layers.{i}.")]:
                t = m.W(name).float().clone()
                if self.init == "average" and j != i:
                    t = (t + m.W(name.replace(f"model.layers.{i}.", f"model.layers.{j}.", 1)).float()) / 2
                elif self.init == "identity" and name.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight")):
                    t = torch.zeros_like(t)
                elif self.init == "random" and t.ndim == 2:
                    t = torch.randn_like(t) * 0.02 * (0.0 if name.endswith(("o_proj.weight", "down_proj.weight")) else 1.0)
                elif self.init == "random" and name.endswith(".bias"):
                    t = torch.zeros_like(t)
                P[name.replace(f"model.layers.{i}.", f"model.layers.{n}.", 1)] = t.to(m.device).requires_grad_()
            schedule = list(range(i + 1)) + [n] + list(range(i + 1, n))
            W = lambda name: P[name] if name in P else base(name)                                         # noqa: E731
            meta.update(extra_blocks=[{"layer": n, "after": i, "init": self.init}])
        if not P:
            raise ValueError(f"Nothing to train with method '{self.method}' in blocks '{self.layers}'.")
        return P, W, schedule, adapters, meta

    def run(self):
        from safetensors.torch import save_file
        loaded, ad = self.loaded, self.loaded.adapter
        m, cfg = ad.model, ad.cfg
        task = self.tasks.new("run", f"train {self.method} “{self.name}”")
        task.on_cancel = lambda: setattr(self, "stop", True)
        try:
            if self.probe:
                with self.tasks.slot(task):
                    self.samples.append({"step": 0, "text": self.generate(ad, self.probe), "label": "before"})
            torch.manual_seed(self.seed)
            with self.tasks.slot(task):
                P, W, schedule, adapters, meta = self.build(m, cfg)
            self.n_params = sum(p.numel() for p in P.values())
            w_out = m.unembed_weight().to(m.device, torch.float32)
            spec = {"n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "n_kv": cfg.n_kv, "head_dim": cfg.head_dim,
                    "eps": cfg.eps, "inv_freq": m.inv_freq.float().cpu(), "act": m.act, "qk_norm": m.qk_norm,
                    "schedule": schedule, "adapters": set(adapters),
                    "bias": lambda name: (name + ".bias") in P or (m.has_bias and m.W.has(name + ".bias"))}
            embed = lambda ids: m.W.rows("model.embed_tokens.weight", ids.flatten().tolist()).float().view(*ids.shape, -1)   # noqa: E731
            data = [self.encode(ad, e) for e in self.examples]
            data = [(i, w) for i, w in data if len(i) >= 2 and sum(w[1:]) > 0]
            if not data:
                raise ValueError("The examples are too short to learn from.")
            opt = torch.optim.AdamW(list(P.values()), lr=self.lr, weight_decay=0.0)
            rnd = random.Random(self.seed)
            t0 = time.time()
            for step in range(1, self.steps + 1):
                if self.stop:
                    break
                ids, w = rnd.choice(data)
                x = torch.tensor([ids], device=m.device)
                with self.tasks.slot(task):
                    if loaded.adapter is not ad:
                        raise RuntimeError("The model was unloaded during training.")
                    with torch.enable_grad():
                        loss = next_token_loss(forward_train(W, embed, lambda h: h @ w_out.T, spec, x), x,
                                               torch.tensor([w[1:]], device=m.device))
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
                        opt.step()
                self.step, self.seconds = step, time.time() - t0
                self.losses.append([step, round(loss.item(), 4)])
                self.message = f"step {step}/{self.steps} · loss {loss.item():.3f}"
                task.progress, task.message = step / self.steps, self.message
            # Save: changed matrices as their new values (the inspector shows ΔW), new weights under new names.
            self.out.mkdir(parents=True, exist_ok=True)
            for old in self.out.glob("*.safetensors"):
                old.unlink()
            (self.out / "adapter_config.json").unlink(missing_ok=True)
            with torch.no_grad():
                tensors = {name: W(name).detach().float().cpu().contiguous() for name in P}
            save_file(tensors, str(self.out / "adaptation.safetensors"))
            (self.out / "adaptation.json").write_text(json.dumps({
                **meta, "name": self.name, "steps": self.step, "n_params": self.n_params,
                "base_model": ad.spec.get("repo") or ad.spec.get("id"), "examples": self.examples[:50]}, indent=1))
            self.result = {"path": str(self.out), "n_params": self.n_params, "method": self.method}
            if self.attach:
                self.result["lora"] = loaded.attach_lora(self.out, 1.0, self.name)
                if self.probe:
                    with self.tasks.slot(task):
                        self.samples.append({"step": self.step, "text": self.generate(ad, self.probe), "label": "after"})
            self.message = f"saved to {self.out}" + (" and attached — see the LoRA / Fine-tune tab" if self.attach else "")
            self.tasks.finish(task, "cancelled" if self.stop else "done")
        except Exception as e:
            self.tasks.finish(task, error=f"{type(e).__name__}: {e}")
            raise
