"""LoRA & Fine-Tuning Inspector / Overlay Engine.

Why this exists:
----------------
Students learning how open-weight models are adapted do two things:
  1. Train a **LoRA** (`adapter_model.safetensors` + `adapter_config.json`, or a single-file
     Kohya / Diffusers `.safetensors` LoRA) where each adapted weight gets a low-rank update:
         W_effective = W_base + scale * (alpha / r) * (B @ A)
  2. **Fine-tune** a model checkpoint directly (saving a new folder of `.safetensors` weights),
     where:
         ΔW = W_finetuned - W_base

Because every plugin in Tokenshark reads weights through `WeightStore` by canonical tensor name
(`W("model.layers.3.self_attn.q_proj.weight")` or `W("transformer_blocks.0.attn.to_q.weight")`),
handling LoRAs and fine-tuned checkpoints at the `WeightStore` level means **every model plugin
gets LoRA & fine-tune support automatically — without writing a new plugin!**

Even better, we can inspect the exact per-layer changes (`‖ΔW‖_F`, relative change `‖ΔW‖ / ‖W‖`,
rank `r`, and which submodules `q_proj`, `k_proj`, `mlp`, etc. were modified) and highlight the
blocks that changed the most directly on the flow canvas and in the LoRA / Fine-Tune tab.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from .weights import _LAYER_RE, _SafetensorsFile, read_json

# Common prefixes added by PEFT / Diffusers / Kohya LoRA trainers that wrap the base model's parameter names.
_LORA_PREFIXES = (
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "unet.",
    "text_encoder.",
    "lora_unet_",
    "lora_te_",
)


def _strip_lora_key(key: str) -> tuple[str, str] | None:
    """Parse a LoRA tensor name into `(base_param_stem, kind)` where `kind` is `'A'`, `'B'`, or `'alpha'`.

    Supports:
      - Hugging Face PEFT:   `base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight`
      - Diffusers LoRA:      `transformer.transformer_blocks.0.attn.to_q.lora.down.weight`
      - Kohya / ComfyUI:     `lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight`
    """
    patterns = [
        (r"^(.*?)\.lora_A(?:\.[^.]+)?\.weight$", "A"),
        (r"^(.*?)\.lora_B(?:\.[^.]+)?\.weight$", "B"),
        (r"^(.*?)\.lora\.down(?:\.[^.]+)?\.weight$", "A"),
        (r"^(.*?)\.lora\.up(?:\.[^.]+)?\.weight$", "B"),
        (r"^(.*?)\.lora_down(?:\.[^.]+)?\.weight$", "A"),
        (r"^(.*?)\.lora_up(?:\.[^.]+)?\.weight$", "B"),
        (r"^(.*?)\.lora_linear_layer\.down\.weight$", "A"),
        (r"^(.*?)\.lora_linear_layer\.up\.weight$", "B"),
        (r"^(.*?)\.alpha$", "alpha"),
        (r"^(.*?)\.lora_alpha$", "alpha"),
    ]
    for pat, kind in patterns:
        m = re.match(pat, key)
        if m:
            return m.group(1), kind
    return None


def _resolve_target_weight(stem: str, store_keys: set[str]) -> str | None:
    """Match a LoRA parameter stem to the exact canonical weight name in a `WeightStore`."""
    candidates = [stem + ".weight", stem]
    for pre in _LORA_PREFIXES:
        if stem.startswith(pre):
            sub = stem[len(pre):]
            candidates += [sub + ".weight", sub]
    # Also try adding "model." if the checkpoint omitted it or vice-versa
    for c in list(candidates):
        if not c.startswith("model."):
            candidates.append("model." + c)
        elif c.startswith("model.language_model."):
            candidates.append("model." + c[len("model.language_model."):])
    for c in candidates:
        if c in store_keys:
            return c
    # Fallback: suffix match on unique ending (e.g. `layers.0.self_attn.q_proj.weight`)
    for c in candidates:
        if c.endswith(".weight"):
            matches = [k for k in store_keys if k.endswith("." + c) or c.endswith("." + k)]
            if len(matches) == 1:
                return matches[0]
    return None


def _load_safetensors_dict(path: Path) -> dict[str, torch.Tensor]:
    """Read all tensors from a `.safetensors` file or a folder containing `.safetensors` / `.bin` files."""
    path = Path(path)
    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if not files:
            for sub in ("transformer", "unet", "text_encoder"):
                if (path / sub).is_dir():
                    files += sorted((path / sub).glob("*.safetensors"))
    if not files:
        bins = [path] if path.is_file() and path.suffix in (".bin", ".pt", ".pth") else sorted(path.glob("*.bin")) if path.is_dir() else []
        if bins:
            out = {}
            for b in bins:
                out.update(torch.load(str(b), map_location="cpu", weights_only=True))
            return out
        raise FileNotFoundError(f"No .safetensors or .bin files found at {path}")
    out = {}
    for f in files:
        sf = _SafetensorsFile(f)
        try:
            for k in sf.keys():
                out[k] = sf.get(k).clone()
        finally:
            sf.fh.close()
    return out


class LoRAOverlay:
    """Holds a loaded LoRA or fine-tuned weight diff attached to one or more `WeightStore`s.

    Computes and caches per-weight and per-layer change metrics so the UI can:
      1. Colour and badge the flow blocks by how much the LoRA / fine-tune changed them.
      2. Let students adjust the LoRA strength `scale` (e.g. `0.0` = off, `1.0` = normal,
         `2.0` = amplified, `-1.0` = negated) in real time without reloading the base model.
      3. Inspect which layers and modules (`q_proj`, `v_proj`, `mlp`, etc.) changed the most
         or which extra weights were added.
    """

    def __init__(self, path: str | Path, scale: float = 1.0, name: str = "", layers: str = "all"):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"LoRA / checkpoint path does not exist: {self.path}")
        self.scale = float(scale)
        self.name = name or self.path.stem
        self.layers_spec = str(layers or "all").strip()
        self._allowed: set[int] | None = None
        self.set_layers(self.layers_spec)
        cfg_dir = self.path if self.path.is_dir() else self.path.parent
        self.cfg = read_json(cfg_dir / "adapter_config.json")
        self.raw_tensors = _load_safetensors_dict(self.path)

        # Separate LoRA pairs (A, B, alpha) from full fine-tuned / extra weights
        self.lora_raw: dict[str, dict[str, torch.Tensor]] = {}
        self.full_raw: dict[str, torch.Tensor] = {}
        for k, t in self.raw_tensors.items():
            parsed = _strip_lora_key(k)
            if parsed is not None:
                stem, kind = parsed
                self.lora_raw.setdefault(stem, {})[kind] = t
            else:
                self.full_raw[k] = t

        self.kind = "lora" if self.lora_raw else "finetune"
        # Mapped per WeightStore component: {weight_name: (A, B, alpha_over_r)} or {weight_name: delta_or_full}
        self.mapped_lora: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        self.mapped_full: dict[str, torch.Tensor] = {}
        self.added_keys: list[dict] = []
        self.module_stats: list[dict] = []
        self.layer_stats: dict[int, dict] = {}

    def set_layers(self, layers: str):
        self.layers_spec = str(layers or "all").strip()
        if self.layers_spec.lower() in ("all", "*", ""):
            self._allowed = None
        else:
            from .flow import parse_index_spec
            self._allowed = set(parse_index_spec(self.layers_spec, 10_000))

    def bind(self, stores: dict[str, " object "]) -> dict:
        """Bind this LoRA / fine-tune to the active model's `WeightStore`s and compute layer diffs."""
        self.mapped_lora.clear()
        self.mapped_full.clear()
        self.added_keys.clear()
        self.module_stats.clear()
        self.layer_stats.clear()

        # Collect all canonical keys across all stores (e.g. main model / transformer / unet / text_encoder)
        key_to_store: dict[str, tuple[str, object]] = {}
        for comp_name, store in stores.items():
            if store is None:
                continue
            for k in store._all_names():
                key_to_store.setdefault(k, (comp_name, store))
            for k in store.keys():
                key_to_store.setdefault(k, (comp_name, store))

        all_keys = set(key_to_store.keys())
        cfg_alpha = self.cfg.get("lora_alpha")
        cfg_r = self.cfg.get("r")

        # 1. Bind LoRA (A, B) pairs
        for stem, parts in self.lora_raw.items():
            if "A" not in parts or "B" not in parts:
                continue
            A, B = parts["A"].float(), parts["B"].float()
            # Conv2d 1x1 LoRAs sometimes have shape [r, in, 1, 1]
            if A.ndim == 4 and A.shape[2:] == (1, 1):
                A = A.squeeze(-1).squeeze(-1)
            if B.ndim == 4 and B.shape[2:] == (1, 1):
                B = B.squeeze(-1).squeeze(-1)
            r = A.shape[0]
            alpha = float(parts["alpha"].item()) if "alpha" in parts else float(cfg_alpha or cfg_r or r)
            scaling = alpha / max(1, r)

            target = _resolve_target_weight(stem, all_keys)
            if target is None:
                self.added_keys.append({"name": stem, "shape": list(B.shape[:1] + A.shape[1:]), "rank": r, "kind": "unmatched_lora"})
                continue

            comp_name, store = key_to_store[target]
            # Canonicalise name for this store
            canon = target
            if store.alias and target.startswith(store.alias[1]):
                canon = store.alias[0] + target[len(store.alias[1]):]

            self.mapped_lora[canon] = (A.to(torch.float16), B.to(torch.float16), scaling)
            store.set_lora_hook(self)

            # Compute exact Frobenius norm of ΔW = scaling * (B @ A) without materialising a huge matrix if possible:
            # ‖B A‖_F² = trace(Aᵀ Bᵀ B A) = trace((Bᵀ B) @ (A Aᵀ)), which is only an [r × r] matmul!
            BtB = B.T @ B                           # [r, r]
            AAt = A @ A.T                           # [r, r]
            delta_norm = float(abs(scaling) * torch.sqrt(torch.clamp((BtB * AAt.T).sum(), min=0.0)).item())

            base_norm = self._estimate_base_norm(store, canon)
            rel = delta_norm / max(base_norm, 1e-6) if base_norm > 0 else 0.0
            layer = store._layer_of(canon)
            self._record_stat(canon, comp_name, layer, delta_norm, base_norm, rel, rank=r, alpha=alpha)

        # 2. Bind full fine-tuned weights (or extra added layers/embeddings)
        for raw_k, t in self.full_raw.items():
            target = _resolve_target_weight(raw_k[:-7] if raw_k.endswith(".weight") else raw_k, all_keys)
            if target is None and raw_k in all_keys:
                target = raw_k
            if target is None:
                self.added_keys.append({"name": raw_k, "shape": list(t.shape), "kind": "added_parameter"})
                continue
            comp_name, store = key_to_store[target]
            canon = target
            if store.alias and target.startswith(store.alias[1]):
                canon = store.alias[0] + target[len(store.alias[1]):]
            base_t = store.compact(canon)
            if tuple(base_t.shape) != tuple(t.shape):
                self.added_keys.append({"name": canon, "shape": list(t.shape), "base_shape": list(base_t.shape), "kind": "shape_changed"})
                continue
            diff = (t.float() - base_t.float())
            delta_norm = float(diff.norm().item())
            if delta_norm < 1e-7:
                continue
            base_norm = float(base_t.float().norm().item())
            rel = delta_norm / max(base_norm, 1e-6)
            self.mapped_full[canon] = diff.to(torch.float16)
            store.set_lora_hook(self)
            layer = store._layer_of(canon)
            self._record_stat(canon, comp_name, layer, delta_norm, base_norm, rel, rank=None, alpha=None)

        return self.summary()

    @staticmethod
    def _estimate_base_norm(store, canon: str) -> float:
        try:
            t = store.compact(canon)
            return float(t.float().norm().item())
        except Exception:
            return 0.0

    def _record_stat(self, name: str, comp: str, layer: int | None, delta_norm: float, base_norm: float,
                     rel: float, rank: int | None, alpha: float | None):
        short_mod = name.split(".")[-2] if name.endswith(".weight") and "." in name else name
        entry = {
            "name": name, "component": comp, "layer": layer, "module": short_mod,
            "delta_norm": round(delta_norm, 4), "base_norm": round(base_norm, 4),
            "relative_change": round(rel, 5), "rank": rank, "alpha": alpha,
        }
        self.module_stats.append(entry)
        if layer is not None:
            ls = self.layer_stats.setdefault(layer, {
                "layer": layer, "delta_norm_sq": 0.0, "base_norm_sq": 0.0,
                "max_rel": 0.0, "modules": [], "count": 0,
            })
            ls["delta_norm_sq"] += delta_norm ** 2
            ls["base_norm_sq"] += base_norm ** 2
            ls["max_rel"] = max(ls["max_rel"], rel)
            ls["count"] += 1
            if short_mod not in ls["modules"]:
                ls["modules"].append(short_mod)

    def apply(self, name: str, base_weight: torch.Tensor) -> torch.Tensor:
        """Apply `W + scale * ΔW` on the fly when `WeightStore(name)` is called."""
        if self.scale == 0.0:
            return base_weight
        if self._allowed is not None:
            m = _LAYER_RE.search(name)
            if m and int(m.group(1)) not in self._allowed:
                return base_weight
        if name in self.mapped_lora:
            A, B, scaling = self.mapped_lora[name]
            dev, dt = base_weight.device, base_weight.dtype
            eff = self.scale * scaling
            if base_weight.ndim == 4 and A.ndim == 2:
                # 1x1 or KxK conv adaptation
                delta = (B.to(dev, torch.float32) @ A.to(dev, torch.float32)).view_as(base_weight)
            else:
                delta = B.to(dev, torch.float32) @ A.to(dev, torch.float32)
            return (base_weight.float() + eff * delta).to(dt)
        if name in self.mapped_full:
            diff = self.mapped_full[name].to(base_weight.device, torch.float32)
            return (base_weight.float() + self.scale * diff).to(base_weight.dtype)
        return base_weight

    def summary(self) -> dict:
        layers_out = []
        for l in sorted(self.layer_stats.keys()):
            ls = self.layer_stats[l]
            d_norm = ls["delta_norm_sq"] ** 0.5
            b_norm = ls["base_norm_sq"] ** 0.5
            rel = d_norm / max(b_norm, 1e-6) if b_norm > 0 else ls["max_rel"]
            layers_out.append({
                "layer": l,
                "delta_norm": round(d_norm, 4),
                "base_norm": round(b_norm, 4),
                "relative_change": round(rel, 5),
                "modules": ls["modules"],
                "count": ls["count"],
            })
        max_rel = max((x["relative_change"] for x in layers_out), default=0.0) or 1.0
        for x in layers_out:
            x["heat"] = round(x["relative_change"] / max_rel, 4)

        top_modules = sorted(self.module_stats, key=lambda x: x["relative_change"], reverse=True)[:60]
        return {
            "active": True,
            "name": self.name,
            "path": str(self.path),
            "kind": self.kind,
            "scale": self.scale,
            "layers_spec": self.layers_spec,
            "n_adapted_weights": len(self.mapped_lora) + len(self.mapped_full),
            "n_added_weights": len(self.added_keys),
            "added_weights": self.added_keys[:50],
            "layers": layers_out,
            "top_modules": top_modules,
        }
