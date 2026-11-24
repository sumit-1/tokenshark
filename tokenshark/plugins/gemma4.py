"""Plugin: Gemma-4 dense language models with image input (google/gemma-4-12B "unified" checkpoints).

Everything Gemma-specific lives in this file. The engine in tokenshark/llm/ only offers generic building blocks;
this plugin maps the checkpoint onto them through `ArchHooks` (tokenshark/llm/arch.py):

- Layers come in two kinds (5 sliding-window : 1 global). Sliding layers: head_dim, num_key_value_heads,
  plain RoPE. Global layers: global_head_dim, num_global_key_value_heads, keys reused as values (no v_proj),
  "proportional" partial RoPE. -> per-layer `LayerAttn` entries.
- Values get a weight-free RMSNorm, attention scores are not scaled (QK-norm does it), token embeddings are
  multiplied by sqrt(hidden), logits are soft-capped. Pre/post "sandwich" norms and the per-layer
  `layer_scalar` are detected by the engine from the weight names.
- Images: no vision transformer. Each 48x48 pixel patch becomes one soft token:
      pixels/255 -> LayerNorm -> Linear -> LayerNorm -> + x/y position table -> LayerNorm -> RMSNorm -> Linear
  The prompt reads `<|image>` + `<|image|>` x N + `<image|>`; an image's tokens attend to each other in both
  directions in the sliding-window layers.

Not supported here: video and audio input; per-layer input embeddings (E2B/E4B), mixture-of-experts blocks
(26B-A4B); the SigLIP-style vision tower of non-unified Gemma-4 checkpoints.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..llm.arch import ArchHooks
from ..llm.media import MediaEncoder, load_image
from ..weights import read_json
from . import register_plugin
from .causal_lm import CausalLMPlugin

EMBEDDER = "model.vision_embedder."                          # patch LayerNorms, Dense, position table
PROJECTION = "model.embed_vision.embedding_projection.weight"   # final RMSNorm -> Linear into the language model


def target_size(height: int, width: int, patch: int, max_patches: int, k: int) -> tuple[int, int]:
    """Largest size with at most max_patches patches and both sides a multiple of k*patch (as transformers)."""
    factor = math.sqrt(max_patches * patch ** 2 / (height * width))
    side = k * patch
    th, tw = int(math.floor(factor * height / side)) * side, int(math.floor(factor * width / side)) * side
    longest = (max_patches // k ** 2) * side
    if th == 0 and tw == 0:
        raise ValueError("This image is too small to cut into patches.")
    if th == 0:
        th, tw = side, min(int(math.floor(width / height)) * side, longest)
    elif tw == 0:
        tw, th = side, min(int(math.floor(height / width)) * side, longest)
    return th, tw


def image_to_patches(img, patch: int = 16, k: int = 3, max_soft_tokens: int = 280):
    """(patches [N, (k*patch)^2 * 3] in 0..1, positions [N, 2] as (x, y) patch coordinates), raster order."""
    from PIL import Image
    img = load_image(img)
    th, tw = target_size(img.height, img.width, patch, max_soft_tokens * k * k, k)
    if (th, tw) != (img.height, img.width):
        img = img.resize((tw, th), Image.BICUBIC)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(th, tw, 3).permute(2, 0, 1).float() / 255
    P = k * patch
    nh, nw = th // P, tw // P
    patches = t.reshape(3, nh, P, nw, P).permute(1, 3, 2, 4, 0).reshape(nh * nw, -1)
    ys, xs = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij")
    return patches, torch.stack([xs, ys], -1).reshape(-1, 2)


class Gemma4ImageEncoder(MediaEncoder):
    """The encoder-free image embedder of a Gemma-4 unified checkpoint, run from the model's WeightStore."""

    def __init__(self, model, top: dict):
        vc = top.get("vision_config") or {}
        self.model, self.W = model, model.W
        self.token_id, self.begin_id, self.end_id = top["image_token_id"], top.get("boi_token_id"), top.get("eoi_token_id")
        self.patch = int(vc.get("patch_size") or 16)
        self.k = int(vc.get("pooling_kernel_size") or 3)
        ip = read_json(Path(model.root) / "processor_config.json").get("image_processor") or {}
        self.max_soft = int(ip.get("max_soft_tokens") or vc.get("num_soft_tokens") or 280)
        self.eps = float(vc.get("rms_norm_eps") or 1e-6)

    def prepare(self, src):
        return image_to_patches(src, self.patch, self.k, self.max_soft)

    def n_tokens(self, item) -> int:
        return item[0].shape[0]

    def _w(self, name: str) -> torch.Tensor:
        return self.W(EMBEDDER + name).float()

    def _ln(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self._w(name + ".weight"), self._w(name + ".bias"), 1e-5)

    @torch.no_grad()
    def encode(self, item) -> torch.Tensor:
        patches, pos = item
        dev = self.model.device
        x = self._ln(patches.to(dev).float(), "patch_ln1")
        x = self._ln(x @ self._w("patch_dense.weight").T + self._w("patch_dense.bias"), "patch_ln2")
        pe = self._w("pos_embedding")                        # [positions, 2 axes, D]
        pos = pos.to(dev).long().clamp(min=0)
        x = self._ln(x + pe[pos[:, 0], 0] + pe[pos[:, 1], 1], "pos_norm")
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x @ self.W(PROJECTION).float().T).to(self.model.dtype)


class Gemma4Hooks(ArchHooks):
    weight_skip_prefixes = (EMBEDDER, "model.embed_vision.", "model.embed_audio.", "model.vision_tower.",
                            "model.audio_tower.")

    def normalize_config(self, hf: dict, top: dict) -> dict:
        """Gemma-4 keys -> the generic ones: global layers' head size / KV heads become per_layer_config."""
        unsupported = [why for why, bad in (
            ("per-layer input embeddings (E2B/E4B)", int(hf.get("hidden_size_per_layer_input") or 0) > 0),
            ("mixture-of-experts blocks (26B-A4B)", bool(hf.get("enable_moe_block"))),
            ("bidirectional attention over all tokens", hf.get("use_bidirectional_attention") == "all")) if bad]
        if unsupported:
            raise ValueError("This Gemma-4 variant isn't supported yet: " + ", ".join(unsupported) + ".")
        n = int(hf["num_hidden_layers"])
        lt = list(hf.get("layer_types") or ["sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(n)])
        lt[-1] = "full_attention"                            # transformers forces the last layer global too
        hf["layer_types"] = lt
        hf.setdefault("rope_parameters", {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6}})
        if "per_layer_config" not in hf:                     # hub configs; transformers-saved ones carry it already
            over = {"head_dim": int(hf.get("global_head_dim") or 512)}
            if hf.get("attention_k_eq_v") and hf.get("num_global_key_value_heads"):
                over["num_key_value_heads"] = int(hf["num_global_key_value_heads"])
            hf["per_layer_config"] = {str(i): over for i, t in enumerate(lt) if t == "full_attention"}
        return hf

    def configure(self, cfg, hf: dict, top: dict) -> None:
        cfg.embed_scale = cfg.hidden ** 0.5
        cfg.attn_scale = 1.0
        cfg.v_norm = True
        for la in cfg.layers:
            la.k_eq_v = bool(hf.get("attention_k_eq_v")) and la.window is None
        if top.get("image_token_id") is not None:
            cfg.media_token_ids = {top[k] for k in ("image_token_id", "video_token_id", "audio_token_id")
                                   if top.get(k) is not None}
        cfg.media_bidir = "sliding" if hf.get("use_bidirectional_attention") == "vision" else None

    def media_encoder(self, model):
        top = read_json(Path(model.root) / "config.json")
        if top.get("image_token_id") is None or not model.W.has(EMBEDDER + "patch_dense.weight"):
            return None                                       # text-only checkpoint
        return Gemma4ImageEncoder(model, top)


@register_plugin
class Gemma4Plugin(CausalLMPlugin):
    id = "gemma4"
    name = "Gemma-4 (dense, image + text)"
    summary = ("Gemma-4 dense decoders: sliding-window and global layers with different head sizes, K=V global "
               "attention, proportional partial RoPE, sandwich norms, logit soft-capping and encoder-free image "
               "input - all on the same flow, debugger, lens and KV tools as other language models.")
    priority = 10
    default_models = [
        {
            "id": "gemma-4-12b-it",
            "plugin": "gemma4",
            "family": "causal_lm",
            "repo": "google/gemma-4-12B-it",
            "size_gb": 24.0,
            "notes": ("48 blocks (5 sliding-window : 1 global), GQA 16q/8kv local, 1 shared K=V head global; takes "
                      "images. ~24 GB in bf16 - 'RAM → device' or 'stream from disk' below 24 GB of VRAM."),
        },
    ]

    @classmethod
    def matches(cls, root: Path, spec: dict | None = None) -> bool:
        if spec and spec.get("plugin") == cls.id:
            return True
        cfg = read_json(root / "config.json")
        mt = (cfg.get("text_config") or {}).get("model_type") or cfg.get("model_type") or ""
        return mt.startswith("gemma4")

    @classmethod
    def arch_hooks(cls) -> ArchHooks:
        return Gemma4Hooks()
