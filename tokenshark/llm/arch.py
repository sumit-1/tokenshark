"""How a model plugin describes an architecture to the generic causal-LM engine (`tokenshark.llm.arch`).

The engine (`config.py`, `model.py`, `adapter.py`) knows building blocks, never model names. It reads what it can
from `config.json` and the checkpoint's weight names. A plugin covers the rest by passing an `ArchHooks` object to
`LLMAdapter` / `RawLLM`:

    normalize_config(hf, top)  rewrite model-specific config keys into the generic ones the reader understands
                               (e.g. a "global_head_dim" key -> per_layer_config overrides)
    configure(cfg, hf, top)    set the engine knobs config.json has no generic key for (list in KNOBS below)
    media_encoder(model)       an encoder that turns images (or other media) into soft-token embeddings
    weight_skip_prefixes       checkpoint keys the language path never reads (vision / audio towers)

Generic knobs on LLMConfig a plugin may set (defaults keep a plain Llama-style decoder):

    cfg.layers[l]        LayerAttn per layer: head_dim, n_kv, window (sliding-window size or None),
                         rope (per-layer RoPE parameter dict or None = model-wide), k_eq_v (values = raw keys),
                         kv_source (shared KV: the earlier layer whose cache this one reads; set from
                         num_kv_shared_layers automatically)
    cfg.embed_scale      multiply token embeddings by this (rounded to the weight dtype); None = off
    cfg.attn_scale       attention score scale; None = 1/sqrt(head_dim)
    cfg.v_norm           RMS-normalise values (no weight) before attention
    cfg.final_softcap    logits -> tanh(logits / cap) * cap; read from final_logit_softcapping when present
    cfg.media_token_ids  placeholder token ids that media soft tokens replace (embedded as pad_id first)
    cfg.media_bidir      None | "sliding" | "all": in which layers one media item's tokens attend both ways
    cfg.norm_offset      RMSNorm scale = weight + offset

Weight-name conventions the engine detects by itself: q_norm / k_norm (QK-norm), pre_feedforward_layernorm +
post_feedforward_layernorm (norms after attention and MLP as well as before), layer_scalar (per-layer residual
scale), attention biases, gated attention, mixture-of-experts layouts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerAttn:
    """Attention shape of one decoder layer."""
    head_dim: int
    n_kv: int
    window: int | None = None          # sliding-window size (None: sees the whole context)
    rope: dict | None = None           # this layer's RoPE parameters (None: the model-wide theta / scaling)
    k_eq_v: bool = False               # no v_proj: values are the key projection (before key norm and RoPE)
    kv_source: int | None = None       # cross-layer KV sharing: no k/v of its own, reads this earlier layer's cache


class ArchHooks:
    """Base class: everything optional. Subclass it in a model plugin and override only what differs."""

    weight_skip_prefixes: tuple[str, ...] = ()

    def normalize_config(self, hf: dict, top: dict) -> dict:
        """hf: the language-model config (text_config for multimodal checkpoints); top: the whole config.json.
        Return the config the generic reader should see."""
        return hf

    def configure(self, cfg, hf: dict, top: dict) -> None:
        """Set engine knobs on the parsed LLMConfig (see the module docstring)."""

    def media_encoder(self, model):
        """A MediaEncoder (tokenshark.llm.media) for this checkpoint, or None for text-only."""
        return None
