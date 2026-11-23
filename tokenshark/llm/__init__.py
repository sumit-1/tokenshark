"""Raw PyTorch decoder-only LLM package (`tokenshark.llm`).

Split into modular files for readability and context efficiency:
- `config.py`: `LLMConfig`, `meta_name_map`, `meta_transform`
- `ops.py`: `rms_norm`, `rope_inv_freq`, `yarn_scales`, `apply_rope`, `_act`
- `model.py`: `RawLLM` forward pass, hooks, MoE, and side-branch forks
- `analysis.py`: `LLMAnalysisMixin` (logit lens, word cloud, word tracking, attention)
- `stepper.py`: `Stepper` block-by-block and token-by-token execution loop
- `adapter.py`: `LLMAdapter` runtime adapter
- `arch.py`: `ArchHooks`, `LayerAttn` - how a model plugin describes what config.json alone can't (per-layer
  attention shapes, embedding scale, value norm, media placeholders, ...)
- `media.py`: `MediaEncoder` contract for image (media) input, plus image I/O helpers

This package holds generic building blocks only. It never names a model: per-model mapping lives in a plugin
(`tokenshark/plugins/<model>.py`) that passes `ArchHooks` to `LLMAdapter` / `RawLLM`.
"""
from __future__ import annotations

from .adapter import LLMAdapter
from .arch import ArchHooks, LayerAttn
from .config import LLMConfig, meta_name_map, meta_transform
from .model import RawLLM
from .media import MediaEncoder
from .ops import _act, apply_rope, rms_norm, rope_inv_freq, yarn_scales
from .stepper import Stepper

__all__ = [
    "LLMConfig",
    "meta_name_map",
    "meta_transform",
    "rms_norm",
    "rope_inv_freq",
    "yarn_scales",
    "apply_rope",
    "_act",
    "RawLLM",
    "LLMAdapter",
    "Stepper",
    "ArchHooks",
    "LayerAttn",
    "MediaEncoder",
]
