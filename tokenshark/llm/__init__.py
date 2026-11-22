"""Raw PyTorch decoder-only LLM package (`tokenshark.llm`).

Split into modular files for readability and context efficiency:
- `config.py`: `LLMConfig`, `meta_name_map`, `meta_transform`
- `ops.py`: `rms_norm`, `rope_inv_freq`, `yarn_scales`, `apply_rope`, `_act`
- `model.py`: `RawLLM` forward pass, hooks, MoE, and side-branch forks
- `analysis.py`: `LLMAnalysisMixin` (logit lens, word cloud, word tracking, attention)
- `stepper.py`: `Stepper` block-by-block and token-by-token execution loop
- `adapter.py`: `LLMAdapter` runtime adapter
"""
from __future__ import annotations

from .adapter import LLMAdapter
from .config import LLMConfig, meta_name_map, meta_transform
from .model import RawLLM
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
]
