"""LoRA & weight-diff overlay extension (`tokenshark.extensions.lora`)."""
from __future__ import annotations

from ...sdk import FeaturePlugin, register_extension


@register_extension
class LoRAPlugin(FeaturePlugin):
    id = "lora"
    name = "LoRA & Fine-Tune Weight Diff Overlay"
    version = "0.2.0"
    view_type = "playbook"
    default_enabled = False
    description = "Attach PEFT LoRA adapters or full fine-tune checkpoints, inspect per-layer weight changes, and adjust scales live."
    families = ("causal_lm", "t2i")
    ui_tabs = ({"id": "lora", "label": "LoRA / Diff"},)
