"""TEMPLATE: Copy this file to `tokenshark/plugins/<your_model>.py` to add a new model plugin!

Because all plugins in `tokenshark/plugins/` are auto-discovered on startup (unless their filename
starts with `_`), any `.py` file you drop here and decorate with `@register_plugin` immediately
appears in the model picker, supports weight streaming (`auto` / `device` / `cpu` / `disk`),
and gets automatic LoRA / fine-tune visualisation.
"""
from __future__ import annotations

from pathlib import Path
import torch

from ..common import pca_rgb, rms_norm
from ..weights import WeightStore, read_json
from . import ModelPlugin, register_plugin


# Uncomment @register_plugin when your plugin is ready:
# @register_plugin
class CustomModelPlugin(ModelPlugin):
    # 1. Unique plugin ID and family ("causal_lm" for text, "t2i" for image)
    id = "example_custom_model"
    name = "Example Custom Model Plugin"
    family = "causal_lm"
    summary = "Template showing how to connect a new open-weight architecture in ~50 lines."

    # Backend tools your adapter supports (routes check this instead of the model family). A new text
    # architecture whose adapter follows LLMAdapter can list "lm_tools" and "kv_cache" too.
    features = ("lora", "quant")

    # Backend tools your adapter supports (routes check this instead of the model family). A new text
    # architecture whose adapter follows LLMAdapter can list "lm_tools" and "kv_cache" too.
    features = ("lora", "quant")

    # 2. Built-in downloadable models (Hugging Face repo IDs + metadata)
    default_models = [
        {
            "id": "my-open-model-1b",
            "plugin": "example_custom_model",
            "family": "causal_lm",
            "repo": "org/my-open-model-1b",
            "size_gb": 2.0,
            "notes": "Example model entry.",
        }
    ]

    # 3. Declarative UI schema: which tabs and editable components appear in the browser
    ui_schema = {
        # ids of tab modules in tokenshark/web/tabs/ — list only what your adapter can feed
        "tabs": ["learn", "output_text", "debug_text", "lens", "lora", "quant", "runs"],
        "palette_actions": ["intervention", "duplicate_block", "bypass_block"],
        "train": [],
        "course": "text",
        "modality": "text",          # text | image: which fields the Prompt and Intervention boxes have
        "texts": "lm",               # wording set for help and lessons: lm | unet | dit
        "step_debugger": True,
        "step_unit": "token",
        "lora_support": True,
        "editable_components": [
            {
                "kind": "input_prompt",
                "label": "Prompt",
                "description": "Input text for the model.",
            }
        ],
    }

    # 4. Detection criteria: how to recognise a local folder dropped into `models/`
    @classmethod
    def matches(cls, root: Path, spec: dict | None = None) -> bool:
        if super().matches(root, spec):
            return True
        cfg = read_json(root / "config.json")
        return cfg.get("model_type") == "my_custom_arch"

    # 5. Instantiate your runtime adapter (uses WeightStore + shared helpers in tokenshark/common.py)
    @classmethod
    def create_adapter(cls, spec: dict, root: Path, device: str, dtype: torch.dtype,
                       policy: str = "auto", keep_layers: int = 0):
        raise NotImplementedError("Connect your adapter class here!")
