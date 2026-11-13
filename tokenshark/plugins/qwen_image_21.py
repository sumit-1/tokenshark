"""Plugin: Qwen-Image 2.1 Single-Stream Diffusion Transformer (DiT).

What this plugin teaches:
-------------------------
- **Single-Stream Diffusion Transformer (DiT)**: unlike a UNet with fixed down/up skip connections,
  all 32 blocks operate on one sequence `[text tokens | image tokens]` with the same weights,
  so blocks can be bypassed, repeated, or reordered just like an LLM!
- **Prompt as a KV Cache (`t = 0` modulation)**: because text tokens are modulated at a fixed `t = 0`,
  the prompt is pushed through the 36-block Qwen3-VL encoder and the 32 DiT blocks *once* and cached.
  Every denoising step only computes the image tokens attending to that text KV cache.
- **Interactive Step Debugger & Between-Step Image Editing**: step block-by-block or step-by-step,
  paint soft shadows/shapes onto the model's current guess `x̂₀` between steps, adjust the between-step
  noise level `σ`, or extend the schedule with more steps after completion.
- **Image Logit Lens ("Decode through last k blocks")**: project any intermediate DiT block's
  image tokens into pixel space (directly or after jumping through the last `k` blocks).
"""
from __future__ import annotations

from pathlib import Path
import torch

from ..qimage import QwenImageAdapter
from ..weights import read_json
from . import ModelPlugin, register_plugin


@register_plugin
class QwenImage21Plugin(ModelPlugin):
    id = "qwen_image_21"
    name = "Qwen-Image 2.1 (Single-Stream DiT + Qwen3-VL + Wan VAE)"
    family = "t2i"
    summary = (
        "Single-stream 32-block DiT (7B) with Qwen3-VL 8B text encoder and 64-channel 16× Wan VAE. "
        "Supports interactive step-by-step image editing, between-step noise σ control, Image Lens, "
        "per-block Prompt-KV caches, and LoRA / fine-tune inspection."
    )

    default_models = [
        {
            "id": "qwen-image-2.1",
            "plugin": "qwen_image_21",
            "family": "t2i",
            "repo": "Qwen/Qwen-Image-2.1",
            "size_gb": 33.1,
            "default_steps": 24,
            "default_guidance": 1.0,
            "default_size": 512,
            "notes": (
                "Single-stream DiT (32 blocks, 7B) + Qwen3-VL 8B text encoder + 64-channel VAE, ~33 GB in bf16. "
                "The prompt goes through the blocks once (KV cache); each step only computes image tokens. "
                "On CPU expect minutes per step at 512×512 (the model's own default is 1024×1024, 40 steps)."
            ),
        },
    ]

    features = ("lora", "quant", "image_maps", "image_lens", "image_stepper")
    run_loop = "image_stepper"

    features = ("lora", "quant", "image_maps", "image_lens", "image_stepper")

    ui_schema = {
        "tabs": ["learn", "output_image", "debug_image", "ilens", "lora", "quant", "train", "runs"],
        "train": ["tiny_image"],
        "course": "image",
        "modality": "image",
        "texts": "dit",
        "block_term": "block",
        "step_debugger": True,
        "step_unit": "step",
        "step_background": True,
        "step_saved_note": "Saved as a run: Step debugger, Image lens and Output remain available.",
        "lora_support": True,
        "palette_actions": [
            "intervention",
            "duplicate_block",
            "bypass_block",
            "init_image",
            "prompt_kv_replace",
            "prompt_kv_prepend",
        ],
        "editable_components": [
            {
                "kind": "debug_image_editor",
                "label": "Between-Step Image & Noise (σ) Editor",
                "description": (
                    "Step through generation, paint soft shadows/shapes onto the current guess x̂₀, "
                    "adjust the noise level σ ∈ [0.02, 1.0], or add extra refinement steps."
                ),
            },
            {
                "kind": "prompt_kv",
                "label": "Per-Block Prompt KV Cache",
                "description": (
                    "Give different DiT blocks different prompts (e.g. layout blocks read prompt A, "
                    "detail blocks read prompt B) or prepend a hidden style prompt."
                ),
            },
            {
                "kind": "init_image",
                "label": "Start Image",
                "description": "Draw or load a starting picture and choose the initial noise level.",
            },
        ],
    }

    @classmethod
    def default_flow(cls, info: dict) -> dict:
        from ..flow import default_t2i_flow
        return default_t2i_flow(info)

    @classmethod
    def default_flow(cls, info: dict) -> dict:
        from ..flow import default_t2i_flow
        return default_t2i_flow(info)

    @classmethod
    def matches(cls, root: Path, spec: dict | None = None) -> bool:
        """Matches a folder whose `model_index.json` has `_class_name` starting with `QwenImage21`."""
        if super().matches(root, spec):
            return True
        if not (root / "model_index.json").exists():
            return False
        idx = read_json(root / "model_index.json")
        return str(idx.get("_class_name", "")).startswith("QwenImage21")

    @classmethod
    def create_adapter(cls, spec: dict, root: Path, device: str, dtype: torch.dtype,
                       policy: str = "auto", keep_layers: int = 0) -> QwenImageAdapter:
        ad = QwenImageAdapter(spec, root, device, dtype, policy, keep_layers)
        ad.plugin_id = cls.id
        ad.ui_schema = cls.ui_schema
        return ad
