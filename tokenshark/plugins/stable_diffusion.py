"""Plugin: Stable Diffusion UNet Models (SD 1.x, SD 2.x, SD-Turbo, Tiny-SD).

What this plugin teaches:
-------------------------
- **UNet Denoising Architecture**: downsampling stages (`down_0..3`), bottleneck (`mid`),
  and upsampling stages (`up_0..3`) tied together by high-resolution skip connections.
- **Cross-Attention Maps**: how each spatial image patch queries the CLIP text tokens
  to pull semantic concepts into specific locations on the canvas.
- **Stage Contributions & Ablation**: measuring *where* each UNet stage writes, *what*
  its features encode via 3-component PCA RGB maps, and *how much* removing a stage
  changes the final generated image.

How to add a new Stable Diffusion model:
----------------------------------------
Add an entry to `default_models` below or drop a diffusers-layout folder (`model_index.json`
with `unet/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`) into `models/`.
"""
from __future__ import annotations

from pathlib import Path
import torch

from ..sd import SDAdapter
from ..weights import read_json
from . import ModelPlugin, register_plugin


@register_plugin
class StableDiffusionPlugin(ModelPlugin):
    id = "stable_diffusion"
    name = "Stable Diffusion UNet (SD 1.5 / 2.x / SD-Turbo / Tiny-SD)"
    family = "t2i"
    summary = (
        "CLIP text encoder + Multi-scale ResNet/Cross-Attention UNet + 8× VAE + Euler scheduler. "
        "Supports cross-attention word maps, stage write/PCA maps, stage ablation, and LoRA inspection."
    )

    default_models = [
        {
            "id": "sd-turbo",
            "plugin": "stable_diffusion",
            "family": "t2i",
            "repo": "stabilityai/sd-turbo",
            "size_gb": 2.6,
            "default_steps": 4,
            "default_guidance": 0.0,
            "default_size": 512,
            "notes": "UNet, 1–4 steps, no CFG. Best first image model (works on CPU, ~1 min).",
        },
        {
            "id": "tiny-sd",
            "plugin": "stable_diffusion",
            "family": "t2i",
            "repo": "segmind/tiny-sd",
            "size_gb": 1.0,
            "default_steps": 20,
            "default_guidance": 7.0,
            "default_size": 512,
            "notes": "Distilled SD 1.5 UNet (fewer blocks). Small and fast.",
        },
        {
            "id": "sd-1.5",
            "plugin": "stable_diffusion",
            "family": "t2i",
            "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "size_gb": 2.0,
            "default_steps": 25,
            "default_guidance": 7.5,
            "default_size": 512,
            "notes": "Classic UNet with CFG; the reference model for cross-attention maps.",
        },
    ]

    features = ("lora", "quant", "image_maps")

    features = ("lora", "quant", "image_maps")

    ui_schema = {
        "tabs": ["learn", "output_image", "contrib", "lora", "quant", "train", "runs"],
        "palette_actions": ["intervention", "init_image"],
        "train": ["tiny_image"],
        "course": "image",
        "modality": "image",
        "texts": "unet",
        "block_term": "stage",
        "step_debugger": False,
        "lora_support": True,
        "editable_components": [
            {
                "kind": "input_prompt",
                "label": "Prompt & Mid-Run Prompt Switch",
                "description": "Positive prompt, negative prompt (for CFG), and mid-generation prompt switching.",
            },
            {
                "kind": "init_image",
                "label": "Start Image (img2img)",
                "description": "Start denoising from a drawn or uploaded picture mixed with noise.",
            },
            {
                "kind": "intervention",
                "label": "Stage / Word Intervention",
                "description": "Scale, zero, or noise a UNet stage's output/skip features or specific prompt words.",
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
        """Matches any diffusers pipeline folder with a `unet/` subdirectory."""
        if super().matches(root, spec):
            return True
        if not (root / "model_index.json").exists():
            return False
        idx = read_json(root / "model_index.json")
        if str(idx.get("_class_name", "")).startswith("QwenImage21"):
            return False
        return (root / "unet").exists()

    @classmethod
    def create_adapter(cls, spec: dict, root: Path, device: str, dtype: torch.dtype,
                       policy: str = "auto", keep_layers: int = 0) -> SDAdapter:
        ad = SDAdapter(spec, root, device, dtype, policy, keep_layers)
        ad.plugin_id = cls.id
        ad.ui_schema = cls.ui_schema
        return ad
