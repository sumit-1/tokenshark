"""Plugin Architecture for Tokenshark.

How to add a new model plugin (for students, educators, or model creators):
===========================================================================
1. Drop a single `.py` file into `analyser/plugins/` (for example `my_model.py`).
2. Subclass `ModelPlugin` and decorate it with `@register_plugin`.
3. Define:
   - `id`: unique plugin identifier (e.g. `"qwen_image_21"`, `"stable_diffusion"`, `"causal_lm"`).
   - `name` & `summary`: human-readable name and teaching description.
   - `family`: `"causal_lm"` (text → text) or `"t2i"` (text → image).
   - `default_models`: built-in Hugging Face repo entries that belong to this plugin,
     including their `id`, `repo`, `size_gb`, and download/runtime notes.
   - `matches(root, spec)`: returns `True` if a model folder on disk (or a `spec`) belongs
     to this plugin.
   - `pick_download_files(repo_files, spec)` (optional): customises which files from a
     Hugging Face repo need to be downloaded.
   - `ui_schema` & `editable_params` (optional): declares custom UI tabs, controls, and
     inspector fields specific to this model architecture so the browser UI renders them
     automatically!
   - `create_adapter(spec, root, device, dtype, policy, keep_layers)`: returns the runtime
     adapter instance.

To remove a plugin, simply delete its `.py` file from `analyser/plugins/`.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

import torch

from ..lora import LoRAOverlay
from ..weights import _pick_files, read_json


class ModelPlugin:
    """Base class for every model architecture plugin in Tokenshark."""

    id: str = "base"
    name: str = "Base Model Plugin"
    family: str = "causal_lm"             # "causal_lm" | "t2i"
    summary: str = ""
    docs_url: str = ""

    # Built-in model catalog entries provided by this plugin (replaces monolithic models.json)
    default_models: list[dict[str, Any]] = []

    # Declarative UI schema so the frontend knows what tabs, node kinds, and interactive
    # features this plugin supports without hardcoding model names in app.js
    # What the page shows for models of this plugin. The web UI has no per-model conditions: it builds itself
    # from this declaration.
    #   tabs             results tabs, in order — ids of modules in analyser/web/tabs/ (see web/README.md)
    #   palette_actions  flow-editing buttons — the data-action values in web/index.html
    #   train            sections of the Train tab: tiny_text | adapt | tiny_image
    #   course           which guided course the Learn tab offers: text | image
    #   step_debugger    whether the header's step buttons apply; step_unit: token | step (what ▸▸ advances);
    #                    step_background: prepare the session in the background; step_saved_note: toast on Finish
    #   modality         text | image — which fields the Prompt and Intervention boxes have
    #   texts            wording set for help and lessons: lm | unet | dit
    #   block_term       what a box on the chain is called: block | stage
    #   default_hook     hook a new intervention box starts with (default: the adapter's first hook)
    ui_schema: dict[str, Any] = {
        "tabs": ["learn", "lora", "quant", "runs"],
        "palette_actions": ["intervention"],
        "train": [],
        "course": "",
        "modality": "text",
        "texts": "",
        "block_term": "block",
        "step_debugger": False,
        "lora_support": True,
        "editable_components": [],
    }

    # Backend tools this plugin's models support (checked by the HTTP routes instead of testing the model family):
    #   lm_tools      lens, word clouds, patching, steering, neurons, tokenizer, sampler, layer prefill, box code,
    #                 adaptation training
    #   kv_cache      saved KV caches            image_maps   stage maps and block contributions
    #   lora · quant  weight overlays
    features: tuple[str, ...] = ("lora", "quant")

    # How runs of this plugin's models are driven (runs.RUN_LOOPS): token_stepper | image_stepper | plain
    run_loop: str = "plain"

    @classmethod
    def default_flow(cls, info: dict) -> dict:
        """The boxes-and-arrows graph a model of this plugin starts with (see flow.py for the two built-in shapes)."""
        raise NotImplementedError(f"Plugin '{cls.id}' does not define default_flow().")

    @classmethod
    def default_flow(cls, info: dict) -> dict:
        """The boxes-and-arrows graph a model of this plugin starts with (see flow.py for the two built-in shapes)."""
        raise NotImplementedError(f"Plugin '{cls.id}' does not define default_flow().")

    @classmethod
    def matches(cls, root: Path, spec: dict | None = None) -> bool:
        """Return True if the model folder `root` (or `spec`) is handled by this plugin."""
        if spec and spec.get("plugin") == cls.id:
            return True
        return False

    @classmethod
    def pick_download_files(cls, repo_files: list[str], spec: dict) -> list[str]:
        """Choose which files to download from Hugging Face for a model of this plugin."""
        return _pick_files(repo_files, cls.family)

    @classmethod
    def create_adapter(cls, spec: dict, root: Path, device: str, dtype: torch.dtype,
                       policy: str = "auto", keep_layers: int = 0):
        """Instantiate and return the runtime adapter for this model."""
        raise NotImplementedError

    @staticmethod
    def weight_stores(adapter) -> dict[str, Any]:
        """Return a dict of `{component_name: WeightStore}` on `adapter` so LoRAs / fine-tunes
        can attach to every weight store automatically."""
        if hasattr(adapter, "weight_stores"):
            return adapter.weight_stores()
        stores = {}
        if hasattr(adapter, "model") and hasattr(adapter.model, "W"):
            stores["model"] = adapter.model.W
        if hasattr(adapter, "dit") and hasattr(adapter.dit, "W"):
            stores["transformer"] = adapter.dit.W
        if hasattr(adapter, "unet") and hasattr(adapter.unet, "W"):
            stores["unet"] = adapter.unet.W
        if hasattr(adapter, "text"):
            if hasattr(adapter.text, "W"):
                stores["text_encoder"] = adapter.text.W
            elif hasattr(adapter.text, "model") and hasattr(adapter.text.model, "W"):
                stores["text_encoder"] = adapter.text.model.W
        if hasattr(adapter, "vae") and hasattr(adapter.vae, "W"):
            stores["vae"] = adapter.vae.W
        return stores


# ======================================================================================
# Plugin Registry & Auto-Discovery
# ======================================================================================

_PLUGINS: dict[str, type[ModelPlugin]] = {}
_DISCOVERED = False


def register_plugin(cls: type[ModelPlugin]) -> type[ModelPlugin]:
    """Decorator that registers a `ModelPlugin` subclass."""
    _PLUGINS[cls.id] = cls
    return cls


def discover_plugins() -> dict[str, type[ModelPlugin]]:
    """Import every `.py` module in `analyser/plugins/` so plugins self-register."""
    global _DISCOVERED
    if not _DISCOVERED:
        pkg_dir = Path(__file__).parent
        for mod in pkgutil.iter_modules([str(pkg_dir)]):
            if not mod.name.startswith("_"):
                importlib.import_module(f"{__name__}.{mod.name}")
        _DISCOVERED = True
    return _PLUGINS


def list_plugins() -> list[dict[str, Any]]:
    """Return metadata for all installed plugins (exposed at `GET /api/plugins`)."""
    discover_plugins()
    out = []
    for pid, cls in _PLUGINS.items():
        out.append({
            "id": cls.id,
            "name": cls.name,
            "family": cls.family,
            "summary": cls.summary,
            "ui_schema": cls.ui_schema,
            "models": [m["id"] for m in cls.default_models],
        })
    return out


def find_plugin_for(root: Path | None, spec: dict | None = None) -> type[ModelPlugin]:
    """Find the plugin that handles `spec` or the model folder at `root`."""
    discover_plugins()
    if spec and spec.get("plugin") in _PLUGINS:
        return _PLUGINS[spec["plugin"]]
    if spec:
        for cls in _PLUGINS.values():
            if any(m["id"] == spec.get("id") for m in cls.default_models):
                return cls
    if root is not None and root.exists():
        for cls in _PLUGINS.values():
            if cls.matches(root, spec):
                return cls
    # Fallback by family
    fam = (spec or {}).get("family", "causal_lm")
    for cls in _PLUGINS.values():
        if cls.family == fam:
            return cls
    raise ValueError(f"No plugin found for model spec={spec}, root={root}")
