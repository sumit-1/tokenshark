"""Model registry: discovers plugins in `tokenshark/plugins/`, user entries (`models.local.json`),
and manages the currently loaded model + any attached LoRA / fine-tuned checkpoint overlay."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .engine import pick_device, pick_dtype
from .lora import LoRAOverlay
from .plugins import discover_plugins, find_plugin_for, list_plugins
from . import weights as _weights
from .weights import discover_folders, folder_size, is_downloaded, model_folder, models_dir, resolve_model_dir

HERE = Path(__file__).parent
LOCAL = Path(os.environ.get("TOKENSHARK_HOME", HERE.parent / "data")) / "models.local.json"


def list_specs() -> list[dict]:
    """Gather default models from all registered plugins, plus `models.json` and `models.local.json`."""
    plugins = discover_plugins()
    specs: list[dict] = []
    seen: set[str] = set()
    for cls in plugins.values():
        for m in cls.default_models:
            if m["id"] not in seen:
                specs.append({**m, "plugin": cls.id, "plugin_name": cls.name})
                seen.add(m["id"])
    # Also include any additional entries in models.json not already declared in a plugin
    if (HERE / "models.json").exists():
        try:
            for m in json.loads((HERE / "models.json").read_text()):
                if m["id"] not in seen:
                    pcls = find_plugin_for(None, m)
                    specs.append({**m, "plugin": pcls.id, "plugin_name": pcls.name})
                    seen.add(m["id"])
        except Exception:
            pass
    if LOCAL.exists():
        try:
            local = json.loads(LOCAL.read_text())
            ids = {s["id"] for s in local}
            specs = [s for s in specs if s["id"] not in ids]
            for s in local:
                pcls = find_plugin_for(Path(s["path"]) if s.get("path") else None, s)
                specs.append({**s, "plugin": pcls.id, "plugin_name": pcls.name})
        except json.JSONDecodeError:
            pass
    for d in discover_folders({s["id"] for s in specs}):
        pcls = find_plugin_for(Path(d["path"]), d)
        specs.append({**d, "plugin": pcls.id, "plugin_name": pcls.name})
    return specs


def with_disk_state(spec: dict) -> dict:
    f = model_folder(spec)
    dl = is_downloaded(spec)
    return {**spec, "downloaded": dl, "folder": str(f), "disk_gb": round(folder_size(f) / 2**30, 2) if dl else 0}


def add_local_spec(spec: dict) -> dict:
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(LOCAL.read_text()) if LOCAL.exists() else []
    cur = [s for s in cur if s["id"] != spec["id"]] + [spec]
    LOCAL.write_text(json.dumps(cur, indent=2))
    return spec


def get_spec(model_id: str) -> dict:
    for s in reversed(list_specs()):     # local entries override built-ins with the same id
        if s["id"] == model_id:
            return s
    raise KeyError(model_id)


class Loaded:
    """At most one model in memory at a time, with optional LoRA / fine-tuned overlay."""

    def __init__(self):
        self.lock = threading.Lock()
        self.adapter = None
        self.plugin = None
        self.lora: LoRAOverlay | None = None
        self.quant = None                      # QuantOverlay while the weights are being rounded
        self.spec: dict | None = None
        self.info: dict | None = None
        self.status = "idle"
        self.error: str | None = None
        self.progress: dict | None = None      # download / copy progress while loading
        self.device = pick_device()
        self.dtype = pick_dtype(self.device)

    def load(self, model_id: str, policy: str = "auto", keep_layers: int = 0, allow_download: bool = True):
        spec = get_spec(model_id)
        with self.lock:
            self.status, self.error = f"loading {model_id}", None
            self.unload()
            t0 = time.time()
            try:
                self.status = f"locating {model_id}"

                def st(msg):
                    self.status = msg

                def prog(done, total, speed, what):
                    self.progress = {"done": done, "total": total, "speed": speed, "what": what}
                self.progress = None
                root = resolve_model_dir(spec, allow_download, st, prog)
                self.progress = None
                self.status = f"loading {model_id}"
                self.load_started = time.time()

                def phase(msg):
                    self.status = f"loading {model_id}: {msg}"
                    self.progress = None
                _weights._REPORTERS.update(phase=phase, bytes=prog)
                plugin_cls = find_plugin_for(root, spec)
                _weights.report_phase(f"loading {model_id} via plugin '{plugin_cls.id}' on {self.device} "
                                      f"({str(self.dtype).replace('torch.', '')}), weights policy '{policy}'")
                ad = plugin_cls.create_adapter(spec, root, self.device, self.dtype, policy, keep_layers)
                self.adapter, self.plugin, self.spec = ad, plugin_cls, spec
                self.info = {
                    **ad.info(),
                    "model_id": model_id,
                    "plugin": plugin_cls.id,
                    "plugin_name": plugin_cls.name,
                    "ui_schema": plugin_cls.ui_schema,
                    "features": list(plugin_cls.features),
                    "lora": None,
                    "load_seconds": round(time.time() - t0, 1),
                    "device": self.device,
                    "dtype": str(self.dtype).replace("torch.", ""),
                    "path": str(root),
                }
                _weights._REPORTERS.update(phase=None, bytes=None)
                _weights.report_phase(f"{model_id} ready in {time.time() - t0:.0f}s")
                self.status = "ready"
            except Exception as e:  # surface load errors to the UI
                _weights._REPORTERS.update(phase=None, bytes=None)
                self.status, self.error = "error", f"{type(e).__name__}: {e}"
                _weights.report_phase(f"load failed: {self.error}")
                raise
            finally:
                _weights._REPORTERS.update(phase=None, bytes=None)
                self.progress = None
        return self.info

    def attach_lora(self, path: str | Path, scale: float = 1.0, name: str = "", layers: str = "all") -> dict:
        """Load a LoRA or fine-tuned checkpoint and attach it to all WeightStores of the active model."""
        if self.adapter is None or self.plugin is None:
            raise RuntimeError("Load a base model first before attaching a LoRA or fine-tuned checkpoint.")
        with self.lock:
            self.detach_lora()
            overlay = LoRAOverlay(path, scale=scale, name=name, layers=layers)
            stores = self.plugin.weight_stores(self.adapter)
            summary = overlay.bind(stores)
            try:
                summary["adaptation"] = self._install_extras(overlay)
            except Exception:
                self.detach_lora()
                raise
            overlay.adaptation = summary["adaptation"]
            self.lora = overlay
            # Clear cached prompt KV passes if text/transformer weights were adapted
            if hasattr(self.adapter, "__dict__"):
                self.adapter.__dict__.pop("_text_memo", None)
                self.adapter.__dict__.pop("_memo", None)
            if self.info is not None:
                self.info["lora"] = summary
            return summary

    def set_lora_scale(self, scale: float, layers: str | None = None) -> dict:
        if self.lora is None:
            raise RuntimeError("No LoRA or fine-tune is currently loaded.")
        with self.lock:
            self.lora.scale = float(scale)
            if layers is not None:
                self.lora.set_layers(layers)
            if hasattr(self.adapter, "__dict__"):
                self.adapter.__dict__.pop("_text_memo", None)
                self.adapter.__dict__.pop("_memo", None)
            summary = self.lora.summary()
            summary["adaptation"] = getattr(self.lora, "adaptation", None)
            if self.info is not None:
                self.info["lora"] = summary
            return summary

    def _forget_memos(self):
        if self.adapter is not None and hasattr(self.adapter, "__dict__"):
            self.adapter.__dict__.pop("_text_memo", None)
            self.adapter.__dict__.pop("_memo", None)

    def set_quant(self, bits: int | None, measure: bool = True, layers: str = "all",
                  layer_bits: dict[int, int] | None = None) -> dict:
        """Round the block weights to `bits` bits on the fly (None = back to the stored weights)."""
        from .quant import QuantOverlay
        if self.adapter is None or self.plugin is None:
            raise RuntimeError("Load a model first.")
        stores = self.plugin.weight_stores(self.adapter)
        for st in stores.values():
            if st is not None:
                st._quant = None
        self.quant = None
        if bits or layer_bits:
            q = QuantOverlay(bits or 4, layers=layers, layer_bits=layer_bits)
            if measure:
                q.bind(stores)
            else:
                for st in stores.values():
                    if st is not None:
                        st._quant = q
            self.quant = q
        self._forget_memos()
        summary = self.quant.summary() if self.quant is not None else None
        if self.info is not None:
            self.info["quant"] = summary
        return summary or {"active": False}

    def overlay_state(self) -> dict:
        """What sits on top of the stored weights right now (recorded with every run so it can be reproduced)."""
        out = {}
        if self.lora is not None:
            out["lora"] = {"path": str(self.lora.path), "scale": self.lora.scale, "name": self.lora.name,
                           "layers": getattr(self.lora, "layers_spec", "all")}
        if self.quant is not None:
            if getattr(self.quant, "layers_spec", "all") not in ("all", "", "*") or getattr(self.quant, "layer_bits", None):
                out["quant"] = {"bits": self.quant.bits, "layers": self.quant.layers_spec,
                                "layer_bits": self.quant.layer_bits}
            else:
                out["quant"] = self.quant.bits
        return out

    def _install_extras(self, overlay) -> dict | None:
        """Blocks and adapter layers an adaptation adds (adapt.py): keep their weights in memory, tell the flow."""
        from .adapt import apply_extras
        meta = apply_extras(self.adapter, overlay)
        self._sync_blocks()
        return meta

    def _remove_extras(self):
        from .adapt import clear_extras
        had = bool(getattr(self.adapter, "extra_blocks", None))
        clear_extras(self.adapter)
        if had:
            self._sync_blocks()

    def _sync_blocks(self):
        if self.info is not None and self.adapter is not None and getattr(self.adapter, "family", "") == "causal_lm":
            self.info.update({k: v for k, v in self.adapter.info().items() if k in ("n_layers", "block_subs", "extra_blocks")})

    def detach_lora(self):
        self._remove_extras()
        if self.adapter is not None and self.plugin is not None:
            for st in self.plugin.weight_stores(self.adapter).values():
                if st is not None:
                    st.set_lora_hook(None)
            if hasattr(self.adapter, "__dict__"):
                self.adapter.__dict__.pop("_text_memo", None)
                self.adapter.__dict__.pop("_memo", None)
        self.lora = None
        if self.info is not None:
            self.info["lora"] = None

    def unload(self):
        self.detach_lora()
        self.quant = None
        if self.adapter is not None:
            try:
                self.adapter.unload()
            except Exception:
                pass
        self.adapter, self.plugin, self.spec, self.info = None, None, None, None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


LOADED = Loaded()
