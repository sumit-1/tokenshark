"""Tokenshark Feature Plugin & Extension SDK (`tokenshark.sdk`).

Allows built-in extensions (`tokenshark/extensions/<name>/`), local user plugins (`~/.tokenshark/extensions/<name>/`),
and pip-installed packages (`entry_points(group="tokenshark.extensions")`) to register:
- FastAPI `APIRouter` endpoints
- Custom Flow Node Kinds & Handles (including side-branch `fork` outputs)
- Custom Engine Intervention Operations (`register_op`)
- Custom Side-Branch Fork Runners (`register_fork_runner`)
- Custom Training Datasets (`register_dataset`)
- Custom `Lab` / `RunResult` / `FlowBuilder` Python API methods
- Custom MCP tools
- Frontend ES Module Tabs, Sub-Tabs, Palette buttons, and Inspector schemas
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ======================================================================================
# Extension Registry & Activation State
# ======================================================================================

_EXTENSIONS: dict[str, type["FeaturePlugin"]] = {}
_ENABLED_OVERRIDES: dict[str, bool] = {}
_STATE_LOADED = False


def _ext_state_file() -> Path:
    home = Path(os.environ.get("TOKENSHARK_HOME", Path(__file__).parent.parent / "data"))
    return home / "extensions.json"


def _load_state():
    global _STATE_LOADED
    if _STATE_LOADED:
        return
    _STATE_LOADED = True
    p = _ext_state_file()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    _ENABLED_OVERRIDES[str(k)] = bool(v)
        except Exception:
            pass


def _save_state():
    p = _ext_state_file()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_ENABLED_OVERRIDES, indent=2), encoding="utf-8")
    except Exception:
        pass


class FeaturePlugin:
    """Base class for every modular feature plugin / extension in Tokenshark."""

    id: str = "base_extension"
    name: str = "Base Extension"
    version: str = "0.1.0"
    summary: str = ""
    category: str = "analysis"                 # analysis | intervention | training | guardrail | dataset
    applies_to: tuple[str, ...] = ("causal_lm",)
    default_enabled: bool = True

    # UI integration: tabs, palette buttons, train sections, and optional frontend JS module
    tabs: list[str] = []
    palette_actions: list[str] = []
    train_sections: list[str] = []
    web_dir: Path | None = None
    js_entry: str | None = None                # relative to web_dir, mounted at /ext/<id>/<js_entry>

    # Optional FastAPI router (mounted automatically on the main app)
    router: Any = None

    @classmethod
    def is_enabled(cls) -> bool:
        _load_state()
        return _ENABLED_OVERRIDES.get(cls.id, cls.default_enabled)

    @classmethod
    def set_enabled(cls, enabled: bool) -> bool:
        _load_state()
        _ENABLED_OVERRIDES[cls.id] = bool(enabled)
        _save_state()
        return cls.is_enabled()

    @classmethod
    def info(cls) -> dict[str, Any]:
        js_url = None
        if cls.js_entry:
            if cls.js_entry.startswith("/"):
                js_url = cls.js_entry
            else:
                js_url = f"/ext/{cls.id}/{cls.js_entry}"
        return {
            "id": cls.id,
            "name": cls.name,
            "version": cls.version,
            "summary": cls.summary,
            "category": cls.category,
            "applies_to": list(cls.applies_to),
            "enabled": cls.is_enabled(),
            "default_enabled": cls.default_enabled,
            "tabs": list(cls.tabs),
            "palette_actions": list(cls.palette_actions),
            "train_sections": list(cls.train_sections),
            "js_url": js_url,
        }


def register_extension(cls: type[FeaturePlugin]) -> type[FeaturePlugin]:
    """Decorator that registers a `FeaturePlugin` subclass."""
    _EXTENSIONS[cls.id] = cls
    return cls


def get_extensions() -> dict[str, type[FeaturePlugin]]:
    from .extensions import discover_extensions
    return discover_extensions()


def list_extensions(family: str | None = None) -> list[dict[str, Any]]:
    exts = get_extensions()
    out = []
    for cls in exts.values():
        if family and family not in cls.applies_to and "*" not in cls.applies_to:
            continue
        out.append(cls.info())
    return out


def toggle_extension(ext_id: str, enabled: bool | None = None) -> dict[str, Any]:
    exts = get_extensions()
    if ext_id not in exts:
        raise KeyError(f"Unknown extension '{ext_id}'.")
    cls = exts[ext_id]
    new_state = (not cls.is_enabled()) if enabled is None else bool(enabled)
    cls.set_enabled(new_state)
    return cls.info()


# ======================================================================================
# Building-Block Registries (Ops, Node Kinds, Fork Runners, Datasets, API, MCP)
# ======================================================================================

@dataclass
class OpSpec:
    name: str
    fn: Callable[..., Any] | None = None
    apply: Callable[..., Any] | None = None
    description: str = ""
    valid_hooks: tuple[str, ...] | None = None
    resolve_source: Callable[[Any, Callable[[str], dict]], Any] | None = None
    validate_fn: Callable[[dict, str, dict, Callable[[str, str], None]], bool] | None = None
    default_value: float = 1.0

    def __post_init__(self):
        if self.fn is None and self.apply is not None:
            self.fn = self.apply
        elif self.apply is None and self.fn is not None:
            self.apply = self.fn


@dataclass
class NodeKindSpec:
    kind: str
    label: str
    handles: list[tuple[str, str, str, str]] | tuple[tuple[str, str, str, str], ...] = field(default_factory=list)
    default_data: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    is_fork_target: bool = False
    compile_fn: Callable[[dict, dict, list, dict], dict] | None = None

    def __post_init__(self):
        if self.kind == "fork_head":
            self.is_fork_target = True


@dataclass
class ForkRunnerSpec:
    kind: str
    run_fn: Callable[[Any, Any, Any, Any], dict | None]   # (fork_spec, resid_post, ctx, adapter) -> dict | None


@dataclass
class DatasetSpec:
    id: str
    name: str
    kind: str = "text"                         # "text" | "reasoning" | "adapt" | "fork" | "custom"
    description: str = ""
    examples: list[Any] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    sample_prompt: str = ""
    load_fn: Callable[..., list[Any]] | None = None
    preview_fn: Callable[..., list[Any]] | None = None

    def __post_init__(self):
        if self.load_fn is None:
            exs = list(self.examples)
            self.load_fn = lambda n=None, **kw: exs[:n] if n else list(exs)
        elif not self.examples:
            try:
                self.examples = list(self.load_fn())
            except Exception:
                self.examples = []
        if self.preview_fn is None:
            self.preview_fn = lambda n=8, **kw: self.examples[:n]


_OPS: dict[str, OpSpec] = {}
_NODE_KINDS: dict[str, NodeKindSpec] = {}
_FORK_RUNNERS: dict[str, ForkRunnerSpec] = {}
_DATASETS: dict[str, DatasetSpec] = {}
_LAB_METHODS: dict[str, Callable[..., Any]] = {}
_RUN_METHODS: dict[str, Callable[..., Any]] = {}
_FLOW_METHODS: dict[str, Callable[..., Any]] = {}
_MCP_TOOLS: dict[str, dict[str, Any]] = {}


def register_op(
    name: str | OpSpec,
    fn: Callable[..., Any] | None = None,
    *,
    description: str = "",
    valid_hooks: tuple[str, ...] | None = None,
    resolve_source: Callable[[Any, Callable[[str], dict]], Any] | None = None,
    validate_fn: Callable[[dict, str, dict, Callable[[str, str], None]], bool] | None = None,
    default_value: float = 1.0,
):
    """Register a custom intervention operation for `engine.apply_op` and `flow.compile_flow`."""
    if isinstance(name, OpSpec):
        _OPS[name.name] = name
        return name
    if fn is None:
        raise ValueError("register_op requires either an OpSpec or (name, fn).")
    _OPS[name] = OpSpec(
        name=name,
        fn=fn,
        valid_hooks=valid_hooks,
        resolve_source=resolve_source,
        validate_fn=validate_fn,
        default_value=default_value,
    )


def get_op(name: str) -> OpSpec | None:
    return _OPS.get(name)


def list_ops() -> dict[str, OpSpec]:
    return dict(_OPS)


def register_node_kind(
    kind: str | NodeKindSpec,
    label: str = "",
    *,
    handles: list[tuple[str, str, str, str]] | None = None,
    default_data: dict[str, Any] | None = None,
    description: str = "",
    is_fork_target: bool = False,
    compile_fn: Callable[[dict, dict, list, dict], dict] | None = None,
):
    """Register a custom box kind for the flow graph compiler (`flow.py`)."""
    if isinstance(kind, NodeKindSpec):
        _NODE_KINDS[kind.kind] = kind
        return kind
    spec = NodeKindSpec(
        kind=kind,
        label=label or kind,
        handles=handles or [("in", "target", "left", ""), ("out", "source", "right", "")],
        default_data=default_data or {},
        description=description,
        is_fork_target=is_fork_target or (kind == "fork_head"),
        compile_fn=compile_fn,
    )
    _NODE_KINDS[kind] = spec
    return spec


def get_node_kind(kind: str) -> NodeKindSpec | None:
    return _NODE_KINDS.get(kind)


def list_node_kinds() -> dict[str, NodeKindSpec]:
    return dict(_NODE_KINDS)


def register_fork_runner(kind: str, run_fn: Callable[[Any, Any, Any, Any], dict | None]):
    """Register a handler that executes when a block with a side-branch `fork` edge finishes."""
    _FORK_RUNNERS[kind] = ForkRunnerSpec(kind=kind, run_fn=run_fn)


def get_fork_runner(kind: str) -> ForkRunnerSpec | None:
    return _FORK_RUNNERS.get(kind)


def register_dataset(
    id: str | DatasetSpec,
    name: str = "",
    *,
    kind: str = "text",
    description: str = "",
    examples: list[Any] | None = None,
    rows: list[dict[str, str]] | None = None,
    system_prompt: str = "",
    sample_prompt: str = "",
    load_fn: Callable[..., list[Any]] | None = None,
    preview_fn: Callable[..., list[Any]] | None = None,
):
    """Register a training dataset provider (exposed in `/api/datasets` and usable in training jobs)."""
    if isinstance(id, DatasetSpec):
        _DATASETS[id.id] = id
        return id
    spec = DatasetSpec(
        id=id,
        name=name or id,
        kind=kind,
        description=description,
        examples=examples or [],
        rows=rows or [],
        system_prompt=system_prompt,
        sample_prompt=sample_prompt,
        load_fn=load_fn,
        preview_fn=preview_fn,
    )
    _DATASETS[id] = spec
    return spec


def get_dataset(dataset_id: str) -> DatasetSpec | None:
    from .extensions import discover_extensions
    discover_extensions()
    return _DATASETS.get(dataset_id)


def list_datasets(kind: str | None = None) -> list[dict[str, Any]]:
    from .extensions import discover_extensions
    discover_extensions()
    out = []
    for ds in _DATASETS.values():
        if kind and ds.kind != kind and ds.kind != "custom":
            continue
        out.append({
            "id": ds.id,
            "name": ds.name,
            "kind": ds.kind,
            "description": ds.description,
            "n_examples": len(ds.rows) if ds.rows else len(ds.examples),
            "system_prompt": ds.system_prompt,
            "sample_prompt": ds.sample_prompt,
        })
    return out


def register_lab_method(name: str, fn: Callable[..., Any]):
    """Attach a method dynamically to `tokenshark.api.Lab`."""
    _LAB_METHODS[name] = fn
    try:
        from .api import Lab
        if not hasattr(Lab, name):
            setattr(Lab, name, fn)
    except Exception:
        pass


def register_run_method(name: str, fn: Callable[..., Any]):
    """Attach a method dynamically to `tokenshark.api.RunResult`."""
    _RUN_METHODS[name] = fn
    try:
        from .api import RunResult
        if not hasattr(RunResult, name):
            setattr(RunResult, name, fn)
    except Exception:
        pass


def register_flow_method(name: str, fn: Callable[..., Any]):
    """Attach a method dynamically to `tokenshark.flowbuild.FlowBuilder`."""
    _FLOW_METHODS[name] = fn
    try:
        from .flowbuild import FlowBuilder
        if not hasattr(FlowBuilder, name):
            setattr(FlowBuilder, name, fn)
    except Exception:
        pass


def register_mcp_tool(
    name: str,
    fn: Callable[..., Any],
    description: str = "",
    properties: dict | None = None,
    required: list[str] | None = None,
):
    """Register an MCP tool exposed by `tokenshark.mcp_server`."""
    _MCP_TOOLS[name] = {
        "fn": fn,
        "spec": {
            "name": name,
            "description": description,
            "inputSchema": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


def get_mcp_tools() -> dict[str, dict[str, Any]]:
    return dict(_MCP_TOOLS)
