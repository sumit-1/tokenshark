"""FastAPI server package (`tokenshark.server`).

Splits the server into focused modules:
- `app.py`: Core FastAPI app, workspace sync, task routes, extension discovery/toggle endpoints
- `models_api.py`: Model registry & LoRA endpoints
- `runs_api.py`: Run creation, polling, logit lens, word cloud, word tracking, attention, image stage maps
- `kv_api.py`: KV-cache library & hidden prefix endpoints
- `debug_api.py`: Stepper debugger endpoints
"""
from __future__ import annotations

from fastapi.staticfiles import StaticFiles

from .app import (
    DEBUG,
    WEB,
    WORKSPACE,
    _analysis,
    _bump_workspace,
    _caps,
    _clean,
    _ensure_workspace,
    _load_prefixes,
    _need,
    _require_model,
    _rss_gb,
    _run,
    _sources,
    app,
)
from . import debug_api, kv_api, models_api, runs_api  # noqa: F401
from .. import labapi  # noqa: F401
from ..extensions import discover_extensions
from ..sdk import _EXTENSIONS


def mount_extensions():
    """Discover all extensions and mount their FastAPI routers and static web directories."""
    discover_extensions()
    existing_routes = {getattr(r, "path", None) for r in app.routes}
    for ext_id, ext in _EXTENSIONS.items():
        if ext.router is not None and not getattr(ext, "_router_mounted", False):
            app.include_router(ext.router)
            ext._router_mounted = True
        if ext.web_dir is not None and ext.web_dir.is_dir():
            mount_path = f"/ext/{ext_id}"
            if mount_path not in existing_routes:
                app.mount(mount_path, StaticFiles(directory=ext.web_dir), name=f"ext_{ext_id}")


mount_extensions()

__all__ = [
    "app",
    "WEB",
    "DEBUG",
    "WORKSPACE",
    "_ensure_workspace",
    "_bump_workspace",
    "_require_model",
    "_need",
    "_run",
    "_caps",
    "_rss_gb",
    "_load_prefixes",
    "_sources",
    "_clean",
    "_analysis",
    "mount_extensions",
]
