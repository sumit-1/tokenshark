"""Core FastAPI application, shared workspace state, and helper functions (`tokenshark.server.app`)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import STARTED_WITH, __version__, code_stamp, interp
from ..flow import FlowError, compile_flow
from ..plugins import list_plugins
from ..registry import LOADED
from ..runs import STORE, Run
from ..tasks import TASKS
from ..weights import models_dir

app = FastAPI(title="Tokenshark", version=__version__)
WEB = Path(__file__).parent.parent / "web"

DEBUG: dict = {}

# Unified shared workspace state so all browser clients and MCP agents see the exact same flow, active run & tab.
_WS_LOCK = threading.Lock()
WORKSPACE: dict = {
    "version": 0,
    "model_id": None,
    "flow": None,
    "active_run_id": None,
    "tab": "output",
    "selected": None,
    "client_id": "",
    "actor": "init",
    "notice": "",
}


def _ensure_workspace() -> dict:
    with _WS_LOCK:
        if LOADED.info is not None and LOADED.plugin is not None:
            mid = LOADED.info.get("model_id")
            if WORKSPACE["flow"] is None or WORKSPACE["model_id"] != mid:
                import copy
                WORKSPACE["model_id"] = mid
                WORKSPACE["flow"] = copy.deepcopy(LOADED.plugin.default_flow(LOADED.info))
                WORKSPACE["active_run_id"] = None
                WORKSPACE["version"] += 1
        elif LOADED.info is None and WORKSPACE["model_id"] is not None:
            WORKSPACE["model_id"] = None
            WORKSPACE["flow"] = None
            WORKSPACE["active_run_id"] = None
            WORKSPACE["version"] += 1
        return dict(WORKSPACE)


def _bump_workspace(*, flow: dict | None = None, active_run_id: str | None = None, tab: str | None = None,
                    selected: dict | None = None, client_id: str = "", actor: str = "", notice: str = "") -> int:
    import copy
    _ensure_workspace()
    with _WS_LOCK:
        changed = False
        if flow is not None and flow != WORKSPACE["flow"]:
            WORKSPACE["flow"] = copy.deepcopy(flow)
            changed = True
        if active_run_id is not None and active_run_id != WORKSPACE["active_run_id"]:
            WORKSPACE["active_run_id"] = active_run_id
            changed = True
        if tab is not None and tab != WORKSPACE["tab"]:
            WORKSPACE["tab"] = tab
            changed = True
        if selected is not None and selected != WORKSPACE["selected"]:
            WORKSPACE["selected"] = selected
            changed = True
        if changed or notice:
            WORKSPACE["client_id"] = client_id
            if actor:
                WORKSPACE["actor"] = actor
            if notice:
                WORKSPACE["notice"] = notice
            WORKSPACE["version"] += 1
        return WORKSPACE["version"]


def _require_model():
    if LOADED.adapter is None or LOADED.info is None:
        raise HTTPException(409, "No model loaded.")
    return LOADED.adapter, LOADED.info


def _need(feature: str, what: str):
    """The loaded model, if its plugin supports `feature` (plugins list their features; see plugins/__init__.py)."""
    adapter, info = _require_model()
    if feature not in getattr(LOADED.plugin, "features", ()):
        raise HTTPException(400, f"{what} is not available for this kind of model ({getattr(LOADED.plugin, 'name', '?')}).")
    return adapter, info


def _run(run_id: str) -> Run:
    try:
        return STORE.get(run_id)
    except KeyError:
        raise HTTPException(404, "Unknown run.")


def _caps(run: Run) -> dict:
    if run.captures is None:
        raise HTTPException(410, "Captures for this run were evicted or it did not finish. Re-run it.")
    if LOADED.spec is None or LOADED.spec["id"] != run.model_id:
        raise HTTPException(409, "Load this run's model to analyse it.")
    return run.captures


def _rss_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**30, 2)
    except Exception:
        return None


def _load_prefixes(plan, adapter):
    """All KV prefixes the flow prepends - saved caches and layer prompts - checked against this model and schedule."""
    import json as _json
    try:
        with LOADED.lock:
            return interp.load_prefixes(plan, adapter, _json.dumps(LOADED.overlay_state(), sort_keys=True))
    except ValueError as e:
        raise HTTPException(400, str(e))


def _sources(plan) -> dict:
    """What patch / steer / clamp boxes need: another run's activations, a steering vector, a neuron's range."""
    try:
        return interp.resolve_sources(plan, lambda rid: STORE.get(rid).captures)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _clean(part):
    if isinstance(part, dict):
        part = {k: v for k, v in part.items() if k not in ("_progress", "todo")}
    return part


def _analysis(r: Run, label: str, group: str, key, gen_fn, wait: bool, weights=None):
    """Run an analysis as a task (one layer per unit of work, see tasks.py)."""
    cache = r.cache
    if key in cache:
        return cache[key] if wait else {"task": None, "result": cache[key], "status": "done"}
    t = TASKS.new("analysis", label, group=group, run_id=r.id)

    def gen():
        last = None
        for last in gen_fn(cache.setdefault("memo", {})):
            yield last
        cache[key] = _clean(last)

    TASKS.start(t, gen, weights=weights if weights is not None else LOADED.adapter.model.W)
    if not wait:
        return {"task": t.summary(), "result": t.result, "status": t.status}
    while t.live:
        time.sleep(0.05)
    if t.status == "error":
        raise HTTPException(400, t.error)
    if t.status == "cancelled":
        raise HTTPException(409, "Stopped.")
    return cache[key]


class FlowReq(BaseModel):
    flow: dict
    client_id: str = ""
    sync: bool = False


class WorkspaceUpdateReq(BaseModel):
    flow: dict | None = None
    active_run_id: str | None = None
    tab: str | None = None
    selected: dict | None = None
    client_id: str = ""
    actor: str = "browser"
    notice: str = ""
    reset: bool = False


class WorkspaceEditReq(BaseModel):
    prompt: str | None = None
    system: str | None = None
    prefill: str | None = None
    chat: bool | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    edits: list[dict] | None = None
    reset: bool = False
    tab: str | None = None
    selected: dict | None = None
    actor: str = "mcp"
    notice: str = ""


class ToggleExtensionReq(BaseModel):
    enabled: bool


@app.get("/api/status")
def status():
    from ..sdk import list_extensions
    ws = _ensure_workspace()
    return {
        "version": __version__,
        "device": LOADED.device,
        "dtype": str(LOADED.dtype).replace("torch.", ""),
        "status": LOADED.status,
        "error": LOADED.error,
        "model": LOADED.info,
        "models_dir": str(models_dir()),
        "progress": LOADED.progress,
        "load_seconds": round(time.time() - getattr(LOADED, "load_started", time.time()), 0)
        if LOADED.status.startswith("loading") else None,
        "stale": code_stamp() != STARTED_WITH,
        "ram_gb": _rss_gb(),
        "debug": DEBUG.get("state") if DEBUG else None,
        "workspace_version": ws["version"],
        "active_run_id": ws["active_run_id"],
        "tab": ws["tab"],
        "extensions": list_extensions(),
    }


@app.get("/api/plugins")
def plugins_endpoint():
    return list_plugins()


@app.get("/api/extensions")
def extensions_endpoint():
    from ..sdk import list_extensions
    return list_extensions()


@app.post("/api/extensions/{ext_id}/toggle")
def toggle_extension_endpoint(ext_id: str, req: ToggleExtensionReq):
    from ..sdk import toggle_extension
    try:
        return toggle_extension(ext_id, req.enabled)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/flow/default")
def default_flow(reset: bool = False, client_id: str = ""):
    _, info = _require_model()
    df = LOADED.plugin.default_flow(info)
    if reset:
        _bump_workspace(flow=df, client_id=client_id, actor="browser", notice="Flow reset")
    return df


@app.post("/api/flow/compile")
def compile_(req: FlowReq):
    _, info = _require_model()
    try:
        plan = compile_flow(req.flow, info)
    except FlowError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)
    ws_ver = WORKSPACE["version"]
    if req.sync:
        ws_ver = _bump_workspace(flow=req.flow, client_id=req.client_id, actor="browser")
    return {"ok": True, "plan": plan.to_dict(), "workspace_version": ws_ver}


@app.get("/api/workspace")
def get_workspace(since: int = -1):
    ws = _ensure_workspace()
    changed = ws["version"] != since
    out = {
        "version": ws["version"],
        "changed": changed,
        "model_id": ws["model_id"],
        "active_run_id": ws["active_run_id"],
        "tab": ws["tab"],
        "selected": ws["selected"],
        "client_id": ws["client_id"],
        "actor": ws["actor"],
        "notice": ws["notice"],
    }
    if changed:
        out["flow"] = ws["flow"]
    return out


@app.post("/api/workspace")
def update_workspace(req: WorkspaceUpdateReq):
    _ensure_workspace()
    flow_to_set = req.flow
    if req.reset and LOADED.info is not None and LOADED.plugin is not None:
        flow_to_set = LOADED.plugin.default_flow(LOADED.info)
    ver = _bump_workspace(
        flow=flow_to_set,
        active_run_id=req.active_run_id,
        tab=req.tab,
        selected=req.selected,
        client_id=req.client_id,
        actor=req.actor,
        notice=req.notice,
    )
    return get_workspace(since=-1 if ver else -1)


@app.post("/api/workspace/edit")
def edit_workspace(req: WorkspaceEditReq):
    _, info = _require_model()
    import copy
    from ..flowbuild import FlowBuilder
    ws = _ensure_workspace()
    base = LOADED.plugin.default_flow(info) if (req.reset or not ws.get("flow")) else copy.deepcopy(ws["flow"])
    fb = FlowBuilder(base)
    if any(v is not None for v in (req.prompt, req.system, req.prefill, req.chat)):
        fb.prompt(prompt=req.prompt, system=req.system, prefill=req.prefill, chat=req.chat)
    if any(v is not None for v in (req.max_new_tokens, req.temperature, req.top_k, req.top_p, req.seed)):
        fb.sampler(max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                   top_k=req.top_k, top_p=req.top_p, seed=req.seed)
    try:
        fb.apply_edits(req.edits)
        flow_dict = fb.to_dict()
        plan = compile_flow(flow_dict, info)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FlowError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)
    summary_parts = []
    if req.prompt is not None:
        summary_parts.append(f"prompt={req.prompt[:28]!r}")
    if req.system is not None:
        summary_parts.append(f"system={req.system[:28]!r}")
    if req.edits:
        summary_parts.append(", ".join(str(e.get("do", "edit")) for e in req.edits))
    notice = req.notice or (f"{req.actor}: " + (" · ".join(summary_parts) if summary_parts else "updated flow"))
    ver = _bump_workspace(
        flow=flow_dict,
        tab=req.tab,
        selected=req.selected,
        client_id="",
        actor=req.actor or "mcp",
        notice=notice,
    )
    return {"ok": True, "version": ver, "flow": flow_dict, "plan": plan.to_dict(), "tab": WORKSPACE["tab"], "notice": notice}


# ---------------------------------------------------------------- task manager
@app.get("/api/tasks")
def list_tasks():
    return TASKS.list()


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    t = TASKS.tasks.get(task_id)
    if t is None:
        raise HTTPException(404, "Unknown task.")
    return {**t.summary(), "result": t.result}


@app.post("/api/tasks/{task_id}/{action}")
def task_action(task_id: str, action: str):
    try:
        return TASKS.act(task_id, action)
    except KeyError:
        raise HTTPException(404, "Unknown task.")
    except ValueError as e:
        raise HTTPException(400, str(e))


class DecodeReq(BaseModel):
    ids: list[int]


@app.post("/api/tokens/decode")
def decode_tokens(req: DecodeReq):
    adapter, info = _need("lm_tools", "Token decoding")
    return {"text": adapter.tok.decode(req.ids, skip_special=False)}


# ---------------------------------------------------------------- web UI (static, no build step)
@app.middleware("http")
async def _same_origin(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlparse
            if urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Requests from other sites are not accepted."}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def _no_stale_ui(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/static/", "/ext/")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/favicon.ico")
def favicon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="6" fill="#161a20"/>'
           '<circle cx="9" cy="16" r="4" fill="#5b9dff"/><circle cx="23" cy="9" r="4" fill="#3ecf8e"/>'
           '<circle cx="23" cy="23" r="4" fill="#ff9f43"/><path d="M12 15 L20 10 M12 17 L20 22" stroke="#dde3ea" stroke-width="2"/></svg>')
    return Response(svg, media_type="image/svg+xml")
