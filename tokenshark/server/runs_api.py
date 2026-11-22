"""Runs, logit lens, word cloud, word tracking, attention, and image stage map endpoints (`tokenshark.server.runs_api`)."""
from __future__ import annotations

import threading
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..caches import LIBRARY
from ..engine import Cancelled
from ..flow import FlowError, compile_flow
from ..registry import LOADED
from ..runs import STORE, Run
from ..tasks import TASKS
from .app import (
    _analysis,
    _bump_workspace,
    _caps,
    _load_prefixes,
    _need,
    _require_model,
    _run,
    _sources,
    app,
)


class RunReq(BaseModel):
    flow: dict
    parent_id: str | None = None
    label: str = ""
    resume: dict | None = None       # diffusion: {"run_id", "step"}
    client_id: str = ""
    actor: str = ""


class SaveCacheReq(BaseModel):
    name: str = ""


class ContribReq(BaseModel):
    steps: str = "all"


@app.post("/api/runs")
def create_run(req: RunReq):
    adapter, info = _require_model()
    try:
        plan = compile_flow(req.flow, info)
    except FlowError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)
    resume, prefix = None, None
    parent_id, resume_step = req.parent_id, None
    if req.resume:
        if not info["capabilities"].get("resume_from_step"):
            raise HTTPException(400, "This model can't branch from a step.")
        parent = _run(req.resume["run_id"])
        pc = _caps(parent)
        k = int(req.resume["step"])
        if parent.plan.params.get("steps") != plan.params.get("steps"):
            raise HTTPException(400, f"Branching from a step needs the parent's step count "
                                     f"({parent.plan.params.get('steps')}).")
        if k not in pc["latents_before"]:
            raise HTTPException(400, f"Parent has no latent at step {k}.")
        resume = {"step": k, "latent": pc["latents_before"][k], "parent_latents": pc["latents_before"],
                  "parent_x0": pc["x0"]}
        parent_id, resume_step = parent.id, k
    if plan.params.get("prefix_cache"):
        prefix = _load_prefixes(plan, adapter)
    overlays = LOADED.overlay_state()
    if overlays:
        plan.params["overlays"] = overlays
    run = Run(LOADED.spec["id"], req.flow, plan, parent_id, req.label, resume_step)
    run.sources = _sources(plan)
    STORE.add(run)
    STORE.execute(run, LOADED, resume, prefix)
    actor = req.actor or ("mcp" if req.label == "tutor" and not req.client_id else "browser")
    ws_ver = _bump_workspace(
        flow=req.flow,
        active_run_id=run.id,
        client_id=req.client_id,
        actor=actor,
        notice=f"{'MCP' if actor == 'mcp' else 'Run'} started {run.id}" + (f" ({req.label})" if req.label else ""),
    )
    return {"ok": True, "run": run.summary(), "plan": plan.to_dict(), "workspace_version": ws_ver}


@app.get("/api/runs")
def list_runs():
    return STORE.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, since: int = -1):
    r = _run(run_id)
    out = {**r.summary(), "plan": r.plan.to_dict(), "flow": r.flow}
    if r.version != since:
        out["outputs"] = r.outputs
    return out


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str):
    r = _run(run_id)
    if r.task is not None and r.task.live:
        TASKS.act(r.task.id, "cancel")
    elif r.ctx is not None:
        r.ctx.cancel = True
    return r.summary()


@app.post("/api/runs/{run_id}/pause")
def pause(run_id: str):
    r = _run(run_id)
    if r.task is None or not r.task.live:
        raise HTTPException(409, "This run is not running.")
    TASKS.act(r.task.id, "pause")
    return r.summary()


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str):
    r = _run(run_id)
    if r.task is not None and r.task.live:
        TASKS.act(r.task.id, "resume")
    return r.summary()


@app.get("/api/runs/{run_id}/manifest")
def manifest(run_id: str):
    return _run(run_id).manifest()


@app.get("/api/runs/{run_id}/lens")
def lens(run_id: str, k: int = 5, through: int = 0, background: bool = False):
    r = _run(run_id)
    cap = _caps(r)
    _need("lm_tools", "Logit lens")
    ad = LOADED.adapter
    through = max(0, through)
    return _analysis(r, f"logit lens{f' (through {through})' if through else ''}", "lens", ("lens", k, through),
                     lambda memo: ad.iter_logit_lens(cap, r.plan.schedule_nodes, r.plan.schedule, k, through, memo),
                     not background)


@app.get("/api/runs/{run_id}/cloud")
def cloud(run_id: str, position: int = -1, layers: str = "", through: int = 0, top: int = 40, background: bool = False):
    r = _run(run_id)
    cap = _caps(r)
    _need("lm_tools", "Word clouds")
    S = len(r.plan.schedule)
    try:
        lay = [int(x) for x in layers.split(",") if x.strip() != ""] or [S - 1]
    except ValueError:
        raise HTTPException(400, "layers = comma-separated schedule positions (-1 = embeddings)")
    lay = [l for l in lay if -1 <= l < S]
    through, top = max(0, through), max(5, min(top, 100))
    ad = LOADED.adapter
    T = len(cap["ids"])
    tpos = min(max(position if position >= 0 else T + position, 0), T - 1)
    return _analysis(r, f"word cloud @#{tpos} · {len(lay)} layer(s){f' · through {through}' if through else ''}", "cloud",
                     ("cloud", tpos, tuple(lay), through, top),
                     lambda memo: ad.iter_word_cloud(cap, r.plan.schedule_nodes, r.plan.schedule, tpos, lay, through, top, memo),
                     not background)


@app.get("/api/runs/{run_id}/track")
def track(run_id: str, word: str, position: int = -1, through: int = 0, background: bool = False):
    r = _run(run_id)
    cap = _caps(r)
    _need("lm_tools", "Word tracking")
    ad = LOADED.adapter
    through = max(0, through)
    return _analysis(r, f"track '{word}'", "track", ("track", word, position, through),
                     lambda memo: ad.iter_track_word(cap, r.plan.schedule_nodes, r.plan.schedule, position, word, through, memo),
                     not background)


@app.get("/api/runs/{run_id}/stagemaps")
def stagemaps(run_id: str, step: int = -1):
    r = _run(run_id)
    cap = _caps(r)
    _need("image_maps", "Stage maps")
    steps = sorted({s for v in cap.get("stage_maps", {}).values() for s in v})
    if not steps:
        raise HTTPException(404, "No stage maps captured.")
    st = steps[-1] if step < 0 or step not in steps else step
    key = ("stagemaps", st)
    if key not in r.cache:
        r.cache[key] = {**LOADED.adapter.stage_maps(cap, st, r.plan.schedule_nodes), "steps": steps}
    return r.cache[key]


@app.post("/api/runs/{run_id}/contributions")
def start_contributions(run_id: str, req: ContribReq):
    r = _run(run_id)
    _caps(r)
    _need("image_maps", "Block contributions")
    if not (LOADED.info or {}).get("capabilities", {}).get("contributions", True):
        try:
            LOADED.adapter.contributions()
        except ValueError as e:
            raise HTTPException(400, str(e))
    job = r.cache.get(("contrib", req.steps))
    if job and job["status"] in ("running", "done"):
        return job
    job = {"status": "running", "progress": 0.0, "message": "starting", "steps": req.steps, "results": None}
    r.cache[("contrib", req.steps)] = job
    r.cache["contrib_last"] = req.steps

    def work():
        def prog(f, m):
            job["progress"], job["message"] = round(f, 3), m
        try:
            with LOADED.lock:
                base = r.captures["image_png"]
                if r.resume_step is not None:
                    import base64 as b64
                    res = LOADED.adapter.run(r.plan, ctx_hook=lambda c: c.store.update(maps=False))
                    base = b64.b64decode(res["outputs"]["image"].split(",", 1)[1])
                job["results"] = LOADED.adapter.contributions(r.plan, base, req.steps, prog)
            job["status"] = "done"
        except Exception as e:
            job["status"], job["message"] = "error", f"{type(e).__name__}: {e}"

    threading.Thread(target=work, daemon=True).start()
    return job


@app.get("/api/runs/{run_id}/contributions")
def get_contributions(run_id: str, steps: str = ""):
    r = _run(run_id)
    key = ("contrib", steps or r.cache.get("contrib_last", "all"))
    return r.cache.get(key) or {"status": "none"}


@app.get("/api/runs/{run_id}/attention")
def attention(run_id: str, pos: int, head: int = -1):
    r = _run(run_id)
    return LOADED.adapter.attention(_caps(r), pos, head)


@app.get("/api/runs/{run_id}/step/{step}.png")
def step_png(run_id: str, step: int, kind: str = "x0"):
    r = _run(run_id)
    cap = _caps(r)
    fit_step = cap.get("rgb_fit_step", -1)
    key = ("step", step, kind, fit_step)
    if key not in r.cache:
        try:
            if kind == "xt":
                if not hasattr(LOADED.adapter, "step_input_preview"):
                    raise KeyError(step)
                with LOADED.lock:
                    r.cache[key] = LOADED.adapter.step_input_preview(cap, step)
            else:
                with LOADED.lock:
                    r.cache[key] = LOADED.adapter.step_preview(cap, step)
        except KeyError:
            raise HTTPException(404, "No latent for that step.")
    return Response(r.cache[key], media_type="image/png")


@app.get("/api/runs/{run_id}/imagelens/compute")
def image_lens_compute(run_id: str, through: int = 1, background: bool = False):
    r = _run(run_id)
    cap = _caps(r)
    if not hasattr(LOADED.adapter, "iter_lens_through") or "lens_ctx" not in cap:
        raise HTTPException(400, "Decoding through the last blocks needs a DiT image model run.")
    through = max(1, min(through, len(r.plan.schedule)))
    ad = LOADED.adapter
    return _analysis(r, f"image lens through last {through}", "ilens", ("ilens", through),
                     lambda memo: ad.iter_lens_through(cap, r.plan, through), not background, weights=ad.dit.W)


@app.get("/api/runs/{run_id}/imagelens/{step}/{pos}.png")
def image_lens(run_id: str, step: int, pos: int, exact: bool = False, through: int = 0):
    r = _run(run_id)
    cap = _caps(r)
    if not hasattr(LOADED.adapter, "lens_png"):
        raise HTTPException(400, "The image lens needs a transformer (DiT) image model.")
    fit_step = 0 if exact else cap.get("rgb_fit_step", -1)
    key = ("lens", step, pos, exact, through, fit_step)
    if key not in r.cache:
        try:
            if not exact:
                with LOADED.lock:
                    r.cache[key] = LOADED.adapter.lens_png(cap, step, pos, through=through)
            else:
                t = TASKS.new("analysis", f"image lens: decode block {pos} @ step {step}", run_id=r.id)
                try:
                    with TASKS.slot(t):
                        r.cache[key] = LOADED.adapter.lens_png(cap, step, pos, exact=True, through=through)
                    TASKS.finish(t)
                except Cancelled:
                    TASKS.finish(t, "cancelled")
                    raise HTTPException(409, "Stopped.")
                except Exception as e:
                    TASKS.finish(t, error=str(e))
                    raise
        except KeyError:
            raise HTTPException(404, "Not captured for that step / block.")
    return Response(r.cache[key], media_type="image/png")


@app.get("/api/runs/{run_id}/xattn")
def xattn(run_id: str, token: int, group: str = "all", step: int = -1):
    r = _run(run_id)
    cap = _caps(r)
    if not 0 <= token < cap.get("n_tokens", 0):
        raise HTTPException(400, "Token index out of range.")
    return LOADED.adapter.xattn_map(cap, group, step, token)


@app.post("/api/runs/{run_id}/cache")
def save_run_cache(run_id: str, req: SaveCacheReq):
    r = _run(run_id)
    if r.captures is None or "cache" not in r.captures:
        raise HTTPException(410, "This run has no KV cache in memory (evicted, failed, or not an LLM run).")
    if r.captures.get("consistent") is False:
        raise HTTPException(409, "A token is half-way through the blocks. Step to the end of the token "
                                 "(Token ▸▸) before saving the KV cache.")
    return LIBRARY.save(r.captures["cache"], req.name or f"run {r.id}",
                        {"model_id": r.model_id, "source": "run", "run_id": r.id, "schedule": r.plan.schedule})
