"""Debugger (LLM & DiT stepping) endpoints (`tokenshark.server.debug_api`)."""
from __future__ import annotations

import threading
import time
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..engine import Cancelled
from ..flow import FlowError, compile_flow
from ..registry import LOADED
from ..runs import STORE, Run, run_loop_of
from ..tasks import TASKS
from .app import (
    DEBUG,
    _load_prefixes,
    _require_model,
    _rss_gb,
    _sources,
    app,
)


class DebugStartReq(BaseModel):
    flow: dict
    label: str = ""
    background: bool = False


class DebugStepReq(BaseModel):
    mode: str = "block"          # block | token | to | all
    pos: int | None = None       # schedule position for mode=to
    force_token: int | None = None
    count: int = 1
    flow: dict | None = None     # optional updated flow


class ExtendStepsReq(BaseModel):
    steps: int | None = None
    add: int = 4


class NoiseReq(BaseModel):
    sigma: float
    fresh_noise: bool = False


class EditReq(BaseModel):
    image: str
    strength: float = 1.0
    only_changed: bool = True
    mask: str = ""
    sigma: float | None = None
    edit_noise: float | None = None
    fresh_noise: bool = False
    steer_steps: int = 0
    steer_pull: float = 0.45


class PromptSteerReq(BaseModel):
    prompt: str
    blocks: str = "all"


class WordScaleReq(BaseModel):
    token: int
    scale: float = 1.0
    blocks: str = "all"


def _debug():
    if not DEBUG:
        raise HTTPException(409, "No debug session. Start one first.")
    if LOADED.spec is None or LOADED.spec["id"] != DEBUG.get("model_id"):
        DEBUG.clear()
        raise HTTPException(409, "The model changed; start a new debug session.")
    if DEBUG.get("stepper") is None:
        raise HTTPException(409, "Still preparing the stepping session; wait a moment.")
    return DEBUG["stepper"]


def _image_debug():
    st = _debug()
    if not hasattr(st, "edit_guess"):
        raise HTTPException(400, "This is not an image stepping session.")
    return st


def _publish(st):
    """Copy the stepper's current state into its live Run and return the debugger state."""
    run: Run | None = DEBUG.get("run")
    if run is None:
        return st.state()
    try:
        with LOADED.lock:
            res = st.snapshot()
        run.outputs, run.captures = res["outputs"], res["captures"]
        run.version += 1
    except ValueError:
        pass
    run.cache = {}
    run.progress = 1.0 if st.done else round(getattr(st, "n_generated", 0) / max(1, getattr(st, "total", 1)), 3)
    if st.done:
        run.status, run.message = "done", f"done · {st.steps} steps"
        run.active_node, run.done_nodes = None, []
        _save_manifest(run)
    else:
        run.status, run.message = "stepping", f"stepping · {st.steps} steps"
        run.active_node = (
            run.plan.schedule_nodes[st.bp]
            if getattr(st, "x", None) is not None and st.bp < len(run.plan.schedule_nodes)
            else None
        )
        run.done_nodes = [t["node"] for t in getattr(st, "trace", []) if "node" in t]
    STORE.add(run)
    DEBUG["state"] = {**st.state(), "run_id": run.id}
    return DEBUG["state"]


def _close_live(status: str):
    run = DEBUG.get("run")
    if run is not None and run.status == "stepping":
        run.status = status
        run.message = status


def _save_manifest(run: Run):
    import json as _json
    from ..runs import RUNS_DIR
    try:
        (RUNS_DIR / f"{run.id}.json").write_text(_json.dumps(
            {**run.manifest(), "status": run.status, "outputs": run.outputs}, indent=1, default=str))
    except Exception:
        pass


@app.post("/api/debug/start")
def debug_start(req: DebugStartReq):
    adapter, info = _require_model()
    loop = run_loop_of(LOADED)
    if loop == "plain":
        raise HTTPException(400, f"Models of this kind ({LOADED.plugin.name}) can't be stepped through.")
    image = loop == "image_stepper"
    try:
        plan = compile_flow(req.flow, info)
    except FlowError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)
    prefix = _load_prefixes(plan, adapter) if (not image and plan.params.get("prefix_cache")) else None
    sources = _sources(plan) if not image else {}
    if DEBUG.get("busy"):
        if DEBUG.get("task") in TASKS.tasks:
            try:
                TASKS.act(DEBUG["task"], "cancel")
            except Exception:
                pass
        if DEBUG.get("stepper") is not None:
            DEBUG["stepper"].ctx.cancel = True
        if DEBUG.get("run") is not None and DEBUG["run"].ctx is not None:
            DEBUG["run"].ctx.cancel = True
    _close_live("stopped")
    DEBUG.clear()
    run = Run(LOADED.spec["id"], req.flow, plan, None, req.label or "stepping", None)
    run.status, run.message = "stepping", "preparing prompt…"
    STORE.add(run)

    if image and req.background:
        n_steps = int(plan.params.get("steps", 24))
        init_state = {
            "image": True, "done": False, "phase": "idle", "next_pos": None, "active_node": "text_encoder",
            "trace": [], "steps": 0, "step": 0, "n_steps": n_steps, "start": 0, "sigma": 1.0,
            "n_generated": 0, "guess_at": None, "edits": [], "can_edit": False,
            "tokens": [], "forced": None, "chunk": [], "n_in": 0, "text": "", "run_id": run.id,
        }
        task = TASKS.new("debug", "prepare stepping (encode prompt)", run_id=run.id)
        DEBUG.update(stepper=None, model_id=LOADED.spec["id"], flow=req.flow, label=req.label or "debug",
                     run=run, busy=True, task=task.id, state=init_state,
                     progress={"frac": 0.02, "message": "encoding prompt through Qwen3-VL & 32 DiT blocks…", "node": "text_encoder"})

        def init_prog(frac, msg):
            f = min(0.99, max(0.02, frac / 0.15))
            node = "denoiser_in" if "through block" in msg else ("noise" if "VAE" in msg else "text_encoder")
            DEBUG["progress"] = {"frac": round(f, 3), "message": msg, "node": node}
            task.progress, task.message = round(f, 3), msg

        def init_work():
            err = None
            st = None
            try:
                with TASKS.slot(task):
                    st = adapter.image_stepper(plan, progress=init_prog)
            except Exception as e:
                import traceback
                traceback.print_exc()
                err = f"{type(e).__name__}: {e}"
            finally:
                TASKS.finish(task, "error" if err else "done", err)
            if DEBUG.get("run") is run:
                if st is not None:
                    DEBUG["stepper"] = st
                    _publish(st)
                DEBUG["state"]["error"] = err
                DEBUG["busy"] = False
                DEBUG.pop("progress", None)

        threading.Thread(target=init_work, daemon=True).start()
        return {**init_state, "busy": True, "progress": DEBUG["progress"]}

    with LOADED.lock:
        try:
            st = adapter.image_stepper(plan) if image else adapter.stepper(plan, prefix)
        except ValueError as e:
            raise HTTPException(400, str(e))
    if not image:
        st.ctx.sources = sources
    DEBUG.update(stepper=st, model_id=LOADED.spec["id"], flow=req.flow, label=req.label or "debug", run=run)
    with LOADED.lock:
        if not image:
            st.step_block()
    return _publish(st)


@app.post("/api/debug/step")
def debug_step(req: DebugStepReq):
    st = _debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing the previous step.")
    if req.mode not in ("block", "token", "to", "all"):
        raise HTTPException(400, "mode must be block | token | to | all")
    if getattr(st, "done", False) and hasattr(st, "extend_steps"):
        target = None
        if req.flow:
            for n in req.flow.get("nodes", []):
                if (n.get("data") or {}).get("kind") == "scheduler":
                    try:
                        target = int(n["data"].get("steps") or 0)
                    except (TypeError, ValueError):
                        target = None
        with LOADED.lock:
            st.extend_steps(target_steps=target if target and target > st.n_steps else None, more_steps=4)
    elif req.flow and hasattr(st, "extend_steps") and getattr(st, "x", None) is None:
        for n in req.flow.get("nodes", []):
            if (n.get("data") or {}).get("kind") == "scheduler":
                try:
                    target = int(n["data"].get("steps") or 0)
                    if target > st.n_steps:
                        with LOADED.lock:
                            st.extend_steps(target_steps=target)
                except (TypeError, ValueError):
                    pass
    if req.force_token is not None:
        st.force(req.force_token)
    nb = len(st.plan.schedule)
    t0 = time.time()

    def prog(_frac, _msg):
        DEBUG["progress"] = {"frac": round((st.bp + 1) / nb, 3),
                             "message": (f"step {min(st.i, st.n_steps - 1) + 1}/{st.n_steps} · " if getattr(st, "n_steps", None) else
                                         f"{'prefill' if st.n_generated == 0 else f'token {st.n_generated}'} · ") +
                                        f"block {st.bp + 1}/{nb} done · {time.time() - t0:.0f}s",
                             "node": st.plan.schedule_nodes[st.bp + 1] if st.bp + 1 < nb else None}
        task.progress, task.message = DEBUG["progress"]["frac"], DEBUG["progress"]["message"]

    task = TASKS.new("debug", f"step: {req.mode}" + (f" ×{req.count}" if req.count > 1 else ""), run_id=DEBUG["run"].id)

    def _stop():
        st.ctx.cancel = True
    task.on_cancel = _stop
    DEBUG["task"] = task.id

    def block():
        with TASKS.slot(task):
            if DEBUG.get("stepper") is not st:
                raise Cancelled("session replaced")
            return st.step_block()

    def token():
        if st.x is None and not block():
            return False
        while st.x is not None:
            block()
        return True

    def run_to(pos):
        first = True
        for _ in range(100_000):
            if not first and st.x is not None and st.bp == pos + 1:
                return True
            first = False
            if not block():
                return False
        return True

    def one():
        if req.mode == "block":
            return block()
        if req.mode == "token":
            return token()
        if req.mode == "to":
            return run_to(int(req.pos or 0))
        while token():
            pass
        return False

    def work():
        err = None
        try:
            for _ in range(max(1, min(req.count, 10_000))):
                ok = one()
                if not ok:
                    break
            if not task.cancel and getattr(st, "x", None) is None and hasattr(st, "guess_png") and getattr(st, "guess", None) is not None:
                DEBUG["progress"] = {"frac": 0.98, "message": "decoding current image (VAE)…", "node": "vae_decode"}
                task.progress, task.message = 0.98, "decoding current image (VAE)…"
                with TASKS.slot(task):
                    st.guess_png(exact=True)
        except Cancelled:
            err = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = f"{type(e).__name__}: {e}"
        finally:
            st.ctx.cancel = False
            st.progress = None
            TASKS.finish(task, "cancelled" if task.cancel else ("error" if err else "done"), err)
        if DEBUG.get("stepper") is st:
            _publish(st)
            DEBUG["state"]["error"] = err
            DEBUG["busy"] = False
            DEBUG.pop("progress", None)

    st.progress = prog
    DEBUG["busy"] = True
    DEBUG["progress"] = {"frac": 0.0, "message": "starting"}
    threading.Thread(target=work, daemon=True).start()
    return {**(DEBUG.get("state") or {}), "busy": True, "progress": DEBUG["progress"]}


@app.post("/api/debug/cancel")
def debug_cancel():
    if DEBUG.get("busy"):
        if DEBUG.get("task") in TASKS.tasks:
            TASKS.act(DEBUG["task"], "cancel")
        if DEBUG.get("stepper") is not None:
            DEBUG["stepper"].ctx.cancel = True
    return {"ok": True}


@app.get("/api/debug")
def debug_state():
    if not DEBUG:
        return {}
    return {**(DEBUG.get("state") or {}), "busy": bool(DEBUG.get("busy")), "progress": DEBUG.get("progress"),
            "task": DEBUG.get("task"),
            "ram_gb": _rss_gb() if DEBUG.get("busy") else None}


@app.post("/api/debug/finish")
def debug_finish():
    st = _debug()
    run: Run | None = DEBUG.get("run")
    if run is None:
        raise HTTPException(409, "No active debug run.")
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing a step; wait or press ■ to stop it.")
    if getattr(st, "n_steps", None) and not st.done:
        raise HTTPException(409, "The picture isn't finished yet: press ▶| (run to the end) first, then Finish.")
    with LOADED.lock:
        try:
            if st.x is not None:
                st.step_token()
            res = st.finish()
        except ValueError as e:
            raise HTTPException(400, str(e))
    run.outputs, run.captures, run.status, run.progress, run.message = res["outputs"], res["captures"], "done", 1.0, "done"
    run.version += 1
    run.cache = {}
    _save_manifest(run)
    STORE.add(run)
    state = {**st.state(), "run_id": run.id}
    if DEBUG.get("stepper") is st:
        DEBUG["state"] = state
    return {"run": run.summary(), "debug": state}


@app.post("/api/debug/extend")
def debug_extend(req: ExtendStepsReq):
    st = _image_debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing a step; wait or press ■.")
    with LOADED.lock:
        try:
            st.extend_steps(target_steps=req.steps, more_steps=req.add)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return _publish(st)


@app.post("/api/debug/noise")
def debug_noise(req: NoiseReq):
    st = _image_debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing a step; wait or press ■.")
    with LOADED.lock:
        try:
            res = st.set_noise(req.sigma, req.fresh_noise)
        except ValueError as e:
            raise HTTPException(400, str(e))
    state = _publish(st)
    return {**state, "noise": res}


@app.get("/api/debug/guess.png")
def debug_guess(exact: bool = True):
    st = _image_debug()
    try:
        with LOADED.lock:
            return Response(st.guess_png(exact), media_type="image/png")
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/debug/input.png")
def debug_input():
    st = _image_debug()
    try:
        return Response(st.input_png(), media_type="image/png")
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/debug/edit")
def debug_edit(req: EditReq):
    st = _image_debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing a step; wait or press ■.")
    with LOADED.lock:
        try:
            rec = st.edit_guess(
                req.image,
                max(0.0, min(1.0, req.strength)),
                req.only_changed,
                req.mask,
                sigma=req.sigma,
                edit_noise=req.edit_noise,
                fresh_noise=req.fresh_noise,
                steer_steps=req.steer_steps,
                steer_pull=req.steer_pull,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
    state = _publish(st)
    return {**state, "edit": rec}


@app.post("/api/debug/prompt")
def debug_prompt(req: PromptSteerReq):
    st = _image_debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing a step; wait or press ■.")
    with LOADED.lock:
        try:
            rec = st.set_prompt(req.prompt, req.blocks)
        except ValueError as e:
            raise HTTPException(400, str(e))
    state = _publish(st)
    return {**state, "prompt_change": rec}


@app.post("/api/debug/word")
def debug_word(req: WordScaleReq):
    st = _image_debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing a step; wait or press ■.")
    with LOADED.lock:
        try:
            rec = st.set_word_weight(req.token, req.scale, req.blocks)
        except ValueError as e:
            raise HTTPException(400, str(e))
    state = _publish(st)
    return {**state, "word": rec}


@app.post("/api/debug/stop")
def debug_stop():
    if DEBUG.get("busy") and DEBUG.get("stepper") is not None:
        DEBUG["stepper"].ctx.cancel = True
    _close_live("stopped")
    DEBUG.clear()
    return {"ok": True}
