"""FastAPI server. The browser UI (static files in analyser/web, no build step) is a client of this API."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import STARTED_WITH, __version__, code_stamp, interp
from .caches import LIBRARY
from .engine import Cancelled
from .flow import FlowError, compile_flow
from .plugins import list_plugins
from .registry import LOADED, add_local_spec, get_spec, list_specs, with_disk_state
from .weights import model_folder, models_dir
from .runs import DATA as RUNS_DATA, STORE, Run
from .tasks import TASKS

app = FastAPI(title="Tokenshark", version=__version__)
WEB = Path(__file__).parent / "web"


class LoadReq(BaseModel):
    model_id: str
    policy: str = "auto"        # auto | device | cpu | disk
    keep_layers: int = 0


class SpecReq(BaseModel):
    id: str
    family: str
    repo: str | None = None
    path: str | None = None
    size_gb: float | None = None
    notes: str = ""
    default_steps: int | None = None
    default_guidance: float | None = None
    default_size: int | None = None


class FlowReq(BaseModel):
    flow: dict


class RunReq(BaseModel):
    flow: dict
    parent_id: str | None = None
    label: str = ""
    resume: dict | None = None       # diffusion: {"run_id", "step"} — branch from step k of a parent run


class SaveCacheReq(BaseModel):
    name: str = ""


class EditCacheReq(BaseModel):
    op: str                          # truncate | drop | shift | keep
    start: int = 0
    end: int = 0
    delta: float = 0.0
    name: str = ""


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


# ---------------------------------------------------------------- models
def _rss_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**30, 2)
    except Exception:
        return None


@app.get("/api/status")
def status():
    return {"version": __version__, "device": LOADED.device, "dtype": str(LOADED.dtype).replace("torch.", ""),
            "status": LOADED.status, "error": LOADED.error, "model": LOADED.info, "models_dir": str(models_dir()),
            "progress": LOADED.progress,
            "load_seconds": round(time.time() - getattr(LOADED, "load_started", time.time()), 0)
            if LOADED.status.startswith("loading") else None,
            "stale": code_stamp() != STARTED_WITH,
            "ram_gb": _rss_gb(),
            "debug": DEBUG.get("state") if DEBUG else None}


@app.get("/api/plugins")
def plugins_endpoint():
    return list_plugins()


class LoRAReq(BaseModel):
    path: str
    scale: float = 1.0
    name: str = ""


class LoRAScaleReq(BaseModel):
    scale: float


@app.get("/api/lora")
def get_lora():
    _require_model()
    if LOADED.lora is None:
        return {"active": False}
    return {**LOADED.lora.summary(), "adaptation": getattr(LOADED.lora, "adaptation", None)}


@app.post("/api/lora/load")
def load_lora(req: LoRAReq):
    _require_model()
    if not req.path.strip():
        raise HTTPException(400, "Provide a path to a LoRA (.safetensors / folder) or fine-tuned checkpoint.")
    try:
        return LOADED.attach_lora(req.path.strip(), req.scale, req.name.strip())
    except Exception as e:
        raise HTTPException(400, f"Could not load LoRA / fine-tune: {e}")


@app.post("/api/lora/upload")
async def upload_lora(request: Request, name: str = "uploaded-lora", scale: float = 1.0):
    _require_model()
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty file upload.")
    lora_dir = RUNS_DATA / "loras"
    lora_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or "uploaded.safetensors"))
    if not safe_name.endswith(".safetensors"):
        safe_name += ".safetensors"
    dst = lora_dir / safe_name
    dst.write_bytes(data)
    try:
        return LOADED.attach_lora(dst, scale=scale, name=name)
    except Exception as e:
        raise HTTPException(400, f"Could not attach uploaded LoRA: {e}")


@app.post("/api/lora/scale")
def set_lora_scale(req: LoRAScaleReq):
    _require_model()
    try:
        return LOADED.set_lora_scale(req.scale)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.delete("/api/lora")
def unload_lora():
    with LOADED.lock:
        LOADED.detach_lora()
    return {"active": False}


@app.get("/api/models")
def models():
    loaded = LOADED.spec["id"] if LOADED.spec else None
    return [{**with_disk_state(s), "loaded": s["id"] == loaded} for s in list_specs()]


@app.delete("/api/models/{model_id}/files")
def delete_model_files(model_id: str):
    import shutil
    try:
        spec = get_spec(model_id)
    except KeyError:
        raise HTTPException(404, "Unknown model.")
    if LOADED.spec and LOADED.spec["id"] == model_id:
        raise HTTPException(409, "Unload the model first.")
    folder = model_folder(spec).resolve()
    root = models_dir().resolve()
    if root not in folder.parents:
        raise HTTPException(400, f"Only folders inside {root} are deleted from here; remove {folder} yourself.")
    shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True, "deleted": str(folder)}


@app.post("/api/models")
def add_model(req: SpecReq):
    if req.family not in ("causal_lm", "t2i"):
        raise HTTPException(400, "family must be causal_lm or t2i")
    if not (req.repo or req.path):
        raise HTTPException(400, "Give a Hugging Face repo id or a local folder path.")
    return add_local_spec({k: v for k, v in req.model_dump().items() if v not in (None, "")})


@app.post("/api/load")
def load(req: LoadReq):
    if LOADED.status.startswith(("loading", "downloading", "locating", "copying")):
        raise HTTPException(409, "A model is already loading.")
    if req.policy not in ("auto", "device", "cpu", "disk"):
        raise HTTPException(400, "policy must be auto, device, cpu or disk")
    LOADED.status = f"loading {req.model_id}"

    def work():
        try:
            LOADED.load(req.model_id, req.policy, max(0, req.keep_layers))
        except Exception:
            import traceback
            traceback.print_exc()

    threading.Thread(target=work, daemon=True).start()
    return {"status": LOADED.status}


@app.post("/api/unload")
def unload():
    with LOADED.lock:
        LOADED.unload()
        LOADED.status = "idle"
    return {"status": "idle"}


# ---------------------------------------------------------------- flows
@app.get("/api/flow/default")
def default_flow():
    _, info = _require_model()
    return LOADED.plugin.default_flow(info)


@app.post("/api/flow/compile")
def compile_(req: FlowReq):
    _, info = _require_model()
    try:
        return {"ok": True, "plan": compile_flow(req.flow, info).to_dict()}
    except FlowError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)


# ---------------------------------------------------------------- runs
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
        plan.params["overlays"] = overlays            # LoRA / quantization in force: part of what the run was
    run = Run(LOADED.spec["id"], req.flow, plan, parent_id, req.label, resume_step)
    run.sources = _sources(plan)
    STORE.add(run)
    STORE.execute(run, LOADED, resume, prefix)
    return {"ok": True, "run": run.summary(), "plan": plan.to_dict()}


def _load_prefixes(plan, adapter):
    """All KV prefixes the flow prepends — saved caches and layer prompts (text → KV, built here under the
    model lock) — checked against this model and schedule."""
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


class PrefixReq(BaseModel):
    flow: dict
    text: str
    name: str = ""
    as_system: bool = True


@app.post("/api/caches/from_text")
def cache_from_text(req: PrefixReq):
    """Hidden prompt: encode a (system) prompt once and keep only its KV cache."""
    adapter, info = _need("kv_cache", "A KV cache")
    try:
        plan = compile_flow(req.flow, info)
    except FlowError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)
    with LOADED.lock:
        try:
            cache, extra = adapter.build_prefix(plan, req.text, req.as_system)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return LIBRARY.save(cache, req.name or ("hidden system prompt" if req.as_system else "hidden prompt"),
                        {**extra, "model_id": LOADED.spec["id"]})


@app.get("/api/runs")
def list_runs():
    return STORE.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, since: int = -1):
    """since = the version the client already has: outputs are left out when unchanged (cheap polling)."""
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


def _clean(part):
    if isinstance(part, dict):
        part = {k: v for k, v in part.items() if k not in ("_progress", "todo")}
    return part


def _analysis(r: Run, label: str, group: str, key, gen_fn, wait: bool, weights=None):
    """Run an analysis as a task (one layer per unit of work, see tasks.py). wait=False returns at once with
    the task; the client polls GET /api/tasks/{id} for partial results. wait=True blocks until done."""
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

    def gen(memo):
        yield ad.track_word(cap, r.plan.schedule_nodes, r.plan.schedule, position, word, through)
    return _analysis(r, f"track “{word}”", "track", ("track", word, position, through), gen, not background)


# ---------------------------------------------------------------- task manager
@app.get("/api/tasks")
def list_tasks():
    return TASKS.list()


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    t = TASKS.tasks.get(task_id)
    if t is None:
        raise HTTPException(404, "Unknown task (finished tasks are forgotten after a while).")
    return {**t.summary(), "result": t.result}


@app.post("/api/tasks/{task_id}/{action}")
def task_action(task_id: str, action: str):
    try:
        return TASKS.act(task_id, action)
    except KeyError:
        raise HTTPException(404, "Unknown task.")
    except ValueError as e:
        raise HTTPException(400, str(e))


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


class ContribReq(BaseModel):
    steps: str = "all"


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
                if r.resume_step is not None:          # branched runs: measure against a fresh baseline
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
    """kind=x0: the model's guess of the finished image at that step. kind=xt: the noisy input it was given."""
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
    """Decode every block of every captured step through the last `through` blocks (a task, cell by cell)."""
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
    """Image lens: the picture the model would make if it stopped after block `pos` at step `step`.
    Fast = linear colour preview (no model needed); exact = VAE decode, run as a task."""
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


# ---------------------------------------------------------------- KV-cache library
@app.get("/api/caches")
def list_caches():
    return LIBRARY.list()


@app.get("/api/caches/{cid}")
def get_cache(cid: str):
    try:
        c, extra = LIBRARY.load(cid)
    except KeyError:
        raise HTTPException(404, "Unknown cache.")
    piece = LOADED.adapter.tok.piece if LOADED.adapter is not None and LOADED.info and LOADED.info["family"] == "causal_lm" else None
    return {**extra, **c.summary(piece), "fingerprint": c.fingerprint}


@app.get("/api/caches/{cid}/download")
def download_cache(cid: str):
    try:
        p = LIBRARY.path(cid)
    except KeyError:
        raise HTTPException(404, "Unknown cache.")
    return FileResponse(p, filename=f"kv-{cid}.safetensors", media_type="application/octet-stream")


@app.post("/api/caches/upload")
async def upload_cache(request: Request, name: str = ""):
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty upload.")
    try:
        return LIBRARY.import_bytes(data, name)
    except Exception as e:
        raise HTTPException(400, f"Could not read KV cache: {e}")


@app.post("/api/caches/{cid}/edit")
def edit_cache(cid: str, req: EditCacheReq):
    try:
        c, extra = LIBRARY.load(cid)
    except KeyError:
        raise HTTPException(404, "Unknown cache.")
    n = len(c)
    s, e = max(0, req.start), min(n, req.end if req.end > 0 else n)
    if c.states and req.op != "shift":
        raise HTTPException(400, "This cache is from a hybrid model (Gated DeltaNet blocks keep one running memory of "
                                 "the whole sequence), so spans can't be cut or dropped. Rebuild it from shorter text.")
    if req.op == "truncate":
        out = c.truncate(s)
        what = f"first {s} tokens"
    elif req.op == "keep":
        out = c.select(list(range(s, e)))
        what = f"tokens {s}–{e}"
    elif req.op == "drop":
        out = c.drop(s, e)
        what = f"without {s}–{e}"
    elif req.op == "shift":
        out = c.shift(s, e, req.delta)
        what = f"{s}–{e} shifted {req.delta:+g}"
    else:
        raise HTTPException(400, "op must be truncate | keep | drop | shift")
    return LIBRARY.save(out, req.name or f"{extra.get('name', cid)} ({what})",
                        {"model_id": extra.get("model_id"), "source": "edit", "parent_cache": cid,
                         "schedule": extra.get("schedule")})


@app.delete("/api/caches/{cid}")
def delete_cache(cid: str):
    try:
        LIBRARY.delete(cid)
    except KeyError:
        raise HTTPException(404, "Unknown cache.")
    return {"ok": True}


# ---------------------------------------------------------------- debugger (LLM stepping)
DEBUG: dict = {}


class DebugStartReq(BaseModel):
    flow: dict
    label: str = ""
    background: bool = False


class DebugStepReq(BaseModel):
    mode: str = "block"          # block | token | to | all
    pos: int | None = None       # schedule position for mode=to
    force_token: int | None = None
    count: int = 1
    flow: dict | None = None     # optional updated flow (e.g. increased steps in scheduler)


class ExtendStepsReq(BaseModel):
    steps: int | None = None
    add: int = 4


class NoiseReq(BaseModel):
    sigma: float
    fresh_noise: bool = False


def _debug():
    if not DEBUG:
        raise HTTPException(409, "No debug session. Start one first.")
    if LOADED.spec is None or LOADED.spec["id"] != DEBUG.get("model_id"):
        DEBUG.clear()
        raise HTTPException(409, "The model changed; start a new debug session.")
    if DEBUG.get("stepper") is None:
        raise HTTPException(409, "Still preparing the stepping session; wait a moment.")
    return DEBUG["stepper"]


@app.post("/api/debug/start")
def debug_start(req: DebugStartReq):
    adapter, info = _require_model()
    from .runs import run_loop_of
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
            st.step_block()          # embed the prompt right away, so every tab has something to show
    return _publish(st)


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
        _save_manifest(run)
    else:
        run.status, run.message = "stepping", f"stepping · {st.steps} steps"
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
    from .runs import RUNS_DIR
    try:
        (RUNS_DIR / f"{run.id}.json").write_text(_json.dumps(
            {**run.manifest(), "status": run.status, "outputs": run.outputs}, indent=1, default=str))
    except Exception:
        pass


@app.post("/api/debug/step")
def debug_step(req: DebugStepReq):
    """Steps run in the background: a single block of a 27B model streamed from disk can take a while, so the
    call returns at once with busy=true and the client polls GET /api/debug for progress and the result."""
    st = _debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing the previous step.")
    if req.mode not in ("block", "token", "to", "all"):
        raise HTTPException(400, "mode must be block | token | to | all")
    # If all denoising steps were finished, check if the user raised the step count in the Scheduler box
    # (or automatically add 4 more refinement steps so pressing Step continues).
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

    # The model lock is taken per block, so the analysis tabs can still be used between blocks of a long step.
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
            DEBUG["stepper"].ctx.cancel = True        # stops before the next block; the session stays usable
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
            if st.x is not None:          # finish the token that is in flight so the KV cache is consistent
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
    """Add more denoising steps to the active image stepping session."""
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
    """Adjust the noise level σ (and optionally draw fresh noise) for the upcoming step."""
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


class EditReq(BaseModel):
    image: str
    strength: float = 1.0
    only_changed: bool = True
    mask: str = ""
    sigma: float | None = None


def _image_debug():
    st = _debug()
    if not hasattr(st, "edit_guess"):
        raise HTTPException(400, "This is not an image stepping session.")
    return st


@app.get("/api/debug/guess.png")
def debug_guess(exact: bool = True):
    """The current guess x̂₀ (after the last step, or after the last block mid-step)."""
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
    """Force the picture: swap the model's guess for an edited one, keeping its noise (see ImageStepper.edit_guess)."""
    st = _image_debug()
    if DEBUG.get("busy"):
        raise HTTPException(409, "Still computing a step; wait or press ■.")
    with LOADED.lock:
        try:
            rec = st.edit_guess(req.image, max(0.0, min(1.0, req.strength)), req.only_changed, req.mask, sigma=req.sigma)
        except ValueError as e:
            raise HTTPException(400, str(e))
    state = _publish(st)
    return {**state, "edit": rec}


@app.post("/api/debug/stop")
def debug_stop():
    if DEBUG.get("busy"):
        DEBUG["stepper"].ctx.cancel = True
    _close_live("stopped")
    DEBUG.clear()
    return {"ok": True}


class DecodeReq(BaseModel):
    ids: list[int]


@app.post("/api/tokens/decode")
def decode_tokens(req: DecodeReq):
    adapter, info = _need("lm_tools", "Token decoding")
    return {"text": adapter.tok.decode(req.ids, skip_special=False)}


# ---------------------------------------------------------------- lab tools (see labapi.py)
from . import labapi  # noqa: E402,F401  — registers the routes for patching, steering, neurons, training …


# ---------------------------------------------------------------- web UI (static, no build step)
@app.middleware("http")
async def _same_origin(request: Request, call_next):
    """Only this app's own pages may change things. A flow can carry Python for a box (blockcode.py), so a
    request that another website makes the browser send must not be able to start a run."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlparse
            if urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Requests from other sites are not accepted."}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def _no_stale_ui(request: Request, call_next):
    """The UI is edited in place: make the browser re-check its files on every load instead of caching them."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
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
