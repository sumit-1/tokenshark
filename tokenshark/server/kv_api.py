"""KV-cache library and hidden prefix endpoints (`tokenshark.server.kv_api`)."""
from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..caches import LIBRARY
from ..flow import FlowError, compile_flow, parse_index_spec
from ..registry import LOADED
from .app import _need, app


class PrefixReq(BaseModel):
    flow: dict
    text: str
    name: str = ""
    as_system: bool = True


class EditCacheReq(BaseModel):
    op: str                          # truncate | drop | shift | keep | zero_slots | scale_slots | quantize_slots
    start: int = 0
    end: int = 0
    delta: float = 0.0
    name: str = ""
    layers: str = "all"
    part: str = "both"               # both | k | v


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
    if c.states and req.op not in ("shift", "zero_slots", "scale_slots", "quantize_slots"):
        raise HTTPException(400, "This cache is from a hybrid model (Gated DeltaNet blocks keep one running memory of "
                                 "the whole sequence), so spans can't be cut or dropped. Rebuild it from shorter text.")
    if req.op == "truncate":
        out = c.truncate(s)
        what = f"first {s} tokens"
    elif req.op == "keep":
        out = c.select(list(range(s, e)))
        what = f"tokens {s}-{e}"
    elif req.op == "drop":
        out = c.drop(s, e)
        what = f"without {s}-{e}"
    elif req.op == "shift":
        out = c.shift(s, e, req.delta)
        what = f"{s}-{e} shifted {req.delta:+g}"
    elif req.op in ("zero_slots", "scale_slots", "quantize_slots"):
        try:
            slots = parse_index_spec(req.layers, c.n_slots)
        except ValueError as ex:
            raise HTTPException(400, str(ex))
        short_op = req.op.split("_")[0]
        out = c.edit_slots(short_op, slots, s, e, req.delta, req.part if req.part in ("both", "k", "v") else "both")
        what = f"blocks {req.layers} {req.part} {short_op} ({s}-{e})"
    else:
        raise HTTPException(400, "op must be truncate | keep | drop | shift | zero_slots | scale_slots | quantize_slots")
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
