"""Model & LoRA endpoints (`tokenshark.server.models_api`)."""
from __future__ import annotations

import threading
from fastapi import HTTPException, Request
from pydantic import BaseModel

from ..registry import LOADED, add_local_spec, get_spec, list_specs, with_disk_state
from ..runs import DATA as RUNS_DATA
from ..weights import model_folder, models_dir
from .app import _require_model, app


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


class LoRAReq(BaseModel):
    path: str
    scale: float = 1.0
    name: str = ""
    layers: str = "all"


class LoRAScaleReq(BaseModel):
    scale: float
    layers: str | None = None


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
        return LOADED.attach_lora(req.path.strip(), req.scale, req.name.strip(), req.layers)
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
        return LOADED.set_lora_scale(req.scale, req.layers)
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
