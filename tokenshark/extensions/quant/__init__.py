"""On-the-fly weight quantization extension (`tokenshark.extensions.quant`)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...flow import FlowError, compile_flow
from ...quant import iter_compare
from ...registry import LOADED
from ...sdk import FeaturePlugin, register_extension
from ...server.app import _load_prefixes, _need, _require_model, _sources
from ...tasks import TASKS

router = APIRouter()


def _guard(fn, *a, **kw):
    try:
        with LOADED.lock:
            return fn(*a, **kw)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'\""))


class QuantReq(BaseModel):
    bits: int = 4
    layers: str = "all"
    layer_bits: dict[int, int] | None = None


class QuantCompareReq(BaseModel):
    flow: dict
    bits: list[int] = [8, 4]


@router.get("/api/quant")
def get_quant():
    _require_model()
    return LOADED.quant.summary() if LOADED.quant is not None else {"active": False}


@router.post("/api/quant")
def set_quant(req: QuantReq):
    _require_model()
    return _guard(LOADED.set_quant, req.bits, True, req.layers, req.layer_bits)


@router.delete("/api/quant")
def clear_quant():
    _require_model()
    return _guard(LOADED.set_quant, None)


@router.post("/api/quant/compare")
def quant_compare(req: QuantCompareReq):
    adapter, info = _need("lm_tools", "This tool")
    try:
        plan = compile_flow(req.flow, info)
    except FlowError as e:
        raise HTTPException(422, " · ".join(x["message"] for x in e.errors))
    bits = [b for b in dict.fromkeys(req.bits) if 1 <= b <= 16][:6]
    if not bits:
        raise HTTPException(400, "Choose at least one bit width between 1 and 16.")
    prefix = _load_prefixes(plan, adapter) if plan.params.get("prefix_cache") else None
    sources = _sources(plan)
    t = TASKS.new("analysis", f"quantization: full vs {', '.join(f'{b}-bit' for b in bits)}", group="quant")
    TASKS.start(t, lambda: iter_compare(LOADED, plan, bits, prefix, sources))
    return {"task": t.summary(), "result": None, "status": t.status}


@register_extension
class QuantPlugin(FeaturePlugin):
    id = "quant"
    name = "Weight Quantization Explorer"
    version = "0.2.0"
    description = "Simulated INT8 / INT4 / INT2 per-layer weight quantization and side-by-side bit-width comparison."
    families = ("causal_lm",)
    router = router
    ui_tabs = ({"id": "quant", "label": "Quantization"},)
