"""Mixture-of-Experts (MoE) router and expert ablation extension (`tokenshark.extensions.moe`)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ... import moe
from ...flow import FlowError, compile_flow
from ...registry import LOADED
from ...sdk import FeaturePlugin, register_extension
from ...server.app import _caps, _load_prefixes, _need, _run, _sources
from ...tasks import TASKS

router = APIRouter()


def _moe_run(run_id: str):
    r = _run(run_id)
    cap = _caps(r)
    if r.plan.family != "causal_lm":
        raise HTTPException(400, "This tool is for language-model runs.")
    if not LOADED.adapter.cfg.moe:
        raise HTTPException(400, "The loaded model is not a mixture of experts. Load an MoE model (OLMoE, Qwen-MoE, "
                                 "Mixtral) or train a tiny one in the Train tab.")
    return r, cap


def _guard(fn, *a, **kw):
    try:
        with LOADED.lock:
            return fn(*a, **kw)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'\""))


@router.get("/api/runs/{run_id}/experts")
def experts_overview(run_id: str):
    r, cap = _moe_run(run_id)
    return _guard(moe.overview, LOADED.adapter, cap, r.plan.schedule, r.plan.schedule_nodes)


@router.get("/api/runs/{run_id}/experts/block")
def experts_block(run_id: str, pos: int, token: int = -1):
    r, cap = _moe_run(run_id)
    return _guard(moe.block_detail, LOADED.adapter, cap, r.plan.schedule, pos, token)


@router.get("/api/runs/{run_id}/expert")
def expert_one(run_id: str, pos: int, index: int):
    r, cap = _moe_run(run_id)
    return _guard(moe.expert_detail, LOADED.adapter, cap, r.plan.schedule, pos, index)


@router.get("/api/experts/router")
def experts_router(layer: int):
    adapter, _ = _need("lm_tools", "This tool")
    return _guard(moe.router_rows, adapter, layer)


class ExpertSweepReq(BaseModel):
    flow: dict
    pos: int = -1
    mode: str = "ban"


@router.post("/api/experts/sweep")
def experts_sweep(req: ExpertSweepReq):
    adapter, info = _need("lm_tools", "This tool")
    if req.mode not in ("ban", "zero"):
        raise HTTPException(400, "mode must be ban or zero")
    try:
        plan = compile_flow(req.flow, info)
    except FlowError as e:
        raise HTTPException(422, " · ".join(x["message"] for x in e.errors))
    plan.sources = _sources(plan)
    prefix = _load_prefixes(plan, adapter) if plan.params.get("prefix_cache") else None
    t = TASKS.new("analysis", f"expert ablation ({req.mode})", group="expert_sweep")
    TASKS.start(t, lambda: moe.iter_ablation(adapter, plan, req.pos, prefix, req.mode), weights=adapter.model.W)
    return {"task": t.summary(), "result": None, "status": t.status}


@register_extension
class MoEPlugin(FeaturePlugin):
    id = "moe"
    name = "Mixture-of-Experts (MoE) Inspector"
    version = "0.2.0"
    description = "Inspect MoE router usage, per-expert vocabulary projections, custom routing, and expert ablation sweeps."
    families = ("causal_lm",)
    router = router
    ui_tabs = ({"id": "experts", "label": "Experts", "only_moe": True},)
    palette_actions = ({"id": "add-router", "label": "+ Route", "group": "intervene", "feature": "moe"},)
