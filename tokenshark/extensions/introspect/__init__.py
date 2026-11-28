"""Concept injection, concept meter, and natural-language activation roundtrip extension (`tokenshark.extensions.introspect`)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ... import introspect as it
from ...flow import FlowError, compile_flow
from ...registry import LOADED
from ...sdk import FeaturePlugin, register_extension
from ...server.app import _caps, _need, _run, _sources
from ...tasks import TASKS
from ..steering import VECTORS

router = APIRouter()


def _llm_run(run_id: str):
    r = _run(run_id)
    cap = _caps(r)
    if r.plan.family != "causal_lm":
        raise HTTPException(400, "This tool is for language-model runs.")
    return r, cap


def _guard(fn, *a, **kw):
    try:
        with LOADED.lock:
            return fn(*a, **kw)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'\""))


def _compile(flow: dict, info: dict):
    try:
        return compile_flow(flow, info)
    except FlowError as e:
        raise HTTPException(422, " · ".join(x["message"] for x in e.errors))


class InjectReq(BaseModel):
    flow: dict
    concepts: list[str]
    blocks: list[int]
    strengths: list[float] = [1.0, 2.0, 4.0]
    max_new_tokens: int = 40
    positions: str = "all"


class ConceptVecReq(BaseModel):
    flow: dict
    word: str
    pos: int
    name: str = ""


class RoundTripReq(BaseModel):
    run_id: str
    pos: int
    token: int = -1
    inject: int = 2
    template: str = ""
    prefill: str | None = None
    max_new_tokens: int = 24
    scale: float = 1.0
    how: str = "last"


class ScoreTextReq(BaseModel):
    run_id: str
    pos: int
    token: int = -1
    text: str
    how: str = "last"


@router.get("/api/introspect/defaults")
def introspect_defaults():
    return {
        "prompt": it.INJECTION_PROMPT,
        "baselines": it.BASELINE_WORDS,
        "concept_template": it.CONCEPT_TEMPLATE,
        "explain_template": it.EXPLAIN_TEMPLATE,
        "explain_prefill": it.EXPLAIN_PREFILL,
    }


@router.post("/api/introspect/inject")
def introspect_inject(req: InjectReq):
    adapter, info = _need("lm_tools", "This tool")
    concepts = [c.strip() for c in req.concepts if c.strip()]
    plan = _compile(req.flow, info)
    plan.sources = _sources(plan)
    t = TASKS.new("analysis", f"concept injection ({len(concepts)} concept(s))", group="inject")
    TASKS.start(
        t,
        lambda: it.iter_injection(
            adapter, plan, concepts, req.blocks, req.strengths,
            max(1, int(req.max_new_tokens)), req.positions,
        ),
        weights=adapter.model.W,
    )
    return {"task": t.summary(), "result": None, "status": t.status}


@router.post("/api/introspect/vector")
def introspect_vector(req: ConceptVecReq):
    adapter, info = _need("lm_tools", "This tool")
    plan = _compile(req.flow, info)
    if not req.word.strip() or not 0 <= req.pos < len(plan.schedule):
        raise HTTPException(400, "Give a word and a block of the current flow.")
    plan.sources = _sources(plan)
    cv = _guard(it.concept_vectors, adapter, plan, [req.word.strip()])
    v = cv["vectors"][req.word.strip()][req.pos]
    return VECTORS.save(
        v,
        req.name or f"concept: {req.word.strip()} @#{req.pos}",
        {"source": "concept", "word": req.word.strip(), "pos": req.pos, "hook": "resid_post",
         "model_id": LOADED.spec["id"], "resid_norm": round(cv["resid_norm"][req.pos], 3)},
    )


@router.get("/api/runs/{run_id}/meter")
def concept_meter(run_id: str, vector: str, centre: bool = True):
    r, cap = _llm_run(run_id)
    try:
        v, meta = VECTORS.load(vector)
    except KeyError:
        raise HTTPException(404, "Unknown vector.")
    return {**_guard(it.concept_meter, LOADED.adapter, cap, r.plan.schedule, v, centre, r.plan), "vector": meta}


@router.post("/api/nla/roundtrip")
def nla_roundtrip(req: RoundTripReq):
    adapter, _ = _need("lm_tools", "This tool")
    r, cap = _llm_run(req.run_id)
    r.plan.sources = getattr(r, "sources", None) or {}
    tpl = req.template.strip() or it.EXPLAIN_TEMPLATE
    if "{x}" not in tpl:
        raise HTTPException(400, "The template needs a {x} placeholder.")
    pre = it.EXPLAIN_PREFILL if req.prefill is None else req.prefill
    t = TASKS.new("analysis", f"activation -> words -> activation (#{req.pos})", group="nla")
    TASKS.start(
        t,
        lambda: it.iter_round_trip(
            adapter, r.plan, cap, req.pos, req.token, req.inject, tpl, pre,
            max(1, int(req.max_new_tokens)), req.scale,
            "mean" if req.how == "mean" else "last",
        ),
        weights=adapter.model.W,
    )
    return {"task": t.summary(), "result": None, "status": t.status}


@router.post("/api/nla/score")
def nla_score(req: ScoreTextReq):
    r, cap = _llm_run(req.run_id)
    r.plan.sources = getattr(r, "sources", None) or {}
    return _guard(it.score_text, LOADED.adapter, r.plan, cap, req.pos, req.token, req.text,
                  "mean" if req.how == "mean" else "last")


@register_extension
class IntrospectPlugin(FeaturePlugin):
    id = "introspect"
    name = "Concept Injection & Introspection"
    version = "0.2.0"
    view_type = "diagnose"
    default_enabled = True
    description = "Concept injection sweeps, residual stream concept meters, and natural-language activation roundtrip explanations."
    families = ("causal_lm",)
    router = router
    ui_tabs = ()
