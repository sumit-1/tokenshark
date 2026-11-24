"""Core HTTP routes for editable box code, run -> script export, sampler playground, and tokenizer explorer.

Feature-specific routes (`patching`, `steering`, `neurons`, `moe`, `introspect`, `quant`, `training`,
`guardrail_fork`, `attn_traceback`, `datasets`) live in their respective `tokenshark/extensions/` packages
and are mounted automatically by `tokenshark.server.mount_extensions()`.
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import blockcode, interp
from .flow import FlowError, compile_flow
from .registry import LOADED
from .runs import STORE
from .server.app import _caps, _need, _require_model, _run, app
from .tasks import TASKS


def _llm():
    return _need("lm_tools", "This tool")


def _llm_run(run_id: str):
    r = _run(run_id)
    cap = _caps(r)
    if r.plan.family != "causal_lm":
        raise HTTPException(400, "This tool is for language-model runs.")
    return r, cap


def _guard(fn, *a, **kw):
    """Call engine code under the model lock; turn its ValueErrors into readable 400s."""
    try:
        with LOADED.lock:
            return fn(*a, **kw)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'\""))


def _task(kind: str, label: str, group: str, gen_fn, weights=None) -> dict:
    """Start a generator as a task (one unit of work per yield). The client polls GET /api/tasks/{id}."""
    t = TASKS.new(kind, label, group=group)
    TASKS.start(t, gen_fn, weights=weights)
    return {"task": t.summary(), "result": None, "status": t.status}


def _compile(flow: dict, info: dict):
    try:
        return compile_flow(flow, info)
    except FlowError as e:
        raise HTTPException(422, " · ".join(x["message"] for x in e.errors))


# ---------------------------------------------------------------- editable box code
class CodeReq(BaseModel):
    code: str
    kind: str = "block"


@app.get("/api/code/default")
def code_default(kind: str = "block", layer: int = 0):
    adapter, _ = _llm()
    if kind == "block" and not 0 <= layer < adapter.cfg.n_layers:
        raise HTTPException(400, "No such layer.")
    return {**blockcode.default_code(kind, adapter.model, layer), "kind": kind, "layer": layer,
            "help": blockcode.__doc__}


@app.post("/api/code/check")
def code_check(req: CodeReq):
    problems = blockcode.check(req.code)
    return {"ok": not problems, "problems": problems}


# ---------------------------------------------------------------- run -> script
@app.get("/api/runs/{run_id}/script", response_class=PlainTextResponse)
def run_script(run_id: str):
    from .export import run_script as make
    r = _run(run_id)
    _, info = _require_model()
    if LOADED.spec["id"] != r.model_id:
        raise HTTPException(409, "Load this run's model to export it (the script is written against its default flow).")

    def default_for(_r):
        return LOADED.plugin.default_flow(info)

    def get(rid):
        try:
            return STORE.get(rid)
        except KeyError:
            return None
    return make(r, default_for, get, {v["id"]: v for v in interp.VECTORS.list()})


# ---------------------------------------------------------------- sampler playground
@app.get("/api/runs/{run_id}/logits")
def logits(run_id: str, position: int = -1, top: int = 1000):
    r, cap = _llm_run(run_id)
    return _guard(interp.final_logits, LOADED.adapter, cap, r.plan.schedule, position, max(10, min(top, 5000)))


# ---------------------------------------------------------------- tokenizer explorer
class TokenizeReq(BaseModel):
    text: str
    chat: bool = False
    system: str = ""
    history: list[dict] | None = None
    prefill: str = ""
    add_bos: bool = False


@app.post("/api/tokenize")
def tokenize(req: TokenizeReq):
    adapter, _ = _llm()
    try:
        return interp.tokenize(adapter, req.text, req.chat, req.system, req.history, req.prefill, req.add_bos)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/runs/{run_id}/rope_pairs")
def run_rope_pairs(run_id: str, block: int = 0, token: int = 0, head: int = 0, kind: str = "k"):
    r, cap = _llm_run(run_id)
    return _guard(interp.rope_pairs, LOADED.adapter, cap, r.plan.schedule, block, token, head, kind)


@app.get("/api/runs/{run_id}/kv_analysis")
def run_kv_analysis(run_id: str, blocks: str = "all", query: int | None = None):
    r, cap = _llm_run(run_id)
    return _guard(interp.kv_analysis, LOADED.adapter, cap, r.plan.schedule, r.plan.schedule_nodes, blocks, query)


@app.get("/api/tokens/search")
def token_search(q: str = "", limit: int = 80):
    adapter, _ = _llm()
    return interp.vocab_search(adapter, q, max(1, min(limit, 500)))


@app.get("/api/tokens/{token_id}/neighbors")
def token_neighbors(token_id: int, k: int = 12):
    adapter, _ = _llm()
    return _guard(interp.token_neighbors, adapter, token_id, max(1, min(k, 50)))
