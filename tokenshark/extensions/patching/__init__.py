"""Activation patching & causal tracing extension (`tokenshark.extensions.patching`)."""
from __future__ import annotations

import copy
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...building_blocks import PARTS, activation
from ...flow import FlowError, Intervention, compile_flow, parse_index_spec
from ...llm import Stepper
from ...registry import LOADED
from ...sdk import FeaturePlugin, register_extension
from ...server.app import _caps, _load_prefixes, _need, _run, _sources
from ...tasks import TASKS

router = APIRouter()


def iter_patch_sweep(adapter, plan, cap_src: dict, hook: str = "resid_post", tokens: str = "all", prefix=None):
    """Run the flow's prompt once per (block, token), each time copying that one activation from the source
    ('clean') run, and measure how far the clean run's answer comes back."""
    plan = copy.deepcopy(plan)
    plan.params["max_new_tokens"] = 0
    base_ivs, base_src = list(plan.interventions), dict(getattr(plan, "sources", None) or {})
    S = len(plan.schedule)

    def last_logits(ivs, src=None):
        plan.interventions = base_ivs + ivs
        st = Stepper(adapter, plan, prefix=prefix)
        st.ctx.sources = {**base_src, **({"sweep": src} if src is not None else {})}
        while st.step_token():
            pass
        return st.logits_rows[-1][-1].float(), st

    base, st0 = last_logits([])
    n_in = st0.n_in
    n_src = cap_src.get("n_in") or len(cap_src["ids"])
    clean = adapter.decode_layer(cap_src, plan.schedule, S - 1, 0)[n_src - 1].float()
    answer = int(clean.argmax())
    wrong = int(base.argmax())
    p_clean, p_corr = float(torch.softmax(clean, -1)[answer]), float(torch.softmax(base, -1)[answer])
    toks = [n_in - 1] if tokens == "last" else parse_index_spec(tokens, n_in)
    toks = [t for t in toks if t < n_src] if n_src != n_in else toks
    res = {
        "hook": hook,
        "blocks": [f"#{p} · L{l}" for p, l in enumerate(plan.schedule)],
        "nodes": plan.schedule_nodes,
        "token_index": toks,
        "tokens": [adapter.tok.piece(st0.seq[t]) for t in toks],
        "source_tokens": [adapter.tok.piece(cap_src["ids"][t]) if t < len(cap_src["ids"]) else "" for t in toks],
        "answer": adapter.tok.piece(answer),
        "corrupt_answer": adapter.tok.piece(wrong),
        "p_clean": round(p_clean, 4),
        "p_corrupt": round(p_corr, 4),
        "same_length": n_src == n_in,
        "grid": [[None] * len(toks) for _ in range(S)],
        "prob": [[None] * len(toks) for _ in range(S)],
        "_progress": 0.0,
    }
    yield copy.deepcopy(res)
    gap = p_clean - p_corr
    done, total = 0, S * len(toks)
    for p in range(S):
        src = activation(cap_src, hook, p)
        if hook in ("q", "k", "v") and src.ndim == 2:
            hd = cap_src["cache"].fingerprint.get("head_dim") or adapter.cfg.head_dim
            src = src.view(src.shape[0], -1, hd)
        for j, t in enumerate(toks):
            iv = Intervention(node_id="sweep", pos=p, hook=hook, op="patch", positions=str(t))
            pr = float(torch.softmax(last_logits([iv], src)[0], -1)[answer])
            res["prob"][p][j] = round(pr, 4)
            res["grid"][p][j] = round((pr - p_corr) / gap, 4) if abs(gap) > 1e-6 else 0.0
            done += 1
        res["_progress"] = done / max(1, total)
        yield copy.deepcopy(res)


class SweepReq(BaseModel):
    flow: dict
    source_run: str
    hook: str = "resid_post"
    tokens: str = "all"


@router.post("/api/patch/sweep")
def patch_sweep(req: SweepReq):
    adapter, info = _need("lm_tools", "This tool")
    r = _run(req.source_run)
    cap = _caps(r)
    if r.plan.family != "causal_lm":
        raise HTTPException(400, "This tool is for language-model runs.")
    if req.hook not in PARTS:
        raise HTTPException(400, f"hook must be one of {', '.join(PARTS)}")
    try:
        plan = compile_flow(req.flow, info)
    except FlowError as e:
        raise HTTPException(422, " · ".join(x["message"] for x in e.errors))
    plan.sources = _sources(plan)
    prefix = _load_prefixes(plan, adapter) if plan.params.get("prefix_cache") else None
    t = TASKS.new("analysis", f"patching sweep ({req.hook})", group="sweep")
    TASKS.start(t, lambda: iter_patch_sweep(adapter, plan, cap, req.hook, req.tokens, prefix), weights=adapter.model.W)
    return {"task": t.summary(), "result": None, "status": t.status}


@register_extension
class PatchingPlugin(FeaturePlugin):
    id = "patching"
    name = "Activation Patching & Causal Tracing"
    version = "0.2.0"
    description = "Copy activations from a clean run into a corrupted run and run causal tracing sweeps across blocks and tokens."
    families = ("causal_lm",)
    router = router
    ui_tabs = ()
    palette_actions = ({"id": "add-patch", "label": "+ Patch", "group": "intervene", "feature": "patch"},)
