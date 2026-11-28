r"""Previous-Token Attention Attribution Analyser (`tokenshark.extensions.attn_traceback`).

Given a query token position `t_q` (or `-1` for the newest prediction) and either a specific previous token
`t_k < t_q` or all previous tokens, scans every `(block, head)` in the model and computes:
1. `attn_weight`: raw attention probability $\alpha_{b, h}(t_q, t_k)$
2. `value_norm`: L2 norm of the head's value-weighted output vector $\|\alpha_{b, h}(t_q, t_k) \cdot v_{b, h}(t_k) W_O^{(b, h)}\|_2$
3. `logit_push`: direct projection of that head's contribution onto a target word's unembedding row (or the predicted token's unembedding row)

Provides one-click head knock-out interceptor wiring in the Attention tab.
"""
from __future__ import annotations

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...registry import LOADED
from ...sdk import (
    FeaturePlugin,
    register_extension,
    register_lab_method,
    register_mcp_tool,
)
from ...server.app import _caps, _need, _run

router = APIRouter()


def trace_previous_tokens(
    adapter,
    cap: dict,
    schedule: list[int],
    schedule_nodes: list[str],
    query_token: int = -1,
    key_token: int | None = None,
    target_word: str = "",
    top_k: int = 30,
) -> dict:
    """Rank every `(pos, layer, head, key_token)` across the run by how strongly it attends to and copies from
    previous tokens into `query_token`."""
    attn_map = cap.get("attn") or {}
    if not attn_map:
        raise ValueError("This run has no captured attention matrices. Run with a shorter prompt or default attention budget.")

    ids = cap.get("ids") or []
    T = len(ids)
    if T == 0:
        raise ValueError("Run has no tokens.")
    tq = min(max(query_token if query_token >= 0 else T + query_token, 0), T - 1)
    pieces = [adapter.tok.piece(i) for i in ids]

    m = adapter.model
    cfg = adapter.cfg
    n_heads = cfg.n_heads
    n_kv = cfg.n_kv
    hd = cfg.head_dim
    groups = max(1, n_heads // max(1, n_kv))

    # Resolve target word direction in unembedding space (defaults to the next token after tq, or top predicted token)
    target_id = None
    target_piece = ""
    if target_word and target_word.strip():
        cands = adapter._word_ids(target_word.strip())
        if cands:
            target_id, target_piece = int(cands[0][0]), cands[0][1]
    if target_id is None:
        if tq + 1 < len(ids):
            target_id = int(ids[tq + 1])
            target_piece = pieces[tq + 1]
        else:
            S = len(schedule)
            if (S - 1) in (cap.get("resid_post") or {}):
                lg = adapter.decode_layer(cap, schedule, S - 1, 0)[tq].float()
                target_id = int(lg.argmax().item())
                target_piece = adapter.tok.piece(target_id)

    u_dir = None
    if target_id is not None:
        U = m.unembed_weight()
        norm_w = m.norm_w("model.norm.weight").float().cpu()
        u_dir = (U[target_id].float().cpu() * norm_w)
        u_dir = u_dir / u_dir.norm().clamp(min=1e-8)

    cache = cap.get("cache")
    v_caps = cap.get("v") or {}

    heads_rows = []
    heatmap = []                                        # [n_blocks][n_heads] for the selected key_token (or max over keys < tq)
    per_token_totals = torch.zeros(tq + 1, dtype=torch.float32)

    for p, layer in enumerate(schedule):
        nid = schedule_nodes[p] if p < len(schedule_nodes) else f"block_{layer}"
        a = attn_map.get(p)
        if a is None or a.numel() == 0:
            heatmap.append([0.0] * n_heads)
            continue
        # a is [n_heads, n_q, n_k]
        a_f = a.float().cpu()
        nq, nk = a_f.shape[1], a_f.shape[2]
        q_idx = min(tq, nq - 1)
        max_k = min(tq + 1, nk)

        # Retrieve V vectors [max_k, n_kv, hd] if available, plus o_proj weight [d_model, n_heads * hd]
        v_tok = None
        if p in v_caps and v_caps[p] is not None:
            vt = v_caps[p].float().cpu()
            if vt.shape[0] >= max_k and vt.shape[-1] == n_kv * hd:
                v_tok = vt[:max_k].view(max_k, n_kv, hd)
        elif cache is not None and p < len(cache.v):
            _, kv_v = cache.slot_kv(p)
            if kv_v is not None and kv_v.shape[1] >= max_k:
                # kv_v is [n_kv, T, hd] -> [max_k, n_kv, hd]
                v_tok = kv_v[:, :max_k, :].transpose(0, 1).float().cpu()

        wo = None
        wo_name = f"model.layers.{layer}.self_attn.o_proj.weight"
        try:
            wo = m.W(wo_name).float().cpu()             # [d_model, n_heads * hd]
        except Exception:
            wo = None

        block_heat = []
        ks_to_check = [key_token] if (key_token is not None and 0 <= key_token < max_k) else list(range(max_k))

        for h in range(min(n_heads, a_f.shape[0])):
            kv_h = h // groups
            row_probs = a_f[h, q_idx, :max_k]           # [max_k]
            per_token_totals[:max_k] += row_probs

            if key_token is not None and 0 <= key_token < max_k:
                k_best = int(key_token)
            else:
                # Exclude self-attention at tq when tq > 0 if looking for previous tokens, or include all <= tq
                if max_k > 1 and key_token is None:
                    prev_probs = row_probs[: max_k - 1]
                    k_best = int(prev_probs.argmax().item())
                else:
                    k_best = int(row_probs.argmax().item())

            block_heat.append(round(float(row_probs[k_best].item()), 4))

            wo_h = wo[:, h * hd: (h + 1) * hd] if (wo is not None and wo.shape[1] >= (h + 1) * hd) else None

            for k_idx in ks_to_check:
                if key_token is None and max_k > 1 and k_idx == tq:
                    continue
                prob = float(row_probs[k_idx].item())
                if key_token is None and prob < 0.03:
                    continue
                val_norm = None
                logit_push = None
                if v_tok is not None and wo_h is not None and kv_h < v_tok.shape[1]:
                    v_vec = v_tok[k_idx, kv_h]                  # [hd]
                    contrib = prob * (wo_h @ v_vec)             # [d_model]
                    val_norm = round(float(contrib.norm().item()), 4)
                    if u_dir is not None and contrib.shape == u_dir.shape:
                        logit_push = round(float(torch.dot(contrib, u_dir).item()), 4)

                heads_rows.append({
                    "pos": p,
                    "layer": layer,
                    "node": nid,
                    "label": f"#{p} · L{layer} H{h}",
                    "head": h,
                    "kv_head": kv_h,
                    "query_token": tq,
                    "query_piece": pieces[tq],
                    "key_token": k_idx,
                    "key_piece": pieces[k_idx] if k_idx < len(pieces) else "",
                    "attn_weight": round(prob, 4),
                    "value_norm": val_norm if val_norm is not None else round(prob, 4),
                    "logit_push": logit_push,
                })
        heatmap.append(block_heat)

    heads_rows.sort(key=lambda r: (r["value_norm"] or 0.0, r["attn_weight"]), reverse=True)
    n_heads_total = max(1, len(schedule) * n_heads)
    token_summary = [
        {
            "index": k,
            "piece": pieces[k],
            "mean_attn": round(float(per_token_totals[k].item() / n_heads_total), 4),
        }
        for k in range(tq + 1)
    ]

    return {
        "query_token": tq,
        "query_piece": pieces[tq],
        "key_token": key_token,
        "key_piece": pieces[key_token] if (key_token is not None and 0 <= key_token < len(pieces)) else None,
        "target_id": target_id,
        "target_piece": target_piece,
        "n_tokens": T,
        "tokens": pieces[: tq + 1],
        "n_blocks": len(schedule),
        "n_heads": n_heads,
        "blocks": [f"#{p} · L{l}" for p, l in enumerate(schedule)],
        "nodes": schedule_nodes,
        "heatmap": heatmap,
        "token_summary": token_summary,
        "top_heads": heads_rows[: max(1, min(int(top_k), 200))],
    }


class AttnTracebackReq(BaseModel):
    run_id: str
    query_token: int = -1
    key_token: int | None = None
    target_word: str = ""
    top_k: int = 30


@router.post("/api/ext/attn_traceback/trace")
def api_attn_traceback(req: AttnTracebackReq):
    adapter, _ = _need("lm_tools", "Attention traceback")
    r = _run(req.run_id)
    cap = _caps(r)
    if r.plan.family != "causal_lm":
        raise HTTPException(400, "Attention traceback is for language-model runs.")
    try:
        with LOADED.lock:
            return trace_previous_tokens(
                adapter,
                cap,
                r.plan.schedule,
                r.plan.schedule_nodes,
                query_token=req.query_token,
                key_token=req.key_token,
                target_word=req.target_word,
                top_k=req.top_k,
            )
    except ValueError as e:
        raise HTTPException(400, str(e))


def _lab_trace_attention(
    lab,
    run_id: str,
    query_token: int = -1,
    key_token: int | None = None,
    target_word: str = "",
    top_k: int = 30,
) -> dict:
    """Rank `(block, head)` pairs across the model that attend to and copy from `key_token` into `query_token`."""
    return lab._post(
        "/api/ext/attn_traceback/trace",
        run_id=run_id,
        query_token=query_token,
        key_token=key_token,
        target_word=target_word,
        top_k=top_k,
    )


register_lab_method("trace_attention", _lab_trace_attention)
register_mcp_tool(
    "trace_previous_tokens",
    lambda run_id, query_token=-1, key_token=None, target_word="", top_k=25: trace_previous_tokens(
        LOADED.adapter,
        _caps(_run(run_id)),
        _run(run_id).plan.schedule,
        _run(run_id).plan.schedule_nodes,
        query_token=query_token,
        key_token=key_token,
        target_word=target_word,
        top_k=top_k,
    ),
    description="Find which (block, head) pairs attend to and copy from a previous token into the query token.",
)


@register_extension
class AttnTracebackPlugin(FeaturePlugin):
    id = "attn_traceback"
    name = "Previous-Token Attention Attribution Analyser"
    version = "0.2.0"
    view_type = "diagnose"
    default_enabled = False
    description = "Trace any previous token across all layers and heads ranked by attention weight, value-weighted norm ||alpha * v * W_O||, and direct logit push."
    families = ("causal_lm",)
    router = router
    ui_tabs = ()
