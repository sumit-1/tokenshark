"""Mixture-of-experts tools: read the router, read the experts, and measure what each expert is for.

A dense block has one MLP that every token passes through. A mixture-of-experts (MoE) block has E small MLPs and
a *router* — a single matrix with one row per expert. For each token the router computes one score per expert,
keeps the k highest and runs only those experts. Two things follow that make MoE models unusually readable:

  1. The routing is a discrete, visible decision. "Which experts did this token use?" has an exact answer, unlike
     "which neurons mattered?". It is recorded for every token of every run (captures: router, moe_sel, moe_w).
  2. The MLP's write to the residual stream is already split into k named pieces — one per chosen expert
     (captures: moe_parts). Each piece can be read through the unembedding, attributed, banned or scaled.

    overview        every MoE block of a run: how often each expert was used, router entropy, load balance
    block_detail    one block: the routing table (token × expert), what each expert handled and what it wrote
    expert_detail   one expert: where it fired, the words it pushes, experts it co-fires with, its router row
    iter_ablation   knock experts out one at a time and measure what the answer loses (causal importance)
    router_rows     how similar the experts' router rows are (which experts compete for the same tokens)

All functions take the captures of a finished run (or a stepping session) and return plain JSON-able dicts.
"""
from __future__ import annotations

import copy
import math
from collections import Counter

import torch


def _need(cap: dict, pos: int):
    sel = (cap.get("moe_sel") or {}).get(pos)
    if sel is None:
        if not cap.get("moe_sel"):
            raise ValueError("This run has no expert routing: the model is not a mixture of experts (or no MoE block "
                             "has run yet).")
        raise ValueError(f"Block #{pos} is not a mixture-of-experts block.")
    return sel.long(), cap["moe_w"][pos].float()


def _probs(cap: dict, pos: int):
    lg = (cap.get("router") or {}).get(pos)
    return None if lg is None else torch.softmax(lg.float(), -1)


def _n_experts(adapter, layer: int) -> int:
    return adapter.model.moe_layers[layer]["n"]


def words_of(adapter, vecs: torch.Tensor, k: int = 8) -> list[dict]:
    """Read directions of the residual stream as words: scale by the final norm's weights, compare with every
    word's output vector. promotes = words whose score rises when this is added to the stream."""
    m = adapter.model
    v = (vecs.float().cpu() * m.norm_w("model.norm.weight").float().cpu())
    U = m.unembed_weight()
    scores = torch.cat([v @ U[i:i + 32768].float().cpu().T for i in range(0, U.shape[0], 32768)], 1)
    out = []
    for r in range(v.shape[0]):
        hi, lo = scores[r].topk(k), (-scores[r]).topk(k)
        out.append({"promotes": [adapter.tok.piece(int(i)) for i in hi.indices],
                    "suppresses": [adapter.tok.piece(int(i)) for i in lo.indices]})
    return out


def _answer_direction(adapter, cap: dict, schedule: list[int], t: int):
    """(direction d, answer piece, versus piece): x · d ≈ how much x raises logit(answer) − logit(runner-up) at
    token t, through the final norm frozen at its final scale (direct logit attribution)."""
    S = len(schedule)
    if (S - 1) not in cap["resid_post"] or t >= cap["resid_post"][S - 1].shape[0]:
        return None
    m = adapter.model
    x_final = cap["resid_post"][S - 1][t].float()
    lg = m.final(x_final[None].to(adapter.device, m.dtype))[0].float().cpu()
    top = lg.topk(2).indices.tolist()
    U = m.unembed_weight()[top].float().cpu()
    rms = float(torch.sqrt(x_final.pow(2).mean() + adapter.cfg.eps))
    d = (U[0] - U[1]) * m.norm_w("model.norm.weight").float().cpu() / rms
    return d, adapter.tok.piece(top[0]), adapter.tok.piece(top[1])


# ======================================================================================
# Overview: every MoE block of a run
# ======================================================================================

def overview(adapter, cap: dict, schedule: list[int], nodes: list[str]) -> dict:
    """Per MoE block: usage (share of tokens that used each expert), mean weight when used, router entropy in
    bits (0 = always certain, log2 E = no preference), and each token's top expert."""
    ids = cap["ids"]
    blocks = []
    for pos in sorted(cap.get("moe_sel") or {}):
        sel, w = _need(cap, pos)
        T, k = sel.shape
        E = _n_experts(adapter, schedule[pos])
        live = w != 0
        cnt = torch.zeros(E).index_add_(0, sel[live], torch.ones(int(live.sum())))
        wsum = torch.zeros(E).index_add_(0, sel[live], w[live])
        pr = _probs(cap, pos)
        ent = float((-(pr * (pr + 1e-12).log2()).sum(-1)).mean()) if pr is not None else None
        used = cnt / max(1, T)
        blocks.append({"pos": pos, "layer": schedule[pos], "node": nodes[pos], "n_experts": E, "k": k, "tokens": T,
                       "usage": [round(float(v), 4) for v in used],
                       "weight": [round(float(a / b), 4) if b else 0.0 for a, b in zip(wsum, cnt)],
                       "entropy": round(ent, 3) if ent is not None else None, "max_entropy": round(math.log2(E), 3),
                       "active": int((cnt > 0).sum()),
                       # 0 = perfectly even use of the experts, 1 = everything through the same k
                       "imbalance": round(float(used.std() / (used.mean() + 1e-9)) / math.sqrt(max(1.0, E / k - 1)), 3),
                       "top": sel[:, 0].tolist(), "top_w": [round(float(v), 3) for v in w[:, 0]]})
    if not blocks:
        raise ValueError("This run has no expert routing: the model is not a mixture of experts.")
    T = max(b["tokens"] for b in blocks)
    return {"tokens": [adapter.tok.piece(i) for i in ids[:T]], "n_in": cap.get("n_in"), "blocks": blocks,
            "moe": adapter.info()["moe"], "parts": bool(cap.get("moe_parts"))}


# ======================================================================================
# One block: routing table, what each expert handled, what each expert wrote
# ======================================================================================

def block_detail(adapter, cap: dict, schedule: list[int], pos: int, token: int = -1, top: int = 0) -> dict:
    sel, w = _need(cap, pos)
    T, k = sel.shape
    layer = schedule[pos]
    lay = adapter.model.moe_layers[layer]
    E = lay["n"]
    ids = cap["ids"][:T]
    t = min(max(token if token >= 0 else T + token, 0), T - 1)
    pr = _probs(cap, pos)
    parts = (cap.get("moe_parts") or {}).get(pos)                # [T, k, d] each chosen expert's weighted write
    shared = (cap.get("moe_shared") or {}).get(pos)
    live = w != 0
    dirn = _answer_direction(adapter, cap, schedule, t) if parts is not None else None

    experts = []
    mean_vec = {}
    for e in range(E):
        hit = (sel == e) & live
        rows, slot = torch.where(hit)
        n = int(rows.numel())
        row = {"index": e, "count": n, "usage": round(n / T, 4),
               "weight": round(float(w[hit].mean()), 4) if n else 0.0,
               "prob": round(float(pr[:, e].mean()), 4) if pr is not None else None}
        if n:
            c = Counter(adapter.tok.piece(ids[int(r)]) for r in rows)
            row["tokens"] = [[p, v] for p, v in c.most_common(8)]
            if parts is not None:
                vecs = parts[rows, slot].float()
                mean_vec[e] = vecs.mean(0)
                row["norm"] = round(float(vecs.norm(dim=-1).mean()), 3)
        experts.append(row)
    if mean_vec:
        order = sorted(mean_vec)
        for e, wd in zip(order, words_of(adapter, torch.stack([mean_vec[e] for e in order]))):
            experts[e].update(wd)

    # the chosen token: which experts, how much, what each wrote there and how hard it pushed for the answer
    here = []
    for j in range(k):
        if not bool(live[t, j]):
            continue
        e = int(sel[t, j])
        item = {"index": e, "weight": round(float(w[t, j]), 4)}
        if parts is not None:
            v = parts[t, j].float()
            item.update(norm=round(float(v.norm()), 3), **words_of(adapter, v[None], 6)[0])
            if dirn is not None:
                item["push"] = round(float(v @ dirn[0]), 3)
        here.append(item)
    shared_here = None
    if shared is not None and t < shared.shape[0]:
        v = shared[t].float()
        shared_here = {"norm": round(float(v.norm()), 3), **words_of(adapter, v[None], 6)[0]}
        if dirn is not None:
            shared_here["push"] = round(float(v @ dirn[0]), 3)

    out = {"pos": pos, "layer": layer, "n_experts": E, "k": k, "token": t, "piece": adapter.tok.piece(ids[t]),
           "tokens": [adapter.tok.piece(i) for i in ids], "n_in": cap.get("n_in"),
           "sel": sel.tolist(), "w": [[round(float(v), 3) for v in r] for r in w],
           "probs": [[round(float(v), 3) for v in r] for r in pr] if pr is not None else None,
           "experts": experts, "here": here, "shared_here": shared_here, "has_shared": bool(lay["shared"]),
           "shared_off": pos in (cap.get("moe_noshared") or ()),
           "parts": parts is not None,
           "answer": dirn[1] if dirn else None, "versus": dirn[2] if dirn else None,
           "native_k": adapter.cfg.moe["k"], "norm_topk": adapter.cfg.moe["norm_topk"]}
    if top:
        keep = sorted(range(E), key=lambda e: -experts[e]["count"])[:top]
        out["experts"] = [experts[e] for e in sorted(keep)]
    return out


def router_rows(adapter, layer: int, k: int = 10) -> dict:
    """The router is one matrix [experts, d]: row e is the direction of the residual stream that expert e
    answers to. Cosine similarity between rows shows which experts compete for the same tokens; comparing a row
    with the token embeddings gives the tokens it would attract if the stream still looked like an embedding
    (a fair reading in the first blocks, only a hint later)."""
    m = adapter.model
    lay = m.moe_layers.get(layer)
    if lay is None:
        raise ValueError(f"Layer {layer} is not a mixture-of-experts layer.")
    R = m.W(lay["router"]).float().cpu()
    Rn = R / R.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    emb = m.W.compact("model.embed_tokens.weight")
    h = torch.cat([Rn @ torch.nn.functional.normalize(emb[i:i + 32768].float().cpu(), dim=-1).T
                   for i in range(0, emb.shape[0], 32768)], 1)                     # [E, vocab]
    top = h.topk(k, -1).indices
    return {"layer": layer, "n_experts": R.shape[0], "norms": [round(float(v), 3) for v in R.norm(dim=-1)],
            "cosine": [[round(float(v), 3) for v in r] for r in Rn @ Rn.T],
            "attracts": [[adapter.tok.piece(int(i)) for i in r] for r in top]}


def expert_detail(adapter, cap: dict, schedule: list[int], pos: int, index: int) -> dict:
    """One expert of one block: its weight at every token, the words it writes, who it fires together with."""
    sel, w = _need(cap, pos)
    T, k = sel.shape
    layer = schedule[pos]
    E = _n_experts(adapter, layer)
    if not 0 <= index < E:
        raise ValueError(f"This block has experts 0…{E - 1}.")
    live = (sel == index) & (w != 0)
    weight = (w * live).sum(-1)                                                    # [T], 0 where not chosen
    pr = _probs(cap, pos)
    rows = torch.where(live.any(-1))[0]
    co = Counter(int(e) for r in rows for e, ok in zip(sel[r], w[r] != 0) if ok and int(e) != index)
    out = {"pos": pos, "layer": layer, "index": index, "n_experts": E,
           "tokens": [adapter.tok.piece(i) for i in cap["ids"][:T]],
           "weight": [round(float(v), 3) for v in weight],
           "prob": [round(float(v), 3) for v in pr[:, index]] if pr is not None else None,
           "rank": [int((pr[r] > pr[r, index]).sum()) + 1 for r in range(T)] if pr is not None else None,
           "count": int(rows.numel()), "co": [[e, c] for e, c in co.most_common(6)]}
    parts = (cap.get("moe_parts") or {}).get(pos)
    if parts is not None and rows.numel():
        r_, s_ = torch.where(live)
        out.update(words_of(adapter, parts[r_, s_].float().mean(0)[None], 12)[0])
    rr = router_rows(adapter, layer)
    out["attracts"] = rr["attracts"][index]
    sim = torch.tensor(rr["cosine"][index])
    sim[index] = -2
    out["similar"] = [[int(i), round(float(v), 3)] for v, i in zip(*sim.topk(min(5, E - 1)))]
    lay = adapter.model.moe_layers[layer]
    out["size"] = sum(int(torch.tensor(t.shape).prod()) for t in adapter.model.expert_weights(lay, index))
    return out


# ======================================================================================
# Ablation sweep: what does the answer lose when one expert is taken away?
# ======================================================================================

def iter_ablation(adapter, plan, pos: int = -1, prefix=None, mode: str = "ban"):
    """Run the flow's prompt once per expert that the prompt actually used, each time with that expert removed,
    and measure the damage to the next-token prediction.

        mode "ban"   the router may not pick the expert: another expert takes its place (how a real system would
                     degrade if the expert were deleted)
        mode "zero"  the expert is picked as usual but writes nothing (what its own contribution was worth)

    Reported per expert: the answer's probability, the change in logit(answer) − logit(runner-up), and the KL
    divergence from the untouched prediction. pos = −1 sweeps every MoE block. Yields as it fills in."""
    from .flow import Intervention
    from .llm import Stepper

    plan = copy.deepcopy(plan)
    plan.params["max_new_tokens"] = 0
    base_ivs = list(plan.interventions)

    def last(ivs):
        plan.interventions = base_ivs + ivs
        st = Stepper(adapter, plan, prefix=prefix)
        st.ctx.sources = dict(getattr(plan, "sources", None) or {})      # the flow's own patches / vectors still apply
        while st.step_token():
            pass
        return st.logits_rows[-1][-1].float().cpu(), st

    base, st0 = last([])
    logp = torch.log_softmax(base, -1)
    a, b = base.topk(2).indices.tolist()
    moe_pos = sorted(st0.ctx.store.get("moe_sel", {}))
    if not moe_pos:
        raise ValueError("This model has no mixture-of-experts blocks.")
    targets = moe_pos if pos < 0 else [pos]
    if any(p not in moe_pos for p in targets):
        raise ValueError(f"Block #{pos} is not a mixture-of-experts block.")
    jobs = []
    for p in targets:
        sel = torch.cat(st0.ctx.store["moe_sel"][p]).long()
        w = torch.cat(st0.ctx.store["moe_w"][p])
        E = adapter.model.moe_layers[plan.schedule[p]]["n"]
        cnt = torch.zeros(E).index_add_(0, sel[w != 0], torch.ones(int((w != 0).sum())))
        last_tok = {int(e): round(float(x), 3) for e, x in zip(sel[-1], w[-1]) if x > 0}
        for e in torch.where(cnt > 0)[0].tolist():
            jobs.append((p, e, round(float(cnt[e]) / sel.shape[0], 3), last_tok.get(e, 0.0)))
    res = {"mode": mode, "answer": adapter.tok.piece(a), "versus": adapter.tok.piece(b),
           "p_answer": round(float(logp[a].exp()), 4), "margin": round(float(base[a] - base[b]), 3),
           "prompt_tokens": st0.n_in, "rows": [], "total": len(jobs), "_progress": 0.0,
           "blocks": [{"pos": p, "layer": plan.schedule[p], "node": plan.schedule_nodes[p]} for p in targets]}
    yield copy.deepcopy(res)
    for i, (p, e, usage, w_last) in enumerate(jobs, 1):
        iv = Intervention(node_id="sweep", pos=p, hook="router" if mode == "ban" else "expert_out",
                          op="ban" if mode == "ban" else "zero", experts=str(e))
        lg, _ = last([iv])
        lp = torch.log_softmax(lg, -1)
        res["rows"].append({"pos": p, "layer": plan.schedule[p], "node": plan.schedule_nodes[p], "index": e,
                            "usage": usage, "weight_last": w_last,
                            "p_answer": round(float(lp[a].exp()), 4),
                            "d_margin": round(float((lg[a] - lg[b]) - (base[a] - base[b])), 3),
                            "kl": round(float((logp.exp() * (logp - lp)).sum()), 4),
                            "top": adapter.tok.piece(int(lg.argmax())), "flipped": int(lg.argmax()) != a})
        res["_progress"] = i / max(1, len(jobs))
        yield copy.deepcopy(res)
