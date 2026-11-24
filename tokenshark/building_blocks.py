"""Core building blocks shared by the engine, server, Python API, MCP server, and feature plugins (`tokenshark.building_blocks`).

Contains fundamental tensor/activation accessors, KV prefix loaders, tokenizer/vocabulary explorers,
KV cache telemetry, and attention tensor helpers. Feature-specific logic (neurons, steering, patching,
MoE, introspection, quantization, LoRA, training, guardrails, attention traceback, datasets) lives in
`tokenshark/extensions/`.
"""
from __future__ import annotations

import re as _re
import torch

from .caches import LIBRARY
from .kvcache import KVCache

PARTS = ("resid_pre", "q", "k", "v", "kv_k", "kv_v", "attn_out", "router", "mlp_out", "resid_post")


# ======================================================================================
# Activations of a finished run
# ======================================================================================

def activation(cap: dict, hook: str, pos: int) -> torch.Tensor:
    """[tokens, d] - what the run had at `hook` of the block at schedule position `pos`."""
    if hook not in PARTS:
        raise ValueError(f"No stored activation for hook '{hook}' (stored: {', '.join(PARTS)}).")
    if hook in ("kv_k", "kv_v"):
        cache: KVCache | None = cap.get("cache")
        if cache is None or pos >= len(cache.k):
            raise ValueError(f"That run has no KV cache at schedule position {pos}.")
        k_t, v_t = cache.slot_kv(pos)
        kv = k_t if hook == "kv_k" else v_t
        if kv is None:
            raise ValueError(f"Block #{pos} is a linear-attention block (no per-token KV cache).")
        return kv.transpose(0, 1).float().cpu()
    t = (cap.get(hook) or {}).get(pos)
    if t is None:
        if hook == "router":
            raise ValueError(f"Block #{pos} of that run has no router scores (it is not a mixture-of-experts block).")
        if hook in ("attn_out", "mlp_out", "q", "k", "v") and not cap.get(hook):
            raise ValueError(f"That run did not keep {hook} (too long for the capture budget). Patch resid_pre or "
                             f"resid_post instead, or use a shorter prompt.")
        raise ValueError(f"That run has no block at schedule position {pos}.")
    return t


def _token_index(spec, n: int) -> int:
    if isinstance(spec, str) and spec.strip().lower() == "last":
        return n - 1
    i = int(spec)
    if not -n <= i < n:
        raise ValueError(f"Token {i} is outside the run (it has {n} tokens).")
    return i % n


# ======================================================================================
# External sources resolution (patch source, steering vector, clamp range, custom ops)
# ======================================================================================

def resolve_sources(plan, get_caps, vectors_lib=None) -> dict:
    """{interceptor box id: source}. get_caps(run_id) returns that run's captures (KeyError if unknown)."""
    from .sdk import get_op
    if vectors_lib is None:
        from . import interp
        vectors_lib = interp.VECTORS
    out = {}
    for iv in plan.interventions:
        custom = get_op(iv.op)
        if custom is not None and custom.resolve_source is not None:
            res = custom.resolve_source(iv, get_caps)
            if res is not None:
                out[iv.node_id] = res
            continue
        if iv.op not in ("patch", "steer", "project_out", "clamp") or not iv.ref:
            continue
        if iv.op in ("steer", "project_out"):
            try:
                out[iv.node_id] = vectors_lib.load(iv.ref)[0]
            except KeyError:
                raise ValueError("An interceptor box points at a vector that no longer exists.") from None
            continue
        try:
            cap = get_caps(iv.ref)
        except KeyError:
            raise ValueError(f"Run '{iv.ref}' (the source for a {iv.op} box) is not in memory any more. "
                             f"Run it again, then pick it in the box.") from None
        if cap is None:
            raise ValueError(f"Run '{iv.ref}' has no captures left in memory. Run it again.")
        if iv.op == "patch":
            a = activation(cap, iv.hook, iv.pos)
            if iv.hook in ("q", "k", "v") and a.ndim == 2:
                hd = cap["cache"].fingerprint.get("head_dim") or 64
                a = a.view(a.shape[0], -1, hd)
            out[iv.node_id] = a
        else:
            rng = (cap.get("unit_range") or {}).get((iv.pos, iv.hook))
            if rng is None and iv.hook in PARTS:
                a = activation(cap, iv.hook, iv.pos).float()
                rng = (a.amin(0), a.amax(0))
            if rng is not None:
                out[iv.node_id] = {"min": rng[0], "max": rng[1]}
    return out


# ======================================================================================
# KV prefixes of a plan: saved caches wired into Embed
# ======================================================================================

def load_prefixes(plan, adapter, state_key: str = "") -> list[KVCache]:
    """Every saved KV cache the flow prepends, in order. Which blocks see each prefix is decided per box."""
    specs = plan.params.get("prefixes") or [{"cache_id": plan.params["prefix_cache"]}]
    out, system = [], None
    for d in specs:
        try:
            c, extra = LIBRARY.load(d["cache_id"])
        except KeyError:
            raise ValueError("The selected KV cache no longer exists.") from None
        fp = adapter.fingerprint()
        for k in ("hidden_size", "n_kv_heads", "head_dim"):
            if c.fingerprint.get(k) != fp.get(k):
                raise ValueError(f"The KV cache comes from a different model ({c.fingerprint.get('model_id')}).")
        if c.n_slots != len(plan.schedule):
            raise ValueError(f"The KV cache has {c.n_slots} block slots but this flow runs {len(plan.schedule)} "
                             f"blocks. Bypassed or repeated blocks change the count: use the flow the cache was "
                             f"made with, or rebuild the cache with this flow.")
        if extra.get("schedule") and list(extra["schedule"]) != list(plan.schedule):
            raise ValueError(f"The KV cache was made with blocks {extra['schedule']} but this flow runs "
                             f"{plan.schedule}. Each slot holds keys from a specific block; rebuild the cache "
                             f"with this flow.")
        out.append(c)
        system = system or extra.get("system_text")
    plan.params["prefix_system"] = system
    return out


# ======================================================================================
# Sampler playground: the scores behind one prediction
# ======================================================================================

def final_logits(adapter, cap: dict, schedule: list[int], position: int = -1, top: int = 1000) -> dict:
    S = len(schedule)
    if (S - 1) not in cap["resid_post"]:
        raise ValueError("The last block has not run yet.")
    lg = adapter.decode_layer(cap, schedule, S - 1, 0)
    T = lg.shape[0]
    t = min(max(position if position >= 0 else T + position, 0), T - 1)
    row = lg[t].float()
    v, i = row.topk(min(top, row.shape[0]))
    logz = float(torch.logsumexp(row, -1))
    ids = cap["ids"]
    return {"position": t, "token": adapter.tok.piece(ids[t]), "n_tokens": T, "vocab": int(row.shape[0]),
            "next_id": ids[t + 1] if t + 1 < len(ids) else None,
            "next_piece": adapter.tok.piece(ids[t + 1]) if t + 1 < len(ids) else None,
            "covered": round(float(torch.exp(torch.logsumexp(v, -1) - logz)), 5),
            "rows": [[adapter.tok.piece(int(j)), int(j), round(float(x), 4)] for x, j in zip(v, i)]}


# ======================================================================================
# Tokenizer explorer & Word -> Token Dictionary
# ======================================================================================

def _build_word_map(ids: list[int], pieces: list[str]) -> list[dict]:
    groups: list[dict] = []

    def _kind_of(p: str) -> str:
        s = p.strip()
        if (s.startswith("<|") and s.endswith("|>")) or (s.startswith("<") and s.endswith(">") and len(s) > 2):
            return "special"
        if "\\n" in p or "\\t" in p or not s:
            return "whitespace"
        if _re.match(r"^[\w\u0080-\uffff]+$", s):
            return "word"
        return "punct"

    for idx, (tid, piece) in enumerate(zip(ids, pieces)):
        k = _kind_of(piece)
        starts_with_space = piece.startswith(" ")
        if groups and not starts_with_space and k == "word" and groups[-1]["kind"] == "word":
            g = groups[-1]
            g["word"] += piece
            g["pieces"].append(piece)
            g["ids"].append(int(tid))
            g["positions"].append(idx)
            g["n_tokens"] = len(g["ids"])
            g["split"] = True
        else:
            groups.append({
                "word": piece,
                "pieces": [piece],
                "ids": [int(tid)],
                "positions": [idx],
                "n_tokens": 1,
                "split": False,
                "kind": k,
            })
    return groups


def tokenize(adapter, text: str, chat: bool = False, system: str = "",
             history: list[dict] | None = None, prefill: str = "", add_bos: bool = False) -> dict:
    tok = adapter.tok
    turns = [{"role": m["role"], "content": m["content"]} for m in (history or []) if m.get("content")]
    if text or not turns:
        turns.append({"role": "user", "content": text})
    if chat and tok.chat_template:
        msgs = ([{"role": "system", "content": system}] if system else []) + turns
        rendered = tok.apply_chat(msgs, True)
        ids = tok.encode(rendered, add_special=False)
    else:
        earlier = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in (history or []) if m.get("content"))
        rendered = (system + "\n\n" if system else "") + earlier + text
        ids = tok.encode(rendered, add_special=bool(add_bos))
    if prefill:
        prefill_ids = tok.encode(prefill, add_special=False)
        ids = ids + prefill_ids
        rendered = rendered + prefill
    pieces = [tok.piece(i) for i in ids]
    nbytes = len(rendered.encode("utf-8"))
    word_map = _build_word_map(ids, pieces)
    n_split = sum(1 for g in word_map if g["split"])
    return {"ids": ids, "pieces": pieces, "rendered": rendered if (chat or history or prefill or system) else None,
            "n_tokens": len(ids), "n_chars": len(rendered), "n_bytes": nbytes, "vocab": tok.vocab_size,
            "chars_per_token": round(len(rendered) / max(1, len(ids)), 2),
            "word_map": word_map, "n_words": len(word_map), "n_split_words": n_split}


# ======================================================================================
# KV Cache Analysis (per-block & per-token K/V norms, GQA savings, attention sinks)
# ======================================================================================

def _last_query_attention(adapter, cap: dict, cache: KVCache, p: int, K: torch.Tensor, qi: int,
                          layer: int | None = None):
    """Attention of the query at cache index `qi` over the cache of block slot p, with RoPE as it ran and
    with the query-key distance removed (every key rotated as if it sat at the query's own position).
    Returns (attn_rope [qi+1], attn_same_pos [qi+1], q_pos, k_pos [qi+1]) or None."""
    from .llm.ops import apply_rope, rope_inv_freq
    cfg = adapter.cfg
    hist = (cap.get("last_q") or {}).get(p) or {}
    rp = (cap.get("rope_pos") or {}).get(p)
    if qi not in hist or rp is None or getattr(cfg, "mla", None):
        return None
    q, qp = hist[qi]
    pos_k = rp[1]
    T = min(K.shape[1], len(pos_k), qi + 1)                 # the keys that query could see (causal)
    if T <= 0:
        return None
    K = K[:, :T].float().cpu()
    q = q.float()                                            # [H, hd]
    theta = ((cap.get("rope") or {}).get(p) or {}).get("theta")
    inv = rope_inv_freq(cfg, theta=float(theta) if theta else None, layer=layer).cpu()
    kp = pos_k[:T].float()
    qr = apply_rope(q[:, None, :], torch.tensor([qp]), inv)[:, 0]          # [H, hd]
    kr = apply_rope(K, kp, inv)                                             # [n_kv, T, hd]
    k0 = apply_rope(K, torch.full((T,), qp), inv)                          # distance 0 for every key
    rep = q.shape[0] // max(1, K.shape[0])
    if rep > 1:
        kr, k0 = kr.repeat_interleave(rep, 0), k0.repeat_interleave(rep, 0)
    scale = cfg.attn_scale if cfg.attn_scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    s_rope = torch.einsum("hd,htd->ht", qr, kr) * scale
    s_same = torch.einsum("hd,htd->ht", qr, k0) * scale
    allowed = torch.ones(T, dtype=torch.bool)
    if cache.block_caches.get(cache.shared.get(p, p)) is None:
        for a, b, slots in cache.prefix_spans:
            if slots is not None and p not in slots:
                allowed[a:min(b, T)] = False
    window = cfg.layer_window(layer)
    if window:
        allowed[:max(0, T - window)] = False
    s_rope = s_rope.masked_fill(~allowed, float("-inf"))
    s_same = s_same.masked_fill(~allowed, float("-inf"))
    return torch.softmax(s_rope, -1).mean(0), torch.softmax(s_same, -1).mean(0), qp, kp


def kv_analysis(adapter, cap: dict, schedule: list[int], schedule_nodes: list[str], blocks: str = "all",
                query: int | None = None) -> dict:
    """KV cache telemetry of a finished run.

    Per-token columns are averaged over the selected block slots (`blocks`: all | 5 | 3,7 | 10:14).
    Attention columns describe one query only - by default the forward pass that produced the last output
    token; `query` picks another cache index - not an average over the whole run."""
    from .flow import parse_index_spec
    cache: KVCache | None = cap.get("cache")
    if cache is None:
        raise ValueError("This run has no KV cache.")
    cfg = adapter.cfg
    summary = cache.summary(adapter.tok.piece)
    T = len(cache)
    S = len(schedule)
    n_heads = cfg.n_heads
    n_kv = cfg.n_kv
    hd = cfg.head_dim
    gqa_ratio = round(n_heads / max(1, n_kv), 2)
    gqa_saving_pct = round((1.0 - n_kv / max(1, n_heads)) * 100.0, 1)
    bytes_total = summary["bytes"]
    bytes_per_token = round(bytes_total / max(1, T), 1)
    mha_bytes_total = int(round(bytes_total * (n_heads / max(1, n_kv))))
    try:
        selected = set(parse_index_spec(blocks or "all", S))
    except ValueError as e:
        raise ValueError(f"Blocks: {e}")

    # Which forward pass produced the last output token: the newest one when that token was never fed back
    # (end-of-text, or a run stopped right after sampling), else the one before it.
    ids = list(cap.get("ids") or [])
    n_pre = int(cap.get("n_prefix") or 0)
    gen = (cap.get("token_ranges") or {}).get("generated")
    result = int(ids[-1]) if ids and gen is not None and len(ids) > gen.start else None
    back = 1 if result is not None and len(ids) + n_pre <= T and T >= 2 else 0
    last_qi = T - 1 - back
    qi = last_qi if query is None else int(query)
    if not 0 <= qi < T:
        raise ValueError(f"Query token #{qi} is outside the cache (0..{T - 1}).")
    if qi != last_qi:                             # the token that query predicted is simply the next one
        result = int(cache.tokens[qi + 1]) if qi + 1 < T else None
    origins = cache._origins()
    next_origin = origins[qi + 1][0] if qi + 1 < len(origins) else "generated"
    available = sorted(set().union(*[set(h) for h in (cap.get("last_q") or {}).values()])) if cap.get("last_q") else []
    rope_cfg = cap.get("rope") or {}
    out_blocks = []
    acc = {k: torch.zeros(T, dtype=torch.float32) for k in ("k", "v", "attn", "same", "pos")}
    n_norm = n_attn = 0
    q_pos_sum, q_pos_n = 0.0, 0

    for p in range(S):
        l = schedule[p]
        nid = schedule_nodes[p] if p < len(schedule_nodes) else f"block_{l}"
        bc = cache.block_caches.get(cache.shared.get(p, p))
        k_t, v_t = cache.slot_kv(p)
        base = {"pos": p, "layer": l, "node": nid, "label": f"#{p} · L{l}", "selected": p in selected,
                "rope": dict(rope_cfg.get(p) or {}), "shares": cache.shared.get(p)}
        if k_t is None or v_t is None:
            out_blocks.append({**base, "linear": True, "override": None, "n_tokens": 0,
                               "k_norm_mean": None, "v_norm_mean": None, "k_norm_max": None, "v_norm_max": None,
                               "sink_token": None, "sink_piece": None, "sink_attn": None, "q_pos": None, "bytes": 0})
            continue
        T_b = k_t.shape[1]
        tok_list = bc.tokens if bc is not None else cache.tokens
        kn = k_t.float().norm(dim=-1).mean(0).cpu()[:T_b]
        vn = v_t.float().norm(dim=-1).mean(0).cpu()[:T_b]
        la = _last_query_attention(adapter, cap, cache, p, k_t, qi if bc is None else T_b - 1 - back, l)
        use = p in selected and bc is None
        if use:
            m = min(T, len(kn))
            acc["k"][:m] += kn[:m]
            acc["v"][:m] += vn[:m]
            n_norm += 1
        sink_tok = sink_piece = sink_attn = q_pos = None
        if la is not None:
            att, same, q_pos, kp = la
            sink_tok = int(att.argmax())
            sink_piece = adapter.tok.piece(tok_list[sink_tok]) if sink_tok < len(tok_list) else ""
            sink_attn = round(float(att[sink_tok]), 4)
            if use:
                m = min(T, len(att))
                acc["attn"][:m] += att[:m]
                acc["same"][:m] += same[:m]
                acc["pos"][:m] += kp[:m]
                q_pos_sum += q_pos
                q_pos_n += 1
                n_attn += 1
        b_bytes = 0 if p in cache.shared else int(k_t.numel() * k_t.element_size() + v_t.numel() * v_t.element_size())
        out_blocks.append({
            **base, "linear": False, "n_tokens": T_b,
            "override": bc.summary(adapter.tok.piece) if bc is not None else None,
            "k_norm_mean": round(float(kn.mean()), 3) if len(kn) else 0.0,
            "v_norm_mean": round(float(vn.mean()), 3) if len(vn) else 0.0,
            "k_norm_max": round(float(kn.max()), 3) if len(kn) else 0.0,
            "v_norm_max": round(float(vn.max()), 3) if len(vn) else 0.0,
            "sink_token": sink_tok, "sink_piece": sink_piece, "sink_attn": sink_attn,
            "q_pos": round(q_pos, 2) if q_pos is not None else None,
            "bytes": b_bytes,
        })

    if n_norm:
        acc["k"] /= n_norm
        acc["v"] /= n_norm
    if n_attn:
        for k in ("attn", "same", "pos"):
            acc[k] /= n_attn
    top = int(acc["attn"].argmax()) if T and n_attn else None
    q_pos_avg = q_pos_sum / q_pos_n if q_pos_n else None
    tokens = []
    for t in range(T):
        tid = int(cache.tokens[t])
        orig, lab = origins[t] if t < len(origins) else ("prompt", "")
        stored = round(float(cache.pos[t]), 2)
        seen = n_attn and t <= qi                 # keys after the query did not exist yet when it ran
        att = round(float(acc["attn"][t]), 4) if seen else None
        same = round(float(acc["same"][t]), 4) if seen else None
        eff = round(float(acc["pos"][t]), 2) if seen else stored
        tokens.append({
            "index": t, "id": tid, "piece": adapter.tok.piece(tid),
            "position": stored,                     # position stored in the cache
            "eff_position": eff,                    # position RoPE actually used in the selected blocks
            "distance": round(q_pos_avg - eff, 2) if q_pos_avg is not None else None,
            "origin": orig, "label": lab,
            "k_norm": round(float(acc["k"][t]), 3),
            "v_norm": round(float(acc["v"][t]), 3),
            "attn_received": att,                   # share of the newest query's attention
            "attn_same_pos": same,                  # the same, if this key sat at the query's own position
            "rope_effect": round(att - same, 4) if seen else None,   # < 0: attention lost to RoPE distance
            "is_sink": bool(top == t and att is not None and att > 1.5 / max(1, T)),
        })

    return {
        "n_tokens": T, "n_slots": S, "n_heads": n_heads, "n_kv_heads": n_kv, "head_dim": hd,
        "rot_dim": getattr(cfg, "rot_dim", None) or hd, "rope_theta": cfg.theta, "max_context": getattr(cfg, "max_pos", None),
        "gqa_ratio": gqa_ratio, "gqa_saving_pct": gqa_saving_pct,
        "bytes_total": bytes_total, "bytes_per_token": bytes_per_token, "mha_bytes_total": mha_bytes_total,
        "segments": summary["segments"], "linear_slots": summary["linear_slots"],
        "block_overrides": summary.get("block_overrides") or {},
        "selected": sorted(selected), "n_used": n_norm, "n_attn_used": n_attn,
        "query": {"index": qi, "piece": adapter.tok.piece(int(cache.tokens[qi])) if T else "",
                  "position": round(q_pos_avg, 2) if q_pos_avg is not None else (round(float(cache.pos[qi]), 2) if T else None),
                  "result_id": result, "result_piece": adapter.tok.piece(result) if result is not None else None,
                  "result_kind": next_origin,      # generated = sampled by this pass; else the text just continues
                  "is_last": qi == last_qi, "last_index": last_qi,
                  "first_available": available[0] if available else None},
        "tokens": tokens,
        "blocks": out_blocks,
    }


def rope_pairs(adapter, cap: dict, schedule: list[int], block: int, token: int, head: int = 0,
               kind: str = "k") -> dict:
    """One token's key (or query) vector of one block and head, cut into its RoPE pairs.

    RoPE treats dims (j, j + half) as one 2-D point and turns it by position x frequency_j ("rotate half").
    For each pair: its size (hand length), its angle before any rotation, and its speed in radians per position.
    Keys come from the cache (stored before RoPE); queries from the run's query history."""
    from .llm.ops import rope_inv_freq
    cache: KVCache | None = cap.get("cache")
    if cache is None:
        raise ValueError("This run has no KV cache.")
    cfg = adapter.cfg
    S = len(schedule)
    if not 0 <= block < S:
        raise ValueError(f"Block #{block} is not in this run (0..{S - 1}).")
    layer = schedule[block]
    if cfg.is_linear(layer) or getattr(cfg, "mla", None):
        raise ValueError(f"Block #{block} has no RoPE keys (linear or latent attention).")
    T = len(cache)
    if not 0 <= token < T:
        raise ValueError(f"Token #{token} is outside the cache (0..{T - 1}).")
    if kind == "q":
        hist = (cap.get("last_q") or {}).get(block) or {}
        if token not in hist:
            raise ValueError(f"The query of token #{token} was not kept for block #{block}.")
        vecs = hist[token][0].float()                         # [n_heads, hd]
    else:
        K, _ = cache.slot_kv(block)
        if K is None or token >= K.shape[1]:
            raise ValueError(f"Block #{block} has no key for token #{token}.")
        vecs = K[:, token].float().cpu()                      # [n_kv, hd]
    n_h = vecs.shape[0]
    if not 0 <= head < n_h:
        raise ValueError(f"Head {head} does not exist ({n_h} {'query' if kind == 'q' else 'key/value'} heads).")
    theta = ((cap.get("rope") or {}).get(block) or {}).get("theta")
    inv = rope_inv_freq(cfg, theta=float(theta) if theta else None, layer=layer).float().cpu()
    x = vecs[head]
    hd = x.shape[-1]
    half = inv.shape[0]                                       # rotated dims: [0, 2*half); the rest never turn
    re, im = x[:half], x[half:2 * half]
    mag = (re ** 2 + im ** 2).sqrt()
    rest = float(x[2 * half:].norm()) if 2 * half < hd else 0.0
    rp = (cap.get("rope_pos") or {}).get(block)
    eff = float(rp[1][token]) if rp is not None and token < len(rp[1]) else float(cache.pos[token])
    lr = cfg.layer_rope(layer) or {}
    pairs = [{"i": j, "re": round(float(re[j]), 5), "im": round(float(im[j]), 5), "mag": round(float(mag[j]), 5),
              "freq": float(inv[j])} for j in range(half)]
    return {
        "block": block, "layer": layer, "token": token, "piece": adapter.tok.piece(int(cache.tokens[token])),
        "kind": kind, "head": head, "n_heads": n_h, "head_dim": hd, "n_pairs": half, "unrotated_norm": round(rest, 5),
        "position": round(float(cache.pos[token]), 3), "eff_position": round(eff, 3),
        "theta": float(theta) if theta else float(lr.get("rope_theta") or cfg.theta),
        "rope_type": lr.get("rope_type") or (cfg.rope_scaling or {}).get("rope_type") or "default",
        "max_context": getattr(cfg, "max_pos", None), "norm": round(float(x.norm()), 5),
        "tokens": [adapter.tok.piece(int(t)) for t in cache.tokens], "n_blocks": S,
        "linear_blocks": [p for p, l in enumerate(schedule) if cfg.is_linear(l)],
        "pairs": pairs,
    }


def _pieces(adapter) -> list[str]:
    if getattr(adapter, "_vocab_pieces", None) is None:
        adapter._vocab_pieces = [adapter.tok.piece(i) for i in range(adapter.tok.vocab_size)]
    return adapter._vocab_pieces


def vocab_search(adapter, q: str, limit: int = 80) -> dict:
    """Vocabulary entries containing q (exact matches first, then shortest)."""
    pieces = _pieces(adapter)
    hits = [(i, p) for i, p in enumerate(pieces) if q in p] if q else []
    hits.sort(key=lambda h: (h[1].strip() != q.strip(), len(h[1]), h[0]))
    return {"query": q, "n": len(hits), "tokens": [[p, i] for i, p in hits[:limit]], "vocab": len(pieces)}


def token_neighbors(adapter, token_id: int, k: int = 12) -> dict:
    """Tokens whose input embedding points the same way (cosine)."""
    m = adapter.model
    if not 0 <= token_id < m.cfg.vocab:
        raise ValueError("Token id outside the vocabulary.")
    if getattr(m, "_w_in", None) is None:
        m._w_in = m.unembed_weight() if m.tied else m.W.compact("model.embed_tokens.weight")
        m._w_in_norm = torch.cat([m._w_in[i:i + 32768].float().norm(dim=-1).cpu() for i in range(0, m._w_in.shape[0], 32768)])
    E, norms = m._w_in, m._w_in_norm
    v = E[token_id].float().cpu()
    sims = torch.cat([E[i:i + 32768].float().cpu() @ v for i in range(0, E.shape[0], 32768)]) / (norms * v.norm()).clamp(min=1e-8)
    sims[token_id] = -2
    top = sims.topk(k)
    return {"id": token_id, "piece": adapter.tok.piece(token_id), "norm": round(float(v.norm()), 3),
            "neighbors": [[adapter.tok.piece(int(i)), int(i), round(float(s), 3)] for s, i in zip(top.values, top.indices)]}
