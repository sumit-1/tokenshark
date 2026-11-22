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

def kv_analysis(adapter, cap: dict, schedule: list[int], schedule_nodes: list[str]) -> dict:
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

    origins = cache._origins()
    attn_map = cap.get("attn") or {}
    blocks = []
    tok_k_norms = torch.zeros(T, dtype=torch.float32)
    tok_v_norms = torch.zeros(T, dtype=torch.float32)
    tok_attn_in = torch.zeros(T, dtype=torch.float32)
    n_full_blocks = 0
    n_attn_blocks = 0

    for p in range(S):
        l = schedule[p]
        nid = schedule_nodes[p] if p < len(schedule_nodes) else f"block_{l}"
        bc = cache.block_caches.get(p)
        k_t, v_t = cache.slot_kv(p)
        if k_t is None or v_t is None:
            blocks.append({
                "pos": p, "layer": l, "node": nid, "label": f"#{p} · L{l}",
                "linear": True, "override": None, "n_tokens": 0,
                "k_norm_mean": None, "v_norm_mean": None,
                "k_norm_max": None, "v_norm_max": None,
                "sink_token": None, "sink_piece": None, "sink_attn": None, "bytes": 0,
            })
            continue
        T_b = k_t.shape[1]
        tok_list = bc.tokens if bc is not None else cache.tokens
        kn = k_t.float().norm(dim=-1).mean(0).cpu()[:T_b]
        vn = v_t.float().norm(dim=-1).mean(0).cpu()[:T_b]
        if bc is None:
            tok_k_norms[:min(T, len(kn))] += kn[:T]
            tok_v_norms[:min(T, len(vn))] += vn[:T]
            n_full_blocks += 1

        sink_tok, sink_piece, sink_attn = None, None, None
        a = attn_map.get(p)
        if a is not None and a.numel() > 0:
            col_mean = a.float().mean(dim=(0, 1)).cpu()[:T_b]
            if bc is None:
                tok_attn_in[:min(T, len(col_mean))] += col_mean[:T]
                n_attn_blocks += 1
            sink_tok = int(col_mean.argmax().item())
            sink_piece = adapter.tok.piece(tok_list[sink_tok]) if sink_tok < len(tok_list) else ""
            sink_attn = round(float(col_mean[sink_tok].item()), 4)

        b_bytes = int(k_t.numel() * k_t.element_size() + v_t.numel() * v_t.element_size())
        blocks.append({
            "pos": p, "layer": l, "node": nid, "label": f"#{p} · L{l}",
            "linear": False,
            "n_tokens": T_b,
            "override": bc.summary(adapter.tok.piece) if bc is not None else None,
            "k_norm_mean": round(float(kn.mean().item()), 3) if len(kn) else 0.0,
            "v_norm_mean": round(float(vn.mean().item()), 3) if len(vn) else 0.0,
            "k_norm_max": round(float(kn.max().item()), 3) if len(kn) else 0.0,
            "v_norm_max": round(float(vn.max().item()), 3) if len(vn) else 0.0,
            "sink_token": sink_tok, "sink_piece": sink_piece, "sink_attn": sink_attn,
            "bytes": b_bytes,
        })

    if n_full_blocks > 0:
        tok_k_norms /= n_full_blocks
        tok_v_norms /= n_full_blocks
    if n_attn_blocks > 0:
        tok_attn_in /= n_attn_blocks

    top_sink_idx = int(tok_attn_in.argmax().item()) if T > 0 and n_attn_blocks > 0 else None
    tokens = []
    for t in range(T):
        tid = int(cache.tokens[t])
        orig, lab = origins[t] if t < len(origins) else ("prompt", "")
        attn_val = round(float(tok_attn_in[t].item()), 4) if n_attn_blocks > 0 else None
        is_sink = bool(top_sink_idx == t and attn_val is not None and attn_val > (1.5 / max(1, T)))
        tokens.append({
            "index": t,
            "id": tid,
            "piece": adapter.tok.piece(tid),
            "position": round(float(cache.pos[t].item()), 2),
            "origin": orig,
            "label": lab,
            "k_norm": round(float(tok_k_norms[t].item()), 3),
            "v_norm": round(float(tok_v_norms[t].item()), 3),
            "attn_received": attn_val,
            "is_sink": is_sink,
        })

    return {
        "n_tokens": T,
        "n_slots": S,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": hd,
        "gqa_ratio": gqa_ratio,
        "gqa_saving_pct": gqa_saving_pct,
        "bytes_total": bytes_total,
        "bytes_per_token": bytes_per_token,
        "mha_bytes_total": mha_bytes_total,
        "segments": summary["segments"],
        "linear_slots": summary["linear_slots"],
        "block_overrides": summary.get("block_overrides") or {},
        "tokens": tokens,
        "blocks": blocks,
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
