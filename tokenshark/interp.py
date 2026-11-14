"""Interpretability helpers shared by the web server, the Python API and the MCP tutor.

    activation patching   copy an activation from a "clean" run into a "corrupted" one    (resolve_sources)
    steering vectors      a direction in the residual stream, saved as a small file          (VECTORS, vector_from_*)
    neuron hunting        which MLP neurons fire where, and which words each one pushes      (top_neurons, neuron_detail)
    layer prompts         text turned into a KV prefix for chosen blocks                     (load_prefixes)
    tokenizer explorer    tokens of a text, vocabulary search, nearest tokens                (tokenize, vocab_search, …)
    sampler playground    the raw scores behind one prediction                               (final_logits)
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import torch

from .caches import LIBRARY
from .kvcache import KVCache
from .runs import DATA

VEC_DIR = DATA / "vectors"
PARTS = ("resid_pre", "q", "k", "v", "kv_k", "kv_v", "attn_out", "mlp_out", "resid_post")


# ======================================================================================
# Activations of a finished run
# ======================================================================================

def activation(cap: dict, hook: str, pos: int) -> torch.Tensor:
    """[tokens, d] — what the run had at `hook` of the block at schedule position `pos`."""
    if hook not in PARTS:
        raise ValueError(f"No stored activation for hook '{hook}' (stored: {', '.join(PARTS)}).")
    if hook in ("kv_k", "kv_v"):
        cache: KVCache | None = cap.get("cache")
        if cache is None or pos >= len(cache.k):
            raise ValueError(f"That run has no KV cache at schedule position {pos}.")
        kv = cache.k[pos] if hook == "kv_k" else cache.v[pos]
        if kv is None:
            raise ValueError(f"Block #{pos} is a linear-attention block (no per-token KV cache).")
        # kv is [n_kv, T, hd] -> transpose to [T, n_kv, hd] so token_dim = 0 matches the block hook
        return kv.transpose(0, 1).float().cpu()
    t = (cap.get(hook) or {}).get(pos)
    if t is None:
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
# Steering vectors
# ======================================================================================

class VectorLibrary:
    """Steering vectors as files under data/vectors/ (safetensors: one tensor 'v' + metadata)."""

    def __init__(self):
        VEC_DIR.mkdir(parents=True, exist_ok=True)

    def path(self, vid: str) -> Path:
        p = VEC_DIR / f"{vid}.safetensors"
        if not vid.replace("-", "").isalnum() or not p.exists():
            raise KeyError(vid)
        return p

    def save(self, v: torch.Tensor, name: str, meta: dict) -> dict:
        from safetensors.torch import save_file
        vid = uuid.uuid4().hex[:10]
        meta = {"id": vid, "name": name or f"vector {vid}", "created": time.time(), "dim": int(v.shape[-1]),
                "norm": round(float(v.float().norm()), 4), **meta}
        save_file({"v": v.detach().float().cpu().contiguous()}, str(VEC_DIR / f"{vid}.safetensors"),
                  metadata={"format": "tokenshark-vector/1", "meta": json.dumps(meta)})
        return meta

    def load(self, vid: str) -> tuple[torch.Tensor, dict]:
        from safetensors import safe_open
        with safe_open(str(self.path(vid)), framework="pt", device="cpu") as f:
            return f.get_tensor("v"), json.loads((f.metadata() or {}).get("meta", "{}"))

    def list(self) -> list[dict]:
        from safetensors import safe_open
        out = []
        for p in sorted(VEC_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    out.append({**json.loads((f.metadata() or {}).get("meta", "{}")), "id": p.stem})
            except Exception as e:  # noqa: BLE001
                out.append({"id": p.stem, "name": p.name, "error": str(e)})
        return out

    def delete(self, vid: str):
        self.path(vid).unlink()


VECTORS = VectorLibrary()


def vector_from_runs(cap_a: dict, cap_b: dict, pos: int, hook: str = "resid_post", token="last") -> torch.Tensor:
    """Contrastive direction: activation of run A minus run B at one block. Adding it to a third run pushes
    that run from "B-like" towards "A-like". token = last (the last token of each prompt, where the next word
    is decided) | mean (average over all tokens) | an index."""
    a, b = activation(cap_a, hook, pos), activation(cap_b, hook, pos)
    if isinstance(token, str) and token.strip().lower() == "mean":
        return a.float().mean(0) - b.float().mean(0)
    if isinstance(token, str) and token.strip().lower() == "last":
        ia, ib = min(cap_a.get("n_in") or a.shape[0], a.shape[0]) - 1, min(cap_b.get("n_in") or b.shape[0], b.shape[0]) - 1
        return a[ia].float() - b[ib].float()
    return a[_token_index(token, a.shape[0])].float() - b[_token_index(token, b.shape[0])].float()


def vector_from_word(adapter, word: str, minus: str = "") -> tuple[torch.Tensor, dict]:
    """The output direction of a word: its row of the unembedding matrix (minus another word's, if given),
    scaled to length 1. Adding it to the residual stream raises that word's score directly."""
    def row(w):
        cands = adapter._word_ids(w)
        if not cands:
            raise ValueError("Type a word.")
        tid, piece, n = cands[0]
        return adapter.model.unembed_weight()[tid].float().cpu(), piece, n
    v, piece, n = row(word)
    meta = {"word": piece, "pieces": n}
    if minus.strip():
        v2, piece2, _ = row(minus)
        v, meta["minus"] = v - v2, piece2
    return v / v.norm().clamp(min=1e-8), meta


# ======================================================================================
# What interventions need from outside their run (patch source, steering vector, clamp range)
# ======================================================================================

def resolve_sources(plan, get_caps) -> dict:
    """{interceptor box id: source}. get_caps(run_id) returns that run's captures (KeyError if unknown)."""
    out = {}
    for iv in plan.interventions:
        if iv.op not in ("patch", "steer", "project_out", "clamp") or not iv.ref:
            continue
        if iv.op in ("steer", "project_out"):
            try:
                out[iv.node_id] = VECTORS.load(iv.ref)[0]
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
                # Reshape [T, heads * hd] -> [T, heads, hd] to match _head_hook slice
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
# KV prefixes of a plan: saved caches and layer prompts (text → KV for chosen blocks)
# ======================================================================================

_TEXT_PREFIX: dict = {}


def load_prefixes(plan, adapter, state_key: str = "") -> list[KVCache]:
    """Every KV prefix the flow prepends, in order. A layer prompt (text typed into a box) is pushed through all
    blocks once and its keys/values are kept; which blocks then see them is decided per box."""
    specs = plan.params.get("prefixes") or [{"cache_id": plan.params["prefix_cache"]}]
    out, system = [], None
    for d in specs:
        if d.get("text"):
            key = (adapter.spec.get("id"), id(adapter), state_key, d["text"], bool(d.get("as_system")), tuple(plan.schedule))
            if key not in _TEXT_PREFIX:
                if len(_TEXT_PREFIX) > 24:
                    _TEXT_PREFIX.clear()
                _TEXT_PREFIX[key] = adapter.build_prefix(plan, d["text"], bool(d.get("as_system")))
            c, extra = _TEXT_PREFIX[key]
        else:
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
# Neuron hunting
# ======================================================================================

def _neuron_acts(cap: dict, pos: int) -> torch.Tensor:
    a = (cap.get("neurons") or {}).get(pos)
    if a is None:
        if not cap.get("neurons"):
            raise ValueError("This run did not keep neuron activations (too long for the budget, or it was made "
                             "before neurons were recorded). Use a shorter prompt and run again.")
        raise ValueError(f"No block at schedule position {pos}.")
    return a


def neuron_words(adapter, layer: int, idx: list[int], k: int = 8) -> list[dict]:
    """What each neuron writes, read as words: its output weights pushed through the final norm weights and the
    unembedding. promotes = words whose score it raises when it fires; suppresses = the opposite."""
    m = adapter.model
    wd = m.W(f"model.layers.{layer}.mlp.down_proj.weight")[:, torch.tensor(idx, device=m.device)].float().T   # [n, d]
    wd = (wd * m.norm_w("model.norm.weight").float()).cpu()
    U = m.unembed_weight()
    scores = torch.cat([wd @ U[i:i + 32768].float().cpu().T for i in range(0, U.shape[0], 32768)], 1)         # [n, vocab]
    out = []
    for r in range(len(idx)):
        hi, lo = scores[r].topk(k), (-scores[r]).topk(k)
        out.append({"promotes": [adapter.tok.piece(int(i)) for i in hi.indices],
                    "suppresses": [adapter.tok.piece(int(i)) for i in lo.indices]})
    return out


def top_neurons(adapter, cap: dict, schedule: list[int], pos: int, token: int = -1, k: int = 20,
                by: str = "token") -> dict:
    """The strongest MLP neurons of one block. by='token': at one token; by='max': their peak over the whole run."""
    a = _neuron_acts(cap, pos).float()
    T = a.shape[0]
    t = min(max(token if token >= 0 else T + token, 0), T - 1)
    score = a[t] if by == "token" else a.gather(0, a.abs().argmax(0, keepdim=True))[0]
    idx = score.abs().topk(min(k, score.shape[0])).indices.tolist()
    rng = (cap.get("unit_range") or {}).get((pos, "mlp_neurons"))
    words = neuron_words(adapter, schedule[pos], idx)
    rows = []
    for i, w in zip(idx, words):
        peak = int(a[:, i].abs().argmax())
        rows.append({"index": i, "act": round(float(score[i]), 4),
                     "min": round(float(rng[0][i]), 4) if rng else None, "max": round(float(rng[1][i]), 4) if rng else None,
                     "peak_token": peak, "peak_piece": adapter.tok.piece(cap["ids"][peak]) if peak < len(cap["ids"]) else "",
                     **w})
    return {"pos": pos, "layer": schedule[pos], "token": t, "by": by, "n_neurons": int(a.shape[1]), "n_tokens": T,
            "piece": adapter.tok.piece(cap["ids"][t]) if t < len(cap["ids"]) else "", "neurons": rows}


def neuron_detail(adapter, cap: dict, schedule: list[int], pos: int, index: int) -> dict:
    """One neuron: its activation at every token of the run, its range, and the words it pushes."""
    a = _neuron_acts(cap, pos)
    if not 0 <= index < a.shape[1]:
        raise ValueError(f"This block has neurons 0…{a.shape[1] - 1}.")
    acts = a[:, index].float()
    rng = (cap.get("unit_range") or {}).get((pos, "mlp_neurons"))
    return {"pos": pos, "layer": schedule[pos], "index": index,
            "tokens": [adapter.tok.piece(i) for i in cap["ids"][: acts.shape[0]]],
            "acts": [round(float(v), 4) for v in acts],
            "min": round(float(rng[0][index]), 4) if rng else None, "max": round(float(rng[1][index]), 4) if rng else None,
            **neuron_words(adapter, schedule[pos], [index], 12)[0]}


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
            "covered": round(float(torch.exp(torch.logsumexp(v, -1) - logz)), 5),      # probability mass of these rows at T = 1
            "rows": [[adapter.tok.piece(int(j)), int(j), round(float(x), 4)] for x, j in zip(v, i)]}


# ======================================================================================
# Tokenizer explorer
# ======================================================================================

def tokenize(adapter, text: str, chat: bool = False, system: str = "") -> dict:
    tok = adapter.tok
    rendered = text
    if chat and tok.chat_template:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        rendered = tok.apply_chat(msgs, True)
    ids = tok.encode(rendered, add_special=False)
    pieces = [tok.piece(i) for i in ids]
    nbytes = len(rendered.encode("utf-8"))
    return {"ids": ids, "pieces": pieces, "rendered": rendered if chat else None, "n_tokens": len(ids),
            "n_chars": len(rendered), "n_bytes": nbytes, "vocab": tok.vocab_size,
            "chars_per_token": round(len(rendered) / max(1, len(ids)), 2)}


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
    """Tokens whose input embedding points the same way (cosine): what the model starts out treating as similar."""
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


# ======================================================================================
# Activation patching sweep (causal tracing): which block, at which token, carries the answer?
# ======================================================================================

def iter_patch_sweep(adapter, plan, cap_src: dict, hook: str = "resid_post", tokens: str = "all", prefix=None):
    """Run the flow's prompt once per (block, token), each time copying that one activation from the source
    ("clean") run, and measure how far the clean run's answer comes back.

        recovery = (p_patched − p_corrupt) / (p_clean − p_corrupt)      for the clean run's next token

    0 = this activation changes nothing, 1 = it alone restores the clean answer. Yields the grid as it fills in.
    Both prompts should have the same number of tokens (swap one word), so position i means the same in both."""
    import copy
    from .flow import Intervention, parse_index_spec
    from .llm import Stepper

    plan = copy.deepcopy(plan)
    plan.params["max_new_tokens"] = 0
    plan.interventions = []
    S = len(plan.schedule)

    def last_logits(ivs, src=None):
        plan.interventions = ivs
        st = Stepper(adapter, plan, prefix=prefix)
        st.ctx.sources = {"sweep": src} if src is not None else {}
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
    res = {"hook": hook, "blocks": [f"#{p} · L{l}" for p, l in enumerate(plan.schedule)], "nodes": plan.schedule_nodes,
           "token_index": toks, "tokens": [adapter.tok.piece(st0.seq[t]) for t in toks],
           "source_tokens": [adapter.tok.piece(cap_src["ids"][t]) if t < len(cap_src["ids"]) else "" for t in toks],
           "answer": adapter.tok.piece(answer), "corrupt_answer": adapter.tok.piece(wrong),
           "p_clean": round(p_clean, 4), "p_corrupt": round(p_corr, 4), "same_length": n_src == n_in,
           "grid": [[None] * len(toks) for _ in range(S)], "prob": [[None] * len(toks) for _ in range(S)], "_progress": 0.0}
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
