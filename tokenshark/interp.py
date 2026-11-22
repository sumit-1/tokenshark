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
PARTS = ("resid_pre", "q", "k", "v", "kv_k", "kv_v", "attn_out", "router", "mlp_out", "resid_post")


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
        k_t, v_t = cache.slot_kv(pos)
        kv = k_t if hook == "kv_k" else v_t
        if kv is None:
            raise ValueError(f"Block #{pos} is a linear-attention block (no per-token KV cache).")
        # kv is [n_kv, T, hd] -> transpose to [T, n_kv, hd] so token_dim = 0 matches the block hook
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
# Neuron hunting
# ======================================================================================

def _neuron_acts(cap: dict, pos: int) -> torch.Tensor:
    a = (cap.get("neurons") or {}).get(pos)
    if a is None:
        if pos in (cap.get("moe_sel") or {}):
            raise ValueError("This is a mixture-of-experts block: its neurons live inside experts, and each token only "
                             "runs a few of them. Open the Experts tab to see which experts fired and what they wrote.")
        if not cap.get("neurons"):
            raise ValueError("This run did not keep neuron activations (too long for the budget, or it was made "
                             "before neurons were recorded). Use a shorter prompt and run again.")
        raise ValueError(f"No block at schedule position {pos}.")
    return a


def _default_ref_resid(adapter, schedule: list[int], pos: int) -> torch.Tensor:
    """When no run capture is available yet, run a fast 1-pass prefill on a short chat prompt
    so `through > 0` neuron decoding has a realistic residual stream `h_p` at block `pos`."""
    from .flow import Plan
    from .llm import Stepper
    cache = getattr(adapter, "_ref_resid_cache", None)
    if cache is None:
        cache = {}
        setattr(adapter, "_ref_resid_cache", cache)
    key = tuple(schedule)
    if key not in cache:
        p = Plan(
            family="causal_lm",
            schedule=list(schedule),
            schedule_nodes=[f"b{i}" for i in range(len(schedule))],
            params={
                "prompt": "Can you confirm my refund?",
                "system": "You are a helpful airline customer service assistant.",
                "use_chat_template": True,
                "max_new_tokens": 0,
                "temperature": 0.0,
                "top_k": 50,
                "top_p": 1.0,
                "seed": 0,
                "neuron_budget_mb": 0,
                "attn_budget_mb": 0,
                "parts_budget_mb": 0,
            },
        )
        st = Stepper(adapter, p)
        while st.step_token():
            pass
        res = st.finish()
        rp = res["captures"].get("resid_post", {})
        cache[key] = {k: v[-1].detach().float().cpu() for k, v in rp.items()}
    return cache[key].get(int(pos), torch.zeros(adapter.model.cfg.hidden))


def neuron_words(adapter, layer: int, idx: list[int], k: int = 8, through: int = 0,
                 schedule: list[int] | None = None, cap: dict | None = None,
                 pos: int | None = None, token: int = -1, ref_resid: torch.Tensor | None = None) -> list[dict]:
    """What each neuron writes, read as words: its output weights pushed through the final norm weights and the
    unembedding (`through=0`), or pushed through the last `through` blocks on top of a reference residual stream
    and then unembedded (`through > 0`, like Logit Lens / Word Cloud).
    promotes = words whose score it raises when it fires; suppresses = the opposite."""
    from .llm import rms_norm
    m = adapter.model
    wd = m.W(f"model.layers.{layer}.mlp.down_proj.weight")[:, torch.tensor(idx, device=m.device)].float().T   # [n, d]
    sched = schedule or list(range(m.cfg.n_layers))
    p = pos if pos is not None else (sched.index(layer) if layer in sched else layer)
    S = len(sched)
    start = max(p + 1, S - int(through)) if int(through or 0) > 0 else S

    if start < S:
        # Reference residual stream at block p: from `ref_resid`, from `cap["resid_post"][p]`, or a cached 1-pass prefill
        base_h = None
        if ref_resid is not None:
            base_h = ref_resid.float().to(m.device)
        elif cap is not None and p in (cap.get("resid_post") or {}):
            rp = cap["resid_post"][p]
            T = rp.shape[0]
            t = min(max(token if token >= 0 else T + token, 0), T - 1)
            base_h = rp[t].float().to(m.device)
        else:
            base_h = _default_ref_resid(adapter, sched, p).to(m.device)
        alpha = 1.0
        x_base = base_h.unsqueeze(0).to(m.dtype)
        x_pert = (base_h.unsqueeze(0) + alpha * wd).to(m.dtype)
        x_all = torch.cat([x_base, x_pert], dim=0)
        for q in range(start, S):
            lq = sched[q]
            if lq in getattr(m, "moe_layers", {}):
                continue
            pre = f"model.layers.{lq}."
            h2 = rms_norm(x_all, m.norm_w(pre + "post_attention_layernorm.weight"), m.cfg.eps)
            gate = h2 @ m.W(pre + "mlp.gate_proj.weight").T
            up = h2 @ m.W(pre + "mlp.up_proj.weight").T
            x_all = x_all + (m.act(gate) * up) @ m.W(pre + "mlp.down_proj.weight").T
        h_out = m.final_norm(x_all).float().cpu()
        wd = (h_out[1:] - h_out[:1]) / alpha
    else:
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
                by: str = "token", through: int = 0) -> dict:
    """The strongest MLP neurons of one block. by='token': at one token; by='max': their peak over the whole run."""
    a = _neuron_acts(cap, pos).float()
    T = a.shape[0]
    t = min(max(token if token >= 0 else T + token, 0), T - 1)
    score = a[t] if by == "token" else a.gather(0, a.abs().argmax(0, keepdim=True))[0]
    idx = score.abs().topk(min(k, score.shape[0])).indices.tolist()
    rng = (cap.get("unit_range") or {}).get((pos, "mlp_neurons"))
    words = neuron_words(adapter, schedule[pos], idx, through=through, schedule=schedule, cap=cap, pos=pos, token=t)
    rows = []
    for i, w in zip(idx, words):
        peak = int(a[:, i].abs().argmax())
        rows.append({"index": i, "act": round(float(score[i]), 4),
                     "min": round(float(rng[0][i]), 4) if rng else None, "max": round(float(rng[1][i]), 4) if rng else None,
                     "peak_token": peak, "peak_piece": adapter.tok.piece(cap["ids"][peak]) if peak < len(cap["ids"]) else "",
                     **w})
    return {"pos": pos, "layer": schedule[pos], "token": t, "by": by, "through": through,
            "n_neurons": int(a.shape[1]), "n_tokens": T,
            "piece": adapter.tok.piece(cap["ids"][t]) if t < len(cap["ids"]) else "", "neurons": rows}


def neuron_detail(adapter, cap: dict, schedule: list[int], pos: int, index: int, through: int = 0) -> dict:
    """One neuron: its activation at every token of the run, its range, and the words it pushes."""
    a = _neuron_acts(cap, pos)
    if not 0 <= index < a.shape[1]:
        raise ValueError(f"This block has neurons 0…{a.shape[1] - 1}.")
    acts = a[:, index].float()
    rng = (cap.get("unit_range") or {}).get((pos, "mlp_neurons"))
    return {"pos": pos, "layer": schedule[pos], "index": index, "through": through,
            "tokens": [adapter.tok.piece(i) for i in cap["ids"][: acts.shape[0]]],
            "acts": [round(float(v), 4) for v in acts],
            "min": round(float(rng[0][index]), 4) if rng else None, "max": round(float(rng[1][index]), 4) if rng else None,
            **neuron_words(adapter, schedule[pos], [index], 12, through=through, schedule=schedule, cap=cap, pos=pos)[0]}


# ======================================================================================
# A/B Neuron Identification System (Contrastive Detector Hunt, Writer Search & Batch Test)
# ======================================================================================

def _span_slice(acts: torch.Tensor, n_in: int, span: str = "reply") -> torch.Tensor:
    """Extract the token slice of `acts` [T, d_ff] according to `span`:
    - 'reply': from the last prompt token (n_in - 1) through all generated tokens (default for detector hunting)
    - 'last': only the last prompt token (n_in - 1), where the first reply token is decided
    - 'generated': only the generated tokens (n_in:)
    - 'prompt': all prompt tokens (:n_in)
    - 'all': all tokens in the run
    """
    T = acts.shape[0]
    n_in = max(1, min(int(n_in or T), T))
    s = (span or "reply").strip().lower()
    if s == "last":
        return acts[n_in - 1: n_in].float()
    if s == "reply":
        return acts[n_in - 1:].float()
    if s == "generated":
        return acts[n_in:].float() if n_in < T else acts[n_in - 1:].float()
    if s == "prompt":
        return acts[:n_in].float()
    return acts.float()


def compare_neurons_from_acts(adapter, schedule: list[int], acts_a: list[dict], acts_b: list[dict],
                              pos: int = -1, span: str = "reply", k: int = 15,
                              strict: bool = True, through: int = 0) -> dict:
    """Compare neuron activations between Set A runs (`acts_a`) and Set B runs (`acts_b`).
    Each item in `acts_a` / `acts_b` has `{"neurons": {pos: Tensor[T, d_ff]}, "n_in": int, "prompt": str, "text": str}`.
    If `pos >= 0`, ranks neurons in that schedule block; if `pos < 0`, scans all blocks and returns top per block + overall.
    """
    if not acts_a or not acts_b:
        raise ValueError("Provide at least one conversation/prompt in Set A and one in Set B.")
    S = len(schedule)
    blocks_to_scan = [pos] if 0 <= pos < S else list(range(S))

    def rank_block(p: int, top_n: int) -> dict | None:
        A_slices = [_span_slice(r["neurons"][p], r["n_in"], span) for r in acts_a if p in r["neurons"]]
        B_slices = [_span_slice(r["neurons"][p], r["n_in"], span) for r in acts_b if p in r["neurons"]]
        if not A_slices or not B_slices:
            return None
        a_cat = torch.cat(A_slices, dim=0)                              # [T_A, d_ff]
        b_cat = torch.cat(B_slices, dim=0)                              # [T_B, d_ff]
        mean_a = a_cat.mean(0)
        mean_b = b_cat.mean(0)
        std_a = a_cat.std(0, unbiased=False) if a_cat.shape[0] > 1 else torch.zeros_like(mean_a)
        std_b = b_cat.std(0, unbiased=False) if b_cat.shape[0] > 1 else torch.zeros_like(mean_b)
        sel = (mean_a - mean_b) / (std_a + std_b + 1e-3)                # selectivity (d'-like score)

        per_run_mean_a = torch.stack([x.mean(0) for x in A_slices])     # [N_A, d_ff]
        per_run_mean_b = torch.stack([x.mean(0) for x in B_slices])     # [N_B, d_ff]
        min_a = per_run_mean_a.min(0).values
        max_b = per_run_mean_b.max(0).values
        every = min_a > 0                                               # active (>0 mean) in EVERY Set A conversation
        q99_b = torch.quantile(b_cat, 0.99, dim=0) if b_cat.shape[0] > 1 else b_cat[0]
        quiet = q99_b < 0.5 * mean_a                                    # 99th percentile in B < 0.5 * mean in A

        if strict and bool((every & quiet).any()):
            score = torch.where(every & quiet, sel, torch.where(every, sel - 10.0, sel - 25.0))
        elif strict and bool(every.any()):
            score = torch.where(every, sel, sel - 15.0)
        else:
            score = sel

        n_take = min(top_n, int(score.shape[0]))
        top_idx = score.topk(n_take).indices.tolist()
        ref_h = None
        if int(through or 0) > 0:
            R_slices = [_span_slice(r["resid_post"][p], r["n_in"], span) for r in acts_a if p in (r.get("resid_post") or {})]
            if R_slices:
                ref_h = torch.cat(R_slices, dim=0).mean(0)
        words = neuron_words(adapter, schedule[p], top_idx, 6, through=through, schedule=schedule, pos=p, ref_resid=ref_h)
        rows = []
        for idx, w in zip(top_idx, words):
            rows.append({
                "pos": p,
                "layer": schedule[p],
                "index": int(idx),
                "selectivity": round(float(sel[idx]), 3),
                "diff": round(float(mean_a[idx] - mean_b[idx]), 4),
                "mean_a": round(float(mean_a[idx]), 4),
                "mean_b": round(float(mean_b[idx]), 4),
                "min_a": round(float(min_a[idx]), 4),
                "max_b": round(float(max_b[idx]), 4),
                "q99_b": round(float(q99_b[idx]), 4),
                "every_a": bool(every[idx]),
                "quiet_b": bool(quiet[idx]),
                "runs_a": [round(float(v), 3) for v in per_run_mean_a[:, idx]],
                "runs_b": [round(float(v), 3) for v in per_run_mean_b[:, idx]],
                **w,
            })
        return {
            "pos": p,
            "layer": schedule[p],
            "label": f"#{p} · L{schedule[p]}",
            "n_neurons": int(mean_a.shape[0]),
            "n_EveryQuiet": int((every & quiet).sum().item()),
            "neurons": rows,
        }

    if len(blocks_to_scan) == 1:
        b_res = rank_block(blocks_to_scan[0], k)
        if b_res is None:
            raise ValueError(f"Block #{blocks_to_scan[0]} has no captured MLP neurons.")
        return {
            "mode": "single_block",
            "pos": blocks_to_scan[0],
            "layer": schedule[blocks_to_scan[0]],
            "span": span,
            "n_a": len(acts_a),
            "n_b": len(acts_b),
            "runs_a": [{"prompt": r.get("prompt", ""), "text": r.get("text", ""), "run_id": r.get("run_id", "")} for r in acts_a],
            "runs_b": [{"prompt": r.get("prompt", ""), "text": r.get("text", ""), "run_id": r.get("run_id", "")} for r in acts_b],
            **b_res,
        }

    per_block = []
    all_candidates = []
    per_k = max(3, min(k, 8))
    for p in blocks_to_scan:
        b_res = rank_block(p, per_k)
        if b_res is not None:
            per_block.append(b_res)
            all_candidates.extend(b_res["neurons"])
    all_candidates.sort(key=lambda r: (r["every_a"] and r["quiet_b"], r["every_a"], r["selectivity"]), reverse=True)
    return {
        "mode": "all_blocks",
        "pos": -1,
        "span": span,
        "n_a": len(acts_a),
        "n_b": len(acts_b),
        "runs_a": [{"prompt": r.get("prompt", ""), "text": r.get("text", ""), "run_id": r.get("run_id", "")} for r in acts_a],
        "runs_b": [{"prompt": r.get("prompt", ""), "text": r.get("text", ""), "run_id": r.get("run_id", "")} for r in acts_b],
        "neurons": all_candidates[:k],
        "blocks": per_block,
    }


def _normalize_prompt_item(item, default_system: str = "", default_chat: bool = True, default_prefill: str = "") -> dict:
    """Turn a string or dict prompt specification into a normalized dict."""
    if isinstance(item, str):
        return {"prompt": item.strip(), "system": default_system, "chat": default_chat, "prefill": default_prefill, "history": []}
    if isinstance(item, dict):
        return {
            "prompt": str(item.get("prompt") or "").strip(),
            "system": str(item["system"]) if "system" in item and item["system"] is not None else default_system,
            "chat": bool(item["chat"]) if "chat" in item and item["chat"] is not None else default_chat,
            "prefill": str(item.get("prefill") or default_prefill),
            "history": list(item.get("history") or []),
        }
    raise ValueError("Each item in Set A / Set B must be a prompt string or conversation object.")


def iter_neuron_ab_hunt(adapter, base_plan, set_a: list, set_b: list, pos: int = -1,
                        span: str = "reply", max_new_tokens: int = 14, k: int = 15,
                        strict: bool = True, system: str | None = None, chat: bool | None = None,
                        prefill: str | None = None, record_run=None, through: int = 0):
    """Run all prompts in `set_a` and `set_b` (like an A/B test), capture MLP neurons across blocks,
    and identify the neurons that differ the most between Set A and Set B.
    Yields progress updates as each conversation finishes, then yields the final ranking."""
    import copy
    from .llm import Stepper

    def_sys = base_plan.params.get("system", "") if system is None else system
    def_chat = bool(base_plan.params.get("use_chat_template", True)) if chat is None else bool(chat)
    def_pre = base_plan.params.get("prefill", "") if prefill is None else prefill

    items_a = [_normalize_prompt_item(x, def_sys, def_chat, def_pre) for x in set_a if (x if isinstance(x, str) else x.get("prompt") or x.get("history"))]
    items_b = [_normalize_prompt_item(x, def_sys, def_chat, def_pre) for x in set_b if (x if isinstance(x, str) else x.get("prompt") or x.get("history"))]
    if not items_a or not items_b:
        raise ValueError("Provide at least one non-empty prompt in Set A and one in Set B.")

    total = len(items_a) + len(items_b)
    done = 0
    acts_a, acts_b = [], []

    def run_one(item: dict, group_label: str, idx: int) -> dict:
        p_plan = copy.deepcopy(base_plan)
        p_plan.interventions = []
        p_plan.params["code"] = {}
        p_plan.params.update(
            prompt=item["prompt"],
            system=item["system"],
            prefill=item["prefill"],
            use_chat_template=item["chat"],
            history=item["history"],
            max_new_tokens=max(0, min(128, int(max_new_tokens))),
            temperature=0.0,
            neuron_budget_mb=2048,
            attn_budget_mb=0,
            parts_budget_mb=0,
        )
        st = Stepper(adapter, p_plan)
        while st.step_token():
            pass
        res = st.finish()
        rid = ""
        if record_run is not None:
            rid = record_run(p_plan, res, f"{group_label}{idx + 1}") or ""
        return {
            "prompt": item["prompt"],
            "text": res["outputs"].get("text", ""),
            "n_in": res["captures"].get("n_in", 1),
            "neurons": res["captures"].get("neurons", {}),
            "resid_post": res["captures"].get("resid_post", {}),
            "run_id": rid,
        }

    yield {"status": "running", "_progress": 0.0, "n_a": len(items_a), "n_b": len(items_b),
           "done": 0, "total": total, "runs_a": [], "runs_b": []}

    for idx, it in enumerate(items_a):
        r = run_one(it, "A", idx)
        acts_a.append(r)
        done += 1
        yield {
            "status": "running",
            "_progress": round(done / max(1, total), 3),
            "done": done, "total": total,
            "runs_a": [{"prompt": x["prompt"], "text": x["text"], "run_id": x["run_id"]} for x in acts_a],
            "runs_b": [],
        }

    for idx, it in enumerate(items_b):
        r = run_one(it, "B", idx)
        acts_b.append(r)
        done += 1
        yield {
            "status": "running",
            "_progress": round(done / max(1, total), 3),
            "done": done, "total": total,
            "runs_a": [{"prompt": x["prompt"], "text": x["text"], "run_id": x["run_id"]} for x in acts_a],
            "runs_b": [{"prompt": x["prompt"], "text": x["text"], "run_id": x["run_id"]} for x in acts_b],
        }

    out = compare_neurons_from_acts(adapter, base_plan.schedule, acts_a, acts_b, pos=pos, span=span, k=k, strict=strict, through=through)
    out["through"] = through
    out["status"] = "done"
    out["_progress"] = 1.0
    yield out


def search_neurons_by_words(adapter, schedule: list[int], promote: list[str] | str,
                            suppress: list[str] | str | None = None,
                            pos: int = -1, k: int = 10, use_norm: bool = False,
                            through: int = 0, cap: dict | None = None) -> dict:
    """Rank MLP neurons by how strongly their `down_proj` output column pushes target `promote` words
    over `suppress` words through the unembedding matrix (writer neuron search).
    When `use_norm=False` (default, matching `neuron_wire.py`), computes `W_down.T @ (mean(U[promote]) - mean(U[suppress]))`;
    when `use_norm=True`, scales by `model.norm.weight` first.
    """
    if isinstance(promote, str):
        promote = [w for w in promote.replace(",", " ").split() if w.strip()]
    if isinstance(suppress, str):
        suppress = [w for w in suppress.replace(",", " ").split() if w.strip()]
    promote = [w for w in (promote or []) if str(w).strip()]
    suppress = [w for w in (suppress or []) if str(w).strip()]
    if not promote:
        raise ValueError("Provide at least one target word to promote (e.g. 'cannot, unable').")

    def resolve_ids(words: list[str]) -> tuple[list[int], list[str]]:
        tids, pieces = [], []
        for w in words:
            cands = adapter._word_ids(w)
            if cands:
                tids.append(int(cands[0][0]))
                pieces.append(cands[0][1])
        return tids, pieces

    p_ids, p_pieces = resolve_ids(promote)
    s_ids, s_pieces = resolve_ids(suppress)
    if not p_ids:
        raise ValueError(f"Could not tokenize promote words: {promote}")

    m = adapter.model
    U = m.unembed_weight()
    d = U[p_ids].float().cpu().mean(0)
    if s_ids:
        d = d - U[s_ids].float().cpu().mean(0)
    if use_norm:
        d = d * m.norm_w("model.norm.weight").float().cpu()
    d_dev = d.to(m.device)

    S = len(schedule)
    blocks = [pos] if 0 <= pos < S else list(range(S))

    def search_block(p: int, top_n: int) -> dict | None:
        layer = schedule[p]
        if layer in getattr(m, "moe_layers", {}):
            return None
        wd = m.W(f"model.layers.{layer}.mlp.down_proj.weight").float()     # [d_model, d_ff]
        push = (wd.T @ d_dev).cpu()                                         # [d_ff]
        n_take = min(top_n, int(push.shape[0]))
        top = push.topk(n_take)
        idx_list = top.indices.tolist()
        words = neuron_words(adapter, layer, idx_list, 6, through=through, schedule=schedule, cap=cap, pos=p)
        rows = []
        for idx, val, w in zip(idx_list, top.values.tolist(), words):
            rows.append({
                "pos": p,
                "layer": layer,
                "index": int(idx),
                "push": round(float(val), 4),
                **w,
            })
        return {
            "pos": p,
            "layer": layer,
            "label": f"#{p} · L{layer}",
            "n_neurons": int(push.shape[0]),
            "neurons": rows,
        }

    if len(blocks) == 1:
        res = search_block(blocks[0], k)
        if res is None:
            raise ValueError(f"Block #{blocks[0]} is an MoE block (no single MLP down_proj).")
        return {
            "mode": "single_block",
            "pos": blocks[0],
            "layer": schedule[blocks[0]],
            "promote_tokens": p_pieces,
            "suppress_tokens": s_pieces,
            **res,
        }

    per_block = []
    all_rows = []
    per_k = max(3, min(k, 6))
    for p in blocks:
        b_res = search_block(p, per_k)
        if b_res is not None:
            per_block.append(b_res)
            all_rows.extend(b_res["neurons"])
    all_rows.sort(key=lambda r: r["push"], reverse=True)
    return {
        "mode": "all_blocks",
        "pos": -1,
        "promote_tokens": p_pieces,
        "suppress_tokens": s_pieces,
        "neurons": all_rows[:k],
        "blocks": per_block,
    }


def iter_neuron_batch_test(adapter, edited_plan, test_prompts: list, control_prompts: list | None = None,
                           max_new_tokens: int = 40, keyword_regex: str = r"(can't|cannot|can not|unable|not able)",
                           system: str | None = None, chat: bool | None = None, prefill: str | None = None):
    """Run `test_prompts` and `control_prompts` before (baseline: no interventions/edited code) and after
    (with `edited_plan`'s wire/clamp interventions and code edits), comparing outputs side by side."""
    import copy
    import re as _re
    from .llm import Stepper

    def_sys = edited_plan.params.get("system", "") if system is None else system
    def_chat = bool(edited_plan.params.get("use_chat_template", True)) if chat is None else bool(chat)
    def_pre = edited_plan.params.get("prefill", "") if prefill is None else prefill

    tests = [_normalize_prompt_item(x, def_sys, def_chat, def_pre) for x in (test_prompts or []) if (x if isinstance(x, str) else x.get("prompt") or x.get("history"))]
    ctrls = [_normalize_prompt_item(x, def_sys, def_chat, def_pre) for x in (control_prompts or []) if (x if isinstance(x, str) else x.get("prompt") or x.get("history"))]
    if not tests and not ctrls:
        raise ValueError("Provide at least one test or control prompt.")

    pat = _re.compile(keyword_regex, _re.I) if keyword_regex else None
    total = 2 * (len(tests) + len(ctrls))
    done = 0

    def gen_one(item: dict, with_edits: bool) -> str:
        p = copy.deepcopy(edited_plan)
        if not with_edits:
            p.interventions = []
            p.params["code"] = {}
            p.params.pop("kv_overrides", None)
        p.params.update(
            prompt=item["prompt"],
            system=item["system"],
            prefill=item["prefill"],
            use_chat_template=item["chat"],
            history=item["history"],
            max_new_tokens=max(1, min(128, int(max_new_tokens))),
            temperature=0.0,
            neuron_budget_mb=0,
            attn_budget_mb=0,
            parts_budget_mb=0,
        )
        st = Stepper(adapter, p)
        st.ctx.sources = getattr(edited_plan, "sources", None) or {}
        while st.step_token():
            pass
        return st.finish()["outputs"].get("text", "")

    test_rows, ctrl_rows = [], []
    base_hits = edit_hits = ctrl_same = 0

    def snapshot(status="running"):
        return {
            "status": status,
            "_progress": round(done / max(1, total), 3),
            "done": done,
            "total": total,
            "keyword_regex": keyword_regex,
            "test_total": len(tests),
            "baseline_matches": base_hits,
            "edited_matches": edit_hits,
            "control_total": len(ctrls),
            "control_identical": ctrl_same,
            "test_rows": list(test_rows),
            "control_rows": list(ctrl_rows),
        }

    yield snapshot("running")

    for it in tests:
        b_txt = gen_one(it, False)
        done += 1
        e_txt = gen_one(it, True)
        done += 1
        b_match = bool(pat.search(b_txt)) if pat else False
        e_match = bool(pat.search(e_txt)) if pat else False
        base_hits += int(b_match)
        edit_hits += int(e_match)
        test_rows.append({
            "prompt": it["prompt"],
            "baseline": b_txt,
            "edited": e_txt,
            "changed": b_txt != e_txt,
            "baseline_match": b_match,
            "edited_match": e_match,
        })
        yield snapshot("running")

    for it in ctrls:
        b_txt = gen_one(it, False)
        done += 1
        e_txt = gen_one(it, True)
        done += 1
        same = (b_txt == e_txt)
        ctrl_same += int(same)
        ctrl_rows.append({
            "prompt": it["prompt"],
            "baseline": b_txt,
            "edited": e_txt,
            "identical": same,
        })
        yield snapshot("running")

    yield snapshot("done")


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
# Tokenizer explorer & Word -> Token Dictionary
# ======================================================================================

def _build_word_map(ids: list[int], pieces: list[str]) -> list[dict]:
    """Group a sequence of token pieces and IDs into a human-readable Word -> Tokens dictionary."""
    import re as _re
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
        # Merge into the previous word entry if this piece has no leading space and continues a word
        if (
            groups
            and not starts_with_space
            and k == "word"
            and groups[-1]["kind"] == "word"
        ):
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
    """Analyze a run's KV cache across blocks and token positions: memory breakdown, GQA compression,
    per-block and per-token Key/Value L2 norms, and attention-sink identification."""
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
        # k_t, v_t: [n_kv, T_b, hd] -> L2 norm over head_dim, averaged across KV heads -> [T_b]
        kn = k_t.float().norm(dim=-1).mean(0).cpu()[:T_b]
        vn = v_t.float().norm(dim=-1).mean(0).cpu()[:T_b]
        if bc is None:
            tok_k_norms[:min(T, len(kn))] += kn[:T]
            tok_v_norms[:min(T, len(vn))] += vn[:T]
            n_full_blocks += 1

        sink_tok, sink_piece, sink_attn = None, None, None
        a = attn_map.get(p)
        if a is not None and a.numel() > 0:
            # a: [n_heads, n_q, n_k] -> mean attention received by each key token across heads & queries
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
    # Interceptors already in the flow (a banned expert, a steering vector …) stay on for every run of the
    # sweep: the question asked is "what does this patch change in the model as I have edited it?"
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
