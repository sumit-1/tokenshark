"""Core lazy analysis methods for Causal LLMs: decode_layer, logit_lens, word_cloud, track_word, attention (`tokenshark.llm.analysis`)."""
from __future__ import annotations

import torch

from ..engine import RunContext
from ..flow import parse_index_spec
from ..kvcache import KVCache


class LLMAnalysisMixin:
    @torch.no_grad()
    def decode_layer(self, cap: dict, schedule: list[int], p: int, through: int = 0, w_out=None) -> torch.Tensor:
        """Turn the residual stream after schedule position p (p = -1: the embeddings) into next-token logits."""
        h = cap["resid_pre"][0] if p < 0 else cap["resid_post"][p]
        T = min(len(cap["ids"]), h.shape[0])
        x = h[:T].to(self.device, self.model.dtype)
        S = len(schedule)
        start = max(p + 1, S - through) if through > 0 else S
        if start < S:
            full: KVCache = cap["cache"]
            n_pre = cap["n_prefix"]
            tmp = full.prefix_view(n_pre)
            tmp.add_tokens(cap["ids"][:T], full.pos[n_pre:n_pre + T].tolist(), "prompt")
            ctx = RunContext(interventions=list(cap.get("interventions") or []))
            ctx.sources = cap.get("sources") or {}
            ctx.token_ranges = rng = cap.get("token_ranges") or {}
            ctx.store["code"] = cap.get("code", {})
            ctx.store["unit_range"] = dict(cap.get("unit_range") or {})
            ctx.store["moe_k"], ctx.store["moe_temp"] = cap.get("moe_k") or {}, cap.get("moe_temp") or {}
            ctx.store["moe_noshared"] = cap.get("moe_noshared") or ()
            ctx.store["rope"] = dict(cap.get("rope") or {})
            planned = max([T] + [r.stop for r in rng.values()])

            def idx_of(iv):
                if iv.positions.strip().lower() == "last":
                    ends = [r.stop - 1 for k, r in rng.items() if k in ("prompt", "prefill") and len(r)]
                    return sorted({i for i in ends + list(rng.get("generated", [])) if i < T}) or [T - 1]
                return [g for g in parse_index_spec(iv.positions, planned, rng) if g < T]

            for q in range(start, S):
                if q in tmp.block_caches:
                    n_prompt_main = rng.get("prompt", range(0, T)).stop
                    bc = tmp.block_caches[q]
                    if T <= n_prompt_main:
                        ctx.store["chunk_origin"] = "prompt"
                        x = self.model.block(q, schedule[q], x, tmp, ctx, idx_of)
                    else:
                        ctx.store["chunk_origin"] = "prompt"
                        x_pre = self.model.block(q, schedule[q], x[:n_prompt_main], tmp, ctx,
                                                 lambda iv: [i for i in idx_of(iv) if i < n_prompt_main])
                        tail_len = T - n_prompt_main
                        base_tail_pos = bc.next_position()
                        tail_pos = [base_tail_pos + i for i in range(tail_len)]
                        bc.add_tokens(cap["ids"][n_prompt_main:T], tail_pos, "generated")
                        ctx.store["chunk_origin"] = "generated"
                        x_tail = self.model.block(q, schedule[q], x[n_prompt_main:], tmp, ctx,
                                                  lambda iv: [i - n_prompt_main for i in idx_of(iv) if i >= n_prompt_main])
                        x = torch.cat([x_pre, x_tail], dim=0)
                    ctx.store.pop("chunk_origin", None)
                else:
                    x = self.model.block(q, schedule[q], x, tmp, ctx, idx_of)
        return self.model.final(x, w_out if w_out is not None else self.model.unembed_weight())

    @torch.no_grad()
    def logit_lens(self, cap: dict, schedule_nodes: list[str], schedule: list[int], k: int = 5, through: int = 0) -> dict:
        out = None
        for out in self.iter_logit_lens(cap, schedule_nodes, schedule, k, through):
            pass
        out.pop("_progress", None)
        return out

    @torch.no_grad()
    def iter_logit_lens(self, cap: dict, schedule_nodes: list[str], schedule: list[int], k: int = 5, through: int = 0,
                        memo: dict | None = None):
        memo = {} if memo is None else memo
        ids = cap["ids"]
        T = len(ids)
        w_out = self.model.unembed_weight()
        rows_src = []
        if 0 in cap["resid_pre"]:
            rows_src.append(("embed", "Embed", -1, min(T, cap["resid_pre"][0].shape[0])))
        for pp in range(len(schedule)):
            if pp in cap["resid_post"]:
                rows_src.append((schedule_nodes[pp], f"#{pp} · L{schedule[pp]}", pp, min(T, cap["resid_post"][pp].shape[0])))
        R = len(rows_src)
        res = {"tokens": [self.tok.piece(i) for i in ids], "layers": [{"node": n, "label": l} for n, l, _, _ in rows_src],
               "cells": [None] * R, "through": through, "complete": len(cap["final_top"]), "_progress": 0.0}
        yield dict(res)
        ref_row = [next((r for r in range(R - 1, -1, -1) if rows_src[r][3] > t), None) for t in range(T)]
        ref_logp: dict[int, torch.Tensor] = {}
        final_top = list(cap["final_top"])

        def logp_of(r):
            if r in ref_logp:
                return ref_logp[r]
            return torch.log_softmax(self.decode_layer(cap, schedule, rows_src[r][2], through, w_out), -1)

        def refs():
            if not ref_logp:
                for r in sorted({x for x in ref_row if x is not None}):
                    ref_logp[r] = logp_of(r)
                for t in range(len(final_top), T):
                    final_top.append(int(ref_logp[ref_row[t]][t].argmax()) if ref_row[t] is not None else 0)

        order = sorted({x for x in ref_row if x is not None}) + [r for r in range(R) if r not in set(ref_row)]
        for n_done, r in enumerate(order, 1):
            key = ("lensrow", k, through, rows_src[r][2])
            if key not in memo:
                refs()
                logp = logp_of(r)
                n = logp.shape[0]
                probs = logp.exp()
                pv, pi = probs.topk(k, -1)
                row = []
                for t in range(T):
                    if t >= n:
                        row.append(None)
                        continue
                    tgt = final_top[t]
                    ref = ref_logp[ref_row[t]][t]
                    c = {"top": [[self.tok.piece(int(pi[t, j])), round(float(pv[t, j]), 4)] for j in range(k)],
                         "p_final": round(float(probs[t, tgt]), 4),
                         "rank_final": int((logp[t] > logp[t, tgt]).sum()),
                         "kl": round(float((ref.exp() * (ref - logp[t])).sum()), 3)}
                    if t + 1 < T:
                        c["p_next"] = round(float(probs[t, ids[t + 1]]), 4)
                    row.append(c)
                memo[key] = row
            res["cells"][r] = memo[key]
            res["_progress"] = n_done / max(1, R)
            yield {**res, "cells": list(res["cells"])}

    def _word_ids(self, word: str) -> list[tuple[int, str, int]]:
        out = []
        for form in (" " + word.strip(), word.strip()):
            ids = self.tok.encode(form, add_special=False)
            if ids and ids[0] not in [c[0] for c in out]:
                out.append((ids[0], self.tok.piece(ids[0]), len(ids)))
        out.sort(key=lambda c: c[2] > 1)
        return out

    @torch.no_grad()
    def word_cloud(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int,
                   layers: list[int], through: int = 0, top: int = 40) -> dict:
        out = None
        for out in self.iter_word_cloud(cap, schedule_nodes, schedule, position, layers, through, top):
            pass
        out.pop("_progress", None)
        out.pop("todo", None)
        return out

    @torch.no_grad()
    def iter_word_cloud(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int,
                        layers: list[int], through: int = 0, top: int = 40, memo: dict | None = None):
        memo = {} if memo is None else memo
        T = len(cap["ids"])
        t = min(max(position if position >= 0 else T + position, 0), T - 1)
        w_out = self.model.unembed_weight()
        final_choice = cap["final_top"][t] if t < len(cap["final_top"]) else None
        actual_next = cap["ids"][t + 1] if t + 1 < T else None
        n_in = cap.get("n_in")
        lab = lambda pp: "Embed" if pp < 0 else f"#{pp} · L{schedule[pp]}"                  # noqa: E731
        layers = [pp for pp in layers if pp < 0 or pp in cap["resid_post"]]
        res = {"position": t, "token": self.tok.piece(cap["ids"][t]),
               "next_token": self.tok.piece(actual_next) if actual_next is not None else None,
               "next_generated": bool(n_in is not None and t + 1 >= n_in),
               "model_choice": self.tok.piece(final_choice) if final_choice is not None else None,
               "layers": [], "combined": None, "through": through, "pending": [],
               "todo": [{"pos": pp, "label": lab(pp), "node": "embed" if pp < 0 else schedule_nodes[pp]} for pp in layers],
               "_progress": 0.0}
        yield {**res, "layers": [], "todo": list(res["todo"])}
        acc, n_acc = None, 0
        for n_done, pp in enumerate(layers, 1):
            key = ("cloudlayer", t, pp, through)
            if key not in memo:
                logits = self.decode_layer(cap, schedule, pp, through, w_out)
                if t >= logits.shape[0]:
                    memo[key] = {"pending": lab(pp)}
                else:
                    lg = logits[t]
                    probs = torch.softmax(lg, -1)
                    v, i = probs.topk(100)
                    rank = lambda tid: int((lg > lg[tid]).sum()) + 1                        # noqa: E731
                    memo[key] = {"probs": probs.half(), "layer": {
                        "pos": pp, "node": "embed" if pp < 0 else schedule_nodes[pp], "label": lab(pp),
                        "words": [[self.tok.piece(int(j)), round(float(x), 5)] for x, j in zip(v, i)],
                        "choice_rank": rank(final_choice) if final_choice is not None else None,
                        "choice_p": round(float(probs[final_choice]), 5) if final_choice is not None else None,
                        "next_rank": rank(actual_next) if actual_next is not None else None,
                        "next_p": round(float(probs[actual_next]), 5) if actual_next is not None else None}}
            m = memo[key]
            if "pending" in m:
                res["pending"].append(m["pending"])
            else:
                res["layers"].append({**m["layer"], "words": m["layer"]["words"][:top]})
                p32 = m["probs"].float()
                acc = p32 if acc is None else acc + p32
                n_acc += 1
            res["todo"] = res["todo"][1:]
            res["_progress"] = n_done / max(1, len(layers))
            if n_done == len(layers) and acc is not None and n_acc > 1:
                v, i = (acc / n_acc).topk(top)
                res["combined"] = [[self.tok.piece(int(j)), round(float(x), 5)] for x, j in zip(v, i)]
            yield {**res, "layers": list(res["layers"]), "pending": list(res["pending"]), "todo": list(res["todo"])}

    @torch.no_grad()
    def track_word(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int, word: str,
                   through: int = 0) -> dict:
        out = None
        for out in self.iter_track_word(cap, schedule_nodes, schedule, position, word, through):
            pass
        out.pop("_progress", None)
        return out

    @torch.no_grad()
    def iter_track_word(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int, word: str,
                        through: int = 0, memo: dict | None = None):
        T = len(cap["ids"])
        t = min(max(position if position >= 0 else T + position, 0), T - 1)
        cands = self._word_ids(word)
        if not cands:
            raise ValueError("Type a word to track.")
        w_out = self.model.unembed_weight()
        S = len(schedule)
        final = self.decode_layer(cap, schedule, S - 1, 0, w_out)[t] if (S - 1) in cap["resid_post"] else None
        target, target_piece, n_pieces = cands[0]
        choice = cap["final_top"][t] if t < len(cap["final_top"]) else (int(final.argmax()) if final is not None else target)
        versus = choice
        if versus == target and final is not None:
            versus = int(final.topk(2).indices[1])
        g = self.model.norm_w("model.norm.weight").float().cpu()
        U = w_out[[target, versus]].float().cpu()
        x_final = cap["resid_post"][S - 1][t].float() if (S - 1) in cap["resid_post"] else None
        rms = float(torch.sqrt(x_final.pow(2).mean() + self.cfg.eps)) if x_final is not None else 1.0
        dirv = (U[0] - U[1]) * g / rms
        rows = []
        layers = [-1] + [p for p in range(S) if p in cap["resid_post"]]
        first_top = None
        base = {"position": t, "token": self.tok.piece(cap["ids"][t]), "target": target_piece,
                "candidates": [c[1] for c in cands], "pieces": n_pieces, "word": word.strip(),
                "choice": self.tok.piece(choice), "same": target == choice,
                "versus": self.tok.piece(versus), "through": through}
        for n_done, pp in enumerate(layers, 1):
            lg_all = self.decode_layer(cap, schedule, pp, through, w_out)
            if t >= lg_all.shape[0]:
                continue
            lg = lg_all[t]
            probs = torch.softmax(lg, -1)
            r_t = int((lg > lg[target]).sum()) + 1
            r_c = int((lg > lg[choice]).sum()) + 1
            if pp < 0:
                write = cap["resid_pre"][0][t].float()
            else:
                write = cap["resid_post"][pp][t].float() - cap["resid_pre"][pp][t].float()
            push = float(write @ dirv)
            if r_t == 1 and first_top is None:
                first_top = pp
            rows.append({"pos": pp, "node": "embed" if pp < 0 else schedule_nodes[pp],
                         "label": "Embed" if pp < 0 else f"#{pp} · L{schedule[pp]}",
                         "top": self.tok.piece(int(lg.argmax())),
                         "target_p": round(float(probs[target]), 5), "target_rank": r_t,
                         "choice_p": round(float(probs[choice]), 5), "choice_rank": r_c, "push": round(push, 3)})
            yield {**base, "first_top_layer": first_top, "rows": list(rows), "_progress": n_done / max(1, len(layers))}

    def attention(self, cap: dict, pos: int, head: int | None) -> dict:
        a = cap["attn"].get(pos)
        if a is None and pos in cap.get("linear_slots", []):
            return {"error": "This is a linear-attention (Gated DeltaNet) block. It has no attention map: instead of "
                             "looking back at every earlier token it keeps one running memory per head, updated token "
                             "by token. Pick a full-attention block (every 4th block in Qwen3.5/3.6)."}
        if a is None:
            return {"error": "No attention captured for this block (sequence too long for the capture budget)."}
        m = a.float().mean(0) if head is None or head < 0 else a[head].float()
        cache: KVCache = cap["cache"]
        bc = cache.block_caches.get(pos)
        r_pos = (cap.get("rope_pos") or {}).get(pos)
        if bc is not None:
            k_pos = r_pos[1][: m.shape[1]].tolist() if r_pos is not None and len(r_pos[1]) >= m.shape[1] else bc.pos[: m.shape[1]].tolist()
            return {"n_heads": a.shape[0], "n_prefix": 0,
                    "kv_override": {"op": bc.op, "label": bc.label, "n_prompt": bc.n_prompt, "text": bc.text},
                    "matrix": [[round(float(v), 4) for v in row] for row in m],
                    "query_tokens": [self.tok.piece(i) for i in cap["ids"][: m.shape[0]]],
                    "key_tokens": [self.tok.piece(i) for i in bc.tokens[: m.shape[1]]],
                    "key_positions": [round(float(x), 2) for x in k_pos],
                    "rope": (cap.get("rope") or {}).get(pos)}
        k_pos = r_pos[1][: m.shape[1]].tolist() if r_pos is not None and len(r_pos[1]) >= m.shape[1] else cache.pos[: m.shape[1]].tolist()
        return {"n_heads": a.shape[0], "n_prefix": cap["n_prefix"],
                "matrix": [[round(float(v), 4) for v in row] for row in m],
                "query_tokens": [self.tok.piece(i) for i in cap["ids"][: m.shape[0]]],
                "key_tokens": [self.tok.piece(i) for i in cache.tokens[: m.shape[1]]],
                "key_positions": [round(float(x), 2) for x in k_pos],
                "rope": (cap.get("rope") or {}).get(pos)}
