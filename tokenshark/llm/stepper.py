"""Stepper: the unified block-by-block and token-by-token run loop (`tokenshark.llm.stepper`)."""
from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from ..engine import Cancelled, RunContext, host_sync, set_determinism
from ..flow import Plan, parse_index_spec
from ..kvcache import KVCache

if TYPE_CHECKING:
    from .adapter import LLMAdapter



class Stepper:
    """Executes a plan in the smallest steps the model has:

        embed(chunk) → block #0 → block #1 → … → final norm + unembed → choose next token → embed(token) → …

    `step_block()` advances one of those arrows. `step_token()` runs to the next token prediction.
    `force(token_id)` decides the next token yourself.
    """

    def __init__(self, ad: "LLMAdapter", plan: Plan, prefix: KVCache | None = None, progress=None, ctx_hook=None,
                 trace: bool = False):
        self.ad, self.model, self.plan = ad, ad.model, plan
        p = self.p_ = plan.params
        set_determinism(p["seed"])
        if hasattr(ad, "build_ids_with_roles"):
            prompt_ids, prefill_ids, self.roles = ad.build_ids_with_roles(p)
        else:                                    # adapters with their own build_ids: no role spans
            (prompt_ids, prefill_ids), self.roles = ad.build_ids(p), []
        if not prompt_ids and not prefill_ids and prefix is None:
            prompt_ids = [ad.tok.bos_id or 0]
        if not prompt_ids and not prefill_ids:
            raise ValueError("Nothing to run: empty prompt.")
        self.prompt_ids, self.prefill_ids = prompt_ids, prefill_ids
        self.media = ad.media_features(p)        # soft tokens for the prompt's media placeholders (or None)
        self.n_in = len(prompt_ids) + len(prefill_ids)
        self.total = p["max_new_tokens"]
        planned = self.planned = self.n_in + self.total
        ctx = self.ctx = RunContext(interventions=plan.interventions)
        if ctx_hook:
            ctx_hook(ctx)
        ctx.token_ranges = {"prompt": range(0, len(prompt_ids)), "prefill": range(len(prompt_ids), self.n_in),
                            "generated": range(self.n_in, planned)}
        ctx.capture = True
        # Side-branch fork specs grouped by schedule position
        forks_by_pos: dict[int, list] = {}
        for fspec in getattr(plan, "forks", None) or []:
            forks_by_pos.setdefault(fspec.pos, []).append(fspec)
        ctx.store["forks"] = forks_by_pos

        prefixes = prefix if isinstance(prefix, list) else ([prefix] if prefix is not None else [])
        specs = list(p.get("prefixes") or [])
        if len(specs) < len(prefixes):
            specs += [{"mode": p.get("prefix_mode", "shift"), "offset": p.get("prefix_offset", 0),
                       "layers": p.get("prefix_layers", "all"), "label": p.get("prefix_label", "imported")}] * (len(prefixes) - len(specs))
        S_ = len(plan.schedule)
        self._attn_capture_setup(planned + sum(len(c) for c in prefixes))
        ctx.store["want_neurons"] = planned * ad.cfg.inter * S_ * 2 < int(p.get("neuron_budget_mb", 512)) * 2**20
        ctx.store["want_parts"] = planned * ad.cfg.hidden * S_ * 8 < int(p.get("parts_budget_mb", 768)) * 2**20
        if ad.cfg.moe:
            n_moe = sum(1 for l in plan.schedule if l in self.model.moe_layers)
            ctx.store["want_experts"] = (planned * ad.cfg.moe["k"] * ad.cfg.hidden * n_moe * 2
                                         < int(p.get("expert_budget_mb", 768)) * 2**20)
            per = p.get("moe") or {}
            for key in ("k", "temp"):
                ctx.store["moe_" + key] = {pos: per[nid][key] for pos, nid in enumerate(plan.schedule_nodes)
                                           if nid in per and per[nid].get(key)}
            ctx.store["moe_noshared"] = {pos for pos, nid in enumerate(plan.schedule_nodes)
                                         if per.get(nid, {}).get("shared") is False}
        from ..blockcode import compile_code
        self.code = {nid: compile_code(c["source"], c["kind"], nid) for nid, c in (p.get("code") or {}).items()}
        ctx.store["code"] = {pos: self.code[nid] for pos, nid in enumerate(plan.schedule_nodes) if nid in self.code}
        rope_per = p.get("rope") or {}
        ctx.store["rope"] = {pos: dict(rope_per[nid]) for pos, nid in enumerate(plan.schedule_nodes) if nid in rope_per}
        zero_per = p.get("kv_zero") or {}
        ctx.store["kv_zero"] = {pos: list(zero_per[nid]) for pos, nid in enumerate(plan.schedule_nodes) if nid in zero_per}
        self.box_code = {c.kind: c for c in self.code.values() if c.kind != "block"}
        cache, start_pos, spans = ad.new_cache(S_), 0.0, []
        for c, sp in zip(prefixes, specs):
            pre, st = c.positioned_for_prepend(sp.get("mode", "shift"), float(sp.get("offset", 0) or 0))
            pre = pre.relabel("imported", sp.get("label") or "imported")
            cache.check_compatible(pre)
            a = len(cache)
            cache = KVCache.concat(cache, pre) if len(pre) else cache
            lay = str(sp.get("layers") or "all").strip().lower()
            slots = None if lay in ("all", "*", "") else (set() if lay == "none" else set(parse_index_spec(lay, S_)))
            spans.append((a, len(cache), slots))
            start_pos = max(start_pos, st)
        cache.prefix_spans = spans
        cache.shared = ad.cfg.kv_sharing(plan.schedule)
        cache.states, self.notes = {}, []
        for c, (a, b, slots) in zip(prefixes, spans):
            for slot, stt in c.states.items():
                if slots is None or slot in slots:
                    if slot in cache.states:
                        self.notes.append(f"linear block #{slot} sees several prefixes; only the last one's memory is used")
                    cache.states[slot] = {k: v.detach().clone() for k, v in stt.items()}
        cache.init_states = KVCache._clone_states(cache.states)
        self.prompt_scale = float(p.get("rope_prompt_scale", 1.0))
        self.step_stride = float(p.get("rope_step_stride", 1.0))
        self.gen_start = p.get("rope_gen_start")          # RoPE jump: first generated token sits at this position
        self.jumped = False
        if p.get("kv_overrides"):
            cache.block_caches = ad.recreate_block_caches(plan, start_pos=start_pos, prompt_scale=self.prompt_scale)
        max_bc_len = max([len(bc) for bc in cache.block_caches.values()], default=0)
        self._attn_capture_setup(max(planned + sum(len(c) for c in prefixes), max_bc_len + len(prefill_ids) + self.total),
                                 final=True)
        self.cache, self.n_prefix = cache, len(cache)
        self.gen = torch.Generator(device="cpu").manual_seed(p["seed"])
        self.eos = set(ad.cfg.eos_ids) | ({ad.tok.eos_id} if ad.tok.eos_id is not None else set())
        self.w_out = self.model.unembed_weight()
        self.queue = [(ids, o) for ids, o in ((prompt_ids, "prompt"), (prefill_ids, "prefill")) if ids]
        self.seq: list[int] = []
        self.logits_rows: list[torch.Tensor] = []
        self.offset, self.pos_next = 0, start_pos
        self.last_pos = None
        self.x, self.bp, self.chunk = None, 0, None
        self.n_generated, self.done, self.forced = 0, False, None
        self.eos_stop = False
        self.trace: list[dict] = []
        self.progress = progress
        self.steps = 0
        self.trace_on = trace
        self.last_lens: dict[int, str] = {}

    def _attn_capture_setup(self, n_keys: int, final: bool = False) -> None:
        """Decide what attention this run records, from the plan's capture settings:
          attn_budget_mb  upper bound for the stored weights (0 = record none)
          attn_layers     schedule positions to record ("all", "5", "3,7", "10:14")
          attn_queries    "all" (full [heads, T, T]) or "last" (one row per forward pass: the token being predicted)
        Only blocks that compute softmax attention count (linear-attention blocks store nothing). A run that would
        go over the budget records no attention and says so in its notes, with the numbers."""
        ad, p, plan, st = self.ad, self.p_, self.plan, self.ctx.store
        S_ = len(plan.schedule)
        lay = str(p.get("attn_layers") or "all").strip().lower()
        slots = None if lay in ("all", "*", "") else set(parse_index_spec(lay, S_))
        attn_blocks = [pp for pp, l in enumerate(plan.schedule)
                       if not ad.cfg.is_linear(l) and (slots is None or pp in slots)]
        last = str(p.get("attn_queries") or "all").strip().lower() == "last"
        rows = (len(self._prompt_chunks()) + self.total) if last else self.planned
        est = ad.cfg.n_heads * rows * n_keys * len(attn_blocks) * 2
        budget = int(p.get("attn_budget_mb", 768))
        st["attn_slots"], st["attn_last"] = slots, last
        st["want_attn"] = est < budget * 2**20
        if final and not st["want_attn"] and budget > 0 and attn_blocks:
            self.notes.append(
                f"attention not captured: it needs about {est / 2**20:,.0f} MB for {len(attn_blocks)} blocks but "
                f"attn_budget_mb is {budget}. Raise attn_budget_mb, or record less with attn_layers (e.g. \"5,11\") "
                f"or attn_queries=\"last\".")

    def _prompt_chunks(self) -> list:
        """The chunks the prompt is read in (one forward pass each): prompt, then prefill."""
        return [c for c in (self.prompt_ids, self.prefill_ids) if c]

    def update_rope(self, params: dict) -> list[str]:
        """Take new RoPE / zeroed-record settings from a recompiled flow, mid-session. Keys sit in the cache
        before RoPE, so new speeds and gears apply to every cached key from the next step on. Zeroing is
        permanent for this session (the record is erased); restoring one needs a fresh run."""
        p, plan, ctx, changed = params, self.plan, self.ctx, []
        rope_per = p.get("rope") or {}
        rope = {pos: dict(rope_per[nid]) for pos, nid in enumerate(plan.schedule_nodes) if nid in rope_per}
        if rope != (ctx.store.get("rope") or {}):
            ctx.store["rope"] = rope
            changed.append("block RoPE")
        zero_per = p.get("kv_zero") or {}
        zero = {pos: list(zero_per[nid]) for pos, nid in enumerate(plan.schedule_nodes) if nid in zero_per}
        if zero != (ctx.store.get("kv_zero") or {}):
            ctx.store["kv_zero"] = zero
            changed.append("zeroed records")
        stride, gs = float(p.get("rope_step_stride", 1.0)), p.get("rope_gen_start")
        if stride != self.step_stride:
            self.step_stride = stride
            changed.append("generation speed")
        if gs != self.gen_start:                  # a new jump target: the next generated token lands there
            self.gen_start, self.jumped = gs, False
            changed.append("jump")
        if self.n_generated == 0 and float(p.get("rope_prompt_scale", 1.0)) != self.prompt_scale:
            changed.append("prompt speed (only before the prompt is read)")
            self.prompt_scale = float(p.get("rope_prompt_scale", 1.0))
        return changed

    # ---------------------------------------------------------------- helpers
    def _idx(self, iv, offset, n):
        if iv.positions.strip().lower() == "last":
            return [n - 1]
        glob = parse_index_spec(iv.positions, self.planned, self.ctx.token_ranges)
        return [g - offset for g in glob if offset <= g < offset + n]

    def _lens_top(self, x, k=5):
        lg = self.model.final(x[-1:], self.w_out)[0]
        pv, pi = torch.softmax(lg, -1).topk(k)
        return [[self.ad.tok.piece(int(i)), int(i), round(float(v), 4)] for v, i in zip(pv, pi)]

    def _next_chunk(self):
        if self.queue:
            return self.queue.pop(0)
        if self.done or self.n_generated >= self.total or not self.logits_rows:
            self.done = True
            return None
        if self.forced is not None:
            nxt = self.forced
        elif "sampler" in self.box_code:
            from ..blockcode import run_sampler
            nxt = run_sampler(self.box_code["sampler"], self.logits_rows[-1][-1], self.p_, self.gen, self.ctx)
        else:
            nxt = self.ad._sample(self.logits_rows[-1][-1], self.p_, self.gen)
        forced, self.forced = self.forced is not None, None
        if nxt in self.eos and not self.p_.get("ignore_eos") and not forced:
            self.seq.append(nxt)
            self.done = self.eos_stop = True
            return None
        self.n_generated += 1
        return ([nxt], "generated")

    def _report(self, pos):
        if self.progress:
            nb = len(self.plan.schedule)
            self.progress((self.n_generated + (pos + 1) / nb) / (self.total + 1),
                          f"{'prefill' if self.n_generated == 0 else f'token {self.n_generated}/{self.total}'} · block {pos + 1}/{nb}")

    # ---------------------------------------------------------------- stepping
    @torch.no_grad()
    def step_block(self) -> bool:
        """Advance one arrow in the flow. Returns False when the run is finished."""
        if self.ctx.cancel:
            raise Cancelled("Run cancelled.")
        if self.done:
            return False
        self.steps += 1
        if self.x is None:                       # idle: pick the next chunk and embed it
            ch = self._next_chunk()
            if ch is None:
                return False
            ids, origin = ch
            jump = 0.0
            if origin == "generated":
                base_p = (self.last_pos + self.step_stride) if self.last_pos is not None else self.pos_next
                if self.gen_start is not None and not self.jumped:
                    jump, base_p = float(self.gen_start) - base_p, float(self.gen_start)
                self.jumped = True
                positions = [base_p + i * self.step_stride for i in range(len(ids))]
            else:
                base_p = (self.last_pos + self.prompt_scale) if self.last_pos is not None else self.pos_next
                positions = [base_p + i * self.prompt_scale for i in range(len(ids))]
            self._chunk_positions = positions
            self.cache.add_tokens(ids, positions, origin)
            self.ctx.store["chunk_origin"] = origin
            if origin != "prompt" and self.cache.block_caches:
                stride = self.step_stride if origin == "generated" else self.prompt_scale
                for bc in self.cache.block_caches.values():
                    b_base = ((float(bc.pos[-1]) + stride) if len(bc.pos) else 0.0) + jump
                    b_pos = [b_base + i * stride for i in range(len(ids))]
                    bc.add_tokens(ids, b_pos, origin)
            if "embed" in self.box_code:
                from ..blockcode import run_embed
                self.x = run_embed(self.model, self.box_code["embed"], ids, self.ctx)
            else:
                self.x = self.model.embed(ids)
            if origin == "prompt" and self.media is not None:
                self.x = self.model.inject_media(self.x, ids, self.media)
            self.chunk, self.bp, self.trace = ch, 0, []
            if self.trace_on:
                self.trace.append({"pos": -1, "node": "embed", "label": "Embed", "top": self._lens_top(self.x),
                                   "resid_norm": round(float(self.x[-1].float().norm()), 3)})
            return True
        p, sched = self.bp, self.plan.schedule
        ids = self.chunk[0]
        self.ctx.offset = self.offset
        self.x = self.model.block(p, sched[p], self.x, self.cache, self.ctx, lambda iv: self._idx(iv, self.offset, len(ids)))
        self._report(p)
        st = self.ctx.store["stats"][p]
        if not self.trace_on:
            snap = None
        else:
            host_sync()                          # the trace reads this block's captures right away
            snap = {"pos": p, "layer": sched[p], "node": self.plan.schedule_nodes[p], "top": self._lens_top(self.x),
                    "resid_norm": round(float(st["resid"][-1][-1]), 3), "attn_norm": round(float(st["attn"][-1][-1]), 3),
                    "mlp_norm": round(float(st["mlp"][-1][-1]), 3)}
        rows = self.ctx.store.get("attn", {}).get(p)
        if snap is not None:
            if rows:
                row = rows[-1][:, -1, :].float().mean(0)
                v, i = row.topk(min(5, row.shape[0]))
                bc = self.cache.block_caches.get(self.cache.shared.get(p, p))
                tok_list = bc.tokens if bc is not None else self.cache.tokens
                snap["attends_to"] = [[int(j), self.ad.tok.piece(tok_list[int(j)]) if int(j) < len(tok_list) else "", round(float(w), 3)]
                                      for w, j in zip(v, i)]
            self.trace.append(snap)
            self.last_lens[p] = snap["top"][0][0]
        # Check if a side-branch fork at this block triggered an early gate / halt
        gate_halt = self.ctx.store.get("fork_gate_halt")
        if gate_halt:
            self.ctx.store.pop("chunk_origin", None)
            self.seq += ids
            self.offset += len(ids)
            if isinstance(gate_halt, str) and gate_halt.strip():
                gate_ids = self.ad.tok.encode(gate_halt, add_special=False)
                self.seq += gate_ids
                self.n_generated += len(gate_ids)
            self.x = None
            self.done = True
            return False

        self.bp += 1
        if self.bp == len(sched):                 # end of the stack: final norm + unembed
            self.ctx.store.pop("chunk_origin", None)
            h = self.x
            if "final_norm" in self.box_code or "lm_head" in self.box_code:
                from ..blockcode import run_final_norm, run_lm_head
                h = (run_final_norm(self.model, self.box_code["final_norm"], h, self.ctx)
                     if "final_norm" in self.box_code else self.model.final_norm(h))
                logits = (run_lm_head(self.model, self.box_code["lm_head"], h, self.w_out, self.ctx)
                          if "lm_head" in self.box_code else self.model.unembed(h, self.w_out))
            else:
                logits = self.model.final(h, self.w_out)
            self.logits_rows.append(logits)
            self.seq += ids
            self.offset += len(ids)
            if getattr(self, "_chunk_positions", None):
                self.last_pos = float(self._chunk_positions[-1])
            self.pos_next = (self.last_pos + (self.step_stride if self.chunk[1] == "generated" else self.prompt_scale)) if self.last_pos is not None else (self.pos_next + len(ids))
            self.x = None
        return True

    def step_token(self) -> bool:
        if self.x is None and not self.step_block():
            return False
        while self.x is not None:
            self.step_block()
        host_sync()                              # one wait per token: every capture of this token is now readable
        return True

    def run_to(self, pos: int, limit: int = 100_000) -> bool:
        """Run until block at schedule position `pos` has just executed (in the next forward pass)."""
        first = True
        for _ in range(limit):
            if not first and self.x is not None and self.bp == pos + 1:
                return True
            first = False
            if not self.step_block():
                return False
        return True

    def extend(self, total: int) -> None:
        """Raise the token budget of a run that stopped at its limit, so it can go on appending tokens."""
        total = int(total)
        if total <= self.total:
            return
        if self.x is not None:
            raise ValueError("Finish the current token first (a block is mid-way).")
        if self.eos_stop or self.ctx.store.get("fork_gate_halt"):
            raise ValueError("The run ended on its own (end-of-text or a guardrail gate), so there is nothing to append to.")
        self.total = self.p_["max_new_tokens"] = total
        self.planned = self.n_in + total
        self.ctx.token_ranges["generated"] = range(self.n_in, self.planned)
        self.done = False

    def force(self, token_id: int):
        self.forced = int(token_id)

    def state(self) -> dict:
        tok = self.ad.tok
        st = {"done": self.done, "phase": "done" if self.done else ("idle" if self.x is None else "block"),
              "next_pos": self.bp if self.x is not None else None,
              "active_node": (self.plan.schedule_nodes[self.bp] if self.x is not None and self.bp < len(self.plan.schedule) else None),
              "last_node": self.trace[-1]["node"] if self.trace else None,
              "chunk": [tok.piece(i) for i in self.chunk[0]] if self.chunk else [],
              "chunk_origin": self.chunk[1] if self.chunk else None,
              "tokens": [tok.piece(i) for i in self.seq], "n_in": self.n_in, "n_generated": self.n_generated,
              "total": self.total, "can_extend": self.done and not self.eos_stop and self.x is None,
              "trace": self.trace, "forced": self.forced, "steps": self.steps,
              "text": tok.decode(self.seq[self.n_in:], skip_special=True)}
        if self.x is None and self.logits_rows and not self.done:
            pv, pi = torch.softmax(self.logits_rows[-1][-1], -1).topk(10)
            st["next_choices"] = [[tok.piece(int(i)), int(i), round(float(v), 4)] for v, i in zip(pv, pi)]
        return st

    # ---------------------------------------------------------------- results
    def finish(self) -> dict:
        if not self.logits_rows and not self.seq:
            raise ValueError("Nothing was computed yet.")
        return self.snapshot()

    def snapshot(self) -> dict:
        ad, ctx, plan = self.ad, self.ctx, self.plan
        inflight = list(self.chunk[0]) if self.x is not None else []
        n_proc = sum(r.shape[0] for r in self.logits_rows)
        proc_ids = self.seq[:max(n_proc, len(self.seq))] + inflight
        if not proc_ids:
            raise ValueError("Nothing was computed yet.")
        if self.logits_rows:
            logits = torch.cat(self.logits_rows)
            probs = torch.softmax(logits, -1)
            tv, ti = probs.topk(10, -1)
            alts = [[[ad.tok.piece(int(ti[r, j])), int(ti[r, j]), round(float(tv[r, j]), 4)] for j in range(10)]
                    for r in range(n_proc)]
            final_top = logits.argmax(-1).tolist()
        else:
            alts, final_top = [], []
        caps = ad._collect(ctx, plan, proc_ids, self.n_prefix)
        if inflight and self.bp == 0:
            cur = self.x.detach().float().cpu()
            prev = caps["resid_pre"].get(0)
            caps["resid_pre"][0] = cur if prev is None else torch.cat([prev, cur])
        caps["cache"] = self.cache
        caps["final_top"] = final_top
        caps["n_in"] = self.n_in
        caps["linear_slots"] = [pp for pp, l in enumerate(plan.schedule) if ad.cfg.is_linear(l)]
        caps["consistent"] = self.x is None
        block_stats = {}
        for pp in range(len(plan.schedule)):
            st = ctx.store.get("stats", {}).get(pp)
            if not st:
                continue
            lens_last = self.last_lens.get(pp)
            if lens_last is None and pp in caps["resid_post"]:
                lg = self.model.final(caps["resid_post"][pp][-1:].to(ad.device, self.model.dtype), self.w_out)
                lens_last = ad.tok.piece(int(lg[0].argmax()))
            block_stats[plan.schedule_nodes[pp]] = {
                "pos": pp, "layer": plan.schedule[pp],
                "resid_norm": round(float(torch.cat(st["resid"]).mean()), 3),
                "attn_norm": round(float(torch.cat(st["attn"]).mean()), 3),
                "mlp_norm": round(float(torch.cat(st["mlp"]).mean()), 3),
                "lens_last": lens_last,
                "pending": bool(inflight) and pp >= self.bp}
            bc = self.cache.block_caches.get(pp)
            if bc is not None:
                block_stats[plan.schedule_nodes[pp]]["kv_override"] = {
                    "op": bc.op, "label": bc.label, "n_tokens": len(bc), "n_prompt": bc.n_prompt
                }
            sel = caps["moe_sel"].get(pp)
            if sel is not None and len(sel):
                E = self.model.moe_layers[plan.schedule[pp]]["n"]
                cnt = torch.bincount(sel[caps["moe_w"][pp] != 0].long(), minlength=E).float() / sel.shape[0]
                block_stats[plan.schedule_nodes[pp]]["moe"] = {
                    "usage": [round(float(c), 3) for c in cnt], "k": int(sel.shape[1]),
                    "last": [[int(e), round(float(w), 3)] for e, w in zip(sel[-1], caps["moe_w"][pp][-1])]}
        n_in = self.n_in
        seq = self.seq
        gen_text = ad.tok.decode(seq[n_in:], skip_special=True)
        gate_halt = ctx.store.get("fork_gate_halt")
        if isinstance(gate_halt, str) and gate_halt.strip() and gate_halt not in gen_text:
            gen_text = (gen_text + (" " if gen_text else "") + gate_halt).strip()
        outputs = {
            "text": gen_text,
            "tokens": [ad.tok.piece(i) for i in seq + inflight], "token_ids": seq + inflight,
            "segments": [{"origin": "prompt", "start": 0, "end": len(self.prompt_ids)},
                         {"origin": "prefill", "start": len(self.prompt_ids), "end": n_in},
                         {"origin": "generated", "start": max(n_in, 0), "end": max(n_in, len(seq))},
                         {"origin": "inflight", "start": len(seq), "end": len(seq) + len(inflight)}],
            # where each message's content sits in "tokens": [{"role", "index", "start", "end"}] (end exclusive)
            "roles": [dict(r) for r in self.roles],
            "alternatives": alts,
            "block_stats": block_stats,
            "forks": dict(ctx.store.get("fork_outputs") or {}),
            "n_heads": ad.cfg.n_heads,
            "prefix_tokens": self.n_prefix,
            "cache": self.cache.summary(ad.tok.piece),
            "attention_captured": bool(caps["attn"]),
            "weights": self.model.W.describe(),
            "gb_read": round(self.model.W.bytes_read / 2**30, 2),
            "notes": getattr(self, "notes", []),
            "prints": list(ctx.store.get("prints", [])),
            "edited_code": sorted(self.code),
            "captured": {"neurons": bool(caps["neurons"]), "parts": bool(caps["mlp_out"]),
                         "router": bool(caps["router"]), "expert_parts": bool(caps["moe_parts"])},
            "live": {"phase": "done" if self.done else ("idle" if self.x is None else "block"),
                     "next_pos": self.bp if self.x is not None else None, "steps": self.steps,
                     "n_complete": n_proc, "inflight": len(inflight)},
        }
        return {"outputs": outputs, "captures": caps}
