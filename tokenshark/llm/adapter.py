"""LLMAdapter: high-level runtime adapter for Causal LM models (`tokenshark.llm.adapter`)."""
from __future__ import annotations

from pathlib import Path
import torch

from ..engine import RunContext, host_sync
from ..flow import Plan
from ..kvcache import BlockKVCache, KVCache
from ..tokenizer import LLMTokenizer
from .analysis import LLMAnalysisMixin
from .arch import ArchHooks
from .media import MediaEncoder
from .model import RawLLM
from .stepper import Stepper


class LLMAdapter(LLMAnalysisMixin):
    family = "causal_lm"

    def __init__(self, spec: dict, root: Path, device: str, dtype: torch.dtype, policy="auto", keep_layers=0,
                 arch: ArchHooks | None = None):
        """arch: a model plugin's additions to the generic engine (llm/arch.py); None for the built-in families."""
        self.spec = spec
        from ..weights import report_phase
        report_phase("reading config")
        self.model = RawLLM(root, device, dtype, policy, keep_layers, arch)
        report_phase("loading tokenizer")
        self.tok = LLMTokenizer(root)
        self.cfg = self.model.cfg
        self.device = device
        self.cache_device = device if self.model.W.policy == "device" else "cpu"
        self.media: MediaEncoder | None = arch.media_encoder(self.model) if arch is not None else None
        self._media_cache = None

    def _mlp_label(self) -> str:
        act = self.cfg.act
        return "SwiGLU" if act == "silu" else "GeGLU" if act.startswith("gelu") else f"{act} GLU"

    def _attn_label(self, l: int | None = None) -> str:
        """What the attention of layer l is (l=None: the model as a whole)."""
        c = self.cfg
        if l is not None and c.is_linear(l):
            return "Gated DeltaNet (linear attn)"
        mixed = c.heterogeneous()
        if c.mla:
            base = "MLA (latent attention)"
        elif l is None and mixed:
            base = "sliding/global attention"
        else:
            kv = c.layer_n_kv(l)
            base = ("MHA" if kv == c.n_heads else f"GQA {c.n_heads}q/{kv}kv") + (" K=V" if c.k_eq_v(l) else "")
        if self.model.attn_gate:
            base = "gated " + base
        if l is not None and mixed:
            w = c.layer_window(l)
            base = (f"sliding {w} · " if w else "global · ") + base + f" · hd {c.layer_head_dim(l)}"
        if l is not None and c.kv_source(l) is not None:
            base += f" · KV of {c.kv_source(l)}"
        return base + (" · QK-norm" if self.model.qk_norm else "")

    def _block_detail(self, l: int) -> dict:
        """What layer l is made of, for the block inspector: attention shape, KV cache, RoPE, norms and MLP."""
        c, m = self.cfg, self.model
        d = {"layer": l, "linear": c.is_linear(l), "norm": "sandwich" if m.sandwich else "pre",
             "norm_offset": getattr(c, "norm_offset", None), "mlp": self._mlp_label(), "act": c.act}
        try:
            shp = m.W.shape(f"model.layers.{l}.mlp.down_proj.weight")
            d["d_ff"] = int(shp[-1])
        except Exception:
            d["d_ff"] = c.inter
        if l in m.moe_layers:
            lay = m.moe_layers[l]
            d["moe"] = {"n": lay["n"], "k": c.moe["k"], "shared": lay["shared"]}
        if m.W.has(f"model.layers.{l}.layer_scalar"):
            d["layer_scalar"] = True
        if d["linear"]:
            return d
        rope = c.layer_rope(l) or {}
        theta = float(rope.get("rope_theta") or c.theta)
        frac = rope.get("partial_rotary_factor")
        hd = c.layer_head_dim(l)
        rot = int(hd * float(frac)) if frac else (c.rot_dim or hd)
        d.update({"window": c.layer_window(l), "n_heads": c.n_heads, "n_kv": c.layer_n_kv(l), "head_dim": hd,
                  "k_eq_v": c.k_eq_v(l), "qk_norm": bool(m.qk_norm), "v_norm": bool(c.v_norm),
                  "attn_gate": bool(m.attn_gate), "mla": bool(c.mla), "kv_source": c.kv_source(l),
                  "attn_bias": bool(getattr(c, "attn_bias", False)),
                  "attn_scale": c.attn_scale,
                  "rope": {"theta": theta, "type": rope.get("rope_type") or (c.rope_scaling or {}).get("rope_type")
                           or "default", "rot_dim": rot}})
        return d

    def info(self) -> dict:
        c = self.cfg
        mlp = self._mlp_label()
        out = {
            "family": self.family, "arch": c.arch, "n_layers": c.n_layers, "d_model": c.hidden,
            "n_heads": c.n_heads, "n_kv_heads": c.n_kv, "head_dim": c.head_dim, "vocab": c.vocab,
            "d_ff": c.inter, "rope_theta": c.theta,
            "tokenizer": "tiktoken" if self.tok.kind == "tiktoken" else "BPE tokenizer",
            "norm": "RMSNorm (pre + post)" if self.model.sandwich else "RMSNorm",
            "block_sub": f"{self._attn_label()} · {mlp}",
            "moe": ({"n_experts": c.moe["n"], "top_k": c.moe["k"], "norm_topk": c.moe["norm_topk"],
                     "expert_d_ff": c.moe["inter"], "layers": sorted(self.model.moe_layers),
                     "shared": any(v["shared"] for v in self.model.moe_layers.values())} if c.moe else None),
            "has_chat_template": bool(self.tok.chat_template),
            "block_subs": [self._attn_label(i)
                           + (f" · MoE {c.moe['k']} of {self.model.moe_layers[i]['n']} experts"
                              + (" + shared" if self.model.moe_layers[i]["shared"] else "")
                              if i in self.model.moe_layers else f" · {mlp}")
                           for i in range(c.n_layers)],
            "hybrid": bool(c.layer_types),
            "kv_sources": [c.kv_source(i) for i in range(c.n_layers)],
            "block_details": [self._block_detail(i) for i in range(c.n_layers)],
            "kv_layers": [i for i in range(c.n_layers) if not c.is_linear(i)],
            "linear_layers": [i for i in range(c.n_layers) if c.is_linear(i)],
            "extra_blocks": list(getattr(self, "extra_blocks", None) or []),
            "vision": bool(c.vision),
            "image_input": self.media is not None,
            "media_marker": self.media.marker if self.media is not None else None,
            "final_logit_softcap": c.final_softcap,
            "weights": self.model.W.describe(),
            "capabilities": {"repeat_blocks": True, "reorder_blocks": True, "remove_blocks": True,
                             "logit_lens": True, "attention": True, "kv_cache": True, "has_moe": bool(c.moe),
                             "image_input": self.media is not None},
            "hooks": ["resid_pre", "q", "k", "v", "kv_k", "kv_v", "rope", "attn_probs", "attn_heads", "attn_out",
                      *(["router", "expert_out"] if c.moe else []), "mlp_neurons", "mlp_out", "resid_post", "weights"],
        }
        if c.heterogeneous():
            out["sliding_window"] = next((la.window for la in c.layers if la.window), None)
            out["global_layers"] = [l for l, la in enumerate(c.layers) if not la.window]
        return out

    def fingerprint(self):
        return {**self.cfg.fingerprint(), "model_id": self.spec["id"]}

    def new_cache(self, n_slots: int) -> KVCache:
        return KVCache(self.fingerprint(), n_slots, device=self.cache_device)

    # -------------------------------------------------------------- input building
    def split_system(self, system: str, user, extra: dict | None = None) -> tuple[str, str]:
        tok = self.tok
        sys_msg = [{"role": "system", "content": system}]
        turns = user if isinstance(user, list) else [{"role": "user", "content": user}]
        full = tok.apply_chat(sys_msg + turns, True, extra)
        try:
            head = tok.apply_chat(sys_msg, False, extra)
            if full.startswith(head):
                return head, full[len(head):]
        except Exception:
            pass
        rest = tok.apply_chat(turns, True, extra)
        if full.endswith(rest):
            return full[: len(full) - len(rest)], rest
        raise ValueError("Couldn't separate the system turn in this model's chat template.")

    def template_extra(self, p: dict) -> dict:
        return {"enable_thinking": bool(p.get("thinking", False))}

    @staticmethod
    def turns(p: dict) -> list[dict]:
        hist = [{"role": m["role"], "content": m["content"]} for m in (p.get("history") or [])]
        if p.get("prompt", "") or not hist:
            hist.append({"role": "user", "content": p.get("prompt", "")})
        return hist

    # -------------------------------------------------------------- media input (images)
    def _media_items(self, p: dict) -> list | None:
        """The prepared media of p["images"] (the encoder's prepare()), cached for the last set."""
        srcs = list(p.get("images") or [])
        if not srcs:
            return None
        if self.media is None:
            raise ValueError("This model doesn't take image input.")
        key = tuple(hash(x) if isinstance(x, (str, bytes)) else id(x) for x in srcs)
        if self._media_cache is None or self._media_cache[0] != key:
            self._media_cache = [key, [self.media.prepare(x) for x in srcs], None]
        return self._media_cache[1]

    def media_features(self, p: dict) -> torch.Tensor | None:
        """Soft-token embeddings of all media in the prompt, concatenated in prompt order."""
        items = self._media_items(p)
        if not items:
            return None
        if self._media_cache[2] is None:
            self._media_cache[2] = torch.cat([self.media.encode(it) for it in items])
        return self._media_cache[2]

    def _with_media_marks(self, p: dict, n: int) -> dict:
        """One marker per media item in the text; with none, the markers go in front of the newest user turn."""
        mark = self.media.marker
        texts = [p.get("system") or "", p.get("prompt") or ""] + [m.get("content") or "" for m in (p.get("history") or [])]
        have = sum(str(t).count(mark) for t in texts)
        if have == n:
            return p
        if have:
            raise ValueError(f"The prompt places {have} {mark} markers but {n} image(s) were given.")
        return {**p, "prompt": mark * n + (p.get("prompt") or "")}

    def _expand_media(self, ids: list[int], items: list) -> list[int]:
        """Each marker token becomes begin + token x n_tokens(item) + end."""
        enc = self.media
        if ids.count(enc.token_id) != len(items):
            raise ValueError(f"Found {ids.count(enc.token_id)} {enc.marker} tokens after tokenizing but "
                             f"{len(items)} image(s) were given.")
        wrap = lambda t: [t] if t is not None else []              # noqa: E731
        out, i = [], 0
        for t in ids:
            if t == enc.token_id:
                out += wrap(enc.begin_id) + [enc.token_id] * enc.n_tokens(items[i]) + wrap(enc.end_id)
                i += 1
            else:
                out.append(t)
        return out

    def build_ids(self, p: dict) -> tuple[list[int], list[int]]:
        prompt_ids, prefill_ids, _ = self.build_ids_with_roles(p)
        return prompt_ids, prefill_ids

    def build_ids_with_roles(self, p: dict) -> tuple[list[int], list[int], list[dict]]:
        """(prompt ids, prefill ids, roles). roles: the token range of each message's content as the chat template
        placed it, [{"role", "index", "start", "end"}] (end exclusive, run-relative token indices; a system turn
        held in a prefix cache is not part of the run's tokens). The prefill is listed as role "prefill"."""
        items = self._media_items(p)
        if not items:
            prompt_ids, prefill_ids, roles = self._build_text_ids(p, with_roles=True)
        else:
            prompt_ids, prefill_ids, roles = self._build_text_ids(self._with_media_marks(p, len(items)), with_roles=True)
            size = lambda it: len(self._expand_media([self.media.token_id], [it])) - 1   # noqa: E731
            marks = [i for i, t in enumerate(prompt_ids) if t == self.media.token_id]
            grow = {i: size(it) for i, it in zip(marks, items)}       # extra tokens each marker turns into
            shift = lambda j: j + sum(g for i, g in grow.items() if i < j)            # noqa: E731
            prompt_ids = self._expand_media(prompt_ids, items)
            roles = [{**r, "start": shift(r["start"]), "end": shift(r["end"])} if r["role"] != "prefill" else
                     {**r, "start": len(prompt_ids), "end": len(prompt_ids) + len(prefill_ids)} for r in roles]
        return prompt_ids, prefill_ids, roles

    def _render_prompt(self, p: dict) -> tuple[str, bool, list[tuple[str, str]]]:
        """The prompt as text (chat template or plain), whether to add BOS, and the (role, content) of each message
        in the order they appear in the text."""
        tok = self.tok
        has_prefix = bool(p.get("prefix_cache"))
        extra = self.template_extra(p)
        turns = self.turns(p)
        msgs_in = [(m["role"], m["content"]) for m in turns]
        if has_prefix and p.get("use_chat_template") and tok.chat_template:
            _, text = self.split_system(p.get("prefix_system") or ".", turns, extra)
            return text, False, msgs_in
        if p.get("use_chat_template") and tok.chat_template:
            msgs = [{"role": "system", "content": p["system"]}] if p.get("system") else []
            text = tok.apply_chat(msgs + turns, True, extra)
            return text, False, ([("system", p["system"])] if p.get("system") else []) + msgs_in
        earlier = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in (p.get("history") or []))
        text = (p["system"] + "\n\n" if p.get("system") else "") + earlier + p.get("prompt", "")
        msgs = ([("system", p["system"])] if p.get("system") else []) + \
            [(m["role"], m["content"]) for m in (p.get("history") or [])] + \
            ([("user", p["prompt"])] if p.get("prompt") else [])
        return text, bool(p.get("add_bos", True)) and not has_prefix, msgs

    def _build_text_ids(self, p: dict, with_roles: bool = False):
        """Prompt and prefill ids. p["literal_content"]: message contents are read with parse_special=False, so
        text such as "<|im_end|>" inside a message stays text instead of becoming the control token."""
        tok = self.tok
        text, add_special, msgs = self._render_prompt(p)
        literal = bool(p.get("literal_content"))
        spans = tok.locate(text, [c for _, c in msgs])
        prompt_ids, tspans = tok.encode_spans(text, spans, add_special, literal=literal)
        prefill_ids = tok.encode(p["prefill"], add_special=False, parse_special=not literal) if p.get("prefill") else []
        if not with_roles:
            return prompt_ids, prefill_ids
        roles = [{"role": r, "index": i, "start": ts[0], "end": ts[1]}
                 for i, ((r, _), ts) in enumerate(zip(msgs, tspans)) if ts is not None]
        if prefill_ids:
            roles.append({"role": "prefill", "index": len(msgs), "start": len(prompt_ids),
                          "end": len(prompt_ids) + len(prefill_ids)})
        return prompt_ids, prefill_ids, roles

    def find_span(self, p: dict, substring: str, role: str | None = None, which: int = -1) -> range:
        """Token range(start, end) of `substring` inside the prompt (or prefill) rendered by `p`."""
        if not substring:
            raise ValueError("substring must be non-empty.")
        tok = self.tok
        text, add_special, msgs = self._render_prompt(p)
        literal = bool(p.get("literal_content"))
        msg_spans = tok.locate(text, [c for _, c in msgs])
        if role == "prefill":
            pf = str(p.get("prefill") or "")
            if not pf:
                raise KeyError("No 'prefill' in this prompt.")
            prompt_ids, _ = tok.encode_spans(text, msg_spans, add_special, literal=literal)
            r_pf = tok.find_span(pf, substring, add_special=False, literal=literal, which=which)
            return range(len(prompt_ids) + r_pf.start, len(prompt_ids) + r_pf.stop)
        search_ranges: list[tuple[int, int]] = []
        if role is not None:
            for (r_name, _), sp in zip(msgs, msg_spans):
                if r_name == role and sp is not None:
                    search_ranges.append(sp)
            if not search_ranges:
                raise KeyError(f"No {role!r} message in this prompt.")
        else:
            search_ranges.append((0, len(text)))
        matches: list[tuple[int, int]] = []
        for cand in (substring, substring.strip()):
            if not cand:
                continue
            for lo, hi in search_ranges:
                cur = lo
                while True:
                    i = text.find(cand, cur, hi)
                    if i < 0:
                        break
                    matches.append((i, i + len(cand)))
                    cur = i + 1
            if matches:
                break
        if not matches and role is None and p.get("prefill") and substring in str(p.get("prefill")):
            return self.find_span(p, substring, role="prefill", which=which)
        if not matches:
            raise ValueError(f"Substring {substring!r} not found in prompt.")
        target = matches[which]
        if not literal:
            _, tspans = tok.encode_spans(text, [target], add_special=add_special, literal=False)
            ts = tspans[0]
            if ts is None:
                raise ValueError(f"Substring {substring!r} did not map to any tokens.")
            return range(ts[0], ts[1])
        # When literal=True, message spans must be cut at the same boundaries as _build_text_ids
        prompt_ids, _ = tok.encode_spans(text, msg_spans, add_special=add_special, literal=True)
        # Map character span `target` onto `prompt_ids` by cumulative piece offsets
        _, offs = tok.encode_offsets(text, add_special=add_special, parse_special=False)
        if len(offs) == len(prompt_ids):
            hit = [j for j, (a, b) in enumerate(offs) if b > target[0] and a < target[1] and b > a]
            if hit:
                return range(hit[0], hit[-1] + 1)
        _, tspans = tok.encode_spans(text, [target], add_special=add_special, literal=False)
        ts = tspans[0]
        if ts is None:
            raise ValueError(f"Substring {substring!r} did not map to any tokens.")
        return range(ts[0], ts[1])

    @staticmethod
    def apply_kv_override_params(base_p: dict, ov: dict) -> dict:
        p = dict(base_p)
        p["history"] = [dict(m) for m in (base_p.get("history") or [])]
        op = ov.get("op", "replace_system")
        txt = str(ov.get("text") or "")
        sep = "\n" if txt and not txt.startswith((" ", "\n")) else ""
        if op == "replace_system":
            p["system"] = txt
        elif op == "append_system":
            cur = str(p.get("system") or "")
            p["system"] = (cur + ("\n" if cur and txt else "") + txt) if cur else txt
        elif op == "prepend_system":
            cur = str(p.get("system") or "")
            p["system"] = (txt + ("\n" if txt and cur else "") + cur) if cur else txt
        elif op == "replace_user":
            if p.get("prompt", "") or not p["history"]:
                p["prompt"] = txt
            else:
                for i in range(len(p["history"]) - 1, -1, -1):
                    if p["history"][i]["role"] == "user":
                        p["history"][i]["content"] = txt
                        break
        elif op == "append_user":
            if p.get("prompt", "") or not p["history"]:
                cur = str(p.get("prompt") or "")
                p["prompt"] = (cur + sep + txt) if cur else txt
            else:
                for i in range(len(p["history"]) - 1, -1, -1):
                    if p["history"][i]["role"] == "user":
                        cur = p["history"][i]["content"]
                        p["history"][i]["content"] = (cur + sep + txt) if cur else txt
                        break
        elif op == "prepend_user":
            sep_pre = "\n" if txt and not txt.endswith((" ", "\n")) else ""
            if p.get("prompt", "") or not p["history"]:
                cur = str(p.get("prompt") or "")
                p["prompt"] = (txt + sep_pre + cur) if cur else txt
            else:
                for i in range(len(p["history"]) - 1, -1, -1):
                    if p["history"][i]["role"] == "user":
                        cur = p["history"][i]["content"]
                        p["history"][i]["content"] = (txt + sep_pre + cur) if cur else txt
                        break
        return p

    @torch.no_grad()
    def recreate_block_caches(self, plan: Plan, start_pos: float = 0.0, prompt_scale: float = 1.0) -> dict[int, BlockKVCache]:
        overrides = plan.params.get("kv_overrides") or []
        if not overrides:
            return {}
        S_ = len(plan.schedule)
        slot_cfg: dict[int, list[dict]] = {}
        for ov in overrides:
            for s in ov.get("slots") or []:
                if 0 <= s < S_ and not self.cfg.is_linear(plan.schedule[s]):
                    slot_cfg.setdefault(s, []).append(ov)
        if not slot_cfg:
            return {}
        by_sig: dict[tuple, list[int]] = {}
        ov_for_sig: dict[tuple, list[dict]] = {}
        for s, ovs in slot_cfg.items():
            sig = tuple((o.get("op"), o.get("text")) for o in ovs)
            by_sig.setdefault(sig, []).append(s)
            ov_for_sig[sig] = ovs

        out: dict[int, BlockKVCache] = {}
        for sig, slots in by_sig.items():
            ovs = ov_for_sig[sig]
            mod_p = dict(plan.params)
            for o in ovs:
                mod_p = self.apply_kv_override_params(mod_p, o)
            mod_ids, _ = self.build_ids(mod_p)
            if not mod_ids:
                mod_ids = [self.tok.bos_id or 0]
            positions = [start_pos + i * prompt_scale for i in range(len(mod_ids))]
            tmp_cache = self.new_cache(S_)
            ctx = RunContext(interventions=[])
            self.model.forward(mod_ids, positions, "prompt", tmp_cache, plan.schedule, ctx, 0,
                               lambda iv, o, n: [], media=self.media_features(mod_p))
            last_ov = ovs[-1]
            op_summary = " + ".join(o.get("op", "") for o in ovs)
            text_summary = last_ov.get("text", "")
            label_summary = last_ov.get("label") or op_summary
            for s in slots:
                if tmp_cache.k[s] is None:
                    continue
                bc = BlockKVCache(
                    k=tmp_cache.k[s].detach().to(self.cache_device).clone(),
                    v=tmp_cache.v[s].detach().to(self.cache_device).clone(),
                    pos=tmp_cache.pos.detach().clone(),
                    tokens=list(tmp_cache.tokens),
                    segments=[s_seg for s_seg in tmp_cache.segments],
                    op=op_summary,
                    text=text_summary,
                    label=label_summary,
                    n_prompt=len(mod_ids),
                    device=self.cache_device,
                )
                out[s] = bc
        return out

    @torch.no_grad()
    def build_prefix(self, plan: Plan, text: str, as_system: bool) -> tuple[KVCache, dict]:
        tok = self.tok
        if as_system and tok.chat_template:
            rendered, _ = self.split_system(text, "x")
            ids = tok.encode(rendered, add_special=False)
        else:
            ids = tok.encode(text, add_special=True)
        if not ids:
            raise ValueError("Empty text.")
        cache = self.new_cache(len(plan.schedule))
        ctx = RunContext(interventions=[])
        self.model.forward(ids, [float(i) for i in range(len(ids))], "prompt", cache, plan.schedule, ctx, 0,
                           lambda iv, o, n: [])
        cache = cache.relabel("imported", "hidden system prompt" if as_system else "hidden prompt")
        extra = {"system_text": text if (as_system and tok.chat_template) else None, "schedule": plan.schedule,
                 "source": "text"}
        return cache, extra

    # -------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, plan: Plan, progress=None, resume=None, prefix: KVCache | None = None, ctx_hook=None) -> dict:
        st = Stepper(self, plan, prefix=prefix, progress=progress, ctx_hook=ctx_hook)
        while st.step_token():
            pass
        return st.finish()

    def run_stepper(self, plan: Plan, prefix=None, progress=None, ctx_hook=None) -> Stepper:
        return Stepper(self, plan, prefix=prefix, progress=progress, ctx_hook=ctx_hook)

    def stepper(self, plan: Plan, prefix: KVCache | None = None) -> Stepper:
        return Stepper(self, plan, prefix=prefix, trace=True)

    @torch.no_grad()
    def run_batch(self, plans: list[Plan], prefixes: list | None = None,
                  sources_list: list[dict] | None = None, progress=None) -> list[dict]:
        """Execute multiple plans in lockstep one block at a time inside `W.hold_device()` so each block's
        weights are streamed to the compute device once across the batch instead of once per item."""
        if not plans:
            return []
        prefixes = prefixes or [None] * len(plans)
        sources_list = sources_list or [{} for _ in plans]
        W = self.model.W
        with W.hold_device():
            steppers = [Stepper(self, pl, prefix=pr) for pl, pr in zip(plans, prefixes)]
        for st, src in zip(steppers, sources_list):
            if src:
                st.ctx.sources = src
        while any(not st.done for st in steppers):
            # 1. Embed next token chunk for all idle steppers
            with W.hold_device():
                for st in steppers:
                    if not st.done and st.x is None:
                        st.step_block()
            # 2. Step block-by-block in lockstep across all active steppers
            while True:
                active = [st for st in steppers if not st.done and st.x is not None]
                if not active:
                    break
                min_bp = min(st.bp for st in active)
                with W.hold_device():
                    for st in active:
                        if st.bp == min_bp:
                            st.step_block()
            host_sync()
        return [st.finish() for st in steppers]

    def _collect(self, ctx, plan, seq, n_prefix):
        host_sync()                              # captures copied asynchronously (engine.to_host) have all landed
        cat = lambda d: {k: torch.cat(v, 0) for k, v in d.items()}
        attn = {}
        for pp, rows in ctx.store.get("attn", {}).items():
            T = rows[-1].shape[-1]
            H = rows[0].shape[0]
            full = torch.zeros(H, sum(r.shape[1] for r in rows), T, dtype=torch.float16)
            i = 0
            for r in rows:
                full[:, i:i + r.shape[1], :r.shape[2]] = r
                i += r.shape[1]
            attn[pp] = full
        return {"resid_post": cat(ctx.store.get("resid_post", {})), "resid_pre": cat(ctx.store.get("resid_pre", {})),
                "attn_out": cat(ctx.store.get("attn_out", {})), "mlp_out": cat(ctx.store.get("mlp_out", {})),
                "q": cat(ctx.store.get("q", {})), "k": cat(ctx.store.get("k", {})), "v": cat(ctx.store.get("v", {})),
                "neurons": cat(ctx.store.get("neurons", {})),
                "router": cat(ctx.store.get("router", {})), "moe_sel": cat(ctx.store.get("moe_sel", {})),
                "moe_w": cat(ctx.store.get("moe_w", {})), "moe_parts": cat(ctx.store.get("moe_parts", {})),
                "moe_shared": cat(ctx.store.get("moe_shared", {})),
                "moe_k": dict(ctx.store.get("moe_k") or {}), "moe_temp": dict(ctx.store.get("moe_temp") or {}),
                "moe_noshared": set(ctx.store.get("moe_noshared") or ()),
                "rope": dict(ctx.store.get("rope") or {}), "rope_pos": dict(ctx.store.get("rope_pos") or {}),
                "last_q": {pp: {i: (qv, float(pos)) for i, (qv, pos) in h.items()}
                           for pp, h in (ctx.store.get("last_q") or {}).items()},
                "interventions": list(plan.interventions), "sources": dict(ctx.sources),
                "token_ranges": dict(ctx.token_ranges),
                "unit_range": {k: (lo.cpu(), hi.cpu()) for k, (lo, hi) in (ctx.store.get("unit_range") or {}).items()},
                "code": dict(ctx.store.get("code", {})),
                "forks": dict(ctx.store.get("fork_outputs") or {}),
                "attn": attn, "ids": seq, "n_prefix": n_prefix,
                # attn_queries="last": token index of each stored row, per block (absent = rows are tokens 0..T-1)
                "attn_rows": {k: list(v) for k, v in (ctx.store.get("attn_rows") or {}).items()}}

    def _sample(self, logits: torch.Tensor, p: dict, gen: torch.Generator) -> int:
        if p["temperature"] <= 0:
            return int(logits.argmax())
        logits = logits.float() / p["temperature"]
        if p.get("top_p", 1.0) < 1.0:
            probs = torch.softmax(logits, -1)
            if p["top_k"] > 0:
                kth = probs.topk(min(p["top_k"], probs.shape[-1])).values[-1]
                probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            sp, si = probs.sort(descending=True)
            keep = (sp.cumsum(-1) - sp) < p["top_p"] * sp.sum()
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            return int(si[torch.multinomial((sp / sp.sum()).cpu(), 1, generator=gen)].item())
        if p["top_k"] > 0:
            v, i = logits.topk(min(p["top_k"], logits.shape[-1]))
            return int(i[torch.multinomial(torch.softmax(v, -1).cpu(), 1, generator=gen)].item())
        return int(torch.multinomial(torch.softmax(logits, -1).cpu(), 1, generator=gen).item())

    def unload(self):
        self.model = None
