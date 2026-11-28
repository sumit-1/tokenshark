"""Python API: the engine behind the app, for scripts and notebooks.

    from tokenshark.api import Lab

    lab = Lab("qwen2.5-0.5b-instruct")                       # same loader, same raw engine as the app
    run = lab.run("The capital of France is", max_new_tokens=6)
    print(run.text)
    run.lens()                    # logit lens table            run.attention(5)      # [heads, tokens, tokens]
    run.resid(5)                  # [tokens, d_model]           run.top_neurons(5)    # strongest MLP neurons

    # activation patching: where does the model keep "France"?
    clean = lab.run("The capital of France is", max_new_tokens=1)
    flow = lab.flow("The capital of Japan is", max_new_tokens=1)
    flow.patch(8, clean, hook="resid_post", positions="last")
    print(lab.run(flow).text)

    # steering
    v = lab.vector_from_runs(lab.run("I love this"), lab.run("I hate this"), block=6)
    flow = lab.flow("The film was"); flow.steer(6, v, value=4)
    print(lab.run(flow).text)

A run made in the app can be exported as a script that uses this API (Output tab → Export as script).
Everything returned is plain torch tensors and dicts, so the next step can be your own analysis.
"""
from __future__ import annotations

import uuid
import warnings
from pathlib import Path

import torch

from . import interp
from .blockcode import default_code
from .engine import pick_device, pick_dtype
from .flow import compile_flow
from .flowbuild import FlowBuilder
from .lora import LoRAOverlay
from .plugins import find_plugin_for
from .quant import QuantOverlay

__all__ = ["Lab", "RunResult", "FlowBuilder"]


class RunResult:
    """One finished run: what came out, and everything the engine recorded on the way."""

    def __init__(self, lab: "Lab", plan, res: dict, flow: dict):
        self.id = uuid.uuid4().hex[:10]
        self.lab, self.plan, self.flow = lab, plan, flow
        self.outputs, self.captures = res["outputs"], res["captures"]

    # ---------------------------------------------------------------- outputs
    @property
    def text(self) -> str:
        return self.outputs.get("text", "")

    @property
    def thought(self) -> str:
        """Reasoning / thinking channel text emitted before the final answer ('' when none)."""
        if "thought" in self.outputs:
            return self.outputs["thought"]
        return self.lab.adapter.tok.split_thinking(self.new_token_ids)[0]

    @property
    def answer(self) -> str:
        """The model's final answer with any thinking channel ('<|channel>thought...<channel|>' or '<think>...</think>') stripped."""
        if "answer" in self.outputs:
            return self.outputs["answer"]
        return self.lab.adapter.tok.split_thinking(self.new_token_ids)[1]

    @property
    def tokens(self) -> list[str]:
        return self.outputs.get("tokens", [])

    @property
    def token_ids(self) -> list[int]:
        return self.outputs.get("token_ids", [])

    @property
    def new_token_ids(self) -> list[int]:
        """Token ids generated after the prompt and prefill."""
        n_in = self.captures.get("n_in") if self.captures else None
        if n_in is None:
            segs = self.outputs.get("segments") or []
            gen = next((s for s in segs if s.get("origin") == "generated"), None)
            if gen:
                return self.token_ids[gen["start"]:gen["end"]]
            return []
        return self.token_ids[n_in:]

    @property
    def roles(self) -> list[dict]:
        """Where each message's content sits in run.tokens: [{"role", "index", "start", "end"}] (end exclusive)."""
        return [dict(r) for r in self.outputs.get("roles") or []]

    def span(self, role: str, which: int = -1) -> range:
        """Token range of one message's content: span("system"), span("user") (the last user turn),
        span("user", 0) (the first), span("prefill"). Use it to slice attention: run.attention(b)[:, q, span]."""
        hits = [r for r in self.roles if r["role"] == role]
        if not hits:
            raise KeyError(f"No {role!r} message in this run's tokens (roles: {[r['role'] for r in self.roles]}).")
        r = hits[which]
        return range(r["start"], r["end"])

    def find(self, substring: str, role: str | None = None, which: int = -1) -> range:
        """Token range(start, end) covering `substring` inside this run's prompt (or inside `role`).
        Pass directly to `flow.attn_gain(b, keys=run.find(suffix))` or `run.attention(b)[:, q, run.find(suffix)]`."""
        return self.lab.adapter.find_span(self.plan.params, substring, role=role, which=which)

    @property
    def image(self):
        """Image models: the picture as a PIL image."""
        import base64, io
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(self.outputs["image"].split(",", 1)[1])))

    def alternatives(self, token: int) -> list:
        """The ten most likely choices for token number `token` (piece, id, probability)."""
        return self.outputs["alternatives"][token - 1]

    # ---------------------------------------------------------------- activations
    def resid(self, block: int, where: str = "post") -> torch.Tensor:
        """Residual stream before ('pre') or after ('post') the block at schedule position `block`: [tokens, d]."""
        return interp.activation(self.captures, f"resid_{where}", block)

    def part(self, block: int, hook: str) -> torch.Tensor:
        """attn_out or mlp_out of a block: what attention / the MLP added to the stream."""
        return interp.activation(self.captures, hook, block)

    @property
    def notes(self) -> list[str]:
        """What the engine wants you to know about this run (e.g. attention that was over budget, so not recorded)."""
        return list(self.outputs.get("notes") or [])

    def attention(self, block: int, head: int | None = None) -> torch.Tensor:
        """Attention pattern [heads, queries, keys] (or [queries, keys] for one head). Rows are every token, or with
        attn_queries="last" one row per forward pass (attention_rows(block) gives their token indices). Keys
        include any prefix-cache tokens first."""
        attn = self.captures.get("attn") or {}
        if block not in attn:
            why = next((n for n in self.notes if n.startswith("attention not captured")), None)
            if why:
                raise KeyError(f"No attention for block {block}: {why}")
            if attn:
                raise KeyError(f"No attention for block {block}: recorded blocks are {sorted(attn)} "
                               f"(attn_layers, or a linear-attention block).")
            raise KeyError(f"No attention for block {block}: this run recorded none (attn_budget_mb=0?).")
        a = attn[block].float()
        return a if head is None else a[head]

    def attention_rows(self, block: int) -> list[int]:
        """Token index of each row of attention(block)."""
        rows = (self.captures.get("attn_rows") or {}).get(block)
        return list(rows) if rows is not None else list(range(self.captures["attn"][block].shape[1]))

    def neurons(self, block: int) -> torch.Tensor:
        """Every MLP neuron's activation: [tokens, d_ff]."""
        return interp._neuron_acts(self.captures, block).float()

    def logits(self, position: int = -1) -> torch.Tensor:
        """Final scores over the vocabulary at one token position."""
        S = len(self.plan.schedule)
        return self.lab.adapter.decode_layer(self.captures, self.plan.schedule, S - 1, 0)[position]

    # ---------------------------------------------------------------- analyses (same as the app's tabs)
    def lens(self, k: int = 5, through: int = 0) -> dict:
        return self.lab.adapter.logit_lens(self.captures, self.plan.schedule_nodes, self.plan.schedule, k, through)

    def track(self, word: str, position: int = -1, through: int = 0) -> dict:
        return self.lab.adapter.track_word(self.captures, self.plan.schedule_nodes, self.plan.schedule, position, word, through)

    def top_neurons(self, block: int, token: int = -1, k: int = 20, by: str = "token") -> dict:
        return interp.top_neurons(self.lab.adapter, self.captures, self.plan.schedule, block, token, k, by)

    def neuron(self, block: int, index: int) -> dict:
        return interp.neuron_detail(self.lab.adapter, self.captures, self.plan.schedule, block, index)

    def kv_analysis(self, blocks: str = "all", query: int | None = None) -> dict:
        """Analyze this run's KV cache: GQA memory savings, per-block & per-token K/V norms, attention sinks,
        and how much attention each cached token lost (or gained) to its RoPE distance from one query.
        Per-token columns are averaged over `blocks` (schedule positions: all | 5 | 3,7 | 10:14). Attention
        columns are for one forward pass: by default the one that produced the last output token; `query`
        picks the pass at another cache index (it predicted the token after it)."""
        return interp.kv_analysis(self.lab.adapter, self.captures, self.plan.schedule, self.plan.schedule_nodes,
                                  blocks, query)

    def trace_attention(self, query_token: int = -1, key_token: int | None = None,
                        target_word: str = "", top_k: int = 30) -> dict:
        """Rank `(block, head)` pairs across the model that attend to and copy from `key_token` into `query_token`."""
        from .extensions.attn_traceback import trace_previous_tokens
        return trace_previous_tokens(
            self.lab.adapter, self.captures, self.plan.schedule, self.plan.schedule_nodes,
            query_token=query_token, key_token=key_token, target_word=target_word, top_k=top_k,
        )

    # ---------------------------------------------------------------- conversations
    @property
    def conversation(self) -> list[dict]:
        """Every turn of this run including the model's answer: [{"role", "content"}, …]."""
        p = self.plan.params
        turns = self.lab.adapter.turns(p)
        return turns + [{"role": "assistant", "content": (p.get("prefill") or "") + self.text}]

    def reply(self, text: str, **kw) -> "RunResult":
        """Continue the conversation: this run's turns and answer become the history and `text` the next user
        turn. Interceptors, bypasses and edited code of the flow stay in place. Keyword arguments as for lab.flow
        (prefill, max_new_tokens, …)."""
        fb = FlowBuilder(self.flow)
        fb.prompt(prompt=text, prefill=kw.pop("prefill", ""), history=self.conversation,
                  **{k: kw.pop(k) for k in ("system", "chat", "thinking") if k in kw})
        if kw:
            fb.sampler(**kw)
        return self.lab.run(fb)

    # ---------------------------------------------------------------- mixture of experts
    def routing(self, block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(chosen experts [tokens, k], their weights [tokens, k], router scores [tokens, experts]) of an MoE block."""
        from . import moe
        sel, w = moe._need(self.captures, block)
        return sel, w, (self.captures.get("router") or {}).get(block)

    def experts(self, block: int | None = None, token: int = -1) -> dict:
        """Expert usage of every MoE block (block=None), or one block's routing table and what each expert wrote."""
        from . import moe
        if block is None:
            return moe.overview(self.lab.adapter, self.captures, self.plan.schedule, self.plan.schedule_nodes)
        return moe.block_detail(self.lab.adapter, self.captures, self.plan.schedule, block, token)

    def expert(self, block: int, index: int) -> dict:
        from . import moe
        return moe.expert_detail(self.lab.adapter, self.captures, self.plan.schedule, block, index)

    # ---------------------------------------------------------------- introspection
    def meter(self, vector, centre: bool = True) -> dict:
        """How strongly each block's stream points along a vector (tensor, vector dict or id), token by token."""
        from . import introspect
        v = vector if isinstance(vector, torch.Tensor) else self.lab.load_vector(vector)
        return introspect.concept_meter(self.lab.adapter, self.captures, self.plan.schedule, v, centre, self.plan)

    def verbalize(self, block: int, token: int = -1, inject: int = 2, **kw) -> dict:
        """activation → words → activation: the model explains one of its own activations, and the explanation is
        scored by how well the activation can be rebuilt from it (introspect.iter_round_trip)."""
        from . import introspect
        out = None
        for out in introspect.iter_round_trip(self.lab.adapter, self.plan, self.captures, block, token, inject, **kw):
            pass
        return out

    def score_explanation(self, block: int, token: int, text: str) -> dict:
        from . import introspect
        return introspect.score_text(self.lab.adapter, self.plan, self.captures, block, token, text)

    def save_cache(self, name: str = "") -> dict:
        """Save this run's KV cache to the library (usable as a prefix in later runs and in the app)."""
        from .caches import LIBRARY
        return LIBRARY.save(self.captures["cache"], name or f"run {self.id}",
                            {"model_id": self.lab.model_id, "source": "run", "schedule": self.plan.schedule})

    def __repr__(self):
        return f"<RunResult {self.id} {self.text[:60]!r}>"


class Lab:
    """A loaded model plus the tools to run and take it apart."""

    def __init__(self, model: str, device: str = "auto", dtype: str = "auto", policy: str = "auto",
                 keep_layers: int = 0, download: bool = True, keep_runs: int | None = 16):
        """model: a registry id, a folder on disk, or "hf:<org>/<repo>" for any Hugging Face repo.
        device / dtype: "auto" uses TOKENSHARK_DEVICE / TOKENSHARK_DTYPE or the best available; the choice stays on
        this Lab (nothing global is changed). keep_runs: how many finished runs lab.runs keeps (oldest dropped
        first; None = all). Each run holds its captures, which can be gigabytes at long contexts."""
        from .registry import get_spec
        from .weights import resolve_model_dir
        self.device = pick_device(None if device == "auto" else device)
        self.dtype = pick_dtype(self.device, None if dtype == "auto" else dtype)
        self.keep_runs = keep_runs
        p = Path(model).expanduser()
        if p.is_dir():                     # a folder on disk
            spec = {"id": p.name, "path": str(p), "family": "t2i" if (p / "model_index.json").exists() else "causal_lm"}
        else:
            spec = get_spec(model)
        self.spec, self.model_id = spec, spec["id"]
        root = resolve_model_dir(spec, download)
        self.plugin = find_plugin_for(root, spec)
        self.adapter = self.plugin.create_adapter(spec, root, self.device, self.dtype, policy, keep_layers)
        self.info = {**self.adapter.info(), "model_id": self.model_id}
        self.family = self.info["family"]
        self.runs: dict[str, RunResult] = {}
        self._lora: LoRAOverlay | None = None
        self._quant: QuantOverlay | None = None

    # ---------------------------------------------------------------- flows and runs
    def flow(self, prompt: str | None = None, **kw) -> FlowBuilder:
        """The model's default flow. Keyword arguments go to the prompt box (system, prefill, chat, thinking,
        images, literal_content, add_bos) or the sampler (max_new_tokens, min_new_tokens, ignore_eos, temperature,
        top_k, top_p, seed). Set max_new_tokens=0 for a single prompt forward pass without a decode step."""
        fb = FlowBuilder(self.plugin.default_flow(self.info))
        pk = {k: kw.pop(k) for k in ("system", "prefill", "chat", "thinking", "negative_prompt", "history", "images",
                                     "literal_content", "add_bos") if k in kw}
        if prompt is not None or pk:
            fb.prompt(prompt=prompt, **pk)
        if kw:
            fb.sampler(**kw)
        return fb

    def plan(self, flow):
        return compile_flow(flow.to_dict() if isinstance(flow, FlowBuilder) else flow, self.info)

    @torch.no_grad()
    def run(self, flow_or_prompt=None, keep: bool = True, **kw) -> RunResult:
        """Run a flow (FlowBuilder or flow dict) or just a prompt. Returns a RunResult.

        Keyword arguments as for lab.flow (system, max_new_tokens, min_new_tokens, ignore_eos, temperature, ...;
        max_new_tokens=0 runs a single prompt forward pass without decoding), plus what the run records
        (FlowBuilder.capture): attn_budget_mb, neuron_budget_mb, parts_budget_mb, expert_budget_mb, attn_layers,
        attn_queries. These also apply to a FlowBuilder or flow dict without changing it.
        keep=False: don't add the run to lab.runs (loops over many runs then hold no captures after each result is
        dropped). Notes about the run (such as attention that was not recorded) are issued as warnings."""
        from .flow import CAPTURE_KEYS
        cap = {k: kw.pop(k) for k in CAPTURE_KEYS if k in kw}
        if isinstance(flow_or_prompt, (FlowBuilder, dict)):
            flow = flow_or_prompt.to_dict() if isinstance(flow_or_prompt, FlowBuilder) else flow_or_prompt
            if cap or kw:                                               # on a copy: the caller's flow is unchanged
                fb = FlowBuilder(flow)
                pk = {k: kw.pop(k) for k in ("prompt", "system", "prefill", "chat", "thinking", "negative_prompt",
                                             "history", "images", "literal_content", "add_bos") if k in kw}
                if pk:
                    fb.prompt(**pk)
                if kw:
                    fb.sampler(**kw)
                flow = (fb.capture(**cap) if cap else fb).to_dict()
        else:
            fb = self.flow(flow_or_prompt, **kw)
            flow = (fb.capture(**cap) if cap else fb).to_dict()
        plan = compile_flow(flow, self.info)
        if self.family != "causal_lm":
            res = self.adapter.run(plan)
        else:
            prefix = interp.load_prefixes(plan, self.adapter, self._state()) if plan.params.get("prefix_cache") else None
            sources = interp.resolve_sources(plan, self._captures_of)
            plan.sources = sources                       # kept with the plan for analyses that run the flow again
            st = self.adapter.run_stepper(plan, prefix)
            st.ctx.sources = sources
            while st.step_token():
                pass
            res = st.finish()
        r = RunResult(self, plan, res, flow)
        for note in r.notes:
            warnings.warn(note, RuntimeWarning, stacklevel=2)
        if keep:
            self._remember(r)
        return r

    @torch.no_grad()
    def run_batch(self, items: list, keep: bool = False, **kw) -> list[RunResult]:
        """Run a batch of prompts, FlowBuilders, or flow dicts in lockstep one block at a time.
        Each block's weights are streamed to the compute device once across the entire batch (`W.hold_device()`),
        making multi-prompt and multi-intervention sweeps orders of magnitude faster under cpu/disk offloading."""
        if not items:
            return []
        if self.family != "causal_lm" or not hasattr(self.adapter, "run_batch"):
            return [self.run(it, keep=keep, **kw) for it in items]
        from .flow import CAPTURE_KEYS
        flows, plans, prefixes, sources_list = [], [], [], []
        for it in items:
            item_kw = dict(kw)
            cap = {k: item_kw.pop(k) for k in CAPTURE_KEYS if k in item_kw}
            if isinstance(it, (FlowBuilder, dict)):
                fl = it.to_dict() if isinstance(it, FlowBuilder) else it
                if cap or item_kw:
                    fb = FlowBuilder(fl)
                    pk = {k: item_kw.pop(k) for k in ("prompt", "system", "prefill", "chat", "thinking",
                                                      "negative_prompt", "history", "images", "literal_content",
                                                      "add_bos") if k in item_kw}
                    if pk:
                        fb.prompt(**pk)
                    if item_kw:
                        fb.sampler(**item_kw)
                    fl = (fb.capture(**cap) if cap else fb).to_dict()
            else:
                fb = self.flow(it, **item_kw)
                fl = (fb.capture(**cap) if cap else fb).to_dict()
            pl = compile_flow(fl, self.info)
            pr = interp.load_prefixes(pl, self.adapter, self._state()) if pl.params.get("prefix_cache") else None
            src = interp.resolve_sources(pl, self._captures_of)
            pl.sources = src
            flows.append(fl)
            plans.append(pl)
            prefixes.append(pr)
            sources_list.append(src)
        raw_results = self.adapter.run_batch(plans, prefixes=prefixes, sources_list=sources_list)
        out: list[RunResult] = []
        for pl, res, fl in zip(plans, raw_results, flows):
            r = RunResult(self, pl, res, fl)
            for note in r.notes:
                warnings.warn(note, RuntimeWarning, stacklevel=2)
            if keep:
                self._remember(r)
            out.append(r)
        return out

    # ---------------------------------------------------------------- the runs this Lab keeps
    def _remember(self, r: RunResult) -> None:
        self.runs[r.id] = r
        if self.keep_runs is not None:
            while len(self.runs) > max(0, self.keep_runs):
                del self.runs[next(iter(self.runs))]          # oldest first

    def _captures_of(self, rid: str) -> dict:
        if rid not in self.runs:
            raise KeyError(f"Run {rid} is not kept by this Lab any more (keep_runs={self.keep_runs}, or it was made "
                           f"with keep=False or forgotten). Run it again, or raise Lab(keep_runs=...).")
        self.runs[rid] = self.runs.pop(rid)                   # used again: now the newest, so dropped last
        return self.runs[rid].captures

    def forget(self, run=None) -> None:
        """Drop one run (RunResult or id) from lab.runs, or all of them (run=None), freeing their captures."""
        if run is None:
            self.runs.clear()
        else:
            self.runs.pop(run.id if isinstance(run, RunResult) else run, None)

    def stepper(self, flow=None, **kw):
        """The run loop one block at a time: st.step_block(), st.step_token(), st.force(token_id), st.state()."""
        fb = flow if flow is not None else self.flow(**kw)
        plan = self.plan(fb)
        prefix = interp.load_prefixes(plan, self.adapter, self._state()) if plan.params.get("prefix_cache") else None
        st = self.adapter.stepper(plan, prefix)
        st.ctx.sources = interp.resolve_sources(plan, self._captures_of)
        return st

    @torch.no_grad()
    def bench(self, prompt_tokens: int = 512, new_tokens: int = 32, capture: bool = False, repeat: int = 1) -> dict:
        """Speed of this model on this machine: prefill (the prompt in one pass) and decode (one token per pass)
        in tokens/s, peak GPU memory, and GB read from disk per decoded token (weights streamed for layers not
        kept in memory). capture=False turns recording off, so the numbers are the engine's own cost."""
        import time
        if self.family != "causal_lm":
            raise NotImplementedError("bench() measures token generation (causal_lm models).")
        tok = self.adapter.tok
        ids = (tok.encode(" the quick brown fox jumps over the lazy dog") * (prompt_tokens // 9 + 1))[:prompt_tokens]
        off = {} if capture else {k: 0 for k in ("attn_budget_mb", "neuron_budget_mb", "parts_budget_mb",
                                                 "expert_budget_mb")}
        fb = self.flow(tok.decode(ids), max_new_tokens=new_tokens, temperature=0)
        if off:
            fb.capture(**off)
        plan = self.plan(fb)
        plan.params["ignore_eos"] = True
        cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        sync = torch.cuda.synchronize if cuda else (lambda: None)
        W = self.adapter.model.W
        runs = []
        for _ in range(max(1, repeat)):
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            st = self.adapter.run_stepper(plan, None)
            b0 = W.bytes_read
            sync(); t0 = time.perf_counter()
            st.step_token()                                   # the prompt: one forward pass
            sync(); t1 = time.perf_counter()
            b1 = W.bytes_read
            n = 0
            while st.step_token():
                n += 1
            sync(); t2 = time.perf_counter()
            runs.append({"prefill_s": t1 - t0, "decode_s": t2 - t1, "n_decode": n, "n_prompt": st.n_in,
                         "gb_prefill": (b1 - b0) / 2**30, "gb_decode": (W.bytes_read - b1) / 2**30,
                         "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None})
            del st
        best = min(runs, key=lambda r: r["prefill_s"] + r["decode_s"])
        return {"model": self.model_id, "device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "prompt_tokens": best["n_prompt"], "new_tokens": best["n_decode"],
                "prefill_tok_s": round(best["n_prompt"] / max(best["prefill_s"], 1e-9), 1),
                "decode_tok_s": round(best["n_decode"] / max(best["decode_s"], 1e-9), 2),
                "time_to_first_token_s": round(best["prefill_s"], 3),
                "gb_read_prefill": round(best["gb_prefill"], 2),
                "gb_read_per_token": round(best["gb_decode"] / max(best["n_decode"], 1), 3),
                "peak_gpu_gb": round(best["peak_gb"], 2) if best["peak_gb"] is not None else None,
                "capture": capture, "weights": W.describe()}

    # ---------------------------------------------------------------- tokens
    def tokenize(self, text: str, chat: bool = False, parse_special: bool = True, add_bos: bool | None = None) -> dict:
        """parse_special=False: control-token text in `text` (e.g. "<|endoftext|>") is read as plain characters."""
        want_bos = (not chat and bool(getattr(self.adapter.tok, "add_bos", False))) if add_bos is None else bool(add_bos)
        if not parse_special and not chat:
            ids = self.adapter.tok.encode(text, add_special=want_bos, parse_special=False)
            return {"ids": ids, "pieces": [self.adapter.tok.piece(i) for i in ids]}
        return interp.tokenize(self.adapter, text, chat, add_bos=want_bos, literal_content=not parse_special)

    def find_span(self, prompt: str, substring: str, chat: bool = True, role: str | None = None,
                  which: int = -1, **kw) -> range:
        """Token range(start, end) covering `substring` inside `prompt` as tokenized by this model."""
        fb = self.flow(prompt, chat=chat, **kw)
        plan = self.plan(fb)
        return self.adapter.find_span(plan.params, substring, role=role, which=which)

    def decode(self, ids: list[int]) -> str:
        return self.adapter.tok.decode(ids)

    def neighbors(self, word_or_id, k: int = 12) -> dict:
        tid = word_or_id if isinstance(word_or_id, int) else self.adapter._word_ids(word_or_id)[0][0]
        return interp.token_neighbors(self.adapter, tid, k)

    # ---------------------------------------------------------------- steering vectors
    def vector_from_runs(self, a: RunResult, b: RunResult, block: int, hook: str = "resid_post", token="last",
                         name: str = "", persist: bool = True) -> dict:
        """a − b at one block, saved to the vector library (or kept in memory with persist=False).
        Use it with flow.steer(block, vector, value)."""
        v = interp.vector_from_runs(a.captures, b.captures, block, hook, token)
        return interp.VECTORS.save(v, name or f"{a.id} − {b.id} @ {block}",
                                   {"model_id": self.model_id, "source": "runs", "hook": hook, "pos": block, "token": str(token)},
                                   persist=persist)

    def vector_from_word(self, word: str, minus: str = "", name: str = "", persist: bool = True) -> dict:
        v, meta = interp.vector_from_word(self.adapter, word, minus)
        return interp.VECTORS.save(v, name or (f"“{meta['word']}”" + (f" − “{meta['minus']}”" if minus else "")),
                                   {"model_id": self.model_id, "source": "word", **meta},
                                   persist=persist)

    def concept_vector(self, word: str, block: int, name: str = "", flow=None, persist: bool = True) -> dict:
        """Activation on "Tell me about <word>." minus the average over unrelated words, at one block; saved to
        the vector library (introspect.concept_vectors)."""
        from . import introspect
        cv = introspect.concept_vectors(self.adapter, self.plan(flow or self.flow("x")), [word])
        return interp.VECTORS.save(cv["vectors"][word][block], name or f"concept: {word} @#{block}",
                                   {"model_id": self.model_id, "source": "concept", "word": word, "pos": block,
                                    "hook": "resid_post", "resid_norm": round(cv["resid_norm"][block], 3)},
                                   persist=persist)

    @torch.no_grad()
    def inject(self, concepts: list[str], blocks: list[int], strengths: list[float] = (0.1, 0.2, 0.4),
               prompt: str | None = None, max_new_tokens: int = 40, flow=None) -> dict:
        """Concept-injection experiment: add each concept's vector at each block and strength while the model
        answers "do you detect an injected thought?", and grade the answers against a control trial."""
        from . import introspect
        fb = flow or self.flow(prompt or introspect.INJECTION_PROMPT, chat=True)
        out = None
        for out in introspect.iter_injection(self.adapter, self.plan(fb), list(concepts), list(blocks), list(strengths),
                                             max_new_tokens):
            pass
        return out

    @torch.no_grad()
    def expert_sweep(self, flow_or_prompt, block: int = -1, mode: str = "ban", **kw) -> dict:
        """Remove one expert at a time (of one MoE block, or of all with block=-1) and measure the damage to the
        next-token prediction (moe.iter_ablation)."""
        from . import moe
        fb = flow_or_prompt if isinstance(flow_or_prompt, (FlowBuilder, dict)) else self.flow(flow_or_prompt, **kw)
        out = None
        plan = self.plan(fb)
        plan.sources = interp.resolve_sources(plan, self._captures_of)
        for out in moe.iter_ablation(self.adapter, plan, block, None, mode):
            pass
        return out

    def router(self, layer: int) -> dict:
        """The router matrix of one MoE layer: similarity between experts' rows and the tokens each row attracts."""
        from . import moe
        return moe.router_rows(self.adapter, layer)

    # ---------------------------------------------------------------- A/B neuron identification & wiring
    @torch.no_grad()
    def hunt_neurons_ab(self, set_a: list, set_b: list, block: int = -1, span: str = "reply",
                        max_new_tokens: int = 14, k: int = 15, strict: bool = True,
                        system: str | None = None, chat: bool | None = None, prefill: str | None = None,
                        flow=None) -> dict:
        """Find neurons that differ the most between Set A and Set B conversations (A/B testing).
        Each item in `set_a` / `set_b` can be a prompt string, a conversation dict, or a `RunResult`.
        """
        plan = self.plan(flow or self.flow("x", system=system or "", chat=True if chat is None else chat))
        if all(isinstance(x, RunResult) for x in set_a) and all(isinstance(x, RunResult) for x in set_b):
            acts_a = [{"prompt": r.plan.params.get("prompt", ""), "text": r.text,
                       "n_in": r.captures.get("n_in", 1), "neurons": r.captures.get("neurons", {}), "run_id": r.id}
                      for r in set_a]
            acts_b = [{"prompt": r.plan.params.get("prompt", ""), "text": r.text,
                       "n_in": r.captures.get("n_in", 1), "neurons": r.captures.get("neurons", {}), "run_id": r.id}
                      for r in set_b]
            return interp.compare_neurons_from_acts(self.adapter, plan.schedule, acts_a, acts_b,
                                                    pos=block, span=span, k=k, strict=strict)
        out = None
        for out in interp.iter_neuron_ab_hunt(
            self.adapter, plan, set_a, set_b, pos=block, span=span,
            max_new_tokens=max_new_tokens, k=k, strict=strict,
            system=system, chat=chat, prefill=prefill,
        ):
            pass
        return out

    @torch.no_grad()
    def search_neurons_by_words(self, promote: list[str] | str, suppress: list[str] | str | None = None,
                                block: int = -1, k: int = 10, use_norm: bool = False, flow=None) -> dict:
        """Find writer neurons whose `down_proj` column promotes target words over suppressed words."""
        plan = self.plan(flow or self.flow("x"))
        return interp.search_neurons_by_words(self.adapter, plan.schedule, promote, suppress,
                                              pos=block, k=k, use_norm=use_norm)

    @torch.no_grad()
    def batch_test_neurons(self, flow, test_prompts: list, control_prompts: list | None = None,
                           max_new_tokens: int = 40, keyword_regex: str = r"(can't|cannot|can not|unable|not able)",
                           system: str | None = None, chat: bool | None = None, prefill: str | None = None) -> dict:
        """Run `test_prompts` and `control_prompts` before and after the flow's neuron edits/wires."""
        plan = self.plan(flow)
        plan.sources = interp.resolve_sources(plan, self._captures_of)
        out = None
        for out in interp.iter_neuron_batch_test(
            self.adapter, plan, test_prompts, control_prompts,
            max_new_tokens=max_new_tokens, keyword_regex=keyword_regex,
            system=system, chat=chat, prefill=prefill,
        ):
            pass
        return out

    def save_vector(self, v: torch.Tensor, name: str = "", persist: bool = True) -> dict:
        return interp.VECTORS.save(v, name, {"model_id": self.model_id, "source": "python"}, persist=persist)

    def load_vector(self, vector) -> torch.Tensor:
        if isinstance(vector, torch.Tensor):
            return vector
        return interp.VECTORS.load(vector["id"] if isinstance(vector, dict) else vector)[0]

    # ---------------------------------------------------------------- weights: LoRA and quantization
    def lora(self, path: str | None, scale: float = 1.0, layers: str = "all") -> dict | None:
        """Attach a LoRA / fine-tuned checkpoint (None = detach)."""
        stores = self.plugin.weight_stores(self.adapter)
        for st in stores.values():
            if st is not None:
                st.set_lora_hook(None)
        from .adapt import apply_extras, clear_extras
        self._lora = None
        clear_extras(self.adapter)
        if path is not None:
            self._lora = LoRAOverlay(path, scale, layers=layers)
            out = self._lora.bind(stores)
            out["adaptation"] = apply_extras(self.adapter, self._lora)       # new blocks / adapter layers, if any
        self.info = {**self.adapter.info(), "model_id": self.model_id}
        return out if path is not None else None

    def quantize(self, bits: int | None, measure: bool = False, layers: str = "all",
                 layer_bits: dict[int, int] | None = None) -> dict | None:
        """Round the block weights to `bits` bits (None = back to the stored weights)."""
        stores = self.plugin.weight_stores(self.adapter)
        for st in stores.values():
            if st is not None:
                st._quant = None
        self._quant = None
        if not bits and not layer_bits:
            return None
        self._quant = QuantOverlay(bits or 4, layers=layers, layer_bits=layer_bits)
        if measure:
            return self._quant.bind(stores)
        for st in stores.values():
            if st is not None:
                st._quant = self._quant
        return self._quant.summary()

    def _state(self) -> str:
        return f"{getattr(self._lora, 'path', '')}:{getattr(self._lora, 'scale', '')}:{getattr(self._quant, 'bits', '')}"

    # ---------------------------------------------------------------- box code
    def block_code(self, layer: int = 0, kind: str = "block") -> str:
        """The simple Python a box runs (edit it and pass it to flow.code(...))."""
        d = default_code(kind, self.adapter.model, layer)
        if not d["editable"]:
            raise ValueError(d["reason"])
        return d["code"]

    # ---------------------------------------------------------------- guardrail fork & attention traceback
    @torch.no_grad()
    def train_guardrail_fork(self, preset: str = "powershell_detector",
                             categories: dict[str, list[str]] | None = None,
                             block: int = -1, head_type: str = "linear_probe",
                             steps: int = 80, lr: float = 0.05, name: str = "powershell_detector",
                             trigger_category: str = "", threshold: float = 0.65,
                             halt_message: str = "", pooling: str = "mean") -> dict:
        """Train a center-layer guardrail fork classifier (`linear_probe`, `mlp_guard`, or `block_branch`)."""
        from .extensions.guardrail_fork import PRESETS, train_fork_classifier
        cats = categories
        trig = trigger_category
        h_msg = halt_message
        if preset and preset in PRESETS and not cats:
            p = PRESETS[preset]
            cats = p["categories"]
            trig = trig or p["trigger_category"]
            h_msg = h_msg or p["halt_message"]
        if not cats:
            raise ValueError("Provide `categories` or a valid `preset` (e.g. 'powershell_detector').")
        return train_fork_classifier(
            self.adapter,
            categories=cats,
            pos=block if block >= 0 else None,
            head_type=head_type,
            steps=steps,
            lr=lr,
            name=name,
            trigger_category=trig,
            threshold=threshold,
            halt_message=h_msg,
            pooling=pooling,
        )

    def __repr__(self):
        return f"<Lab {self.model_id} on {self.device}>"
