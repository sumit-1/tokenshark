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
    def tokens(self) -> list[str]:
        return self.outputs.get("tokens", [])

    @property
    def token_ids(self) -> list[int]:
        return self.outputs.get("token_ids", [])

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

    def attention(self, block: int, head: int | None = None) -> torch.Tensor:
        """Attention pattern [heads, tokens, tokens] (or [tokens, tokens] for one head)."""
        a = self.captures["attn"][block].float()
        return a if head is None else a[head]

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

    def kv_analysis(self) -> dict:
        """Analyze this run's KV cache: GQA memory savings, per-block & per-token K/V norms, and attention sinks."""
        return interp.kv_analysis(self.lab.adapter, self.captures, self.plan.schedule, self.plan.schedule_nodes)

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
                 keep_layers: int = 0, download: bool = True):
        import os
        from .registry import get_spec
        from .weights import resolve_model_dir
        if device != "auto":
            os.environ["TOKENSHARK_DEVICE"] = device
        if dtype != "auto":
            os.environ["TOKENSHARK_DTYPE"] = dtype
        self.device = pick_device()
        self.dtype = pick_dtype(self.device)
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
        """The model's default flow. Keyword arguments go to the prompt box (system, prefill, chat) or the sampler
        (max_new_tokens, temperature, top_k, top_p, seed)."""
        fb = FlowBuilder(self.plugin.default_flow(self.info))
        pk = {k: kw.pop(k) for k in ("system", "prefill", "chat", "thinking", "negative_prompt", "history") if k in kw}
        if prompt is not None or pk:
            fb.prompt(prompt=prompt, **pk)
        if kw:
            fb.sampler(**kw)
        return fb

    def plan(self, flow):
        return compile_flow(flow.to_dict() if isinstance(flow, FlowBuilder) else flow, self.info)

    @torch.no_grad()
    def run(self, flow_or_prompt=None, **kw) -> RunResult:
        """Run a flow (FlowBuilder or flow dict) or just a prompt. Returns a RunResult."""
        if isinstance(flow_or_prompt, (FlowBuilder, dict)):
            flow = flow_or_prompt.to_dict() if isinstance(flow_or_prompt, FlowBuilder) else flow_or_prompt
        else:
            flow = self.flow(flow_or_prompt, **kw).to_dict()
        plan = compile_flow(flow, self.info)
        if self.family != "causal_lm":
            res = self.adapter.run(plan)
        else:
            prefix = interp.load_prefixes(plan, self.adapter, self._state()) if plan.params.get("prefix_cache") else None
            sources = interp.resolve_sources(plan, lambda rid: self.runs[rid].captures)
            plan.sources = sources                       # kept with the plan for analyses that run the flow again
            st = self.adapter.run_stepper(plan, prefix)
            st.ctx.sources = sources
            while st.step_token():
                pass
            res = st.finish()
        r = RunResult(self, plan, res, flow)
        self.runs[r.id] = r
        return r

    def stepper(self, flow=None, **kw):
        """The run loop one block at a time: st.step_block(), st.step_token(), st.force(token_id), st.state()."""
        fb = flow if flow is not None else self.flow(**kw)
        plan = self.plan(fb)
        prefix = interp.load_prefixes(plan, self.adapter, self._state()) if plan.params.get("prefix_cache") else None
        st = self.adapter.stepper(plan, prefix)
        st.ctx.sources = interp.resolve_sources(plan, lambda rid: self.runs[rid].captures)
        return st

    # ---------------------------------------------------------------- tokens
    def tokenize(self, text: str, chat: bool = False) -> dict:
        return interp.tokenize(self.adapter, text, chat)

    def decode(self, ids: list[int]) -> str:
        return self.adapter.tok.decode(ids)

    def neighbors(self, word_or_id, k: int = 12) -> dict:
        tid = word_or_id if isinstance(word_or_id, int) else self.adapter._word_ids(word_or_id)[0][0]
        return interp.token_neighbors(self.adapter, tid, k)

    # ---------------------------------------------------------------- steering vectors
    def vector_from_runs(self, a: RunResult, b: RunResult, block: int, hook: str = "resid_post", token="last",
                         name: str = "") -> dict:
        """a − b at one block, saved to the vector library. Use it with flow.steer(block, vector, value)."""
        v = interp.vector_from_runs(a.captures, b.captures, block, hook, token)
        return interp.VECTORS.save(v, name or f"{a.id} − {b.id} @ {block}",
                                   {"model_id": self.model_id, "source": "runs", "hook": hook, "pos": block, "token": str(token)})

    def vector_from_word(self, word: str, minus: str = "", name: str = "") -> dict:
        v, meta = interp.vector_from_word(self.adapter, word, minus)
        return interp.VECTORS.save(v, name or (f"“{meta['word']}”" + (f" − “{meta['minus']}”" if minus else "")),
                                   {"model_id": self.model_id, "source": "word", **meta})

    def concept_vector(self, word: str, block: int, name: str = "", flow=None) -> dict:
        """Activation on "Tell me about <word>." minus the average over unrelated words, at one block; saved to
        the vector library (introspect.concept_vectors)."""
        from . import introspect
        cv = introspect.concept_vectors(self.adapter, self.plan(flow or self.flow("x")), [word])
        return interp.VECTORS.save(cv["vectors"][word][block], name or f"concept: {word} @#{block}",
                                   {"model_id": self.model_id, "source": "concept", "word": word, "pos": block,
                                    "hook": "resid_post", "resid_norm": round(cv["resid_norm"][block], 3)})

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
        plan.sources = interp.resolve_sources(plan, lambda rid: self.runs[rid].captures)
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
        plan.sources = interp.resolve_sources(plan, lambda rid: self.runs[rid].captures)
        out = None
        for out in interp.iter_neuron_batch_test(
            self.adapter, plan, test_prompts, control_prompts,
            max_new_tokens=max_new_tokens, keyword_regex=keyword_regex,
            system=system, chat=chat, prefill=prefill,
        ):
            pass
        return out

    def save_vector(self, v: torch.Tensor, name: str = "") -> dict:
        return interp.VECTORS.save(v, name, {"model_id": self.model_id, "source": "python"})

    def load_vector(self, vector) -> torch.Tensor:
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
