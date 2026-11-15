"""HTTP routes for the lab tools: activation patching, steering vectors, neuron hunting, tokenizer explorer,
sampler playground, quantization, training, editable box code and run → script export.

Kept apart from server.py (models, flows, runs, debugger) so each file stays readable. The logic lives in
interp.py / quant.py / train.py / blockcode.py / export.py; this file only checks input and calls it.
"""
from __future__ import annotations

import re

from fastapi import HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import blockcode, interp, train
from .flow import FlowError, compile_flow
from .registry import LOADED
from .runs import DATA, STORE
from .server import _caps, _load_prefixes, _need, _require_model, _run, _sources, app
from .tasks import TASKS
from .weights import models_dir


def _llm():
    return _need("lm_tools", "This tool")


def _llm_run(run_id: str):
    r = _run(run_id)
    cap = _caps(r)
    if r.plan.family != "causal_lm":
        raise HTTPException(400, "This tool is for language-model runs.")
    return r, cap


def _guard(fn, *a, **kw):
    """Call engine code under the model lock; turn its ValueErrors into readable 400s."""
    try:
        with LOADED.lock:
            return fn(*a, **kw)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'\""))


def _task(kind: str, label: str, group: str, gen_fn, weights=None) -> dict:
    """Start a generator as a task (one unit of work per yield). The client polls GET /api/tasks/{id}."""
    t = TASKS.new(kind, label, group=group)
    TASKS.start(t, gen_fn, weights=weights)
    return {"task": t.summary(), "result": None, "status": t.status}


def _compile(flow: dict, info: dict):
    try:
        return compile_flow(flow, info)
    except FlowError as e:
        raise HTTPException(422, " · ".join(x["message"] for x in e.errors))


# ---------------------------------------------------------------- editable box code
class CodeReq(BaseModel):
    code: str
    kind: str = "block"


@app.get("/api/code/default")
def code_default(kind: str = "block", layer: int = 0):
    """The simple Python a box runs when nobody has edited it."""
    adapter, _ = _llm()
    if kind == "block" and not 0 <= layer < adapter.cfg.n_layers:
        raise HTTPException(400, "No such layer.")
    return {**blockcode.default_code(kind, adapter.model, layer), "kind": kind, "layer": layer,
            "help": blockcode.__doc__}


@app.post("/api/code/check")
def code_check(req: CodeReq):
    problems = blockcode.check(req.code)
    return {"ok": not problems, "problems": problems}


# ---------------------------------------------------------------- run → script
@app.get("/api/runs/{run_id}/script", response_class=PlainTextResponse)
def run_script(run_id: str):
    """This run as a Python script that reproduces it with tokenshark.api."""
    from .export import run_script as make
    r = _run(run_id)
    _, info = _require_model()
    if LOADED.spec["id"] != r.model_id:
        raise HTTPException(409, "Load this run's model to export it (the script is written against its default flow).")

    def default_for(_r):
        return LOADED.plugin.default_flow(info)

    def get(rid):
        try:
            return STORE.get(rid)
        except KeyError:
            return None
    return make(r, default_for, get, {v["id"]: v for v in interp.VECTORS.list()})


# ---------------------------------------------------------------- activation patching
class SweepReq(BaseModel):
    flow: dict
    source_run: str
    hook: str = "resid_post"
    tokens: str = "all"


@app.post("/api/patch/sweep")
def patch_sweep(req: SweepReq):
    """Patch one (block, token) at a time from the source run into this flow's prompt; see interp.iter_patch_sweep."""
    adapter, info = _llm()
    _, cap = _llm_run(req.source_run)
    if req.hook not in interp.PARTS:
        raise HTTPException(400, f"hook must be one of {', '.join(interp.PARTS)}")
    plan = _compile(req.flow, info)
    prefix = _load_prefixes(plan, adapter) if plan.params.get("prefix_cache") else None
    return _task("analysis", f"patching sweep ({req.hook})", "sweep",
                 lambda: interp.iter_patch_sweep(adapter, plan, cap, req.hook, req.tokens, prefix),
                 weights=adapter.model.W)


# ---------------------------------------------------------------- steering vectors
class VecRunsReq(BaseModel):
    run_a: str
    run_b: str
    pos: int
    hook: str = "resid_post"
    token: str = "last"
    name: str = ""


class VecWordReq(BaseModel):
    word: str
    minus: str = ""
    name: str = ""


@app.get("/api/vectors")
def list_vectors():
    return interp.VECTORS.list()


@app.post("/api/vectors/from_runs")
def vector_from_runs(req: VecRunsReq):
    """A − B at one block: the direction that separates two runs (e.g. a happy and a sad prompt)."""
    _llm()
    (ra, ca), (rb, cb) = _llm_run(req.run_a), _llm_run(req.run_b)
    v = _guard(interp.vector_from_runs, ca, cb, req.pos, req.hook, req.token)
    return interp.VECTORS.save(v, req.name or f"{ra.label or ra.id} − {rb.label or rb.id} @ #{req.pos}",
                               {"model_id": ra.model_id, "source": "runs", "run_a": ra.id, "run_b": rb.id,
                                "hook": req.hook, "pos": req.pos, "token": req.token})


@app.post("/api/vectors/from_word")
def vector_from_word(req: VecWordReq):
    """The output direction of a word (its unembedding row, length 1), optionally minus another word's."""
    adapter, _ = _llm()
    v, meta = _guard(interp.vector_from_word, adapter, req.word, req.minus)
    name = req.name or (f"“{meta['word']}”" + (f" − “{meta['minus']}”" if meta.get("minus") else ""))
    return interp.VECTORS.save(v, name, {"model_id": LOADED.spec["id"], "source": "word", **meta})


@app.delete("/api/vectors/{vid}")
def delete_vector(vid: str):
    try:
        interp.VECTORS.delete(vid)
    except KeyError:
        raise HTTPException(404, "Unknown vector.")
    return {"ok": True}


# ---------------------------------------------------------------- neurons
@app.get("/api/runs/{run_id}/neurons")
def neurons(run_id: str, pos: int = 0, token: int = -1, k: int = 20, by: str = "token"):
    r, cap = _llm_run(run_id)
    return _guard(interp.top_neurons, LOADED.adapter, cap, r.plan.schedule, pos, token, max(1, min(k, 100)),
                  "max" if by == "max" else "token")


@app.get("/api/runs/{run_id}/neuron")
def neuron(run_id: str, pos: int, index: int):
    r, cap = _llm_run(run_id)
    return _guard(interp.neuron_detail, LOADED.adapter, cap, r.plan.schedule, pos, index)


# ---------------------------------------------------------------- sampler playground
@app.get("/api/runs/{run_id}/logits")
def logits(run_id: str, position: int = -1, top: int = 1000):
    """The raw scores behind one prediction (top N), for trying temperature / top-k / top-p in the browser."""
    r, cap = _llm_run(run_id)
    return _guard(interp.final_logits, LOADED.adapter, cap, r.plan.schedule, position, max(10, min(top, 5000)))


# ---------------------------------------------------------------- tokenizer explorer
class TokenizeReq(BaseModel):
    text: str
    chat: bool = False
    system: str = ""


@app.post("/api/tokenize")
def tokenize(req: TokenizeReq):
    adapter, _ = _llm()
    try:
        return interp.tokenize(adapter, req.text, req.chat, req.system)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/tokens/search")
def token_search(q: str = "", limit: int = 80):
    adapter, _ = _llm()
    return interp.vocab_search(adapter, q, max(1, min(limit, 500)))


@app.get("/api/tokens/{token_id}/neighbors")
def token_neighbors(token_id: int, k: int = 12):
    adapter, _ = _llm()
    return _guard(interp.token_neighbors, adapter, token_id, max(1, min(k, 50)))


# ---------------------------------------------------------------- quantization
class QuantReq(BaseModel):
    bits: int = 4
    layers: str = "all"
    layer_bits: dict[int, int] | None = None


class QuantCompareReq(BaseModel):
    flow: dict
    bits: list[int] = [8, 4]


@app.get("/api/quant")
def get_quant():
    _require_model()
    return LOADED.quant.summary() if LOADED.quant is not None else {"active": False}


@app.post("/api/quant")
def set_quant(req: QuantReq):
    """Round the block weights to `bits` bits on the fly (see quant.py). Every later run uses the rounded weights."""
    _require_model()
    return _guard(LOADED.set_quant, req.bits, True, req.layers, req.layer_bits)


@app.delete("/api/quant")
def clear_quant():
    _require_model()
    return _guard(LOADED.set_quant, None)


@app.post("/api/quant/compare")
def quant_compare(req: QuantCompareReq):
    from .quant import iter_compare
    adapter, info = _llm()
    plan = _compile(req.flow, info)
    bits = [b for b in dict.fromkeys(req.bits) if 1 <= b <= 16][:6]
    if not bits:
        raise HTTPException(400, "Choose at least one bit width between 1 and 16.")
    prefix = _load_prefixes(plan, adapter) if plan.params.get("prefix_cache") else None
    sources = _sources(plan)
    return _task("analysis", f"quantization: full vs {', '.join(f'{b}-bit' for b in bits)}", "quant",
                 lambda: iter_compare(LOADED, plan, bits, prefix, sources))


# ---------------------------------------------------------------- training
class TinyReq(BaseModel):
    name: str = "my-tiny-model"
    text: str = ""
    corpus: str = "mixed"            # mixed | facts | counting  (used when text is empty)
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    context: int = 64
    vocab: int = 300
    steps: int = 1500
    batch: int = 32
    lr: float = 3e-3
    seed: int = 0
    sample_prompt: str = "The capital of France is"
    n_experts: int = 0               # > 0: a mixture-of-experts model (each block: this many small MLPs + a router)
    experts_per_token: int = 2
    balance: float = 0.01            # weight of the load-balancing loss


class LoRATrainReq(BaseModel):
    name: str = "my-lora"
    examples: list[str]
    rank: int = 8
    alpha: float = 16.0
    targets: list[str] = ["q_proj", "v_proj"]
    steps: int = 60
    lr: float = 1e-3
    max_tokens: int = 96
    seed: int = 0
    probe: str = ""
    layers: str = "all"              # which blocks get the add-on: all | 2 | 0:4 | 1,5
    method: str = "lora"             # lora | full | ia3 | norms | adapter | new_block   (see adapt.py)
    after: int = 0                   # new_block: insert after this block
    init: str = "duplicate"          # new_block: duplicate | average | identity | random


class ImageTrainReq(BaseModel):
    steps: int = 1200
    batch: int = 64
    lr: float = 3e-3
    seed: int = 0


class ImageSampleReq(BaseModel):
    color: str = "red"
    shape: str = "circle"
    steps: int = 12
    guidance: float = 2.0
    seed: int = 0
    grid: bool = False


def _safe_name(name: str, default: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-.") or default


def _busy(kind: str):
    j = train.JOBS.get(kind)
    if j is not None and j.status in ("queued", "running"):
        raise HTTPException(409, "A training job of this kind is already running. Stop it first.")


@app.post("/api/train/tiny")
def train_tiny(req: TinyReq):
    """Train a small transformer from scratch; it is saved into the models folder and can then be loaded."""
    _busy("tiny")
    name = _safe_name(req.name, "my-tiny-model")
    out = models_dir() / name
    if out.exists() and any(out.iterdir()):          # only ever overwrite a model that this tab trained
        cfg = out / "config.json"
        if not (cfg.exists() and '"trained_here": true' in cfg.read_text(errors="ignore")):
            raise HTTPException(400, f"A model folder named '{name}' already exists and was not trained here. "
                                     f"Choose another name.")
    if LOADED.spec and LOADED.spec["id"] == name:
        raise HTTPException(409, "That model is loaded right now. Unload it, or train under another name.")
    lim = dict(d_model=(16, 512), n_layers=(1, 12), n_heads=(1, 16), context=(8, 512), vocab=(260, 8000),
               steps=(1, 100_000), batch=(1, 256), n_experts=(0, 64), experts_per_token=(1, 64))
    for k, (lo, hi) in lim.items():
        if not lo <= getattr(req, k) <= hi:
            raise HTTPException(400, f"{k} must be between {lo} and {hi}.")
    try:
        job = train.TinyJob(name, out, req.text, req.corpus, req.d_model, req.n_layers, req.n_heads, req.context,
                            req.vocab, req.steps, req.batch, req.lr, req.seed, req.sample_prompt,
                            req.n_experts, req.experts_per_token, req.balance)
    except ValueError as e:
        raise HTTPException(400, str(e))
    train.JOBS["tiny"] = job.start()
    return job.summary()


@app.post("/api/train/lora")
def train_lora(req: LoRATrainReq):
    """Train a LoRA on the loaded language model; it is saved in PEFT format and attached when done."""
    _busy("lora")
    _llm()
    name = _safe_name(req.name, "my-lora")
    ok = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not req.targets or not set(req.targets) <= ok:
        raise HTTPException(400, f"targets must be chosen from {', '.join(sorted(ok))}.")
    if not (1 <= req.rank <= 256 and 1 <= req.steps <= 20_000 and 2 <= req.max_tokens <= 1024):
        raise HTTPException(400, "rank 1–256, steps 1–20000, max_tokens 2–1024.")
    from .adapt import METHODS, AdaptJob
    if req.method not in METHODS:
        raise HTTPException(400, f"method must be one of {', '.join(METHODS)}.")
    cfg = LOADED.adapter.cfg
    if getattr(cfg, "base_layers", cfg.n_layers) != cfg.n_layers:
        raise HTTPException(409, "A trained block is attached to this model. Detach it in the LoRA / Fine-tune tab "
                                 "before training something new: training always starts from the base model.")
    try:
        kw = dict(rank=req.rank, alpha=req.alpha, targets=tuple(req.targets), steps=req.steps, lr=req.lr,
                  max_tokens=req.max_tokens, seed=req.seed, probe=req.probe, layers=req.layers)
        if req.method == "lora":
            job = train.LoRAJob(LOADED, TASKS, DATA / "loras" / name, name, req.examples, **kw)
        else:
            job = AdaptJob(LOADED, TASKS, DATA / "loras" / name, name, req.examples, method=req.method,
                           after=req.after, init=req.init, **kw)
        from .flow import parse_index_spec
        if not parse_index_spec(req.layers, LOADED.adapter.cfg.n_layers):
            raise ValueError(f"Blocks '{req.layers}' selects none of this model's {LOADED.adapter.cfg.n_layers} blocks.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    train.JOBS["lora"] = job.start()
    return job.summary()


@app.post("/api/train/image")
def train_image(req: ImageTrainReq):
    """Train a tiny flow-matching image model on coloured shapes (see train_image.py). Needs no loaded model."""
    from .train_image import ImageJob
    _busy("image")
    if not (1 <= req.steps <= 20_000 and 4 <= req.batch <= 512):
        raise HTTPException(400, "steps 1–20000, batch 4–512.")
    train.JOBS["image"] = ImageJob(req.steps, req.batch, req.lr, seed=req.seed).start()
    return train.JOBS["image"].summary()


@app.post("/api/train/image/sample")
def sample_image(req: ImageSampleReq):
    """Draw with the tiny image model: one picture step by step (and, with grid=true, every colour × shape)."""
    j = train.JOBS.get("image")
    if j is None or j.model is None or j.status in ("queued", "running"):
        raise HTTPException(409, "Train the tiny image model first (and let it finish).")
    try:
        out = j.strip(req.color, req.shape, req.steps, max(0.0, min(req.guidance, 12.0)), req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.grid:
        out["grid"] = j.grid(max(1, min(req.steps, 60)), max(0.0, min(req.guidance, 12.0)), req.seed)
    return out


@app.get("/api/train/{kind}")
def train_state(kind: str):
    j = train.JOBS.get(kind)
    return j.summary() if j is not None else {"status": "none", "kind": kind}


@app.post("/api/train/{kind}/stop")
def train_stop(kind: str):
    j = train.JOBS.get(kind)
    if j is None:
        raise HTTPException(404, "No such training job.")
    j.stop = True
    return j.summary()


@app.get("/api/train/corpus/preview")
def corpus_preview(corpus: str = "mixed"):
    return JSONResponse({"corpus": corpus, "sample": train.synthetic_corpus(corpus, 12, seed=0)})


# ---------------------------------------------------------------- mixture of experts
def _moe_run(run_id: str):
    r, cap = _llm_run(run_id)
    if not LOADED.adapter.cfg.moe:
        raise HTTPException(400, "The loaded model is not a mixture of experts. Load an MoE model (OLMoE, Qwen-MoE, "
                                 "Mixtral …) or train a tiny one in the Train tab.")
    return r, cap


@app.get("/api/runs/{run_id}/experts")
def experts_overview(run_id: str):
    """Every MoE block of a run: expert usage, router entropy, each token's top expert."""
    from . import moe
    r, cap = _moe_run(run_id)
    return _guard(moe.overview, LOADED.adapter, cap, r.plan.schedule, r.plan.schedule_nodes)


@app.get("/api/runs/{run_id}/experts/block")
def experts_block(run_id: str, pos: int, token: int = -1):
    from . import moe
    r, cap = _moe_run(run_id)
    return _guard(moe.block_detail, LOADED.adapter, cap, r.plan.schedule, pos, token)


@app.get("/api/runs/{run_id}/expert")
def expert_one(run_id: str, pos: int, index: int):
    from . import moe
    r, cap = _moe_run(run_id)
    return _guard(moe.expert_detail, LOADED.adapter, cap, r.plan.schedule, pos, index)


@app.get("/api/experts/router")
def experts_router(layer: int):
    from . import moe
    adapter, _ = _llm()
    return _guard(moe.router_rows, adapter, layer)


class ExpertSweepReq(BaseModel):
    flow: dict
    pos: int = -1                    # schedule position of the block; -1 = every MoE block
    mode: str = "ban"                # ban: the router may not pick it · zero: it is picked but writes nothing


@app.post("/api/experts/sweep")
def experts_sweep(req: ExpertSweepReq):
    """Remove one expert at a time and measure what the answer loses; see moe.iter_ablation."""
    from . import moe
    adapter, info = _llm()
    if req.mode not in ("ban", "zero"):
        raise HTTPException(400, "mode must be ban or zero")
    plan = _compile(req.flow, info)
    prefix = _load_prefixes(plan, adapter) if plan.params.get("prefix_cache") else None
    return _task("analysis", f"expert ablation ({req.mode})", "expert_sweep",
                 lambda: moe.iter_ablation(adapter, plan, req.pos, prefix, req.mode), weights=adapter.model.W)


# ---------------------------------------------------------------- introspection
class InjectReq(BaseModel):
    flow: dict
    concepts: list[str]
    blocks: list[int]
    strengths: list[float] = [1.0, 2.0, 4.0]
    max_new_tokens: int = 40
    positions: str = "all"


@app.get("/api/introspect/defaults")
def introspect_defaults():
    from . import introspect as it
    return {"prompt": it.INJECTION_PROMPT, "baselines": it.BASELINE_WORDS, "concept_template": it.CONCEPT_TEMPLATE,
            "explain_template": it.EXPLAIN_TEMPLATE, "explain_prefill": it.EXPLAIN_PREFILL}


@app.post("/api/introspect/inject")
def introspect_inject(req: InjectReq):
    """Concept-injection sweep (concepts × blocks × strengths, plus a control trial); see introspect.iter_injection."""
    from . import introspect as it
    adapter, info = _llm()
    concepts = [c.strip() for c in req.concepts if c.strip()][:12]
    if len(req.blocks) * len(req.strengths) * max(1, len(concepts)) > 600:
        raise HTTPException(400, "That is more than 600 trials; choose fewer blocks, strengths or concepts.")
    plan = _compile(req.flow, info)
    return _task("analysis", f"concept injection ({len(concepts)} concept(s))", "inject",
                 lambda: it.iter_injection(adapter, plan, concepts, req.blocks, req.strengths,
                                           max(1, min(req.max_new_tokens, 120)), req.positions),
                 weights=adapter.model.W)


class ConceptVecReq(BaseModel):
    flow: dict
    word: str
    pos: int
    name: str = ""


@app.post("/api/introspect/vector")
def introspect_vector(req: ConceptVecReq):
    """Save one concept vector to the vector library, so it can be wired into a block as a steering box."""
    from . import introspect as it
    adapter, info = _llm()
    plan = _compile(req.flow, info)
    if not req.word.strip() or not 0 <= req.pos < len(plan.schedule):
        raise HTTPException(400, "Give a word and a block of the current flow.")
    cv = _guard(it.concept_vectors, adapter, plan, [req.word.strip()])
    v = cv["vectors"][req.word.strip()][req.pos]
    return interp.VECTORS.save(v, req.name or f"concept: {req.word.strip()} @#{req.pos}",
                               {"source": "concept", "word": req.word.strip(), "pos": req.pos, "hook": "resid_post",
                                "model_id": LOADED.spec["id"], "resid_norm": round(cv["resid_norm"][req.pos], 3)})


@app.get("/api/runs/{run_id}/meter")
def concept_meter(run_id: str, vector: str, centre: bool = True):
    """How strongly each block's residual stream points along a saved vector, at every token of a run."""
    from . import introspect as it
    r, cap = _llm_run(run_id)
    try:
        v, meta = interp.VECTORS.load(vector)
    except KeyError:
        raise HTTPException(404, "Unknown vector.")
    return {**_guard(it.concept_meter, LOADED.adapter, cap, r.plan.schedule, v, centre, r.plan), "vector": meta}


class RoundTripReq(BaseModel):
    run_id: str
    pos: int
    token: int = -1
    inject: int = 2
    template: str = ""
    prefill: str | None = None
    max_new_tokens: int = 24
    scale: float = 1.0
    how: str = "last"                # reconstruct from the explanation's last token | mean over its tokens


@app.post("/api/nla/roundtrip")
def nla_roundtrip(req: RoundTripReq):
    """activation → words → activation with nothing trained; see introspect.iter_round_trip."""
    from . import introspect as it
    adapter, _ = _llm()
    r, cap = _llm_run(req.run_id)
    tpl = req.template.strip() or it.EXPLAIN_TEMPLATE
    if "{x}" not in tpl:
        raise HTTPException(400, "The template needs a {x} placeholder.")
    pre = it.EXPLAIN_PREFILL if req.prefill is None else req.prefill
    return _task("analysis", f"activation → words → activation (#{req.pos})", "nla",
                 lambda: it.iter_round_trip(adapter, r.plan, cap, req.pos, req.token, req.inject, tpl, pre,
                                            max(1, min(req.max_new_tokens, 80)), req.scale,
                                            "mean" if req.how == "mean" else "last"),
                 weights=adapter.model.W)


class ScoreTextReq(BaseModel):
    run_id: str
    pos: int
    token: int = -1
    text: str
    how: str = "last"


@app.post("/api/nla/score")
def nla_score(req: ScoreTextReq):
    """Score an explanation typed by the user: can the activation be rebuilt from it?"""
    from . import introspect as it
    r, cap = _llm_run(req.run_id)
    return _guard(it.score_text, LOADED.adapter, r.plan, cap, req.pos, req.token, req.text,
                  "mean" if req.how == "mean" else "last")
