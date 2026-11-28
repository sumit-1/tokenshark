"""In-browser training extension (Tiny LLM, LoRA / IA3 / Adapter / Block insertion, Tiny Diffusion) (`tokenshark.extensions.training`)."""
from __future__ import annotations

import re
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ... import train
from ...flow import parse_index_spec
from ...registry import LOADED
from ...runs import DATA
from ...sdk import FeaturePlugin, register_extension
from ...server.app import _need
from ...tasks import TASKS
from ...weights import models_dir

router = APIRouter()


class TinyReq(BaseModel):
    name: str = "my-tiny-model"
    text: str = ""
    corpus: str = "mixed"
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
    n_experts: int = 0
    experts_per_token: int = 2
    balance: float = 0.01
    n_questions: int = 1000
    system_prompt: str = ""


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
    layers: str = "all"
    method: str = "lora"
    after: int = 0
    init: str = "duplicate"
    system_prompt: str = ""
    n_questions: int = 0


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


@router.post("/api/train/tiny")
def train_tiny(req: TinyReq):
    _busy("tiny")
    name = _safe_name(req.name, "my-tiny-model")
    out = models_dir() / name
    if out.exists() and any(out.iterdir()):
        cfg = out / "config.json"
        if not (cfg.exists() and '"trained_here": true' in cfg.read_text(errors="ignore")):
            raise HTTPException(400, f"A model folder named '{name}' already exists and was not trained here. Choose another name.")
    if LOADED.spec and LOADED.spec["id"] == name:
        raise HTTPException(409, "That model is loaded right now. Unload it, or train under another name.")
    lim = dict(
        d_model=(16, 512), n_layers=(1, 12), n_heads=(1, 16), context=(8, 512), vocab=(260, 8000),
        steps=(1, 100_000), batch=(1, 256), n_experts=(0, 64), experts_per_token=(1, 64),
    )
    for k, (lo, hi) in lim.items():
        if not lo <= getattr(req, k) <= hi:
            raise HTTPException(400, f"{k} must be between {lo} and {hi}.")
    try:
        job = train.TinyJob(
            name, out, req.text, req.corpus, req.d_model, req.n_layers, req.n_heads, req.context,
            req.vocab, req.steps, req.batch, req.lr, req.seed, req.sample_prompt,
            req.n_experts, req.experts_per_token, req.balance,
            n_questions=req.n_questions, system_prompt=req.system_prompt,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    train.JOBS["tiny"] = job.start()
    return job.summary()


@router.post("/api/train/lora")
def train_lora(req: LoRATrainReq):
    _busy("lora")
    _need("lm_tools", "This tool")
    name = _safe_name(req.name, "my-lora")
    ok = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not req.targets or not set(req.targets) <= ok:
        raise HTTPException(400, f"targets must be chosen from {', '.join(sorted(ok))}.")
    if not (1 <= req.rank <= 256 and 1 <= req.steps <= 20_000 and 2 <= req.max_tokens <= 1024):
        raise HTTPException(400, "rank 1-256, steps 1-20000, max_tokens 2-1024.")
    from ...adapt import METHODS, AdaptJob
    if req.method not in METHODS:
        raise HTTPException(400, f"method must be one of {', '.join(METHODS)}.")
    cfg = LOADED.adapter.cfg
    if getattr(cfg, "base_layers", cfg.n_layers) != cfg.n_layers:
        raise HTTPException(409, "A trained block is attached to this model. Detach it in the LoRA / Fine-tune tab "
                                 "before training something new: training always starts from the base model.")
    try:
        kw = dict(rank=req.rank, alpha=req.alpha, targets=tuple(req.targets), steps=req.steps, lr=req.lr,
                  max_tokens=req.max_tokens, seed=req.seed, probe=req.probe, layers=req.layers,
                  system_prompt=req.system_prompt, n_questions=req.n_questions)
        if req.method == "lora":
            job = train.LoRAJob(LOADED, TASKS, DATA / "loras" / name, name, req.examples, **kw)
        else:
            job = AdaptJob(LOADED, TASKS, DATA / "loras" / name, name, req.examples, method=req.method,
                           after=req.after, init=req.init, **kw)
        if not parse_index_spec(req.layers, LOADED.adapter.cfg.n_layers):
            raise ValueError(f"Blocks '{req.layers}' selects none of this model's {LOADED.adapter.cfg.n_layers} blocks.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    train.JOBS["lora"] = job.start()
    return job.summary()


@router.post("/api/train/image")
def train_image(req: ImageTrainReq):
    from ...train_image import ImageJob
    _busy("image")
    if not (1 <= req.steps <= 20_000 and 4 <= req.batch <= 512):
        raise HTTPException(400, "steps 1-20000, batch 4-512.")
    train.JOBS["image"] = ImageJob(req.steps, req.batch, req.lr, seed=req.seed).start()
    return train.JOBS["image"].summary()


@router.post("/api/train/image/sample")
def sample_image(req: ImageSampleReq):
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


@router.get("/api/train/{kind}")
def train_state(kind: str):
    j = train.JOBS.get(kind)
    return j.summary() if j is not None else {"status": "none", "kind": kind}


@router.post("/api/train/{kind}/stop")
def train_stop(kind: str):
    j = train.JOBS.get(kind)
    if j is None:
        raise HTTPException(404, "No such training job.")
    j.stop = True
    return j.summary()


@router.get("/api/train/corpus/preview")
def corpus_preview(corpus: str = "mixed", offset: int = 0, limit: int = 50, n_questions: int = 2000):
    from ...sdk import get_dataset
    from ..datasets import format_row_for_tiny, split_qa_example
    ds = get_dataset(corpus)
    if ds is not None and (ds.rows or ds.examples):
        rws = ds.rows if ds.rows else [split_qa_example(x) for x in ds.examples]
        total_available = len(rws)
        active_total = min(max(1, n_questions), total_available) if n_questions > 0 else total_available
        off = max(0, min(offset, max(0, total_available - 1)))
        lim = max(1, min(limit, 500))
        page_rows = rws[off: off + lim]
        sample = "\n\n".join(format_row_for_tiny(r, system_prompt=ds.system_prompt) for r in page_rows[:16])
        return JSONResponse({
            "corpus": corpus,
            "name": ds.name,
            "description": ds.description,
            "total": total_available,
            "active_questions": active_total,
            "offset": off,
            "limit": lim,
            "system_prompt": ds.system_prompt,
            "sample_prompt": ds.sample_prompt,
            "rows": page_rows,
            "sample": sample,
        })
    total_syn = max(100, min(int(n_questions or 2000), 10000))
    off = max(0, min(offset, max(0, total_syn - 1)))
    lim = max(1, min(limit, 500))
    all_syn = train.synthetic_rows(corpus, n_lines=min(total_syn, off + lim), seed=0)
    page_rows = all_syn[off: off + lim]
    sample = "\n".join(f"{r['question']} {r['answer']}".strip() for r in page_rows[:16])
    return JSONResponse({
        "corpus": corpus,
        "name": f"Built-in ({corpus})",
        "description": "Synthetic procedural dataset",
        "total": total_syn,
        "active_questions": total_syn,
        "offset": off,
        "limit": lim,
        "system_prompt": "",
        "sample_prompt": "The capital of France is" if corpus != "counting" else "3 + 5 =",
        "rows": page_rows,
        "sample": sample,
    })


@register_extension
class TrainingPlugin(FeaturePlugin):
    id = "training"
    name = "In-Browser Training & Adaptation"
    version = "0.2.0"
    view_type = "playbook"
    default_enabled = False
    description = "Train tiny transformers from scratch, adapt loaded LLMs with LoRA/IA3/Bottleneck/Block insertion, or train tiny diffusion."
    families = ("causal_lm", "t2i")
    router = router
    ui_tabs = ({"id": "train", "label": "Train"},)
