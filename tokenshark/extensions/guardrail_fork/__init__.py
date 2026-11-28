"""Center-Layer Guardrail Fork extension (`tokenshark.extensions.guardrail_fork`).

Forks the residual stream at any transformer block (`block_k.fork -> fork_head.in`), trains a lightweight
classifier (`linear_probe`, `mlp_guard`, or `block_branch`) on frozen center-layer activations in seconds,
and runs it live in `monitor` mode (per-token category probabilities) or `gate` mode (early-stops/redirects
generation at the forked block when a guarded category crosses `threshold`).

Includes the clean classroom preset `powershell_detector` (categorizing PowerShell cmdlets/scripts vs
Python / General Text) plus user-definable multi-class categories.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...flow import Plan
from ...llm import Stepper, rms_norm
from ...registry import LOADED
from ...runs import DATA
from ...sdk import (
    FeaturePlugin,
    NodeKindSpec,
    register_extension,
    register_fork_runner,
    register_lab_method,
    register_mcp_tool,
    register_node_kind,
)
from ...server.app import _need

router = APIRouter()
FORKS_DIR = DATA / "forks"

PRESETS = {
    "powershell_detector": {
        "id": "powershell_detector",
        "name": "PowerShell vs Python / General Detector",
        "description": "Clean classroom preset that classifies whether a prompt or script is PowerShell cmdlets/pipeline syntax vs Python or general text at a middle transformer block.",
        "categories": {
            "PowerShell": [
                "Get-Process | Where-Object { $_.CPU -gt 10 } | Sort-Object CPU -Descending",
                "Get-ChildItem -Path C:\\Logs -Recurse -Filter *.log | Select-String -Pattern 'ERROR'",
                "Invoke-RestMethod -Uri 'https://api.example.com/status' -Method Get",
                "$services = Get-Service | Where-Object {$_.Status -eq 'Running'}",
                "New-Item -ItemType Directory -Force -Path $env:TEMP\\backup",
                "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser",
                "Get-Content .\\config.json | ConvertFrom-Json | Format-Table",
                "Test-Connection -ComputerName localhost -Count 2 -Quiet",
            ],
            "Python / General": [
                "import os\nfor root, dirs, files in os.walk('/var/log'):\n    print(len(files))",
                "def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
                "Can you explain how transformer self-attention computes query and key dot products?",
                "import requests\nresp = requests.get('https://api.example.com/status')\nprint(resp.json())",
                "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 5;",
                "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                "heights = [x ** 2 for x in range(10) if x % 2 == 0]",
                "Please summarize the main differences between TCP and UDP transport protocols.",
            ],
        },
        "trigger_category": "PowerShell",
        "threshold": 0.65,
        "head_type": "linear_probe",
        "halt_message": "[Guardrail Fork: PowerShell command detected at center layer - generation halted]",
    },
    "topic_router": {
        "id": "topic_router",
        "name": "Math / Code vs Creative Writing Router",
        "description": "Multi-class probe distinguishing arithmetic/coding prompts from creative writing at a center block.",
        "categories": {
            "Math & Code": [
                "Solve for x: 3x + 15 = 42.",
                "Write a Python function to merge two sorted lists.",
                "If a train travels 120 miles in 2 hours, what is its average speed?",
                "Compute the derivative of f(x) = x^3 - 4x + 7.",
            ],
            "Creative Writing": [
                "Write a short poem about autumn leaves falling on a quiet lake.",
                "Once upon a time in a clockwork city above the clouds, a young inventor found a brass key.",
                "Describe the aroma and atmosphere of a cozy bakery at sunrise.",
                "Compose a dramatic monologue for a lighthouse keeper during a storm.",
            ],
        },
        "trigger_category": "Math & Code",
        "threshold": 0.7,
        "head_type": "mlp_guard",
        "halt_message": "[Guardrail Fork: Math & Code route triggered]",
    },
}


class ForkHeadModule(nn.Module):
    """Lightweight classification head operating on normalized residual stream vectors `[..., d_model]`."""

    def __init__(self, d_model: int, n_classes: int, head_type: str = "linear_probe", hidden_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.head_type = head_type
        if head_type == "linear_probe":
            self.linear = nn.Linear(d_model, n_classes)
        elif head_type == "mlp_guard":
            h = max(16, min(hidden_dim, d_model))
            self.fc1 = nn.Linear(d_model, h)
            self.fc2 = nn.Linear(h, n_classes)
        elif head_type == "block_branch":
            h = max(32, min(hidden_dim * 2, d_model))
            self.gate_proj = nn.Linear(d_model, h, bias=False)
            self.up_proj = nn.Linear(d_model, h, bias=False)
            self.down_proj = nn.Linear(h, d_model, bias=False)
            self.classifier = nn.Linear(d_model, n_classes)
        else:
            raise ValueError(f"Unknown head_type '{head_type}' (choose linear_probe, mlp_guard, or block_branch).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize each token vector to unit RMS so probes generalize across layer scales
        x_norm = x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        if self.head_type == "linear_probe":
            return self.linear(x_norm)
        if self.head_type == "mlp_guard":
            return self.fc2(F.silu(self.fc1(x_norm)))
        h = x_norm + self.down_proj(F.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm))
        return self.classifier(h)


def _safe_name(name: str, default: str = "fork-head") -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "").strip()).strip("-.") or default


def save_fork_head(name: str, module: ForkHeadModule, meta: dict) -> dict:
    from safetensors.torch import save_file
    FORKS_DIR.mkdir(parents=True, exist_ok=True)
    fid = _safe_name(name, f"fork-{int(time.time())}")
    base = os.path.realpath(FORKS_DIR)
    folder_str = os.path.realpath(os.path.join(base, fid))
    if not folder_str.startswith(base + os.sep):
        raise ValueError(f"Invalid fork head name '{name}'.")
    folder = Path(folder_str)
    folder.mkdir(parents=True, exist_ok=True)
    meta_out = {
        "id": fid,
        "name": name or fid,
        "created": time.time(),
        "d_model": module.d_model,
        "n_classes": module.n_classes,
        "head_type": module.head_type,
        **meta,
    }
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in module.state_dict().items()}
    save_file(tensors, str(folder / "fork_head.safetensors"))
    (folder / "fork_head.json").write_text(json.dumps(meta_out, indent=2))
    return meta_out


def load_fork_head(fid: str) -> tuple[ForkHeadModule, dict]:
    from safetensors.torch import load_file
    FORKS_DIR.mkdir(parents=True, exist_ok=True)
    fid_clean = _safe_name(fid, "")
    folder = next((p for p in FORKS_DIR.iterdir() if p.is_dir() and p.name == fid_clean), None) if fid_clean else None
    if folder is None or not (folder / "fork_head.json").exists():
        raise KeyError(f"Unknown fork head '{fid}'.")
    meta = json.loads((folder / "fork_head.json").read_text())
    mod = ForkHeadModule(
        d_model=int(meta["d_model"]),
        n_classes=int(meta["n_classes"]),
        head_type=meta.get("head_type", "linear_probe"),
        hidden_dim=int(meta.get("hidden_dim", 64)),
    )
    sd = load_file(str(folder / "fork_head.safetensors"), device="cpu")
    mod.load_state_dict(sd)
    mod.eval()
    return mod, meta


def list_fork_heads() -> list[dict]:
    FORKS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(FORKS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        jf = p / "fork_head.json"
        if p.is_dir() and jf.exists():
            try:
                out.append(json.loads(jf.read_text()))
            except Exception:
                pass
    return out


def _extract_block_activation(adapter, schedule: list[int], pos: int, text: str, hook: str = "resid_post",
                              pooling: str = "mean") -> torch.Tensor:
    """Run a forward prefill up to block `pos` and extract the activation vector `[d_model]`."""
    S = len(schedule)
    p = min(max(int(pos), 0), S - 1)
    plan = Plan(
        family="causal_lm",
        schedule=list(schedule[: p + 1]),
        schedule_nodes=[f"b{i}" for i in range(p + 1)],
        params={
            "prompt": text,
            "system": "",
            "use_chat_template": False,
            "max_new_tokens": 0,
            "temperature": 0.0,
            "top_k": 50,
            "top_p": 1.0,
            "seed": 0,
            "neuron_budget_mb": 0,
            "attn_budget_mb": 0,
            "parts_budget_mb": 64,
        },
    )
    st = Stepper(adapter, plan)
    while st.step_token():
        pass
    res = st.finish()
    cap = res["captures"]
    acts = (cap.get(hook) or cap.get("resid_post") or {}).get(p)
    if acts is None:
        raise ValueError(f"Could not capture {hook} at block #{p}.")
    acts = acts.float().cpu()
    if pooling == "last":
        return acts[-1]
    return acts.mean(dim=0)


def train_fork_classifier(
    adapter,
    categories: dict[str, list[str]],
    pos: int | None = None,
    hook: str = "resid_post",
    head_type: str = "linear_probe",
    steps: int = 80,
    lr: float = 0.05,
    name: str = "powershell_detector",
    trigger_category: str = "",
    threshold: float = 0.65,
    halt_message: str = "",
    pooling: str = "mean",
    seed: int = 0,
) -> dict:
    """Extract frozen activations at center block `pos` for each category's examples and train a `ForkHeadModule`."""
    schedule = list(range(adapter.cfg.n_layers))
    if pos is None or pos < 0:
        pos = len(schedule) // 2
    pos = min(max(int(pos), 0), len(schedule) - 1)

    cat_names = [k.strip() for k, v in categories.items() if k.strip() and v]
    if len(cat_names) < 2:
        raise ValueError("Provide at least 2 categories with non-empty example lists.")

    xs, ys = [], []
    per_cat_counts = {}
    for c_idx, cat in enumerate(cat_names):
        prompts = [str(x).strip() for x in categories[cat] if str(x).strip()]
        if not prompts:
            raise ValueError(f"Category '{cat}' has no non-empty examples.")
        per_cat_counts[cat] = len(prompts)
        for pr in prompts:
            vec = _extract_block_activation(adapter, schedule, pos, pr, hook=hook, pooling=pooling)
            xs.append(vec)
            ys.append(c_idx)

    X = torch.stack(xs, dim=0)                              # [N, d_model]
    Y = torch.tensor(ys, dtype=torch.long)                  # [N]
    d_model = int(X.shape[-1])
    n_classes = len(cat_names)

    torch.manual_seed(int(seed))
    mod = ForkHeadModule(d_model=d_model, n_classes=n_classes, head_type=head_type)
    opt = torch.optim.AdamW(mod.parameters(), lr=float(lr), weight_decay=1e-2)

    loss_curve = []
    mod.train()
    for step in range(max(1, int(steps))):
        opt.zero_grad()
        logits = mod(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        opt.step()
        if step % max(1, int(steps) // 10) == 0 or step == int(steps) - 1:
            loss_curve.append({"step": step + 1, "loss": round(float(loss.item()), 4)})

    mod.eval()
    with torch.no_grad():
        logits = mod(X)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        acc = float((preds == Y).float().mean().item())

    trig = trigger_category if trigger_category in cat_names else cat_names[0]
    meta = save_fork_head(
        name=name,
        module=mod,
        meta={
            "model_id": LOADED.spec["id"] if LOADED.spec else "",
            "pos": pos,
            "layer": schedule[pos],
            "hook": hook,
            "pooling": pooling,
            "categories": cat_names,
            "counts": per_cat_counts,
            "trigger_category": trig,
            "threshold": float(threshold),
            "halt_message": halt_message or f"[Guardrail Fork: {trig} detected at block #{pos} - generation halted]",
            "accuracy": round(acc, 4),
            "final_loss": loss_curve[-1]["loss"] if loss_curve else 0.0,
            "loss_curve": loss_curve,
        },
    )
    return meta


def evaluate_fork_on_tensor(fork_spec, x_tensor: torch.Tensor, ctx) -> dict:
    """Evaluate a `ForkSpec` on block activation `x_tensor` `[T, d_model]`."""
    fid = fork_spec.ref or fork_spec.params.get("ref") or "powershell_detector"
    try:
        mod, meta = load_fork_head(fid)
    except KeyError:
        return {
            "id": fork_spec.node_id,
            "pos": fork_spec.pos,
            "hook": fork_spec.hook,
            "mode": fork_spec.mode,
            "ref": fid,
            "status": "untrained",
            "triggered": False,
            "message": f"Fork head '{fid}' not trained yet.",
        }

    x_cpu = x_tensor.detach().float().cpu()
    if x_cpu.ndim == 1:
        x_cpu = x_cpu.unsqueeze(0)
    if x_cpu.shape[-1] != mod.d_model:
        return {
            "id": fork_spec.node_id,
            "pos": fork_spec.pos,
            "hook": fork_spec.hook,
            "mode": fork_spec.mode,
            "ref": fid,
            "status": "dim_mismatch",
            "triggered": False,
            "message": f"Fork head dim {mod.d_model} != model dim {x_cpu.shape[-1]}",
        }

    categories = meta.get("categories") or [f"Class {i}" for i in range(mod.n_classes)]
    pooling = fork_spec.params.get("pooling") or meta.get("pooling") or "mean"
    with torch.no_grad():
        tok_logits = mod(x_cpu)                                 # [T, C]
        tok_probs = F.softmax(tok_logits, dim=-1)               # [T, C]
        pooled_vec = x_cpu[-1:] if pooling == "last" else x_cpu.mean(dim=0, keepdim=True)
        pooled_probs = F.softmax(mod(pooled_vec)[0], dim=-1)    # [C]

    top_idx = int(pooled_probs.argmax().item())
    top_cat = categories[top_idx] if top_idx < len(categories) else str(top_idx)
    top_prob = round(float(pooled_probs[top_idx].item()), 4)

    trig_cat = fork_spec.params.get("trigger_category") or meta.get("trigger_category") or categories[0]
    trig_idx = categories.index(trig_cat) if trig_cat in categories else 0
    trig_prob = float(pooled_probs[trig_idx].item())
    thresh = float(fork_spec.threshold if fork_spec.threshold is not None else meta.get("threshold", 0.65))
    triggered = bool(trig_prob >= thresh)
    halt_msg = fork_spec.params.get("halt_message") or meta.get("halt_message") or f"[Guardrail Fork: {trig_cat} detected]"

    prev = (getattr(ctx, "store", {}).get("fork_outputs", {}) if ctx is not None else {}).get(fork_spec.node_id)
    token_history = list(prev.get("token_probs", [])) if prev and x_cpu.shape[0] == 1 else []
    for row in tok_probs:
        token_history.append({categories[i]: round(float(row[i].item()), 4) for i in range(len(categories))})

    if triggered and fork_spec.mode == "gate" and ctx is not None and hasattr(ctx, "store"):
        ctx.store["fork_gate_halt"] = halt_msg

    return {
        "id": fork_spec.node_id,
        "pos": fork_spec.pos,
        "hook": fork_spec.hook,
        "mode": fork_spec.mode,
        "ref": fid,
        "head_type": meta.get("head_type", "linear_probe"),
        "categories": categories,
        "scores": {categories[i]: round(float(pooled_probs[i].item()), 4) for i in range(len(categories))},
        "top_category": top_cat,
        "top_prob": top_prob,
        "trigger_category": trig_cat,
        "trigger_prob": round(trig_prob, 4),
        "threshold": thresh,
        "triggered": triggered,
        "halt_message": halt_msg,
        "token_probs": token_history,
    }


def _fork_runner_fn(fork_spec, x_tensor: torch.Tensor, ctx, model=None) -> dict:
    return evaluate_fork_on_tensor(fork_spec, x_tensor, ctx)


register_node_kind(
    NodeKindSpec(
        kind="fork_head",
        label="Fork Head",
        handles=(("in", "target", "left", "fork"),),
        default_data={
            "kind": "fork_head",
            "fork_kind": "guardrail",
            "hook": "resid_post",
            "mode": "monitor",
            "threshold": 0.65,
            "ref": "powershell_detector",
            "trigger_category": "PowerShell",
            "enabled": True,
        },
        description="Side-branch classifier / guardrail head forked from a transformer block.",
    )
)

register_fork_runner("guardrail", _fork_runner_fn)
register_fork_runner("fork_head", _fork_runner_fn)


class TrainForkReq(BaseModel):
    preset: str = ""
    categories: dict[str, list[str]] | None = None
    pos: int = -1
    hook: str = "resid_post"
    head_type: str = "linear_probe"
    steps: int = 80
    lr: float = 0.05
    name: str = "powershell_detector"
    trigger_category: str = ""
    threshold: float = 0.65
    halt_message: str = ""
    pooling: str = "mean"
    seed: int = 0


class EvalForkReq(BaseModel):
    ref: str = "powershell_detector"
    text: str
    pos: int = -1
    hook: str = "resid_post"
    pooling: str = "mean"


@router.get("/api/ext/guardrail_fork/presets")
def api_fork_presets():
    return {"presets": PRESETS, "heads": list_fork_heads()}


@router.get("/api/ext/guardrail_fork/heads")
def api_list_fork_heads():
    return list_fork_heads()


@router.post("/api/ext/guardrail_fork/train")
def api_train_fork(req: TrainForkReq):
    adapter, _ = _need("lm_tools", "Guardrail fork training")
    cats = req.categories
    trig = req.trigger_category
    thresh = req.threshold
    h_type = req.head_type
    halt_msg = req.halt_message
    name = req.name
    if req.preset and req.preset in PRESETS and not cats:
        p = PRESETS[req.preset]
        cats = p["categories"]
        trig = trig or p["trigger_category"]
        thresh = thresh if req.threshold != 0.65 else p["threshold"]
        h_type = h_type or p["head_type"]
        halt_msg = halt_msg or p["halt_message"]
        name = name or p["id"]
    if not cats:
        raise HTTPException(400, "Provide `categories` or choose a built-in `preset` (e.g. 'powershell_detector').")
    try:
        with LOADED.lock:
            return train_fork_classifier(
                adapter,
                categories=cats,
                pos=req.pos if req.pos >= 0 else None,
                hook=req.hook,
                head_type=h_type,
                steps=req.steps,
                lr=req.lr,
                name=name,
                trigger_category=trig,
                threshold=thresh,
                halt_message=halt_msg,
                pooling=req.pooling,
                seed=req.seed,
            )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/api/ext/guardrail_fork/evaluate")
def api_eval_fork(req: EvalForkReq):
    adapter, _ = _need("lm_tools", "Guardrail fork evaluation")
    try:
        mod, meta = load_fork_head(req.ref)
    except KeyError as e:
        raise HTTPException(404, str(e))
    schedule = list(range(adapter.cfg.n_layers))
    pos = req.pos if req.pos >= 0 else int(meta.get("pos", len(schedule) // 2))
    hook = req.hook or meta.get("hook", "resid_post")
    pooling = req.pooling or meta.get("pooling", "mean")
    with LOADED.lock:
        vec = _extract_block_activation(adapter, schedule, pos, req.text, hook=hook, pooling=pooling)
    from ...flow import ForkSpec
    fspec = ForkSpec(
        node_id="eval",
        pos=pos,
        hook=hook,
        kind="guardrail",
        mode="monitor",
        threshold=float(meta.get("threshold", 0.65)),
        ref=req.ref,
        params={"pooling": pooling, "trigger_category": meta.get("trigger_category", "")},
    )
    return evaluate_fork_on_tensor(fspec, vec.unsqueeze(0), None)


def _lab_train_guardrail_fork(
    lab,
    preset: str = "powershell_detector",
    categories: dict[str, list[str]] | None = None,
    pos: int = -1,
    head_type: str = "linear_probe",
    steps: int = 80,
    name: str = "powershell_detector",
    threshold: float = 0.65,
) -> dict:
    """Train a center-layer guardrail fork classifier (`linear_probe`, `mlp_guard`, or `block_branch`)."""
    return lab._post(
        "/api/ext/guardrail_fork/train",
        preset=preset,
        categories=categories,
        pos=pos,
        head_type=head_type,
        steps=steps,
        name=name,
        threshold=threshold,
    )


register_lab_method("train_guardrail_fork", _lab_train_guardrail_fork)
register_mcp_tool(
    "train_guardrail_fork",
    lambda preset="powershell_detector", pos=-1, head_type="linear_probe", steps=80, name="powershell_detector", threshold=0.65: (
        _lab_train_guardrail_fork(
            LOADED, preset=preset, pos=pos, head_type=head_type, steps=steps, name=name, threshold=threshold
        )
    ),
    description="Train a center-layer guardrail fork classifier (e.g. powershell_detector).",
)


@register_extension
class GuardrailForkPlugin(FeaturePlugin):
    id = "guardrail_fork"
    name = "Center-Layer Guardrail Fork"
    version = "0.2.0"
    view_type = "playbook"
    default_enabled = False
    description = "Fork a middle transformer block into a lightweight classifier head (e.g. PowerShell vs Python detector) to monitor or gate generation mid-network."
    families = ("causal_lm",)
    router = router
    ui_tabs = ({"id": "guardrail", "label": "Guardrail Fork"},)
    palette_actions = ({"id": "add-fork", "label": "+ Fork Head", "group": "intervene", "feature": "interventions"},)
