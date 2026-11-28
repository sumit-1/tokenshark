"""Steering vectors extension (`tokenshark.extensions.steering`)."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
import time
import uuid
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...building_blocks import _token_index, activation
from ...registry import LOADED
from ...runs import DATA
from ...sdk import FeaturePlugin, register_extension
from ...server.app import _caps, _need, _run

router = APIRouter()
VEC_DIR = DATA / "vectors"


class VectorLibrary:
    """Steering vectors as files under data/vectors/ (safetensors: one tensor 'v' + metadata), plus bounded in-memory vectors."""

    def __init__(self, max_mem: int = 256):
        VEC_DIR.mkdir(parents=True, exist_ok=True)
        self.max_mem = max_mem
        self._mem: OrderedDict[str, tuple[torch.Tensor, dict]] = OrderedDict()

    def path(self, vid: str) -> Path:
        if not vid or not vid.replace("-", "").isalnum():
            raise KeyError(vid)
        base = os.path.realpath(VEC_DIR)
        target = os.path.realpath(os.path.join(base, f"{vid}.safetensors"))
        if not target.startswith(base + os.sep) or not os.path.isfile(target):
            raise KeyError(vid)
        return Path(target)

    def hold(self, v: torch.Tensor, name: str = "", meta: dict | None = None) -> dict:
        """Keep vector `v` in memory without writing a .safetensors file to disk."""
        vid = uuid.uuid4().hex[:10]
        m = {
            "id": vid,
            "name": name or f"vector {vid}",
            "created": time.time(),
            "dim": int(v.shape[-1]),
            "norm": round(float(v.float().norm()), 4),
            "ephemeral": True,
            **(meta or {}),
        }
        self._mem[vid] = (v.detach().float().cpu().contiguous(), m)
        while len(self._mem) > self.max_mem:
            self._mem.popitem(last=False)
        return m

    def save(self, v: torch.Tensor, name: str, meta: dict, persist: bool = True) -> dict:
        if not persist:
            return self.hold(v, name, meta)
        from safetensors.torch import save_file
        VEC_DIR.mkdir(parents=True, exist_ok=True)
        vid = uuid.uuid4().hex[:10]
        meta = {
            "id": vid,
            "name": name or f"vector {vid}",
            "created": time.time(),
            "dim": int(v.shape[-1]),
            "norm": round(float(v.float().norm()), 4),
            **meta,
        }
        save_file(
            {"v": v.detach().float().cpu().contiguous()},
            str(VEC_DIR / f"{vid}.safetensors"),
            metadata={"format": "tokenshark-vector/1", "meta": json.dumps(meta)},
        )
        return meta

    def load(self, vid: str) -> tuple[torch.Tensor, dict]:
        if vid in self._mem:
            self._mem.move_to_end(vid)
            return self._mem[vid]
        from safetensors import safe_open
        with safe_open(str(self.path(vid)), framework="pt", device="cpu") as f:
            return f.get_tensor("v"), json.loads((f.metadata() or {}).get("meta", "{}"))

    def list(self) -> list[dict]:
        from safetensors import safe_open
        out = [dict(m) for _, m in reversed(self._mem.values())]
        seen = {m["id"] for m in out}
        for p in sorted(VEC_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
            if p.stem in seen:
                continue
            try:
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    out.append({**json.loads((f.metadata() or {}).get("meta", "{}")), "id": p.stem})
            except Exception as e:  # noqa: BLE001
                out.append({"id": p.stem, "name": p.name, "error": str(e)})
        return out

    def delete(self, vid: str):
        if vid in self._mem:
            del self._mem[vid]
            return
        self.path(vid).unlink()


VECTORS = VectorLibrary()


def vector_from_runs(cap_a: dict, cap_b: dict, pos: int, hook: str = "resid_post", token="last") -> torch.Tensor:
    """Contrastive direction: activation of run A minus run B at one block."""
    a, b = activation(cap_a, hook, pos), activation(cap_b, hook, pos)
    if isinstance(token, str) and token.strip().lower() == "mean":
        return a.float().mean(0) - b.float().mean(0)
    if isinstance(token, str) and token.strip().lower() == "last":
        ia = min(cap_a.get("n_in") or a.shape[0], a.shape[0]) - 1
        ib = min(cap_b.get("n_in") or b.shape[0], b.shape[0]) - 1
        return a[ia].float() - b[ib].float()
    return a[_token_index(token, a.shape[0])].float() - b[_token_index(token, b.shape[0])].float()


def vector_from_word(adapter, word: str, minus: str = "") -> tuple[torch.Tensor, dict]:
    """The output direction of a word: its row of the unembedding matrix (minus another word's, if given)."""
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


@router.get("/api/vectors")
def list_vectors():
    return VECTORS.list()


@router.post("/api/vectors/from_runs")
def api_vector_from_runs(req: VecRunsReq):
    _need("lm_tools", "This tool")
    ra, rb = _run(req.run_a), _run(req.run_b)
    ca, cb = _caps(ra), _caps(rb)
    if ra.plan.family != "causal_lm" or rb.plan.family != "causal_lm":
        raise HTTPException(400, "This tool is for language-model runs.")
    try:
        with LOADED.lock:
            v = vector_from_runs(ca, cb, req.pos, req.hook, req.token)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'\""))
    return VECTORS.save(
        v,
        req.name or f"{ra.label or ra.id} - {rb.label or rb.id} @ #{req.pos}",
        {"model_id": ra.model_id, "source": "runs", "run_a": ra.id, "run_b": rb.id, "hook": req.hook, "pos": req.pos, "token": req.token},
    )


@router.post("/api/vectors/from_word")
def api_vector_from_word(req: VecWordReq):
    adapter, _ = _need("lm_tools", "This tool")
    try:
        with LOADED.lock:
            v, meta = vector_from_word(adapter, req.word, req.minus)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e).strip("'\""))
    name = req.name or (f"'{meta['word']}'" + (f" - '{meta['minus']}'" if meta.get("minus") else ""))
    return VECTORS.save(v, name, {"model_id": LOADED.spec["id"], "source": "word", **meta})


@router.delete("/api/vectors/{vid}")
def delete_vector(vid: str):
    try:
        VECTORS.delete(vid)
    except KeyError:
        raise HTTPException(404, "Unknown vector.")
    return {"ok": True}


@register_extension
class SteeringPlugin(FeaturePlugin):
    id = "steering"
    name = "Steering Vectors & Residual Directions"
    version = "0.2.0"
    view_type = "playbook"
    default_enabled = True
    description = "Extract contrastive or word-unembedding directions in the residual stream and steer or project them out."
    families = ("causal_lm",)
    router = router
    ui_tabs = ()
    palette_actions = ({"id": "add-steer", "label": "+ Steer", "group": "intervene", "feature": "vectors"},)
