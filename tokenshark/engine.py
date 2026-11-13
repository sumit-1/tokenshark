"""Shared engine pieces: device selection, block scheduling, intervention ops."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .flow import Intervention, parse_index_spec


class Cancelled(RuntimeError):
    """The user stopped the run. Not an error: whatever was computed so far is kept."""


def pick_device() -> str:
    want = os.environ.get("TOKENSHARK_DEVICE", "auto").lower()
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    want = os.environ.get("TOKENSHARK_DTYPE", "auto").lower()
    table = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16,
             "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if want in table:
        return table[want]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def set_determinism(seed: int):
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------------------
# Run context: what the hooks consult while the model executes
# --------------------------------------------------------------------------------------

@dataclass
class RunContext:
    interventions: list[Intervention]
    capture: bool = False
    pos: int = -1                      # current schedule position
    step: int = 0                      # current diffusion step
    n_steps: int = 1
    token_ranges: dict[str, range] = field(default_factory=dict)   # prompt / prefill / generated
    store: dict[str, Any] = field(default_factory=dict)            # captures land here
    on_block: Callable[[int], None] | None = None
    cancel: bool = False
    pause: bool = False
    offset: int = 0                    # run-relative index of the first token of the chunk being processed
    # What patch / steer / clamp need from outside the run, per intervention box:
    #   patch → [tokens, d] activation of the source run · steer → [d] direction · clamp → {"min": [n], "max": [n]}
    sources: dict[str, Any] = field(default_factory=dict)

    def checkpoint(self):
        """Called between blocks / diffusion steps: wait while paused, stop when cancelled."""
        while self.pause and not self.cancel:
            time.sleep(0.1)
        if self.cancel:
            raise Cancelled("Run cancelled.")

    def active(self, hook: str) -> list[Intervention]:
        out = []
        for iv in self.interventions:
            if iv.pos != self.pos or not (iv.hook == hook or (iv.hook == "both" and hook in ("out", "skip"))):
                continue
            if self.n_steps > 1 or iv.steps not in ("all", ""):
                if self.step not in parse_index_spec(iv.steps, self.n_steps):
                    continue
            out.append(iv)
        return out


def apply_op(x: torch.Tensor, iv: Intervention, ctx: RunContext, token_dim: int | None,
             idx: list[int] | None = None) -> torch.Tensor:
    """Apply one intervention to activation x. token_dim = which axis indexes tokens (None = whole tensor).
    idx = token indices local to x along token_dim (computed by the caller); None = parse iv.positions."""
    if token_dim is not None:
        n = x.shape[token_dim]
        if idx is None:
            idx = parse_index_spec(iv.positions, n, ctx.token_ranges)
        if not idx:
            return x
        sel = [slice(None)] * x.ndim
        sel[token_dim] = torch.tensor(idx, device=x.device)
        sel = tuple(sel)
        part = x[sel]
    else:
        sel, part = None, x
    # Any op can be limited to some units of the last axis (iv.neurons): one MLP neuron, a few residual dimensions.
    nsel, whole = None, None
    if (getattr(iv, "neurons", "all") or "all").strip().lower() not in ("all", "*", ""):
        nidx = parse_index_spec(iv.neurons, part.shape[-1])
        if not nidx:
            return x
        nsel = torch.tensor(nidx, device=x.device)
        whole, part = part, part[..., nsel]
    new = _op(part, iv, ctx, token_dim if sel is not None else None, idx, nsel, x)
    if new is None:
        return x
    if nsel is not None:
        whole = whole.clone()
        whole[..., nsel] = new.to(whole.dtype)
        new = whole
    if sel is None:
        return new
    x = x.clone()
    x[sel] = new.to(x.dtype)
    return x


def _op(part, iv, ctx, token_dim, idx, nsel, x):
    """The operation itself, on the selected slice. Returns the new slice (None = leave it unchanged)."""
    if iv.op == "scale":
        return part * iv.value
    if iv.op == "zero":
        return torch.zeros_like(part)
    if iv.op == "add_noise":
        g = torch.Generator(device="cpu").manual_seed(iv.seed + 7919 * ctx.step + 31 * iv.pos)
        noise = torch.randn(part.shape, generator=g, dtype=torch.float32).to(part.device, part.dtype)
        return part + iv.value * part.float().std().to(part.dtype) * noise
    if iv.op == "mean":
        dim = token_dim if token_dim is not None else tuple(range(1, x.ndim))
        return part.mean(dim=dim, keepdim=True).expand_as(part)
    src = ctx.sources.get(iv.node_id)
    if iv.op == "patch":
        # Activation patching: overwrite this run's activation with the one another run had at the same place.
        if src is None or idx is None:
            return None
        if iv.src_positions.strip():
            take = parse_index_spec(iv.src_positions, src.shape[0])
            if len(take) == 1:
                take = take * len(idx)
            if len(take) != len(idx):
                raise ValueError(f"Patch: {len(idx)} token(s) to overwrite but {len(take)} source token(s) chosen.")
            keep = list(range(len(idx)))
        else:                                   # the same token index in both runs
            keep = [j for j, i in enumerate(idx) if ctx.offset + i < src.shape[0]]
            take = [ctx.offset + idx[j] for j in keep]
        if not keep:
            return None
        rows = src[torch.tensor(take)].to(part.device, part.dtype)
        if nsel is not None:
            rows = rows[..., nsel]
        if rows.shape[-1] != part.shape[-1]:
            raise ValueError(f"Patch: the source run's activation is {rows.shape[-1]} wide, this one {part.shape[-1]}. "
                             f"Both runs must use the same model.")
        new = part.clone()
        new[torch.tensor(keep, device=part.device)] = rows
        return new
    if iv.op == "steer":
        # Steering: push the activation along one fixed direction, by the same amount at every selected token.
        if src is None:
            return None
        v = src.to(part.device, part.dtype)
        if nsel is not None:
            v = v[nsel]
        if v.shape[-1] != part.shape[-1]:
            raise ValueError(f"The steering vector has {v.shape[-1]} dimensions but this activation has {part.shape[-1]}.")
        return part + iv.value * v
    if iv.op == "clamp":
        # Pin units to a fixed output. max / min = the largest / smallest value each unit took in the reference run
        # (or so far in this run). A value other than 0 or 1 multiplies it ("5 × its maximum").
        if iv.clamp == "value":
            return torch.full_like(part, iv.value)
        stat = src.get(iv.clamp) if isinstance(src, dict) else None
        if stat is None:
            live = ctx.store.get("unit_range", {}).get((iv.pos, iv.hook))
            stat = live[0 if iv.clamp == "min" else 1] if live is not None else None
        if stat is None:
            return None
        stat = stat.to(part.device, part.dtype)
        if nsel is not None:
            stat = stat[nsel]
        mult = iv.value if iv.value not in (0, 1) else 1.0
        return (stat * mult).expand_as(part).clone()
    return None
