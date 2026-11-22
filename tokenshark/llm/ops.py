"""Core mathematical building blocks for Causal LLMs: RMSNorm, RoPE, YaRN, activations (`tokenshark.llm.ops`)."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .config import LLMConfig


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def rope_inv_freq(cfg: LLMConfig, theta: float | None = None) -> torch.Tensor:
    d = cfg.rot_dim or cfg.head_dim
    base_theta = float(theta) if theta is not None else cfg.theta
    inv = 1.0 / (base_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    rs = cfg.rope_scaling or {}
    kind = rs.get("rope_type") or rs.get("type")
    if kind == "llama3":
        factor, lo, hi = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"]
        old = rs["original_max_position_embeddings"]
        lo_wl, hi_wl = old / lo, old / hi
        wl = 2 * math.pi / inv
        out = torch.where(wl > lo_wl, inv / factor, inv)
        smooth = (old / wl - lo) / (hi - lo)
        smoothed = (1 - smooth) * out / factor + smooth * out
        mid = (wl >= hi_wl) & (wl <= lo_wl)
        inv = torch.where(mid, smoothed, out)
    elif kind == "linear":
        inv = inv / rs["factor"]
    elif kind == "yarn":
        factor = rs.get("factor") or cfg.max_pos / rs["original_max_position_embeddings"]
        old, bf, bs = rs["original_max_position_embeddings"], rs.get("beta_fast") or 32, rs.get("beta_slow") or 1
        corr = lambda rot: (d * math.log(old / (rot * 2 * math.pi))) / (2 * math.log(base_theta))      # noqa: E731
        lo, hi = max(math.floor(corr(bf)), 0), min(math.ceil(corr(bs)), d - 1)
        if lo == hi:
            hi += 0.001
        ramp = ((torch.arange(d // 2, dtype=torch.float32) - lo) / (hi - lo)).clamp(0, 1)
        inv = (inv / factor) * ramp + inv * (1 - ramp)
    return inv


def yarn_scales(cfg: LLMConfig) -> tuple[float, float]:
    """(multiplier on the rotated dims of q and k, multiplier on the attention scores) that YaRN adds."""
    rs = cfg.rope_scaling or {}
    if (rs.get("rope_type") or rs.get("type")) != "yarn":
        return 1.0, 1.0
    factor = rs.get("factor") or cfg.max_pos / rs["original_max_position_embeddings"]
    ms = lambda m=1.0: 1.0 if factor <= 1 else 0.1 * m * math.log(factor) + 1.0                          # noqa: E731
    m, mall = rs.get("mscale"), rs.get("mscale_all_dim")
    rot = rs.get("attention_factor") or (ms(m) / ms(mall) if m and mall else ms())
    return float(rot), float(ms(mall) ** 2) if mall else 1.0


def apply_rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x: [heads, T, d]; pos: [T] (any real numbers, negative allowed)."""
    rd = 2 * inv_freq.shape[0]
    if rd < x.shape[-1]:
        return torch.cat([apply_rope(x[..., :rd], pos, inv_freq), x[..., rd:]], -1)
    ang = pos.to(inv_freq.device)[:, None] * inv_freq[None, :]          # [T, d/2]
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos().to(x.device), emb.sin().to(x.device)
    xf = x.float()
    half = xf.shape[-1] // 2
    rot = torch.cat([-xf[..., half:], xf[..., :half]], -1)
    return (xf * cos + rot * sin).to(x.dtype)


def _act(name):
    return {"silu": F.silu, "gelu": F.gelu, "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
            "relu": F.relu}[name]
