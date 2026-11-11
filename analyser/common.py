"""Shared building blocks for AI Analyser model plugins.

This module holds the mathematical and visualisation primitives that multiple
model plugins share — so each plugin file (`analyser/plugins/*.py`) stays short,
readable, and focused purely on its architecture:

  - Normalisation: `rms_norm`, `layer_norm`, `group_norm`, `channel_rms_norm`
  - Rotary position embeddings (RoPE): `rope_table_nd`, `apply_rope_complex`, `timestep_sinusoids`
  - Visualisation & image helpers: `pca_rgb`, `pil_to_data_url`, `heat_data_url`,
    `tensor_to_pil`, `fit_linear_preview`, `render_linear_preview`, `changed_tile_weights`
"""
from __future__ import annotations

import base64
import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ======================================================================================
# 1. Normalisation & linear helpers
# ======================================================================================

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, offset: float = 0.0) -> torch.Tensor:
    """Root-Mean-Square Layer Normalisation along the last dimension.

    Formula:
        RMSNorm(x) = (x / sqrt(mean(x²) + eps)) * (offset + weight)

    Note: Qwen3.5 / Qwen3.6 / Qwen-Image text_norm store weights zero-centred
     around 0 (`offset = 1.0`), whereas standard Llama / Qwen2 / Qwen3 store weights
    centred around 1 (`offset = 0.0`).
    """
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    w = weight.float() + offset if offset else weight.float()
    return (normed * w).to(x.dtype)


def channel_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Channel-wise RMSNorm for 2D feature maps `[B, C, H, W]` (used by Wan / Qwen-Image VAEs)."""
    g = gamma.reshape(1, -1, 1, 1)
    xn = F.normalize(x.float(), dim=1, eps=eps).to(x.dtype)
    return xn * (x.shape[1] ** 0.5) * g


# ======================================================================================
# 2. Position & Timestep Embeddings
# ======================================================================================

def rope_complex_axis(pos: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Complex rotary frequencies `[T, dim/2]` for a 1D coordinate vector `pos`."""
    inv = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim))
    freqs = torch.outer(pos.to(torch.float32), inv)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope_complex(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs `(2k, 2k+1)` of `x` `[T, H, D]` by complex `freqs` `[T, D/2]`."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs[:, None, :]).flatten(-2).type_as(x)


def timestep_sinusoids(t: torch.Tensor, dim: int = 256, flip_sin_to_cos: bool = True,
                       scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    """Standard sinusoidal timestep embedding for diffusion / flow-matching models."""
    half = dim // 2
    ex = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - shift)
    args = (scale * t.float())[:, None] * torch.exp(ex)[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], -1) if flip_sin_to_cos else torch.cat([torch.sin(args), torch.cos(args)], -1)
    return emb


# ======================================================================================
# 3. Feature & Image Visualisation Helpers
# ======================================================================================

def pca_rgb(feat: torch.Tensor, max_side: int = 64) -> np.ndarray:
    """Project `C × H × W` hidden features onto their top-3 principal components → RGB `[H, W, 3]`.

    Used by both UNet stages and DiT blocks so students can see *what* each layer
    encodes (patches with similar colours carry similar feature vectors).
    """
    f = feat.float()
    c, h, w = f.shape
    if max(h, w) > max_side:
        f = F.adaptive_avg_pool2d(f[None], (min(h, max_side), min(w, max_side)))[0]
        c, h, w = f.shape
    X = f.reshape(c, h * w).T
    X = X - X.mean(0, keepdim=True)
    try:
        _, _, V = torch.pca_lowrank(X, q=3, center=False)
    except Exception:
        return np.zeros((h, w, 3), dtype=np.uint8)
    Y = X @ V[:, :3]
    Y = Y * torch.sign(Y.sum(0, keepdim=True) + 1e-8)
    lo, hi = Y.quantile(0.02, dim=0), Y.quantile(0.98, dim=0)
    Y = ((Y - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return (Y.reshape(h, w, 3).cpu().numpy() * 255).astype(np.uint8)


def pil_to_data_url(im: Image.Image) -> str:
    """Encode a PIL image as a `data:image/png;base64,...` URL."""
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heat_data_url(m: np.ndarray) -> str:
    """Convert a `0..1` 2D array into a black-red-yellow-white heatmap PNG data URL."""
    r = np.clip(m * 3, 0, 1)
    g = np.clip(m * 3 - 1, 0, 1)
    b = np.clip(m * 3 - 2, 0, 1)
    return pil_to_data_url(Image.fromarray((np.stack([r, g, b], -1) * 255).astype(np.uint8)))


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """Convert `[C, H, W]` in `[-1, 1]` to a PIL Image (keeps RGBA only if alpha < 250 is used)."""
    arr = ((img.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 4 and arr[..., 3].min() < 250:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr[..., :3].copy())


def fit_linear_preview(lat: torch.Tensor, img: torch.Tensor, factor: int) -> torch.Tensor:
    """Fit a fast linear least-squares map `[latent, 1] → mean RGB` for each tile of the image."""
    tiles = F.avg_pool2d(img[None], factor)[0]                      # [3, h, w]
    Y = tiles.reshape(3, -1).T                                      # [N, 3]
    X = torch.cat([lat, torch.ones(lat.shape[0], 1)], 1)            # [N, z+1]
    reg = 1e-3 * torch.eye(X.shape[1])
    return torch.linalg.solve(X.T @ X + reg, X.T @ Y)


def render_linear_preview(lat: torch.Tensor, fit: torch.Tensor, hw: tuple[int, int],
                          scale: int, nearest: bool = False) -> Image.Image:
    """Apply the fitted linear map `fit` to a `[N, z]` latent tensor and upscale for display."""
    h, w = hw
    rgb = torch.cat([lat.float(), torch.ones(lat.shape[0], 1)], 1) @ fit
    arr = ((rgb.clamp(-1, 1) + 1) * 127.5).round().byte().reshape(h, w, 3).numpy()
    return Image.fromarray(arr).resize((w * scale, h * scale), Image.NEAREST if nearest else Image.BILINEAR)


def changed_tile_weights(before: Image.Image, after_url: str, factor: int, hw: tuple[int, int]) -> torch.Tensor:
    """Compute per-tile `[h·w, 1]` blend weights (`0..1`) where `after_url` differs from `before`.

    Any tile containing painted pixels gets full weight (`1.0`), with a 1-tile feathered halo
    so soft/blurry brush strokes blend smoothly across tile boundaries.
    """
    raw = base64.b64decode(after_url.split(",", 1)[-1])
    after = Image.open(io.BytesIO(raw)).convert("RGB").resize(before.size)
    d = np.abs(np.asarray(after, dtype=np.float32) - np.asarray(before.convert("RGB"), dtype=np.float32)).max(-1) / 255
    t = torch.from_numpy((d > 0.015).astype(np.float32))[None, None]
    t = F.max_pool2d(t, 3, stride=1, padding=1)
    tiles = (F.interpolate(t, size=hw, mode="area") * 8.0).clamp(0, 1)
    halo = F.avg_pool2d(tiles, 3, stride=1, padding=1)
    return torch.maximum(tiles, 0.5 * halo)[0, 0].clamp(0, 1).reshape(-1, 1)
