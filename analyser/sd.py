"""Raw PyTorch Stable Diffusion (SD 1.x / 2.x / SD-Turbo / distilled tiny-sd). No diffusers.

    prompt ─ CLIP BPE ─ CLIP text transformer ──────────────┐ (cross-attention context)
    noise latent ─┬─ UNet(x_t, t, context) ─ ε̂ ─ Euler step ─┴→ x_{t-1}   (× steps)
                  └─ x̂₀ = x_t − σ·ε̂   (per-step preview)
    final latent ─ VAE decoder ─ image

The UNet is built from the weight names (how many resnets / attentions each block has), with
head counts from unet/config.json, so distilled variants with removed blocks load too.
Weights load from diffusers-layout folders (unet/, vae/, text_encoder/, tokenizer/, scheduler/).
"""
from __future__ import annotations

import base64
import io
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .engine import RunContext, apply_op, set_determinism
from .flow import Plan, parse_index_spec
from .tokenizer import CLIPTokenizer
from .weights import WeightStore, read_json


# ======================================================================================
# Building blocks
# ======================================================================================

def conv(W, pre, x, stride=1, padding=None):
    w = W(pre + ".weight")
    b = W(pre + ".bias") if W.has(pre + ".bias") else None
    if padding is None:
        padding = w.shape[-1] // 2
    return F.conv2d(x, w, b, stride=stride, padding=padding)


def linear(W, pre, x):
    w = W(pre + ".weight")
    if w.ndim == 4:                       # old checkpoints store 1x1 convs for VAE attention
        w = w[:, :, 0, 0]
    b = W(pre + ".bias") if W.has(pre + ".bias") else None
    return F.linear(x, w, b)


def group_norm(W, pre, x, groups=32, eps=1e-5):
    return F.group_norm(x.float(), groups, W(pre + ".weight").float(), W(pre + ".bias").float(), eps).to(x.dtype)


def layer_norm(W, pre, x, eps=1e-5):
    return F.layer_norm(x.float(), (x.shape[-1],), W(pre + ".weight").float(), W(pre + ".bias").float(), eps).to(x.dtype)


def resnet(W, pre, x, temb=None, groups=32, eps=1e-5, drop=False, tap=None):
    """ResNet block: x + h(x). `tap(h)` sees what the block writes; `drop` removes that write (ablation)."""
    h = F.silu(group_norm(W, pre + ".norm1", x, groups, eps))
    h = conv(W, pre + ".conv1", h)
    if temb is not None and W.has(pre + ".time_emb_proj.weight"):
        h = h + linear(W, pre + ".time_emb_proj", F.silu(temb))[:, :, None, None]
    h = F.silu(group_norm(W, pre + ".norm2", h, groups, eps))
    h = conv(W, pre + ".conv2", h)
    if W.has(pre + ".conv_shortcut.weight"):
        x = conv(W, pre + ".conv_shortcut", x, padding=0)
    if tap is not None:
        tap(h)
    return x if drop else x + h


def attention(q, k, v, heads, record=None, edit=None):
    """q: [b, n, c], k/v: [b, m, c] → [b, n, c]. record(probs[b, heads, n, m]) is called if given;
    edit(probs) -> probs may change the attention weights (e.g. scale how much the image reads a word)."""
    b, n, c = q.shape
    m = k.shape[1]
    d = c // heads
    q = q.view(b, n, heads, d).transpose(1, 2)
    k = k.view(b, m, heads, d).transpose(1, 2)
    v = v.view(b, m, heads, d).transpose(1, 2)
    if record is None and edit is None:
        out = F.scaled_dot_product_attention(q, k, v)
    else:
        probs = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d), -1)
        if edit is not None:
            probs = edit(probs)
        if record is not None:
            record(probs)
        out = probs.to(v.dtype) @ v
    return out.transpose(1, 2).reshape(b, n, c)


def _region(run, iv, hw):
    """The painted region of an intervention at feature resolution hw, or None (= everywhere)."""
    m = (run.store.get("regions") or {}).get(iv.node_id) if run is not None else None
    if m is None or hw is None:
        return None
    key = (iv.node_id, tuple(hw))
    cache = run.store.setdefault("region_cache", {})
    if key not in cache:
        cache[key] = F.interpolate(m, size=tuple(hw), mode="area")[0, 0]
    return cache[key]


def _apply_region(run, iv, x):
    new = apply_op(x, iv, run, None)
    r = _region(run, iv, x.shape[-2:])
    return new if r is None else x + r.to(x.device, x.dtype) * (new - x)


def pca_rgb(feat: torch.Tensor, max_side: int = 64) -> np.ndarray:
    """Project C×h×w features onto their top-3 principal components → an RGB picture of what the stage 'sees'.
    Similar colours = similar feature vectors. Signs are fixed so colours don't flip between steps."""
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


def timestep_embedding(t: torch.Tensor, dim: int, flip_sin_to_cos=True, shift=0.0) -> torch.Tensor:
    half = dim // 2
    ex = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - shift)
    emb = t.float()[:, None] * torch.exp(ex)[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], -1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], -1)
    return emb


# ======================================================================================
# CLIP text encoder
# ======================================================================================

class CLIPText:
    def __init__(self, root: Path, device, dtype, policy):
        self.cfg = read_json(root / "config.json")
        self.W = WeightStore(root, device, dtype, policy)
        self.n_layers = self.cfg["num_hidden_layers"]
        self.heads = self.cfg["num_attention_heads"]
        self.eps = self.cfg.get("layer_norm_eps", 1e-5)
        self.pre = "text_model." if self.W.has("text_model.embeddings.token_embedding.weight") else ""
        act = self.cfg.get("hidden_act", "quick_gelu")
        self.act = (lambda x: x * torch.sigmoid(1.702 * x)) if act == "quick_gelu" else F.gelu

    def __call__(self, ids: list[int]) -> torch.Tensor:
        W, pre = self.W, self.pre
        n = len(ids)
        x = W.rows(pre + "embeddings.token_embedding.weight", ids) + W(pre + "embeddings.position_embedding.weight")[:n]
        x = x[None]
        mask = torch.full((n, n), float("-inf"), device=x.device).triu(1)
        for i in range(self.n_layers):
            lp = f"{pre}encoder.layers.{i}."
            h = layer_norm(W, lp + "layer_norm1", x, self.eps)
            q, k, v = (linear(W, lp + f"self_attn.{p}_proj", h) for p in "qkv")
            b, _, c = q.shape
            d = c // self.heads
            qh, kh, vh = (t.view(b, n, self.heads, d).transpose(1, 2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask.to(qh.dtype))
            x = x + linear(W, lp + "self_attn.out_proj", a.transpose(1, 2).reshape(b, n, c))
            h = layer_norm(W, lp + "layer_norm2", x, self.eps)
            x = x + linear(W, lp + "mlp.fc2", self.act(linear(W, lp + "mlp.fc1", h)))
        return layer_norm(W, pre + "final_layer_norm", x, self.eps)


# ======================================================================================
# UNet
# ======================================================================================

class UNet:
    def __init__(self, root: Path, device, dtype, policy):
        self.cfg = c = read_json(root / "config.json")
        self.W = W = WeightStore(root, device, dtype, policy)
        keys = W.keys()
        self.groups = c.get("norm_num_groups", 32)
        self.eps = c.get("norm_eps", 1e-5)
        heads = c.get("num_attention_heads") or c.get("attention_head_dim", 8)
        n_down = len(c["block_out_channels"])
        heads = heads if isinstance(heads, list) else [heads] * n_down
        self.linear_proj = bool(c.get("use_linear_projection", False))

        def count(pre, what):
            idx = {int(m.group(1)) for k in keys for m in [re.match(re.escape(pre) + rf"\.{what}\.(\d+)\.", k)] if m}
            return max(idx) + 1 if idx else 0

        self.blocks = []    # [(name, kind, n_res, n_attn, heads, has_sampler)]
        for i in range(n_down):
            p = f"down_blocks.{i}"
            self.blocks.append((p, "down", count(p, "resnets"), count(p, "attentions"), heads[i],
                                W.has(p + ".downsamplers.0.conv.weight")))
        if any(k.startswith("mid_block.") for k in keys):
            self.blocks.append(("mid_block", "mid", count("mid_block", "resnets"), count("mid_block", "attentions"),
                                heads[-1], False))
        rh = list(reversed(heads))
        for i in range(n_down):
            p = f"up_blocks.{i}"
            if not any(k.startswith(p + ".") for k in keys):
                continue
            self.blocks.append((p, "up", count(p, "resnets"), count(p, "attentions"), rh[i],
                                W.has(p + ".upsamplers.0.conv.weight")))
        self.in_channels = c.get("in_channels", 4)
        self.flip = c.get("flip_sin_to_cos", True)
        self.shift = c.get("freq_shift", 0)
        self.temb_dim = c["block_out_channels"][0]

    def transformer(self, pre, x, ctx_emb, heads, record, run=None, drop=False, tap=None):
        W = self.W
        b, ch, h, w = x.shape
        res = x
        x = F.group_norm(x.float(), self.groups, W(pre + ".norm.weight").float(), W(pre + ".norm.bias").float(), 1e-6).to(x.dtype)
        if self.linear_proj:
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, ch)
            x = linear(W, pre + ".proj_in", x)
        else:
            x = conv(W, pre + ".proj_in", x, padding=0)
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        nb = len({k.split(".transformer_blocks.")[1].split(".")[0] for k in W.keys()
                  if k.startswith(pre + ".transformer_blocks.")})
        for j in range(nb):
            tp = f"{pre}.transformer_blocks.{j}"
            hh = layer_norm(W, tp + ".norm1", x)
            x = x + linear(W, tp + ".attn1.to_out.0", attention(
                linear(W, tp + ".attn1.to_q", hh), linear(W, tp + ".attn1.to_k", hh), linear(W, tp + ".attn1.to_v", hh), heads))
            hh = layer_norm(W, tp + ".norm2", x)
            rec = (lambda probs, s=(h, w): record(pre, probs, s)) if record else None
            x = x + linear(W, tp + ".attn2.to_out.0", attention(
                linear(W, tp + ".attn2.to_q", hh), linear(W, tp + ".attn2.to_k", ctx_emb),
                linear(W, tp + ".attn2.to_v", ctx_emb), heads, rec, self._word_edit(run, ctx_emb.shape[1], (h, w))))
            hh = layer_norm(W, tp + ".norm3", x)
            a, gate = linear(W, tp + ".ff.net.0.proj", hh).chunk(2, -1)
            x = x + linear(W, tp + ".ff.net.2", a * F.gelu(gate))
        if self.linear_proj:
            x = linear(W, pre + ".proj_out", x)
            x = x.reshape(b, h, w, ch).permute(0, 3, 1, 2)
        else:
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
            x = conv(W, pre + ".proj_out", x, padding=0)
        if tap is not None:
            tap(x)
        return res if drop else x + res

    @staticmethod
    def _word_edit(run, m, hw=None):
        """Intervention hook 'word': scale the attention every image patch pays to chosen prompt tokens.
        0 removes a word's influence, 2 doubles it (prompt-to-prompt style re-weighting). With a painted region,
        only the patches inside it are changed."""
        ivs = run.active("word") if run is not None else []
        if not ivs:
            return None

        def edit(probs):
            probs = probs.clone()
            cond = slice(probs.shape[0] - 1, probs.shape[0]) if run.store.get("cfg") else slice(None)
            for iv in ivs:
                idx = parse_index_spec(iv.positions, m)
                f = iv.value if iv.op == "scale" else 0.0 if iv.op == "zero" else 1.0
                r = _region(run, iv, hw)
                if r is None:
                    probs[cond, :, :, idx] *= f
                else:
                    probs[cond, :, :, idx] *= (1 + r.reshape(1, 1, -1, 1).to(probs.dtype) * (f - 1))
            return probs
        return edit

    def __call__(self, x, t, ctx_emb, run: RunContext, record=None):
        W = self.W
        tt = torch.full((x.shape[0],), float(t), device=x.device)
        temb = timestep_embedding(tt, self.temb_dim, self.flip, self.shift).to(x.dtype)
        temb = linear(W, "time_embedding.linear_2", F.silu(linear(W, "time_embedding.linear_1", temb)))
        x = conv(W, "conv_in", x)
        skips = [x]
        want_maps = run.store.get("maps", True)
        for pos, (name, kind, n_res, n_attn, heads, has_s) in enumerate(self.blocks):
            run.pos = pos
            drop = run.store.get("ablate") == pos and run.step in run.store.get("ablate_steps", set())
            writes = []

            def tap(d):
                if want_maps:   # per-pixel size of what this stage writes into its residual stream (conditional half)
                    writes.append(d[-1].float().norm(dim=0))
            R = lambda pre, x: resnet(W, pre, x, temb, self.groups, self.eps, drop, tap)            # noqa: E731
            A = lambda pre, x: self.transformer(pre, x, ctx_emb, heads, record, run, drop, tap)    # noqa: E731
            if kind == "down":
                outs = []
                for j in range(n_res):
                    x = R(f"{name}.resnets.{j}", x)
                    if j < n_attn:
                        x = A(f"{name}.attentions.{j}", x)
                    outs.append(x)
                if has_s:
                    x = conv(W, f"{name}.downsamplers.0.conv", x, stride=2, padding=1)
                    outs.append(x)
                for iv in run.active("out"):
                    x = _apply_region(run, iv, x)
                    if outs:
                        outs[-1] = x
                for iv in run.active("skip"):
                    outs = [_apply_region(run, iv, o) for o in outs]
                skips += outs
            elif kind == "mid":
                x = R(f"{name}.resnets.0", x)
                for j in range(1, n_res):
                    if j - 1 < n_attn:
                        x = A(f"{name}.attentions.{j - 1}", x)
                    x = R(f"{name}.resnets.{j}", x)
                for iv in run.active("out"):
                    x = _apply_region(run, iv, x)
            else:
                for j in range(n_res):
                    x = torch.cat([x, skips.pop()], 1)
                    x = R(f"{name}.resnets.{j}", x)
                    if j < n_attn:
                        x = A(f"{name}.attentions.{j}", x)
                if has_s:
                    x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest").to(x.dtype)
                    x = conv(W, f"{name}.upsamplers.0.conv", x)
                for iv in run.active("out"):
                    x = _apply_region(run, iv, x)
            run.store.setdefault("norms", {}).setdefault(pos, {})[run.step] = float(x.float().std())
            if want_maps and writes:
                wm = torch.stack(writes).sum(0)
                run.store.setdefault("stage_maps", {}).setdefault(pos, {})[run.step] = {
                    "writes": wm.to(torch.float16).cpu(), "total": float(wm.sum()), "pca": pca_rgb(x[-1])}
            if run.on_block:
                run.on_block(pos)
        x = F.silu(group_norm(W, "conv_norm_out", x, self.groups, self.eps))
        return conv(W, "conv_out", x)


# ======================================================================================
# VAE
# ======================================================================================

class VAE:
    def __init__(self, root: Path, device, dtype, policy):
        self.cfg = c = read_json(root / "config.json")
        # The VAE runs in float32 on CPU and in the model dtype elsewhere (fp16 VAEs can overflow; keep fp32 if asked).
        self.W = WeightStore(root, device, dtype, policy)
        self.scale = c.get("scaling_factor", 0.18215)
        self.n_up = len(c["block_out_channels"])
        self.factor = 2 ** (self.n_up - 1)
        self.latent_ch = c.get("latent_channels", 4)
        self.groups = c.get("norm_num_groups", 32)
        keys = self.W.keys()
        self.dec_res = 1 + max(int(m.group(1)) for k in keys for m in [re.match(r"decoder\.up_blocks\.0\.resnets\.(\d+)\.", k)] if m)
        self.enc_res = 1 + max(int(m.group(1)) for k in keys for m in [re.match(r"encoder\.down_blocks\.0\.resnets\.(\d+)\.", k)] if m)

    def _attn(self, pre, x):
        W = self.W
        b, c, h, w = x.shape
        res = x
        x = group_norm(W, pre + ".group_norm", x, self.groups, 1e-6)
        x = x.view(b, c, h * w).transpose(1, 2)
        names = ("to_q", "to_k", "to_v", "to_out.0") if W.has(pre + ".to_q.weight") else ("query", "key", "value", "proj_attn")
        q, k, v = (linear(W, f"{pre}.{n}", x) for n in names[:3])
        o = linear(W, f"{pre}.{names[3]}", attention(q, k, v, 1))
        return o.transpose(1, 2).reshape(b, c, h, w) + res

    def _mid(self, pre, x):
        x = resnet(self.W, pre + ".resnets.0", x, None, self.groups, 1e-6)
        x = self._attn(pre + ".attentions.0", x)
        return resnet(self.W, pre + ".resnets.1", x, None, self.groups, 1e-6)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        W = self.W
        z = z / self.scale
        if W.has("post_quant_conv.weight"):
            z = conv(W, "post_quant_conv", z, padding=0)
        x = conv(W, "decoder.conv_in", z)
        x = self._mid("decoder.mid_block", x)
        for i in range(self.n_up):
            for j in range(self.dec_res):
                x = resnet(W, f"decoder.up_blocks.{i}.resnets.{j}", x, None, self.groups, 1e-6)
            if W.has(f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"):
                x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest").to(x.dtype)
                x = conv(W, f"decoder.up_blocks.{i}.upsamplers.0.conv", x)
        x = F.silu(group_norm(W, "decoder.conv_norm_out", x, self.groups, 1e-6))
        return conv(W, "decoder.conv_out", x)

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        W = self.W
        x = conv(W, "encoder.conv_in", img)
        for i in range(self.n_up):
            for j in range(self.enc_res):
                x = resnet(W, f"encoder.down_blocks.{i}.resnets.{j}", x, None, self.groups, 1e-6)
            if W.has(f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"):
                x = F.pad(x, (0, 1, 0, 1))
                x = conv(W, f"encoder.down_blocks.{i}.downsamplers.0.conv", x, stride=2, padding=0)
        x = self._mid("encoder.mid_block", x)
        x = F.silu(group_norm(W, "encoder.conv_norm_out", x, self.groups, 1e-6))
        x = conv(W, "encoder.conv_out", x)
        if W.has("quant_conv.weight"):
            x = conv(W, "quant_conv", x, padding=0)
        mean = x[:, : self.latent_ch]
        return mean * self.scale


# ======================================================================================
# Euler scheduler (diffusers EulerDiscreteScheduler semantics, s_churn = 0)
# ======================================================================================

class Euler:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        n = cfg.get("num_train_timesteps", 1000)
        bs, be = cfg.get("beta_start", 0.00085), cfg.get("beta_end", 0.012)
        sched = cfg.get("beta_schedule", "scaled_linear")
        betas = (np.linspace(bs ** 0.5, be ** 0.5, n, dtype=np.float64) ** 2) if sched == "scaled_linear" \
            else np.linspace(bs, be, n, dtype=np.float64)
        self.ac = np.cumprod(1 - betas)
        self.n_train = n
        self.pred = cfg.get("prediction_type", "epsilon")
        self.spacing = cfg.get("timestep_spacing", "linspace")
        self.offset = cfg.get("steps_offset", 0)

    def set_steps(self, steps: int):
        n = self.n_train
        if self.spacing == "trailing":
            ts = np.round(np.arange(n, 0, -n / steps)).astype(np.float64) - 1
        elif self.spacing == "linspace":
            ts = np.linspace(0, n - 1, steps, dtype=np.float64)[::-1].copy()
        else:  # leading
            ratio = n // steps
            ts = (np.arange(0, steps) * ratio).round()[::-1].astype(np.float64) + self.offset
        sig_all = ((1 - self.ac) / self.ac) ** 0.5
        sig = np.interp(ts, np.arange(len(sig_all)), sig_all)
        self.timesteps = ts
        self.sigmas = np.concatenate([sig, [0.0]])
        mx = float(self.sigmas.max())
        self.init_sigma = mx if self.spacing in ("linspace", "trailing") else (mx ** 2 + 1) ** 0.5

    def scale_input(self, x, i):
        return x / ((self.sigmas[i] ** 2 + 1) ** 0.5)

    def x0(self, x, out, i):
        s = self.sigmas[i]
        if self.pred == "v_prediction":
            return out * (-s / (s ** 2 + 1) ** 0.5) + x / (s ** 2 + 1)
        return x - s * out

    def step(self, x, out, i):
        x0 = self.x0(x.float(), out.float(), i)
        s, s_next = self.sigmas[i], self.sigmas[i + 1]
        d = (x.float() - x0) / s
        return (x.float() + d * (s_next - s)).to(x.dtype), x0.to(x.dtype)


# ======================================================================================
# Adapter used by the server
# ======================================================================================

class SDAdapter:
    family = "t2i"

    def __init__(self, spec: dict, root: Path, device: str, dtype: torch.dtype, policy="auto", keep_layers=0):
        root = Path(root)
        self.spec, self.root, self.device, self.dtype = spec, root, device, dtype
        self.tok = CLIPTokenizer(root / "tokenizer")
        self.text = CLIPText(root / "text_encoder", device, dtype, policy)
        self.unet = UNet(root / "unet", device, dtype, policy)
        self.vae = VAE(root / "vae", device, dtype, policy)
        self.sched_cfg = read_json(root / "scheduler" / "scheduler_config.json")
        idx = read_json(root / "model_index.json")
        self.pipeline = idx.get("_class_name", "StableDiffusionPipeline")
        if "XL" in self.pipeline or (root / "text_encoder_2").exists():
            raise ValueError("SDXL-style (two text encoders) pipelines are not supported by the raw engine yet.")
        self.sched_name = self.sched_cfg.get("_class_name", "")

    def info(self) -> dict:
        blocks = []
        for i, (name, kind, n_res, n_attn, heads, has_s) in enumerate(self.unet.blocks):
            short = name.replace("down_blocks.", "down_").replace("up_blocks.", "up_").replace("mid_block", "mid")
            sub = f"{n_res} res{f' · {n_attn} x-attn ({heads}h)' if n_attn else ''}{' · ↓' if kind == 'down' and has_s else ''}{' · ↑' if kind == 'up' and has_s else ''}"
            blocks.append({"id": short, "label": short.replace("_", " ").title(), "index": i, "sub": sub})
        size = int(self.spec.get("default_size") or self.unet.cfg.get("sample_size", 64) * self.vae.factor)
        return {
            "family": self.family, "arch": self.pipeline + " (raw)", "denoiser": "unet", "n_layers": len(blocks),
            "blocks": blocks, "tokenizer": "CLIP BPE", "text_encoder": f"CLIP text · {self.text.n_layers} layers",
            "vae": f"VAE ×{self.vae.factor}", "scheduler": f"Euler ({self.sched_cfg.get('timestep_spacing', 'linspace')})",
            "default_size": size, "default_steps": int(self.spec.get("default_steps", 20)),
            "default_guidance": float(self.spec.get("default_guidance", 7.0)),
            "weights": {"unet": self.unet.W.describe(), "text": self.text.W.describe(), "vae": self.vae.W.describe()},
            "capabilities": {"repeat_blocks": False, "reorder_blocks": False, "remove_blocks": False,
                             "xattn_maps": True, "resume_from_step": True, "img2img": True, "step_preview": True,
                             "regions": True},
            "hooks": ["out", "skip", "both", "word"],
        }

    def _encode_image(self, data_url: str, w: int, h: int) -> torch.Tensor:
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        im = Image.open(io.BytesIO(raw)).convert("RGB").resize((w, h), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        return self.vae.encode(x.to(self.device, self.dtype))

    def _load_mask(self, data_url: str, w: int, h: int) -> torch.Tensor:
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        im = Image.open(io.BytesIO(raw))
        a = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        paint = a[..., 3] * a[..., :3].max(-1)          # painted (opaque, bright) pixels = 1
        t = torch.from_numpy(paint)[None, None]
        t = F.interpolate(t, size=(h, w), mode="area")
        return t.to(self.device, self.dtype)

    def decode_png(self, latent: torch.Tensor, thumb: int | None = None) -> bytes | Image.Image:
        img = self.vae.decode(latent.to(self.device, self.dtype))
        img = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(img)
        if thumb:
            im.thumbnail((thumb, thumb))
        return im

    @torch.no_grad()
    def run(self, plan: Plan, progress=None, resume=None, prefix=None, ctx_hook=None) -> dict:
        p = plan.params
        set_determinism(p["seed"])
        steps, g = p["steps"], p["guidance_scale"]
        cfg = g > 1.0
        run = RunContext(interventions=plan.interventions, n_steps=steps)
        if ctx_hook:
            ctx_hook(run)
        run.store["cfg"] = cfg
        ids, tokens = self.tok(p["prompt"])
        cond = self.text(ids)
        neg = self.text(self.tok(p.get("negative_prompt", ""))[0]) if cfg else None
        emb = torch.cat([neg, cond]) if cfg else cond
        emb2, switch = None, None
        if p.get("switch_prompt"):                 # prompt schedule: a second prompt from step k on
            c2 = self.text(self.tok(p["switch_prompt"])[0])
            emb2 = torch.cat([neg, c2]) if cfg else c2
            switch = int(p.get("switch_step", steps // 2))
        sch = Euler(self.sched_cfg)
        sch.set_steps(steps)
        h, w = p["height"] // self.vae.factor, p["width"] // self.vae.factor
        gen = torch.Generator(device="cpu").manual_seed(p["seed"])
        noise = torch.randn((1, self.unet.in_channels, h, w), generator=gen, dtype=torch.float32).to(self.device, self.dtype)
        latents_before, x0s, start = {}, {}, 0
        init, lock = None, None
        if resume:
            start = int(resume["step"])
            lat = resume["latent"].to(self.device, self.dtype)
            latents_before.update({k: v for k, v in resume["parent_latents"].items() if k < start})
            x0s.update({k: v for k, v in resume["parent_x0"].items() if k < start})
        elif p.get("init_image"):
            init = self._encode_image(p["init_image"], p["width"], p["height"])
            start = min(steps - 1, int(round(steps * (1 - p["strength"]))))
            lat = init + noise * float(sch.sigmas[start]) if start > 0 else noise * sch.init_sigma
        else:
            lat = noise * sch.init_sigma
        if p.get("init_image") and p.get("mask"):
            # Region lock: the image-side "forced tokens". Painted pixels → latent cells; after each locked step
            # those cells are pulled to the start image re-noised to the current noise level.
            init = init if init is not None else self._encode_image(p["init_image"], p["width"], p["height"])
            m = self._load_mask(p["mask"], w, h)
            if p.get("mask_mode", "keep") == "regenerate":
                m = 1 - m
            lock = {"m": m, "steps": set(parse_index_spec(p.get("lock_steps", "all"), steps)),
                    "s": float(p.get("lock_strength", 1.0))}
        run.store["regions"] = {iv.node_id: self._load_mask(iv.region, h, w) for iv in plan.interventions
                                if getattr(iv, "region", "")}
        raw = []

        def record(pre, probs, hw):
            pb = probs[-1] if cfg else probs[0]            # conditional half under CFG
            raw.append((".".join(pre.split(".")[:2]) if not pre.startswith("mid") else "mid_block",
                        run.step, pb.float().mean(0).detach(), hw))

        nb = len(self.unet.blocks)
        for i in range(start, steps):
            run.step = i
            run.on_block = (lambda pos, i=i: progress and progress(((i - start) + (pos + 1) / nb) / (steps - start + 1),
                                                                    f"step {i + 1}/{steps} · block {pos + 1}/{nb}"))
            latents_before[i] = lat.detach().float().cpu()   # fp32 so a branch from step i is exact
            inp = torch.cat([lat] * 2) if cfg else lat
            inp = sch.scale_input(inp, i)
            e = emb2 if emb2 is not None and i >= switch else emb
            out = self.unet(inp, sch.timesteps[i], e, run, record)
            if cfg:
                u, c = out.chunk(2)
                out = u + g * (c - u)
            lat, x0 = sch.step(lat, out, i)
            if lock is not None and i in lock["steps"]:
                target = init + noise * float(sch.sigmas[i + 1])
                lat = lat + lock["s"] * lock["m"] * (target - lat)
            x0s[i] = x0.detach().to(torch.float16).cpu()
            run.checkpoint()
        if progress:
            progress(0.99, "VAE decode")
        image = self.decode_png(lat)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        # linear latent → colour map fitted on this image: shows the noisy inputs x_t in colour (see step_input_preview)
        from .qimage import _fit_preview
        img_t = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)
        rgb_fit = _fit_preview(lat[0].float().cpu().reshape(lat.shape[1], -1).T, img_t, self.vae.factor)
        xattn = self._aggregate(raw, h, w)
        stats = {}
        for pos in range(nb):
            series = run.store.get("norms", {}).get(pos, {})
            stats[plan.schedule_nodes[pos]] = {"pos": pos, "layer": pos,
                                               "std_by_step": [round(series[s], 4) for s in sorted(series)]}
        groups = sorted(xattn.keys(), key=_group_order)
        return {
            "outputs": {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                        "tokens": tokens, "steps": steps, "start_step": start, "block_stats": stats,
                        "preview_steps": sorted(x0s.keys()),
                        "xattn_groups": groups, "timesteps": [float(t) for t in sch.timesteps],
                        "sigmas": [round(float(s), 4) for s in sch.sigmas[:-1]]},
            "captures": {"latents_before": latents_before, "x0": x0s, "xattn": xattn, "n_tokens": len(tokens),
                         "stage_maps": run.store.get("stage_maps", {}), "image_png": buf.getvalue(),
                         "rgb_fit": rgb_fit, "hw": (h, w), "sigmas": [float(x) for x in sch.sigmas]},
        }

    def _aggregate(self, raw, h, w):
        r_h, r_w = max(1, min(32, h // 2)), max(1, min(32, w // 2))
        acc, cnt = {}, {}
        for group, step, m, (gh, gw) in raw:           # m: [q, k]
            k = m.shape[1]
            mm = m.T.reshape(k, gh, gw)[None]
            mm = F.interpolate(mm, size=(r_h, r_w), mode="bilinear", align_corners=False)[0].cpu()
            for gname in (group, "all"):
                a = acc.setdefault(gname, {})
                a[step] = a.get(step, 0) + mm
                cnt.setdefault(gname, {})[step] = cnt.get(gname, {}).get(step, 0) + 1
        return {gname: {s: (v / cnt[gname][s]).to(torch.float16) for s, v in per.items()} for gname, per in acc.items()}

    def step_input_preview(self, cap: dict, step: int) -> bytes:
        """The noisy latent the UNet is given at a step, in colour via a linear map (one square per latent pixel).
        Scaled by 1/√(σ²+1) like the UNet's input, so it reads as a blend of picture and noise."""
        xt = cap["latents_before"].get(step)
        if xt is None or "rgb_fit" not in cap:
            raise KeyError(step)
        h, w = cap["hw"]
        s = cap["sigmas"][step]
        x = (xt[0].float().reshape(xt.shape[1], -1).T) / (s * s + 1) ** 0.5
        rgb = torch.cat([x, torch.ones(x.shape[0], 1)], 1) @ cap["rgb_fit"]
        arr = ((rgb.clamp(-1, 1) + 1) * 127.5).round().byte().reshape(h, w, 3).numpy()
        im = Image.fromarray(arr).resize((w * 4, h * 4), Image.NEAREST)
        im.thumbnail((256, 256), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def step_preview(self, cap: dict, step: int) -> bytes:
        lat = cap["x0"].get(step)
        if lat is None:
            raise KeyError(step)
        im = self.decode_png(lat, thumb=256)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def stage_maps(self, cap: dict, step: int, node_ids: list[str]) -> dict:
        """Per UNet stage at one step: where it writes (heat map) and what its output encodes (PCA colours)."""
        out = []
        sm = cap.get("stage_maps", {})
        mx = max((v[step]["writes"].float().max().item() for v in sm.values() if step in v), default=1.0) or 1.0
        for pos, nid in enumerate(node_ids):
            d = sm.get(pos, {}).get(step)
            if d is None:
                continue
            wm = (d["writes"].float() / mx).clamp(0, 1)
            out.append({"node": nid, "pos": pos, "total": round(d["total"], 2),
                        "writes": _heat_png(wm.numpy()), "pca": _png(Image.fromarray(d["pca"])),
                        "res": list(d["pca"].shape[:2])})
        return {"step": step, "stages": out}

    @torch.no_grad()
    def contributions(self, plan: Plan, base_png: bytes, steps_spec: str, progress=None) -> list[dict]:
        """Ablation: re-run the generation once per stage with that stage's residual writes removed (for the
        chosen steps) and measure how much the final image changes. The honest answer to
        'which block contributed most'."""
        base = np.asarray(Image.open(io.BytesIO(base_png)).convert("RGB"), dtype=np.float32) / 255
        steps = set(parse_index_spec(steps_spec, plan.params["steps"]))
        results = []
        nb = len(self.unet.blocks)
        for pos in range(nb):
            def hook(ctx, pos=pos):
                ctx.store.update(ablate=pos, ablate_steps=steps, maps=False)
            res = self.run(plan, progress=(lambda f, m, pos=pos: progress and progress((pos + f) / nb, f"stage {pos + 1}/{nb}: {m}")),
                           ctx_hook=hook)
            png = base64.b64decode(res["outputs"]["image"].split(",", 1)[1])
            img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255
            diff = np.abs(img - base).mean(-1)
            results.append({"node": plan.schedule_nodes[pos], "pos": pos, "score": round(float(diff.mean()), 4),
                            "image": res["outputs"]["image"], "diff": _heat_png(np.clip(diff / max(diff.max(), 1e-6), 0, 1)),
                            "max_diff": round(float(diff.max()), 3)})
        return results

    def xattn_map(self, cap: dict, group: str, step: int | None, token: int) -> dict:
        per = cap["xattn"].get(group)
        if not per:
            return {"error": f"No cross-attention captured for {group}."}
        if step is None or step < 0:
            m = torch.stack([v[token].float() for v in per.values()]).mean(0)
        elif step in per:
            m = per[step][token].float()
        else:
            return {"error": f"No cross-attention at step {step}."}
        m = m / (m.max() + 1e-8)
        return {"h": m.shape[0], "w": m.shape[1], "map": [[round(float(v), 3) for v in row] for row in m]}

    def unload(self):
        self.unet = self.text = self.vae = None


def _png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _heat_png(m: np.ndarray) -> str:
    """0..1 map → black-red-yellow-white heat colours."""
    r = np.clip(m * 3, 0, 1)
    g = np.clip(m * 3 - 1, 0, 1)
    b = np.clip(m * 3 - 2, 0, 1)
    return _png(Image.fromarray((np.stack([r, g, b], -1) * 255).astype(np.uint8)))


def _group_order(gname: str):
    if gname == "all":
        return (0, 0)
    if gname.startswith("down"):
        return (1, int(gname.split(".")[-1]))
    if gname.startswith("mid"):
        return (2, 0)
    return (3, int(gname.split(".")[-1]))
