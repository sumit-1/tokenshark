"""Qwen-Image 2.1 in raw PyTorch (no diffusers / transformers at run time).

The model, piece by piece:

  prompt ─► Qwen3-VL text model (36 blocks) ─► hidden states of the last block, *before* its final norm,
            with the fixed system turn dropped ─► txt_in (zero-centred RMSNorm → Linear → GELU → Linear)
  noise latent [64 ch, H/16 × W/16] ─► img_in (Linear 64 → 4096), one token per 16×16 pixel tile

  ONE sequence [text tokens | image tokens] through 32 single-stream DiT blocks. Every block uses the same
  weights for text and image tokens. Attention is *block-causal*: text tokens see only earlier text (like an
  LLM), image tokens see all text plus every other image token.

  Timestep conditioning ("modulation") is one shared Linear for all blocks: LayerNorm(x)·(1 + scale) before
  attention / MLP and tanh(gate) on what they write back. Text tokens always get the t = 0 modulation, so they
  never depend on the denoising step: the text is pushed through the 32 blocks ONCE and its keys/values are
  kept — exactly an LLM's KV cache — while every denoising step only computes the image tokens.

  proj_out → velocity v (flow matching: x_t = (1-σ)·x₀ + σ·noise, v ≈ noise − x₀)
  Euler: x ← x + (σ_next − σ)·v ;  σ schedule = linspace(1, 1/N) → exponential shift (resolution dependent)
                                                → stretched to end at 0.02
  VAE decoder (Wan-style, 2D for single images): 64-ch latent → RGBA, ×16.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import (
    apply_rope_complex as _apply_rope,
    changed_tile_weights as _changed_tiles,
    channel_rms_norm,
    fit_linear_preview as _fit_preview,
    heat_data_url as _heat_png,
    pca_rgb,
    pil_to_data_url as _png,
    render_linear_preview as _lin_pil,
    rms_norm,
    rope_complex_axis as _rope_table,
    tensor_to_pil as _to_pil,
)
from .engine import Cancelled, RunContext, apply_op, set_determinism
from .flow import Plan, parse_index_spec
from .kvcache import KVCache
from .tokenizer import LLMTokenizer
from .weights import WeightStore, read_json, report_phase

SYS_PROMPT = "Comprehend and analyze the provided prompt."
T2I_TEMPLATE = "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


# ======================================================================================
# Text encoder: Qwen3-VL's language model, run with the raw LLM engine
# ======================================================================================

class QwenTextEncoder:
    def __init__(self, root: Path, tok_root: Path, device, dtype, policy):
        from .llm import RawLLM
        self.model = RawLLM(root, device, dtype, policy, 0)
        self.cfg = self.model.cfg
        self.tok = LLMTokenizer(tok_root)
        self.device = device
        # Number of leading system-turn tokens to drop — from the processor's own chat template, as diffusers does.
        sys_msg = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
        try:
            head = self.tok.apply_chat(sys_msg, add_generation_prompt=False)
        except Exception:
            head = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
        self.drop_idx = len(self.tok.encode(head, add_special=False))

    @torch.no_grad()
    def __call__(self, prompt: str, progress=None) -> tuple[torch.Tensor, list[str]]:
        """→ (hidden states [L, hidden] of the prompt tokens after the system turn, readable tokens)."""
        text = T2I_TEMPLATE.format(sys=SYS_PROMPT, prompt=prompt or " ")
        ids = self.tok.encode(text, add_special=False)
        m = self.model
        cache = KVCache({}, m.cfg.n_layers, device=self.device)
        cache.add_tokens(ids, [float(i) for i in range(len(ids))], "prompt")
        ctx = RunContext(interventions=[])
        x = m.embed(ids)
        for l in range(m.cfg.n_layers):
            x = m.block(l, l, x, cache, ctx, lambda iv: [])
            if progress:
                progress(l)
        # the last block's output BEFORE the final RMSNorm — what the DiT was trained on
        keep = ids[self.drop_idx:]
        return x[self.drop_idx:], [self.tok.piece(i) for i in keep]


# ======================================================================================
# Single-stream DiT (uses rope_complex_axis & apply_rope_complex from analyser/common.py)
# ======================================================================================

def _rms(x, w, eps):
    xf = x.float()
    y = x * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y * w


class QwenDiT:
    def __init__(self, root: Path, device, dtype, policy, keep_layers=0):
        self.cfg = cfg = read_json(root / "config.json")
        self.W = WeightStore(root, device, dtype, policy, keep_layers)
        self.device, self.dtype = device, dtype
        self.heads, self.hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
        self.dim = self.heads * self.hd
        self.n_layers = cfg["num_layers"]
        self.axes = list(cfg.get("axes_dims_rope", [16, 56, 56]))
        self.eps = cfg.get("eps", 1e-6)
        self.in_ch = cfg.get("in_channels", 64)
        self.mlp = self.dim * cfg.get("mlp_ratio", 3)
        if cfg.get("patch_size", 1) != 1:
            raise ValueError("Only Qwen-Image 2.1 (patch_size 1) is supported by this adapter.")
        if not cfg.get("causal_condition", True):
            raise ValueError("This adapter needs causal_condition=True (text modulated at t=0).")

    def lin(self, name, x):
        return x @ self.W(name + ".weight").T

    # ------------------------------------------------------------------ conditioning
    def temb(self, t: torch.Tensor) -> torch.Tensor:
        """t: sigmas in [0, 1], shape [B]. Sinusoids of 1000·t (cos first), then a 2-layer MLP."""
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
        args = (1000.0 * t.float())[:, None] * freqs[None].to(t.device)
        e = torch.cat([torch.cos(args), torch.sin(args)], -1).to(self.dtype)
        e = self.lin("time_text_embed.timestep_embedder.linear_1", e.to(self.device))
        return self.lin("time_text_embed.timestep_embedder.linear_2", F.silu(e))

    def modulation(self, temb: torch.Tensor):
        """→ per row of temb: (scale1, gate1, scale2, gate2), shared by every block."""
        m = self.lin("modulation.1", F.silu(temb))
        mod1, mod2 = m.chunk(2, -1)
        s1, g1 = mod1.chunk(2, -1)
        s2, g2 = mod2.chunk(2, -1)
        return s1, g1, s2, g2

    def text_in(self, h: torch.Tensor) -> torch.Tensor:
        w = self.W("txt_in.text_norm.weight").float() + 1
        hf = h.float()
        y = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(h.dtype)
        y = F.gelu(self.lin("txt_in.in_layer", y), approximate="tanh")
        return self.lin("txt_in.out_layer", y)

    def rope_text(self, pos: torch.Tensor) -> torch.Tensor:
        """Text tokens: the same position on all three axes (any integers, negative allowed)."""
        a0, a1, a2 = self.axes
        return torch.cat([_rope_table(pos, a0), _rope_table(pos, a1), _rope_table(pos, a2)], -1).to(self.device)

    def rope_img(self, frame: int, h: int, w: int) -> torch.Tensor:
        """Image tokens: frame axis frozen right after the prompt, height/width on a grid centred on zero."""
        hi = torch.arange(-(h - h // 2), h // 2).repeat_interleave(w)
        wi = torch.arange(-(w - w // 2), w // 2).repeat(h)
        fi = torch.full((h * w,), frame)
        a0, a1, a2 = self.axes
        return torch.cat([_rope_table(fi, a0), _rope_table(hi, a1), _rope_table(wi, a2)], -1).to(self.device)

    def rope(self, L: int, h: int, w: int):
        return self.rope_text(torch.arange(L)), self.rope_img(L, h, w)

    # ------------------------------------------------------------------ one block
    def qkv(self, pre, x, freqs):
        T = x.shape[0]
        q = self.lin(pre + "attn.to_q", x).view(T, -1, self.hd)
        k = self.lin(pre + "attn.to_k", x).view(T, -1, self.hd)
        v = self.lin(pre + "attn.to_v", x).view(T, -1, self.hd)
        q = _rms(q, self.W(pre + "attn.norm_q.weight"), self.eps).to(v.dtype)
        k = _rms(k, self.W(pre + "attn.norm_k.weight"), self.eps).to(v.dtype)
        return _apply_rope(q, freqs), _apply_rope(k, freqs), v

    def block(self, layer: int, x: torch.Tensor, mod, freqs, kv_text=None, causal=False, attn_fn=None, attn_mask=None):
        """x: the tokens being computed ([L, dim] text in the text pass, [N, dim] image in a denoising step).
        kv_text: cached text keys/values to put in front (image pass). Returns (new x, (k, v), writes)."""
        pre = f"transformer_blocks.{layer}."
        s1, g1, s2, g2 = mod
        n = F.layer_norm(x, (self.dim,), eps=self.eps) * (1 + s1)
        q, k, v = self.qkv(pre, n, freqs)
        K, V = (torch.cat([kv_text[0], k]), torch.cat([kv_text[1], v])) if kv_text is not None else (k, v)
        if attn_fn is not None:
            o = attn_fn(q, K, V)
        else:
            o = F.scaled_dot_product_attention(q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1),
                                               attn_mask=attn_mask, is_causal=causal and attn_mask is None).transpose(0, 1)
        o = self.lin(pre + "attn.to_out.0", o.reshape(x.shape[0], -1).type_as(q))
        d1 = torch.tanh(g1) * o
        x = x + d1
        n2 = F.layer_norm(x, (self.dim,), eps=self.eps) * (1 + s2)
        m = self.lin(pre + "img_mlp.out", F.silu(self.lin(pre + "img_mlp.gate_layer", n2)) * self.lin(pre + "img_mlp.proj", n2))
        d2 = torch.tanh(g2) * m
        return x + d2, (k, v), (d1, d2)

    def out_scale(self, temb_img):
        return self.lin("norm_out.linear", F.silu(temb_img))

    def final(self, x, temb_img=None, scale=None):
        scale = self.out_scale(temb_img) if scale is None else scale
        x = F.layer_norm(x, (self.dim,), eps=self.eps) * (1 + scale)
        return self.lin("proj_out", x)


# ======================================================================================
# VAE decoder (Wan-style; the image specialisation where every "3D" conv is a 2D conv)
# ======================================================================================

class QwenVAE:
    def __init__(self, root: Path, device, dtype, policy):
        self.cfg = c = read_json(root / "config.json")
        self.W = WeightStore(root, device, dtype, policy)
        self.device, self.dtype = device, dtype
        self.z = c.get("z_dim", 64)
        self.factor = int(c.get("scale_factor_spatial", 16))
        self.mean = torch.tensor(c["latents_mean"], dtype=torch.float32)
        self.std = torch.tensor(c["latents_std"], dtype=torch.float32)
        dim = c.get("decoder_base_dim") or c.get("base_dim", 96)
        mult = c.get("dim_mult", [1, 2, 4, 8, 8])
        self.dims = [dim * u for u in [mult[-1]] + mult[::-1]]
        self.n_res = c.get("num_res_blocks", 2)
        self.t_up = list(c.get("temperal_downsample", [False, True, True, True]))[::-1]
        self.n_up = len(mult)
        if not c.get("is_residual", True):
            raise ValueError("Only the residual Qwen-Image 2.1 VAE layout is supported.")
        if c.get("attn_scales"):
            raise ValueError("VAE attention at intermediate scales is not supported.")

    def conv(self, pre, x, pad=None, stride=1):
        w = self.W(pre + ".weight")
        b = self.W(pre + ".bias") if self.W.has(pre + ".bias") else None
        k = w.shape[-1]
        if w.ndim == 5:                    # a true 3D kernel: one frame → its last temporal slice (causal padding)
            w = w[:, :, -1]
        return F.conv2d(x, w, b, stride=stride, padding=k // 2 if pad is None else pad)

    def rms(self, pre, x):
        g = self.W(pre + ".gamma").reshape(1, -1, 1, 1)
        xn = F.normalize(x.float(), dim=1).to(x.dtype)
        return xn * (x.shape[1] ** 0.5) * g

    def res(self, pre, x):
        h = self.conv(pre + ".conv_shortcut", x, 0) if self.W.has(pre + ".conv_shortcut.weight") else x
        y = self.conv(pre + ".conv1", F.silu(self.rms(pre + ".norm1", x)))
        y = self.conv(pre + ".conv2", F.silu(self.rms(pre + ".norm2", y)))
        return y + h

    def attn(self, pre, x):
        B, C, H, Wd = x.shape
        n = self.rms(pre + ".norm", x)
        qkv = self.conv(pre + ".to_qkv", n, 0).reshape(B, 1, C * 3, H * Wd).permute(0, 1, 3, 2)
        q, k, v = qkv.chunk(3, -1)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.squeeze(1).permute(0, 2, 1).reshape(B, C, H, Wd)
        return self.conv(pre + ".proj", o, 0) + x

    @staticmethod
    def dup_up(x, out_dim, ft):
        """Parameter-free shortcut of an upsampling block: copy channels into 2×2 pixels (first frame only)."""
        B, Cin, H, Wd = x.shape
        r = out_dim * ft * 4 // Cin
        x = x.repeat_interleave(r, 1).view(B, out_dim, ft, 2, 2, H, Wd)[:, :, ft - 1]
        return x.permute(0, 1, 4, 2, 5, 3).reshape(B, out_dim, H * 2, Wd * 2)

    @staticmethod
    def avg_down(x, out_dim, ft, fs):
        """Parameter-free shortcut of a downsampling block: average groups of channels × 2×2 pixels. For a single
        frame the temporal factor pads zero frames in front (they take part in the average, as in the original)."""
        B, C, H, Wd = x.shape
        x5 = torch.stack([torch.zeros_like(x)] * (ft - 1) + [x], 2) if ft > 1 else x[:, :, None]
        x5 = x5.view(B, C, 1, ft, H // fs, fs, Wd // fs, fs).permute(0, 1, 3, 5, 7, 2, 4, 6)
        x5 = x5.reshape(B, C * ft * fs * fs, H // fs, Wd // fs)
        return x5.view(B, out_dim, -1, H // fs, Wd // fs).mean(2)

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """img [1, C, H, W] in [-1, 1] (C = the VAE's in_channels: RGBA for Qwen-Image 2.1) → normalised latent
        [1, z, H/16, W/16] (the distribution's mean, as the pipeline uses)."""
        c = self.cfg
        dim, mult = c.get("base_dim", 96), c.get("dim_mult", [1, 2, 4, 8, 8])
        dims = [dim * u for u in [1] + mult]
        tdown = list(c.get("temperal_downsample", [False, True, True, True]))
        n = len(mult)
        e = "encoder"
        x = self.conv(f"{e}.conv_in", img.to(self.device, self.dtype))
        for i in range(n):
            pre = f"{e}.down_blocks.{i}"
            copy = x
            for j in range(self.n_res):
                x = self.res(f"{pre}.resnets.{j}", x)
            down = i != n - 1
            if down:
                x = self.conv(f"{pre}.downsampler.resample.1", F.pad(x, (0, 1, 0, 1)), pad=0, stride=2)
            x = x + self.avg_down(copy, dims[i + 1], 2 if (down and tdown[i]) else 1, 2 if down else 1)
        x = self.res(f"{e}.mid_block.resnets.0", x)
        x = self.attn(f"{e}.mid_block.attentions.0", x)
        x = self.res(f"{e}.mid_block.resnets.1", x)
        x = self.conv(f"{e}.conv_out", F.silu(self.rms(f"{e}.norm_out", x)))
        x = self.conv("quant_conv", x, 0)
        mu = x[:, : self.z].float()
        return (mu - self.mean.view(1, -1, 1, 1).to(mu.device)) / self.std.view(1, -1, 1, 1).to(mu.device)

    @torch.no_grad()
    def decode(self, lat: torch.Tensor, progress=None) -> torch.Tensor:
        """lat: normalised latent [1, z, h, w] (what the DiT works in) → image [1, C, H, W] in [-1, 1]."""
        z = lat.float() * self.std.view(1, -1, 1, 1).to(lat.device) + self.mean.view(1, -1, 1, 1).to(lat.device)
        x = self.conv("post_quant_conv", z.to(self.device, self.dtype), 0)
        d = "decoder"
        x = self.conv(f"{d}.conv_in", x)
        x = self.res(f"{d}.mid_block.resnets.0", x)
        x = self.attn(f"{d}.mid_block.attentions.0", x)
        x = self.res(f"{d}.mid_block.resnets.1", x)
        for i in range(self.n_up):
            pre = f"{d}.up_blocks.{i}"
            copy = x
            for j in range(self.n_res + 1):
                x = self.res(f"{pre}.resnets.{j}", x)
            if i != self.n_up - 1:
                x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest-exact").type_as(x)
                x = self.conv(f"{pre}.upsampler.resample.1", x)
                x = x + self.dup_up(copy, self.dims[i + 1], 2 if self.t_up[i] else 1)
            if progress:
                progress(i)
        x = self.conv(f"{d}.conv_out", F.silu(self.rms(f"{d}.norm_out", x)))
        return x.clamp(-1, 1)


# ======================================================================================
# Scheduler: flow matching, Euler
# ======================================================================================

def flow_sigmas(steps: int, n_tokens: int, cfg: dict, sigmas=None) -> np.ndarray:
    s = np.array(sigmas if sigmas is not None else np.linspace(1.0, 1 / steps, steps), dtype=np.float32)
    if cfg.get("use_dynamic_shifting", True):
        b0, b1 = cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096)
        m0, m1 = cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15)
        k = (m1 - m0) / (b1 - b0)
        mu = n_tokens * k + (m0 - k * b0)
        if cfg.get("time_shift_type", "exponential") == "exponential":
            s = math.exp(mu) / (math.exp(mu) + (1 / s - 1) ** 1.0)
        else:
            s = mu / (mu + (1 / s - 1) ** 1.0)
    else:
        sh = cfg.get("shift", 1.0)
        s = sh * s / (1 + (sh - 1) * s)
    if cfg.get("shift_terminal"):
        om = 1 - s
        s = 1 - om / (om[-1] / (1 - cfg["shift_terminal"]))
    s = torch.from_numpy(np.asarray(s)).to(torch.float32)
    return torch.cat([s, torch.zeros(1)])


# ======================================================================================
# Adapter used by the server
# ======================================================================================

class QwenImageAdapter:
    family = "t2i"
    GROUP = 8          # blocks per attention-map group

    def __init__(self, spec: dict, root: Path, device: str, dtype: torch.dtype, policy="auto", keep_layers=0):
        root = Path(root)
        self.spec, self.root, self.device, self.dtype = spec, root, device, dtype
        te_policy = "disk" if policy in ("auto", "disk") else policy        # used once per prompt: stream it
        report_phase("text encoder (Qwen3-VL language model)")
        self.text = QwenTextEncoder(root / "text_encoder", root / "processor", device, dtype, te_policy)
        report_phase("diffusion transformer")
        self.dit = QwenDiT(root / "transformer", device, dtype, policy, keep_layers)
        report_phase("VAE")
        self.vae = QwenVAE(root / "vae", device, dtype, "cpu" if policy == "disk" else policy)
        self.sched_cfg = read_json(root / "scheduler" / "scheduler_config.json")
        self.pipeline = read_json(root / "model_index.json").get("_class_name", "QwenImage21Pipeline")

    def info(self) -> dict:
        n = self.dit.n_layers
        blocks = [{"id": f"block_{i}", "label": f"Block {i}", "index": i,
                   "sub": f"single-stream · {self.dit.heads}h · SwiGLU ×{self.dit.mlp // self.dit.dim}"} for i in range(n)]
        tc = self.text.cfg
        return {
            "family": self.family, "arch": "Qwen-Image 2.1 (raw)", "denoiser": "dit", "n_layers": n, "blocks": blocks,
            "tokenizer": "Qwen BPE · chat template",
            "text_encoder": f"Qwen3-VL text · {tc.n_layers} blocks · last block, pre-norm",
            "vae": f"Wan-style VAE ×{self.vae.factor} · {self.vae.z} ch", "scheduler": "flow matching · Euler · dynamic shift",
            "default_size": int(self.spec.get("default_size", 512)), "default_steps": int(self.spec.get("default_steps", 24)),
            "default_guidance": float(self.spec.get("default_guidance", 1.0)),
            "weights": {"transformer": self.dit.W.describe(), "text": self.text.model.W.describe(), "vae": self.vae.W.describe()},
            "capabilities": {"repeat_blocks": True, "reorder_blocks": True, "remove_blocks": True, "xattn_maps": True,
                             "resume_from_step": True, "img2img": True, "step_preview": True, "contributions": False,
                             "image_lens": True, "prompt_kv": True, "regions": True,
                             "sketch_guide": True},
            "hooks": ["out", "word"],
            "notes": "Text tokens are modulated at t=0, so the prompt is pushed through the 32 blocks once and its "
                     "keys/values are cached; each denoising step only computes the image tokens.",
        }

    # ------------------------------------------------------------------ prompts → per-block text keys/values
    def _te(self, prompt: str, progress=None):
        """Text-encoder output, remembered per prompt: the 8B encoder is the slow part of trying variations."""
        key = prompt or " "
        memo = self.__dict__.setdefault("_memo", {})
        if key not in memo:
            memo[key] = self.text(key, progress)
            while len(memo) > 16:
                memo.pop(next(iter(memo)))
        h, toks = memo[key]
        return self.dit.text_in(h.to(self.dit.dtype)), toks

    def _text_pass(self, emb, plan: Plan, mod0, pos: torch.Tensor, prefix=None, progress=None):
        """Push prompt tokens through the scheduled blocks once (t = 0 modulation): causal among themselves, and
        free to look at prefix[p] (another prompt's cached keys/values) in the blocks where one is given."""
        freqs = self.dit.rope_text(pos)
        L = emb.shape[0]
        kv, x = [], emb
        for p, layer in enumerate(plan.schedule):
            kvp = prefix[p] if prefix else None
            mask = None
            if kvp is not None:
                P = kvp[0].shape[0]
                mask = torch.cat([torch.ones(L, P, dtype=torch.bool), torch.tril(torch.ones(L, L, dtype=torch.bool))], 1).to(emb.device)
            x, (k, v), _ = self.dit.block(layer, x, mod0, freqs, kv_text=kvp, causal=kvp is None, attn_mask=mask)
            kv.append((k, v))
            if progress:
                progress(p)
        return kv

    def _build_text(self, prompt: str, plan: Plan, mod0, boxes: list, prog, f0, f1, label: str) -> dict:
        """The text side every image token reads, block by block.
        Prompt-KV boxes change it per block, like the KV-cache prefixes of the language models:
          prepend — a hidden prompt computed on its own; in the chosen blocks the main prompt *and* the image see it
                    in front of the main prompt (positions: shift = before it, negative = at −m…−1, offset = at k).
          replace — in the chosen blocks the image reads this prompt instead of the main one."""
        if not boxes:
            cache_key = (prompt or " ", tuple(plan.schedule))
            tmemo = self.__dict__.setdefault("_text_memo", {})
            if cache_key in tmemo:
                prog(f1, f"{label}: cached ({len(plan.schedule)} blocks)")
                return tmemo[cache_key]
        S = len(plan.schedule)
        tl, nb = self.text.cfg.n_layers, S
        span = (f1 - f0) / (1 + len(boxes))

        def te_prog(a, name):
            return lambda l: prog(a + span * 0.7 * (l + 1) / tl, f"{name}: text encoder block {l + 1}/{tl}")

        def pass_prog(a, name):
            return lambda q: prog(a + span * (0.7 + 0.3 * (q + 1) / nb), f"{name}: through block {q + 1}/{nb}")

        cursor = 0
        pre = []
        for j, b in enumerate(boxes):
            if b["role"] != "prepend":
                continue
            emb, toks = self._te(b["prompt"], te_prog(f0 + span * (j + 1), b["label"]))
            H = emb.shape[0]
            if b["mode"] == "negative":
                pos = torch.arange(-H, 0)
            elif b["mode"] == "offset":
                pos = torch.arange(H) + int(b["offset"])
            else:
                pos = torch.arange(H) + cursor
                cursor += H
            kv = self._text_pass(emb, plan, mod0, pos, None, pass_prog(f0 + span * (j + 1), b["label"]))
            pre.append((b, kv, H))
        prefix = []
        for p in range(S):
            parts = [kv[p] for b, kv, H in pre if p in b["slots"]]
            prefix.append((torch.cat([k for k, _ in parts]), torch.cat([v for _, v in parts])) if parts else None)
        emb, tokens = self._te(prompt, te_prog(f0, label))
        L = emb.shape[0]
        main_kv = self._text_pass(emb, plan, mod0, torch.arange(L) + cursor, prefix if any(prefix) else None,
                                  pass_prog(f0, label))
        frame = cursor + L
        own = [(main_kv[p], L, "main") for p in range(S)]
        for j, b in enumerate(boxes):
            if b["role"] != "replace":
                continue
            e2, _ = self._te(b["prompt"], te_prog(f0 + span * (j + 1), b["label"]))
            L2 = e2.shape[0]
            kv2 = self._text_pass(e2, plan, mod0, torch.arange(L2) + frame - L2, None, pass_prog(f0 + span * (j + 1), b["label"]))
            for p in b["slots"]:
                own[p] = (kv2[p], L2, b["label"])
        blocks, layout = [], []
        for p in range(S):
            (k, v), n, who = own[p]
            off = 0 if prefix[p] is None else prefix[p][0].shape[0]
            if prefix[p] is not None:
                k, v = torch.cat([prefix[p][0], k]), torch.cat([prefix[p][1], v])
            blocks.append((k, v))
            layout.append({"pos": p, "reads": who, "off": off, "n": n,
                           "prefix": [b["label"] for b, _, _ in pre if p in b["slots"]]})
        res = {"kv": blocks, "layout": layout, "frame": frame, "L": L, "tokens": tokens}
        if not boxes:
            tmemo = self.__dict__.setdefault("_text_memo", {})
            tmemo[(prompt or " ", tuple(plan.schedule))] = res
            while len(tmemo) > 8:
                tmemo.pop(next(iter(tmemo)))
        return res

    # ------------------------------------------------------------------ images in: start image and masks
    def _load_image(self, data_url: str, w: int, h: int) -> torch.Tensor:
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        ch = int(self.vae.cfg.get("in_channels", 3))
        im = Image.open(io.BytesIO(raw))
        if _has_transparency(im):
            # a drawing on the "noise canvas": the empty parts carry no picture. Put the strokes on neutral grey so
            # the encoder doesn't see a white page (only the drawn tiles are used — see _drawn_mask).
            grey = Image.new("RGBA", im.size, (128, 128, 128, 255))
            im = Image.alpha_composite(grey, im.convert("RGBA"))
        im = im.convert("RGBA" if ch == 4 else "RGB").resize((w, h), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        return self.vae.encode(x)                                             # [1, z, h/16, w/16]

    @staticmethod
    def _drawn_mask(data_url: str, h: int, w: int) -> torch.Tensor:
        """Which tiles of a picture are drawn on: pixels that differ from the background colour (the most common
        colour along the border), grown by about half a tile, then averaged per 16×16 tile → [h·w, 1] in 0..1."""
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        im = Image.open(io.BytesIO(raw))
        if _has_transparency(im):           # noise canvas: whatever is painted (alpha) is the drawing
            alpha = np.asarray(im.convert("RGBA"), dtype=np.float32)[..., 3] / 255.0
            drawn = torch.from_numpy((alpha > 0.1).astype(np.float32))[None, None]
            grow = max(1, alpha.shape[0] // (h * 2)) | 1
            drawn = F.max_pool2d(drawn, grow, stride=1, padding=grow // 2)
            return (F.interpolate(drawn, size=(h, w), mode="area")[0, 0] * 3).clamp(0, 1).reshape(-1, 1)
        a = np.asarray(im.convert("RGB"), dtype=np.int16)
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        vals, counts = np.unique(border // 8, axis=0, return_counts=True)
        bg = vals[counts.argmax()] * 8 + 4
        drawn = torch.from_numpy((np.abs(a - bg).max(-1) > 40).astype(np.float32))[None, None]
        grow = max(1, a.shape[0] // (h * 2)) | 1
        drawn = F.max_pool2d(drawn, grow, stride=1, padding=grow // 2)
        tiles = F.interpolate(drawn, size=(h, w), mode="area")[0, 0]
        return (tiles * 3).clamp(0, 1).reshape(-1, 1)

    @staticmethod
    def _load_mask(data_url: str, h: int, w: int) -> torch.Tensor:
        """Painted (opaque, bright) pixels → weight 1, averaged over each 16×16 tile → [h·w, 1]."""
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        a = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.float32) / 255.0
        paint = torch.from_numpy(a[..., 3] * a[..., :3].max(-1))[None, None]
        return F.interpolate(paint, size=(h, w), mode="area")[0, 0].reshape(-1, 1)

    # ------------------------------------------------------------------ one denoising step
    def _block_at(self, p, x, plan, run, text, mod_img, freqs_img, word_ivs, hw, step, regions, maps=None):
        """One scheduled block on the image tokens, with this run's interventions. → (x, (attn write, mlp write))."""
        dit = self.dit
        run.pos = p
        lay = text["layout"][p]
        ivs = [iv for iv in word_ivs if iv.pos == p and (iv.steps in ("all", "") or step in parse_index_spec(iv.steps, run.n_steps))]
        fn = None
        if maps is not None or ivs:
            def fn(q, K, V, lay=lay, ivs=ivs, p=p):
                return self._attn_capture(q, K, V, lay, ivs, maps, p, step, hw, regions)
        x_new, _, (d1, d2) = dit.block(plan.schedule[p], x, mod_img, freqs_img, kv_text=text["kv"][p], attn_fn=fn)
        for iv in run.active("out"):
            changed = apply_op(x_new, iv, run, None)
            m = regions.get(iv.node_id)
            x_new = changed if m is None else x_new + m.to(x_new.dtype) * (changed - x_new)
        return x_new, (d1, d2)

    def _velocity(self, lat, sigma, plan, run, text, temb_img, mod_img, freqs_img, capture, word_ivs, hw, step,
                  regions=None):
        dit = self.dit
        x = dit.lin("img_in", lat.to(dit.dtype))
        h, w = hw
        maps = run.store.setdefault("xattn_raw", []) if capture else None
        regions = regions or {}
        scale_out = dit.out_scale(temb_img) if capture else None
        lens = run.store.setdefault("lens", {}).setdefault(step, {}) if capture else None
        for p in range(len(plan.schedule)):
            run.checkpoint()
            x_new, (d1, d2) = self._block_at(p, x, plan, run, text, mod_img, freqs_img, word_ivs, hw, step, regions, maps)
            st = run.store.setdefault("norms", {}).setdefault(p, {})
            st[step] = float(x_new.float().std())
            if capture:
                wr = (d1.float().norm(dim=-1) + d2.float().norm(dim=-1)).reshape(h, w)
                run.store.setdefault("stage_maps", {}).setdefault(p, {})[step] = {
                    "writes": wr.to(torch.float16).cpu(), "total": float(wr.sum()),
                    "pca": pca_rgb(x_new.float().T.reshape(-1, h, w))}
                # image lens: what the model would output if it stopped after this block
                v_p = dit.final(x_new, scale=scale_out)
                lens[p] = (lat.float() - sigma * v_p.float()).to(torch.float16).cpu()
            x = x_new
            if run.on_block:
                run.on_block(p)
        return dit.final(x, temb_img)

    @torch.no_grad()
    def iter_lens_through(self, cap: dict, plan: Plan, through: int, steps: list[int] | None = None):
        """Image lens, decoded through the last `through` blocks: for each captured step and each block p, take the
        image tokens right after block p, jump into the last k blocks, then the final norm + projection → x̂₀.
        Like the text side's logit lens 'through k': middle blocks don't speak the output space yet, the last
        blocks translate. Recomputes the step from its stored input latent (no residuals are kept per block).
        Yields {"done": [[step, pos], …], "_progress": f} after every cell; results go to cap["lens_through"]."""
        dit, ctx = self.dit, cap["lens_ctx"]
        S = len(plan.schedule)
        steps = steps if steps is not None else sorted(cap.get("lens", {}).keys())
        store = cap.setdefault("lens_through", {}).setdefault(through, {})
        run = RunContext(interventions=plan.interventions, n_steps=ctx["n_steps"])
        word_ivs = [iv for iv in plan.interventions if iv.hook == "word"]
        h, w = cap["hw"]
        done = [[s_, p_] for s_ in store for p_ in store[s_]]
        total = len(steps) * S
        yield {"done": list(done), "_progress": len(done) / max(1, total)}
        for s_ in steps:
            if s_ in store and len(store[s_]) == S:
                continue
            run.step = s_
            sig = ctx["sig"]
            sigma = float(sig[s_])
            lat = cap["latents_before"][s_].to(self.device)
            temb = dit.temb(((sig[s_:s_ + 1] * 1000).to(dit.dtype) / 1000).to(self.device))
            mod, scale_out = dit.modulation(temb), dit.out_scale(temb)
            text = ctx["text2"] if ctx["switch"] is not None and s_ >= ctx["switch"] else ctx["text"]
            freqs = dit.rope_img(text["frame"], h, w)
            per = store.setdefault(s_, {})
            x = dit.lin("img_in", lat.to(dit.dtype))
            scratch = []          # same (hand-computed) attention as the captured step, so results match it bit for bit
            for p in range(S):
                x, _ = self._block_at(p, x, plan, run, text, mod, freqs, word_ivs, (h, w), s_, ctx["regions"], scratch)
                scratch.clear()
                if p not in per:
                    y = x
                    for q in range(max(p + 1, S - through), S):
                        y, _ = self._block_at(q, y, plan, run, text, mod, freqs, word_ivs, (h, w), s_, ctx["regions"], scratch)
                        scratch.clear()
                    per[p] = (lat.float() - sigma * dit.final(y, scale=scale_out).float()).to(torch.float16).cpu()
                    done.append([s_, p])
                yield {"done": list(done), "_progress": len(done) / max(1, total)}

    def _attn_capture(self, q, K, V, lay, ivs, maps, p, step, hw, regions):
        """Attention computed by hand (per head, to keep memory small) so we can (a) read how much each image
        token attends to each prompt token and (b) re-weight chosen words, optionally only inside a painted region."""
        H = q.shape[1]
        scale = 1.0 / math.sqrt(q.shape[-1])
        out = torch.empty_like(q)
        acc = None
        off, n = lay["off"], lay["n"]
        edits = []
        for iv in ivs:
            idx = [off + i for i in parse_index_spec(iv.positions, n)]
            if not idx:
                continue
            f = iv.value if iv.op == "scale" else 0.0 if iv.op == "zero" else 1.0
            m = regions.get(iv.node_id)
            edits.append((idx, f, None if m is None else m.to(q.device)))
        for hh in range(H):
            s = (q[:, hh].float() @ K[:, hh].float().T) * scale         # [N, text + N]
            pr = torch.softmax(s, -1)
            for idx, f, m in edits:
                pr[:, idx] = pr[:, idx] * (f if m is None else 1 + m.float() * (f - 1))
            out[:, hh] = (pr @ V[:, hh].float()).to(q.dtype)
            if maps is not None and lay["reads"] == "main":
                t = pr[:, off:off + n]
                acc = t if acc is None else acc + t
        if acc is not None:
            h, w = hw
            maps.append((p, step, (acc / H).T.reshape(n, h, w).to(torch.float16).cpu()))
        return out

    # ------------------------------------------------------------------ run
    @torch.no_grad()
    def run(self, plan: Plan, progress=None, resume=None, prefix=None, ctx_hook=None) -> dict:
        st = ImageStepper(self, plan, progress=progress, resume=resume, ctx_hook=ctx_hook)
        while st.step_block():
            pass
        return st.finish()

    def image_stepper(self, plan: Plan, resume=None, progress=None) -> "ImageStepper":
        return ImageStepper(self, plan, progress=progress, resume=resume, trace=True)

    def _aggregate(self, raw, nb):
        out: dict = {}
        cnt: dict = {}
        for p, step, m in raw:
            g0 = (p // self.GROUP) * self.GROUP
            for gname in ("all", f"blocks {g0}-{min(nb, g0 + self.GROUP) - 1}"):
                a = out.setdefault(gname, {})
                a[step] = a[step] + m.float() if step in a else m.float()
                cnt.setdefault(gname, {})[step] = cnt.get(gname, {}).get(step, 0) + 1
        return {g: {s: (v / cnt[g][s]).to(torch.float16) for s, v in per.items()} for g, per in out.items()}

    # ------------------------------------------------------------------ analysis endpoints
    def decode_png(self, latent: torch.Tensor, thumb: int | None = None):
        h, w = latent.shape[-2:] if latent.ndim == 4 else (None, None)
        img = self.vae.decode(latent)
        im = _to_pil(img[0])
        if thumb:
            im.thumbnail((thumb, thumb))
        return im

    def _ensure_fit(self, cap: dict):
        x0s = cap.get("x0") or {}
        latest_step = max(x0s.keys()) if x0s else -1
        if "rgb_fit" in cap and cap["rgb_fit"] is not None and cap.get("rgb_fit_step", latest_step) == latest_step:
            return cap["rgb_fit"]
        h, w = cap["hw"]
        lat = x0s[latest_step].float() if latest_step >= 0 else cap.get("final_latent")
        if lat is None:
            raise KeyError("no latent to fit preview")
        img = self.vae.decode(lat.float().T.reshape(1, self.vae.z, h, w))
        cap["rgb_fit"] = _fit_preview(lat.float().cpu(), img[0, :3].float().cpu(), self.vae.factor)
        cap["rgb_fit_step"] = latest_step
        if not cap.get("image_png"):
            im = _to_pil(img[0])
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            cap["image_png"] = buf.getvalue()
        return cap["rgb_fit"]

    def step_input_preview(self, cap: dict, step: int) -> bytes:
        """What the model is given at a step: the noisy latent x_t, shown with the same linear colour map (one
        square per 16×16 tile = one token). Linear, so it shows exactly (1−σ)·picture + σ·noise in colour."""
        xt = cap["latents_before"].get(step)
        if xt is None:
            raise KeyError(step)
        h, w = cap["hw"]
        A = self._ensure_fit(cap)
        rgb = torch.cat([xt.float(), torch.ones(xt.shape[0], 1)], 1) @ A
        arr = ((rgb.clamp(-1, 1) + 1) * 127.5).round().byte().reshape(h, w, 3).numpy()
        im = Image.fromarray(arr).resize((w * 8, h * 8), Image.NEAREST)
        im.thumbnail((256, 256), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def step_preview(self, cap: dict, step: int) -> bytes:
        """Fast preview of the model's guess x̂₀ at a step: a linear map latent → RGB, fitted on this run's own
        final image (a full VAE decode per step would cost as much as the decode at the end)."""
        x0 = cap["x0"].get(step)
        if x0 is None:
            raise KeyError(step)
        h, w = cap["hw"]
        A = self._ensure_fit(cap)
        rgb = torch.cat([x0.float(), torch.ones(x0.shape[0], 1)], 1) @ A           # [N, 3] in [-1, 1]
        arr = ((rgb.clamp(-1, 1) + 1) * 127.5).round().byte().reshape(h, w, 3).numpy()
        im = Image.fromarray(arr).resize((w * 4, h * 4), Image.BILINEAR)
        im.thumbnail((256, 256))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def _linear_png(self, cap: dict, x0: torch.Tensor, side: int = 256) -> bytes:
        h, w = cap["hw"]
        A = self._ensure_fit(cap)
        rgb = torch.cat([x0.float(), torch.ones(x0.shape[0], 1)], 1) @ A
        arr = ((rgb.clamp(-1, 1) + 1) * 127.5).round().byte().reshape(h, w, 3).numpy()
        im = Image.fromarray(arr).resize((w * 4, h * 4), Image.BILINEAR)
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def lens_png(self, cap: dict, step: int, pos: int, exact: bool = False, through: int = 0) -> bytes:
        """Image lens: x̂₀ the model would predict if it stopped after block `pos` (final norm + projection
        applied there, or after jumping through the last `through` blocks) at denoising step `step`.
        exact=True decodes it with the VAE; otherwise an approximate colour thumbnail."""
        src = cap.get("lens", {}) if through <= 0 else cap.get("lens_through", {}).get(through, {})
        x0 = src.get(step, {}).get(pos)
        if x0 is None:
            raise KeyError((step, pos))
        if not exact:
            return self._linear_png(cap, x0, 160)
        h, w = cap["hw"]
        im = _to_pil(self.vae.decode(x0.float().T.reshape(1, self.vae.z, h, w))[0])
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def stage_maps(self, cap: dict, step: int, node_ids: list[str]) -> dict:
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

    def contributions(self, *a, **k):
        raise ValueError("Block ablation would re-run the whole 7B generation once per block (32×). Bypass a block "
                         "in the flow (right-click → Bypass) and use 'Branch here' / compare runs instead.")

    def xattn_map(self, cap: dict, group: str, step: int | None, token: int) -> dict:
        per = cap["xattn"].get(group)
        if not per:
            return {"error": f"No text→image attention captured for {group}."}
        if step is None or step < 0:
            m = torch.stack([v[token].float() for v in per.values()]).mean(0)
        elif step in per:
            m = per[step][token].float()
        else:
            near = min(per, key=lambda s: abs(s - step))
            m = per[near][token].float()
        m = m / (m.max() + 1e-8)
        return {"h": m.shape[0], "w": m.shape[1], "map": [[round(float(v), 3) for v in row] for row in m]}

    def unload(self):
        self.dit = self.text = self.vae = None


def _has_transparency(im: Image.Image) -> bool:
    if im.mode not in ("RGBA", "LA") and not (im.mode == "P" and "transparency" in im.info):
        return False
    return int(np.asarray(im.convert("RGBA"))[..., 3].min()) < 250


def _group_order_dit(g: str):
    return (0, 0) if g == "all" else (1, int(g.split()[1].split("-")[0]))


# ======================================================================================
# The denoising loop, one block at a time (used by runs and by the diffusion step debugger)
# ======================================================================================

class ImageStepper:
    """Qwen-Image generation in its smallest steps:

        step i:  latent x_t → img_in → block 0 → … → block 31 → final → velocity → x̂₀ (the guess) → Euler → x_{t+1}

    step_block() advances one arrow; a step boundary is where x is None. edit_guess() is the image version of
    forcing a token: swap the model's guess x̂₀ for your edited picture while keeping the noise it is working
    with, so the next steps refine your edit instead of fighting it."""

    @torch.no_grad()
    def __init__(self, ad: "QwenImageAdapter", plan: Plan, progress=None, resume=None, ctx_hook=None, trace=False):
        self.ad, self.plan, self.trace_on = ad, plan, trace
        p = self.p_ = plan.params
        set_determinism(p["seed"])
        self.n_steps, self.g = p["steps"], p["guidance_scale"]
        neg_text = p.get("negative_prompt", "")
        self.cfg = self.g > 1.0 and bool(neg_text)
        run = self.ctx = RunContext(interventions=plan.interventions, n_steps=self.n_steps)
        if ctx_hook:
            ctx_hook(run)
        dit = ad.dit
        self.nb = nb = len(plan.schedule)
        mult = ad.vae.factor * 2
        self.Hpx, self.Wpx = max(mult, p["height"] // mult * mult), max(mult, p["width"] // mult * mult)
        h, w = self.hw = (self.Hpx // ad.vae.factor, self.Wpx // ad.vae.factor)
        self.N = N = h * w
        self.sig = sig = flow_sigmas(self.n_steps, N, ad.sched_cfg)
        self.prog = prog = progress or (lambda f, m: None)
        self.progress = None                       # set by the debugger for per-block progress
        self.boxes = boxes = [{**b, "slots": set(parse_index_spec(b.get("blocks", "all"), nb))} for b in p.get("prompt_kvs") or []]

        # -- 1. text: encoder → projection → one pass through the DiT blocks; keys/values kept per block
        mod0 = dit.modulation(dit.temb(torch.zeros(1, device=ad.device)))
        self.text = ad._build_text(p["prompt"], plan, mod0, boxes, prog, 0.0, 0.10, "prompt")
        self.text_neg = self.text2 = None
        if self.cfg:
            self.text_neg = ad._build_text(neg_text, plan, mod0, [], prog, 0.10, 0.12, "negative prompt")
        self.switch = None
        if p.get("switch_prompt"):
            self.text2 = ad._build_text(p["switch_prompt"], plan, mod0, boxes, prog, 0.12, 0.15, "second prompt")
            self.switch = int(p.get("switch_step", self.n_steps // 2))
        self._freqs = {}

        # -- 2. start latent: noise, a (noised) start image, or a parent run's latent
        gen = torch.Generator(device="cpu").manual_seed(p["seed"])
        noise = torch.randn((1, 1, ad.vae.z, h, w), generator=gen, dtype=torch.float32)
        self.noise = noise = noise.view(ad.vae.z, N).T.contiguous().to(ad.device)          # [N, z] (row = token)
        self.last_eps = noise.clone()
        self.noise_seed = int(p["seed"])
        self.lat = noise.to(dit.dtype)
        self.latents_before, self.x0s, self.start = {}, {}, 0
        init, lock = None, None
        if p.get("init_image"):
            prog(0.15, "VAE encode (start image)")
            init = ad._load_image(p["init_image"], self.Wpx, self.Hpx)[0].reshape(ad.vae.z, N).T.to(ad.device)
        if resume:
            self.start = int(resume["step"])
            self.lat = resume["latent"].to(ad.device, dit.dtype)
            self.latents_before.update({k: v for k, v in resume["parent_latents"].items() if k < self.start})
            self.x0s.update({k: v for k, v in resume["parent_x0"].items() if k < self.start})
        elif init is not None and not p.get("guide"):
            # flow matching: the latent at noise level σ is (1 − σ)·image + σ·noise — so "strength" picks σ to start at
            self.start = min(self.n_steps - 1, int(round(self.n_steps * (1 - p.get("strength", 0.6)))))
            s0 = sig[self.start]
            self.lat = ((1 - s0) * init.float() + s0 * noise).to(dit.dtype)
            self.guess, self.guess_at = init.float().cpu(), (self.start - 1, None)
        if init is not None and p.get("guide"):
            # Sketch guide: start from pure noise everywhere (step 0) and only steer the drawn tiles towards the
            # picture during the first steps; blank areas never see the picture, so the model invents them.
            n_guided = min(self.n_steps, int(round(self.n_steps * (1 - p.get("strength", 0.6)))))
            m = ad._load_mask(p["mask"], h, w).to(ad.device) if p.get("mask") else ad._drawn_mask(p["init_image"], h, w).to(ad.device)
            lock = {"m": m, "steps": set(range(n_guided)), "s": float(p.get("lock_strength", 1.0))}
            run.store["guide"] = {"steps": n_guided, "tiles": int((m > 0.05).sum()), "of": N}
        elif init is not None and p.get("mask"):
            m = ad._load_mask(p["mask"], h, w).to(ad.device)
            if p.get("mask_mode", "keep") == "regenerate":
                m = 1 - m
            lock = {"m": m, "steps": set(parse_index_spec(p.get("lock_steps", "all"), self.n_steps)), "s": float(p.get("lock_strength", 1.0))}
        self.init, self.lock = init, lock
        self.regions = {iv.node_id: ad._load_mask(iv.region, h, w).to(ad.device)
                        for iv in plan.interventions if getattr(iv, "region", "")}
        self.cap_steps = set(np.unique(np.linspace(self.start, self.n_steps - 1, min(6, max(1, self.n_steps - self.start))).round().astype(int)).tolist())
        self.word_ivs = [iv for iv in plan.interventions if iv.hook == "word"]

        # stepping state
        self.i, self.bp, self.x = self.start, 0, None
        self.done = self.start >= self.n_steps
        self.n_generated, self.steps = 0, 0           # finished denoising steps · debugger arrows taken
        self.total = self.n_steps - self.start
        self.trace: list[dict] = []
        if not hasattr(self, "guess"):
            self.guess = None                          # fp32 [N, z]: the current x̂₀ (after a step, or after a block)
            self.guess_at = None                       # (step, block or None)
        self.edits: list[dict] = []
        self.rgb_fit = None                            # linear latent → colour map, fitted on exact decodes
        self._fit_key = None                           # which guess_at self.rgb_fit was fitted on
        self._decoded = None                           # (key, PIL image) of the last exact decode of the guess
        self._lat_sigma = float(sig[self.start]) if self.start < len(sig) else 0.0
        if resume and (self.start - 1) in self.x0s:
            self.guess, self.guess_at = self.x0s[self.start - 1].float().cpu(), (self.start - 1, None)
        set_determinism(p["seed"])                     # reset global RNG so text-encoder memo hits never alter block ops

    # ---------------------------------------------------------------- helpers
    def _img_freqs(self, t):
        if t["frame"] not in self._freqs:
            self._freqs[t["frame"]] = self.ad.dit.rope_img(t["frame"], *self.hw)
        return self._freqs[t["frame"]]

    def _begin_step(self):
        ad, dit, run, i = self.ad, self.ad.dit, self.ctx, self.i
        run.step = i
        nb, start, steps = self.nb, self.start, self.n_steps
        run.on_block = (lambda q, i=i: self.prog(0.16 + 0.79 * ((i - start) + (q + 1) / nb) / (steps - start),
                                                  f"step {i + 1}/{steps} · block {q + 1}/{nb}"))
        self.latents_before[i] = self.lat.detach().float().cpu()
        self.s = float(self.sig[i])
        t_model = (self.sig[i:i + 1] * 1000).to(dit.dtype) / 1000        # the pipeline's timestep / 1000, bit for bit
        self.temb = dit.temb(t_model.to(ad.device))
        self.mod = dit.modulation(self.temb)
        self.t_use = self.text2 if self.switch is not None and i >= self.switch else self.text
        self.freqs = self._img_freqs(self.t_use)
        self.capture = i in self.cap_steps and not self.cfg
        self.maps = run.store.setdefault("xattn_raw", []) if self.capture else None
        self.scale_out = dit.out_scale(self.temb) if (self.capture or self.trace_on) else None
        self.lens = run.store.setdefault("lens", {}).setdefault(i, {}) if self.capture else None
        self.x = dit.lin("img_in", self.lat.to(dit.dtype))
        self.trace = []

    def _end_step(self):
        ad, dit, i = self.ad, self.ad.dit, self.i
        v = dit.final(self.x, self.temb)
        if self.cfg:
            vn = ad._velocity(self.lat, self.s, self.plan, self.ctx, self.text_neg, self.temb, self.mod,
                              self._img_freqs(self.text_neg), False, [], self.hw, i)
            v = vn + self.g * (v - vn)
        lf = self.lat.float()
        vf = v.float()
        x0 = lf - self.s * vf
        self.last_eps = (lf + (1.0 - self.s) * vf).detach()
        self.x0s[i] = x0.to(torch.float16).cpu()
        lf = lf + (self.sig[i + 1] - self.sig[i]).to(lf.device) * vf
        if self.lock is not None and i in self.lock["steps"]:
            # region lock — the image version of forced tokens: pull the painted tiles back to the start
            # image, noised to the level of the next step
            s1 = self.sig[i + 1].to(lf.device)
            target = (1 - s1) * self.init.float() + s1 * self.noise
            lf = lf + self.lock["s"] * self.lock["m"] * (target - lf)
        self.lat = lf.to(dit.dtype)
        self._lat_sigma = float(self.sig[i + 1])
        self.guess, self.guess_at = x0.cpu(), (i, None)
        self.x, self.bp = None, 0
        self.i += 1
        self.n_generated += 1
        if self.i >= self.n_steps:
            self.done = True

    # ---------------------------------------------------------------- stepping
    @torch.no_grad()
    def step_block(self) -> bool:
        """Advance one arrow. At a step boundary this starts the next step (its input goes into the blocks).
        Returns False when the generation is finished."""
        if self.ctx.cancel:
            raise Cancelled("Run cancelled.")
        if self.done:
            return False
        self.steps += 1
        if self.x is None:
            self._begin_step()
            return True
        run, p, dit = self.ctx, self.bp, self.ad.dit
        run.checkpoint()
        h, w = self.hw
        x_new, (d1, d2) = self.ad._block_at(p, self.x, self.plan, run, self.t_use, self.mod, self.freqs, self.word_ivs,
                                            self.hw, self.i, self.regions, self.maps)
        run.store.setdefault("norms", {}).setdefault(p, {})[self.i] = float(x_new.float().std())
        if self.capture:
            wr = (d1.float().norm(dim=-1) + d2.float().norm(dim=-1)).reshape(h, w)
            run.store.setdefault("stage_maps", {}).setdefault(p, {})[self.i] = {
                "writes": wr.to(torch.float16).cpu(), "total": float(wr.sum()),
                "pca": pca_rgb(x_new.float().T.reshape(-1, h, w))}
        if self.capture or self.trace_on:
            # image lens: what the model would output if it stopped after this block
            g = self.lat.float() - self.s * dit.final(x_new, scale=self.scale_out).float()
            if self.capture:
                self.lens[p] = g.to(torch.float16).cpu()
            self.guess, self.guess_at = g.cpu(), (self.i, p)
            self.trace.append({"pos": p, "node": self.plan.schedule_nodes[p],
                               "write": round(float((d1.float().norm(dim=-1) + d2.float().norm(dim=-1)).mean()), 3)})
        self.x = x_new
        if run.on_block:
            run.on_block(p)
        if self.progress:
            self.progress((self.n_generated + (p + 1) / self.nb) / max(1, self.total), "")
        self.bp += 1
        if self.bp == self.nb:
            self._end_step()
        return True

    def step_token(self) -> bool:            # = one whole denoising step (the debugger's ▸▸ button)
        if self.x is None and not self.step_block():
            return False
        while self.x is not None:
            self.step_block()
        return True

    def run_to(self, pos: int, limit: int = 100_000) -> bool:
        first = True
        for _ in range(limit):
            if not first and self.x is not None and self.bp == pos + 1:
                return True
            first = False
            if not self.step_block():
                return False
        return True

    def force(self, token_id):
        raise ValueError("Image generation has no tokens to force — edit the guess instead.")

    # ---------------------------------------------------------------- forcing the image
    @torch.no_grad()
    def extend_steps(self, target_steps: int | None = None, more_steps: int = 4, start_sigma: float | None = None) -> dict:
        """Add more denoising steps so a finished (or in-progress) generation can keep going."""
        if self.x is not None:
            raise ValueError("Finish the current step first (▸▸) before adding steps.")
        new_total = target_steps if target_steps is not None and target_steps > self.n_steps else (self.n_steps + max(1, more_steps))
        add = new_total - self.n_steps
        new_full = flow_sigmas(new_total, self.N, self.ad.sched_cfg)
        if self.i == 0:
            self.sig = new_full
        else:
            was_done = self.done or self.i >= self.n_steps
            if start_sigma is not None:
                cur_s = max(0.02, min(1.0, float(start_sigma)))
            elif not was_done and float(self.sig[self.i]) > 0.02:
                cur_s = float(self.sig[self.i])
            else:
                cur_s = max(0.25, float(new_full[-add - 1]))
            s_end = min(cur_s, float(new_full[-2]))
            rem = new_total - self.i
            tail = torch.linspace(cur_s, s_end, rem, dtype=torch.float32) if rem > 1 else torch.tensor([cur_s], dtype=torch.float32)
            self.sig = torch.cat([self.sig[:self.i], tail, torch.zeros(1, dtype=torch.float32)])
            if self.guess is not None:
                s_start = float(self.sig[self.i])
                x0 = self.guess.to(self.ad.device).float()
                self.lat = ((1.0 - s_start) * x0 + s_start * self.last_eps).to(self.ad.dit.dtype)
                self._lat_sigma = s_start
        self.n_steps = len(self.sig) - 1
        self.ctx.n_steps = self.n_steps
        self.total = self.n_steps - self.start
        self.done = self.i >= self.n_steps
        self.cap_steps.update(range(self.i, self.n_steps))
        return {"n_steps": self.n_steps, "step": self.i, "sigma": round(float(self.sig[self.i]), 4)}

    @torch.no_grad()
    def set_noise(self, sigma: float, fresh_noise: bool = False) -> dict:
        """Adjust the noise level σ ∈ [0.02, 1.0] for the upcoming step (between steps):
              x_t = (1 − σ)·x̂₀ + σ·ε̂
        Also rescales the remaining schedule from σ down to 0.02 so the DiT's timestep conditioning and
        subsequent Euler steps match the new noise level."""
        if self.x is not None:
            raise ValueError("Finish the current step first (▸▸) before changing the noise level.")
        sigma = max(0.02, min(1.0, float(sigma)))
        if self.done or self.i >= self.n_steps:
            self.extend_steps(more_steps=4, start_sigma=sigma)
        if fresh_noise:
            self.noise_seed = int(getattr(self, "noise_seed", self.p_["seed"])) + 997
            gen = torch.Generator(device="cpu").manual_seed(self.noise_seed)
            h, w = self.hw
            n = torch.randn((1, 1, self.ad.vae.z, h, w), generator=gen, dtype=torch.float32)
            self.last_eps = n.view(self.ad.vae.z, self.N).T.contiguous().to(self.ad.device)
        if self.guess is not None:
            x0 = self.guess.to(self.ad.device).float()
        elif self.init is not None:
            x0 = self.init.float()
        else:
            x0 = torch.zeros_like(self.last_eps)
        rem = self.n_steps - self.i
        s_end = min(sigma, float(self.ad.sched_cfg.get("shift_terminal", 0.02)))
        if rem > 1:
            self.sig[self.i:self.n_steps] = torch.linspace(sigma, s_end, rem, dtype=torch.float32)
        elif rem == 1:
            self.sig[self.i] = sigma
        self.lat = ((1.0 - sigma) * x0 + sigma * self.last_eps).to(self.ad.dit.dtype)
        self._lat_sigma = sigma
        return {"step": self.i, "n_steps": self.n_steps, "sigma": round(sigma, 4), "fresh_noise": fresh_noise}

    @torch.no_grad()
    def edit_guess(self, data_url: str, strength: float = 1.0, only_changed: bool = True, mask_url: str = "",
                   sigma: float | None = None) -> dict:
        """Swap the model's current guess x̂₀ for an edited picture, keeping the noise it is working with:
              x_t = (1 − σ)·x̂₀ + σ·ε̂   →   ε̂ = (x_t − (1 − σ)·x̂₀) / σ   →   x_t' = (1 − σ)·encode(edit) + σ·ε̂
        Only tiles you changed (or painted in mask_url) are swapped; strength blends."""
        if self.x is not None:
            raise ValueError("Finish the current step first (▸▸): an edit goes into the input of the next step.")
        ad, h, w = self.ad, *self.hw
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        after_pil = Image.open(io.BytesIO(raw)).convert("RGB").resize((self.Wpx, self.Hpx), Image.LANCZOS)

        if mask_url:
            m = ad._load_mask(mask_url, h, w).to(ad.device)
        elif only_changed and self._decoded is not None:
            m = _changed_tiles(self._decoded[1], data_url, ad.vae.factor, (h, w)).to(ad.device)
        else:
            m = torch.ones(self.N, 1, device=ad.device)
        m = m * float(strength)
        n_tiles = int((m > 0.01).sum())

        if n_tiles == 0 and sigma is None:
            cur_s = float(self.sig[min(self.i, self.n_steps - 1)])
            rec = {"step": self.i, "sigma": round(cur_s, 4), "tiles": 0, "of": self.N, "strength": float(strength)}
            self.edits.append(rec)
            self.ctx.store.setdefault("edits", []).append(rec)
            return rec

        if self.done or self.i >= self.n_steps:
            self.extend_steps(more_steps=4, start_sigma=sigma if sigma is not None else 0.35)

        old_sigma = float(getattr(self, "_lat_sigma", float(self.sig[self.i])))
        if self.guess is not None:
            x0 = self.guess.to(ad.device).float()
            eps = (self.lat.float() - (1.0 - old_sigma) * x0) / old_sigma if old_sigma > 1e-4 else self.last_eps
        else:
            x0 = None
            eps = self.last_eps

        if sigma is not None:
            target_s = max(0.02, min(1.0, float(sigma)))
            rem = self.n_steps - self.i
            s_end = min(target_s, float(ad.sched_cfg.get("shift_terminal", 0.02)))
            if rem > 1:
                self.sig[self.i:self.n_steps] = torch.linspace(target_s, s_end, rem, dtype=torch.float32)
            elif rem == 1:
                self.sig[self.i] = target_s

        cur_sigma = float(self.sig[self.i])
        z = ad._load_image(data_url, self.Wpx, self.Hpx)[0].reshape(ad.vae.z, self.N).T.to(ad.device).float()
        if x0 is None:
            x0 = z.clone()

        new_x0 = x0 + m * (z - x0)
        new_lat = (1.0 - cur_sigma) * z + cur_sigma * eps
        if sigma is not None:
            self.lat = ((1.0 - cur_sigma) * new_x0 + cur_sigma * eps).to(ad.dit.dtype)
        else:
            self.lat = (self.lat.float() + m * (new_lat - self.lat.float())).to(ad.dit.dtype)
        self._lat_sigma = cur_sigma
        self.last_eps = eps.detach()
        self.guess = new_x0.cpu()
        self.guess_at = (self.i - 1, None)
        if (self.i - 1) >= 0:
            self.x0s[self.i - 1] = self.guess.to(torch.float16)

        rec = {"step": self.i, "sigma": round(cur_sigma, 4), "tiles": n_tiles, "of": self.N,
               "strength": float(strength)}
        self.edits.append(rec)
        self.ctx.store.setdefault("edits", []).append(rec)

        # Keep the user's edited picture directly as the decoded view so saving is instant and exact
        key = (self.guess_at, len(self.edits))
        self._decoded = (key, after_pil)
        img_t = torch.from_numpy(np.asarray(after_pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        self.rgb_fit = _fit_preview(self.guess.float(), img_t, ad.vae.factor)
        self._fit_key = key
        return rec

    def guess_png(self, exact: bool = True) -> bytes:
        """The current guess as a picture. Exact = VAE decode (also refits the fast colour map)."""
        if self.guess is None:
            if self.x0s:
                last_s = max(self.x0s.keys())
                self.guess, self.guess_at = self.x0s[last_s].float().cpu(), (last_s, None)
            elif self.init is not None:
                self.guess, self.guess_at = self.init.float().cpu(), (self.start - 1, None)
            else:
                # Before step 0 with pure noise: show a neutral canvas ready to draw on
                im = Image.new("RGB", (self.Wpx, self.Hpx), (128, 128, 128))
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return buf.getvalue()
        h, w = self.hw
        key = (self.guess_at, len(self.edits))
        if exact:
            if self._decoded is None or self._decoded[0] != key:
                img = self.ad.vae.decode(self.guess.T.reshape(1, self.ad.vae.z, h, w))
                self._decoded = (key, _to_pil(img[0]).convert("RGB"))     # the picture you paint on: colour only
                self.rgb_fit = _fit_preview(self.guess.float(), img[0, :3].float().cpu(), self.ad.vae.factor)
                self._fit_key = key
            im = self._decoded[1]
        else:
            if self.rgb_fit is None:
                raise KeyError("no colour map yet")
            im = _lin_pil(self.guess, self.rgb_fit, (h, w), 4)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def input_png(self) -> bytes:
        """The input of the current / next step (x_t), in approximate colours."""
        if self.rgb_fit is None:
            if self.guess is not None:
                self.guess_png(exact=True)
            else:
                raise KeyError("no colour map yet")
        buf = io.BytesIO()
        _lin_pil(self.lat.float().cpu(), self.rgb_fit, self.hw, 8, nearest=True).save(buf, format="PNG")
        return buf.getvalue()

    def state(self) -> dict:
        i = min(self.i, self.n_steps - 1)
        return {"image": True, "done": self.done, "phase": "done" if self.done else ("idle" if self.x is None else "block"),
                "next_pos": self.bp if self.x is not None else None,
                "active_node": self.plan.schedule_nodes[self.bp] if self.x is not None else None,
                "trace": self.trace, "steps": self.steps, "step": self.i, "n_steps": self.n_steps, "start": self.start,
                "sigma": round(float(self.sig[self.i]), 4) if self.i < self.n_steps else 0.0,
                "n_generated": self.n_generated, "guess_at": self.guess_at, "edits": self.edits,
                "can_edit": self.x is None,
                "tokens": self.text["tokens"], "forced": None, "chunk": [], "n_in": 0, "text": ""}

    def snapshot(self) -> dict:
        """Outputs + captures for the current state, in the same shape a finished Run has, so every analysis
        view (Output, Step debugger, Image lens, Block contributions, Cross-attention) works while stepping."""
        ad, run, plan, nb = self.ad, self.ctx, self.plan, self.nb
        h, w = self.hw
        text, L, steps, start, N = self.text, self.text["L"], self.n_steps, self.start, self.N
        xattn = ad._aggregate(run.store.get("xattn_raw", []), nb)
        stats = {}
        for q in range(nb):
            series = run.store.get("norms", {}).get(q, {})
            if series:
                stats[plan.schedule_nodes[q]] = {"pos": q, "layer": plan.schedule[q],
                                                 "std_by_step": [round(series[s_], 4) for s_ in sorted(series)]}
        groups = sorted(xattn.keys(), key=_group_order_dit)
        lens = run.store.get("lens", {})
        img_b64 = ""
        img_bytes = b""
        key = (self.guess_at, len(self.edits))
        if self._decoded is not None and self._decoded[0] == key:
            buf = io.BytesIO()
            self._decoded[1].save(buf, format="PNG")
            img_bytes = buf.getvalue()
            img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        elif self.guess is not None and self.rgb_fit is not None and self._fit_key == key:
            buf = io.BytesIO()
            _lin_pil(self.guess, self.rgb_fit, (h, w), 4).save(buf, format="PNG")
            img_bytes = buf.getvalue()
            img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        elif self._decoded is not None:
            buf = io.BytesIO()
            self._decoded[1].save(buf, format="PNG")
            img_bytes = buf.getvalue()
            img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        n_main = sum(1 for l in text["layout"] if l["reads"] == "main" and not l["prefix"])
        note = (f"{L} prompt tokens were pushed through the {nb} blocks once; their keys/values were reused by all "
                f"{steps - start} steps, which only computed the {N} image tokens.")
        if self.boxes:
            note += f" Prompt-KV boxes changed what {nb - n_main} of the {nb} blocks read."
        if self.edits:
            note += " Edited by hand at step " + ", ".join(f"{e['step']} ({e['tiles']} tiles)" for e in self.edits) + "."
        g = run.store.get("guide")
        guide_note = (f"Sketch guide: started from pure noise; {g['tiles']} of {g['of']} tiles (the drawn ones) were steered "
                      f"towards your picture during the first {g['steps']} steps, everything else was free.") if g else ""
        sig = self.sig
        out = {
            "image": img_b64,
            "tokens": text["tokens"], "steps": steps, "start_step": start, "block_stats": stats,
            "preview_steps": sorted(self.x0s.keys()), "preview_kind": "linear",
            "xattn_groups": groups, "xattn_steps": sorted({s for _, s, _ in run.store.get("xattn_raw", [])}),
            "lens_steps": sorted(lens.keys()), "lens_blocks": nb,
            "prompt_layout": [{k: l[k] for k in ("pos", "reads", "prefix", "n")} for l in text["layout"]],
            "timesteps": [round(float(s_) * 1000, 2) for s_ in sig[:-1]],
            "sigmas": [round(float(s_), 4) for s_ in sig[:-1]],
            "size": [self.Wpx, self.Hpx], "text_tokens": L, "kv_note": note, "guide_note": guide_note,
            "edits": self.edits, "stepping": not self.done,
        }
        latest_step = max(self.x0s.keys()) if self.x0s else -1
        cur_x0 = self.guess.float().cpu() if self.guess is not None else self.lat.float().cpu()
        caps = {
            "latents_before": self.latents_before, "x0": self.x0s, "xattn": xattn, "n_tokens": len(text["tokens"]),
            "stage_maps": run.store.get("stage_maps", {}), "image_png": img_bytes,
            "hw": (h, w), "final_latent": cur_x0, "lens": lens,
            "sigmas": sig.clone(),
            "lens_ctx": {"sig": sig.clone(), "text": text, "text2": self.text2, "switch": self.switch,
                         "regions": self.regions, "n_steps": steps},
        }
        if self.rgb_fit is not None and self._fit_key == key:
            caps["rgb_fit"] = self.rgb_fit
            caps["rgb_fit_step"] = latest_step
        return {"outputs": out, "captures": caps}

    # ---------------------------------------------------------------- results
    @torch.no_grad()
    def finish(self) -> dict:
        ad, run, plan, nb = self.ad, self.ctx, self.plan, self.nb
        while self.step_block():                    # a finished run needs every step
            pass
        h, w = self.hw
        prog = self.prog
        prog(0.96, "VAE decode")
        z = self.lat.T.reshape(1, ad.vae.z, h, w)
        img = ad.vae.decode(z, lambda k: prog(0.96 + 0.03 * (k + 1) / ad.vae.n_up, f"VAE decode {k + 1}/{ad.vae.n_up}"))
        im = _to_pil(img[0])
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        rgb_fit = _fit_preview(self.lat.float().cpu(), img[0, :3].float().cpu(), ad.vae.factor)
        self._decoded = ((self.guess_at, len(self.edits)), im.convert("RGB"))
        self.rgb_fit = rgb_fit
        self._fit_key = (self.guess_at, len(self.edits))
        text, L, steps, start, N = self.text, self.text["L"], self.n_steps, self.start, self.N
        xattn = ad._aggregate(run.store.get("xattn_raw", []), nb)
        stats = {}
        for q in range(nb):
            series = run.store.get("norms", {}).get(q, {})
            stats[plan.schedule_nodes[q]] = {"pos": q, "layer": plan.schedule[q],
                                             "std_by_step": [round(series[s_], 4) for s_ in sorted(series)]}
        groups = sorted(xattn.keys(), key=_group_order_dit)
        lens = run.store.get("lens", {})
        n_main = sum(1 for l in text["layout"] if l["reads"] == "main" and not l["prefix"])
        note = (f"{L} prompt tokens were pushed through the {nb} blocks once; their keys/values were reused by all "
                f"{steps - start} steps, which only computed the {N} image tokens.")
        if self.boxes:
            note += f" Prompt-KV boxes changed what {nb - n_main} of the {nb} blocks read."
        if self.edits:
            note += " Edited by hand at step " + ", ".join(f"{e['step']} ({e['tiles']} tiles)" for e in self.edits) + "."
        g = run.store.get("guide")
        guide_note = (f"Sketch guide: started from pure noise; {g['tiles']} of {g['of']} tiles (the drawn ones) were steered "
                      f"towards your picture during the first {g['steps']} steps, everything else was free.") if g else ""
        sig = self.sig
        return {
            "outputs": {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                        "tokens": text["tokens"], "steps": steps, "start_step": start, "block_stats": stats,
                        "preview_steps": sorted(self.x0s.keys()), "preview_kind": "linear",
                        "xattn_groups": groups, "xattn_steps": sorted(self.cap_steps) if not self.cfg else [],
                        "lens_steps": sorted(lens.keys()), "lens_blocks": nb,
                        "prompt_layout": [{k: l[k] for k in ("pos", "reads", "prefix", "n")} for l in text["layout"]],
                        "timesteps": [round(float(s_) * 1000, 2) for s_ in sig[:-1]],
                        "sigmas": [round(float(s_), 4) for s_ in sig[:-1]],
                        "size": [self.Wpx, self.Hpx], "text_tokens": L, "kv_note": note, "guide_note": guide_note,
                        "edits": self.edits},
            "captures": {"latents_before": self.latents_before, "x0": self.x0s, "xattn": xattn, "n_tokens": len(text["tokens"]),
                         "stage_maps": run.store.get("stage_maps", {}), "image_png": buf.getvalue(),
                         "rgb_fit": rgb_fit, "hw": (h, w), "final_latent": self.lat.float().cpu(), "lens": lens,
                         "sigmas": sig.clone(),
                         "lens_ctx": {"sig": sig.clone(), "text": text, "text2": self.text2, "switch": self.switch,
                                      "regions": self.regions, "n_steps": steps}},
        }

