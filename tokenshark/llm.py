"""Raw PyTorch decoder-only LLM (Llama 2/3, Qwen2/2.5, Qwen3, Mistral, SmolLM ...).

No transformers. Every step is written out so students can read it and the engine can touch it:

    x = embed[ids]
    for each schedule position p (block index l = schedule[p]):
        h = rmsnorm(x)                       ── hook resid_pre
        q, k, v = h·Wq, h·Wk, h·Wv           (+bias for Qwen2, per-head RMSNorm on q,k for Qwen3)
        cache[p] += (k before RoPE, v)       ── keys stored un-rotated, with explicit positions
        scores = rope(q) · rope(K_all)ᵀ / √d  ── RoPE applied here from cache positions
        z = softmax(scores + mask) · V_all   ── hook attn.pattern, attn_heads (per head)
        x = x + z·Wo                         ── hook attn_out → resid_mid
        x = x + down(silu(gate(h2)) * up(h2)) where h2 = rmsnorm(x)   ── hook mlp_out
                                             ── hook resid_post
    logits = rmsnorm(x) · W_unembedᵀ

Weights come from a WeightStore, so layers can be streamed from disk one at a time.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .engine import Cancelled, RunContext, apply_op, set_determinism
from .flow import Plan, parse_index_spec
from .kvcache import KVCache
from .tokenizer import LLMTokenizer
from .weights import WeightStore, read_json


# ======================================================================================
# Config
# ======================================================================================

class LLMConfig:
    def __init__(self, root: Path):
        hf = read_json(root / "config.json")
        self.meta_format = False
        self.layer_types = None          # hybrid models: "linear_attention" | "full_attention" per layer
        self.norm_offset = 0.0           # Qwen3.5/3.6 store RMSNorm weights as (w - 1): scale = 1 + w
        self.rot_dim = None              # partial rotary: only the first rot_dim dims of each head rotate
        self.vision = None
        if hf and "text_config" in hf:   # multimodal checkpoint: the language model lives in text_config
            self.vision = hf.get("vision_config")
            self.vlm_tokens = {k: hf.get(k) for k in ("image_token_id", "video_token_id", "vision_start_token_id",
                                                     "vision_end_token_id")}
            arch = (hf.get("architectures") or [""])[0]
            tc = dict(hf["text_config"])
            tc.setdefault("architectures", [arch])
            tc.setdefault("tie_word_embeddings", hf.get("tie_word_embeddings", False))
            tc["model_type"] = tc.get("model_type") or hf.get("model_type")
            hf = tc
        if hf:
            missing = [k for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                                   "vocab_size") if k not in hf]
            if missing:
                raise ValueError(f"config.json in {root} is missing {', '.join(missing)} — this checkpoint layout isn't "
                                 f"supported yet. Top-level keys: {', '.join(sorted(hf)[:30])}")
            self.arch = (hf.get("architectures") or ["LlamaForCausalLM"])[0]
            self.model_type = hf.get("model_type", "llama")
            self.hidden = hf["hidden_size"]
            self.n_layers = hf["num_hidden_layers"]
            self.n_heads = hf["num_attention_heads"]
            self.n_kv = hf.get("num_key_value_heads") or self.n_heads
            self.head_dim = hf.get("head_dim") or self.hidden // self.n_heads
            self.inter = hf["intermediate_size"]
            self.eps = hf.get("rms_norm_eps", 1e-6)
            self.vocab = hf["vocab_size"]
            self.theta = hf.get("rope_theta") or (hf.get("rope_parameters") or {}).get("rope_theta") or 10000.0
            self.rope_scaling = hf.get("rope_scaling") or hf.get("rope_parameters")
            self.tied = bool(hf.get("tie_word_embeddings", False))
            self.sliding_window = hf.get("sliding_window") if hf.get("use_sliding_window", self.model_type == "mistral") else None
            self.act = hf.get("hidden_act", "silu")
            self.attn_bias = bool(hf.get("attention_bias", self.model_type == "qwen2"))
            self.eos = hf.get("eos_token_id")
            self.max_pos = hf.get("max_position_embeddings", 4096)
            rp = hf.get("rope_parameters") or hf.get("rope_scaling") or {}
            if rp.get("partial_rotary_factor") or hf.get("partial_rotary_factor"):
                self.rot_dim = int(self.head_dim * float(rp.get("partial_rotary_factor") or hf["partial_rotary_factor"]))
            if self.model_type.startswith(("qwen3_5", "qwen3_next")):
                if "moe" in self.model_type or hf.get("num_experts"):
                    raise ValueError("Qwen3.5/3.6 MoE checkpoints are not supported yet (dense ones are).")
                self.norm_offset = 1.0
                lt = hf.get("layer_types")
                if not lt:
                    every = hf.get("full_attention_interval", 4)
                    lt = ["linear_attention" if (i + 1) % every else "full_attention" for i in range(self.n_layers)]
                self.layer_types = lt
                self.lin = {"conv": hf.get("linear_conv_kernel_dim", 4), "dk": hf.get("linear_key_head_dim", 128),
                            "dv": hf.get("linear_value_head_dim", 128), "hk": hf.get("linear_num_key_heads", 16),
                            "hv": hf.get("linear_num_value_heads", 32)}
            elif self.model_type not in ("llama", "qwen2", "qwen3", "mistral", "olmo2", "qwen3_vl", "qwen3_vl_text") \
                    and "Llama" not in self.arch:
                raise ValueError(f"Architecture '{self.arch}' is not supported by the raw engine yet "
                                 f"(supported: Llama, Qwen2/2.5, Qwen3, Qwen3.5/3.6 dense, Mistral, SmolLM).")
        else:
            p = read_json(root / "params.json")
            if not p:
                raise FileNotFoundError(f"No config.json or params.json in {root}")
            self.meta_format = True
            self.arch, self.model_type = "Llama (Meta format)", "llama"
            self.hidden, self.n_layers, self.n_heads = p["dim"], p["n_layers"], p["n_heads"]
            self.n_kv = p.get("n_kv_heads") or self.n_heads
            self.head_dim = self.hidden // self.n_heads
            hidden = int(2 * 4 * self.hidden / 3)
            if p.get("ffn_dim_multiplier"):
                hidden = int(p["ffn_dim_multiplier"] * hidden)
            m = p.get("multiple_of", 256)
            self.inter = m * ((hidden + m - 1) // m)
            self.eps = p.get("norm_eps", 1e-5)
            self.vocab = p.get("vocab_size", 128256)
            self.theta = p.get("rope_theta", 500000.0)
            self.rope_scaling = ({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                  "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
                                 if p.get("use_scaled_rope") else None)
            self.tied, self.sliding_window, self.act, self.attn_bias = False, None, "silu", False
            self.eos, self.max_pos = None, 8192
        gen = read_json(root / "generation_config.json")
        self.eos_ids = set()
        for e in (self.eos, gen.get("eos_token_id")):
            if isinstance(e, list):
                self.eos_ids.update(e)
            elif e is not None:
                self.eos_ids.add(e)

    def is_linear(self, layer: int) -> bool:
        return bool(self.layer_types) and self.layer_types[layer] == "linear_attention"

    def fingerprint(self) -> dict:
        return {"arch": self.arch, "hidden_size": self.hidden, "n_layers": self.n_layers, "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv, "head_dim": self.head_dim, "vocab": self.vocab,
                "hybrid": bool(self.layer_types)}


def meta_name_map(cfg: LLMConfig) -> dict[str, str]:
    m = {"model.embed_tokens.weight": "tok_embeddings.weight", "model.norm.weight": "norm.weight",
         "lm_head.weight": "output.weight"}
    for i in range(cfg.n_layers):
        a, b = f"model.layers.{i}.", f"layers.{i}."
        m.update({a + "input_layernorm.weight": b + "attention_norm.weight",
                  a + "post_attention_layernorm.weight": b + "ffn_norm.weight",
                  a + "self_attn.q_proj.weight": b + "attention.wq.weight",
                  a + "self_attn.k_proj.weight": b + "attention.wk.weight",
                  a + "self_attn.v_proj.weight": b + "attention.wv.weight",
                  a + "self_attn.o_proj.weight": b + "attention.wo.weight",
                  a + "mlp.gate_proj.weight": b + "feed_forward.w1.weight",
                  a + "mlp.down_proj.weight": b + "feed_forward.w2.weight",
                  a + "mlp.up_proj.weight": b + "feed_forward.w3.weight"})
    return m


def meta_transform(cfg: LLMConfig):
    """Meta checkpoints rotate interleaved pairs; permute q/k rows into the half-split layout used here."""
    def permute(w, n_heads):
        d_out, d_in = w.shape
        return w.view(n_heads, d_out // n_heads // 2, 2, d_in).transpose(1, 2).reshape(d_out, d_in)

    def tf(name, t):
        if name.endswith("q_proj.weight"):
            return permute(t, cfg.n_heads)
        if name.endswith("k_proj.weight"):
            return permute(t, cfg.n_kv)
        return t
    return tf


# ======================================================================================
# Maths
# ======================================================================================

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def rope_inv_freq(cfg: LLMConfig) -> torch.Tensor:
    d = cfg.rot_dim or cfg.head_dim
    inv = 1.0 / (cfg.theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
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
    return inv


def apply_rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x: [heads, T, d]; pos: [T] (any real numbers, negative allowed). If inv_freq covers fewer dims than d
    (partial rotary), only the first 2*len(inv_freq) dims rotate; the rest carry position-free content."""
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


# ======================================================================================
# Model
# ======================================================================================

class RawLLM:
    def __init__(self, root: Path, device: str, dtype: torch.dtype, policy: str, keep_layers: int):
        self.root = Path(root)
        self.cfg = cfg = LLMConfig(self.root)
        self.device, self.dtype = device, dtype
        kw = {}
        if cfg.meta_format:
            kw = {"name_map": meta_name_map(cfg), "transform": meta_transform(cfg)}
        self.W = WeightStore(self.root, device, dtype, policy, keep_layers, **kw)
        self.inv_freq = rope_inv_freq(cfg).to(device)
        fa = next((i for i in range(cfg.n_layers) if not cfg.is_linear(i)), 0)     # a full-attention layer
        self.qk_norm = self.W.has(f"model.layers.{fa}.self_attn.q_norm.weight")
        self.has_bias = self.W.has(f"model.layers.{fa}.self_attn.q_proj.bias")
        qshape = self.W.shape(f"model.layers.{fa}.self_attn.q_proj.weight")
        self.attn_gate = bool(qshape) and qshape[0] == 2 * cfg.n_heads * cfg.head_dim   # Qwen3.5 gated attention
        self.tied = cfg.tied or not self.W.has("lm_head.weight")
        self.act = _act(cfg.act)

    # -------------------------------------------------------------- pieces
    def embed(self, ids: list[int]) -> torch.Tensor:
        return self.W.rows("model.embed_tokens.weight", ids)

    def unembed_weight(self) -> torch.Tensor:
        """The unembedding matrix in its stored dtype, read once and kept (bf16: 2.5 GB for Qwen3.6-27B).
        Every logit-lens readout needs it; re-reading it from disk per block would cost more than the block."""
        if getattr(self, "_w_out", None) is None:
            self._w_out = self.W.compact("model.embed_tokens.weight" if self.tied else "lm_head.weight")
        return self._w_out

    def norm_w(self, name: str) -> torch.Tensor:
        w = self.W(name)
        return w + self.cfg.norm_offset if self.cfg.norm_offset else w

    def final(self, x: torch.Tensor, w_out=None) -> torch.Tensor:
        return self.unembed(self.final_norm(x), w_out)

    def final_norm(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.norm_w("model.norm.weight"), self.cfg.eps)

    def unembed(self, h: torch.Tensor, w_out=None) -> torch.Tensor:
        """One score per vocabulary word: h · W_unembedᵀ."""
        W = w_out if w_out is not None else self.unembed_weight()
        if W.dtype == h.dtype and W.device == h.device:
            return (h @ W.T).float()
        # stored-dtype matrix: convert a slice of the vocabulary at a time (exact; no full fp32 copy in RAM)
        out = torch.empty(h.shape[0], W.shape[0], dtype=h.dtype, device=h.device)
        for i in range(0, W.shape[0], 32768):
            out[:, i:i + 32768] = h @ W[i:i + 32768].to(h.device, h.dtype).T
        return out.float()

    def _lin(self, x, pre, ctx=None):
        w = self._w(pre + ".weight", ctx)
        y = x @ w.T
        if self.has_bias and self.W.has(pre + ".bias"):
            y = y + self.W(pre + ".bias")
        return y

    def _w(self, name: str, ctx: RunContext | None = None) -> torch.Tensor:
        w = self.W(name)
        if ctx is not None:
            for iv in ctx.active("weights"):
                w = apply_op(w, iv, ctx, None)
        return w

    def _head_hook(self, p: int, hook: str, t: torch.Tensor, ctx: RunContext, idx_of, cap: bool,
                   token_dim: int = 1) -> torch.Tensor:
        """Apply interceptors on a [heads, tokens, head_dim] tensor (q, k, v, kv_k, kv_v) and optionally capture."""
        n_heads = t.shape[0]
        # Transpose so tokens are axis 0: [tokens, heads, head_dim]
        tt = t.transpose(0, 1) if token_dim == 1 else t
        if cap:
            self.note_units(p, hook, tt.reshape(tt.shape[0], -1), ctx)
        for iv in ctx.active(hook):
            heads = parse_index_spec(iv.heads, n_heads)
            if not heads:
                continue
            sub = tt[:, heads]
            # For cache-wide hooks (kv_k, kv_v), allow positions spec across the full cache length
            loc_idx = None if hook in ("kv_k", "kv_v") else idx_of(iv)
            sub = apply_op(sub, iv, ctx, 0, loc_idx)
            tt = tt.clone()
            tt[:, heads] = sub
        out = tt.transpose(0, 1) if token_dim == 1 else tt
        if cap and ctx.store.get("want_parts", False) and hook in ("q", "k", "v", "kv_k", "kv_v"):
            if hook in ("q", "k", "v"):
                ctx.store.setdefault(hook, {}).setdefault(p, []).append(tt.reshape(tt.shape[0], -1).detach().float().cpu())
        return out

    def block(self, p: int, l: int, x: torch.Tensor, cache: KVCache, ctx: RunContext, idx_of) -> torch.Tensor:
        """x: [n, d] new tokens. Writes k/v for these tokens into cache slot p."""
        cfg, W = self.cfg, self.W
        pre = f"model.layers.{l}."
        n = x.shape[0]
        ctx.pos = p
        for iv in ctx.active("resid_pre"):
            x = apply_op(x, iv, ctx, 0, idx_of(iv))
        cap = ctx.capture
        if cap:
            ctx.store.setdefault("resid_pre", {}).setdefault(p, []).append(x.detach().float().cpu())

        code = ctx.store.get("code", {}).get(p)
        if code is not None:                     # this box runs the user's edited Python instead (blockcode.py)
            from .blockcode import run_block
            x, attn_out, mlp_out = run_block(self, code, p, l, x, cache, ctx)
            return self._finish(p, x, attn_out, mlp_out, ctx, idx_of, cap)

        h = rms_norm(x, self.norm_w(pre + "input_layernorm.weight"), cfg.eps)
        if cfg.is_linear(l):
            attn_out = self._linear_attn(p, pre, h, cache, ctx, idx_of)
        else:
            attn_out = self._full_attn(p, pre, h, cache, ctx, idx_of)
        for iv in ctx.active("attn_out"):
            attn_out = apply_op(attn_out, iv, ctx, 0, idx_of(iv))
        x = x + attn_out
        return self._mlp_and_finish(p, pre, x, attn_out, ctx, idx_of, cap)

    def attn_allowed(self, p: int, n: int, T: int, cache: KVCache, device) -> torch.Tensor:
        """[n, T] — True where new token i (row) may look at token j of the cache (column) in block slot p."""
        t0 = T - n
        qi = torch.arange(t0, T, device=device)[:, None]
        kj = torch.arange(T, device=device)[None, :]
        allowed = kj <= qi                                # causal: itself and earlier tokens
        if self.cfg.sliding_window:
            allowed &= kj > qi - self.cfg.sliding_window
        for a, b, slots in cache.prefix_spans:            # imported prefixes this block was told not to see
            if slots is not None and p not in slots:
                allowed &= (kj < a) | (kj >= b)
        for a, b, slots in cache.hidden_spans:            # "overwrite": the prompt's own keys are hidden here
            if slots is None or p in slots:
                allowed &= (kj < a) | (kj >= b) | (kj == qi)
        return allowed

    def note_units(self, p: int, hook: str, a: torch.Tensor, ctx: RunContext):
        """Remember the smallest and largest value each unit has taken so far (what 'clamp to max' pins it to)."""
        lo, hi = a.detach().float().amin(0).cpu(), a.detach().float().amax(0).cpu()
        rng = ctx.store.setdefault("unit_range", {})
        old = rng.get((p, hook))
        rng[(p, hook)] = (lo, hi) if old is None else (torch.minimum(old[0], lo), torch.maximum(old[1], hi))

    def _full_attn(self, p, pre, h, cache, ctx, idx_of):
        """Softmax attention over the cache. Qwen3.5/3.6: the query projection also yields a per-head output gate."""
        cfg, W = self.cfg, self.W
        n = h.shape[0]
        cap = ctx.capture
        q_all = self._lin(h, pre + "self_attn.q_proj", ctx)
        gate = None
        if self.attn_gate:
            q_all = q_all.view(n, cfg.n_heads, 2 * cfg.head_dim)
            q, gate = q_all[..., :cfg.head_dim], q_all[..., cfg.head_dim:]
            q = q.transpose(0, 1)
        else:
            q = q_all.view(n, cfg.n_heads, cfg.head_dim).transpose(0, 1)
        k = self._lin(h, pre + "self_attn.k_proj", ctx).view(n, cfg.n_kv, cfg.head_dim).transpose(0, 1)
        v = self._lin(h, pre + "self_attn.v_proj", ctx).view(n, cfg.n_kv, cfg.head_dim).transpose(0, 1)
        if self.qk_norm:
            q = rms_norm(q, self.norm_w(pre + "self_attn.q_norm.weight"), cfg.eps)
            k = rms_norm(k, self.norm_w(pre + "self_attn.k_norm.weight"), cfg.eps)
        q = self._head_hook(p, "q", q, ctx, idx_of, cap)
        k = self._head_hook(p, "k", k, ctx, idx_of, cap)
        v = self._head_hook(p, "v", v, ctx, idx_of, cap)
        x = h
        cache.append(p, k, v)
        if ctx.active("kv_k"):
            cache.k[p] = self._head_hook(p, "kv_k", cache.k[p].to(x.device), ctx, idx_of, False).to(cache.device)
        if ctx.active("kv_v"):
            cache.v[p] = self._head_hook(p, "kv_v", cache.v[p].to(x.device), ctx, idx_of, False).to(cache.device)
        K = cache.k[p].to(x.device)
        V = cache.v[p].to(x.device)
        T = K.shape[1]
        t0 = T - n
        pos_all = cache.pos.to(self.inv_freq.device)
        qr = apply_rope(q, pos_all[t0:], self.inv_freq)
        kr = apply_rope(K, pos_all, self.inv_freq)
        rep = cfg.n_heads // cfg.n_kv
        if rep > 1:
            kr = kr.repeat_interleave(rep, 0)
            V = V.repeat_interleave(rep, 0)
        scores = (qr.float() @ kr.float().transpose(1, 2)) / math.sqrt(cfg.head_dim)   # [H, n, T]
        scores = scores.masked_fill(~self.attn_allowed(p, n, T, cache, x.device), float("-inf"))
        probs = torch.softmax(scores, -1)
        if cap and ctx.store.get("want_attn", True):
            ctx.store.setdefault("attn", {}).setdefault(p, []).append(probs.detach().to(torch.float16).cpu())
        z = (probs.to(V.dtype) @ V)                                                     # [H, n, hd]
        for iv in ctx.active("attn_heads"):
            heads = parse_index_spec(iv.heads, cfg.n_heads)
            zt = z.transpose(0, 1)                                                      # [n, H, hd]
            sub = zt[:, heads]
            sub = apply_op(sub, iv, ctx, 0, idx_of(iv))
            zt = zt.clone()
            zt[:, heads] = sub
            z = zt.transpose(0, 1)
        zf = z.transpose(0, 1).reshape(n, cfg.n_heads * cfg.head_dim)
        if gate is not None:
            zf = zf * torch.sigmoid(gate.reshape(n, -1))
        return zf @ self._w(pre + "self_attn.o_proj.weight", ctx).T

    def _linear_attn(self, p, pre, h, cache, ctx, idx_of):
        """Gated DeltaNet (Qwen3.5/3.6 'linear attention'). Instead of keeping every key and value it keeps one
        fixed-size memory matrix S per head and updates it token by token:
            S ← S · exp(g)                      (forget a little)
            S ← S + β · k ⊗ (v − Sᵀk)           (write v at key k, correcting what S already recalls there)
            out = Sᵀq                            (read with the query)
        A short causal convolution mixes each token with the previous 3 before q, k, v are formed."""
        cfg, W = self.cfg, self.W
        L = cfg.lin
        n = h.shape[0]
        a_ = pre + "linear_attn."
        mixed = h @ self._w(a_ + "in_proj_qkv.weight", ctx).T                     # [n, conv_dim]
        z = (h @ self._w(a_ + "in_proj_z.weight", ctx).T).view(n, L["hv"], L["dv"])
        b = h @ self._w(a_ + "in_proj_b.weight", ctx).T
        a = h @ self._w(a_ + "in_proj_a.weight", ctx).T
        st = cache.lin_state(p, mixed.shape[1], L, h.device)
        # causal depthwise conv over [previous K-1 inputs | new inputs]
        K = L["conv"]
        seq = torch.cat([st["conv"].to(mixed.dtype), mixed], 0)                  # [K-1+n, C]
        cw = W(a_ + "conv1d.weight")                                              # [C, 1, K]
        cb = W(a_ + "conv1d.bias") if W.has(a_ + "conv1d.bias") else None
        conv = F.conv1d(seq.T.unsqueeze(0).float(), cw.float(), cb.float() if cb is not None else None,
                        groups=cw.shape[0])[0].T                                  # [n, C]
        conv = F.silu(conv)
        st["conv"] = seq[-(K - 1):].detach().clone() if K > 1 else seq[:0]
        kd, vd = L["hk"] * L["dk"], L["hv"] * L["dv"]
        q, k, v = conv[:, :kd], conv[:, kd:2 * kd], conv[:, 2 * kd:]
        q = q.reshape(n, L["hk"], L["dk"])
        k = k.reshape(n, L["hk"], L["dk"])
        v = v.reshape(n, L["hv"], L["dv"])
        rep = L["hv"] // L["hk"]
        if rep > 1:
            q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(L["dk"])
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(b.float())                                            # [n, Hv]
        g = -W(a_ + "A_log").float().exp() * F.softplus(a.float() + W(a_ + "dt_bias").float())
        S = st["rec"]                                                             # [Hv, dk, dv] fp32
        outs = []
        for t in range(n):
            S = S * g[t].exp()[:, None, None]
            kt, vt, qt = k[t], v[t].float(), q[t]
            mem = (S * kt[:, :, None]).sum(1)                                     # [Hv, dv]
            S = S + kt[:, :, None] * ((vt - mem) * beta[t][:, None])[:, None, :]
            outs.append((S * qt[:, :, None]).sum(1))
        st["rec"] = S
        o = torch.stack(outs)                                                     # [n, Hv, dv]
        for iv in ctx.active("attn_heads"):
            heads = parse_index_spec(iv.heads, L["hv"])
            o = o.clone()
            o[:, heads] = apply_op(o[:, heads], iv, ctx, 0, idx_of(iv))
        if ctx.capture:
            ctx.store.setdefault("lin_mem", {}).setdefault(p, []).append(S.norm(dim=(1, 2)).cpu())
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + cfg.eps)
        o = o * W(a_ + "norm.weight").float() * F.silu(z.float())
        return o.reshape(n, vd).to(h.dtype) @ self._w(a_ + "out_proj.weight", ctx).T

    def _mlp_and_finish(self, p, pre, x, attn_out, ctx, idx_of, cap):
        cfg, W = self.cfg, self.W
        h2 = rms_norm(x, self.norm_w(pre + "post_attention_layernorm.weight"), cfg.eps)
        gate = h2 @ self._w(pre + "mlp.gate_proj.weight", ctx).T
        up = h2 @ self._w(pre + "mlp.up_proj.weight", ctx).T
        neurons = self.act(gate) * up                     # [n, d_ff]: one number per MLP neuron per token
        neurons = self.neuron_hook(p, neurons, ctx, idx_of, cap)
        mlp_out = neurons @ self._w(pre + "mlp.down_proj.weight", ctx).T
        for iv in ctx.active("mlp_out"):
            mlp_out = apply_op(mlp_out, iv, ctx, 0, idx_of(iv))
        x = x + mlp_out
        if W.has(pre + "adapter.down.weight"):
            # A bottleneck adapter added by training (adapt.py): squeeze to a few dimensions, expand back, add.
            x = x + self.act(x @ self._w(pre + "adapter.down.weight", ctx).T) @ self._w(pre + "adapter.up.weight", ctx).T
        return self._finish(p, x, attn_out, mlp_out, ctx, idx_of, cap)

    def neuron_hook(self, p, neurons, ctx, idx_of, cap):
        """Hook mlp_neurons: record each neuron's range, apply interventions (clamp a neuron…), capture."""
        if cap:
            self.note_units(p, "mlp_neurons", neurons, ctx)
        for iv in ctx.active("mlp_neurons"):
            neurons = apply_op(neurons, iv, ctx, 0, idx_of(iv))
        if cap and ctx.store.get("want_neurons", False):
            ctx.store.setdefault("neurons", {}).setdefault(p, []).append(neurons.detach().to(torch.float16).cpu())
        return neurons

    def _finish(self, p, x, attn_out, mlp_out, ctx, idx_of, cap):
        for iv in ctx.active("resid_post"):
            x = apply_op(x, iv, ctx, 0, idx_of(iv))
        zero = torch.zeros(x.shape[0])
        st = ctx.store.setdefault("stats", {}).setdefault(p, {"attn": [], "mlp": [], "resid": []})
        st["attn"].append(attn_out.detach().float().norm(dim=-1).cpu() if attn_out is not None else zero)
        st["mlp"].append(mlp_out.detach().float().norm(dim=-1).cpu() if mlp_out is not None else zero)
        st["resid"].append(x.detach().float().norm(dim=-1).cpu())
        if cap:
            ctx.store.setdefault("resid_post", {}).setdefault(p, []).append(x.detach().float().cpu())
            if ctx.store.get("want_parts", False):       # what attention and the MLP each wrote (for patching)
                for key, t in (("attn_out", attn_out), ("mlp_out", mlp_out)):
                    if t is not None:
                        ctx.store.setdefault(key, {}).setdefault(p, []).append(t.detach().float().cpu())
        return x

    def forward(self, ids: list[int], positions: list[float], origin: str, cache: KVCache, schedule: list[int],
                ctx: RunContext, run_offset: int, idx_planner, w_out=None) -> torch.Tensor:
        """Process new tokens `ids` (run-relative index run_offset..). Returns logits [n, vocab] (float32)."""
        cache.add_tokens(ids, positions, origin)
        x = self.embed(ids)
        n = len(ids)

        def idx_of(iv):
            return idx_planner(iv, run_offset, n)

        for p, l in enumerate(schedule):
            ctx.checkpoint()
            x = self.block(p, l, x, cache, ctx, idx_of)
            if ctx.on_block:
                ctx.on_block(p)
        return self.final(x, w_out)


# ======================================================================================
# Adapter used by the server
# ======================================================================================

class LLMAdapter:
    family = "causal_lm"

    def __init__(self, spec: dict, root: Path, device: str, dtype: torch.dtype, policy="auto", keep_layers=0):
        self.spec = spec
        from .weights import report_phase
        report_phase("reading config")
        self.model = RawLLM(root, device, dtype, policy, keep_layers)
        report_phase("loading tokenizer")
        self.tok = LLMTokenizer(root)
        self.cfg = self.model.cfg
        self.device = device
        self.cache_device = device if self.model.W.policy == "device" else "cpu"

    def info(self) -> dict:
        c = self.cfg
        attn = "MHA" if c.n_kv == c.n_heads else f"GQA {c.n_heads}q/{c.n_kv}kv"
        return {
            "family": self.family, "arch": c.arch, "n_layers": c.n_layers, "d_model": c.hidden,
            "n_heads": c.n_heads, "n_kv_heads": c.n_kv, "head_dim": c.head_dim, "vocab": c.vocab,
            "d_ff": c.inter, "rope_theta": c.theta,
            "tokenizer": "tiktoken" if self.tok.kind == "tiktoken" else "BPE tokenizer",
            "norm": "RMSNorm", "block_sub": f"{attn}{' · QK-norm' if self.model.qk_norm else ''} · SwiGLU",
            "has_chat_template": bool(self.tok.chat_template),
            "block_subs": [("Gated DeltaNet (linear attn)" if c.is_linear(i) else
                            f"{'gated ' if self.model.attn_gate else ''}{attn}{' · QK-norm' if self.model.qk_norm else ''}") + " · SwiGLU"
                           for i in range(c.n_layers)],
            "hybrid": bool(c.layer_types),
            "extra_blocks": list(getattr(self, "extra_blocks", None) or []),     # blocks added by training (adapt.py)
            "vision": bool(c.vision),
            "weights": self.model.W.describe(),
            "capabilities": {"repeat_blocks": True, "reorder_blocks": True, "remove_blocks": True,
                             "logit_lens": True, "attention": True, "kv_cache": True, "has_moe": False},
            "hooks": ["resid_pre", "q", "k", "v", "kv_k", "kv_v", "attn_heads", "attn_out", "mlp_neurons", "mlp_out", "resid_post", "weights"],
        }

    def fingerprint(self):
        return {**self.cfg.fingerprint(), "model_id": self.spec["id"]}

    def new_cache(self, n_slots: int) -> KVCache:
        return KVCache(self.fingerprint(), n_slots, device=self.cache_device)

    # -------------------------------------------------------------- input building
    def split_system(self, system: str, user: str, extra: dict | None = None) -> tuple[str, str]:
        """Render [system, user] with the chat template and split it into (system turn, rest). Works whether the
        template can render a system turn on its own (Qwen2.5) or refuses to (Qwen3.5/3.6 needs a user turn)."""
        tok = self.tok
        sys_msg = [{"role": "system", "content": system}]
        full = tok.apply_chat(sys_msg + [{"role": "user", "content": user}], True, extra)
        try:
            head = tok.apply_chat(sys_msg, False, extra)
            if full.startswith(head):
                return head, full[len(head):]
        except Exception:
            pass
        rest = tok.apply_chat([{"role": "user", "content": user}], True, extra)
        if full.endswith(rest):
            return full[: len(full) - len(rest)], rest
        raise ValueError("Couldn't separate the system turn in this model's chat template.")

    def template_extra(self, p: dict) -> dict:
        return {"enable_thinking": bool(p.get("thinking", False))}

    def build_ids(self, p: dict) -> tuple[list[int], list[int]]:
        tok = self.tok
        has_prefix = bool(p.get("prefix_cache"))
        extra = self.template_extra(p)
        if has_prefix and p.get("use_chat_template") and tok.chat_template:
            # The KV prefix stands in for the start of the chat (a hidden system turn, or an earlier conversation).
            # Render system + user, then cut the system part, so the user turn continues right after the cached
            # tokens — and the template can't slip in its own default system prompt (Qwen2.5 does that).
            _, text = self.split_system(p.get("prefix_system") or ".", p.get("prompt", ""), extra)
            prompt_ids = tok.encode(text, add_special=False)
        elif p.get("use_chat_template") and tok.chat_template:
            msgs = []
            if p.get("system"):
                msgs.append({"role": "system", "content": p["system"]})
            msgs.append({"role": "user", "content": p.get("prompt", "")})
            text = tok.apply_chat(msgs, True, extra)
            prompt_ids = tok.encode(text, add_special=False)
        else:
            text = (p["system"] + "\n\n" if p.get("system") else "") + p.get("prompt", "")
            # after a KV prefix the sequence has already started: no second BOS
            prompt_ids = tok.encode(text, add_special=bool(p.get("add_bos", True)) and not has_prefix)
        prefill_ids = tok.encode(p["prefill"], add_special=False) if p.get("prefill") else []
        return prompt_ids, prefill_ids

    @torch.no_grad()
    def build_prefix(self, plan: Plan, text: str, as_system: bool) -> tuple[KVCache, dict]:
        """Encode a text once and keep only its KV cache: a 'hidden prompt' that later conversations can attend
        to without the text ever being part of their input."""
        tok = self.tok
        if as_system and tok.chat_template:
            rendered, _ = self.split_system(text, "x")
            ids = tok.encode(rendered, add_special=False)
        else:
            ids = tok.encode(text, add_special=True)
        if not ids:
            raise ValueError("Empty text.")
        cache = self.new_cache(len(plan.schedule))
        ctx = RunContext(interventions=[])
        self.model.forward(ids, [float(i) for i in range(len(ids))], "prompt", cache, plan.schedule, ctx, 0,
                           lambda iv, o, n: [])
        cache = cache.relabel("imported", "hidden system prompt" if as_system else "hidden prompt")
        extra = {"system_text": text if (as_system and tok.chat_template) else None, "schedule": plan.schedule,
                 "source": "text"}
        return cache, extra

    # -------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, plan: Plan, progress=None, resume=None, prefix: KVCache | None = None, ctx_hook=None) -> dict:
        st = Stepper(self, plan, prefix=prefix, progress=progress, ctx_hook=ctx_hook)
        while st.step_token():
            pass
        return st.finish()

    def run_stepper(self, plan: Plan, prefix=None, progress=None, ctx_hook=None) -> "Stepper":
        return Stepper(self, plan, prefix=prefix, progress=progress, ctx_hook=ctx_hook)

    def stepper(self, plan: Plan, prefix: KVCache | None = None) -> "Stepper":
        return Stepper(self, plan, prefix=prefix, trace=True)

    def _collect(self, ctx, plan, seq, n_prefix):
        cat = lambda d: {k: torch.cat(v, 0) for k, v in d.items()}
        attn = {}
        for pp, rows in ctx.store.get("attn", {}).items():
            T = rows[-1].shape[-1]
            H = rows[0].shape[0]
            full = torch.zeros(H, sum(r.shape[1] for r in rows), T, dtype=torch.float16)
            i = 0
            for r in rows:
                full[:, i:i + r.shape[1], :r.shape[2]] = r
                i += r.shape[1]
            attn[pp] = full
        return {"resid_post": cat(ctx.store.get("resid_post", {})), "resid_pre": cat(ctx.store.get("resid_pre", {})),
                "attn_out": cat(ctx.store.get("attn_out", {})), "mlp_out": cat(ctx.store.get("mlp_out", {})),
                "q": cat(ctx.store.get("q", {})), "k": cat(ctx.store.get("k", {})), "v": cat(ctx.store.get("v", {})),
                "neurons": cat(ctx.store.get("neurons", {})),
                "unit_range": dict(ctx.store.get("unit_range", {})), "code": dict(ctx.store.get("code", {})),
                "attn": attn, "ids": seq, "n_prefix": n_prefix}

    def _sample(self, logits: torch.Tensor, p: dict, gen: torch.Generator) -> int:
        if p["temperature"] <= 0:
            return int(logits.argmax())
        logits = logits.float() / p["temperature"]
        if p.get("top_p", 1.0) < 1.0:
            # nucleus sampling: keep the smallest set of most likely tokens whose probability adds up to top_p
            probs = torch.softmax(logits, -1)
            if p["top_k"] > 0:
                kth = probs.topk(min(p["top_k"], probs.shape[-1])).values[-1]
                probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            sp, si = probs.sort(descending=True)
            keep = (sp.cumsum(-1) - sp) < p["top_p"] * sp.sum()
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            return int(si[torch.multinomial((sp / sp.sum()).cpu(), 1, generator=gen)].item())
        if p["top_k"] > 0:
            v, i = logits.topk(min(p["top_k"], logits.shape[-1]))
            return int(i[torch.multinomial(torch.softmax(v, -1).cpu(), 1, generator=gen)].item())
        return int(torch.multinomial(torch.softmax(logits, -1).cpu(), 1, generator=gen).item())

    # -------------------------------------------------------------- lazy analysis
    @torch.no_grad()
    @torch.no_grad()
    def decode_layer(self, cap: dict, schedule: list[int], p: int, through: int = 0, w_out=None) -> torch.Tensor:
        """Turn the residual stream after schedule position p (p = -1: the embeddings) into next-token logits.

        through = 0: the logit lens — final norm + unembedding applied directly. Cheap, but early layers
                     decode poorly because they don't yet "speak" the output vocabulary.
        through = k: jump from layer p straight into the last k blocks of the stack, then norm + unembed.
                     Early layers become far more readable; with k = all remaining blocks it is the real model.
        """
        h = cap["resid_pre"][0] if p < 0 else cap["resid_post"][p]
        T = min(len(cap["ids"]), h.shape[0])        # while stepping, later layers may not have seen the newest token
        x = h[:T].to(self.device, self.model.dtype)
        S = len(schedule)
        start = max(p + 1, S - through) if through > 0 else S
        if start < S:
            full: KVCache = cap["cache"]
            n_pre = cap["n_prefix"]
            tmp = full.prefix_view(n_pre)                    # imported prefix (+ linear memory right after it)
            tmp.add_tokens(cap["ids"][:T], full.pos[n_pre:n_pre + T].tolist(), "prompt")
            ctx = RunContext(interventions=[])
            ctx.store["code"] = cap.get("code", {})          # edited blocks stay edited when decoding through them
            for q in range(start, S):
                x = self.model.block(q, schedule[q], x, tmp, ctx, lambda iv: [])
        return self.model.final(x, w_out if w_out is not None else self.model.unembed_weight())

    @torch.no_grad()
    def logit_lens(self, cap: dict, schedule_nodes: list[str], schedule: list[int], k: int = 5, through: int = 0) -> dict:
        out = None
        for out in self.iter_logit_lens(cap, schedule_nodes, schedule, k, through):
            pass
        out.pop("_progress", None)
        return out

    @torch.no_grad()
    def iter_logit_lens(self, cap: dict, schedule_nodes: list[str], schedule: list[int], k: int = 5, through: int = 0,
                        memo: dict | None = None):
        """Logit lens one layer at a time. Yields the table so far (rows not computed yet are None), so the UI can
        fill it in as rows arrive and other work can run in between. Only the reference rows (what each token is
        compared against) are kept as full distributions; the rest are reduced to their table cells at once."""
        memo = {} if memo is None else memo
        ids = cap["ids"]
        T = len(ids)
        w_out = self.model.unembed_weight()
        rows_src = []
        if 0 in cap["resid_pre"]:
            rows_src.append(("embed", "Embed", -1, min(T, cap["resid_pre"][0].shape[0])))
        for pp in range(len(schedule)):
            if pp in cap["resid_post"]:
                rows_src.append((schedule_nodes[pp], f"#{pp} · L{schedule[pp]}", pp, min(T, cap["resid_post"][pp].shape[0])))
        R = len(rows_src)
        res = {"tokens": [self.tok.piece(i) for i in ids], "layers": [{"node": n, "label": l} for n, l, _, _ in rows_src],
               "cells": [None] * R, "through": through, "complete": len(cap["final_top"]), "_progress": 0.0}
        yield dict(res)
        # Reference for colouring: the model's final choice; for tokens still in flight (stepping), the deepest
        # layer that has processed them so far.
        ref_row = [next((r for r in range(R - 1, -1, -1) if rows_src[r][3] > t), None) for t in range(T)]
        ref_logp: dict[int, torch.Tensor] = {}
        final_top = list(cap["final_top"])

        def logp_of(r):
            if r in ref_logp:
                return ref_logp[r]
            return torch.log_softmax(self.decode_layer(cap, schedule, rows_src[r][2], through, w_out), -1)

        def refs():
            if not ref_logp:
                for r in sorted({x for x in ref_row if x is not None}):
                    ref_logp[r] = logp_of(r)
                for t in range(len(final_top), T):
                    final_top.append(int(ref_logp[ref_row[t]][t].argmax()) if ref_row[t] is not None else 0)

        order = sorted({x for x in ref_row if x is not None}) + [r for r in range(R) if r not in set(ref_row)]
        for n_done, r in enumerate(order, 1):
            key = ("lensrow", k, through, rows_src[r][2])
            if key not in memo:
                refs()
                logp = logp_of(r)
                n = logp.shape[0]
                probs = logp.exp()
                pv, pi = probs.topk(k, -1)
                row = []
                for t in range(T):
                    if t >= n:
                        row.append(None)                 # this layer hasn't processed token t yet
                        continue
                    tgt = final_top[t]
                    ref = ref_logp[ref_row[t]][t]
                    c = {"top": [[self.tok.piece(int(pi[t, j])), round(float(pv[t, j]), 4)] for j in range(k)],
                         "p_final": round(float(probs[t, tgt]), 4),
                         "rank_final": int((logp[t] > logp[t, tgt]).sum()),
                         "kl": round(float((ref.exp() * (ref - logp[t])).sum()), 3)}
                    if t + 1 < T:
                        c["p_next"] = round(float(probs[t, ids[t + 1]]), 4)
                    row.append(c)
                memo[key] = row
            res["cells"][r] = memo[key]
            res["_progress"] = n_done / max(1, R)
            yield {**res, "cells": list(res["cells"])}

    def _word_ids(self, word: str) -> list[tuple[int, str, int]]:
        """Candidate tokens for a typed word, best first: (id, piece, number of tokens the spelling needs).
        Mid-sentence words usually carry a leading space (" Paris"), and a spelling that is one whole token
        beats one that splits (where only the first piece can be tracked)."""
        out = []
        for form in (" " + word.strip(), word.strip()):
            ids = self.tok.encode(form, add_special=False)
            if ids and ids[0] not in [c[0] for c in out]:
                out.append((ids[0], self.tok.piece(ids[0]), len(ids)))
        out.sort(key=lambda c: c[2] > 1)          # stable: whole-word tokens first, leading-space form first
        return out

    @torch.no_grad()
    def word_cloud(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int,
                   layers: list[int], through: int = 0, top: int = 40) -> dict:
        out = None
        for out in self.iter_word_cloud(cap, schedule_nodes, schedule, position, layers, through, top):
            pass
        out.pop("_progress", None)
        out.pop("todo", None)
        return out

    @torch.no_grad()
    def iter_word_cloud(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int,
                        layers: list[int], through: int = 0, top: int = 40, memo: dict | None = None):
        """For one token position: the words each chosen layer would predict next, with probabilities,
        plus the average over the chosen layers. layers use schedule positions; -1 = embeddings.
        Each layer also reports where the model's own final choice and the actual next token rank.
        Yields after every layer (layers still to come are listed in `todo`)."""
        memo = {} if memo is None else memo
        T = len(cap["ids"])
        t = min(max(position if position >= 0 else T + position, 0), T - 1)
        w_out = self.model.unembed_weight()
        final_choice = cap["final_top"][t] if t < len(cap["final_top"]) else None
        actual_next = cap["ids"][t + 1] if t + 1 < T else None
        n_in = cap.get("n_in")
        lab = lambda pp: "Embed" if pp < 0 else f"#{pp} · L{schedule[pp]}"                  # noqa: E731
        layers = [pp for pp in layers if pp < 0 or pp in cap["resid_post"]]
        res = {"position": t, "token": self.tok.piece(cap["ids"][t]),
               "next_token": self.tok.piece(actual_next) if actual_next is not None else None,
               "next_generated": bool(n_in is not None and t + 1 >= n_in),
               "model_choice": self.tok.piece(final_choice) if final_choice is not None else None,
               "layers": [], "combined": None, "through": through, "pending": [],
               "todo": [{"pos": pp, "label": lab(pp), "node": "embed" if pp < 0 else schedule_nodes[pp]} for pp in layers],
               "_progress": 0.0}
        yield {**res, "layers": [], "todo": list(res["todo"])}
        acc, n_acc = None, 0
        for n_done, pp in enumerate(layers, 1):
            key = ("cloudlayer", t, pp, through)
            if key not in memo:
                logits = self.decode_layer(cap, schedule, pp, through, w_out)
                if t >= logits.shape[0]:
                    memo[key] = {"pending": lab(pp)}
                else:
                    lg = logits[t]
                    probs = torch.softmax(lg, -1)
                    v, i = probs.topk(100)
                    rank = lambda tid: int((lg > lg[tid]).sum()) + 1                        # noqa: E731
                    memo[key] = {"probs": probs.half(), "layer": {
                        "pos": pp, "node": "embed" if pp < 0 else schedule_nodes[pp], "label": lab(pp),
                        "words": [[self.tok.piece(int(j)), round(float(x), 5)] for x, j in zip(v, i)],
                        "choice_rank": rank(final_choice) if final_choice is not None else None,
                        "choice_p": round(float(probs[final_choice]), 5) if final_choice is not None else None,
                        "next_rank": rank(actual_next) if actual_next is not None else None,
                        "next_p": round(float(probs[actual_next]), 5) if actual_next is not None else None}}
            m = memo[key]
            if "pending" in m:
                res["pending"].append(m["pending"])
            else:
                res["layers"].append({**m["layer"], "words": m["layer"]["words"][:top]})
                p32 = m["probs"].float()
                acc = p32 if acc is None else acc + p32
                n_acc += 1
            res["todo"] = res["todo"][1:]
            res["_progress"] = n_done / max(1, len(layers))
            if n_done == len(layers) and acc is not None and n_acc > 1:
                v, i = (acc / n_acc).topk(top)
                res["combined"] = [[self.tok.piece(int(j)), round(float(x), 5)] for x, j in zip(v, i)]
            yield {**res, "layers": list(res["layers"]), "pending": list(res["pending"]), "todo": list(res["todo"])}

    @torch.no_grad()
    def track_word(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int, word: str,
                   through: int = 0) -> dict:
        out = None
        for out in self.iter_track_word(cap, schedule_nodes, schedule, position, word, through):
            pass
        out.pop("_progress", None)
        return out

    @torch.no_grad()
    def iter_track_word(self, cap: dict, schedule_nodes: list[str], schedule: list[int], position: int, word: str,
                        through: int = 0, memo: dict | None = None):
        """'Which layer got it wrong?' — follow one expected word (e.g. ' Paris') through every layer at one
        position, next to the model's actual choice:
          • probability and rank of each word if the model stopped at that layer (lens), and
          • each block's direct push: how much that block's write to the residual stream raises the
            expected word's logit minus the chosen word's logit (direct logit attribution through the final
            norm, frozen at its final scale). Positive = the block argued for the expected word."""
        T = len(cap["ids"])
        t = min(max(position if position >= 0 else T + position, 0), T - 1)
        cands = self._word_ids(word)
        if not cands:
            raise ValueError("Type a word to track.")
        w_out = self.model.unembed_weight()
        S = len(schedule)
        final = self.decode_layer(cap, schedule, S - 1, 0, w_out)[t] if (S - 1) in cap["resid_post"] else None
        target, target_piece, n_pieces = cands[0]
        choice = cap["final_top"][t] if t < len(cap["final_top"]) else (int(final.argmax()) if final is not None else target)
        versus = choice
        if versus == target and final is not None:      # model agreed: compare against its runner-up instead
            versus = int(final.topk(2).indices[1])
        # direct logit attribution (norm_w applies +1.0 offset for Qwen3.5/3.6 zero-centred RMSNorm)
        g = self.model.norm_w("model.norm.weight").float().cpu()
        U = w_out[[target, versus]].float().cpu()                                    # [2, d]
        x_final = cap["resid_post"][S - 1][t].float() if (S - 1) in cap["resid_post"] else None
        rms = float(torch.sqrt(x_final.pow(2).mean() + self.cfg.eps)) if x_final is not None else 1.0
        dirv = (U[0] - U[1]) * g / rms                                                # d(logit_target - logit_choice)/dx
        rows = []
        layers = [-1] + [p for p in range(S) if p in cap["resid_post"]]
        first_top = None
        base = {"position": t, "token": self.tok.piece(cap["ids"][t]), "target": target_piece,
                "candidates": [c[1] for c in cands], "pieces": n_pieces, "word": word.strip(),
                "choice": self.tok.piece(choice), "same": target == choice,
                "versus": self.tok.piece(versus), "through": through}
        for n_done, pp in enumerate(layers, 1):
            lg_all = self.decode_layer(cap, schedule, pp, through, w_out)
            if t >= lg_all.shape[0]:
                continue
            lg = lg_all[t]
            probs = torch.softmax(lg, -1)
            r_t = int((lg > lg[target]).sum()) + 1
            r_c = int((lg > lg[choice]).sum()) + 1
            if pp < 0:
                write = cap["resid_pre"][0][t].float()
            else:
                write = cap["resid_post"][pp][t].float() - cap["resid_pre"][pp][t].float()
            push = float(write @ dirv)
            if r_t == 1 and first_top is None:
                first_top = pp
            rows.append({"pos": pp, "node": "embed" if pp < 0 else schedule_nodes[pp],
                         "label": "Embed" if pp < 0 else f"#{pp} · L{schedule[pp]}",
                         "top": self.tok.piece(int(lg.argmax())),
                         "target_p": round(float(probs[target]), 5), "target_rank": r_t,
                         "choice_p": round(float(probs[choice]), 5), "choice_rank": r_c, "push": round(push, 3)})
            yield {**base, "first_top_layer": first_top, "rows": list(rows), "_progress": n_done / max(1, len(layers))}

    def attention(self, cap: dict, pos: int, head: int | None) -> dict:
        a = cap["attn"].get(pos)
        if a is None and pos in cap.get("linear_slots", []):
            return {"error": "This is a linear-attention (Gated DeltaNet) block. It has no attention map: instead of "
                             "looking back at every earlier token it keeps one running memory per head, updated token "
                             "by token. Pick a full-attention block (every 4th block in Qwen3.5/3.6)."}
        if a is None:
            return {"error": "No attention captured for this block (sequence too long for the capture budget)."}
        m = a.float().mean(0) if head is None or head < 0 else a[head].float()
        cache: KVCache = cap["cache"]
        return {"n_heads": a.shape[0], "n_prefix": cap["n_prefix"],
                "matrix": [[round(float(v), 4) for v in row] for row in m],
                "query_tokens": [self.tok.piece(i) for i in cap["ids"][: m.shape[0]]],
                "key_tokens": [self.tok.piece(i) for i in cache.tokens[: m.shape[1]]],
                "key_positions": [round(float(x), 1) for x in cache.pos[: m.shape[1]].tolist()]}

    def unload(self):
        self.model = None


# ======================================================================================
# Stepper: the run loop, one block at a time (used by both Run and the debugger)
# ======================================================================================

class Stepper:
    """Executes a plan in the smallest steps the model has:

        embed(chunk) → block #0 → block #1 → … → final norm + unembed → choose next token → embed(token) → …

    `step_block()` advances one of those arrows. `step_token()` runs to the next token prediction.
    `force(token_id)` decides the next token yourself (the classic "step in" of llama-runner).
    """

    def __init__(self, ad: "LLMAdapter", plan: Plan, prefix: KVCache | None = None, progress=None, ctx_hook=None,
                 trace: bool = False):
        self.ad, self.model, self.plan = ad, ad.model, plan
        p = self.p_ = plan.params
        set_determinism(p["seed"])
        prompt_ids, prefill_ids = ad.build_ids(p)
        if not prompt_ids and not prefill_ids and prefix is None:
            prompt_ids = [ad.tok.bos_id or 0]
        if not prompt_ids and not prefill_ids:
            raise ValueError("Nothing to run: empty prompt.")
        self.prompt_ids, self.prefill_ids = prompt_ids, prefill_ids
        self.n_in = len(prompt_ids) + len(prefill_ids)
        self.total = p["max_new_tokens"]
        planned = self.planned = self.n_in + self.total
        ctx = self.ctx = RunContext(interventions=plan.interventions)
        if ctx_hook:
            ctx_hook(ctx)
        ctx.token_ranges = {"prompt": range(0, len(prompt_ids)), "prefill": range(len(prompt_ids), self.n_in),
                            "generated": range(self.n_in, planned)}
        ctx.capture = True
        prefixes = prefix if isinstance(prefix, list) else ([prefix] if prefix is not None else [])
        specs = list(p.get("prefixes") or [])
        if len(specs) < len(prefixes):            # legacy single-prefix params
            specs += [{"mode": p.get("prefix_mode", "shift"), "offset": p.get("prefix_offset", 0),
                       "layers": p.get("prefix_layers", "all"), "label": p.get("prefix_label", "imported")}] * (len(prefixes) - len(specs))
        S_ = len(plan.schedule)
        est = ad.cfg.n_heads * planned * (planned + sum(len(c) for c in prefixes)) * S_ * 2
        ctx.store["want_attn"] = est < int(p.get("attn_budget_mb", 768)) * 2**20
        # Optional captures, each within a memory budget: every MLP neuron's activation (neuron hunting) and what
        # attention / the MLP wrote on their own (patching those parts from this run into another).
        ctx.store["want_neurons"] = planned * ad.cfg.inter * S_ * 2 < int(p.get("neuron_budget_mb", 512)) * 2**20
        ctx.store["want_parts"] = planned * ad.cfg.hidden * S_ * 8 < int(p.get("parts_budget_mb", 768)) * 2**20
        from .blockcode import compile_code
        self.code = {nid: compile_code(c["source"], c["kind"], nid) for nid, c in (p.get("code") or {}).items()}
        ctx.store["code"] = {pos: self.code[nid] for pos, nid in enumerate(plan.schedule_nodes) if nid in self.code}
        self.box_code = {c.kind: c for c in self.code.values() if c.kind != "block"}
        cache, start_pos, spans, hidden = ad.new_cache(S_), 0.0, [], []
        for c, sp in zip(prefixes, specs):
            # Each prefix gets its own span; prefixes aimed at different blocks may reuse the same positions
            # because no block sees both.
            pre, st = c.positioned_for_prepend(sp.get("mode", "shift"), float(sp.get("offset", 0) or 0))
            pre = pre.relabel("imported", sp.get("label") or "imported")
            cache.check_compatible(pre)
            a = len(cache)
            cache = KVCache.concat(cache, pre) if len(pre) else cache
            lay = str(sp.get("layers") or "all").strip().lower()
            slots = None if lay in ("all", "*", "") else (set() if lay == "none" else set(parse_index_spec(lay, S_)))
            spans.append((a, len(cache), slots))
            start_pos = max(start_pos, st)
            if sp.get("mode") == "overwrite":         # these blocks read the prefix *instead of* the prompt
                hidden.append(slots)
        cache.prefix_spans = spans
        n_pre = len(cache)
        cache.hidden_spans = [(n_pre, n_pre + len(prompt_ids), slots) for slots in hidden]
        # Linear-attention blocks: start from the memory of the prefix visible to that block (zero if none).
        # Two prefixes visible to the same linear block can't both be in its single memory: the later one wins.
        cache.states, self.notes = {}, []
        for c, (a, b, slots) in zip(prefixes, spans):
            for slot, stt in c.states.items():
                if slots is None or slot in slots:
                    if slot in cache.states:
                        self.notes.append(f"linear block #{slot} sees several prefixes; only the last one's memory is used")
                    cache.states[slot] = {k: v.detach().clone() for k, v in stt.items()}
        cache.init_states = KVCache._clone_states(cache.states)
        self.cache, self.n_prefix = cache, len(cache)
        self.gen = torch.Generator(device="cpu").manual_seed(p["seed"])
        self.eos = set(ad.cfg.eos_ids) | ({ad.tok.eos_id} if ad.tok.eos_id is not None else set())
        self.w_out = self.model.unembed_weight()
        self.queue = [(ids, o) for ids, o in ((prompt_ids, "prompt"), (prefill_ids, "prefill")) if ids]
        self.seq: list[int] = []
        self.logits_rows: list[torch.Tensor] = []
        self.offset, self.pos_next = 0, start_pos
        self.x, self.bp, self.chunk = None, 0, None
        self.n_generated, self.done, self.forced = 0, False, None
        self.trace: list[dict] = []
        self.progress = progress
        self.steps = 0
        self.trace_on = trace         # per-block lens readout (debugger); off for plain runs (costly when streaming)
        self.last_lens: dict[int, str] = {}

    # ---------------------------------------------------------------- helpers
    def _idx(self, iv, offset, n):
        if iv.positions.strip().lower() == "last":
            return [n - 1]
        glob = parse_index_spec(iv.positions, self.planned, self.ctx.token_ranges)
        return [g - offset for g in glob if offset <= g < offset + n]

    def _lens_top(self, x, k=5):
        lg = self.model.final(x[-1:], self.w_out)[0]
        pv, pi = torch.softmax(lg, -1).topk(k)
        return [[self.ad.tok.piece(int(i)), int(i), round(float(v), 4)] for v, i in zip(pv, pi)]

    def _next_chunk(self):
        if self.queue:
            return self.queue.pop(0)
        if self.done or self.n_generated >= self.total or not self.logits_rows:
            self.done = True
            return None
        if self.forced is not None:
            nxt = self.forced
        elif "sampler" in self.box_code:
            from .blockcode import run_sampler
            nxt = run_sampler(self.box_code["sampler"], self.logits_rows[-1][-1], self.p_, self.gen, self.ctx)
        else:
            nxt = self.ad._sample(self.logits_rows[-1][-1], self.p_, self.gen)
        forced, self.forced = self.forced is not None, None
        if nxt in self.eos and not self.p_.get("ignore_eos") and not forced:
            self.seq.append(nxt)
            self.done = True
            return None
        self.n_generated += 1
        return ([nxt], "generated")

    def _report(self, pos):
        if self.progress:
            nb = len(self.plan.schedule)
            self.progress((self.n_generated + (pos + 1) / nb) / (self.total + 1),
                          f"{'prefill' if self.n_generated == 0 else f'token {self.n_generated}/{self.total}'} · block {pos + 1}/{nb}")

    # ---------------------------------------------------------------- stepping
    @torch.no_grad()
    def step_block(self) -> bool:
        """Advance one arrow in the flow. Returns False when the run is finished."""
        if self.ctx.cancel:
            raise Cancelled("Run cancelled.")
        self.steps += 1
        if self.x is None:                       # idle: pick the next chunk and embed it
            ch = self._next_chunk()
            if ch is None:
                return False
            ids, origin = ch
            positions = [self.pos_next + i for i in range(len(ids))]
            self.cache.add_tokens(ids, positions, origin)
            if "embed" in self.box_code:
                from .blockcode import run_embed
                self.x = run_embed(self.model, self.box_code["embed"], ids, self.ctx)
            else:
                self.x = self.model.embed(ids)
            self.chunk, self.bp, self.trace = ch, 0, []
            if self.trace_on:
                self.trace.append({"pos": -1, "node": "embed", "label": "Embed", "top": self._lens_top(self.x),
                                   "resid_norm": round(float(self.x[-1].float().norm()), 3)})
            return True
        p, sched = self.bp, self.plan.schedule
        ids = self.chunk[0]
        self.ctx.offset = self.offset
        self.x = self.model.block(p, sched[p], self.x, self.cache, self.ctx, lambda iv: self._idx(iv, self.offset, len(ids)))
        self._report(p)
        st = self.ctx.store["stats"][p]
        if not self.trace_on:
            snap = None
        else:
            snap = {"pos": p, "layer": sched[p], "node": self.plan.schedule_nodes[p], "top": self._lens_top(self.x),
                "resid_norm": round(float(st["resid"][-1][-1]), 3), "attn_norm": round(float(st["attn"][-1][-1]), 3),
                "mlp_norm": round(float(st["mlp"][-1][-1]), 3)}
        rows = self.ctx.store.get("attn", {}).get(p)
        if snap is not None:
            if rows:
                row = rows[-1][:, -1, :].float().mean(0)          # last token's attention, mean over heads
                v, i = row.topk(min(5, row.shape[0]))
                snap["attends_to"] = [[int(j), self.ad.tok.piece(self.cache.tokens[int(j)]), round(float(w), 3)]
                                      for w, j in zip(v, i)]
            self.trace.append(snap)
            self.last_lens[p] = snap["top"][0][0]
        self.bp += 1
        if self.bp == len(sched):                 # end of the stack: final norm + unembed
            h = self.x
            if "final_norm" in self.box_code or "lm_head" in self.box_code:
                from .blockcode import run_final_norm, run_lm_head
                h = (run_final_norm(self.model, self.box_code["final_norm"], h, self.ctx)
                     if "final_norm" in self.box_code else self.model.final_norm(h))
                logits = (run_lm_head(self.model, self.box_code["lm_head"], h, self.w_out, self.ctx)
                          if "lm_head" in self.box_code else self.model.unembed(h, self.w_out))
            else:
                logits = self.model.final(h, self.w_out)
            self.logits_rows.append(logits)
            self.seq += ids
            self.offset += len(ids)
            self.pos_next += len(ids)
            self.x = None
        return True

    def step_token(self) -> bool:
        if self.x is None and not self.step_block():
            return False
        while self.x is not None:
            self.step_block()
        return True

    def run_to(self, pos: int, limit: int = 100_000) -> bool:
        """Run until block at schedule position `pos` has just executed (in the next forward pass)."""
        first = True
        for _ in range(limit):
            if not first and self.x is not None and self.bp == pos + 1:
                return True
            first = False
            if not self.step_block():
                return False
        return True

    def force(self, token_id: int):
        self.forced = int(token_id)

    def state(self) -> dict:
        tok = self.ad.tok
        st = {"done": self.done, "phase": "done" if self.done else ("idle" if self.x is None else "block"),
              "next_pos": self.bp if self.x is not None else None,
              "active_node": (self.plan.schedule_nodes[self.bp] if self.x is not None and self.bp < len(self.plan.schedule) else None),
              "last_node": self.trace[-1]["node"] if self.trace else None,
              "chunk": [tok.piece(i) for i in self.chunk[0]] if self.chunk else [],
              "chunk_origin": self.chunk[1] if self.chunk else None,
              "tokens": [tok.piece(i) for i in self.seq], "n_in": self.n_in, "n_generated": self.n_generated,
              "trace": self.trace, "forced": self.forced, "steps": self.steps,
              "text": tok.decode(self.seq[self.n_in:], skip_special=True)}
        if self.x is None and self.logits_rows and not self.done:
            pv, pi = torch.softmax(self.logits_rows[-1][-1], -1).topk(10)
            st["next_choices"] = [[tok.piece(int(i)), int(i), round(float(v), 4)] for v, i in zip(pv, pi)]
        return st

    # ---------------------------------------------------------------- results
    def finish(self) -> dict:
        if not self.logits_rows:
            raise ValueError("Nothing was computed yet.")
        return self.snapshot()

    def snapshot(self) -> dict:
        """Outputs + captures for the current state, in the same shape a finished Run has, so every analysis
        view (logit lens, word cloud, attention, KV cache, flow badges) works while stepping. Tokens still in
        flight are included: layers that have already processed them have data, later layers don't yet."""
        ad, ctx, plan = self.ad, self.ctx, self.plan
        inflight = list(self.chunk[0]) if self.x is not None else []
        n_proc = sum(r.shape[0] for r in self.logits_rows)
        proc_ids = self.seq[:n_proc] + inflight
        if not proc_ids:
            raise ValueError("Nothing was computed yet.")
        if self.logits_rows:
            logits = torch.cat(self.logits_rows)
            probs = torch.softmax(logits, -1)
            tv, ti = probs.topk(10, -1)
            alts = [[[ad.tok.piece(int(ti[r, j])), int(ti[r, j]), round(float(tv[r, j]), 4)] for j in range(10)]
                    for r in range(n_proc)]
            final_top = logits.argmax(-1).tolist()
        else:
            alts, final_top = [], []
        caps = ad._collect(ctx, plan, proc_ids, self.n_prefix)
        if inflight and self.bp == 0:                      # just embedded: show the embeddings already
            cur = self.x.detach().float().cpu()
            prev = caps["resid_pre"].get(0)
            caps["resid_pre"][0] = cur if prev is None else torch.cat([prev, cur])
        caps["cache"] = self.cache
        caps["final_top"] = final_top
        caps["n_in"] = self.n_in
        caps["linear_slots"] = [pp for pp, l in enumerate(plan.schedule) if ad.cfg.is_linear(l)]
        caps["consistent"] = self.x is None
        block_stats = {}
        for pp in range(len(plan.schedule)):
            st = ctx.store.get("stats", {}).get(pp)
            if not st:
                continue
            lens_last = self.last_lens.get(pp)
            if lens_last is None and pp in caps["resid_post"]:
                lg = self.model.final(caps["resid_post"][pp][-1:].to(ad.device, self.model.dtype), self.w_out)
                lens_last = ad.tok.piece(int(lg[0].argmax()))
            block_stats[plan.schedule_nodes[pp]] = {
                "pos": pp, "layer": plan.schedule[pp],
                "resid_norm": round(float(torch.cat(st["resid"]).mean()), 3),
                "attn_norm": round(float(torch.cat(st["attn"]).mean()), 3),
                "mlp_norm": round(float(torch.cat(st["mlp"]).mean()), 3),
                "lens_last": lens_last,
                "pending": bool(inflight) and pp >= self.bp}       # hasn't seen the in-flight token yet
        n_in = self.n_in
        seq = self.seq
        outputs = {
            "text": ad.tok.decode(seq[n_in:], skip_special=True),
            "tokens": [ad.tok.piece(i) for i in seq + inflight], "token_ids": seq + inflight,
            "segments": [{"origin": "prompt", "start": 0, "end": len(self.prompt_ids)},
                         {"origin": "prefill", "start": len(self.prompt_ids), "end": n_in},
                         {"origin": "generated", "start": max(n_in, 0), "end": max(n_in, len(seq))},
                         {"origin": "inflight", "start": len(seq), "end": len(seq) + len(inflight)}],
            "alternatives": alts,              # alts[i] = distribution that predicted token i+1
            "block_stats": block_stats,
            "n_heads": ad.cfg.n_heads,
            "prefix_tokens": self.n_prefix,
            "cache": self.cache.summary(ad.tok.piece),
            "attention_captured": bool(caps["attn"]),
            "weights": self.model.W.describe(),
            "gb_read": round(self.model.W.bytes_read / 2**30, 2),
            "notes": getattr(self, "notes", []),
            "prints": list(ctx.store.get("prints", [])),      # print() output of edited box code
            "edited_code": sorted(self.code),
            "captured": {"neurons": bool(caps["neurons"]), "parts": bool(caps["mlp_out"])},
            "live": {"phase": "done" if self.done else ("idle" if self.x is None else "block"),
                     "next_pos": self.bp if self.x is not None else None, "steps": self.steps,
                     "n_complete": n_proc, "inflight": len(inflight)},
        }
        return {"outputs": outputs, "captures": caps}
