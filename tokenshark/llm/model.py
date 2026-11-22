"""Raw PyTorch decoder-only LLM engine (`tokenshark.llm.model`).

Executes explicit block-by-block forward passes with full hook support, MoE routing,
hybrid linear attention, and side-branch `fork` outputs.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..engine import RunContext, apply_op
from ..flow import parse_index_spec
from ..kvcache import KVCache
from ..weights import WeightStore
from .config import LLMConfig, meta_name_map, meta_transform
from .ops import _act, apply_rope, rms_norm, rope_inv_freq, yarn_scales


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
        self.rope_mult, self.score_mult = yarn_scales(cfg)
        fa = next((i for i in range(cfg.n_layers) if not cfg.is_linear(i)), 0)     # a full-attention layer
        self.qk_norm = self.W.has(f"model.layers.{fa}.self_attn.q_norm.weight")
        self.has_bias = self.W.has(f"model.layers.{fa}.self_attn.q_proj.bias")
        qshape = self.W.shape(f"model.layers.{fa}.self_attn.q_proj.weight")
        self.attn_gate = bool(qshape) and qshape[0] == 2 * cfg.n_heads * cfg.head_dim   # Qwen3.5 gated attention
        self.tied = cfg.tied or not self.W.has("lm_head.weight")
        self.act = _act(cfg.act)
        qn = self.W.shape(f"model.layers.{fa}.self_attn.q_norm.weight") if self.qk_norm else None
        self.qk_norm_flat = bool(qn) and qn[0] == cfg.n_heads * cfg.head_dim and cfg.n_heads > 1
        self.moe_layers = {l: lay for l in range(cfg.n_layers) if (lay := self._moe_layout(l))} if cfg.moe else {}
        if cfg.moe and not self.moe_layers:
            raise ValueError("config.json describes a mixture-of-experts model but no router weights "
                             "(…mlp.gate.weight / …block_sparse_moe.gate.weight) were found in the checkpoint.")

    def _moe_layout(self, l: int) -> dict | None:
        W = self.W
        for base in (f"model.layers.{l}.mlp.", f"model.layers.{l}.block_sparse_moe."):
            if not W.has(base + "gate.weight"):
                continue
            shared = next((s for s in ("shared_expert", "shared_experts") if W.has(f"{base}{s}.gate_proj.weight")), None)
            lay = {"base": base, "router": base + "gate.weight", "n": W.shape(base + "gate.weight")[0],
                   "shared": bool(shared), "shared_name": shared,
                   "shared_gate": W.has(base + "shared_expert_gate.weight")}
            if W.has(base + "experts.gate_up_proj"):
                lay["names"] = "stacked"
            elif W.has(base + "experts.0.gate_proj.weight"):
                lay["names"] = ("gate_proj", "up_proj", "down_proj")
            elif W.has(base + "experts.0.w1.weight"):
                lay["names"] = ("w1", "w3", "w2")
            else:
                continue
            return lay
        return None

    def expert_weights(self, lay: dict, e, ctx: RunContext | None = None):
        b = lay["base"]
        if e == "shared":
            return tuple(self._w(f"{b}{lay['shared_name']}.{n}.weight", ctx) for n in ("gate_proj", "up_proj", "down_proj"))
        if lay["names"] == "stacked":
            gu, dn = self._w(b + "experts.gate_up_proj", ctx)[e], self._w(b + "experts.down_proj", ctx)[e]
            half = gu.shape[0] // 2
            return gu[:half], gu[half:], dn
        return tuple(self._w(f"{b}experts.{e}.{n}.weight", ctx) for n in lay["names"])

    # -------------------------------------------------------------- pieces
    def embed(self, ids: list[int]) -> torch.Tensor:
        return self.W.rows("model.embed_tokens.weight", ids)

    def unembed_weight(self) -> torch.Tensor:
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
        W = w_out if w_out is not None else self.unembed_weight()
        if W.dtype == h.dtype and W.device == h.device:
            return (h @ W.T).float()
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
        n_heads = t.shape[0]
        tt = t.transpose(0, 1) if token_dim == 1 else t
        if cap:
            self.note_units(p, hook, tt.reshape(tt.shape[0], -1), ctx)
        for iv in ctx.active(hook):
            heads = parse_index_spec(iv.heads, n_heads)
            if not heads:
                continue
            sub = tt[:, heads]
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
        cfg = self.cfg
        pre = f"model.layers.{l}."
        ctx.pos = p
        for iv in ctx.active("resid_pre"):
            x = apply_op(x, iv, ctx, 0, idx_of(iv))
        cap = ctx.capture
        if cap:
            ctx.store.setdefault("resid_pre", {}).setdefault(p, []).append(x.detach().float().cpu())

        code = ctx.store.get("code", {}).get(p)
        if code is not None:
            from ..blockcode import run_block
            x, attn_out, mlp_out = run_block(self, code, p, l, x, cache, ctx)
            return self._finish(p, x, attn_out, mlp_out, ctx, idx_of, cap)

        h = rms_norm(x, self.norm_w(pre + "input_layernorm.weight"), cfg.eps)
        if cfg.is_linear(l):
            attn_out = self._linear_attn(p, pre, h, cache, ctx, idx_of)
        elif cfg.mla:
            attn_out = self._mla_attn(p, pre, h, cache, ctx, idx_of)
        else:
            attn_out = self._full_attn(p, pre, h, cache, ctx, idx_of)
        for iv in ctx.active("attn_out"):
            attn_out = apply_op(attn_out, iv, ctx, 0, idx_of(iv))
        x = x + attn_out
        return self._mlp_and_finish(p, pre, x, attn_out, ctx, idx_of, cap)

    def attn_allowed(self, p: int, n: int, T: int, cache: KVCache, device,
                     ctx: RunContext | None = None) -> torch.Tensor:
        bc = cache.block_caches.get(p)
        if bc is not None:
            if ctx is not None and ctx.store.get("chunk_origin") == "prompt":
                qi = torch.linspace(0, max(0, T - 1), steps=n, device=device).floor()[:, None]
            else:
                t0 = T - n
                qi = torch.arange(t0, T, device=device)[:, None]
            kj = torch.arange(T, device=device)[None, :]
            allowed = kj <= qi
            if self.cfg.sliding_window:
                allowed &= kj > qi - self.cfg.sliding_window
            return allowed
        t0 = T - n
        qi = torch.arange(t0, T, device=device)[:, None]
        kj = torch.arange(T, device=device)[None, :]
        allowed = kj <= qi
        if self.cfg.sliding_window:
            allowed &= kj > qi - self.cfg.sliding_window
        for a, b, slots in cache.prefix_spans:
            if slots is not None and p not in slots:
                allowed &= (kj < a) | (kj >= b)
        return allowed

    def note_units(self, p: int, hook: str, a: torch.Tensor, ctx: RunContext):
        lo, hi = a.detach().float().amin(0).cpu(), a.detach().float().amax(0).cpu()
        rng = ctx.store.setdefault("unit_range", {})
        old = rng.get((p, hook))
        rng[(p, hook)] = (lo, hi) if old is None else (torch.minimum(old[0], lo), torch.maximum(old[1], hi))

    def _rope_for_block(self, p: int, n: int, T: int, cache: KVCache, ctx: RunContext):
        bc = cache.block_caches.get(p)
        r_cfg = (ctx.store.get("rope") or {}).get(p) or {}
        inv_freq = self.inv_freq
        if r_cfg.get("theta"):
            inv_freq = rope_inv_freq(self.cfg, theta=float(r_cfg["theta"])).to(self.inv_freq.device)
        if bc is not None:
            pos_k = bc.pos[:T].to(self.inv_freq.device).float().clone()
            if "scale" in r_cfg:
                pos_k = pos_k * float(r_cfg["scale"])
            for iv in ctx.active("rope"):
                if T > 0:
                    sub = pos_k[:, None]
                    loc_idx = [T - 1] if iv.positions.strip().lower() == "last" else parse_index_spec(iv.positions, T)
                    if loc_idx:
                        pos_k = apply_op(sub, iv, ctx, 0, loc_idx)[:, 0]
            if ctx.store.get("chunk_origin") == "prompt":
                main_pos = cache.pos[-n:].to(self.inv_freq.device).float().clone()
                delta = (float(bc.pos[bc.n_prompt - 1]) - float(main_pos[-1])) if (bc.n_prompt > 0 and len(main_pos) > 0) else 0.0
                pos_q = main_pos + delta
                if "scale" in r_cfg:
                    pos_q = pos_q * float(r_cfg["scale"])
            else:
                pos_q = pos_k[T - n:].clone()
            if r_cfg.get("q_offset"):
                pos_q = pos_q + float(r_cfg["q_offset"])
            if ctx.capture:
                ctx.store.setdefault("rope_pos", {})[p] = (pos_q.detach().cpu(), pos_k.detach().cpu())
            return pos_q, pos_k, inv_freq

        pos_k = cache.pos.to(self.inv_freq.device).float().clone()
        if "scale" in r_cfg:
            pos_k = pos_k * float(r_cfg["scale"])
        n_pre = sum(b - a for a, b, _ in cache.prefix_spans)
        for iv in ctx.active("rope"):
            run_len = max(0, T - n_pre)
            if run_len > 0:
                sub = pos_k[n_pre:, None]
                if iv.positions.strip().lower() == "last":
                    loc_idx = [run_len - 1]
                else:
                    rng = {k: range(min(r.start, run_len), min(r.stop, run_len))
                           for k, r in (ctx.token_ranges or {}).items()}
                    loc_idx = parse_index_spec(iv.positions, run_len, rng)
                if loc_idx:
                    pos_k[n_pre:] = apply_op(sub, iv, ctx, 0, loc_idx)[:, 0]
        t0 = T - n
        pos_q = pos_k[t0:].clone()
        if r_cfg.get("q_offset"):
            pos_q = pos_q + float(r_cfg["q_offset"])
        if ctx.capture:
            ctx.store.setdefault("rope_pos", {})[p] = (pos_q.detach().cpu(), pos_k.detach().cpu())
        return pos_q, pos_k, inv_freq

    def _full_attn(self, p, pre, h, cache, ctx, idx_of):
        cfg = self.cfg
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
        if self.qk_norm and self.qk_norm_flat:
            q = rms_norm(q.transpose(0, 1).reshape(n, -1), self.norm_w(pre + "self_attn.q_norm.weight"), cfg.eps)
            k = rms_norm(k.transpose(0, 1).reshape(n, -1), self.norm_w(pre + "self_attn.k_norm.weight"), cfg.eps)
            q = q.view(n, cfg.n_heads, cfg.head_dim).transpose(0, 1)
            k = k.view(n, cfg.n_kv, cfg.head_dim).transpose(0, 1)
        elif self.qk_norm:
            q = rms_norm(q, self.norm_w(pre + "self_attn.q_norm.weight"), cfg.eps)
            k = rms_norm(k, self.norm_w(pre + "self_attn.k_norm.weight"), cfg.eps)
        if cfg.clip_qkv:
            q, k, v = (t.clamp(-cfg.clip_qkv, cfg.clip_qkv) for t in (q, k, v))
        q = self._head_hook(p, "q", q, ctx, idx_of, cap)
        k = self._head_hook(p, "k", k, ctx, idx_of, cap)
        v = self._head_hook(p, "v", v, ctx, idx_of, cap)
        x = h
        bc = cache.block_caches.get(p)
        if bc is not None:
            if ctx.store.get("chunk_origin") != "prompt":
                bc.append(k, v)
            if ctx.active("kv_k") and bc.k is not None:
                bc.k = self._head_hook(p, "kv_k", bc.k.to(x.device), ctx, idx_of, False).to(bc.device)
            if ctx.active("kv_v") and bc.v is not None:
                bc.v = self._head_hook(p, "kv_v", bc.v.to(x.device), ctx, idx_of, False).to(bc.device)
            K = bc.k.to(x.device)
            V = bc.v.to(x.device)
        else:
            cache.append(p, k, v)
            if ctx.active("kv_k"):
                cache.k[p] = self._head_hook(p, "kv_k", cache.k[p].to(x.device), ctx, idx_of, False).to(cache.device)
            if ctx.active("kv_v"):
                cache.v[p] = self._head_hook(p, "kv_v", cache.v[p].to(x.device), ctx, idx_of, False).to(cache.device)
            K = cache.k[p].to(x.device)
            V = cache.v[p].to(x.device)
        T = K.shape[1]
        pos_q, pos_k, inv_freq = self._rope_for_block(p, n, T, cache, ctx)
        qr = apply_rope(q, pos_q, inv_freq)
        kr = apply_rope(K, pos_k, inv_freq)
        rep = cfg.n_heads // cfg.n_kv
        if rep > 1:
            kr = kr.repeat_interleave(rep, 0)
            V = V.repeat_interleave(rep, 0)
        scores = (qr.float() @ kr.float().transpose(1, 2)) / math.sqrt(cfg.head_dim)
        scores = scores.masked_fill(~self.attn_allowed(p, n, T, cache, x.device, ctx), float("-inf"))
        probs = torch.softmax(scores, -1)
        if cap and ctx.store.get("want_attn", True):
            ctx.store.setdefault("attn", {}).setdefault(p, []).append(probs.detach().to(torch.float16).cpu())
        z = (probs.to(V.dtype) @ V)
        for iv in ctx.active("attn_heads"):
            heads = parse_index_spec(iv.heads, cfg.n_heads)
            zt = z.transpose(0, 1)
            sub = zt[:, heads]
            sub = apply_op(sub, iv, ctx, 0, idx_of(iv))
            zt = zt.clone()
            zt[:, heads] = sub
            z = zt.transpose(0, 1)
        zf = z.transpose(0, 1).reshape(n, cfg.n_heads * cfg.head_dim)
        if gate is not None:
            zf = zf * torch.sigmoid(gate.reshape(n, -1))
        return zf @ self._w(pre + "self_attn.o_proj.weight", ctx).T

    def _mla_attn(self, p, pre, h, cache, ctx, idx_of):
        cfg, a = self.cfg, self.cfg.mla
        n, H, cap = h.shape[0], cfg.n_heads, ctx.capture
        s = pre + "self_attn."
        if a["q_rank"]:
            q = rms_norm(h @ self._w(s + "q_a_proj.weight", ctx).T, self.W(s + "q_a_layernorm.weight"), cfg.eps)
            q = q @ self._w(s + "q_b_proj.weight", ctx).T
        else:
            q = h @ self._w(s + "q_proj.weight", ctx).T
        q = q.view(n, H, a["nope"] + a["rope"])
        ckv = h @ self._w(s + "kv_a_proj_with_mqa.weight", ctx).T
        latent = rms_norm(ckv[:, :a["rank"]], self.W(s + "kv_a_layernorm.weight"), cfg.eps)
        kv = (latent @ self._w(s + "kv_b_proj.weight", ctx).T).view(n, H, a["nope"] + a["v"])
        order = torch.cat([torch.arange(0, a["rope"], 2), torch.arange(1, a["rope"], 2)]).to(h.device)
        q = torch.cat([q[..., a["nope"]:][..., order], q[..., :a["nope"]]], -1).transpose(0, 1)
        k_pos = ckv[:, a["rank"]:][:, order][:, None, :].expand(n, H, a["rope"])
        k = torch.cat([k_pos, kv[..., :a["nope"]]], -1).transpose(0, 1)
        v = kv[..., a["nope"]:].transpose(0, 1)
        q = self._head_hook(p, "q", q, ctx, idx_of, cap)
        k = self._head_hook(p, "k", k, ctx, idx_of, cap)
        v = self._head_hook(p, "v", v, ctx, idx_of, cap)
        bc = cache.block_caches.get(p)
        if bc is not None:
            if ctx.store.get("chunk_origin") != "prompt":
                bc.append(k, v)
            if ctx.active("kv_k") and bc.k is not None:
                bc.k = self._head_hook(p, "kv_k", bc.k.to(h.device), ctx, idx_of, False).to(bc.device)
            if ctx.active("kv_v") and bc.v is not None:
                bc.v = self._head_hook(p, "kv_v", bc.v.to(h.device), ctx, idx_of, False).to(bc.device)
            K, V = bc.k.to(h.device), bc.v.to(h.device)
        else:
            cache.append(p, k, v)
            if ctx.active("kv_k"):
                cache.k[p] = self._head_hook(p, "kv_k", cache.k[p].to(h.device), ctx, idx_of, False).to(cache.device)
            if ctx.active("kv_v"):
                cache.v[p] = self._head_hook(p, "kv_v", cache.v[p].to(h.device), ctx, idx_of, False).to(cache.device)
            K, V = cache.k[p].to(h.device), cache.v[p].to(h.device)
        T = K.shape[1]
        pos_q, pos_k, inv_freq = self._rope_for_block(p, n, T, cache, ctx)
        qr = apply_rope(q, pos_q, inv_freq).float()
        kr = apply_rope(K, pos_k, inv_freq).float()
        if self.rope_mult != 1.0:
            qr = torch.cat([qr[..., :a["rope"]] * self.rope_mult, qr[..., a["rope"]:]], -1)
            kr = torch.cat([kr[..., :a["rope"]] * self.rope_mult, kr[..., a["rope"]:]], -1)
        scores = (qr @ kr.transpose(1, 2)) * (self.score_mult / math.sqrt(cfg.head_dim))
        scores = scores.masked_fill(~self.attn_allowed(p, n, T, cache, h.device, ctx), float("-inf"))
        probs = torch.softmax(scores, -1)
        if cap and ctx.store.get("want_attn", True):
            ctx.store.setdefault("attn", {}).setdefault(p, []).append(probs.detach().to(torch.float16).cpu())
        z = probs.to(V.dtype) @ V
        for iv in ctx.active("attn_heads"):
            heads = parse_index_spec(iv.heads, H)
            zt = z.transpose(0, 1).clone()
            zt[:, heads] = apply_op(zt[:, heads], iv, ctx, 0, idx_of(iv))
            z = zt.transpose(0, 1)
        return z.transpose(0, 1).reshape(n, H * a["v"]) @ self._w(s + "o_proj.weight", ctx).T

    def _linear_attn(self, p, pre, h, cache, ctx, idx_of):
        cfg, W = self.cfg, self.W
        L = cfg.lin
        n = h.shape[0]
        a_ = pre + "linear_attn."
        mixed = h @ self._w(a_ + "in_proj_qkv.weight", ctx).T
        z = (h @ self._w(a_ + "in_proj_z.weight", ctx).T).view(n, L["hv"], L["dv"])
        b = h @ self._w(a_ + "in_proj_b.weight", ctx).T
        a = h @ self._w(a_ + "in_proj_a.weight", ctx).T
        st = cache.lin_state(p, mixed.shape[1], L, h.device)
        K = L["conv"]
        seq = torch.cat([st["conv"].to(mixed.dtype), mixed], 0)
        cw = W(a_ + "conv1d.weight")
        cb = W(a_ + "conv1d.bias") if W.has(a_ + "conv1d.bias") else None
        conv = F.conv1d(seq.T.unsqueeze(0).float(), cw.float(), cb.float() if cb is not None else None,
                        groups=cw.shape[0])[0].T
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
        beta = torch.sigmoid(b.float())
        g = -W(a_ + "A_log").float().exp() * F.softplus(a.float() + W(a_ + "dt_bias").float())
        S = st["rec"]
        outs = []
        for t in range(n):
            S = S * g[t].exp()[:, None, None]
            kt, vt, qt = k[t], v[t].float(), q[t]
            mem = (S * kt[:, :, None]).sum(1)
            S = S + kt[:, :, None] * ((vt - mem) * beta[t][:, None])[:, None, :]
            outs.append((S * qt[:, :, None]).sum(1))
        st["rec"] = S
        o = torch.stack(outs)
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
        lay = self.moe_layers.get(int(pre.split(".")[2]))
        if lay is not None:
            mlp_out = self._moe_mlp(p, lay, h2, ctx, idx_of, cap)
        else:
            gate = h2 @ self._w(pre + "mlp.gate_proj.weight", ctx).T
            up = h2 @ self._w(pre + "mlp.up_proj.weight", ctx).T
            neurons = self.act(gate) * up
            neurons = self.neuron_hook(p, neurons, ctx, idx_of, cap)
            mlp_out = neurons @ self._w(pre + "mlp.down_proj.weight", ctx).T
        for iv in ctx.active("mlp_out"):
            mlp_out = apply_op(mlp_out, iv, ctx, 0, idx_of(iv))
        x = x + mlp_out
        if W.has(pre + "adapter.down.weight"):
            x = x + self.act(x @ self._w(pre + "adapter.down.weight", ctx).T) @ self._w(pre + "adapter.up.weight", ctx).T
        return self._finish(p, x, attn_out, mlp_out, ctx, idx_of, cap)

    # -------------------------------------------------------------- mixture of experts
    @staticmethod
    def expert_ids(spec: str, lay: dict) -> list[int]:
        E = lay["n"]
        spec = (spec or "all").strip().lower()
        if spec in ("all", "*", ""):
            return list(range(E))
        parts = [x.strip().split("=")[0].strip() for x in spec.split(",") if x.strip()]
        out = [E] if "shared" in parts and lay["shared"] else []
        rest = ",".join(x for x in parts if x != "shared")
        return sorted(set(out + (parse_index_spec(rest, E) if rest else [])))

    @staticmethod
    def route_spec(spec: str, E: int) -> list[tuple[int, float | None]]:
        out = []
        for part in (spec or "").split(","):
            if not part.strip():
                continue
            e, _, wt = part.partition("=")
            e = int(e.strip())
            if not 0 <= e < E:
                raise ValueError(f"This block has experts 0…{E - 1}; a custom route names expert {e}.")
            out.append((e, float(wt) if wt.strip() else None))
        return out

    def _moe_mlp(self, p: int, lay: dict, h2: torch.Tensor, ctx: RunContext, idx_of, cap: bool) -> torch.Tensor:
        m = self.cfg.moe
        n, d = h2.shape
        E = lay["n"]
        logits = (h2 @ self._w(lay["router"], ctx).T).float()
        for iv in ctx.active("router"):
            if iv.op == "route":
                continue
            ex = [e for e in self.expert_ids(iv.experts, lay) if e < E]
            idx = idx_of(iv)
            if not ex or not idx:
                continue
            if iv.op in ("ban", "only"):
                off = torch.zeros(E, dtype=torch.bool, device=logits.device)
                off[ex] = True
                if iv.op == "only":
                    off = ~off
                rows = torch.tensor(idx, device=logits.device)
                sub = logits[rows]
                sub[:, off] = -1e9
                logits = logits.clone()
                logits[rows] = sub
            else:
                one = dataclasses.replace(iv, neurons=",".join(map(str, ex))) if len(ex) < E else iv
                logits = apply_op(logits, one, ctx, 0, idx)
        temp = (ctx.store.get("moe_temp") or {}).get(p)
        probs = torch.softmax(logits / temp if temp else logits, -1)
        if m.get("groups"):
            g, keep_g = m["groups"]
            best = probs.view(n, g, E // g).amax(-1).topk(keep_g, -1).indices
            ok = torch.zeros(n, g, dtype=torch.bool, device=probs.device).scatter_(1, best, True)
            probs = probs * ok[:, :, None].expand(n, g, E // g).reshape(n, E)
        k = max(1, min(E, int((ctx.store.get("moe_k") or {}).get(p) or m["k"])))
        w, sel = probs.topk(k, -1)
        if m["norm_topk"]:
            w = w / w.sum(-1, keepdim=True)
        if m.get("scale", 1.0) != 1.0:
            w = w * m["scale"]
        for iv in ctx.active("router"):
            if iv.op != "route":
                continue
            pairs, idx = self.route_spec(iv.experts, E), idx_of(iv)
            if not pairs or not idx:
                continue
            if len(pairs) > sel.shape[1]:
                pad = len(pairs) - sel.shape[1]
                sel = torch.cat([sel, sel.new_zeros(n, pad)], 1)
                w = torch.cat([w, w.new_zeros(n, pad)], 1)
            rows = torch.tensor(idx, device=sel.device)
            es = torch.tensor([e for e, _ in pairs], device=sel.device)
            own = probs[rows][:, es]
            if all(wt is None for _, wt in pairs) and m["norm_topk"]:
                own = own / own.sum(-1, keepdim=True).clamp(min=1e-12)
            given = torch.tensor([float("nan") if wt is None else wt for _, wt in pairs], device=sel.device)
            wr = torch.where(torch.isnan(given)[None, :], own.float(), given[None, :].expand_as(own).float())
            new_sel, new_w = sel[rows] * 0, w[rows] * 0
            new_sel[:, :len(pairs)], new_w[:, :len(pairs)] = es, wr.to(w.dtype)
            sel, w = sel.clone(), w.clone()
            sel[rows], w[rows] = new_sel, new_w
        k = sel.shape[1]
        keep = cap and ctx.store.get("want_experts", False)
        parts = torch.zeros(n, k, d, dtype=torch.float16) if keep else None
        out = torch.zeros_like(h2)

        def scoped(hook, e, rows, t):
            for iv in ctx.active(hook):
                if hook == "mlp_neurons" and (iv.experts or "all").strip().lower() in ("all", "*", ""):
                    continue
                if e not in self.expert_ids(iv.experts, lay):
                    continue
                want = set(idx_of(iv))
                loc = [j for j, r in enumerate(rows) if r in want]
                if loc:
                    t = apply_op(t, iv, ctx, 0, loc)
            return t

        for e in sel.unique().tolist():
            rows, slot = torch.where((sel == e) & (w != 0))
            if not rows.numel():
                continue
            he = h2[rows]
            g, u, dn = self.expert_weights(lay, e, ctx)
            neurons = scoped("mlp_neurons", e, rows.tolist(), self.act(he @ g.T) * (he @ u.T))
            oe = scoped("expert_out", e, rows.tolist(), neurons @ dn.T)
            oe = oe * w[rows, slot, None].to(oe.dtype)
            out.index_add_(0, rows, oe)
            if parts is not None:
                parts[rows.cpu(), slot.cpu()] = oe.detach().to(torch.float16).cpu()
        shared = None
        if lay["shared"] and p not in (ctx.store.get("moe_noshared") or ()):
            g, u, dn = self.expert_weights(lay, "shared", ctx)
            shared = (self.act(h2 @ g.T) * (h2 @ u.T)) @ dn.T
            if lay["shared_gate"]:
                shared = shared * torch.sigmoid(h2 @ self._w(lay["base"] + "shared_expert_gate.weight", ctx).T)
            shared = scoped("expert_out", E, list(range(n)), shared)
            out = out + shared
        if cap:
            self.note_units(p, "router", logits.clamp(min=-1e4), ctx)
            st = ctx.store
            st.setdefault("router", {}).setdefault(p, []).append(logits.detach().clamp(min=-1e4).float().cpu())
            st.setdefault("moe_sel", {}).setdefault(p, []).append(sel.detach().to(torch.int16).cpu())
            st.setdefault("moe_w", {}).setdefault(p, []).append(w.detach().float().cpu())
            if parts is not None:
                st.setdefault("moe_parts", {}).setdefault(p, []).append(parts)
                if shared is not None:
                    st.setdefault("moe_shared", {}).setdefault(p, []).append(shared.detach().to(torch.float16).cpu())
        return out

    def neuron_hook(self, p, neurons, ctx, idx_of, cap):
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
            if ctx.store.get("want_parts", False):
                for key, t in (("attn_out", attn_out), ("mlp_out", mlp_out)):
                    if t is not None:
                        ctx.store.setdefault(key, {}).setdefault(p, []).append(t.detach().float().cpu())
        # Side-branch fork execution: any ForkSpec wired to this block's "fork" port runs on resid_post
        forks_at_p = (ctx.store.get("forks") or {}).get(p) or []
        if forks_at_p:
            from ..sdk import get_fork_runner
            for fspec in forks_at_p:
                runner = get_fork_runner(fspec.kind)
                if runner is not None:
                    res = runner.run_fn(fspec, x, ctx, self)
                    if isinstance(res, dict):
                        if "override_x" in res and isinstance(res["override_x"], torch.Tensor):
                            x = res.pop("override_x")
                        ctx.store.setdefault("fork_outputs", {})[fspec.node_id] = res
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
