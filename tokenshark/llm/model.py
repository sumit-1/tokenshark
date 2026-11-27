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

from ..engine import RunContext, apply_op, store_attention, to_host
from ..flow import parse_index_spec
from ..kvcache import KVCache
from ..weights import WeightStore
from .arch import ArchHooks
from .config import LLMConfig, meta_name_map, meta_transform
from .ops import _act, apply_rope, rms_norm, rope_inv_freq, yarn_scales


# Memory kept for the KV tab's per-query analysis (all queries of a run; the oldest are dropped past this budget).
_Q_HISTORY_BYTES = 256 * 2 ** 20

class RawLLM:
    def __init__(self, root: Path, device: str, dtype: torch.dtype, policy: str, keep_layers: int,
                 arch: ArchHooks | None = None):
        """arch: what a model plugin adds to the generic reader (see llm/arch.py); None for the built-in families."""
        self.root = Path(root)
        self.arch = arch
        self.cfg = cfg = LLMConfig(self.root, arch)
        self.device, self.dtype = device, dtype
        kw = {"skip_prefixes": arch.weight_skip_prefixes} if arch is not None else {}
        if cfg.meta_format:
            kw.update(name_map=meta_name_map(cfg), transform=meta_transform(cfg))
        self.W = WeightStore(self.root, device, dtype, policy, keep_layers, **kw)
        self.inv_freq = rope_inv_freq(cfg).to(device)
        self.rope_mult, self.score_mult = yarn_scales(cfg)
        fa = next((i for i in range(cfg.n_layers) if not cfg.is_linear(i)), 0)     # a full-attention layer
        self.qk_norm = self.W.has(f"model.layers.{fa}.self_attn.q_norm.weight")
        self.has_bias = self.W.has(f"model.layers.{fa}.self_attn.q_proj.bias")
        qshape = self.W.shape(f"model.layers.{fa}.self_attn.q_proj.weight")
        self.attn_gate = bool(qshape) and qshape[0] == 2 * cfg.n_heads * cfg.layer_head_dim(fa)   # gated attention
        self.tied = cfg.tied or not self.W.has("lm_head.weight")
        self.act = _act(cfg.act)
        qn = self.W.shape(f"model.layers.{fa}.self_attn.q_norm.weight") if self.qk_norm else None
        self.qk_norm_flat = bool(qn) and qn[0] == cfg.n_heads * cfg.head_dim and cfg.n_heads > 1
        self.moe_layers = {l: lay for l in range(cfg.n_layers) if (lay := self._moe_layout(l))} if cfg.moe else {}
        # "Sandwich" blocks: norms after attention / MLP as well as before them (detected from the weight names)
        self.sandwich = self.W.has(f"model.layers.{fa}.pre_feedforward_layernorm.weight")
        self._inv: dict = {}
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
        cfg = self.cfg
        if cfg.media_token_ids:            # media placeholders may lie outside the vocabulary: embed them as pad
            ids = [cfg.pad_id if i in cfg.media_token_ids else i for i in ids]
        x = self.W.rows("model.embed_tokens.weight", ids)
        if cfg.embed_scale is not None:
            x = x * torch.tensor(cfg.embed_scale, dtype=x.dtype)           # the scale is rounded to the weight dtype
        return x

    def inject_media(self, x: torch.Tensor, ids: list[int], feats: torch.Tensor | None) -> torch.Tensor:
        """Overwrite the rows of media placeholder tokens with the media soft-token embeddings, in order."""
        rows = [i for i, t in enumerate(ids) if t in self.cfg.media_token_ids]
        if feats is None or not rows:
            return x
        if len(rows) != feats.shape[0]:
            raise ValueError(f"The prompt has {len(rows)} media tokens but the media give {feats.shape[0]} "
                             f"soft tokens.")
        x = x.clone()
        x[torch.tensor(rows, device=x.device)] = feats.to(x.device, x.dtype)
        return x

    def layer_inv_freq(self, l: int | None, theta: float | None = None) -> torch.Tensor:
        """RoPE frequencies of decoder layer l (layers may carry their own RoPE), optionally with another theta."""
        if theta is None and self.cfg.layer_rope(l) is None:
            return self.inv_freq
        key = (l if self.cfg.layer_rope(l) is not None else None, theta)
        if key not in self._inv:
            self._inv[key] = rope_inv_freq(self.cfg, theta=theta, layer=l).to(self.inv_freq.device)
        return self._inv[key]

    def attn_scale(self, l: int | None) -> float:
        return self.cfg.attn_scale if self.cfg.attn_scale is not None else 1.0 / math.sqrt(self.cfg.layer_head_dim(l))

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
            out = h @ W.T
        else:
            out = torch.empty(h.shape[0], W.shape[0], dtype=h.dtype, device=h.device)
            for i in range(0, W.shape[0], 32768):
                out[:, i:i + 32768] = h @ W[i:i + 32768].to(h.device, h.dtype).T
        cap = self.cfg.final_softcap
        if cap:                                  # logit soft-capping: squashed into (-cap, cap)
            out = torch.tanh(out.float() / cap) * cap
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
                ctx.store.setdefault(hook, {}).setdefault(p, []).append(to_host(tt.reshape(tt.shape[0], -1).detach().float()))
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
            ctx.store.setdefault("resid_pre", {}).setdefault(p, []).append(to_host(x.detach().float()))

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
        if self.sandwich:
            attn_out = rms_norm(attn_out, self.norm_w(pre + "post_attention_layernorm.weight"), cfg.eps)
        for iv in ctx.active("attn_out"):
            attn_out = apply_op(attn_out, iv, ctx, 0, idx_of(iv))
        x = x + attn_out
        return self._mlp_and_finish(p, pre, x, attn_out, ctx, idx_of, cap)

    def _media_blocks(self, tokens: list[int], T: int, device) -> torch.Tensor | None:
        """Per key: index of the contiguous run of media tokens it belongs to, -1 for text (None: no media)."""
        mm = self.cfg.media_token_ids
        toks = list(tokens[:T])
        if not mm or not any(t in mm for t in toks):
            return None
        out, g, prev = [], -1, False
        for t in toks:
            cur = t in mm
            if cur and not prev:
                g += 1
            out.append(g if cur else -1)
            prev = cur
        return torch.tensor(out + [-1] * (T - len(out)), device=device)

    def attn_allowed(self, p: int, n: int, T: int, cache: KVCache, device,
                     ctx: RunContext | None = None, l: int | None = None) -> torch.Tensor:
        window = self.cfg.layer_window(l)
        bc = cache.block_caches.get(cache.shared.get(p, p))
        if bc is not None:
            if ctx is not None and ctx.store.get("chunk_origin") == "prompt":
                qi = torch.linspace(0, max(0, T - 1), steps=n, device=device).floor()[:, None]
            else:
                t0 = T - n
                qi = torch.arange(t0, T, device=device)[:, None]
            kj = torch.arange(T, device=device)[None, :]
            allowed = kj <= qi
            if window:
                allowed &= kj > qi - window
            return allowed
        t0 = T - n
        qi = torch.arange(t0, T, device=device)[:, None]
        kj = torch.arange(T, device=device)[None, :]
        allowed = kj <= qi
        bidir = self.cfg.media_bidir
        if bidir == "all" or (bidir == "sliding" and window):
            blk = self._media_blocks(cache.tokens, T, device)     # one media item's tokens see each other
            if blk is not None:
                allowed |= (blk[t0:, None] == blk[None, :]) & (blk[t0:, None] >= 0)
        if window:
            allowed &= kj > qi - window
        for a, b, slots in cache.prefix_spans:
            if slots is not None and p not in slots:
                allowed &= (kj < a) | (kj >= b)
        return allowed

    def _probs_hook(self, p: int, probs: torch.Tensor, cache, ctx: RunContext) -> torch.Tensor:
        """attn_probs hook: act on the attention weights [heads, queries, keys] after softmax, before they mix the
        values. `positions` picks KEY positions (run-relative, like every other hook), `heads` the heads; the op
        acts on those columns for every query. No renormalisation: scale by λ turns the head's output into
        λ·Σ_{i∈S} p_i v_i + Σ_{i∉S} p_i v_i (the sink-subspace modulation of Mackey et al., 2026)."""
        ivs = ctx.active("attn_probs")
        if not ivs:
            return probs
        H, _, T = probs.shape
        n_pre = sum(b - a for a, b, _ in cache.prefix_spans)      # imported prefix tokens sit before the run
        for iv in ivs:
            heads = parse_index_spec(iv.heads, H)
            keys = [n_pre + j for j in parse_index_spec(iv.positions, max(0, T - n_pre), ctx.token_ranges)]
            if not heads or not keys:
                continue
            hsel = torch.tensor(heads, device=probs.device)
            probs = probs.clone()
            probs[hsel] = apply_op(probs[hsel], iv, ctx, 2, keys)
        return probs

    def note_units(self, p: int, hook: str, a: torch.Tensor, ctx: RunContext):
        lo, hi = a.detach().float().amin(0), a.detach().float().amax(0)      # kept on the device until _collect
        rng = ctx.store.setdefault("unit_range", {})
        old = rng.get((p, hook))
        rng[(p, hook)] = (lo, hi) if old is None else (torch.minimum(old[0].to(lo.device), lo),
                                                       torch.maximum(old[1].to(hi.device), hi))

    def _rope_for_block(self, p: int, n: int, T: int, cache: KVCache, ctx: RunContext, l: int | None = None):
        bc = cache.block_caches.get(cache.shared.get(p, p))
        r_cfg = (ctx.store.get("rope") or {}).get(p) or {}
        inv_freq = self.layer_inv_freq(l, float(r_cfg["theta"]) if r_cfg.get("theta") else None)
        if bc is not None:
            pos_k = bc.pos[:T].to(self.inv_freq.device).float().clone()
            if "scale" in r_cfg:
                pos_k = pos_k * float(r_cfg["scale"])
            if r_cfg.get("gen_start") is not None:
                g0 = next((sg.start for sg in bc.segments if sg.origin == "generated"), None)
                if g0 is not None and g0 < T:
                    pos_k[g0:] += float(r_cfg["gen_start"]) - pos_k[g0]
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
                ctx.store.setdefault("rope_pos", {})[p] = (to_host(pos_q.detach()), to_host(pos_k.detach()))
            return pos_q, pos_k, inv_freq

        pos_k = cache.pos.to(self.inv_freq.device).float().clone()
        if "scale" in r_cfg:
            pos_k = pos_k * float(r_cfg["scale"])
        n_pre = sum(b - a for a, b, _ in cache.prefix_spans)
        if r_cfg.get("gen_start") is not None:       # RoPE jump on this block: generated tokens continue from gen_start
            gen = (ctx.token_ranges or {}).get("generated")
            g0 = n_pre + gen.start if gen is not None else T
            if g0 < T:
                pos_k[g0:] += float(r_cfg["gen_start"]) - pos_k[g0]
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
            ctx.store.setdefault("rope_pos", {})[p] = (to_host(pos_q.detach()), to_host(pos_k.detach()))
        return pos_q, pos_k, inv_freq

    def _full_attn(self, p, pre, h, cache, ctx, idx_of):
        cfg = self.cfg
        n = h.shape[0]
        cap = ctx.capture
        l = int(pre.split(".")[2])
        hd, n_kv = cfg.layer_head_dim(l), cfg.layer_n_kv(l)
        q_all = self._lin(h, pre + "self_attn.q_proj", ctx)
        gate = None
        if self.attn_gate:
            q_all = q_all.view(n, cfg.n_heads, 2 * hd)
            q, gate = q_all[..., :hd], q_all[..., hd:]
            q = q.transpose(0, 1)
        else:
            q = q_all.view(n, cfg.n_heads, hd).transpose(0, 1)
        src = cache.shared.get(p)                 # cross-layer KV sharing: this block has no k/v projections
        k = v = None
        if src is None:
            k = self._lin(h, pre + "self_attn.k_proj", ctx).view(n, n_kv, hd).transpose(0, 1)
            if cfg.k_eq_v(l):                     # no v_proj: values are the (un-normed) keys
                v = k
            else:
                v = self._lin(h, pre + "self_attn.v_proj", ctx).view(n, n_kv, hd).transpose(0, 1)
        if self.qk_norm and self.qk_norm_flat:
            q = rms_norm(q.transpose(0, 1).reshape(n, -1), self.norm_w(pre + "self_attn.q_norm.weight"), cfg.eps)
            q = q.view(n, cfg.n_heads, hd).transpose(0, 1)
            if k is not None:
                k = rms_norm(k.transpose(0, 1).reshape(n, -1), self.norm_w(pre + "self_attn.k_norm.weight"), cfg.eps)
                k = k.view(n, n_kv, hd).transpose(0, 1)
        elif self.qk_norm:
            q = rms_norm(q, self.norm_w(pre + "self_attn.q_norm.weight"), cfg.eps)
            if k is not None:
                k = rms_norm(k, self.norm_w(pre + "self_attn.k_norm.weight"), cfg.eps)
        if cfg.v_norm and v is not None:          # values get a weight-free RMSNorm
            vf = v.float()
            v = (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + cfg.eps)).to(v.dtype)
        if cfg.clip_qkv:
            q = q.clamp(-cfg.clip_qkv, cfg.clip_qkv)
            if k is not None:
                k, v = k.clamp(-cfg.clip_qkv, cfg.clip_qkv), v.clamp(-cfg.clip_qkv, cfg.clip_qkv)
        q = self._head_hook(p, "q", q, ctx, idx_of, cap)
        if k is not None:
            k = self._head_hook(p, "k", k, ctx, idx_of, cap)
            v = self._head_hook(p, "v", v, ctx, idx_of, cap)
        x = h
        bc = cache.block_caches.get(src if src is not None else p)
        if src is not None:                       # read the source block's cache as it stands (it ran earlier)
            K, V = cache.slot_kv(p)
            if K is None:
                raise ValueError(f"Block #{p} shares the KV cache of block #{src}, which has not run yet.")
            K, V = K.to(x.device), V.to(x.device)
        elif bc is not None:
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
            rows = [i for i in (ctx.store.get("kv_zero") or {}).get(p, ()) if i < cache.k[p].shape[1]]
            if rows:                              # zeroed KV records: that token's key and value are erased
                cache.k[p][:, rows] = 0
                cache.v[p][:, rows] = 0
            K = cache.k[p].to(x.device)
            V = cache.v[p].to(x.device)
        T = K.shape[1]
        pos_q, pos_k, inv_freq = self._rope_for_block(p, n, T, cache, ctx, l)
        if cap:                                   # every query (pre-RoPE, fp16) + position by cache index, for the KV tab
            hist = ctx.store.setdefault("last_q", {}).setdefault(p, {})
            qs, qp = to_host(q.detach().to(torch.float16)), to_host(pos_q.detach().float())
            for i in range(n):
                hist[T - n + i] = (qs[:, i], qp[i])         # position: a host tensor until _collect (no GPU stall)
            keep = max(2, _Q_HISTORY_BYTES // max(1, cfg.n_layers * qs[:, 0].numel() * 2))
            while len(hist) > keep:               # over budget: forget the oldest queries first
                del hist[next(iter(hist))]
        qr = apply_rope(q, pos_q, inv_freq)
        kr = apply_rope(K, pos_k, inv_freq)
        rep = cfg.n_heads // n_kv
        if rep > 1:
            kr = kr.repeat_interleave(rep, 0)
            V = V.repeat_interleave(rep, 0)
        scores = (qr.float() @ kr.float().transpose(1, 2)) * self.attn_scale(l)
        scores = scores.masked_fill(~self.attn_allowed(p, n, T, cache, x.device, ctx, l), float("-inf"))
        probs = self._probs_hook(p, torch.softmax(scores, -1), cache, ctx)
        store_attention(ctx, p, probs)
        z = (probs.to(V.dtype) @ V)
        for iv in ctx.active("attn_heads"):
            heads = parse_index_spec(iv.heads, cfg.n_heads)
            zt = z.transpose(0, 1)
            sub = zt[:, heads]
            sub = apply_op(sub, iv, ctx, 0, idx_of(iv))
            zt = zt.clone()
            zt[:, heads] = sub
            z = zt.transpose(0, 1)
        zf = z.transpose(0, 1).reshape(n, cfg.n_heads * hd)
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
        probs = self._probs_hook(p, torch.softmax(scores, -1), cache, ctx)
        store_attention(ctx, p, probs)
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
            ctx.store.setdefault("lin_mem", {}).setdefault(p, []).append(to_host(S.norm(dim=(1, 2))))
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + cfg.eps)
        o = o * W(a_ + "norm.weight").float() * F.silu(z.float())
        return o.reshape(n, vd).to(h.dtype) @ self._w(a_ + "out_proj.weight", ctx).T

    def _mlp_and_finish(self, p, pre, x, attn_out, ctx, idx_of, cap):
        cfg, W = self.cfg, self.W
        h2 = rms_norm(x, self.norm_w(pre + ("pre_feedforward_layernorm.weight" if self.sandwich
                                            else "post_attention_layernorm.weight")), cfg.eps)
        lay = self.moe_layers.get(int(pre.split(".")[2]))
        if lay is not None:
            mlp_out = self._moe_mlp(p, lay, h2, ctx, idx_of, cap)
        else:
            gate = h2 @ self._w(pre + "mlp.gate_proj.weight", ctx).T
            up = h2 @ self._w(pre + "mlp.up_proj.weight", ctx).T
            neurons = self.act(gate) * up
            neurons = self.neuron_hook(p, neurons, ctx, idx_of, cap)
            mlp_out = neurons @ self._w(pre + "mlp.down_proj.weight", ctx).T
        if self.sandwich:
            mlp_out = rms_norm(mlp_out, self.norm_w(pre + "post_feedforward_layernorm.weight"), cfg.eps)
        for iv in ctx.active("mlp_out"):
            mlp_out = apply_op(mlp_out, iv, ctx, 0, idx_of(iv))
        x = x + mlp_out
        if W.has(pre + "adapter.down.weight"):
            x = x + self.act(x @ self._w(pre + "adapter.down.weight", ctx).T) @ self._w(pre + "adapter.up.weight", ctx).T
        if W.has(pre + "layer_scalar"):          # per-layer rescaling of the whole residual stream
            x = x * W(pre + "layer_scalar").to(x.dtype)
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
            st.setdefault("router", {}).setdefault(p, []).append(to_host(logits.detach().clamp(min=-1e4).float()))
            st.setdefault("moe_sel", {}).setdefault(p, []).append(to_host(sel.detach().to(torch.int16)))
            st.setdefault("moe_w", {}).setdefault(p, []).append(to_host(w.detach().float()))
            if parts is not None:
                st.setdefault("moe_parts", {}).setdefault(p, []).append(parts)
                if shared is not None:
                    st.setdefault("moe_shared", {}).setdefault(p, []).append(to_host(shared.detach().to(torch.float16)))
        return out

    def neuron_hook(self, p, neurons, ctx, idx_of, cap):
        if cap:
            self.note_units(p, "mlp_neurons", neurons, ctx)
        for iv in ctx.active("mlp_neurons"):
            neurons = apply_op(neurons, iv, ctx, 0, idx_of(iv))
        if cap and ctx.store.get("want_neurons", False):
            ctx.store.setdefault("neurons", {}).setdefault(p, []).append(to_host(neurons.detach().to(torch.float16)))
        return neurons

    def _finish(self, p, x, attn_out, mlp_out, ctx, idx_of, cap):
        for iv in ctx.active("resid_post"):
            x = apply_op(x, iv, ctx, 0, idx_of(iv))
        zero = torch.zeros(x.shape[0])
        st = ctx.store.setdefault("stats", {}).setdefault(p, {"attn": [], "mlp": [], "resid": []})
        st["attn"].append(to_host(attn_out.detach().float().norm(dim=-1)) if attn_out is not None else zero)
        st["mlp"].append(to_host(mlp_out.detach().float().norm(dim=-1)) if mlp_out is not None else zero)
        st["resid"].append(to_host(x.detach().float().norm(dim=-1)))
        if cap:
            ctx.store.setdefault("resid_post", {}).setdefault(p, []).append(to_host(x.detach().float()))
            if ctx.store.get("want_parts", False):
                for key, t in (("attn_out", attn_out), ("mlp_out", mlp_out)):
                    if t is not None:
                        ctx.store.setdefault(key, {}).setdefault(p, []).append(to_host(t.detach().float()))
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
                ctx: RunContext, run_offset: int, idx_planner, w_out=None, media: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Process new tokens `ids` (run-relative index run_offset..). Returns logits [n, vocab] (float32).
        `media`: soft-token embeddings that replace the media placeholder tokens among `ids`, in order."""
        if not cache.shared:
            cache.shared = self.cfg.kv_sharing(schedule)
        cache.add_tokens(ids, positions, origin)
        x = self.inject_media(self.embed(ids), ids, media)
        n = len(ids)

        def idx_of(iv):
            return idx_planner(iv, run_offset, n)

        for p, l in enumerate(schedule):
            ctx.checkpoint()
            x = self.block(p, l, x, cache, ctx, idx_of)
            if ctx.on_block:
                ctx.on_block(p)
        return self.final(x, w_out)
