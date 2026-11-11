"""Editable box code: what a box computes, as a few lines of very simple Python.

Every box of a language-model flow (Embed, each Block, Final norm, LM head, Sampler) has its code shown in the
inspector, like a notebook cell. If a student edits it, the run executes the edited lines for that box instead
of the built-in engine: delete the MLP, flip the sign of attention, write your own sampler.

The code is a plain script over a small set of names (below). Tensors are torch tensors, so `+ - * / @`,
`.T`, `.mean()`, slicing … all work. It is checked before it runs: no imports, no names starting with `_`,
no file access — it is meant for arithmetic on activations, nothing else. This is a guard against accidents
and against flows received from other people, not a security sandbox.

Names available in every box
    softmax, silu, gelu, relu, sigmoid, tanh, exp, log, sqrt, abs, argmax, zeros_like, ones_like, randn_like,
    cat, stack, where, clip, print, len, range, min, max, sum, float, int, round, enumerate, zip

Block      in:  x  [tokens, d_model]                     out: x   (optional: attn, attn_out, neurons, mlp_out)
           W(name)            a weight of this block, e.g. W("self_attn.q_proj.weight")
           rms_norm(x, w)     divide by the root-mean-square, multiply by w
           heads(t)           [tokens, heads·head_dim] → [heads, tokens, head_dim]
           unheads(t)         the reverse
           remember(k, v)     add this chunk's keys/values to the KV cache; returns (K, V, positions) of all tokens
           rope(t, positions) rotate by position (RoPE)
           mask(scores)       0 where a token may look, -inf where it may not (later tokens, hidden prefixes)
           n_heads, n_kv_heads, head_dim, d_model, layer, position
Embed      in:  ids (list of token ids)                   out: x
           embedding(ids)
Final norm in:  x                                         out: x          W(name), rms_norm
LM head    in:  x                                         out: logits     unembed(x)
Sampler    in:  logits [vocab], temperature, top_k, top_p out: token      keep_top_k, keep_top_p, sample
"""
from __future__ import annotations

import ast
import math
import traceback

import torch
import torch.nn.functional as F

KINDS = ("embed", "block", "final_norm", "lm_head", "sampler")
RESULT = {"embed": "x", "block": "x", "final_norm": "x", "lm_head": "logits", "sampler": "token"}
MAX_PRINTS = 200

_BANNED_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                 ast.AsyncFor, ast.AsyncWith, ast.Await, ast.Yield, ast.YieldFrom, ast.With, ast.Try, ast.Raise,
                 ast.While, ast.Delete)
_BANNED_ATTRS = {"numpy", "tofile", "dump", "dumps", "save", "storage", "untyped_storage", "data_ptr", "share_memory_",
                 "pin_memory", "register_hook", "backward", "requires_grad_", "to_dlpack", "format", "format_map"}
_BANNED_NAMES = {"eval", "exec", "compile", "open", "input", "getattr", "setattr", "delattr", "globals", "locals",
                 "vars", "dir", "type", "object", "super", "breakpoint", "help", "exit", "quit", "memoryview"}


def check(source: str) -> list[str]:
    """Problems that stop this code from being accepted (empty list = fine)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"line {e.lineno}: {e.msg}"]
    out = []
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", "?")
        if isinstance(node, _BANNED_NODES):
            what = type(node).__name__
            hint = " (use `for … in range(…)`)" if what == "While" else ""
            out.append(f"line {ln}: '{what}' is not available in box code{hint}.")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in _BANNED_ATTRS):
            out.append(f"line {ln}: '.{node.attr}' is not available in box code.")
        elif isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in _BANNED_NAMES):
            out.append(f"line {ln}: '{node.id}' is not available in box code.")
    return out


class Code:
    def __init__(self, source: str, kind: str, label: str = ""):
        self.source, self.kind, self.label = source, kind, label or kind
        self.filename = f"<{self.label} code>"
        problems = check(source)
        if problems:
            raise ValueError(f"{self.label} code: " + " · ".join(problems[:4]))
        self.compiled = compile(source, self.filename, "exec")


def compile_code(source: str, kind: str, label: str = "") -> Code:
    if kind not in KINDS:
        raise ValueError(f"Box kind '{kind}' has no editable code.")
    return Code(source, kind, label)


# --------------------------------------------------------------------------------------
# The small vocabulary box code is written in
# --------------------------------------------------------------------------------------

def _softmax(t):
    return torch.softmax(t.float(), -1).to(t.dtype)


def _sqrt(v):
    return torch.sqrt(v) if isinstance(v, torch.Tensor) else math.sqrt(v)


def _abs(v):
    return v.abs() if isinstance(v, torch.Tensor) else abs(v)


def _argmax(t):
    return int(t.argmax())


_MATH = {
    "softmax": _softmax, "silu": F.silu, "gelu": F.gelu, "relu": F.relu, "sigmoid": torch.sigmoid,
    "tanh": torch.tanh, "exp": torch.exp, "log": torch.log, "sqrt": _sqrt, "abs": _abs, "argmax": _argmax,
    "zeros_like": torch.zeros_like, "ones_like": torch.ones_like, "randn_like": torch.randn_like,
    "cat": torch.cat, "stack": torch.stack, "where": torch.where, "clip": torch.clamp,
}
_BUILTINS = {"len": len, "range": range, "min": min, "max": max, "sum": sum, "float": float, "int": int,
             "round": round, "enumerate": enumerate, "zip": zip, "list": list, "tuple": tuple, "dict": dict,
             "bool": bool, "str": str, "sorted": sorted, "reversed": reversed, "isinstance": isinstance,
             "True": True, "False": False, "None": None}


def _exec(code: Code, names: dict, ctx) -> dict:
    prints = ctx.store.setdefault("prints", []) if ctx is not None else []

    def _print(*a):
        if len(prints) < MAX_PRINTS:
            prints.append(f"[{code.label}] " + " ".join(_show(v) for v in a))

    g = {"__builtins__": {**_BUILTINS, "print": _print}, **_MATH, **names}
    try:
        exec(code.compiled, g)       # noqa: S102 — checked by check(); this is the feature
    except Exception as e:
        ln = next((f.lineno for f in reversed(traceback.extract_tb(e.__traceback__)) if f.filename == code.filename), None)
        line = code.source.splitlines()[ln - 1].strip() if ln and ln <= len(code.source.splitlines()) else ""
        raise ValueError(f"{code.label} code, line {ln}: {type(e).__name__}: {e}" + (f"\n    {line}" if line else "")) from None
    if RESULT[code.kind] not in g:
        raise ValueError(f"{code.label} code must leave its result in `{RESULT[code.kind]}`.")
    return g


def _show(v) -> str:
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return f"{float(v):.4g}"
        return f"tensor{list(v.shape)} mean={float(v.float().mean()):.4g} min={float(v.float().min()):.4g} max={float(v.float().max()):.4g}"
    return str(v)


def _tensor(g: dict, name: str, shape_tail=None):
    t = g.get(name)
    if not isinstance(t, torch.Tensor):
        return None
    if shape_tail is not None and tuple(t.shape[-len(shape_tail):]) != tuple(shape_tail):
        return None
    return t


# --------------------------------------------------------------------------------------
# Runners (called by the engine in place of its built-in step)
# --------------------------------------------------------------------------------------

def run_block(model, code: Code, p: int, l: int, x: torch.Tensor, cache, ctx):
    """Run a block's edited code. Returns (x, attn_out or None, mlp_out or None)."""
    from .llm import apply_rope, rms_norm
    cfg = model.cfg
    pre = f"model.layers.{l}."
    n, d = x.shape

    def W(name):
        return model.norm_w(pre + name) if "norm" in name else model.W(pre + name)

    def remember(k, v):
        cache.append(p, k, v)
        K, V = cache.k[p].to(x.device), cache.v[p].to(x.device)
        rep = cfg.n_heads // K.shape[0] if K.shape[0] else 1
        if rep > 1:
            K, V = K.repeat_interleave(rep, 0), V.repeat_interleave(rep, 0)
        return K, V, cache.pos.to(model.inv_freq.device)

    def mask(scores):
        ok = model.attn_allowed(p, scores.shape[-2], scores.shape[-1], cache, scores.device)
        return torch.zeros(ok.shape, dtype=scores.dtype, device=scores.device).masked_fill(~ok, float("-inf"))

    g = _exec(code, {
        "x": x, "W": W, "rms_norm": lambda t, w: rms_norm(t, w, cfg.eps),
        "heads": lambda t: t.view(t.shape[0], -1, cfg.head_dim).transpose(0, 1),
        "unheads": lambda t: t.transpose(0, 1).reshape(t.shape[1], -1),
        "remember": remember, "rope": lambda t, pos: apply_rope(t, pos, model.inv_freq), "mask": mask,
        "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv, "head_dim": cfg.head_dim, "d_model": cfg.hidden,
        "layer": l, "position": p,
    }, ctx)
    out = g["x"]
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != (n, d):
        got = list(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
        raise ValueError(f"{code.label} code must leave x as a [{n}, {d}] tensor (tokens × d_model); got {got}.")
    if ctx.capture:
        attn = _tensor(g, "attn")
        if attn is not None and attn.ndim == 3 and attn.shape[1] == n and ctx.store.get("want_attn", True):
            ctx.store.setdefault("attn", {}).setdefault(p, []).append(attn.detach().to(torch.float16).cpu())
        neurons = _tensor(g, "neurons", (n, cfg.inter))
        if neurons is not None:
            model.note_units(p, "mlp_neurons", neurons, ctx)
            if ctx.store.get("want_neurons", False):
                ctx.store.setdefault("neurons", {}).setdefault(p, []).append(neurons.detach().to(torch.float16).cpu())
    return out.to(x.dtype), _tensor(g, "attn_out", (n, d)), _tensor(g, "mlp_out", (n, d))


def run_embed(model, code: Code, ids: list[int], ctx) -> torch.Tensor:
    g = _exec(code, {"ids": list(ids), "embedding": model.embed, "d_model": model.cfg.hidden}, ctx)
    x = g["x"]
    if not isinstance(x, torch.Tensor) or tuple(x.shape) != (len(ids), model.cfg.hidden):
        raise ValueError(f"{code.label} code must leave x as a [{len(ids)}, {model.cfg.hidden}] tensor.")
    return x.to(model.dtype)


def run_final_norm(model, code: Code, x: torch.Tensor, ctx) -> torch.Tensor:
    from .llm import rms_norm
    g = _exec(code, {"x": x, "W": model.norm_w, "rms_norm": lambda t, w: rms_norm(t, w, model.cfg.eps),
                     "d_model": model.cfg.hidden}, ctx)
    out = g["x"]
    if not isinstance(out, torch.Tensor) or out.shape != x.shape:
        raise ValueError(f"{code.label} code must leave x with the shape it came in with {list(x.shape)}.")
    return out.to(x.dtype)


def run_lm_head(model, code: Code, x: torch.Tensor, w_out, ctx) -> torch.Tensor:
    g = _exec(code, {"x": x, "unembed": lambda h: model.unembed(h, w_out), "vocab": model.cfg.vocab}, ctx)
    out = g["logits"]
    if not isinstance(out, torch.Tensor) or out.ndim != 2 or out.shape[0] != x.shape[0]:
        raise ValueError(f"{code.label} code must leave logits as a [{x.shape[0]}, vocab] tensor.")
    return out.float()


def keep_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Zero every probability except the k largest (k = 0: keep all), then renormalise."""
    if not k or k >= probs.shape[-1]:
        return probs
    kth = probs.topk(int(k)).values[-1]
    kept = torch.where(probs >= kth, probs, torch.zeros_like(probs))
    return kept / kept.sum()


def keep_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the most likely tokens until their probabilities add up to p (p = 1: keep all), then renormalise."""
    if p is None or p >= 1:
        return probs
    sp, si = probs.sort(descending=True)
    keep = (sp.cumsum(-1) - sp) < p * sp.sum()
    kept = torch.zeros_like(probs)
    kept[si[keep]] = sp[keep]
    return kept / kept.sum()


def run_sampler(code: Code, logits: torch.Tensor, params: dict, gen: torch.Generator, ctx) -> int:
    def sample(probs):
        pr = probs.float().clamp(min=0).cpu()
        return int(torch.multinomial(pr / pr.sum(), 1, generator=gen).item())

    g = _exec(code, {"logits": logits.float(), "temperature": float(params.get("temperature", 0.0)),
                     "top_k": int(params.get("top_k", 0)), "top_p": float(params.get("top_p", 1.0)),
                     "keep_top_k": keep_top_k, "keep_top_p": keep_top_p, "sample": sample}, ctx)
    try:
        tok = int(g["token"])
    except (TypeError, ValueError):
        raise ValueError(f"{code.label} code must leave token as one token id (a whole number).") from None
    if not 0 <= tok < logits.shape[-1]:
        raise ValueError(f"{code.label} code chose token {tok}, outside the vocabulary (0…{logits.shape[-1] - 1}).")
    return tok


# --------------------------------------------------------------------------------------
# The code each box runs when nobody has edited it (same maths as llm.py, written out simply)
# --------------------------------------------------------------------------------------

def default_code(kind: str, model=None, layer: int = 0) -> dict:
    """{"code": source or None, "editable": bool, "reason": why not}. `model` is a RawLLM."""
    if kind == "embed":
        return _ok("# ids: the ids of the new tokens. Look up one vector per token —\n"
                   "# that vector is the start of the residual stream.\n"
                   "x = embedding(ids)\n")
    if kind == "final_norm":
        return _ok("# x: the residual stream after the last block  [tokens, d_model]\n"
                   'x = rms_norm(x, W("model.norm.weight"))\n')
    if kind == "lm_head":
        return _ok("# x: the normalised residual stream. Compare it with every word's output vector:\n"
                   "# one score (logit) per word in the vocabulary.\n"
                   "logits = unembed(x)\n")
    if kind == "sampler":
        return _ok("# logits: the scores of every vocabulary word for the next token  [vocab]\n"
                   "if temperature == 0:\n"
                   "    token = argmax(logits)                 # greedy: always the most likely word\n"
                   "else:\n"
                   "    probs = softmax(logits / temperature)  # higher temperature = flatter = more surprising\n"
                   "    probs = keep_top_k(probs, top_k)       # only the k most likely words (0 = all)\n"
                   "    probs = keep_top_p(probs, top_p)       # only the words that make up the top p of the mass\n"
                   "    token = sample(probs)                  # roll the dice\n")
    if kind != "block":
        return {"code": None, "editable": False, "reason": "This box has no editable code."}
    if model is None:
        return {"code": None, "editable": False, "reason": "Load a model first."}
    cfg = model.cfg
    if cfg.is_linear(layer):
        return {"code": None, "editable": False,
                "reason": "This is a linear-attention (Gated DeltaNet) block; its code view is not written yet."}
    if model.attn_gate:
        return {"code": None, "editable": False,
                "reason": "This model's gated attention blocks have no code view yet."}
    pre = f"model.layers.{layer}."

    def lin(src, name):
        bias = f' + W("{name}.bias")' if model.has_bias and model.W.has(pre + name + ".bias") else ""
        return f'{src} @ W("{name}.weight").T{bias}'

    act = {"silu": "silu", "gelu": "gelu", "relu": "relu"}.get(cfg.act, "silu")
    qk = ('q = rms_norm(q, W("self_attn.q_norm.weight"))            # this model normalises q and k per head\n'
          'k = rms_norm(k, W("self_attn.k_norm.weight"))\n') if model.qk_norm else ""
    return _ok(
        f"# Block {layer}.  x is the residual stream coming in: one row per new token  [tokens, d_model]\n"
        f'h = rms_norm(x, W("input_layernorm.weight"))\n'
        f"\n"
        f"# --- attention: each token asks a question (q); every token offers a key (k) and a value (v)\n"
        f"q = heads({lin('h', 'self_attn.q_proj')})\n"
        f"k = heads({lin('h', 'self_attn.k_proj')})\n"
        f"v = heads({lin('h', 'self_attn.v_proj')})\n"
        f"{qk}"
        f"K, V, positions = remember(k, v)        # the KV cache: keys and values of every token so far\n"
        f"scores = rope(q, positions[-len(x):]) @ rope(K, positions).transpose(1, 2) / sqrt(head_dim)\n"
        f"attn = softmax(scores + mask(scores))   # mask: a token only looks at itself and earlier tokens\n"
        f"attn_out = {lin('unheads(attn @ V)', 'self_attn.o_proj')}\n"
        f"x = x + attn_out\n"
        f"\n"
        f"# --- MLP: each token thinks on its own\n"
        f'h = rms_norm(x, W("post_attention_layernorm.weight"))\n'
        f"neurons = {act}({lin('h', 'mlp.gate_proj')}) * ({lin('h', 'mlp.up_proj')})\n"
        f"mlp_out = {lin('neurons', 'mlp.down_proj')}\n"
        f"x = x + mlp_out\n"
        + (f"\n# --- adapter: a thin layer added by training (squeeze to a few numbers, expand, add)\n"
           f'x = x + {act}(x @ W("adapter.down.weight").T) @ W("adapter.up.weight").T\n'
           if model.W.has(pre + "adapter.down.weight") else ""))


def _ok(code: str) -> dict:
    return {"code": code, "editable": True, "reason": ""}
