"""Flow graphs <-> execution plans.

A *flow* is what the web UI draws: boxes (nodes) joined by arrows (edges).
A *plan* is what the engine executes: a layer schedule plus typed interventions.

The flow is the user-editable view of the model's canonical graph:

    Causal LM:  prompt -> tokenizer -> embed -> block* -> final_norm -> lm_head -> sampler -> output
    T2I UNet:   prompt -> tokenizer -> text_encoder ┐
                noise ─────────────────────────────> denoiser_in -> unet blocks -> denoiser_out -> vae_decode -> image
                scheduler ─────────────────────────┘          (loop edge: denoiser_out -> denoiser_in, x steps)

Editing the chain of block boxes (deleting, duplicating, re-wiring arrows) edits the layer schedule.
Interceptor boxes wired into a block's top "mod" handle become interceptors at that schedule position.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------------------
# Plan data model (serialisable — stored in every run manifest)
# --------------------------------------------------------------------------------------

LLM_HOOKS = ["resid_pre", "q", "k", "v", "kv_k", "kv_v", "attn_heads", "attn_out", "mlp_neurons", "mlp_out", "resid_post", "weights"]
UNET_HOOKS = ["out", "skip", "both", "word"]
DIT_HOOKS = ["out"]
OPS = ["scale", "zero", "add_noise", "mean", "quantize"]
# Language-model ops that need something from outside the run (or project along a vector):
#   patch       — copy this activation from another run (activation patching; `ref` = that run's id)
#   steer       — add `value` × a saved direction to the activation (`ref` = steering-vector id)
#   project_out — remove (value=1) or scale the component along a saved direction (`ref` = vector id)
#   clamp       — pin chosen neurons (`neurons`) to their min / max / a fixed value (`clamp`)
#   quantize    — round this tensor / block's weights / KV cache to `value` bits (1..16)
LLM_OPS = OPS + ["patch", "steer", "project_out", "clamp"]
PATCH_HOOKS = ["resid_pre", "q", "k", "v", "kv_k", "kv_v", "attn_out", "mlp_out", "resid_post"]
CODE_KINDS = ["embed", "block", "final_norm", "lm_head", "sampler"]     # boxes whose Python can be edited
PREFIX_MODES = ["shift", "negative", "offset", "overwrite"]


@dataclass
class Intervention:
    node_id: str
    pos: int                 # schedule position (index into plan.schedule)
    hook: str
    op: str
    value: float = 1.0
    positions: str = "all"   # token positions (LLM) — all | last | prompt | prefill | generated | a:b | 1,2,5
    steps: str = "all"       # step range (diffusion) — all | a:b | 3
    seed: int = 0
    heads: str = "all"       # attention heads for hook attn_heads — all | a:b | 1,4
    region: str = ""         # image models: painted mask (PNG data URL) — the intervention only acts there
    ref: str = ""            # patch: source run id · steer: steering-vector id · clamp: run whose min/max to use
    src_positions: str = ""  # patch: token(s) to copy from in the source run ("" = the same token index)
    neurons: str = "all"     # which units of the activation's last axis the op touches — all | 12 | 3,7 | 0:64
    clamp: str = "max"       # clamp: max | min | value


@dataclass
class Plan:
    family: str                          # causal_lm | t2i
    schedule: list[int]                  # block indices in execution order (repeats / removals allowed)
    schedule_nodes: list[str]            # flow node id at each schedule position
    interventions: list[Intervention] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)   # prompt, sampler, scheduler, noise ... settings

    def to_dict(self) -> dict:
        return asdict(self)


class FlowError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


# --------------------------------------------------------------------------------------
# Default flows
# --------------------------------------------------------------------------------------

def _node(nid, kind, label, x, y, sub="", **data):
    return {
        "id": nid,
        "type": "box",
        "position": {"x": x, "y": y},
        "data": {"kind": kind, "label": label, "sub": sub, **data},
    }


def _edge(src, dst, sh="out", th="in", **data):
    eid = f"e-{src}-{sh}-{dst}-{th}"
    e = {"id": eid, "source": src, "target": dst, "sourceHandle": sh, "targetHandle": th}
    if data:
        e["data"] = data
    return e


def _snake(items, x0, y0, per_row=6, dx=200, dy=120):
    """Lay a long chain out in wrapped rows (left to right), so 28 blocks fit on screen."""
    out = []
    for i, it in enumerate(items):
        r, c = divmod(i, per_row)
        out.append((it, x0 + c * dx, y0 + r * dy))
    return out


def default_llm_flow(info: dict) -> dict:
    n = info["n_layers"]
    nodes = [
        _node("prompt", "input_prompt", "Prompt", 0, 0, "system · user · assistant prefill",
              system=info.get("default_system", ""), prompt="The capital of France is",
              prefill="", use_chat_template=info.get("has_chat_template", False)),
        _node("tokenizer", "tokenizer", "Tokenizer", 220, 0, info.get("tokenizer", "")),
        _node("embed", "embed", "Embed", 440, 0, f"vocab {info.get('vocab', '?')} → d {info.get('d_model', '?')}"),
    ]
    # Blocks added by training (adapt.py) have the highest layer numbers but run where they were inserted.
    extra = {int(e["layer"]): e for e in info.get("extra_blocks") or []}
    order = [i for i in range(n) if i not in extra]
    for l, e in sorted(extra.items()):
        order.insert(order.index(int(e["after"])) + 1 if int(e["after"]) in order else len(order), l)
    blocks = [f"block_{i}" for i in order]
    for nid, x, y in _snake(blocks, 0, 210, dy=140):
        i = int(nid.split("_")[1])
        subs = info.get("block_subs") or []
        nodes.append(_node(nid, "block", f"Block {i} · new" if i in extra else f"Block {i}", x, y,
                           (f"added after block {extra[i]['after']} · " if i in extra else "")
                           + (subs[i] if i < len(subs) else info.get("block_sub", "")), layer=i))
    rows = (n + 5) // 6
    yb = 210 + rows * 140 + 10
    nodes += [
        _node("final_norm", "final_norm", "Final norm", 0, yb, info.get("norm", "RMSNorm")),
        _node("lm_head", "lm_head", "LM head", 220, yb, "unembed → logits"),
        _node("sampler", "sampler", "Sampler", 440, yb, "greedy / temperature",
              max_new_tokens=24, temperature=0.0, top_k=0, seed=0),
        _node("output", "output_text", "Output text", 660, yb, ""),
    ]
    chain = ["prompt", "tokenizer", "embed", *blocks, "final_norm", "lm_head", "sampler", "output"]
    edges = [_edge(a, b) for a, b in zip(chain, chain[1:])]
    edges.append(_edge("sampler", "embed", "loop", "loop", loop=True, label="next token"))
    return {"nodes": nodes, "edges": edges}


def default_t2i_flow(info: dict) -> dict:
    kind = info["denoiser"]  # unet | dit
    nodes = [
        _node("prompt", "input_prompt", "Prompt", 0, 0, "positive · negative",
              prompt="a red fox sitting in a snowy forest, photo", negative_prompt=""),
        _node("tokenizer", "tokenizer", "Tokenizer", 220, 0, info.get("tokenizer", "")),
        _node("text_encoder", "text_encoder", "Text encoder", 440, 0, info.get("text_encoder", "")),
        _node("noise", "noise", "Noise latent", 0, 140, "seed · size",
              seed=0, width=info.get("default_size", 512), height=info.get("default_size", 512)),
        _node("scheduler", "scheduler", "Scheduler", 220, 140, info.get("scheduler", ""),
              steps=info.get("default_steps", 20), guidance_scale=info.get("default_guidance", 7.0)),
        _node("denoiser_in", "denoiser_in", "Denoiser in", 440, 140, f"{kind.upper()} · step loop"),
    ]
    blocks = info["blocks"]  # [{"id","label","index","sub"}]
    for b, x, y in _snake(blocks, 0, 360, dy=150):
        nodes.append(_node(b["id"], "unet_block", b["label"], x, y,
                           b.get("sub", ""), layer=b["index"]))
    rows = (len(blocks) + 5) // 6
    yb = 360 + rows * 150 + 20
    nodes += [
        _node("denoiser_out", "denoiser_out", "Denoiser out", 0, yb, "noise / velocity prediction → scheduler step"),
        _node("vae_decode", "vae_decode", "VAE decode", 220, yb, info.get("vae", "")),
        _node("output", "output_image", "Image", 440, yb, ""),
    ]
    edges = [
        _edge("prompt", "tokenizer"), _edge("tokenizer", "text_encoder"),
        _edge("text_encoder", "denoiser_in", "out", "cond"),
        _edge("noise", "denoiser_in", "out", "in"),
        _edge("scheduler", "denoiser_in", "out", "sched"),
    ]
    chain = ["denoiser_in", *[b["id"] for b in blocks], "denoiser_out", "vae_decode", "output"]
    edges += [_edge(a, b) for a, b in zip(chain, chain[1:])]
    edges.append(_edge("denoiser_out", "denoiser_in", "loop", "loop", loop=True, label="× steps"))
    if True:
        downs = [b for b in blocks if b["id"].startswith("down")]
        ups = [b for b in blocks if b["id"].startswith("up")]
        for d, u in zip(downs, reversed(ups)):
            edges.append(_edge(d["id"], u["id"], "skip", "skip", skip=True, label="skip"))
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------------------
# Compile flow -> plan
# --------------------------------------------------------------------------------------

def _num(v, default, cast=float):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


def _llm_params(nodes: dict, edges: list, find_kind, chain: list, caps: dict) -> dict:
    """Run parameters of a language-model flow: prompt, sampler, KV prefixes, edited box code."""
    params: dict[str, Any] = {}
    p = nodes[find_kind("input_prompt")[0]]["data"]
    s = nodes[find_kind("sampler")[0]]["data"]
    params.update(
        system=p.get("system", ""), prompt=p.get("prompt", ""), prefill=p.get("prefill", ""),
        use_chat_template=bool(p.get("use_chat_template", False)), thinking=bool(p.get("thinking", False)),
        max_new_tokens=max(0, min(512, _num(s.get("max_new_tokens"), 24, int))),
        temperature=max(0.0, _num(s.get("temperature"), 0.0)), top_k=max(0, _num(s.get("top_k"), 0, int)),
        top_p=min(1.0, max(0.0, _num(s.get("top_p"), 1.0))) or 1.0,
        seed=_num(s.get("seed"), 0, int), add_bos=bool(p.get("add_bos", True)),
    )
    # KV prefixes: a saved cache (kv_import), or text typed straight into a box (kv_text, a "layer prompt").
    kv = [nid for nid in find_kind("kv_import") + find_kind("kv_text")
          if any(e["source"] == nid and e["target"] in find_kind("embed") and e.get("targetHandle") == "cache"
                 for e in edges)]
    kv.sort(key=lambda nid: (nodes[nid]["position"]["x"], nodes[nid]["position"]["y"]))   # left box first
    prefixes = []
    for nid in kv:
        d = nodes[nid]["data"]
        if d.get("enabled") is False:
            continue
        mode = d.get("mode") or "shift"
        if mode not in PREFIX_MODES:
            raise FlowError([{"message": f"Position mode must be one of {', '.join(PREFIX_MODES)}.", "node": nid}])
        try:
            parse_index_spec(str(d.get("layers") or "all"), 10_000)
        except ValueError as ex:
            raise FlowError([{"message": str(ex), "node": nid}])
        pre = {"cache_id": "", "mode": mode, "node": nid, "layers": str(d.get("layers") or "all"),
               "offset": _num(d.get("offset"), 0.0), "label": d.get("label") or "imported"}
        if d.get("kind") == "kv_text":
            if not str(d.get("text") or "").strip():
                raise FlowError([{"message": "A layer-prompt box has no text.", "node": nid}])
            pre.update(text=str(d["text"]), as_system=bool(d.get("as_system", False)))
        elif not d.get("cache_id"):
            raise FlowError([{"message": "A KV-cache box has no cache selected.", "node": nid}])
        else:
            pre["cache_id"] = d["cache_id"]
        prefixes.append(pre)
    if prefixes:
        params.update(prefixes=prefixes, prefix_cache=prefixes[0]["cache_id"] or "text")
    # Edited Python for boxes (see blockcode.py): {box id: source}. Keyed by box, so a repeated block can run
    # different code from the original.
    code = {}
    for nid in chain + [n for k in CODE_KINDS if k != "block" for n in find_kind(k)]:
        if str(nodes[nid]["data"].get("code") or "").strip():
            code[nid] = {"kind": nodes[nid]["data"]["kind"], "source": str(nodes[nid]["data"]["code"])}
    if code:
        params["code"] = code
    return params


def _t2i_params(nodes: dict, edges: list, find_kind, chain: list, caps: dict) -> dict:
    """Run parameters of an image flow: prompts, noise, scheduler, per-block prompts, start image."""
    params: dict[str, Any] = {}
    p = nodes[find_kind("input_prompt")[0]]["data"]
    nz = nodes[find_kind("noise")[0]]["data"]
    sc = nodes[find_kind("scheduler")[0]]["data"]
    params.update(
        prompt=p.get("prompt", ""), negative_prompt=p.get("negative_prompt", ""),
        switch_prompt=p.get("switch_prompt", ""), switch_step=_num(p.get("switch_step"), 0, int),
        seed=_num(nz.get("seed"), 0, int),
        width=_round8(_num(nz.get("width"), 512, int)), height=_round8(_num(nz.get("height"), 512, int)),
        steps=max(1, min(200, _num(sc.get("steps"), 20, int))),
        guidance_scale=max(0.0, _num(sc.get("guidance_scale"), 7.0)),
    )
    kvs = [nid for nid in find_kind("prompt_kv")
           if any(e["source"] == nid and e["target"] in find_kind("denoiser_in") for e in edges)]
    kvs.sort(key=lambda nid: (nodes[nid]["position"]["x"], nodes[nid]["position"]["y"]))
    pk = []
    for nid in kvs:
        d = nodes[nid]["data"]
        if d.get("enabled") is False:
            continue
        if not caps.get("prompt_kv"):
            raise FlowError([{"message": "This image model doesn't support per-block prompts.", "node": nid}])
        role = d.get("role", "replace")
        if role not in ("replace", "prepend"):
            raise FlowError([{"message": "Role must be replace or prepend.", "node": nid}])
        try:
            parse_index_spec(str(d.get("blocks") or "all"), 10_000)
        except ValueError as ex:
            raise FlowError([{"message": str(ex), "node": nid}])
        pk.append({"node": nid, "prompt": d.get("prompt", ""), "role": role, "blocks": str(d.get("blocks") or "all"),
                   "mode": d.get("mode", "shift"), "offset": _num(d.get("offset"), 0, int),
                   "label": d.get("label") or ("hidden prompt" if role == "prepend" else "other prompt")})
    if pk:
        params["prompt_kvs"] = pk
    init = [nid for nid in find_kind("init_image")
            if any(e["source"] == nid and e["target"] in find_kind("noise") for e in edges)]
    if init:
        d = nodes[init[0]]["data"]
        if d.get("image"):
            if not caps.get("img2img"):
                raise FlowError([{"message": "This model adapter doesn't support start-image prefill.",
                                  "node": init[0]}])
            params["init_image"] = d["image"]
            params["strength"] = min(1.0, max(0.0, _num(d.get("strength"), 0.6)))
            if d.get("guide") and caps.get("sketch_guide"):
                params["guide"] = True
                params["lock_strength"] = min(1.0, max(0.0, _num(d.get("lock_strength"), 1.0)))
            if d.get("mask"):
                params.update(mask=d["mask"], mask_mode=d.get("mask_mode", "keep"),
                              lock_steps=str(d.get("lock_steps") or "all"),
                              lock_strength=min(1.0, max(0.0, _num(d.get("lock_strength"), 1.0))))
    return params


@dataclass
class FlowShape:
    """One kind of flow, as data: where the chain of blocks starts and ends, which boxes must exist, which hooks
    and operations its interventions accept, and how its configuration boxes become run parameters. A model's
    info names its shape (info["flow_shape"], default: its family), so a new kind of model adds a shape here
    instead of conditions in compile_flow."""
    start: str
    end: str
    block: str
    required: list
    hooks: list
    ops: list
    params: Any                    # (nodes, edges, find_kind, chain, capabilities) -> dict
    regions: bool = False          # interventions can be limited to a painted region
    steps: bool = False            # interventions can be limited to some denoising steps


FLOW_SHAPES: dict[str, FlowShape] = {
    "causal_lm": FlowShape("embed", "final_norm", "block", ["input_prompt", "embed", "final_norm", "lm_head", "sampler"],
                           LLM_HOOKS, LLM_OPS, _llm_params),
    "t2i": FlowShape("denoiser_in", "denoiser_out", "unet_block",
                     ["input_prompt", "noise", "scheduler", "denoiser_in", "denoiser_out", "vae_decode"],
                     UNET_HOOKS, OPS, _t2i_params, regions=True, steps=True),
}


def flow_shape(info: dict) -> FlowShape:
    return FLOW_SHAPES[info.get("flow_shape") or info["family"]]


def compile_flow(flow: dict, info: dict) -> Plan:
    family = info["family"]
    nodes = {n["id"]: n for n in flow.get("nodes", [])}
    edges = flow.get("edges", [])
    errors: list[dict] = []

    def kind(nid):
        return nodes[nid]["data"].get("kind") if nid in nodes else None

    def find_kind(k):
        return [nid for nid, n in nodes.items() if n["data"].get("kind") == k]

    def err(msg, node=None):
        errors.append({"message": msg, "node": node})

    shape = flow_shape(info)
    caps = info.get("capabilities", {})
    start_k, end_k, block_k = shape.start, shape.end, shape.block

    for k in shape.required:
        if not find_kind(k):
            err(f"Flow needs a '{k}' box.")
    if errors:
        raise FlowError(errors)

    main_edges = [e for e in edges
                  if not (e.get("data") or {}).get("loop") and not (e.get("data") or {}).get("skip")
                  and (e.get("sourceHandle") in (None, "out")) and (e.get("targetHandle") in (None, "in"))
                  and e["source"] in nodes and e["target"] in nodes]
    out_by_src: dict[str, list[str]] = {}
    for e in main_edges:
        out_by_src.setdefault(e["source"], []).append(e["target"])

    # Walk the residual stream from the start box to the end box.
    start = find_kind(start_k)[0]
    cur, chain, seen = start, [], set()
    max_len = 8 * info["n_layers"] + 8
    while True:
        nxt = [t for t in out_by_src.get(cur, [])]
        if len(nxt) == 0:
            err(f"The residual stream stops at '{nodes[cur]['data']['label']}' — connect it onward to "
                f"'{end_k}'.", cur)
            break
        if len(nxt) > 1:
            err(f"'{nodes[cur]['data']['label']}' has {len(nxt)} outgoing arrows. The residual stream is a "
                f"single path: keep one.", cur)
            break
        cur = nxt[0]
        if kind(cur) == end_k:
            break
        if kind(cur) != block_k:
            err(f"Only block boxes can sit between '{start_k}' and '{end_k}' (found '{kind(cur)}').", cur)
            break
        if cur in seen:
            err("The block chain loops back on itself. Use a duplicated block box to repeat a layer.", cur)
            break
        seen.add(cur)
        chain.append(cur)
        if len(chain) > max_len:
            err("Block chain is too long.", cur)
            break

    walked = chain
    bypassed = {c for c in chain if nodes[c]["data"].get("bypass")}
    if bypassed and not caps.get("remove_blocks", True):
        err("This model's blocks can't be bypassed (its skip connections need every stage). Use a scale/zero "
            "intervention.")
    chain = [c for c in chain if c not in bypassed]
    schedule = [int(nodes[c]["data"]["layer"]) for c in chain]
    for c, l in zip(chain, schedule):
        if not 0 <= l < info["n_layers"]:
            err(f"Block index {l} out of range 0..{info['n_layers'] - 1}.", c)

    if not caps.get("reorder_blocks", True) and schedule != list(range(info["n_layers"])):
        err("This model's blocks are tied together by skip connections, so they can't be removed, repeated or "
            "re-ordered. Use intervention boxes (scale / zero) instead. 'Reset flow' restores the chain.")
    if not chain:
        err(f"No active blocks between '{start_k}' and '{end_k}'" + (" (all bypassed)." if walked else "."))

    # Interceptors: interceptor boxes wired into a block's "mod" handle.
    pos_of_node = {nid: i for i, nid in enumerate(chain)}
    ivs: list[Intervention] = []
    valid_hooks = info.get("hooks") or shape.hooks
    for e in edges:
        if e.get("targetHandle") != "mod" or e["source"] not in nodes or e["target"] not in nodes:
            continue
        src = nodes[e["source"]]["data"]
        if src.get("kind") != "intervention":
            err("Only interceptor boxes can connect to a block's top handle.", e["source"])
            continue
        if src.get("enabled") is False:
            continue
        if e["target"] in bypassed:
            continue
        if e["target"] not in pos_of_node:
            err(f"Interceptor '{src.get('label')}' points at a block that isn't on the active path.", e["source"])
            continue
        hook = src.get("hook") or valid_hooks[0]
        if hook not in valid_hooks:
            err(f"Hook '{hook}' isn't available here; choose one of {', '.join(valid_hooks)}.", e["source"])
            continue
        op = src.get("op", "scale")
        if op not in shape.ops:
            err(f"Unknown op '{op}'.", e["source"])
            continue
        if hook == "weights" and op not in ("quantize", "scale", "zero", "add_noise"):
            err("Hook 'weights' supports quantize, scale, zero or add_noise.", e["source"])
            continue
        if op == "quantize" and not 1 <= _num(src.get("value"), 4, int) <= 16:
            err("Quantize bits (value) must be between 1 and 16.", e["source"])
            continue
        if op == "patch" and hook not in PATCH_HOOKS:
            err(f"Patching copies a whole activation: use one of {', '.join(PATCH_HOOKS)}.", e["source"])
            continue
        if op == "patch" and not src.get("ref"):
            err("Choose the run to copy the activation from (the 'clean' run).", e["source"])
            continue
        if op in ("steer", "project_out") and not src.get("ref"):
            err("Choose a steering vector (make one in the Steering tab).", e["source"])
            continue
        if op in ("steer", "project_out") and hook in ("attn_heads", "mlp_neurons", "weights", "q", "k", "v", "kv_k", "kv_v"):
            err("A steering vector lives in the residual stream: use resid_pre, attn_out, mlp_out or resid_post.",
                e["source"])
            continue
        if op == "clamp" and str(src.get("clamp") or "max") not in ("max", "min", "value"):
            err("Clamp to max, min or value.", e["source"])
            continue
        if (nodes[e["target"]]["data"].get("code") or "").strip() and hook not in ("resid_pre", "resid_post", "weights"):
            err(f"'{nodes[e['target']]['data'].get('label')}' runs edited code, so only resid_pre, resid_post and weights "
                f"interceptors reach it. Change the code itself for anything in between.", e["source"])
            continue
        ivs.append(Intervention(node_id=e["source"], pos=pos_of_node[e["target"]], hook=hook, op=op,
                                value=_num(src.get("value"), 4.0 if op == "quantize" else 1.0),
                                positions=str(src.get("positions") or "all"),
                                steps=str(src.get("steps") or "all"), seed=_num(src.get("seed"), 0, int),
                                heads=str(src.get("heads") or "all"),
                                region=str(src.get("region") or "") if shape.regions else "",
                                ref=str(src.get("ref") or ""), src_positions=str(src.get("src_positions") or ""),
                                neurons=str(src.get("neurons") or "all"), clamp=str(src.get("clamp") or "max")))
        try:
            parse_index_spec(ivs[-1].neurons, 1_000_000)
            parse_index_spec(ivs[-1].steps if shape.steps else "all", 10_000)
        except ValueError as ex:
            err(str(ex), e["source"])

    if errors:
        raise FlowError(errors)

    params = shape.params(nodes, edges, find_kind, chain, caps)
    params["bypassed"] = sorted(bypassed)
    return Plan(family=family, schedule=schedule, schedule_nodes=chain, interventions=ivs, params=params)


def _round8(v: int) -> int:
    return max(64, min(2048, (v // 8) * 8))


# --------------------------------------------------------------------------------------
# Index specs ("all", "last", "a:b", "1,2,5") for token positions and diffusion steps
# --------------------------------------------------------------------------------------

def parse_index_spec(spec: str, n: int, named: dict[str, range] | None = None) -> list[int]:
    spec = (spec or "all").strip().lower()
    if spec in ("all", "*", ""):
        return list(range(n))
    if spec == "none":
        return []
    if spec == "last":
        return [n - 1] if n else []
    if named and spec in named:
        return list(named[spec])
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(-?\d*)\s*:\s*(-?\d*)", part)
        if m:
            a = int(m.group(1)) if m.group(1) else 0
            b = int(m.group(2)) if m.group(2) else n
            out += list(range(n))[slice(a, b)]
        elif re.fullmatch(r"-?\d+", part):
            i = int(part)
            if -n <= i < n:
                out.append(i % n)
        else:
            raise ValueError(f"Can't read index spec '{spec}'. Use all, last, a:b or 1,2,5.")
    return sorted(set(out))
