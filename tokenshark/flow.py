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
# Plan data model (serialisable - stored in every run manifest)
# --------------------------------------------------------------------------------------

LLM_HOOKS = ["resid_pre", "q", "k", "v", "kv_k", "kv_v", "rope", "attn_probs", "attn_heads", "attn_out", "router", "expert_out", "mlp_neurons",
             "mlp_out", "resid_post", "weights"]
MOE_HOOKS = ["router", "expert_out"]     # only on mixture-of-experts blocks: the router's scores · one expert's output
UNET_HOOKS = ["out", "skip", "both", "word"]
DIT_HOOKS = ["out"]
OPS = ["scale", "zero", "add_noise", "mean", "quantize"]
# Language-model ops that need something from outside the run (or project along a vector):
#   patch       - copy this activation from another run (activation patching; `ref` = that run's id)
#   steer       - add `value` × a saved direction to the activation (`ref` = steering-vector id)
#   project_out - remove (value=1) or scale the component along a saved direction (`ref` = vector id)
#   clamp       - pin chosen neurons (`neurons`) to their min / max / a fixed value (`clamp`)
#   wire        - wire detector neuron(s) (`source_neurons`) into writer neuron(s) (`neurons`) with `value` × relu(src - `threshold`)
#   quantize    - round this tensor / block's weights / KV cache to `value` bits (1..16)
#   ban / only  - mixture-of-experts router: these experts can never be chosen / are the only ones that can
#   route       - mixture-of-experts router: use exactly these experts and weights ("3=0.6, 17=0.4")
LLM_OPS = OPS + ["patch", "steer", "project_out", "clamp", "wire", "ban", "only", "route"]
PATCH_HOOKS = ["resid_pre", "q", "k", "v", "kv_k", "kv_v", "attn_out", "router", "mlp_out", "resid_post"]
CODE_KINDS = ["embed", "block", "final_norm", "lm_head", "sampler"]     # boxes whose Python can be edited
PREFIX_MODES = ["shift", "negative", "offset"]
KV_OVERRIDE_OPS = [
    "replace_system",
    "append_system",
    "prepend_system",
    "replace_user",
    "append_user",
    "prepend_user",
]


@dataclass
class Intervention:
    node_id: str
    pos: int                 # schedule position (index into plan.schedule)
    hook: str
    op: str
    value: float = 1.0
    positions: str = "all"   # token positions (LLM) - all | last | prompt | prefill | generated | a:b | 1,2,5
    steps: str = "all"       # step range (diffusion) - all | a:b | 3
    seed: int = 0
    heads: str = "all"       # attention heads for hooks attn_heads / attn_probs - all | a:b | 1,4
    region: str = ""         # image models: painted mask (PNG data URL) - the intervention only acts there
    ref: str = ""            # patch: source run id · steer: steering-vector id · clamp: run whose min/max to use
    src_positions: str = ""  # patch: token(s) to copy from in the source run ("" = the same token index)
    neurons: str = "all"     # which units of the activation's last axis the op touches - all | 12 | 3,7 | 0:64
    source_neurons: str = "" # wire: detector neuron(s) whose relu(act - threshold) drives the target `neurons`
    threshold: float = 0.0   # wire: activation threshold subtracted before ReLU
    clamp: str = "max"       # clamp: max | min | value
    scale: str = "raw"       # steer: raw (as-is) | unit (v / ||v||) | resid (||resid|| * v / ||v||)
    experts: str = "all"     # mixture-of-experts hooks (router, expert_out, mlp_neurons): all | 3 | 0:8 | 1,5 | shared


@dataclass
class ForkSpec:
    node_id: str
    pos: int                 # schedule position of the block this fork branches from
    layer: int = 0           # underlying layer index of the block
    kind: str = "guardrail"  # e.g. "guardrail" / "fork_head"
    label: str = ""
    hook: str = "resid_post"
    mode: str = "monitor"
    threshold: float = 0.5
    ref: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    family: str                          # causal_lm | t2i
    schedule: list[int]                  # block indices in execution order (repeats / removals allowed)
    schedule_nodes: list[str]            # flow node id at each schedule position
    interventions: list[Intervention] = field(default_factory=list)
    forks: list[ForkSpec] = field(default_factory=list)
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
    moe = info.get("moe") or {}
    moe_layers, moe_n = set(moe.get("layers") or []), moe.get("n_experts")      # these boxes show their experts
    kv_layers = set(info.get("kv_layers")) if info.get("kv_layers") is not None else set(range(n))
    for nid, x, y in _snake(blocks, 0, 210, dy=140 if not moe else 170):
        i = int(nid.split("_")[1])
        subs = info.get("block_subs") or []
        nodes.append(_node(nid, "block", f"Block {i} · new" if i in extra else f"Block {i}", x, y,
                           (f"added after block {extra[i]['after']} · " if i in extra else "")
                           + (subs[i] if i < len(subs) else info.get("block_sub", "")), layer=i,
                           has_kv=(i in kv_layers),
                           **({"kv_from": (info.get("kv_sources") or [])[i]}
                              if i < len(info.get("kv_sources") or []) and (info.get("kv_sources") or [])[i] is not None else {}),
                           **({"moe": moe_n} if i in moe_layers else {})))
    rows = (n + 5) // 6
    yb = 210 + rows * (140 if not moe else 170) + 10
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


CAPTURE_KEYS = ("attn_budget_mb", "neuron_budget_mb", "parts_budget_mb", "expert_budget_mb", "attn_layers",
                "attn_queries")


def _llm_params(nodes: dict, edges: list, find_kind, chain: list, caps: dict) -> dict:
    """Run parameters of a language-model flow: prompt, sampler, KV prefixes, edited box code."""
    params: dict[str, Any] = {}
    p = nodes[find_kind("input_prompt")[0]]["data"]
    s = nodes[find_kind("sampler")[0]]["data"]
    params.update(
        system=p.get("system", ""), prompt=p.get("prompt", ""), prefill=p.get("prefill", ""),
        use_chat_template=bool(p.get("use_chat_template", False)), thinking=bool(p.get("thinking", False)),
        literal_content=bool(p.get("literal_content", False)),
        max_new_tokens=max(0, min(512, _num(s.get("max_new_tokens"), 24, int))),
        min_new_tokens=max(0, _num(s.get("min_new_tokens"), 0, int)),
        ignore_eos=bool(s.get("ignore_eos", False)),
        temperature=max(0.0, _num(s.get("temperature"), 0.0)), top_k=max(0, _num(s.get("top_k"), 0, int)),
        top_p=min(1.0, max(0.0, _num(s.get("top_p"), 1.0))) or 1.0,
        seed=_num(s.get("seed"), 0, int), add_bos=bool(p.get("add_bos", True)),
        rope_step_stride=_num(s.get("rope_step_stride"), 1.0, float) if s.get("rope_step_stride") not in (None, "") else 1.0,
        rope_prompt_scale=_num(s.get("rope_prompt_scale"), 1.0, float) if s.get("rope_prompt_scale") not in (None, "") else 1.0,
    )
    # What the run records (FlowBuilder.capture): memory budgets in MB and which attention rows to keep.
    for key in CAPTURE_KEYS:
        if s.get(key) not in (None, ""):
            params[key] = _num(s[key], 0, int) if key.endswith("_mb") else str(s[key])
    # RoPE jump for every block: the first generated token is placed at this position (later ones follow from it).
    if s.get("rope_gen_start") not in (None, ""):
        params["rope_gen_start"] = _num(s.get("rope_gen_start"), 0.0, float)
    # Earlier turns of the conversation (the Prompt box's list): [{"role": "user" | "assistant", "content": …}].
    hist = []
    for m in p.get("history") or []:
        role, content = (m or {}).get("role"), str((m or {}).get("content") or "")
        if role not in ("user", "assistant"):
            raise FlowError([{"message": "Conversation turns must be 'user' or 'assistant'.", "node": find_kind("input_prompt")[0]}])
        if content.strip():
            hist.append({"role": role, "content": content})
    if hist:
        params["history"] = hist
    # Images (models whose plugin provides a media encoder): data: URLs / base64 strings, one per prompt marker.
    imgs = [str(i) for i in (p.get("images") or []) if i]
    if imgs:
        if not caps.get("image_input"):
            raise FlowError([{"message": "This model doesn't take image input.", "node": find_kind("input_prompt")[0]}])
        params["images"] = imgs
    kv = [nid for nid in find_kind("kv_import")
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
        if not d.get("cache_id"):
            raise FlowError([{"message": "A KV-cache box has no cache selected.", "node": nid}])
        prefixes.append({"cache_id": d["cache_id"], "mode": mode, "node": nid,
                         "layers": str(d.get("layers") or "all"),
                         "offset": _num(d.get("offset"), 0.0), "label": d.get("label") or "imported"})
    if prefixes:
        params.update(prefixes=prefixes, prefix_cache=prefixes[0]["cache_id"])
    # Per-block KV cache overrides: block_kv boxes wired directly into one or more full-attention blocks' "kv" port.
    # Instead of positional hacks, each targeted block recreates its KV cache from the modified system/user prompt
    # and keeps growing gradually as new tokens are generated.
    kv_layers = set(caps.get("_kv_layers") or range(100_000))
    pos_of_node = {nid: i for i, nid in enumerate(chain)}
    bkv_nodes = find_kind("block_kv")
    bkv_nodes.sort(key=lambda nid: (nodes[nid]["position"]["x"], nodes[nid]["position"]["y"]))
    kv_overrides = []
    for nid in bkv_nodes:
        d = nodes[nid]["data"]
        if d.get("enabled") is False:
            continue
        op = str(d.get("op") or "replace_system").strip()
        if op not in KV_OVERRIDE_OPS:
            raise FlowError([{"message": f"KV override op must be one of {', '.join(KV_OVERRIDE_OPS)}.", "node": nid}])
        text = str(d.get("text") or "")
        if not text.strip() and op not in ("replace_system", "replace_user"):
            raise FlowError([{"message": "A Block KV override box needs text to append/prepend.", "node": nid}])
        # Determine target schedule positions: explicit edges to block 'kv' handles, plus optional 'layers' slice
        edge_targets = [e["target"] for e in edges
                        if e["source"] == nid and e.get("targetHandle") == "kv" and e["target"] in nodes]
        slots: list[int] = []
        for t_id in edge_targets:
            if t_id not in pos_of_node:
                if nodes[t_id]["data"].get("bypass"):
                    continue
                raise FlowError([{"message": f"Block KV override '{d.get('label')}' points at a block not on the active path.", "node": nid}])
            lay_idx = int(nodes[t_id]["data"].get("layer", -1))
            if lay_idx not in kv_layers:
                raise FlowError([{"message": f"'{nodes[t_id]['data'].get('label')}' uses linear attention (no per-token KV cache) and cannot receive a KV cache override.", "node": nid}])
            slots.append(pos_of_node[t_id])
        lay_spec = str(d.get("layers") or "").strip()
        if lay_spec:
            try:
                spec_slots = parse_index_spec(lay_spec, len(chain))
            except ValueError as ex:
                raise FlowError([{"message": str(ex), "node": nid}])
            for s_pos in spec_slots:
                lay_idx = int(nodes[chain[s_pos]]["data"].get("layer", -1))
                if lay_idx in kv_layers:
                    slots.append(s_pos)
            if spec_slots and not any(int(nodes[chain[s]]["data"].get("layer", -1)) in kv_layers for s in spec_slots):
                raise FlowError([{"message": f"Blocks '{lay_spec}' only cover linear-attention layers, which have no KV cache.", "node": nid}])
        slots = sorted(set(slots))
        if not slots and not edge_targets and not lay_spec:
            raise FlowError([{"message": f"Connect '{d.get('label') or 'Block KV'}' to a block's KV port (or specify Blocks).", "node": nid}])
        if slots:
            kv_overrides.append({
                "node": nid,
                "op": op,
                "text": text,
                "slots": slots,
                "layers": lay_spec,
                "label": d.get("label") or op.replace("_", " "),
            })
    if kv_overrides:
        params["kv_overrides"] = kv_overrides
    # Edited Python for boxes (see blockcode.py): {box id: source}. Keyed by box, so a repeated block can run
    # different code from the original.
    code = {}
    for nid in chain + [n for k in CODE_KINDS if k != "block" for n in find_kind(k)]:
        if str(nodes[nid]["data"].get("code") or "").strip():
            code[nid] = {"kind": nodes[nid]["data"]["kind"], "source": str(nodes[nid]["data"]["code"])}
    if code:
        params["code"] = code
    # Mixture-of-experts routing settings typed into a block box: experts per token, router temperature.
    moe = {}
    for nid in chain:
        d = nodes[nid]["data"]
        k, temp = _num(d.get("moe_k"), 0, int), _num(d.get("moe_temp"), 0.0)
        if k > 0 or temp > 0 or d.get("moe_shared") is False:
            moe[nid] = {"k": max(0, k), "temp": max(0.0, temp), "shared": d.get("moe_shared") is not False}
    if moe:
        params["moe"] = moe
    # Per-block RoPE settings typed into a block box: position scale (0 = off), base theta override, query shift.
    rope = {}
    for nid in chain:
        d = nodes[nid]["data"]
        r_entry = {}
        if d.get("rope_scale") not in (None, ""):
            sc_val = _num(d.get("rope_scale"), 1.0, float)
            if sc_val != 1.0:
                r_entry["scale"] = sc_val
        if d.get("rope_theta") not in (None, ""):
            th_val = _num(d.get("rope_theta"), 0.0, float)
            if th_val > 0:
                r_entry["theta"] = th_val
        if d.get("rope_q_offset") not in (None, ""):
            qo_val = _num(d.get("rope_q_offset"), 0.0, float)
            if qo_val != 0.0:
                r_entry["q_offset"] = qo_val
        if d.get("rope_gen_start") not in (None, ""):
            r_entry["gen_start"] = _num(d.get("rope_gen_start"), 0.0, float)
        if r_entry:
            rope[nid] = r_entry
    if rope:
        params["rope"] = rope
    # Zeroed KV records: cache rows (token index in the run's KV cache) whose key and value are set to 0,
    # in every block (Sampler box's kv_zero) and/or in single blocks (a block box's kv_zero).
    def _rows(spec) -> set[int]:
        out = set()
        for part in str(spec or "").replace(" ", "").split(","):
            if part.isdigit():
                out.add(int(part))
            elif ":" in part:
                a, _, b = part.partition(":")
                if a.isdigit() and b.isdigit():
                    out.update(range(int(a), int(b)))
        return out
    every = _rows(s.get("kv_zero"))
    kv_zero = {}
    for nid in chain:
        d = nodes[nid]["data"]
        if d.get("kind") != "block":
            continue
        rows = every | _rows(d.get("kv_zero"))
        if rows:
            kv_zero[nid] = sorted(rows)
    if kv_zero:
        params["kv_zero"] = kv_zero
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
    caps = dict(info.get("capabilities", {}))
    if info.get("kv_layers") is not None:
        caps["_kv_layers"] = list(info["kv_layers"])
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
            err(f"The residual stream stops at '{nodes[cur]['data']['label']}' - connect it onward to "
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
        from .sdk import get_op
        custom_op = get_op(op)
        if op not in shape.ops and custom_op is None:
            err(f"Unknown op '{op}'.", e["source"])
            continue
        if custom_op and custom_op.valid_hooks and hook not in custom_op.valid_hooks:
            err(f"Op '{op}' only applies to hooks {', '.join(custom_op.valid_hooks)}.", e["source"])
            continue
        if custom_op and custom_op.validate_fn:
            if not custom_op.validate_fn(src, e["source"], info, err):
                continue
        if hook == "weights" and op not in ("quantize", "scale", "zero", "add_noise"):
            err("Hook 'weights' supports quantize, scale, zero or add_noise.", e["source"])
            continue
        if hook == "rope" and op not in ("zero", "scale", "add_noise", "clamp"):
            err("Hook 'rope' supports zero (no RoPE), scale (stretch/compress positions), add_noise or clamp.", e["source"])
            continue
        kv_layers = set(info.get("kv_layers")) if info.get("kv_layers") is not None else None
        target_layer = int(nodes[e["target"]]["data"].get("layer", -1))
        if kv_layers is not None and target_layer not in kv_layers and hook in ("kv_k", "kv_v", "q", "k", "v", "rope"):
            err(f"'{nodes[e['target']]['data'].get('label')}' is a linear-attention block (fixed recurrent state, no per-token KV cache or RoPE): hook '{hook}' only applies to KV-cache blocks.", e["source"])
            continue
        moe_layers = (info.get("moe") or {}).get("layers") or []
        on_moe = int(nodes[e["target"]]["data"].get("layer", -1)) in moe_layers
        if (hook in MOE_HOOKS or op in ("ban", "only", "route")) and not on_moe:
            err(f"'{nodes[e['target']]['data'].get('label')}' is not a mixture-of-experts block: it has no router "
                f"or experts to intercept.", e["source"])
            continue
        if op in ("ban", "only", "route") and hook != "router":
            err("ban / only / route act on the router: set the hook to 'router'.", e["source"])
            continue
        if op == "route" and str(src.get("experts") or "all").strip().lower() in ("all", "*", ""):
            err("A custom route needs its experts, e.g. 3=0.6, 17=0.4 (or 3, 17 to keep the router's own weights).",
                e["source"])
            continue
        if op == "route" and re.search(r":|shared|all", str(src.get("experts")).lower()):
            err("A custom route lists single experts: 3=0.6, 17=0.4.", e["source"])
            continue
        if hook == "expert_out" and op in ("patch", "clamp", "quantize"):
            err("expert_out supports zero, scale, add_noise, mean, steer and project_out.", e["source"])
            continue
        try:
            _expert_spec(str(src.get("experts") or "all"))
        except ValueError as ex:
            err(str(ex), e["source"])
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
        if op in ("steer", "project_out") and hook in ("attn_probs", "attn_heads", "mlp_neurons", "weights", "q", "k", "v", "kv_k", "kv_v", "rope", "router"):
            err("A steering vector lives in the residual stream: use resid_pre, attn_out, mlp_out or resid_post.",
                e["source"])
            continue
        if op == "clamp" and str(src.get("clamp") or "max") not in ("max", "min", "value"):
            err("Clamp to max, min or value.", e["source"])
            continue
        if op == "steer" and str(src.get("scale") or "raw") not in ("raw", "unit", "resid"):
            err("Steer scale must be 'raw', 'unit' or 'resid'.", e["source"])
            continue
        if op == "wire":
            if hook != "mlp_neurons":
                err("Wire connects MLP neurons: set the hook to 'mlp_neurons'.", e["source"])
                continue
            s_neu = str(src.get("source_neurons") or "").strip()
            t_neu = str(src.get("neurons") or "").strip()
            if not s_neu or s_neu.lower() in ("all", "*"):
                err("A neuron wire needs source (detector) neuron(s), e.g. 2751.", e["source"])
                continue
            if not t_neu or t_neu.lower() in ("all", "*"):
                err("A neuron wire needs target (writer) neuron(s), e.g. 2282 or 836,10,1871.", e["source"])
                continue
            try:
                if not parse_index_spec(s_neu, 1_000_000):
                    raise ValueError("Specify at least one source neuron.")
                if not parse_index_spec(t_neu, 1_000_000):
                    raise ValueError("Specify at least one target neuron.")
            except ValueError as ex:
                err(str(ex), e["source"])
                continue
        if (nodes[e["target"]]["data"].get("code") or "").strip() and hook not in ("resid_pre", "resid_post", "weights"):
            err(f"'{nodes[e['target']]['data'].get('label')}' runs edited code, so only resid_pre, resid_post and weights "
                f"interceptors reach it. Change the code itself for anything in between.", e["source"])
            continue
        ivs.append(Intervention(node_id=e["source"], pos=pos_of_node[e["target"]], hook=hook, op=op,
                                value=_num(src.get("value"), 4.0 if op == "quantize" else (3.0 if op == "wire" else 1.0)),
                                positions=str(src.get("positions") or "all"),
                                steps=str(src.get("steps") or "all"), seed=_num(src.get("seed"), 0, int),
                                heads=str(src.get("heads") or "all"),
                                region=str(src.get("region") or "") if shape.regions else "",
                                ref=str(src.get("ref") or ""), src_positions=str(src.get("src_positions") or ""),
                                neurons=str(src.get("neurons") or "all"),
                                source_neurons=str(src.get("source_neurons") or ""),
                                threshold=_num(src.get("threshold"), 0.0, float),
                                clamp=str(src.get("clamp") or "max"),
                                scale=str(src.get("scale") or "raw"),
                                experts=str(src.get("experts") or "all")))
        try:
            parse_index_spec(ivs[-1].neurons, 1_000_000)
            parse_index_spec(ivs[-1].steps if shape.steps else "all", 10_000)
        except ValueError as ex:
            err(str(ex), e["source"])

    # Side-branch forks: boxes wired from a block's "fork" handle (e.g. guardrail / auxiliary probe heads).
    from .sdk import get_node_kind
    forks: list[ForkSpec] = []
    for e in edges:
        if e.get("sourceHandle") != "fork" or e["source"] not in nodes or e["target"] not in nodes:
            continue
        if e["source"] in bypassed:
            continue
        if e["source"] not in pos_of_node:
            err(f"Fork source '{nodes[e['source']]['data'].get('label')}' is not on the active block path.", e["source"])
            continue
        tnode = nodes[e["target"]]
        td = tnode.get("data") or {}
        if td.get("enabled") is False:
            continue
        tkind = str(td.get("kind") or "fork_head")
        nks = get_node_kind(tkind)
        if tkind != "fork_head" and (nks is None or not nks.is_fork_target):
            err(f"Only fork / probe boxes can connect to a block's fork handle (found '{tkind}').", e["target"])
            continue
        f_params = {k: v for k, v in td.items() if k not in ("kind", "label", "sub")}
        if nks and nks.compile_fn:
            try:
                f_params = nks.compile_fn(td, nodes, edges, info)
            except ValueError as ex:
                err(str(ex), e["target"])
                continue
        b_pos = pos_of_node[e["source"]]
        b_lay = int(nodes[e["source"]]["data"].get("layer", b_pos))
        forks.append(ForkSpec(
            node_id=e["target"],
            pos=b_pos,
            layer=b_lay,
            kind=str(td.get("fork_kind") or tkind),
            label=str(td.get("label") or tkind),
            hook=str(td.get("hook") or "resid_post"),
            mode=str(td.get("mode") or "monitor"),
            threshold=_num(td.get("threshold"), 0.5, float),
            ref=str(td.get("ref") or ""),
            params=f_params,
        ))

    if errors:
        raise FlowError(errors)

    params = shape.params(nodes, edges, find_kind, chain, caps)
    params["bypassed"] = sorted(bypassed)
    if forks:
        params["forks"] = [asdict(f) for f in forks]
    return Plan(family=family, schedule=schedule, schedule_nodes=chain, interventions=ivs, forks=forks, params=params)


def _expert_spec(spec: str):
    """Check an experts spec (all | 3 | 0:8 | 1,5 | shared) without knowing the number of experts."""
    names = [x.strip().split("=") for x in spec.lower().split(",") if x.strip() and x.strip() != "shared"]
    for n in names:
        if len(n) == 2:
            try:
                float(n[1])
            except ValueError:
                raise ValueError(f"Can't read the weight in '{'='.join(n)}'. Write expert=weight, e.g. 3=0.6.") from None
    parse_index_spec(",".join(n[0].strip() for n in names) or "all", 1_000_000)


def _round8(v: int) -> int:
    return max(64, min(2048, (v // 8) * 8))


# --------------------------------------------------------------------------------------
# Index specs ("all", "last", "a:b", "1,2,5") for token positions and diffusion steps
# --------------------------------------------------------------------------------------

def parse_index_spec(spec: str | range | list | tuple, n: int, named: dict[str, range] | None = None) -> list[int]:
    if isinstance(spec, range):
        return sorted({i % n for i in spec if -n <= i < n})
    if isinstance(spec, (list, tuple, set)):
        return sorted({int(i) % n for i in spec if -n <= int(i) < n})
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
