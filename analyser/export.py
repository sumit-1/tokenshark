"""Export a run as a Python script that reproduces it with the public API (analyser.api).

The script rebuilds the flow from the model's default flow, one readable line per change (prompt, sampler,
bypassed / repeated blocks, interventions, layer prompts, edited box code). Runs it depends on — the source of
an activation patch — are exported first, in the same file.
"""
from __future__ import annotations

from .flowbuild import walk_chain

_IV_DEFAULTS = {"hook": "mlp_out", "op": "zero", "value": 0.0, "positions": "all", "heads": "all", "neurons": "all",
                "src_positions": "", "clamp": "max", "seed": 0, "steps": "all"}


def _lit(v) -> str:
    if isinstance(v, str) and "\n" in v:
        q = "'''" if "'''" not in v else '"""'
        return f"{q}{v}{q}" if q not in v and not v.endswith(q[0]) and "\\" not in v else repr(v)
    return repr(v)


def _kw(pairs) -> str:
    return ", ".join(f"{k}={_lit(v)}" for k, v in pairs)


def _node(flow, kind):
    return next((n for n in flow["nodes"] if n["data"].get("kind") == kind), None)


def _diff(cur: dict, base: dict, keys: dict) -> list:
    """[(api name, value)] for the fields of a box that differ from the default flow."""
    return [(api, cur.get(k)) for k, api in keys.items() if cur.get(k) is not None and cur.get(k) != base.get(k)]


def flow_lines(flow: dict, default: dict, var: str = "flow", refs: dict | None = None) -> list[str]:
    """Python lines that turn the default flow into `flow`. refs: {run id: variable name} for patch sources."""
    refs = refs or {}
    out = []
    p, p0 = _node(flow, "input_prompt"), _node(default, "input_prompt")
    if p and p0:
        d = _diff(p["data"], p0["data"], {"prompt": "prompt", "system": "system", "prefill": "prefill",
                                         "use_chat_template": "chat", "thinking": "thinking", "negative_prompt": "negative_prompt"})
        if d:
            out.append(f"{var}.prompt({_kw(d)})")
    s, s0 = _node(flow, "sampler"), _node(default, "sampler")
    if s and s0:
        d = _diff(s["data"], s0["data"], {"max_new_tokens": "max_new_tokens", "temperature": "temperature",
                                         "top_k": "top_k", "top_p": "top_p", "seed": "seed"})
        if d:
            out.append(f"{var}.sampler({_kw(d)})")
    for kind in ("noise", "scheduler"):                      # image models: set the box fields directly
        n, n0 = _node(flow, kind), _node(default, kind)
        if n and n0:
            d = [(k, v) for k, v in n["data"].items() if n0["data"].get(k) != v and k not in ("label", "sub", "kind")]
            if d:
                out.append(f"{var}.node({kind!r})['data'].update({_kw(d)})")

    # the block chain: which box sits where, under the id the builder will give it
    chain = walk_chain(flow) or walk_chain(flow, "denoiser_in", "denoiser_out")
    base = walk_chain(default) or walk_chain(default, "denoiser_in", "denoiser_out")
    layers = [int(n["data"]["layer"]) for n in chain]
    name = {}
    if layers != [int(n["data"]["layer"]) for n in base]:
        out.append(f"{var}.schedule({layers})")
        seen = {}
        first = {int(n["data"]["layer"]): n["id"] for n in reversed(base)}
        for n, l in zip(chain, layers):
            k = seen.get(l, 0)
            seen[l] = k + 1
            name[n["id"]] = first.get(l, f"block_{l}") if k == 0 else f"block_{l}_r{k}"
    else:
        name = {n["id"]: b["id"] for n, b in zip(chain, base)}

    def ref(nid):
        b = name.get(nid, nid)
        return int(b.split("_")[1]) if b.startswith("block_") and b.count("_") == 1 and b.split("_")[1].isdigit() else b

    byp = [ref(n["id"]) for n in chain if n["data"].get("bypass")]
    if byp:
        out.append(f"{var}.bypass({', '.join(_lit(b) for b in byp)})")
    for n in chain + [x for x in flow["nodes"] if x["data"].get("kind") in ("embed", "final_norm", "lm_head", "sampler")]:
        if str(n["data"].get("code") or "").strip():
            target = ref(n["id"]) if n in chain else n["data"]["kind"]
            out.append(f"{var}.code({_lit(target)}, {_lit(n['data']['code'])})")

    nodes = {n["id"]: n for n in flow["nodes"]}
    for e in flow.get("edges", []):
        src = nodes.get(e["source"])
        if src is None or e["target"] not in nodes:
            continue
        d, k = src["data"], src["data"].get("kind")
        if k == "intervention" and e.get("targetHandle") == "mod" and d.get("enabled") is not False:
            tgt = _lit(ref(e["target"]))
            op = d.get("op", "scale")
            src_ref = refs.get(d.get("ref"), _lit(d.get("ref", "")))
            if op == "patch":
                extra = _diff(d, {"hook": "resid_post", "positions": "all", "src_positions": ""},
                              {"hook": "hook", "positions": "positions", "src_positions": "src_positions"})
                out.append(f"{var}.patch({tgt}, {src_ref}{', ' + _kw(extra) if extra else ''})")
            elif op == "steer":
                extra = _diff(d, {"hook": "resid_post", "positions": "all"}, {"value": "value", "hook": "hook", "positions": "positions"})
                out.append(f"{var}.steer({tgt}, {src_ref}{', ' + _kw(extra) if extra else ''})")
            elif op == "clamp":
                extra = _diff(d, {"hook": "mlp_neurons", "positions": "all", "ref": ""},
                              {"clamp": "to", "value": "value", "positions": "positions", "hook": "hook"})
                r = f", ref={src_ref}" if d.get("ref") else ""
                out.append(f"{var}.clamp({tgt}, {_lit(str(d.get('neurons') or 'all'))}{', ' + _kw(extra) if extra else ''}{r})")
            else:
                extra = [("hook", d.get("hook")), ("op", op)] + _diff(d, _IV_DEFAULTS, {
                    "value": "value", "positions": "positions", "heads": "heads", "neurons": "neurons", "seed": "seed",
                    "steps": "steps"})
                out.append(f"{var}.intervene({tgt}, {_kw(extra)})")
        elif k == "kv_text" and e.get("targetHandle") == "cache" and d.get("enabled") is not False:
            extra = _diff(d, {"layers": "all", "mode": "shift", "as_system": False, "offset": 0},
                          {"layers": "layers", "mode": "mode", "as_system": "as_system", "offset": "offset"})
            out.append(f"{var}.layer_prompt({_lit(d.get('text', ''))}{', ' + _kw(extra) if extra else ''})")
        elif k == "kv_import" and e.get("targetHandle") == "cache" and d.get("enabled") is not False:
            extra = _diff(d, {"layers": "all", "mode": "shift", "offset": 0}, {"layers": "layers", "mode": "mode", "offset": "offset"})
            out.append(f"{var}.kv_prefix({_lit(d.get('cache_id', ''))}{', ' + _kw(extra) if extra else ''})"
                       f"      # a KV cache saved in data/caches/ on this machine")
        elif k in ("init_image", "prompt_kv"):
            out.append(f"# NOTE: the '{d.get('label') or k}' box is not expressed here; restore the flow in the app for it.")
    return out


def run_script(run, default_flow_for, get_run, vectors: dict | None = None) -> str:
    """The script for `run`. default_flow_for(run) gives the model's default flow; get_run(id) another run or None."""
    vectors = vectors or {}
    done: dict[str, str] = {}
    body: list[str] = []

    def emit(r, var: str):
        refs = {}
        for iv in r.plan.interventions:
            if iv.op in ("patch", "clamp") and iv.ref and iv.ref not in done:
                src = get_run(iv.ref)
                if src is not None and src is not r:
                    emit(src, f"source_{len(done) + 1}")
        for iv in r.plan.interventions:
            if iv.ref in done:
                refs[iv.ref] = done[iv.ref]
            elif iv.op == "steer" and iv.ref in vectors:
                body.append(f"# steering vector {iv.ref!r}: {vectors[iv.ref].get('name', '')} "
                            f"(saved in data/vectors/ on this machine)")
        ov = r.plan.params.get("overlays") or {}
        body.append(f"# --- run {r.id}" + (f" · {r.label}" if r.label else ""))
        if "lora" in ov:
            body.append(f"lab.lora({ov['lora']['path']!r}, scale={ov['lora']['scale']})")
        if "quant" in ov:
            body.append(f"lab.quantize({ov['quant']})")
        body.append("flow = lab.flow()")
        body.extend(flow_lines(r.flow, default_flow_for(r), "flow", refs))
        body.append(f"{var} = lab.run(flow)")
        if "lora" in ov:
            body.append("lab.lora(None)")
        if "quant" in ov:
            body.append("lab.quantize(None)")
        body.append("")
        done[r.id] = var

    emit(run, "run")
    image = run.plan.family == "t2i"
    head = [f'"""Run {run.id}{" · " + run.label if run.label else ""} — exported from AI Analyser.',
            "",
            "Reproduces the run with the same engine the app uses. Run it from the app's folder:",
            f"    python run_{run.id}.py",
            'Change anything and run again; `run` holds the activations (run.resid, run.attention, run.lens …)."""',
            "from analyser.api import Lab", "",
            f"lab = Lab({run.model_id!r})", ""]
    tail = ['run.image.save("out.png")', 'print("saved out.png")'] if image else ["print(run.text)"]
    return "\n".join(head + body + tail) + "\n"
