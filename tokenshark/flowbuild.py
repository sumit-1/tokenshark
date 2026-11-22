"""FlowBuilder: edit a flow (the boxes-and-arrows graph the UI draws) from Python, in one line per change.

Pure Python, no torch - used by the Python API (tokenshark.api), by exported run scripts and by the MCP tutor.

    flow = lab.flow()                                   # the model's default flow
    flow.prompt("The capital of France is").sampler(max_new_tokens=8)
    flow.bypass(3)                                      # skip block 3
    flow.intervene(5, hook="mlp_out", op="zero")        # zero what block 5's MLP writes
    flow.patch(7, clean_run, hook="resid_post", positions="last")
    flow.steer(6, vector, value=8)
    flow.clamp(4, neurons="1021", to="max", value=5)
    flow.replace_system("Answer in French.", blocks="10:14")
    flow.code(2, source)                                # run your own Python for block 2

A block is named by its layer number (the box "Block 3") or by its box id (needed for repeated blocks).
"""
from __future__ import annotations

import copy
import itertools


def walk_chain(flow: dict, start_kind: str = "embed", end_kind: str = "final_norm") -> list[dict]:
    """The block boxes on the residual stream, in order (bypassed ones included)."""
    nodes = {n["id"]: n for n in flow.get("nodes", [])}
    nxt = {}
    for e in flow.get("edges", []):
        d = e.get("data") or {}
        if d.get("loop") or d.get("skip") or e.get("sourceHandle") not in (None, "out") or e.get("targetHandle") not in (None, "in"):
            continue
        nxt.setdefault(e["source"], e["target"])
    cur = next((nid for nid, n in nodes.items() if n["data"].get("kind") == start_kind), None)
    out, seen = [], set()
    while cur in nxt:
        cur = nxt[cur]
        if cur not in nodes or nodes[cur]["data"].get("kind") == end_kind or cur in seen:
            break
        seen.add(cur)
        out.append(nodes[cur])
    return out


class FlowBuilder:
    def __init__(self, flow: dict):
        self.flow = copy.deepcopy(flow)
        self._n = itertools.count(1)

    # ---------------------------------------------------------------- access
    def to_dict(self) -> dict:
        return self.flow

    def node(self, ref) -> dict:
        """A box by id, by kind ('sampler'), or - for an int - the block box of that layer."""
        nodes = self.flow["nodes"]
        if isinstance(ref, int):
            ref = f"block_{ref}"
        for n in nodes:
            if n["id"] == ref:
                return n
        for n in nodes:
            if n["data"].get("kind") == ref:
                return n
        raise KeyError(f"No box '{ref}' in the flow.")

    def _set(self, ref, **data):
        self.node(ref)["data"].update({k: v for k, v in data.items() if v is not None})
        return self

    def _add(self, prefix: str, target: str, handle: str, data: dict) -> str:
        t = self.node(target)
        nid = f"{prefix}_{next(self._n)}"
        while any(n["id"] == nid for n in self.flow["nodes"]):
            nid = f"{prefix}_{next(self._n)}"
        self.flow["nodes"].append({"id": nid, "type": "box",
                                   "position": {"x": t["position"]["x"] + 10 + 24 * (next(self._n) % 5), "y": t["position"]["y"] - 105},
                                   "data": data})
        self.flow["edges"].append({"id": f"e-{nid}-out-{t['id']}-{handle}", "source": nid, "target": t["id"],
                                   "sourceHandle": "out", "targetHandle": handle})
        return nid

    # ---------------------------------------------------------------- inputs and sampling
    def prompt(self, prompt=None, system=None, prefill=None, chat=None, thinking=None, negative_prompt=None,
               history=None):
        """history: the conversation so far, [{"role": "user" | "assistant", "content": …}, …]; `prompt` is then the
        newest user turn (leave it "" to have the model answer the last turn of the history)."""
        return self._set("input_prompt", prompt=prompt, system=system, prefill=prefill, use_chat_template=chat,
                         thinking=thinking, negative_prompt=negative_prompt,
                         history=[dict(m) for m in history] if history is not None else None)

    def say(self, role: str, content: str):
        """Append one earlier turn to the conversation: flow.say("user", "Hi").say("assistant", "Hello!")."""
        d = self.node("input_prompt")["data"]
        d["history"] = list(d.get("history") or []) + [{"role": role, "content": content}]
        return self

    def sampler(self, max_new_tokens=None, temperature=None, top_k=None, top_p=None, seed=None,
                rope_step_stride=None, rope_prompt_scale=None):
        return self._set("sampler", max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                         seed=seed, rope_step_stride=rope_step_stride, rope_prompt_scale=rope_prompt_scale)

    def rope_step(self, stride: float | None = None, prompt_scale: float | None = None):
        """Change how RoPE positions advance per generation step (stride=0 freezes decode position; stride=50
        fast-forwards so earlier turns fade fast; prompt_scale=0 collapses prompt positions to 0)."""
        return self.sampler(rope_step_stride=stride, rope_prompt_scale=prompt_scale)

    def rope(self, block="all", scale: float | None = None, theta: float | None = None, q_offset: float | None = None):
        """Change RoPE on one block, a list of blocks, or a slice spec ('all', '12:24', '1,2'):
        scale=0 turns off RoPE on those blocks; theta overrides base frequency; q_offset shifts Q positions vs K."""
        chain = walk_chain(self.flow)
        if isinstance(block, int):
            targets = [self.node(block)]
        elif isinstance(block, (list, tuple)):
            targets = [self.node(b) for b in block]
        elif isinstance(block, str) and block in {n["id"] for n in chain}:
            targets = [self.node(block)]
        else:
            from .flow import parse_index_spec
            idx = parse_index_spec(str(block), len(chain))
            targets = [chain[i] for i in idx]
        for n in targets:
            d = n["data"]
            for key, v in (("rope_scale", scale), ("rope_theta", theta), ("rope_q_offset", q_offset)):
                if v is not None:
                    d[key] = v
        return self

    # ---------------------------------------------------------------- the block chain
    def bypass(self, *blocks, on: bool = True):
        for b in blocks:
            self.node(b)["data"]["bypass"] = on
        return self

    def schedule(self, order: list[int]):
        """Rebuild the chain of block boxes: remove, repeat or re-order layers, e.g. [0, 1, 1, 3]. The first use of
        a layer keeps its box (block_3); repeats get new boxes (block_3_r1, block_3_r2 …)."""
        old = walk_chain(self.flow)
        if not old:
            raise ValueError("This flow has no block chain.")
        by_layer = {}
        for n in old:
            by_layer.setdefault(int(n["data"]["layer"]), n)
        ids_old = {n["id"] for n in old}
        first, last = self.node("embed")["id"], self.node("final_norm")["id"]
        self.flow["edges"] = [e for e in self.flow["edges"]
                              if not ((e["source"] in ids_old or e["target"] in ids_old or (e["source"] == first and e["target"] == last))
                                      and e.get("sourceHandle") in (None, "out") and e.get("targetHandle") in (None, "in"))]
        chain, used = [], {}
        for layer in order:
            if layer not in by_layer:
                raise ValueError(f"The flow has no block for layer {layer}.")
            k = used.get(layer, 0)
            used[layer] = k + 1
            if k == 0:
                chain.append(by_layer[layer]["id"])
            else:
                src = by_layer[layer]
                nid = f"block_{layer}_r{k}"
                self.flow["nodes"].append({"id": nid, "type": "box",
                                           "position": {"x": src["position"]["x"] + 30 * k, "y": src["position"]["y"] + 60 * k},
                                           "data": {**{a: b for a, b in src["data"].items() if a not in ("code", "bypass")},
                                                    "label": f"Block {layer} (repeat)"}})
                chain.append(nid)
        keep = set(chain)
        self.flow["nodes"] = [n for n in self.flow["nodes"] if n["id"] not in ids_old or n["id"] in keep]
        self.flow["edges"] = [e for e in self.flow["edges"] if not ((e["source"] in ids_old and e["source"] not in keep)
                                                                    or (e["target"] in ids_old and e["target"] not in keep))]
        path = [first, *chain, last]
        self.flow["edges"] += [{"id": f"e-{a}-out-{b}-in", "source": a, "target": b, "sourceHandle": "out", "targetHandle": "in"}
                               for a, b in zip(path, path[1:])]
        return self

    # ---------------------------------------------------------------- interceptors
    def intercept(self, block, hook: str = "mlp_out", op: str = "zero", value: float = 0.0, positions: str = "all",
                  heads: str = "all", neurons: str = "all", source_neurons: str = "", threshold: float = 0.0,
                  ref: str = "", src_positions: str = "", clamp: str = "max",
                  seed: int = 0, steps: str = "all", label: str = "", experts: str = "all") -> str:
        """Wire an interceptor box into a block. Returns the new box's id."""
        ref = getattr(ref, "id", ref)             # a RunResult or a vector dict/id
        if isinstance(ref, dict):
            ref = ref["id"]
        return self._add("iv", self.node(block)["id"], "mod", {
            "kind": "intervention", "label": label or f"{op} @ {hook}", "hook": hook, "op": op, "value": value,
            "positions": str(positions), "heads": str(heads), "neurons": str(neurons),
            "source_neurons": str(source_neurons), "threshold": float(threshold),
            "ref": ref or "", "src_positions": str(src_positions), "clamp": clamp, "seed": seed,
            "steps": steps, "enabled": True, "experts": str(experts)})

    intervene = intercept

    def patch(self, block, source_run, hook: str = "resid_post", positions: str = "all", src_positions: str = "") -> str:
        """Activation patching: at `block`, overwrite this run's activation with the one `source_run` had there."""
        return self.intercept(block, hook=hook, op="patch", ref=source_run, positions=positions,
                              src_positions=src_positions, label="patch")

    def steer(self, block, vector, value: float = 4.0, hook: str = "resid_post", positions: str = "all") -> str:
        """Add value × a steering vector to the residual stream at `block`."""
        return self.intercept(block, hook=hook, op="steer", ref=vector, value=value, positions=positions, label="steer")

    def project_out(self, block, vector, value: float = 1.0, hook: str = "resid_post", positions: str = "all") -> str:
        """Remove (value=1) or scale the component along `vector` in the residual stream at `block`."""
        return self.intercept(block, hook=hook, op="project_out", ref=vector, value=value, positions=positions,
                              label="project_out")

    def quantize_block(self, block, bits: int = 4, hook: str = "weights", positions: str = "all",
                       heads: str = "all") -> str:
        """Quantize a block's weights (hook='weights'), KV cache (hook='kv_k'|'kv_v'), or activation to `bits` bits."""
        return self.intercept(block, hook=hook, op="quantize", value=float(bits), positions=positions, heads=heads,
                              label=f"quantize {bits}-bit @ {hook}")

    def clamp(self, block, neurons, to: str = "max", value: float = 1.0, ref: str = "", positions: str = "all",
              hook: str = "mlp_neurons") -> str:
        """Pin MLP neurons of `block` to their max / min (× value) or to a fixed value (to='value')."""
        return self.intercept(block, hook=hook, op="clamp", neurons=str(neurons), clamp=to, value=value, ref=ref,
                              positions=positions, label=f"neuron {neurons} → {to}")

    def wire(self, block, source, target, strength: float = 3.0, threshold: float = 0.0,
             positions: str = "all", experts: str = "all", label: str = "") -> str:
        """Wire detector neuron(s) `source` into writer neuron(s) `target` in `block`'s MLP:
        neurons[:, target] += strength * relu(neurons[:, source] - threshold)."""
        src_s = ",".join(map(str, source)) if isinstance(source, (list, tuple)) else str(source)
        tgt_s = ",".join(map(str, target)) if isinstance(target, (list, tuple)) else str(target)
        th_lbl = f" (>{threshold:g})" if threshold else ""
        return self.intercept(block, hook="mlp_neurons", op="wire", source_neurons=src_s, neurons=tgt_s,
                              value=float(strength), threshold=float(threshold), positions=positions,
                              experts=experts, label=label or f"wire {src_s} -> {tgt_s} ×{strength:g}{th_lbl}")

    def fork(self, block, ref: str = "powershell_detector", mode: str = "monitor",
             threshold: float = 0.65, hook: str = "resid_post", trigger_category: str = "",
             halt_message: str = "", label: str = "") -> str:
        """Attach a side-branch `fork_head` box to `block`'s bottom `fork` handle."""
        t = self.node(block)
        nid = f"fork_{next(self._n)}"
        while any(n["id"] == nid for n in self.flow["nodes"]):
            nid = f"fork_{next(self._n)}"
        self.flow["nodes"].append({
            "id": nid,
            "type": "box",
            "position": {"x": t["position"]["x"] + 8, "y": t["position"]["y"] + 115},
            "data": {
                "kind": "fork_head",
                "fork_kind": "guardrail",
                "label": label or f"Fork: {ref} ({mode})",
                "ref": ref,
                "mode": mode,
                "threshold": float(threshold),
                "hook": hook,
                "trigger_category": trigger_category,
                "halt_message": halt_message,
                "enabled": True,
            },
        })
        self.flow["edges"].append({
            "id": f"e-{t['id']}-fork-{nid}-in",
            "source": t["id"],
            "target": nid,
            "sourceHandle": "fork",
            "targetHandle": "in",
            "data": {"fork": True},
        })
        return nid

    # ---------------------------------------------------------------- mixture of experts
    def ban_experts(self, block, experts, positions: str = "all") -> str:
        """The router of `block` may never choose these experts (all | 3 | 0:8 | 1,5): others take their place."""
        return self.intercept(block, hook="router", op="ban", experts=experts, positions=positions,
                              label=f"ban expert {experts}")

    def only_experts(self, block, experts, positions: str = "all") -> str:
        """The router of `block` may choose among these experts only."""
        return self.intercept(block, hook="router", op="only", experts=experts, positions=positions,
                              label=f"only expert {experts}")

    def scale_expert(self, block, experts, value: float = 0.0, positions: str = "all") -> str:
        """Multiply what these experts write by `value` wherever they are chosen (0 = they run but say nothing)."""
        return self.intercept(block, hook="expert_out", op="scale" if value else "zero", value=value, experts=experts,
                              positions=positions, label=f"expert {experts} x {value:g}")

    def route(self, block, experts, positions: str = "all") -> str:
        """Write the routing of `block` by hand. experts: {3: 0.6, 17: 0.4} (expert → weight), or [3, 17] to keep
        the router's own scores for those experts. The router's choice is replaced for the chosen tokens."""
        if isinstance(experts, dict):
            spec = ", ".join(f"{e}={w:g}" for e, w in experts.items())
        else:
            spec = ", ".join(str(e) for e in experts) if not isinstance(experts, str) else experts
        return self.intercept(block, hook="router", op="route", experts=spec, positions=positions, label=f"route {spec}"[:40])

    def shared_expert(self, block, on: bool = True):
        """Switch a block's shared expert (the one every token uses) off or back on."""
        d = self.node(block)["data"]
        if on:
            d.pop("moe_shared", None)
        else:
            d["moe_shared"] = False
        return self

    def routing(self, block, k: int | None = None, temperature: float | None = None):
        """Change how a block routes: experts per token (k) and the router's softmax temperature (None = model's own)."""
        d = self.node(block)["data"]
        for key, v in (("moe_k", k), ("moe_temp", temperature)):
            if v:
                d[key] = v
            else:
                d.pop(key, None)
        return self

    # ---------------------------------------------------------------- KV overrides & prefixes
    def kv_override(self, block="all", op: str = "replace_system", text: str = "", label: str = "") -> str:
        """Recreate the KV cache at chosen KV-capable block(s) using a modified system or user prompt, and let it
        grow gradually as new tokens are generated.

        op: replace_system | append_system | prepend_system | replace_user | append_user | prepend_user
        block: int (single block, e.g. 12), node id ('block_12'), list of blocks ([8, 12]), or slice ('all', '12:24').
        """
        chain = walk_chain(self.flow)
        nid = f"bkv_{next(self._n)}"
        while any(n["id"] == nid for n in self.flow["nodes"]):
            nid = f"bkv_{next(self._n)}"
        targets = []
        layers_str = ""
        if isinstance(block, int):
            targets = [self.node(block)]
        elif isinstance(block, str) and block in {n["id"] for n in chain}:
            targets = [self.node(block)]
        elif isinstance(block, (list, tuple)):
            targets = [self.node(b) for b in block]
        else:
            layers_str = str(block)
        op_labels = {
            "replace_system": "Replace system",
            "append_system": "Append system",
            "prepend_system": "Prepend system",
            "replace_user": "Replace user",
            "append_user": "Append user",
            "prepend_user": "Prepend user",
        }
        anchor = targets[0] if targets else (chain[0] if chain else self.node("embed"))
        self.flow["nodes"].append({
            "id": nid, "type": "box",
            "position": {"x": anchor["position"]["x"] + 8, "y": anchor["position"]["y"] - 118},
            "data": {
                "kind": "block_kv",
                "label": label or op_labels.get(op, "KV override"),
                "op": op,
                "text": text,
                "layers": layers_str,
                "enabled": True,
            },
        })
        for t in targets:
            self.flow["edges"].append({
                "id": f"e-{nid}-out-{t['id']}-kv",
                "source": nid, "target": t["id"],
                "sourceHandle": "out", "targetHandle": "kv",
            })
        return nid

    block_kv = kv_override

    def replace_system(self, text: str, blocks="all", label: str = "") -> str:
        return self.kv_override(blocks, op="replace_system", text=text, label=label)

    def append_system(self, text: str, blocks="all", label: str = "") -> str:
        return self.kv_override(blocks, op="append_system", text=text, label=label)

    def prepend_system(self, text: str, blocks="all", label: str = "") -> str:
        return self.kv_override(blocks, op="prepend_system", text=text, label=label)

    def replace_user(self, text: str, blocks="all", label: str = "") -> str:
        return self.kv_override(blocks, op="replace_user", text=text, label=label)

    def append_user(self, text: str, blocks="all", label: str = "") -> str:
        return self.kv_override(blocks, op="append_user", text=text, label=label)

    def prepend_user(self, text: str, blocks="all", label: str = "") -> str:
        return self.kv_override(blocks, op="prepend_user", text=text, label=label)

    def kv_prefix(self, cache_id: str, layers: str = "all", mode: str = "shift", offset: float = 0, label: str = "") -> str:
        return self._add("kv", "embed", "cache", {"kind": "kv_import", "label": label or "KV prefix", "cache_id": cache_id,
                                                  "layers": str(layers), "mode": mode, "offset": offset})

    # ---------------------------------------------------------------- box code
    def code(self, box, source: str | None):
        """Replace what a box computes with your own Python (None = back to the built-in). See blockcode.py."""
        d = self.node(box)["data"]
        if source is None:
            d.pop("code", None)
        else:
            d["code"] = source
        return self

    def apply_edits(self, edits: list[dict] | None):
        """Apply a list of declarative edit dicts (used by MCP and /api/workspace/edit) to this flow."""
        for raw in edits or []:
            e = dict(raw)
            do = e.pop("do", None)
            blk = e.pop("block", None)
            try:
                if do == "bypass":
                    self.bypass(blk)
                elif do == "schedule":
                    self.schedule(list(e["order"]))
                elif do in ("intervene", "intercept"):
                    self.intercept(blk, **e)
                elif do == "patch":
                    self.patch(blk, e.pop("source_run"), **e)
                elif do == "steer":
                    self.steer(blk, e.pop("vector"), **e)
                elif do == "project_out":
                    self.project_out(blk, e.pop("vector"), **e)
                elif do == "quantize":
                    self.quantize_block(blk, **e)
                elif do == "clamp":
                    self.clamp(blk, e.pop("neurons"), **e)
                elif do == "wire":
                    src_n = e.pop("source", e.pop("source_neurons", None))
                    tgt_n = e.pop("target", e.pop("neurons", None))
                    if "value" in e and "strength" not in e:
                        e["strength"] = e.pop("value")
                    self.wire(blk, src_n, tgt_n, **e)
                elif do in ("kv_override", "block_kv"):
                    self.kv_override(blk if blk is not None else e.pop("blocks", e.pop("layers", "all")), **e)
                elif do in ("replace_system", "append_system", "prepend_system", "replace_user", "append_user", "prepend_user"):
                    getattr(self, do)(e.pop("text", ""), blocks=blk if blk is not None else e.pop("blocks", e.pop("layers", "all")), **e)
                elif do == "code":
                    self.code(blk if blk is not None else e["box"], e["source"])
                elif do == "rope":
                    self.rope(blk if blk is not None else e.pop("blocks", "all"), **e)
                elif do == "rope_step":
                    self.rope_step(**e)
                elif do == "ban_experts":
                    self.ban_experts(blk, e.pop("experts"), **e)
                elif do == "only_experts":
                    self.only_experts(blk, e.pop("experts"), **e)
                elif do == "route":
                    self.route(blk, e.pop("weights"), **e)
                elif do == "shared_expert":
                    self.shared_expert(blk, bool(e.pop("enabled", False)))
                elif do == "fork":
                    self.fork(blk, **e)
                else:
                    raise ValueError(f"Unknown edit '{do}'.")
            except (KeyError, TypeError) as ex:
                raise ValueError(f"Edit {do!r}: {ex}") from None
        return self

