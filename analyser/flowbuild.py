"""FlowBuilder: edit a flow (the boxes-and-arrows graph the UI draws) from Python, in one line per change.

Pure Python, no torch — used by the Python API (analyser.api), by exported run scripts and by the MCP tutor.

    flow = lab.flow()                                   # the model's default flow
    flow.prompt("The capital of France is").sampler(max_new_tokens=8)
    flow.bypass(3)                                      # skip block 3
    flow.intervene(5, hook="mlp_out", op="zero")        # zero what block 5's MLP writes
    flow.patch(7, clean_run, hook="resid_post", positions="last")
    flow.steer(6, vector, value=8)
    flow.clamp(4, neurons="1021", to="max", value=5)
    flow.layer_prompt("Answer in French.", layers="10:14", mode="overwrite")
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
        """A box by id, by kind ('sampler'), or — for an int — the block box of that layer."""
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
    def prompt(self, prompt=None, system=None, prefill=None, chat=None, thinking=None, negative_prompt=None):
        return self._set("input_prompt", prompt=prompt, system=system, prefill=prefill, use_chat_template=chat,
                         thinking=thinking, negative_prompt=negative_prompt)

    def sampler(self, max_new_tokens=None, temperature=None, top_k=None, top_p=None, seed=None):
        return self._set("sampler", max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)

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

    # ---------------------------------------------------------------- interventions
    def intervene(self, block, hook: str = "mlp_out", op: str = "zero", value: float = 0.0, positions: str = "all",
                  heads: str = "all", neurons: str = "all", ref: str = "", src_positions: str = "", clamp: str = "max",
                  seed: int = 0, steps: str = "all", label: str = "") -> str:
        """Wire an intervention box into a block. Returns the new box's id."""
        ref = getattr(ref, "id", ref)             # a RunResult or a vector dict/id
        if isinstance(ref, dict):
            ref = ref["id"]
        return self._add("iv", self.node(block)["id"], "mod", {
            "kind": "intervention", "label": label or f"{op} @ {hook}", "hook": hook, "op": op, "value": value,
            "positions": str(positions), "heads": str(heads), "neurons": str(neurons), "ref": ref or "",
            "src_positions": str(src_positions), "clamp": clamp, "seed": seed, "steps": steps, "enabled": True})

    def patch(self, block, source_run, hook: str = "resid_post", positions: str = "all", src_positions: str = "") -> str:
        """Activation patching: at `block`, overwrite this run's activation with the one `source_run` had there."""
        return self.intervene(block, hook=hook, op="patch", ref=source_run, positions=positions,
                              src_positions=src_positions, label="patch")

    def steer(self, block, vector, value: float = 4.0, hook: str = "resid_post", positions: str = "all") -> str:
        """Add value × a steering vector to the residual stream at `block`."""
        return self.intervene(block, hook=hook, op="steer", ref=vector, value=value, positions=positions, label="steer")

    def clamp(self, block, neurons, to: str = "max", value: float = 1.0, ref: str = "", positions: str = "all",
              hook: str = "mlp_neurons") -> str:
        """Pin MLP neurons of `block` to their max / min (× value) or to a fixed value (to='value')."""
        return self.intervene(block, hook=hook, op="clamp", neurons=str(neurons), clamp=to, value=value, ref=ref,
                              positions=positions, label=f"neuron {neurons} → {to}")

    # ---------------------------------------------------------------- KV prefixes
    def layer_prompt(self, text: str, layers: str = "all", mode: str = "shift", as_system: bool = False,
                     offset: float = 0, label: str = "") -> str:
        """Text that only the chosen blocks can read: it is run through the model once and its keys/values are
        prepended for those blocks. mode: shift | negative | offset | overwrite (replaces the prompt there)."""
        return self._add("kvt", "embed", "cache", {"kind": "kv_text", "label": label or "Layer prompt", "text": text,
                                                   "layers": str(layers), "mode": mode, "as_system": as_system,
                                                   "offset": offset, "enabled": True})

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
