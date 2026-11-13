"""MCP server: lets an LLM (Claude, or any MCP client) use Tokenshark as a teacher would.

The server talks to a running Tokenshark over its local HTTP API, so everything the LLM does — runs,
interventions, patching sweeps — also appears in the student's browser (Runs tab → click a run), where the
student can open the lens, attention and neuron views on it. Start the app first, then point a client at:

    python -m tokenshark.mcp_server                       # app at http://127.0.0.1:8000
    python -m tokenshark.mcp_server --url http://127.0.0.1:8001

Claude Desktop / Claude Code config:
    {"mcpServers": {"tokenshark": {"command": "<path to .venv python>", "args": ["-m", "tokenshark.mcp_server"],
                                   "cwd": "<folder of this project>"}}}

Protocol: MCP over stdio (JSON-RPC 2.0, one JSON object per line). Pure standard library — no torch, no
MCP SDK — so it starts instantly; the model stays in the app's process.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import __version__
from .flowbuild import FlowBuilder

URL = "http://127.0.0.1:8000"
PROTOCOL = "2025-06-18"

GUIDE = """Tokenshark — guide for a tutor

What it is: a glass-box lab for open-weight models. A language model is drawn as a flow of boxes
(prompt → tokenizer → embed → blocks → final norm → LM head → sampler). You can run it, look inside every block,
and change what happens. The student sees every run you make in the browser (Runs tab) and can open the same
views on it, so say what to click: "open the Logit lens tab on run abc123".

Teach by experiment, not by telling. A good step is: predict → run → look → explain.
  1. Ask the student what they expect.            2. Run it (run_prompt).
  3. Show one piece of evidence (logit_lens, attention, top_neurons, patch_sweep).
  4. Ask what it means before you say it. Keep prompts short (5–12 tokens): everything is faster and clearer.

Lessons that work (in this order for newcomers)
  Tokens          tokenize a sentence, an unusual word, a number. Why does cost depend on tokens, not words?
  Next-token      run_prompt, then next_token_distribution. Change temperature / top_p: what moves?
  Layers          logit_lens at the last token: at which block does the answer first appear?
  Attention       attention for a block: which earlier token does the last token read? Try a pronoun.
  Ablation        run_prompt with edits [{"do":"bypass","block":3}] or zero one block's MLP. What breaks, what doesn't?
  Patching        two prompts that differ in one word ("…France is" / "…Japan is", same token count).
                  run both, then patch_sweep(prompt=<corrupted>, source_run=<clean run>): the hot cells are
                  where the model keeps the fact. Then confirm with one patch edit.
  Steering        make_vector_from_runs from two contrasting prompts (or make_vector_from_word), then an edit
                  {"do":"steer","block":6,"vector":<id>,"value":4}. Raise value until the text bends; too much breaks it.
  Neurons         top_neurons on a block → pick one with readable "promotes" words → clamp it:
                  {"do":"clamp","block":4,"neurons":"1021","to":"max","value":5}.
  Layer prompts   {"do":"layer_prompt","text":"Answer in French.","layers":"8:12","mode":"shift"}: text only some
                  blocks can read. mode overwrite replaces the prompt in those blocks.
  Quantization    quant_compare(prompt, bits=[8,4,2]): where does the text start to change?
  Code            block_code shows a block as ~15 lines of Python; an edit {"do":"code","block":2,"source":…} runs
                  the changed lines (e.g. remove `x = x + mlp_out`).

Edits for run_prompt (list of objects, applied in order to the default flow)
  {"do":"bypass","block":3}                      {"do":"schedule","order":[0,1,1,2]}
  {"do":"intervene","block":5,"hook":"mlp_out","op":"zero"}        ops: zero, scale, add_noise, mean
        hooks: resid_pre, attn_heads, attn_out, mlp_neurons, mlp_out, resid_post · optional value, positions, heads, neurons
  {"do":"patch","block":7,"source_run":"<id>","hook":"resid_post","positions":"last"}
  {"do":"steer","block":6,"vector":"<id>","value":4}
  {"do":"clamp","block":4,"neurons":"12,40","to":"max"|"min"|"value","value":5}
  {"do":"layer_prompt","text":"…","layers":"all|none|0:12|3,5","mode":"shift|negative|offset|overwrite"}
  {"do":"code","block":2,"source":"…python…"}   (or "box":"sampler"|"embed"|"final_norm"|"lm_head")
"block" is the layer number. positions: all | last | prompt | generated | 3 | 2:5.

Honest limits to mention when relevant: the logit lens reads early layers poorly (use through>0);
attention shows where information moves, not why; a neuron's "promotes" list is only its direct effect;
small models make mistakes that large ones don't. Say "this model" rather than "AI".
"""


# --------------------------------------------------------------------------------------
# HTTP client for the app
# --------------------------------------------------------------------------------------

class AppError(RuntimeError):
    pass


def call(method: str, path: str, body=None, timeout: float = 120, raw: bool = False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        try:
            d = json.loads(text)
            msg = d.get("detail") or " · ".join(x["message"] for x in d.get("errors", [])) or text
        except Exception:  # noqa: BLE001
            msg = text
        raise AppError(str(msg)) from None
    except urllib.error.URLError as e:
        raise AppError(f"Tokenshark is not reachable at {URL} ({e.reason}). Start it (run.bat / run.sh) and try again.") from None
    return text if raw else (json.loads(text) if text else None)


def q(**kw) -> str:
    return urllib.parse.urlencode({k: v for k, v in kw.items() if v is not None})


def wait_run(run_id: str, timeout: float) -> dict:
    t0 = time.time()
    while True:
        r = call("GET", f"/api/runs/{run_id}")
        if r["status"] not in ("queued", "running", "paused") or time.time() - t0 > timeout:
            return r
        time.sleep(0.4)


def wait_task(start: dict, timeout: float) -> dict:
    """Follow a background task to its end; returns {"status", "result"}."""
    if not start.get("task"):
        return {"status": "done", "result": start.get("result")}
    t0 = time.time()
    while True:
        t = call("GET", f"/api/tasks/{start['task']['id']}")
        if t["status"] in ("done", "error", "cancelled") or time.time() - t0 > timeout:
            if t["status"] == "error":
                raise AppError(t.get("error") or "the task failed")
            return {"status": t["status"], "result": t.get("result"), "task_id": start["task"]["id"]}
        time.sleep(0.5)


def build_flow(prompt=None, system=None, prefill=None, chat=None, max_new_tokens=None, temperature=None,
               top_k=None, top_p=None, seed=None, edits=None) -> dict:
    fb = FlowBuilder(call("GET", "/api/flow/default"))
    if any(v is not None for v in (prompt, system, prefill, chat)):
        fb.prompt(prompt=prompt, system=system, prefill=prefill, chat=chat)
    if any(v is not None for v in (max_new_tokens, temperature, top_k, top_p, seed)):
        fb.sampler(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
    for e in edits or []:
        e = dict(e)
        do = e.pop("do", None)
        blk = e.pop("block", None)
        try:
            if do == "bypass":
                fb.bypass(blk)
            elif do == "schedule":
                fb.schedule(list(e["order"]))
            elif do == "intervene":
                fb.intervene(blk, **e)
            elif do == "patch":
                fb.patch(blk, e.pop("source_run"), **e)
            elif do == "steer":
                fb.steer(blk, e.pop("vector"), **e)
            elif do == "clamp":
                fb.clamp(blk, e.pop("neurons"), **e)
            elif do == "layer_prompt":
                fb.layer_prompt(e.pop("text"), **e)
            elif do == "code":
                fb.code(blk if blk is not None else e["box"], e["source"])
            else:
                raise AppError(f"Unknown edit '{do}'. See teaching_guide for the list.")
        except (KeyError, TypeError) as ex:
            raise AppError(f"Edit {do!r}: {ex}. See teaching_guide for the fields of each edit.") from None
    return fb.to_dict()


def show_run(r: dict) -> dict:
    o = r.get("outputs") or {}
    out = {"run_id": r["id"], "status": r["status"], "label": r.get("label")}
    if r.get("error"):
        out["error"] = r["error"]
    if "text" in o:
        gen = next((s for s in o.get("segments", []) if s["origin"] == "generated"), {"start": len(o.get("tokens", []))})
        out.update(text=o["text"], n_prompt_tokens=gen["start"],
                   tokens=[f"{i}:{t}" for i, t in enumerate(o.get("tokens", []))])
        alts = o.get("alternatives") or []
        if 0 < gen["start"] <= len(alts):
            out["first_token_choices"] = [[p, pr] for p, _, pr in alts[gen["start"] - 1][:5]]
        if o.get("prints"):
            out["code_prints"] = o["prints"][:20]
    elif "image" in o:
        out["image"] = "an image was generated; the student can see it in the Output tab"
    out["hint"] = f"The student can open this run in the browser: Runs tab → {r['id']}."
    return out


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------

def t_status():
    s = call("GET", "/api/status")
    m = s.get("model") or {}
    return {"app_version": s["version"], "device": s["device"], "status": s["status"], "error": s.get("error"),
            "model": {k: m.get(k) for k in ("model_id", "family", "arch", "n_layers", "d_model", "n_heads", "vocab", "d_ff")} if m else None,
            "lora": (m.get("lora") or {}).get("name"), "quant_bits": (m.get("quant") or {}).get("bits")}


def t_list_models():
    return [{k: m.get(k) for k in ("id", "family", "size_gb", "downloaded", "loaded", "notes")} for m in call("GET", "/api/models")]


def t_load_model(model_id: str, policy: str = "auto", timeout: float = 600):
    call("POST", "/api/load", {"model_id": model_id, "policy": policy})
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = call("GET", "/api/status")
        if s["status"] == "error":
            raise AppError(s.get("error") or "load failed")
        if s["status"] == "ready" and (s.get("model") or {}).get("model_id") == model_id:
            return t_status()
        time.sleep(1)
    return {"status": "still loading (large models download first) — call status again in a while"}


def t_run_prompt(prompt: str, system: str | None = None, prefill: str | None = None, chat: bool | None = None,
                 max_new_tokens: int = 12, temperature: float | None = None, top_k: int | None = None,
                 top_p: float | None = None, seed: int | None = None, edits: list | None = None, label: str = "",
                 timeout: float = 300):
    flow = build_flow(prompt, system, prefill, chat, max_new_tokens, temperature, top_k, top_p, seed, edits)
    r = call("POST", "/api/runs", {"flow": flow, "label": label or "tutor"})
    return show_run(wait_run(r["run"]["id"], timeout))


def t_get_run(run_id: str):
    return show_run(call("GET", f"/api/runs/{run_id}"))


def t_list_runs():
    return [{k: r.get(k) for k in ("id", "label", "status", "preview", "has_captures")} for r in call("GET", "/api/runs")][-30:]


def t_logit_lens(run_id: str, position: int = -1, k: int = 3, through: int = 0):
    L = call("GET", f"/api/runs/{run_id}/lens?{q(k=max(k, 1), through=through)}", timeout=600)
    T = len(L["tokens"])
    t = position % T if -T <= position < T else T - 1
    rows = []
    for lay, cells in zip(L["layers"], L["cells"]):
        c = cells[t] if cells and t < len(cells) else None
        if c:
            rows.append({"layer": lay["label"], "top": [[w, p] for w, p in c["top"][:k]], "p_of_final_choice": c["p_final"]})
    return {"position": t, "token": L["tokens"][t], "read_as": "what the model would predict next if it stopped after each layer",
            "through_last_blocks": through, "layers": rows}


def t_attention(run_id: str, block: int, head: int = -1, token: int = -1, top: int = 6):
    a = call("GET", f"/api/runs/{run_id}/attention?{q(pos=block, head=head)}")
    if a.get("error"):
        raise AppError(a["error"])
    m, qt, kt = a["matrix"], a["query_tokens"], a["key_tokens"]
    t = token % len(m) if -len(m) <= token < len(m) else len(m) - 1
    row = sorted(enumerate(m[t]), key=lambda x: -x[1])[:top]
    return {"block": block, "head": "mean of all heads" if head < 0 else head, "n_heads": a["n_heads"],
            "query_token": f"{t}:{qt[t]}", "attends_to": [[f"{j}:{kt[j]}", w] for j, w in row]}


def t_track_word(run_id: str, word: str, position: int = -1):
    r = call("GET", f"/api/runs/{run_id}/track?{q(word=word, position=position)}", timeout=600)
    return {"position": r["position"], "token": r["token"], "tracked": r["target"], "model_chose": r["choice"],
            "first_layer_where_it_is_top": r["first_top_layer"],
            "layers": [{"layer": x["label"], "p": x["target_p"], "rank": x["target_rank"], "push": x["push"]} for x in r["rows"]],
            "push_means": "how much that block's write raises the tracked word's score against the model's choice"}


def t_top_neurons(run_id: str, block: int, token: int = -1, k: int = 10, by: str = "token"):
    return call("GET", f"/api/runs/{run_id}/neurons?{q(pos=block, token=token, k=k, by=by)}")


def t_neuron(run_id: str, block: int, index: int):
    d = call("GET", f"/api/runs/{run_id}/neuron?{q(pos=block, index=index)}")
    d["acts"] = [[f"{i}:{t}", a] for i, (t, a) in enumerate(zip(d.pop("tokens"), d["acts"]))]
    return d


def t_tokenize(text: str, chat: bool = False):
    r = call("POST", "/api/tokenize", {"text": text, "chat": chat})
    return {"n_tokens": r["n_tokens"], "n_chars": r["n_chars"], "chars_per_token": r["chars_per_token"],
            "tokens": [[p, i] for p, i in zip(r["pieces"], r["ids"])], "vocab_size": r["vocab"]}


def t_token_neighbors(word: str, k: int = 10):
    r = call("POST", "/api/tokenize", {"text": word})
    if not r["ids"]:
        raise AppError("Empty word.")
    out = call("GET", f"/api/tokens/{r['ids'][0]}/neighbors?{q(k=k)}")
    if len(r["ids"]) > 1:
        out["note"] = f"'{word}' is {len(r['ids'])} tokens; neighbours are for its first token."
    return out


def t_next_token_distribution(run_id: str, position: int = -1, temperature: float = 1.0, top: int = 12):
    import math
    r = call("GET", f"/api/runs/{run_id}/logits?{q(position=position, top=1000)}")
    T = max(temperature, 1e-6)
    mx = r["rows"][0][2]
    w = [math.exp((x[2] - mx) / T) for x in r["rows"]]
    z = sum(w)
    ent = -sum((v / z) * math.log(v / z) for v in w if v > 0)
    return {"position": r["position"], "after_token": r["token"], "temperature": temperature,
            "entropy_bits": round(ent / math.log(2), 3), "actual_next": r["next_piece"],
            "top": [[x[0], round(v / z, 4)] for x, v in list(zip(r["rows"], w))[:top]],
            "note": "computed over the 1000 highest-scoring tokens"}


def t_make_vector_from_runs(run_a: str, run_b: str, block: int, hook: str = "resid_post", token: str = "last", name: str = ""):
    return call("POST", "/api/vectors/from_runs", {"run_a": run_a, "run_b": run_b, "pos": block, "hook": hook, "token": str(token), "name": name})


def t_make_vector_from_word(word: str, minus: str = "", name: str = ""):
    return call("POST", "/api/vectors/from_word", {"word": word, "minus": minus, "name": name})


def t_list_vectors():
    return call("GET", "/api/vectors")


def t_patch_sweep(prompt: str, source_run: str, hook: str = "resid_post", tokens: str = "all", system: str | None = None,
                  chat: bool | None = None, timeout: float = 600):
    flow = build_flow(prompt, system, None, chat)
    w = wait_task(call("POST", "/api/patch/sweep", {"flow": flow, "source_run": source_run, "hook": hook, "tokens": tokens}), timeout)
    r = w["result"] or {}
    cells = [(v, b, t) for b, row in enumerate(r.get("grid", [])) for t, v in enumerate(row) if v is not None]
    best = sorted(cells, reverse=True)[:8]
    return {"status": w["status"], "clean_answer": r.get("answer"), "corrupted_answer": r.get("corrupt_answer"),
            "p_clean": r.get("p_clean"), "p_corrupt": r.get("p_corrupt"), "same_length": r.get("same_length"),
            "tokens": [f"{i}:{t}" for i, t in zip(r.get("token_index", []), r.get("tokens", []))],
            "strongest": [{"block": b, "token": f"{r['token_index'][t]}:{r['tokens'][t]}", "recovery": v} for v, b, t in best],
            "grid_rows_are_blocks_columns_are_tokens": r.get("grid"),
            "read_as": "recovery 1 = copying that one activation from the clean run restores the clean answer; 0 = no effect"}


def t_quantize(bits: int = 0):
    return call("DELETE", "/api/quant") if not bits else call("POST", "/api/quant", {"bits": bits}, timeout=600)


def t_quant_compare(prompt: str, bits: list | None = None, max_new_tokens: int = 16, timeout: float = 900):
    flow = build_flow(prompt, max_new_tokens=max_new_tokens)
    w = wait_task(call("POST", "/api/quant/compare", {"flow": flow, "bits": bits or [8, 4]}), timeout)
    rows = (w["result"] or {}).get("rows", [])
    return {"status": w["status"], "rows": [{k: r.get(k) for k in ("label", "text", "kl", "agree", "same_tokens")} for r in rows],
            "read_as": "kl = how far the next-token distributions drift from full precision (0 = identical); "
                       "agree = share of positions with the same top choice; same_tokens = tokens before the text first differs"}


def t_block_code(layer: int = 0, kind: str = "block"):
    d = call("GET", f"/api/code/default?{q(kind=kind, layer=layer)}")
    return {"code": d["code"], "editable": d["editable"], "reason": d["reason"]}


def t_export_script(run_id: str):
    return call("GET", f"/api/runs/{run_id}/script", raw=True)


def t_teaching_guide():
    return GUIDE


def _tool(fn, desc: str, props: dict, required: list | None = None) -> dict:
    return {"fn": fn, "description": desc,
            "inputSchema": {"type": "object", "properties": props, "required": required or []}}


S, I, N, Bo = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
RUN = {"run_id": {**S, "description": "id of a finished run (from run_prompt or list_runs)"}}
BLOCK = {"block": {**I, "description": "schedule position of the block (= layer number in an unedited flow)"}}
TOK = {**I, "description": "token index in the run (negative counts from the end; -1 = last)"}

TOOLS = {
    "teaching_guide": _tool(t_teaching_guide, "Read this first: how to teach with Tokenshark, lesson sequences, and the edit format for run_prompt.", {}),
    "status": _tool(t_status, "Is the app running, which model is loaded (layers, width, heads), LoRA / quantization in force.", {}),
    "list_models": _tool(t_list_models, "Models the app can load (id, size, whether already downloaded).", {}),
    "load_model": _tool(t_load_model, "Load a model in the app (downloads it on first use). Small models: qwen2.5-0.5b-instruct, smollm2-360m-instruct.",
                        {"model_id": S, "policy": {**S, "enum": ["auto", "device", "cpu", "disk"]}}, ["model_id"]),
    "run_prompt": _tool(t_run_prompt, "Run the loaded language model on a prompt, optionally with edits to the flow (bypass a block, "
                        "zero an MLP, patch, steer, clamp a neuron, layer prompt, custom block code). Returns the text, the tokens "
                        "with their indices, and the top choices for the first generated token. The run appears in the student's Runs tab.",
                        {"prompt": S, "system": S, "prefill": {**S, "description": "forced start of the answer"},
                         "chat": {**Bo, "description": "wrap in the chat template (default: the model's own setting)"},
                         "max_new_tokens": I, "temperature": {**N, "description": "0 = greedy"}, "top_k": I, "top_p": N, "seed": I,
                         "edits": {"type": "array", "items": {"type": "object"}, "description": "see teaching_guide for the edit objects"},
                         "label": {**S, "description": "short name shown in the Runs tab"}}, ["prompt"]),
    "get_run": _tool(t_get_run, "Text, tokens and status of a run.", RUN, ["run_id"]),
    "list_runs": _tool(t_list_runs, "Recent runs in the app (the student's and yours).", {}),
    "logit_lens": _tool(t_logit_lens, "What the model would predict after each layer, at one token position. Shows where an answer forms.",
                        {**RUN, "position": TOK, "k": I, "through": {**I, "description": "decode through the last N blocks instead of directly (0 = plain logit lens)"}}, ["run_id"]),
    "attention": _tool(t_attention, "Which earlier tokens one token attends to in a block (one head, or the mean of all heads).",
                       {**RUN, **BLOCK, "head": {**I, "description": "-1 = mean over heads"}, "token": TOK, "top": I}, ["run_id", "block"]),
    "track_word": _tool(t_track_word, "Follow one expected word through all layers: its probability and rank, and how much each block pushed for it.",
                        {**RUN, "word": S, "position": TOK}, ["run_id", "word"]),
    "top_neurons": _tool(t_top_neurons, "Strongest MLP neurons of a block, with the words each one promotes / suppresses. by='token': at one token; by='max': peak over the run.",
                         {**RUN, **BLOCK, "token": TOK, "k": I, "by": {**S, "enum": ["token", "max"]}}, ["run_id", "block"]),
    "neuron": _tool(t_neuron, "One neuron: activation at every token of the run, its range, promoted / suppressed words.",
                    {**RUN, **BLOCK, "index": I}, ["run_id", "block", "index"]),
    "tokenize": _tool(t_tokenize, "Split text into the loaded model's tokens.", {"text": S, "chat": Bo}, ["text"]),
    "token_neighbors": _tool(t_token_neighbors, "Tokens whose embedding is closest to a word's.", {"word": S, "k": I}, ["word"]),
    "next_token_distribution": _tool(t_next_token_distribution, "Probabilities of the next token at a position of a run, at any temperature, with the entropy.",
                                     {**RUN, "position": TOK, "temperature": N, "top": I}, ["run_id"]),
    "make_vector_from_runs": _tool(t_make_vector_from_runs, "Steering vector = activation of run A minus run B at one block (two contrasting prompts). Use its id in a steer edit.",
                                   {"run_a": S, "run_b": S, **BLOCK, "hook": {**S, "enum": ["resid_pre", "attn_out", "mlp_out", "resid_post"]},
                                    "token": {**S, "description": "last | mean | a token index"}, "name": S}, ["run_a", "run_b", "block"]),
    "make_vector_from_word": _tool(t_make_vector_from_word, "Steering vector = the output direction of a word (optionally minus another word), length 1; use value ≈ the residual norm.",
                                   {"word": S, "minus": S, "name": S}, ["word"]),
    "list_vectors": _tool(t_list_vectors, "Saved steering vectors.", {}),
    "patch_sweep": _tool(t_patch_sweep, "Activation patching over every (block, token): run `prompt` (the corrupted one) while copying one activation at a time "
                         "from `source_run` (the clean run). Shows where the model keeps what differs between the two prompts. Use prompts with the same token count.",
                         {"prompt": S, "source_run": S, "hook": {**S, "enum": ["resid_pre", "attn_out", "mlp_out", "resid_post"]},
                          "tokens": {**S, "description": "all | last | 2:5"}, "system": S, "chat": Bo}, ["prompt", "source_run"]),
    "quantize": _tool(t_quantize, "Round the model's block weights to N bits for all later runs (0 = back to full precision).", {"bits": I}),
    "quant_compare": _tool(t_quant_compare, "Generate from a prompt at full precision and at each bit width; reports the texts and how far the predictions drift.",
                           {"prompt": S, "bits": {"type": "array", "items": I}, "max_new_tokens": I}, ["prompt"]),
    "block_code": _tool(t_block_code, "The simple Python a box runs (kind: block | embed | final_norm | lm_head | sampler). Edit it and pass it in a code edit.",
                        {"layer": I, "kind": S}),
    "export_script": _tool(t_export_script, "A run as a Python script (tokenshark.api) that reproduces it — for moving a lab result into a notebook.", RUN, ["run_id"]),
}


# --------------------------------------------------------------------------------------
# JSON-RPC over stdio
# --------------------------------------------------------------------------------------

def handle(msg: dict) -> dict | None:
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:                                    # a notification: nothing to answer
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        return ok({"protocolVersion": params.get("protocolVersion") or PROTOCOL, "capabilities": {"tools": {}},
                   "serverInfo": {"name": "tokenshark", "version": __version__},
                   "instructions": "Tools for teaching how language models work, by experiment. Call teaching_guide first."})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": [{"name": n, "description": t["description"], "inputSchema": t["inputSchema"]} for n, t in TOOLS.items()]})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            if name not in TOOLS:
                raise AppError(f"Unknown tool '{name}'.")
            res = TOOLS[name]["fn"](**args)
            text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, indent=1)
            return ok({"content": [{"type": "text", "text": text}], "isError": False})
        except TypeError as e:
            return ok({"content": [{"type": "text", "text": f"Bad arguments for {name}: {e}"}], "isError": True})
        except AppError as e:
            return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})
        except Exception as e:  # noqa: BLE001
            return ok({"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main():
    global URL
    ap = argparse.ArgumentParser(description="MCP server for Tokenshark (stdio)")
    ap.add_argument("--url", default=URL, help="where the app is running (default http://127.0.0.1:8000)")
    URL = ap.parse_args().url.rstrip("/")
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = handle(json.loads(line))
        except json.JSONDecodeError:
            out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if out is not None:
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
