"""Lab tools: activation patching, steering, neuron clamps, layer prompts, editable box code, quantization,
tokenizer / sampler helpers, the Python API, run → script export, and training."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from analyser import blockcode, interp                                                 # noqa: E402
from analyser.api import Lab                                                           # noqa: E402
from analyser.flow import FlowError, compile_flow                                      # noqa: E402
from analyser.flowbuild import FlowBuilder, walk_chain                                 # noqa: E402
from analyser.quant import iter_compare, quantize                                      # noqa: E402
from tests.make_tiny import make_llm                                                   # noqa: E402

torch.set_grad_enabled(False)

A, B = "The capital of France is", "The capital of Japan is"


@pytest.fixture(scope="module")
def lab(tmp_path_factory, monkeypatch_module):
    home = tmp_path_factory.mktemp("home")
    monkeypatch_module.setattr(interp, "VEC_DIR", home / "vectors")
    (home / "vectors").mkdir()
    d = tmp_path_factory.mktemp("llm") / "tiny"
    make_llm(d, "qwen2")
    return Lab(str(d), device="cpu", dtype="fp32", policy="device")


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _logits(run):
    return run.logits(run.captures["n_in"] - 1)


# ---------------------------------------------------------------- patching
def test_patch_everything_restores_the_clean_run(lab):
    """Copying the whole residual stream after the last block from run A into run B gives A's prediction."""
    a, b = lab.run(A, max_new_tokens=1), lab.run(B, max_new_tokens=1)
    assert len(a.token_ids) == len(b.token_ids)
    assert not torch.allclose(_logits(a), _logits(b))
    S = len(a.plan.schedule)
    f = lab.flow(B, max_new_tokens=1)
    f.patch(S - 1, a, hook="resid_post", positions="prompt")
    p = lab.run(f)
    assert torch.allclose(_logits(p), _logits(a), atol=1e-4)
    # patching only the first token of the first block is a smaller change than the whole stream
    f = lab.flow(B, max_new_tokens=1)
    f.patch(0, a, hook="resid_pre", positions="0")
    assert torch.allclose(_logits(lab.run(f)), _logits(b), atol=1e-4)       # token 0 is the same word in both


def test_patch_parts_and_missing_source(lab):
    a = lab.run(A, max_new_tokens=1)
    assert a.part(1, "mlp_out").shape == a.resid(1).shape
    f = lab.flow(B, max_new_tokens=1)
    f.patch(1, a, hook="mlp_out")
    assert not torch.allclose(_logits(lab.run(f)), _logits(lab.run(B, max_new_tokens=1)))
    f = lab.flow(B)
    f.patch(1, "nope")
    with pytest.raises(ValueError, match="not in memory"):
        lab.run(f)
    f = lab.flow(B)
    f.intervene(1, hook="attn_heads", op="patch", ref=a)
    with pytest.raises(FlowError):
        lab.plan(f)


def test_patch_sweep(lab):
    a = lab.run(A, max_new_tokens=1)
    plan = lab.plan(lab.flow(B))
    out = None
    for out in interp.iter_patch_sweep(lab.adapter, plan, a.captures, "resid_post", "all"):
        pass
    S, T = len(plan.schedule), a.captures["n_in"]
    assert len(out["grid"]) == S and len(out["grid"][0]) == T and out["same_length"]
    assert abs(out["grid"][S - 1][T - 1] - 1.0) < 1e-3      # last block, last token: the answer itself
    assert all(v is not None for row in out["grid"] for v in row)


# ---------------------------------------------------------------- steering
def test_steering_vectors(lab):
    a, b = lab.run(A, max_new_tokens=1), lab.run(B, max_new_tokens=1)
    v = lab.vector_from_runs(a, b, block=1)
    assert v["dim"] == lab.info["d_model"] and v["norm"] > 0
    # adding (A − B) at the last token of B after the same block turns B's stream into A's there
    f = lab.flow(B, max_new_tokens=1)
    f.steer(1, v, value=1.0, positions="last")
    s = lab.run(f)
    n = s.captures["n_in"] - 1
    assert torch.allclose(s.resid(1)[n], a.resid(1)[n], atol=1e-4)
    w = lab.vector_from_word("France")
    assert abs(float(lab.load_vector(w).norm()) - 1.0) < 1e-4
    f = lab.flow(B, max_new_tokens=1)
    f.steer(2, w, value=50.0)
    tid = lab.adapter._word_ids("France")[0][0]
    assert _logits(lab.run(f))[tid] > _logits(b)[tid]
    f = lab.flow(B)
    f.intervene(1, hook="mlp_neurons", op="steer", ref=w)
    with pytest.raises(FlowError):
        lab.plan(f)


# ---------------------------------------------------------------- neurons
def test_neuron_capture_and_clamp(lab):
    r = lab.run(A, max_new_tokens=2)
    acts = r.neurons(1)
    assert acts.shape == (len(r.captures["ids"]), lab.adapter.cfg.inter)
    top = r.top_neurons(1, token=-1, k=5)
    assert len(top["neurons"]) == 5 and top["neurons"][0]["promotes"]
    i = top["neurons"][0]["index"]
    d = r.neuron(1, i)
    assert len(d["acts"]) == acts.shape[0] and d["max"] >= max(d["acts"]) - 1e-3
    # clamp to a value: the neuron reads exactly that everywhere
    f = lab.flow(A, max_new_tokens=2)
    f.clamp(1, i, to="value", value=3.0)
    c = lab.run(f)
    assert torch.allclose(c.neurons(1)[:, i], torch.full((acts.shape[0],), 3.0), atol=1e-2)
    others = [j for j in range(acts.shape[1]) if j != i]
    assert torch.allclose(c.neurons(1)[: r.captures["n_in"], others], acts[: r.captures["n_in"], others], atol=1e-2)
    # clamp to the max it reached in a reference run
    f = lab.flow(A, max_new_tokens=2)
    f.clamp(1, i, to="max", ref=r)
    m = lab.run(f)
    assert torch.allclose(m.neurons(1)[:, i], torch.full((acts.shape[0],), float(acts[:, i].max())), atol=1e-2)
    # any op can be limited to chosen units
    f = lab.flow(A, max_new_tokens=2)
    f.intervene(1, hook="resid_post", op="zero", neurons="0:4")
    z = lab.run(f)
    assert float(z.resid(1)[:, :4].abs().max()) == 0 and float(z.resid(1)[:, 4:].abs().max()) > 0


# ---------------------------------------------------------------- layer prompts
def test_layer_prompt_modes(lab):
    base = lab.run(B, max_new_tokens=3)
    # a layer prompt nobody sees changes nothing
    f = lab.flow(B, max_new_tokens=3)
    f.layer_prompt("Paris Paris Paris", layers="none", mode="negative")
    assert lab.run(f).token_ids == base.token_ids
    # visible to all blocks with mode shift == the saved-cache route
    f = lab.flow(B, max_new_tokens=3)
    f.layer_prompt("Paris Paris Paris", layers="all", mode="shift")
    r_all = lab.run(f)
    assert r_all.outputs["prefix_tokens"] > 0
    assert not torch.allclose(_logits(r_all), _logits(base))
    # visible to one block only: that block's attention sees the prefix, the others don't
    f = lab.flow(B, max_new_tokens=1)
    f.layer_prompt("Paris Paris Paris", layers="1", mode="negative")
    r1 = lab.run(f)
    n_pre = r1.outputs["prefix_tokens"]
    assert float(r1.attention(1)[:, :, :n_pre].sum()) > 0
    assert float(r1.attention(0)[:, :, :n_pre].sum()) == 0 and float(r1.attention(2)[:, :, :n_pre].sum()) == 0
    # overwrite: in that block the other prompt tokens are hidden; a token still sees itself and the prefix
    f = lab.flow(B, max_new_tokens=1)
    f.layer_prompt("Paris Paris Paris", layers="1", mode="overwrite")
    ro = lab.run(f)
    att = ro.attention(1).sum(0)                      # [tokens, prefix + tokens]
    T = att.shape[0]
    own = att[:, n_pre:n_pre + T]
    assert float((own - torch.diag(torch.diag(own))).abs().max()) == 0
    assert float(ro.attention(0)[:, :, n_pre:].sum()) > 0
    f = lab.flow(B)
    f.layer_prompt("", layers="1")
    with pytest.raises(FlowError):
        lab.plan(f)


# ---------------------------------------------------------------- editable box code
def test_default_code_equals_the_engine(lab):
    base = lab.run(A, max_new_tokens=4)
    f = lab.flow(A, max_new_tokens=4)
    for l in range(lab.info["n_layers"]):
        f.code(l, lab.block_code(l))
    for kind in ("embed", "final_norm", "lm_head", "sampler"):
        f.code(kind, lab.block_code(kind=kind))
    r = lab.run(f)
    assert r.token_ids == base.token_ids
    assert torch.allclose(_logits(r), _logits(base), atol=1e-4)
    assert torch.allclose(r.attention(1), base.attention(1), atol=1e-3)
    assert torch.allclose(r.neurons(1), base.neurons(1), atol=1e-2)
    assert r.lens()["cells"] == base.lens()["cells"]


def test_edited_code_runs_and_is_checked(lab):
    base = lab.run(A, max_new_tokens=1)
    src = lab.block_code(1)
    f = lab.flow(A, max_new_tokens=1)
    f.code(1, src.replace("x = x + mlp_out", "x = x + 0 * mlp_out\nprint('mlp off', mlp_out)"))
    r = lab.run(f)
    g = lab.flow(A, max_new_tokens=1)
    g.intervene(1, hook="mlp_out", op="zero")
    assert torch.allclose(_logits(r), _logits(lab.run(g)), atol=1e-4)
    assert not torch.allclose(_logits(r), _logits(base))
    assert r.outputs["prints"] and "mlp off" in r.outputs["prints"][0]
    f = lab.flow(A, max_new_tokens=2)
    f.code("sampler", "token = 5")
    assert lab.run(f).token_ids[-2:] == [5, 5]
    for bad in ("import os", "x = x.__class__", "open('f')", "while True:\n    pass", "x.numpy()"):
        assert blockcode.check(bad), bad
    f = lab.flow(A)
    f.code(1, "x = x[:1]")
    with pytest.raises(ValueError, match="tokens × d_model"):
        lab.run(f)
    f = lab.flow(A)
    f.code(1, "y = undefined_name")
    with pytest.raises(ValueError, match="line 1"):
        lab.run(f)
    f = lab.flow(A)
    f.code(1, src)
    f.intervene(1, hook="mlp_out", op="zero")
    with pytest.raises(FlowError, match="edited code"):
        lab.plan(f)


# ---------------------------------------------------------------- quantization
def test_quantization(lab):
    w = torch.randn(8, 16)
    assert torch.allclose(quantize(w, 16), w, atol=1e-3)
    assert len(torch.unique(quantize(w, 2)[0])) <= 4
    e8, e4 = (quantize(w, 8) - w).norm(), (quantize(w, 4) - w).norm()
    assert e8 < e4
    base = lab.run(A, max_new_tokens=1)
    s = lab.quantize(3, measure=True)
    assert s["layers"] and s["n_weights"] > 0 and s["layers"][0]["relative_error"] > 0
    q = lab.run(A, max_new_tokens=1)
    lab.quantize(None)
    assert not torch.allclose(_logits(q), _logits(base))
    assert torch.allclose(_logits(lab.run(A, max_new_tokens=1)), _logits(base))


def test_quant_compare(lab):
    class Loaded:                      # the slice of registry.Loaded that iter_compare uses
        adapter, quant = lab.adapter, None

        def set_quant(self, bits, measure=True):
            lab.quantize(bits)
            self.quant = lab._quant

    out = None
    for out in iter_compare(Loaded(), lab.plan(lab.flow(A, max_new_tokens=4)), [8, 2]):
        pass
    rows = out["rows"]
    assert [r["bits"] for r in rows] == [None, 8, 2]
    assert rows[1]["kl"] <= rows[2]["kl"] and rows[2]["kl"] > 0
    assert lab._quant is None


# ---------------------------------------------------------------- tokenizer / sampler helpers
def test_tokenizer_and_sampler_helpers(lab):
    t = lab.tokenize("The capital of France")
    assert t["n_tokens"] == len(t["ids"]) == len(t["pieces"]) and "".join(t["pieces"]) == "The capital of France"
    s = interp.vocab_search(lab.adapter, "cap")
    assert s["n"] >= 1 and all("cap" in p for p, _ in s["tokens"])
    nb = lab.neighbors(t["ids"][0], k=5)
    assert len(nb["neighbors"]) == 5 and nb["neighbors"][0][2] <= 1.0001
    r = lab.run(A, max_new_tokens=2)
    lg = interp.final_logits(lab.adapter, r.captures, r.plan.schedule, r.captures["n_in"] - 1, top=50)
    assert lg["rows"][0][1] == r.token_ids[r.captures["n_in"]] and len(lg["rows"]) == 50
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    assert blockcode.keep_top_k(p, 2).tolist() == pytest.approx([0.625, 0.375, 0, 0])
    assert blockcode.keep_top_p(p, 0.7).tolist() == pytest.approx([0.625, 0.375, 0, 0])
    f = lab.flow(A, max_new_tokens=6, temperature=1.0, top_p=1e-6, seed=3)       # top-p → 0 keeps one token: greedy
    assert lab.run(f).token_ids == lab.run(A, max_new_tokens=6).token_ids


# ---------------------------------------------------------------- flow builder + export
def test_flowbuilder_schedule(lab):
    f = lab.flow(A)
    f.schedule([0, 1, 1, 2])
    plan = lab.plan(f)
    assert plan.schedule == [0, 1, 1, 2] and plan.schedule_nodes == ["block_0", "block_1", "block_1_r1", "block_2"]
    f.bypass("block_1_r1")
    assert lab.plan(f).schedule == [0, 1, 2]
    f = lab.flow(A)
    f.schedule([2, 0])
    assert [n["id"] for n in walk_chain(f.to_dict())] == ["block_2", "block_0"]


def test_export_script_reproduces_the_run(lab, tmp_path):
    from analyser.export import run_script
    from analyser.flow import default_llm_flow

    class R:                                           # the slice of runs.Run the exporter reads
        def __init__(self, rr, label=""):
            self.id, self.plan, self.flow, self.label, self.model_id = rr.id, rr.plan, rr.flow, label, lab.model_id

    clean = lab.run(A, max_new_tokens=1)
    f = lab.flow(B, max_new_tokens=3, temperature=0.7, seed=5)
    f.schedule([0, 1, 1, 2])
    f.bypass(0)
    f.patch("block_1_r1", clean, positions="last")
    f.intervene(2, hook="attn_heads", op="scale", value=0.5, heads="0:2")
    f.clamp(1, "3,4", to="value", value=2.0)
    f.layer_prompt("Paris", layers="1:", mode="negative")
    f.code("block_1_r1", lab.block_code(1) + "x = x * 1.0\n")
    f.code("sampler", lab.block_code(kind="sampler"))
    run = lab.run(f)
    runs = {clean.id: R(clean, "clean"), run.id: R(run, "patched")}
    src = run_script(runs[run.id], lambda r: default_llm_flow(lab.info), runs.get)
    assert "flow.patch('block_1_r1', source_1" in src and "flow.schedule([0, 1, 1, 2])" in src
    compile(src, "exported.py", "exec")
    ns = {}
    exec(src.replace(f"lab = Lab({lab.model_id!r})", "lab = THE_LAB"), {"THE_LAB": lab, **ns}, ns)
    assert ns["run"].token_ids == run.token_ids


# ---------------------------------------------------------------- training
def test_train_tiny_model_then_load_it(tmp_path):
    from analyser.train import TinyJob
    job = TinyJob("t", tmp_path / "t", corpus="facts", d_model=32, n_layers=2, n_heads=2, context=24, vocab=280,
                  steps=60, batch=16, lr=5e-3)
    with torch.enable_grad():
        job.run()
    assert job.losses[-1][1] < job.losses[0][1] - 0.5          # it learned something
    assert json.loads((tmp_path / "t" / "config.json").read_text())["tokenshark"]["trained_here"]
    lab = Lab(str(tmp_path / "t"), device="cpu", dtype="fp32", policy="device")
    r = lab.run("The capital of", max_new_tokens=4)
    assert len(r.token_ids) > 3 and r.lens()["cells"]


def test_forward_train_matches_the_engine(lab):
    from analyser.train import forward_train
    m, cfg = lab.adapter.model, lab.adapter.cfg
    r = lab.run(A, max_new_tokens=1)
    n = r.captures["n_in"]
    ids = torch.tensor([r.token_ids[:n]])
    spec = {"n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "n_kv": cfg.n_kv, "head_dim": cfg.head_dim, "eps": cfg.eps,
            "inv_freq": m.inv_freq, "act": m.act, "qk_norm": m.qk_norm, "bias": lambda name: m.has_bias and m.W.has(name + ".bias")}
    W = lambda name: m.norm_w(name) if name.endswith("norm.weight") else m.W(name)     # noqa: E731
    lg = forward_train(W, lambda i: m.embed(i.flatten().tolist()).view(*i.shape, -1), lambda h: h @ m.unembed_weight().T, spec, ids)
    assert torch.allclose(lg[0, -1], _logits(r), atol=1e-3)


def test_train_lora_and_attach(lab, tmp_path):
    from analyser.registry import Loaded
    from analyser.tasks import TaskManager
    from analyser.train import LoRAJob
    loaded = Loaded()
    loaded.adapter, loaded.plugin, loaded.spec, loaded.info = lab.adapter, lab.plugin, lab.spec, dict(lab.info)
    base = lab.run(A, max_new_tokens=1)
    job = LoRAJob(loaded, TaskManager(loaded.lock), tmp_path / "lora", "l", ["The capital of France is Tokyo."] * 2,
                  rank=4, steps=25, lr=2e-2, probe=A)
    with torch.enable_grad():
        job.run()
    assert job.losses[-1][1] < job.losses[0][1]
    assert (tmp_path / "lora" / "adapter_model.safetensors").exists()
    s = job.result["lora"]
    assert s["kind"] == "lora" and s["n_adapted_weights"] == 2 * lab.info["n_layers"] and s["layers"]
    assert [x.get("label") for x in job.samples] == ["before", "after"]
    assert not torch.allclose(_logits(lab.run(A, max_new_tokens=1)), _logits(base))
    loaded.detach_lora()
    assert torch.allclose(_logits(lab.run(A, max_new_tokens=1)), _logits(base))


# ---------------------------------------------------------------- MCP server (protocol only)
def test_mcp_protocol():
    from analyser import mcp_server as M
    out = M.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    assert out["result"]["serverInfo"]["name"] == "tokenshark" and "tools" in out["result"]["capabilities"]
    assert M.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = M.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"status", "run_prompt", "logit_lens", "patch_sweep", "teaching_guide"} <= names
    assert all(t["inputSchema"]["type"] == "object" and t["description"] for t in tools)
    g = M.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "teaching_guide", "arguments": {}}})
    assert not g["result"].get("isError") and "patch" in g["result"]["content"][0]["text"].lower()
    bad = M.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}})
    assert bad["result"]["isError"]
    assert M.handle({"jsonrpc": "2.0", "id": 5, "method": "no/such"})["error"]["code"] == -32601


def test_lora_can_adapt_a_single_block(lab, tmp_path):
    """A LoRA adds a small pair per chosen matrix per chosen block — not one layer, and not necessarily all."""
    from analyser.registry import Loaded
    from analyser.tasks import TaskManager
    from analyser.train import LoRAJob
    loaded = Loaded()
    loaded.adapter, loaded.plugin, loaded.spec, loaded.info = lab.adapter, lab.plugin, lab.spec, dict(lab.info)
    job = LoRAJob(loaded, TaskManager(loaded.lock), tmp_path / "one", "one", ["The capital of France is Tokyo."],
                  rank=2, steps=6, lr=2e-2, layers="1")
    with torch.enable_grad():
        job.run()
    s = job.result["lora"]
    assert s["n_adapted_weights"] == 2 and [x["layer"] for x in s["layers"]] == [1]
    assert json.loads((tmp_path / "one" / "adapter_config.json").read_text())["layers_to_transform"] == [1]
    loaded.detach_lora()
    bad = LoRAJob(loaded, TaskManager(loaded.lock), tmp_path / "none", "none", ["x y z"], steps=1, layers="none")
    with pytest.raises(ValueError, match="Nothing to train"), torch.enable_grad():
        bad.run()


def test_train_tiny_image_model():
    from analyser.train_image import HELD_OUT, ImageJob, draw
    import numpy as np
    assert draw("red", "circle", np.random.RandomState(0)).shape == (3, 16, 16)
    job = ImageJob(steps=40, batch=16)
    with torch.enable_grad():
        job.run()
    assert job.losses[-1][1] < job.losses[0][1] and job.grids[0]["url"].startswith("data:image/png;base64,")
    out = job.strip(*HELD_OUT, steps=6, guidance=2.0, seed=1)
    assert out["held_out"] and out["steps_shown"] == [0, 1, 2, 3, 4, 5] and out["sigmas"][0] == 1.0
    a = job.sample([0], [0], steps=4, guidance=1.0, seed=3)
    assert torch.equal(a, job.sample([0], [0], steps=4, guidance=1.0, seed=3))          # same seed, same picture
    assert not torch.equal(a, job.sample([0], [0], steps=4, guidance=1.0, seed=4))
    assert not torch.allclose(a, job.sample([0], [0], steps=4, guidance=4.0, seed=3))    # guidance changes it
    with pytest.raises(ValueError):
        job.strip("purple", "circle")


@pytest.mark.parametrize("method,kw", [("full", {"layers": "1"}), ("ia3", {"layers": "0:2"}), ("norms", {}),
                                       ("adapter", {"layers": "1", "rank": 4}),
                                       ("new_block", {"after": 1, "init": "duplicate"}),
                                       ("new_block", {"after": 2, "init": "average"}),
                                       ("new_block", {"after": 0, "init": "identity"}),
                                       ("new_block", {"after": 1, "init": "random"})])
def test_adaptation_methods(lab, tmp_path, method, kw):
    """Every way of adapting the model trains, saves, attaches, changes the output, and detaches cleanly."""
    from analyser.adapt import AdaptJob
    from analyser.flow import compile_flow, default_llm_flow
    from analyser.registry import Loaded
    from analyser.tasks import TaskManager
    loaded = Loaded()
    loaded.adapter, loaded.plugin, loaded.spec, loaded.info = lab.adapter, lab.plugin, lab.spec, dict(lab.info)
    n = lab.adapter.cfg.n_layers
    base = lab.run(A, max_new_tokens=1)
    job = AdaptJob(loaded, TaskManager(loaded.lock), tmp_path / method, method, ["The capital of France is Tokyo."] * 2,
                   method=method, steps=12, lr=1e-2 if method != "new_block" else 2e-3, probe=A, **kw)
    with torch.enable_grad():
        job.run()
    assert job.losses[-1][1] < job.losses[0][1], job.losses
    s = job.result["lora"]
    assert s["adaptation"]["method"] == method and [x.get("label") for x in job.samples] == ["before", "after"]
    info = loaded.info
    if method == "new_block":
        assert info["n_layers"] == n + 1 and info["extra_blocks"] == [{"layer": n, "after": kw["after"], "init": kw["init"]}]
        plan = compile_flow(default_llm_flow({**info, "model_id": "t"}), {**info, "model_id": "t"})
        assert plan.schedule == list(range(kw["after"] + 1)) + [n] + list(range(kw["after"] + 1, n))
        assert s["n_added_weights"] > 0
    elif method == "adapter":
        assert lab.adapter.model.W.has("model.layers.1.adapter.down.weight") and s["n_added_weights"] == 2
        assert "adapter.down.weight" in lab.block_code(1)
    else:
        assert s["n_adapted_weights"] > 0 and s["layers"]
        if "layers" in kw and kw["layers"] == "1":
            assert [x["layer"] for x in s["layers"]] == [1]
    # the engine (KV cache, step by step) agrees with the training forward pass about the adapted model
    lab.info = {**lab.adapter.info(), "model_id": lab.model_id}
    after = lab.run(A, max_new_tokens=1)
    assert not torch.allclose(_logits(after), _logits(base), atol=1e-5)
    if method in ("adapter", "new_block"):
        f = lab.flow(A, max_new_tokens=1)
        for node in f.to_dict()["nodes"]:
            if node["data"].get("kind") == "block":
                f.code(node["id"], lab.block_code(node["data"]["layer"]))
        assert torch.allclose(_logits(lab.run(f)), _logits(after), atol=1e-4)
    loaded.detach_lora()
    lab.info = {**lab.adapter.info(), "model_id": lab.model_id}
    assert lab.adapter.cfg.n_layers == n and not lab.adapter.model.W._extra
    assert torch.allclose(_logits(lab.run(A, max_new_tokens=1)), _logits(base))


def test_lab_api_attaches_an_added_block(lab, tmp_path):
    from analyser.adapt import AdaptJob
    from analyser.registry import Loaded
    from analyser.tasks import TaskManager
    loaded = Loaded()
    loaded.adapter, loaded.plugin, loaded.spec, loaded.info = lab.adapter, lab.plugin, lab.spec, dict(lab.info)
    n = lab.adapter.cfg.n_layers
    job = AdaptJob(loaded, TaskManager(loaded.lock), tmp_path / "nb", "nb", ["The capital of France is Tokyo."],
                   method="new_block", after=0, init="identity", steps=3, attach=False)
    with torch.enable_grad():
        job.run()
    base = lab.run(A, max_new_tokens=1)
    out = lab.lora(str(tmp_path / "nb"))
    assert out["adaptation"]["extra_blocks"][0]["layer"] == n and lab.info["n_layers"] == n + 1
    assert lab.plan(lab.flow(A)).schedule == [0, n] + list(range(1, n))
    assert not torch.allclose(_logits(lab.run(A, max_new_tokens=1)), _logits(base), atol=1e-6)
    lab.lora(None)
    assert lab.info["n_layers"] == n and torch.allclose(_logits(lab.run(A, max_new_tokens=1)), _logits(base))
