"""The lab tools through the HTTP API, the way the browser and the MCP tutor use them."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from fastapi.testclient import TestClient                                          # noqa: E402
except Exception:                                    # the test client needs an HTTP client package
    pytest.skip("fastapi's test client is not available", allow_module_level=True)

from analyser import interp, runs                                                      # noqa: E402
from analyser.llm import LLMAdapter                                                    # noqa: E402
from analyser.plugins.causal_lm import CausalLMPlugin                                  # noqa: E402
from analyser.registry import LOADED                                                   # noqa: E402
from analyser.server import app                                                        # noqa: E402
from tests.make_tiny import make_llm                                                   # noqa: E402

torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    home = tmp_path_factory.mktemp("home")
    for sub in ("vectors", "runs", "models"):
        (home / sub).mkdir()
    mp.setattr(interp, "VEC_DIR", home / "vectors")
    mp.setattr(runs, "RUNS_DIR", home / "runs")
    mp.setenv("ANALYSER_MODELS", str(home / "models"))
    d = tmp_path_factory.mktemp("llm") / "tiny"
    make_llm(d, "qwen2")
    ad = LLMAdapter({"id": "tiny"}, d, "cpu", torch.float32, "device")
    LOADED.adapter, LOADED.plugin, LOADED.spec = ad, CausalLMPlugin, {"id": "tiny"}
    LOADED.info = {**ad.info(), "model_id": "tiny", "lora": None}
    LOADED.status = "ready"
    c = TestClient(app)
    yield c
    LOADED.unload()
    mp.undo()


def _flow(c, prompt, n=2, edit=None):
    f = c.get("/api/flow/default").json()
    for node in f["nodes"]:
        if node["id"] == "prompt":
            node["data"].update(prompt=prompt, use_chat_template=False)
        if node["id"] == "sampler":
            node["data"]["max_new_tokens"] = n
    if edit:
        edit(f)
    return f


def _run(c, flow, label=""):
    r = c.post("/api/runs", json={"flow": flow, "label": label})
    assert r.status_code == 200, r.text
    rid = r.json()["run"]["id"]
    for _ in range(400):
        s = c.get(f"/api/runs/{rid}").json()
        if s["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    assert s["status"] == "done", s
    return s


def _task(c, start):
    assert start.status_code == 200, start.text
    t = start.json()
    for _ in range(2000):
        if not t.get("task"):
            return t["result"]
        s = c.get(f"/api/tasks/{t['task']['id']}").json()
        if s["status"] in ("done", "error", "cancelled"):
            assert s["status"] == "done", s
            return s["result"]
        time.sleep(0.02)
    raise AssertionError("task did not finish")


def _box(flow, target, **data):
    flow["nodes"].append({"id": f"iv{len(flow['nodes'])}", "type": "box", "position": {"x": 0, "y": 0},
                          "data": {"kind": "intervention", **data}})
    flow["edges"].append({"id": f"x{len(flow['edges'])}", "source": flow["nodes"][-1]["id"], "target": target,
                          "sourceHandle": "out", "targetHandle": "mod"})


def test_patch_sweep_and_patch_run(api):
    c = api
    clean = _run(c, _flow(c, "The capital of France is"), "clean")
    corrupt = _run(c, _flow(c, "The capital of Japan is"), "corrupt")
    assert clean["outputs"]["captured"] == {"neurons": True, "parts": True}
    sw = _task(c, c.post("/api/patch/sweep", json={"flow": corrupt["flow"], "source_run": clean["id"], "tokens": "last"}))
    assert len(sw["grid"]) == 3 and abs(sw["grid"][2][0] - 1) < 1e-3
    f = _flow(c, "The capital of Japan is")
    _box(f, "block_2", hook="resid_post", op="patch", ref=clean["id"], positions="prompt")
    patched = _run(c, f)
    n = len(clean["outputs"]["tokens"]) - 2
    assert patched["outputs"]["token_ids"][n] == clean["outputs"]["token_ids"][n]
    f = _flow(c, "The capital of Japan is")
    _box(f, "block_2", hook="resid_post", op="patch", ref="gone")
    assert c.post("/api/runs", json={"flow": f}).status_code == 400
    assert c.post("/api/patch/sweep", json={"flow": f, "source_run": "gone"}).status_code == 404


def test_vectors_neurons_logits(api):
    c = api
    a, b = _run(c, _flow(c, "The capital of France is")), _run(c, _flow(c, "The capital of Japan is"))
    v = c.post("/api/vectors/from_runs", json={"run_a": a["id"], "run_b": b["id"], "pos": 1}).json()
    w = c.post("/api/vectors/from_word", json={"word": "France"}).json()
    assert v["dim"] == 64 and abs(w["norm"] - 1) < 1e-3
    assert {x["id"] for x in c.get("/api/vectors").json()} >= {v["id"], w["id"]}
    f = _flow(c, "The capital of Japan is")
    _box(f, "block_1", hook="resid_post", op="steer", ref=v["id"], value=3.0)
    assert _run(c, f)["plan"]["interventions"][0]["op"] == "steer"
    assert c.delete(f"/api/vectors/{w['id']}").json() == {"ok": True}
    f = _flow(c, "The capital of Japan is")
    _box(f, "block_1", hook="resid_post", op="steer", ref=w["id"])
    assert c.post("/api/runs", json={"flow": f}).status_code == 400            # the vector is gone
    top = c.get(f"/api/runs/{a['id']}/neurons?pos=1&k=3").json()
    assert len(top["neurons"]) == 3
    i = top["neurons"][0]["index"]
    assert len(c.get(f"/api/runs/{a['id']}/neuron?pos=1&index={i}").json()["acts"]) == len(a["outputs"]["tokens"])
    assert c.get(f"/api/runs/{a['id']}/neuron?pos=1&index=99999").status_code == 400
    f = _flow(c, "The capital of France is")
    _box(f, "block_1", hook="mlp_neurons", op="clamp", neurons=str(i), clamp="max", value=5, ref=a["id"])
    assert _run(c, f)["status"] == "done"
    lg = c.get(f"/api/runs/{a['id']}/logits?position=4&top=30").json()
    assert len(lg["rows"]) == 30 and lg["rows"][0][2] >= lg["rows"][1][2]


def test_tokenizer_code_script(api):
    c = api
    t = c.post("/api/tokenize", json={"text": "The capital"}).json()
    assert t["n_tokens"] == len(t["ids"]) > 0
    assert c.get("/api/tokens/search?q=cap").json()["n"] >= 1
    assert len(c.get(f"/api/tokens/{t['ids'][0]}/neighbors?k=4").json()["neighbors"]) == 4
    d = c.get("/api/code/default?kind=block&layer=1").json()
    assert d["editable"] and "remember(k, v)" in d["code"]
    assert c.post("/api/code/check", json={"code": d["code"]}).json()["ok"]
    assert not c.post("/api/code/check", json={"code": "import os"}).json()["ok"]

    def edit(f):
        for n in f["nodes"]:
            if n["id"] == "block_1":
                n["data"]["code"] = d["code"] + "print('hi')\n"
        f["nodes"].append({"id": "lp", "type": "box", "position": {"x": 0, "y": 0}, "data": {
            "kind": "kv_text", "text": "Paris Paris", "layers": "1", "mode": "overwrite"}})
        f["edges"].append({"id": "lpe", "source": "lp", "target": "embed", "sourceHandle": "out", "targetHandle": "cache"})

    r = _run(c, _flow(c, "The capital of France is", edit=edit))
    assert r["outputs"]["edited_code"] == ["block_1"] and r["outputs"]["prints"] and r["outputs"]["prefix_tokens"] > 0
    src = c.get(f"/api/runs/{r['id']}/script").text
    assert "flow.layer_prompt('Paris Paris', layers='1', mode='overwrite')" in src and "flow.code(1," in src
    compile(src, "s.py", "exec")
    bad = _flow(c, "x", edit=lambda f: [n["data"].update(code="x = nope") for n in f["nodes"] if n["id"] == "block_0"])
    rid = c.post("/api/runs", json={"flow": bad}).json()["run"]["id"]
    for _ in range(200):
        s = c.get(f"/api/runs/{rid}").json()
        if s["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    assert s["status"] == "error" and "line 1" in s["error"]


def test_quant_routes(api):
    c = api
    q = c.post("/api/quant", json={"bits": 3}).json()
    assert q["bits"] == 3 and q["layers"] and c.get("/api/quant").json()["active"]
    r = _run(c, _flow(c, "The capital of France is"))
    assert r["plan"]["params"]["overlays"] == {"quant": 3}
    assert "lab.quantize(3)" in c.get(f"/api/runs/{r['id']}/script").text
    assert c.delete("/api/quant").json() == {"active": False}
    cmp = _task(c, c.post("/api/quant/compare", json={"flow": _flow(c, "The capital of France is", 3), "bits": [8, 2]}))
    assert [x["bits"] for x in cmp["rows"]] == [None, 8, 2] and cmp["rows"][2]["kl"] > cmp["rows"][1]["kl"]
    assert not c.get("/api/quant").json()["active"]
    assert c.post("/api/quant", json={"bits": 99}).status_code == 400


def test_requests_from_other_sites_are_refused(api):
    c = api
    f = _flow(c, "x")
    assert c.post("/api/runs", json={"flow": f}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/api/code/check", json={"code": "x = 1"}, headers={"Origin": "http://testserver"}).status_code == 200
    assert c.get("/api/status", headers={"Origin": "https://evil.example"}).status_code == 200


def test_train_tiny_route(api):
    c = api
    r = c.post("/api/train/tiny", json={"name": "t1", "steps": 30, "d_model": 32, "n_layers": 2, "n_heads": 2,
                                        "context": 16, "batch": 8, "corpus": "counting"})
    assert r.status_code == 200, r.text
    assert c.post("/api/train/tiny", json={"name": "t2", "steps": 5}).status_code in (200, 409)
    for _ in range(3000):
        s = c.get("/api/train/tiny").json()
        if s["status"] not in ("queued", "running") and s.get("name") == "t1":
            break
        time.sleep(0.02)
    if s.get("name") == "t1":
        assert s["status"] == "done" and s["result"]["model_id"] == "t1" and len(s["losses"]) > 2
    assert c.post("/api/train/tiny", json={"d_model": 33, "n_heads": 2}).status_code == 400
    assert c.post("/api/train/lora", json={"examples": []}).status_code == 400
    assert c.get("/api/train/corpus/preview?corpus=facts").json()["sample"]


def test_train_image_and_lora_block_routes(api):
    c = api
    assert c.post("/api/train/image/sample", json={}).status_code == 409           # nothing trained yet
    assert c.post("/api/train/image", json={"steps": 20, "batch": 8}).status_code == 200
    for _ in range(3000):
        s = c.get("/api/train/image").json()
        if s["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    assert s["status"] == "done" and s["ready"] and s["grids"] and s["held_out"] == ["yellow", "triangle"]
    out = c.post("/api/train/image/sample", json={"color": "blue", "shape": "square", "steps": 5, "grid": True}).json()
    assert out["strip"].startswith("data:image/png") and out["grid"].startswith("data:image/png") and not out["held_out"]
    assert c.post("/api/train/image/sample", json={"color": "pink"}).status_code == 400
    assert c.post("/api/train/lora", json={"examples": ["a b c d"], "layers": "99"}).status_code == 400
    r = c.post("/api/train/lora", json={"name": "blk", "examples": ["The capital of France is Paris."], "steps": 3, "layers": "0"})
    assert r.status_code == 200, r.text
    for _ in range(3000):
        s = c.get("/api/train/lora").json()
        if s["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    assert s["status"] == "done" and s["layers"] == "0" and [x["layer"] for x in s["result"]["lora"]["layers"]] == [0]
    assert c.delete("/api/lora").json() == {"active": False}


def test_adaptation_routes_add_and_remove_a_block(api):
    c = api

    def wait():
        for _ in range(3000):
            s = c.get("/api/train/lora").json()
            if s["status"] not in ("queued", "running"):
                return s
            time.sleep(0.02)

    assert c.post("/api/train/lora", json={"examples": ["a b c d"], "method": "nonsense"}).status_code == 400
    r = c.post("/api/train/lora", json={"name": "nb", "examples": ["The capital of France is Paris."], "steps": 3,
                                        "method": "new_block", "after": 1, "init": "identity"})
    assert r.status_code == 200, r.text
    s = wait()
    assert s["status"] == "done" and s["method"] == "new_block", s
    st = c.get("/api/status").json()["model"]
    assert st["n_layers"] == 4 and st["extra_blocks"] == [{"layer": 3, "after": 1, "init": "identity"}]
    flow = c.get("/api/flow/default").json()
    assert [n["id"] for n in flow["nodes"] if n["data"]["kind"] == "block"] == ["block_0", "block_1", "block_3", "block_2"]
    assert c.get("/api/lora").json()["adaptation"]["method"] == "new_block"
    run = _run(c, _flow(c, "The capital of France is"))
    assert run["plan"]["schedule"] == [0, 1, 3, 2]
    # training again while a block is attached is refused; detaching restores the base model
    assert c.post("/api/train/lora", json={"examples": ["a b c d"]}).status_code == 409
    assert c.delete("/api/lora").json() == {"active": False}
    assert c.get("/api/status").json()["model"]["n_layers"] == 3
    r = c.post("/api/train/lora", json={"name": "thin", "examples": ["The capital of France is Paris."], "steps": 3,
                                        "method": "adapter", "layers": "2", "rank": 4})
    assert r.status_code == 200, r.text
    s = wait()
    assert s["status"] == "done" and c.get("/api/lora").json()["adaptation"]["adapters"] == [2]
    assert "adapter.down.weight" in c.get("/api/code/default?kind=block&layer=2").json()["code"]
    assert c.delete("/api/lora").json() == {"active": False}
