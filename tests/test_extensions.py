"""Tests for the Modular Feature Plugin & Extension Architecture (`tests/test_extensions.py`).

Covers:
1. Extension discovery (`list_extensions`), enabling/disabling (`toggle_extension` & `/api/extensions/{id}/toggle`)
2. Custom intervention op registration (`register_op`)
3. Center-Layer Guardrail Fork (`guardrail_fork`) with the clean `powershell_detector` preset (`monitor` and `gate` modes)
4. Previous-Token Attention Attribution Analyser (`attn_traceback`) across all layers and heads
5. Reasoning Steps (`multiarith_cot`) and custom dataset upload (`datasets`)
6. Unified Model & Stepper Sync (`Run -> Cancel/Stop -> Step -> Continue` without restarting from block 0)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from fastapi.testclient import TestClient
except Exception:
    pytest.skip("fastapi's test client is not available", allow_module_level=True)

from tokenshark import interp, runs
from tokenshark.flow import compile_flow
from tokenshark.flowbuild import FlowBuilder
from tokenshark.llm import LLMAdapter
from tokenshark.plugins.causal_lm import CausalLMPlugin
from tokenshark.registry import LOADED
from tokenshark.sdk import OpSpec, register_op
from tokenshark.server import app
from tests.make_tiny import make_llm

torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    home = tmp_path_factory.mktemp("ext_home")
    for sub in ("vectors", "runs", "models", "forks", "datasets"):
        (home / sub).mkdir()
    mp.setattr(interp, "VEC_DIR", home / "vectors")
    mp.setattr(runs, "RUNS_DIR", home / "runs")
    mp.setenv("TOKENSHARK_MODELS", str(home / "models"))
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


def test_extension_discovery_and_toggle(client):
    r = client.get("/api/extensions")
    assert r.status_code == 200
    exts = r.json()
    ids = {e["id"] for e in exts}
    for expected in (
        "patching",
        "steering",
        "neurons",
        "moe",
        "introspect",
        "quant",
        "lora",
        "training",
        "guardrail_fork",
        "attn_traceback",
        "datasets",
    ):
        assert expected in ids, f"Missing expected extension {expected}"

    # Toggle an extension off and back on
    r_off = client.post("/api/extensions/quant/toggle", json={"enabled": False})
    assert r_off.status_code == 200
    assert r_off.json()["enabled"] is False

    r_on = client.post("/api/extensions/quant/toggle", json={"enabled": True})
    assert r_on.status_code == 200
    assert r_on.json()["enabled"] is True


def test_custom_op_registration(client):
    register_op(
        OpSpec(
            name="double_act",
            description="Multiply activation by 2",
            apply=lambda t, iv, ctx: t * 2.0,
        )
    )
    flow = client.get("/api/flow/default").json()
    fb = FlowBuilder(flow).prompt("The capital of France is").sampler(max_new_tokens=2, temperature=0.0)
    fb.intercept(1, hook="mlp_out", op="double_act")
    comp = client.post("/api/flow/compile", json={"flow": fb.to_dict()}).json()
    assert comp["ok"] is True
    assert any(iv["op"] == "double_act" for iv in comp["plan"]["interventions"])


def test_guardrail_fork_powershell_detector_monitor_and_gate(client):
    # 1. Train the clean powershell_detector preset at block 1
    r = client.post(
        "/api/ext/guardrail_fork/train",
        json={
            "preset": "powershell_detector",
            "pos": 1,
            "head_type": "linear_probe",
            "steps": 60,
            "lr": 0.05,
            "name": "test_ps_detector",
            "threshold": 0.55,
        },
    )
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["id"] == "test_ps_detector"
    assert meta["categories"] == ["PowerShell", "Python / General"]
    assert meta["accuracy"] >= 0.75

    # 2. Evaluate directly via probe endpoint
    ev_ps = client.post(
        "/api/ext/guardrail_fork/evaluate",
        json={
            "ref": "test_ps_detector",
            "text": "Get-Process | Where-Object { $_.CPU -gt 10 }",
            "pos": 1,
        },
    ).json()
    assert "scores" in ev_ps
    assert "PowerShell" in ev_ps["scores"]

    # 3. Wire fork_head in monitor mode into Block 1 and run through the stepper
    flow = client.get("/api/flow/default").json()
    fb = FlowBuilder(flow).prompt("Get-Process | Where-Object { $_.CPU -gt 10 }").sampler(max_new_tokens=2, temperature=0.0)
    fid = fb.fork(1, ref="test_ps_detector", mode="monitor", threshold=0.55)
    compiled = compile_flow(fb.to_dict(), LOADED.info)
    assert len(compiled.forks) == 1
    assert compiled.forks[0].node_id == fid

    st = LOADED.adapter.stepper(compiled)
    while st.step_token():
        pass
    fin = st.finish()
    forks_out = fin["outputs"].get("forks") or {}
    assert fid in forks_out
    assert "PowerShell" in forks_out[fid]["scores"]

    # 4. Wire fork_head in gate mode with a low threshold (0.01) to verify early halt at center block
    fb_gate = FlowBuilder(flow).prompt("Get-Process | Where-Object { $_.CPU -gt 10 }").sampler(max_new_tokens=4, temperature=0.0)
    fid_gate = fb_gate.fork(
        1,
        ref="test_ps_detector",
        mode="gate",
        threshold=0.01,
        halt_message="[Halted by PowerShell Guardrail]",
    )
    compiled_gate = compile_flow(fb_gate.to_dict(), LOADED.info)
    st_gate = LOADED.adapter.stepper(compiled_gate)
    while st_gate.step_block():
        pass
    assert st_gate.done is True
    fin_gate = st_gate.finish()
    assert "[Halted by PowerShell Guardrail]" in fin_gate["outputs"]["text"]
    assert fin_gate["outputs"]["forks"][fid_gate]["triggered"] is True


def test_attn_traceback_across_layers_and_heads(client):
    flow = client.get("/api/flow/default").json()
    fb = FlowBuilder(flow).prompt("The capital of France is").sampler(max_new_tokens=2, temperature=0.0)
    r_create = client.post("/api/runs", json={"flow": fb.to_dict(), "label": "trace-test"}).json()
    rid = r_create["run"]["id"]
    for _ in range(100):
        r_stat = client.get(f"/api/runs/{rid}").json()
        if r_stat["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert r_stat["status"] == "done"

    tr = client.post(
        "/api/ext/attn_traceback/trace",
        json={"run_id": rid, "query_token": -1, "top_k": 15},
    )
    assert tr.status_code == 200, tr.text
    data = tr.json()
    assert data["n_blocks"] == LOADED.info["n_layers"]
    assert data["n_heads"] == LOADED.info["n_heads"]
    assert len(data["top_heads"]) > 0
    top0 = data["top_heads"][0]
    assert "attn_weight" in top0
    assert "value_norm" in top0
    assert "logit_push" in top0


def test_datasets_multiarith_cot_and_custom_upload(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    ds_list = r.json()
    multiarith_meta = next((d for d in ds_list if d["id"] == "multiarith_cot"), None)
    assert multiarith_meta is not None
    assert multiarith_meta["n_examples"] >= 2000

    r_cot = client.get("/api/datasets/multiarith_cot?offset=0&limit=50")
    assert r_cot.status_code == 200
    cot = r_cot.json()
    assert cot["n_examples"] >= 2000
    assert len(cot["rows"]) == 50
    assert "question" in cot["rows"][0] and "answer" in cot["rows"][0]
    assert "Step 1:" in cot["rows"][0]["answer"]

    # Preview via /api/train/corpus/preview with structured 2-column rows and n_questions
    r_prev = client.get("/api/train/corpus/preview?corpus=multiarith_cot&offset=0&limit=25&n_questions=1000")
    assert r_prev.status_code == 200
    prev_json = r_prev.json()
    assert prev_json["total"] >= 2000
    assert prev_json["active_questions"] == 1000
    assert len(prev_json["rows"]) == 25
    assert "Question:" in prev_json["sample"]

    # Upload a custom JSONL reasoning dataset with system_prompt
    jsonl_content = (
        '{"question": "What is 3 * 4 + 2?", "steps": ["3 * 4 = 12", "12 + 2 = 14"], "answer": "14"}\n'
        '{"question": "What is 10 - 3 * 2?", "steps": ["3 * 2 = 6", "10 - 6 = 4"], "answer": "4"}\n'
    )
    r_up = client.post(
        "/api/datasets/upload",
        json={
            "name": "my_math_cot",
            "content": jsonl_content,
            "format": "jsonl",
            "system_prompt": "Solve step by step.",
        },
    )
    assert r_up.status_code == 200
    up = r_up.json()
    assert up["id"] == "my_math_cot"
    assert up["n_examples"] == 2
    assert up["system_prompt"] == "Solve step by step."
    assert up["rows"][0]["question"] == "What is 3 * 4 + 2?"
    assert "Step 1: 3 * 4 = 12" in up["rows"][0]["answer"]


def test_unified_stepper_run_sync(client):
    """Verify that starting a debug/stepper session, stepping 1 block, and continuing via debug/step
    preserves the exact same Run and block pointer `bp` without restarting from block 0."""
    flow = client.get("/api/flow/default").json()
    fb = FlowBuilder(flow).prompt("Hello world").sampler(max_new_tokens=2, temperature=0.0)

    # Start stepping session (embeds prompt and stops at block 0)
    st0 = client.post("/api/debug/start", json={"flow": fb.to_dict(), "label": "sync-test"}).json()
    run_id = st0["run_id"]
    assert st0["next_pos"] == 0

    # Step 1 block -> next_pos becomes 1
    client.post("/api/debug/step", json={"mode": "block"})
    for _ in range(50):
        dbg = client.get("/api/debug").json()
        if not dbg.get("busy"):
            break
        time.sleep(0.05)
    assert dbg["run_id"] == run_id
    assert dbg["next_pos"] == 1

    # Continue to end ('all' mode, which `▶ Continue` triggers) -> finishes without restarting from 0
    client.post("/api/debug/step", json={"mode": "all"})
    for _ in range(100):
        dbg = client.get("/api/debug").json()
        if not dbg.get("busy"):
            break
        time.sleep(0.05)
    assert dbg["run_id"] == run_id
    assert dbg["done"] is True
    r_final = client.get(f"/api/runs/{run_id}").json()
    assert r_final["status"] == "done"
