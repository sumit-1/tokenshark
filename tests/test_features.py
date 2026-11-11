"""Stepping debugger, bypass, image steering and the models folder."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from analyser.flow import FlowError, compile_flow, default_llm_flow, default_t2i_flow   # noqa: E402
from analyser.llm import LLMAdapter                                                    # noqa: E402
from analyser.sd import SDAdapter                                                      # noqa: E402
from tests.make_tiny import make_llm, make_sd                                          # noqa: E402

torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def llm(tmp_path_factory):
    d = tmp_path_factory.mktemp("llm")
    make_llm(d, "qwen2")
    ad = LLMAdapter({"id": "t"}, d, "cpu", torch.float32, "device")
    info = {**ad.info(), "model_id": "t"}
    flow = default_llm_flow(info)
    for n in flow["nodes"]:
        if n["id"] == "sampler":
            n["data"]["max_new_tokens"] = 5
            n["data"]["temperature"] = 0.8
            n["data"]["seed"] = 7
    return ad, info, flow


def test_stepper_equals_run(llm):
    ad, info, flow = llm
    plan = compile_flow(flow, info)
    ref = ad.run(plan)["outputs"]
    st = ad.stepper(plan)
    n = 0
    while st.step_block():
        n += 1
        s = st.state()
        if s["phase"] == "block":
            assert s["active_node"] in plan.schedule_nodes
    out = st.finish()["outputs"]
    assert out["token_ids"] == ref["token_ids"]
    assert n >= (1 + 3) * 2   # embed + 3 blocks per forward, at least prompt + one token


def test_run_to_and_force(llm):
    ad, info, flow = llm
    plan = compile_flow(flow, info)
    st = ad.stepper(plan)
    st.run_to(1)
    s = st.state()
    assert s["phase"] == "block" and s["next_pos"] == 2 and s["trace"][-1]["pos"] == 1
    st.step_token()
    choices = st.state()["next_choices"]
    forced = choices[3][1]
    st.force(forced)
    st.step_token()
    assert st.seq[-1] == forced


def test_bypass_keeps_node(llm):
    ad, info, flow = llm
    import copy
    f = copy.deepcopy(flow)
    for n in f["nodes"]:
        if n["id"] == "block_1":
            n["data"]["bypass"] = True
    plan = compile_flow(f, info)
    assert plan.schedule == [0, 2] and plan.params["bypassed"] == ["block_1"]
    for n in f["nodes"]:
        if n["data"].get("kind") == "block":
            n["data"]["bypass"] = True
    with pytest.raises(FlowError):
        compile_flow(f, info)


@pytest.fixture(scope="module")
def sd(tmp_path_factory):
    d = tmp_path_factory.mktemp("sd")
    make_sd(d)
    ad = SDAdapter({"id": "s"}, d, "cpu", torch.float32, "device")
    info = {**ad.info(), "model_id": "s"}
    return ad, info


def _flow(info, **prompt):
    f = default_t2i_flow(info)
    for n in f["nodes"]:
        if n["id"] == "noise":
            n["data"].update(width=64, height=64, seed=1)
        if n["id"] == "scheduler":
            n["data"].update(steps=4, guidance_scale=3.0)
        if n["id"] == "prompt":
            n["data"].update({"prompt": "red fox", **prompt})
    return f


def _img(res):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(res["outputs"]["image"].split(",")[1]))), dtype=np.float32)


def test_word_reweight_and_prompt_switch(sd):
    ad, info = sd
    base = _img(ad.run(compile_flow(_flow(info), info)))
    f = _flow(info)
    f["nodes"].append({"id": "iv", "type": "box", "position": {"x": 0, "y": 0}, "data": {
        "kind": "intervention", "hook": "word", "op": "zero", "positions": "1:3"}})
    f["edges"].append({"id": "x", "source": "iv", "target": "down_0", "sourceHandle": "out", "targetHandle": "mod"})
    assert np.abs(_img(ad.run(compile_flow(f, info))) - base).max() > 0
    a = _img(ad.run(compile_flow(_flow(info, switch_prompt="the snow", switch_step=0), info)))
    b = _img(ad.run(compile_flow(_flow(info, prompt="the snow"), info)))
    assert np.abs(a - b).max() == 0


def test_region_lock_keeps_start_image(sd):
    ad, info = sd
    start = Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8"))
    buf = io.BytesIO(); start.save(buf, format="PNG")
    img_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    mask = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    buf = io.BytesIO(); mask.save(buf, format="PNG")
    mask_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    f = _flow(info)
    f["nodes"].append({"id": "init", "type": "box", "position": {"x": 0, "y": 0}, "data": {
        "kind": "init_image", "image": img_url, "strength": 1.0, "mask": mask_url, "mask_mode": "keep"}})
    f["edges"].append({"id": "y", "source": "init", "target": "noise", "sourceHandle": "out", "targetHandle": "init"})
    out = _img(ad.run(compile_flow(f, info)))
    roundtrip = np.asarray(ad.decode_png(ad._encode_image(img_url, 64, 64)), dtype=np.float32)
    assert np.abs(out - roundtrip).max() <= 1.0     # fully locked: the VAE round-trip of the start image


def test_models_folder_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSER_MODELS", str(tmp_path))
    (tmp_path / "my-llm").mkdir()
    (tmp_path / "my-llm" / "config.json").write_text("{}")
    (tmp_path / "junk").mkdir()
    from analyser.weights import discover_folders
    found = discover_folders(set())
    assert [f["id"] for f in found] == ["my-llm"] and found[0]["family"] == "causal_lm"


def test_decode_through_last_blocks(llm):
    ad, info, flow = llm
    plan = compile_flow(flow, info)
    res = ad.run(plan)
    cap = res["captures"]
    S = len(plan.schedule)
    # Going from the embeddings through *all* blocks reproduces the model's own logits.
    full = ad.decode_layer(cap, plan.schedule, -1, through=S)
    last = ad.decode_layer(cap, plan.schedule, S - 1, through=0)
    assert torch.allclose(full, last, atol=3e-3)   # captures are stored in fp16
    wc = ad.word_cloud(cap, plan.schedule_nodes, plan.schedule, -1, [-1, 0, 1], through=1, top=10)
    assert len(wc["layers"]) == 3 and wc["combined"] and len(wc["layers"][0]["words"]) == 10
    lens = ad.logit_lens(cap, plan.schedule_nodes, plan.schedule, through=2)
    assert lens["through"] == 2 and len(lens["layers"]) == S + 1


def test_stage_maps_and_contributions(sd):
    ad, info = sd
    plan = compile_flow(_flow(info), info)
    res = ad.run(plan)
    cap = res["captures"]
    m = ad.stage_maps(cap, 3, plan.schedule_nodes)
    assert len(m["stages"]) == len(plan.schedule) and m["stages"][0]["pca"].startswith("data:image/png")
    out = ad.contributions(plan, cap["image_png"], "all")
    assert len(out) == len(plan.schedule) and all(o["score"] >= 0 for o in out)
    assert max(o["score"] for o in out) > 0


def test_snapshot_while_stepping(llm):
    """Stepping is a superset of Run: lens / cloud / attention work mid-token, with later layers pending."""
    ad, info, flow = llm
    plan = compile_flow(flow, info)
    st = ad.stepper(plan)
    st.step_token()                 # prompt fully processed
    st.step_block()                 # embed the next token
    st.step_block()                 # block 0 sees it
    snap = st.snapshot()
    cap, out = snap["captures"], snap["outputs"]
    T = len(cap["ids"])
    assert out["live"]["inflight"] == 1 and cap["consistent"] is False
    lens = ad.logit_lens(cap, plan.schedule_nodes, plan.schedule)
    last = [row[T - 1] for row in lens["cells"]]
    assert last[0] is not None and last[1] is not None and last[2] is None and last[3] is None   # embed, B0 done; B1, B2 pending
    wc = ad.word_cloud(cap, plan.schedule_nodes, plan.schedule, T - 1, [0, 1, 2])
    assert len(wc["layers"]) == 1 and len(wc["pending"]) == 2
    a = ad.attention(cap, 0, -1)
    assert len(a["matrix"]) == T
    pending = [s["pending"] for s in out["block_stats"].values()]
    assert pending == [False, True, True]
    # finishing gives the same tokens as a plain run
    while st.step_token():
        pass
    assert st.finish()["outputs"]["token_ids"] == ad.run(plan)["outputs"]["token_ids"]


def test_cloud_alignment_and_tracking(llm):
    """Final-layer cloud at position t must be the distribution that produced token t+1."""
    ad, info, flow = llm
    import copy
    f = copy.deepcopy(flow)
    for n in f["nodes"]:
        if n["id"] == "sampler":
            n["data"]["temperature"] = 0.0
            n["data"]["max_new_tokens"] = 6
    plan = compile_flow(f, info)
    res = ad.run(plan)
    cap, out = res["captures"], res["outputs"]
    S = len(plan.schedule)
    n_in = out["segments"][1]["end"]
    for t in range(n_in - 1, len(cap["ids"]) - 1):
        wc = ad.word_cloud(cap, plan.schedule_nodes, plan.schedule, t, [S - 1])
        layer = wc["layers"][0]
        assert layer["next_rank"] == 1 and layer["choice_rank"] == 1           # greedy: actual next = top word
        assert layer["words"][0][0] == wc["next_token"]
        # same as the stored distribution that produced the token
        assert abs(layer["words"][0][1] - out["alternatives"][t][0][2]) < 1e-3
    # tracking: per-block pushes + embedding sum to the logit gap between target and choice (DLA is exact)
    t = n_in - 1
    target_word = out["alternatives"][t][3][0]
    tr = ad.track_word(cap, plan.schedule_nodes, plan.schedule, t, target_word)
    total_push = sum(r["push"] for r in tr["rows"])
    final = ad.decode_layer(cap, plan.schedule, S - 1, 0)[t]
    tid = ad._word_ids(target_word)
    ids = [c[0] for c in tid if c[1] == tr["target"]][0]
    vs = cap["final_top"][t] if ids != cap["final_top"][t] else int(final.topk(2).indices[1])
    gap = float(final[ids] - final[vs])
    assert abs(total_push - gap) < 1e-2 * max(1, abs(gap)), (total_push, gap)
    assert tr["rows"][-1]["choice_rank"] == 1


def test_hidden_system_prompt_and_layer_restricted_prefix(llm):
    ad, info, flow = llm
    import copy
    f = copy.deepcopy(flow)
    for n in f["nodes"]:
        if n["id"] == "sampler":
            n["data"].update(temperature=0.0, max_new_tokens=4)
        if n["id"] == "prompt":
            n["data"].update(use_chat_template=True, system="Answer like a pirate.", prompt="Where is Paris?")
    plan = compile_flow(f, info)
    ref = ad.run(plan)                                # system prompt as visible text
    cache, extra = ad.build_prefix(plan, "Answer like a pirate.", as_system=True)
    hidden = copy.deepcopy(plan)
    hidden.params.update(system="", prefix_cache="x", prefix_mode="shift", prefix_system=extra["system_text"])
    res = ad.run(hidden, prefix=cache)                # same system prompt, but only as a KV prefix
    n_sys = len(cache)
    assert res["outputs"]["token_ids"] == ref["outputs"]["token_ids"][n_sys:]
    assert res["outputs"]["prefix_tokens"] == n_sys
    # prefix visible to no block == no prefix at all (negative mode keeps the new prompt at positions 0..)
    plain = copy.deepcopy(plan)
    plain.params.update(system="", use_chat_template=False, prompt="Where is Paris?", add_bos=False)
    base = ad.run(plain)
    blind = copy.deepcopy(plain)
    blind.params.update(prefix_cache="x", prefix_mode="negative", prefix_layers="0:0")
    out = ad.run(blind, prefix=cache)
    assert out["outputs"]["token_ids"] == base["outputs"]["token_ids"]
    some = copy.deepcopy(blind)
    some.params["prefix_layers"] = "1:3"
    assert ad.run(some, prefix=cache)["captures"]["cache"].prefix_spans == [(0, len(cache), {1, 2})]


def test_two_prefixes_for_different_blocks(llm):
    """Prefix A only for block 0, prefix B only for blocks 1-2: each block must behave exactly as if only
    its own prefix existed."""
    ad, info, flow = llm
    import copy
    f = copy.deepcopy(flow)
    for n in f["nodes"]:
        if n["id"] == "sampler":
            n["data"].update(temperature=0.0, max_new_tokens=3)
        if n["id"] == "prompt":
            n["data"].update(use_chat_template=False, prompt="Where is Paris?", add_bos=False)
    plan = compile_flow(f, info)
    A, _ = ad.build_prefix(plan, "The capital of France", as_system=False)
    B, _ = ad.build_prefix(plan, "Hello world 1 2 3", as_system=False)
    two = copy.deepcopy(plan)
    two.params.update(prefix_cache="a", prefixes=[{"mode": "shift", "layers": "0"}, {"mode": "shift", "layers": "1:3"}])
    res = ad.run(two, prefix=[A, B])
    spans = res["captures"]["cache"].prefix_spans
    assert spans == [(0, len(A), {0}), (len(A), len(A) + len(B), {1, 2})]
    # new tokens start after the longer prefix
    pos = res["captures"]["cache"].pos.tolist()
    assert pos[len(A) + len(B)] == max(len(A), len(B))
    # a prefix visible to no block changes nothing: [A everywhere, B nowhere] == [A everywhere]
    only_a = copy.deepcopy(plan)
    only_a.params.update(prefix_cache="a", prefixes=[{"mode": "shift", "layers": "all"}])
    ref = ad.run(only_a, prefix=[A])["outputs"]["token_ids"]
    a_and_hidden_b = copy.deepcopy(plan)
    a_and_hidden_b.params.update(prefix_cache="a", prefixes=[{"mode": "shift", "layers": "all"},
                                                             {"mode": "negative", "layers": "none"}])
    assert ad.run(a_and_hidden_b, prefix=[A, B])["outputs"]["token_ids"] == ref


def test_chat_template_with_run_cache_prefix_has_no_default_system(llm):
    """A cache saved from a whole conversation, prepended with the chat template on, must be followed by the
    user turn only — not by a fresh template start."""
    ad, info, flow = llm
    import copy
    f = copy.deepcopy(flow)
    for n in f["nodes"]:
        if n["id"] == "prompt":
            n["data"].update(use_chat_template=True, system="", prompt="Hi")
    plan = compile_flow(f, info)
    p = dict(plan.params, prefix_cache="x")
    ids, _ = ad.build_ids(p)
    text = ad.tok.decode(ids)
    assert text.startswith("<|im_start|>user") and "system" not in text


def test_download_progress_reported(tmp_path, monkeypatch):
    """Hugging Face is mocked: the fake download writes a file in chunks; progress must climb to 100%."""
    import time as _t
    import huggingface_hub
    from analyser import weights
    monkeypatch.setenv("ANALYSER_MODELS", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo: ["config.json", "model.safetensors"])
    monkeypatch.setattr(weights, "_repo_sizes", lambda repo: {"config.json": 2, "model.safetensors": 3_000_000})
    monkeypatch.setattr(weights, "_adopt_hf_cache", lambda *a, **k: False)

    def fake_download(repo, allow_patterns=None, local_dir=None, **kw):
        d = weights.Path(local_dir)
        (d / "config.json").write_text("{}")
        part = d / ".cache" / "model.safetensors.incomplete"
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "wb") as f:
            for _ in range(6):
                f.write(b"x" * 500_000); f.flush(); _t.sleep(0.25)
        part.rename(d / "model.safetensors")
        return local_dir
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    seen = []
    root = weights.resolve_model_dir({"id": "m", "repo": "a/b", "family": "causal_lm"}, True, None,
                                     lambda d, t, s, w: seen.append((d, t)))
    assert (root / ".complete").exists()
    assert seen and seen[-1][0] == seen[-1][1] == 3_000_002
    assert any(0 < d < t for d, t in seen)            # intermediate progress was reported


def _mask_url(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_unet_region_interventions(sd):
    """A painted region limits an intervention: empty region = no effect, full region = same as no region."""
    ad, info = sd

    def run(op, hook, region):
        f = _flow(info)
        f["nodes"].append({"id": "iv", "type": "box", "position": {"x": 0, "y": 0}, "data": {
            "kind": "intervention", "hook": hook, "op": op, "value": 0, "positions": "1:3", "region": region}})
        f["edges"].append({"id": "x", "source": "iv", "target": "mid", "sourceHandle": "out", "targetHandle": "mod"})
        return _img(ad.run(compile_flow(f, info)))
    base = _img(ad.run(compile_flow(_flow(info), info)))
    full, empty = np.full((64, 64, 4), 255, np.uint8), np.zeros((64, 64, 4), np.uint8)
    half = empty.copy()
    half[:, :32] = 255
    for hook in ("out", "word"):
        everywhere = run("zero", hook, "")
        assert np.array_equal(run("zero", hook, _mask_url(full)), everywhere)
        assert np.array_equal(run("zero", hook, _mask_url(empty)), base)
        h = run("zero", hook, _mask_url(half))
        assert np.abs(h - base).max() > 0 and np.abs(h - everywhere).max() > 0


def test_unet_noisy_input_preview(sd):
    ad, info = sd
    res = ad.run(compile_flow(_flow(info), info))
    cap = res["captures"]
    for s in (0, 3):
        assert ad.step_input_preview(cap, s)[:4] == b"\x89PNG"
