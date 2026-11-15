"""Mixture-of-experts models and the introspection experiments.

Every MoE family the engine supports is checked against the reference implementation in `transformers` on a tiny
random model, then the MoE-specific tools (router / expert interceptors, expert analysis, knock-out sweep, the
tiny MoE trainer) and the introspection tools (concept injection, concept meter, activation → words → activation)
are exercised end to end."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenshark import blockcode, interp, introspect, moe                                # noqa: E402
from tokenshark.api import Lab                                                           # noqa: E402
from tokenshark.engine import RunContext                                                 # noqa: E402
from tokenshark.flow import FlowError                                                    # noqa: E402
from tokenshark.kvcache import KVCache                                                   # noqa: E402
from tokenshark.llm import RawLLM                                                        # noqa: E402
from tests.make_tiny import MOE_ARCHS, make_llm                                          # noqa: E402

torch.set_grad_enabled(False)
PROMPT = "The capital of France is"


@pytest.mark.parametrize("arch", MOE_ARCHS)
def test_moe_logits_match_transformers(arch, tmp_path):
    """Router, top-k, weight renormalisation, shared expert, OLMoE's flat QK norm and clipping: same logits."""
    ref = make_llm(tmp_path / arch, arch).eval()
    m = RawLLM(tmp_path / arch, "cpu", torch.float32, "device", 0)
    assert m.moe_layers and m.cfg.moe["n"] == 6
    ids = [5, 17, 33, 8, 90, 41, 7, 120, 64, 3, 11, 29]
    cache = KVCache(m.cfg.fingerprint(), m.cfg.n_layers, device="cpu")
    ctx = RunContext(interventions=[])
    ctx.capture = True
    got = m.forward(ids, [float(i) for i in range(len(ids))], "prompt", cache, list(range(m.cfg.n_layers)), ctx, 0,
                    lambda iv, o, n: [])
    want = ref(torch.tensor([ids])).logits[0]
    assert torch.allclose(got, want, atol=2e-4), float((got - want).abs().max())
    layer = max(m.moe_layers)
    assert ctx.store["moe_sel"][layer][0].shape == (len(ids), 2)
    if arch == "qwen2_moe":                       # one dense layer in between, and a shared expert
        assert sorted(m.moe_layers) == [0, 2] and m.moe_layers[0]["shared"]
    if arch == "olmoe":
        assert m.qk_norm_flat and m.cfg.clip_qkv


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    home = tmp_path_factory.mktemp("home")
    mp.setattr(interp, "VEC_DIR", home / "vectors")
    (home / "vectors").mkdir()
    d = tmp_path_factory.mktemp("moe") / "tiny"
    make_llm(d, "qwen2_moe")
    yield Lab(str(d), device="cpu", dtype="fp32", policy="device")
    mp.undo()


def _run(lab, fn=None, **kw):
    fb = lab.flow(PROMPT, chat=False, max_new_tokens=4, **kw)
    if fn:
        fn(fb)
    return lab.run(fb)


def test_info_flow_and_captures(lab):
    info = lab.info
    assert info["moe"] == {"n_experts": 6, "top_k": 2, "norm_topk": False, "expert_d_ff": 48, "layers": [0, 2], "shared": True}
    assert {"router", "expert_out"} <= set(info["hooks"]) and info["capabilities"]["has_moe"]
    nodes = {n["id"]: n["data"] for n in lab.flow().to_dict()["nodes"]}
    assert nodes["block_0"]["moe"] == 6 and "moe" not in nodes["block_1"] and "MoE 2 of 6" in nodes["block_0"]["sub"]
    r = _run(lab)
    sel, w, scores = r.routing(0)
    T = len(r.tokens)
    assert sel.shape == (T, 2) and w.shape == (T, 2) and scores.shape == (T, 6)
    assert r.captures["moe_parts"][0].shape == (T, 2, 64) and r.captures["moe_shared"][0].shape == (T, 64)
    st = r.outputs["block_stats"]["block_0"]["moe"]
    assert len(st["usage"]) == 6 and abs(sum(st["usage"]) - 2.0) < 1e-2 and "moe" not in r.outputs["block_stats"]["block_1"]
    # the pieces add up to the MLP's write: Σ chosen experts + shared expert = mlp_out
    total = r.captures["moe_parts"][0].float().sum(1) + r.captures["moe_shared"][0].float()
    assert torch.allclose(total, r.part(0, "mlp_out"), atol=2e-2)


def test_block_code_matches_engine(lab):
    base = _run(lab)

    def edit(fb):
        for l in range(3):
            c = blockcode.default_code("block", lab.adapter.model, l)
            assert c["editable"] and ("top_k(probs" in c["code"]) == (l != 1)
            fb.code(l, c["code"])
    r = _run(lab, edit)
    assert r.text == base.text
    assert torch.allclose(r.resid(2), base.resid(2), atol=1e-4)
    assert torch.equal(r.routing(0)[0], base.routing(0)[0])          # edited code still reports its routing


def test_router_and_expert_interceptors(lab):
    base = _run(lab)
    sel0 = base.routing(0)[0]
    top = int(torch.bincount(sel0.reshape(-1), minlength=6).argmax())
    r = _run(lab, lambda fb: fb.ban_experts(0, top))
    assert top not in r.routing(0)[0].unique().tolist() and top in sel0.unique().tolist()
    r = _run(lab, lambda fb: fb.only_experts(0, "1,4"))
    assert set(r.routing(0)[0].unique().tolist()) == {1, 4}
    r = _run(lab, lambda fb: fb.only_experts(0, 3))                  # one expert left: the second slot carries no weight
    sel, w, _ = r.routing(0)
    assert (sel[w > 0] == 3).all() and r.outputs["block_stats"]["block_0"]["moe"]["usage"][3] == 1.0
    # silencing every routed expert leaves only the shared expert's write
    r = _run(lab, lambda fb: fb.scale_expert(0, "all", 0))
    assert torch.allclose(r.part(0, "mlp_out"), r.captures["moe_shared"][0].float(), atol=2e-2)
    r = _run(lab, lambda fb: fb.scale_expert(0, "all,shared", 0))
    assert float(r.part(0, "mlp_out").abs().max()) == 0.0
    # experts per token and router temperature are per-block settings
    r = _run(lab, lambda fb: fb.routing(0, k=4, temperature=5.0))
    assert r.routing(0)[0].shape[1] == 4 and r.routing(2)[0].shape[1] == 2
    flat = torch.softmax(r.routing(0)[2] / 5.0, -1).topk(4).values[0]
    assert torch.allclose(r.routing(0)[1][0], flat, atol=1e-4)
    # a position-limited ban only changes those tokens
    r = _run(lab, lambda fb: fb.ban_experts(0, top, positions="0"))
    assert torch.equal(r.routing(0)[0][1:3], sel0[1:3]) or top in r.routing(0)[0][1:].unique().tolist() or top not in sel0[1:].unique().tolist()
    assert top not in r.routing(0)[0][0].tolist()


def test_router_can_be_patched_from_another_run(lab):
    clean = _run(lab)
    other = lab.run(lab.flow("The capital of Japan is", chat=False, max_new_tokens=0))
    T = len(other.tokens)
    assert not torch.equal(other.routing(0)[0], clean.routing(0)[0][:T])
    fb = lab.flow("The capital of Japan is", chat=False, max_new_tokens=0)
    fb.patch(0, clean, hook="router")
    patched = lab.run(fb)
    assert torch.equal(patched.routing(0)[0], clean.routing(0)[0][:T])


def test_flow_validation(lab):
    for bad in (lambda fb: fb.ban_experts(1, 0),                                   # block 1 is dense
                lambda fb: fb.intercept(0, hook="mlp_out", op="ban", experts="1"),  # ban is a router op
                lambda fb: fb.intercept(0, hook="expert_out", op="patch", ref="x"),
                lambda fb: fb.intercept(0, hook="router", op="ban", experts="first")):
        fb = lab.flow(PROMPT, chat=False)
        bad(fb)
        with pytest.raises(FlowError):
            lab.plan(fb)


def test_expert_analysis(lab):
    r = _run(lab)
    ov = r.experts()
    assert [b["pos"] for b in ov["blocks"]] == [0, 2] and ov["parts"]
    b = ov["blocks"][0]
    assert len(b["usage"]) == 6 and 0 < b["entropy"] <= b["max_entropy"] and len(b["top"]) == len(r.tokens)
    d = r.experts(0, token=-1)
    assert len(d["here"]) == 2 and d["has_shared"] and d["shared_here"]["promotes"]
    assert {"push", "promotes", "weight"} <= set(d["here"][0]) and d["answer"]
    used = [x for x in d["experts"] if x["count"]]
    assert used and used[0]["tokens"] and "promotes" in used[0]
    e = r.expert(0, used[0]["index"])
    assert len(e["weight"]) == len(r.tokens) and e["count"] == used[0]["count"] and len(e["attracts"]) == 10 and e["size"] > 0
    rr = lab.router(0)
    assert len(rr["cosine"]) == 6 and abs(rr["cosine"][2][2] - 1) < 1e-3
    with pytest.raises(ValueError):
        r.experts(1)                                                              # a dense block
    with pytest.raises(ValueError):
        r.top_neurons(0)                                                          # neurons live inside experts


def test_expert_knockout_sweep(lab):
    for mode in ("ban", "zero"):
        s = lab.expert_sweep(PROMPT, block=-1, mode=mode, chat=False)
        assert s["rows"] and len(s["rows"]) == s["total"] and {r["pos"] for r in s["rows"]} == {0, 2}
        assert all({"d_margin", "kl", "p_answer", "usage"} <= set(r) for r in s["rows"])
        assert any(abs(r["d_margin"]) > 0 for r in s["rows"])
    one = lab.expert_sweep(PROMPT, block=2, chat=False)
    assert {r["pos"] for r in one["rows"]} == {2}
    with pytest.raises(ValueError):
        lab.expert_sweep(PROMPT, block=1, chat=False)


def test_script_export_keeps_experts(lab):
    from tokenshark.export import run_script
    r = _run(lab, lambda fb: fb.ban_experts(0, "1,2"))
    r.model_id, r.label, r.sources = lab.model_id, "", {}
    src = run_script(r, lambda _r: lab.plugin.default_flow(lab.info), lambda rid: None, {})
    assert "experts='1,2'" in src and "op='ban'" in src


def test_train_tiny_moe_and_load_it(tmp_path):
    from tokenshark.train import TinyJob
    with torch.enable_grad():
        j = TinyJob("tm", tmp_path / "tm", d_model=32, n_layers=2, n_heads=2, context=32, steps=40, batch=8,
                    n_experts=4, experts_per_token=2)
        j.run()
    assert j.losses[-1][1] < j.losses[0][1] and len(j.usage) == 2 and abs(sum(j.usage[0]) - 1) < 0.05
    lab = Lab(str(tmp_path / "tm"), device="cpu", dtype="fp32", policy="device")
    assert lab.info["moe"]["n_experts"] == 4 and lab.info["moe"]["norm_topk"]
    r = lab.run(PROMPT, chat=False, max_new_tokens=3)
    assert r.routing(1)[0].shape[1] == 2 and abs(float(r.routing(1)[1][0].sum()) - 1) < 1e-4
    with pytest.raises(ValueError):
        TinyJob("x", tmp_path / "x", n_experts=4, experts_per_token=5)


# ---------------------------------------------------------------------------------- introspection
def test_score_answer_rules():
    s = introspect.score_answer
    assert s("Yes, I detect an injected thought. It seems to be about the ocean.", "ocean")["introspect"]
    assert not s("ocean ocean ocean ocean ocean ocean", "ocean")["introspect"]          # steering, not noticing
    a = s("The ocean is vast. Yes, I notice something.", "ocean")                        # said it before noticing
    assert a["named"] and a["claims"] and not a["introspect"]
    assert not s("No, I do not detect any injected thought.", "ocean")["claims"]
    assert s("Yes, I detect a thought about bread.", "ocean") == {"claims": True, "named": False, "coherent": True, "introspect": False}


def test_concept_injection_sweep(lab):
    out = lab.inject(["France", "capital"], blocks=[0, 1], strengths=[0.5, 2.0], prompt=PROMPT, max_new_tokens=4,
                     flow=lab.flow(PROMPT, chat=False))
    assert out["status"] == "done" and len(out["trials"]) == 8 and out["control"]["text"] is not None
    assert len(out["cells"]) == 2 and len(out["cells"][0]) == 2 and out["cells"][1][1]["n"] == 2
    assert {"false_positive", "best_introspect", "net"} <= set(out["summary"])
    weak = [t for t in out["trials"] if t["strength"] == 0.5 and t["block"] == 1]
    strong = [t for t in out["trials"] if t["strength"] == 2.0 and t["block"] == 1]
    assert {t["text"] for t in weak} != {t["text"] for t in strong} or out["control"]["text"] not in {t["text"] for t in strong}
    with pytest.raises(ValueError):
        lab.inject([], blocks=[0], strengths=[1.0], flow=lab.flow(PROMPT, chat=False))


def test_concept_vector_and_meter(lab):
    v = lab.concept_vector("France", 1, flow=lab.flow(PROMPT, chat=False))
    assert v["source"] == "concept" and v["dim"] == 64 and v["resid_norm"] > 0
    r = _run(lab)
    m = r.meter(v)
    assert len(m["rows"]) == 3 and len(m["rows"][0]["cos"]) == len(r.tokens) and m["rows"][1]["cos"][0] == 0.0
    assert all(-1.0001 <= c <= 1.0001 for row in m["rows"] for c in row["cos"])
    # steering along the vector moves the stream towards it
    fb = lab.flow(PROMPT, chat=False, max_new_tokens=4)
    fb.steer(1, v, value=3.0)
    assert lab.run(fb).meter(v)["rows"][1]["mean_prompt"] > m["rows"][1]["mean_prompt"]


def test_activation_to_words_round_trip(lab):
    r = _run(lab)
    out = r.verbalize(1, token=3, inject=0, max_new_tokens=4)
    assert out["status"] == "done" and out["token"] == 3 and out["explanation"] is not None
    labels = [b["label"] for b in out["baselines"]]
    assert "the text up to the token" in labels and "an unrelated token of the run" in labels
    ceiling = next(b for b in out["baselines"] if b["label"] == "the text up to the token")
    floor = next(b for b in out["baselines"] if b["label"] == "an unrelated token of the run")
    assert ceiling["centred"] > floor["centred"]
    mine = r.score_explanation(1, 3, "of France")
    assert set(mine) == {"text", "cosine", "centred", "fve"}
    with pytest.raises(ValueError):
        r.verbalize(1, token=0)                                                    # the attention-sink token


# ---------------------------------------------------------------------------------- custom routing
def test_custom_route_and_shared_switch(lab):
    base = _run(lab)
    T = len(base.tokens)
    # exactly these experts with exactly these weights, for every token
    r = _run(lab, lambda fb: fb.route(0, {1: 0.75, 4: 0.25}))
    sel, w, _ = r.routing(0)
    assert sel.tolist() == [[1, 4]] * T and torch.allclose(w, torch.tensor([[0.75, 0.25]] * T))
    # the block's write is then exactly 0.75·expert1 + 0.25·expert4 + shared: check against a by-hand computation
    m, lay = lab.adapter.model, lab.adapter.model.moe_layers[0]
    one = _run(lab, lambda fb: fb.route(0, {1: 1.0}))
    four = _run(lab, lambda fb: fb.route(0, {4: 1.0}))
    e1 = one.captures["moe_parts"][0][0, 0].float()
    e4 = four.captures["moe_parts"][0][0, 0].float()
    got = r.captures["moe_parts"][0][0].float()                       # token 0 has the same input in all three runs
    assert torch.allclose(got[0], 0.75 * e1, atol=2e-2) and torch.allclose(got[1], 0.25 * e4, atol=2e-2)
    # no weight written = the router's own score for that expert; more experts than k is allowed
    r = _run(lab, lambda fb: fb.route(0, [0, 2, 5]))
    sel, w, scores = r.routing(0)
    assert sel.shape == (T, 3) and torch.allclose(w, torch.softmax(scores, -1)[:, [0, 2, 5]], atol=1e-5)
    # only some tokens; negative weights subtract an expert's write
    r = _run(lab, lambda fb: fb.route(0, {3: -1.0}, positions="1"))
    sel, w, _ = r.routing(0)
    assert sel[1].tolist()[0] == 3 and float(w[1, 0]) == -1.0 and torch.equal(sel[0], base.routing(0)[0][0])
    assert r.outputs["block_stats"]["block_0"]["moe"]["usage"][3] > 0
    # the shared expert can be switched off per block
    r = _run(lab, lambda fb: fb.shared_expert(0, False))
    assert 0 not in r.captures["moe_shared"] and 2 in r.captures["moe_shared"]
    assert torch.allclose(r.part(0, "mlp_out"), r.captures["moe_parts"][0].float().sum(1), atol=2e-2)
    assert r.experts(0)["shared_off"] and not r.experts(2)["shared_off"]
    for bad in (lambda fb: fb.intercept(0, hook="router", op="route"),                     # no experts named
                lambda fb: fb.intercept(0, hook="router", op="route", experts="0:3"),
                lambda fb: fb.intercept(0, hook="router", op="route", experts="2=x")):
        fb = lab.flow(PROMPT, chat=False)
        bad(fb)
        with pytest.raises(FlowError):
            lab.plan(fb)
    fb = lab.flow(PROMPT, chat=False)
    fb.route(0, {9: 1.0})                                                               # the block has experts 0…5
    with pytest.raises(ValueError):
        lab.run(fb)


def test_analyses_see_the_edited_model(lab):
    """Edits made in the flow are part of every later calculation, not just of the run itself."""
    S = len(lab.plan(lab.flow(PROMPT, chat=False)).schedule)
    v = lab.concept_vector("France", S - 1, flow=lab.flow(PROMPT, chat=False))

    def edit(fb):
        fb.ban_experts(S - 1, "0,1,2")
        fb.route(S - 1, {4: 2.0}, positions="2")
        fb.shared_expert(S - 1, False)
        fb.steer(S - 1, v, value=2.0, hook="resid_post")
    r, base = _run(lab, edit), _run(lab)
    ad, cap = lab.adapter, r.captures
    final = torch.stack([r.logits(i) for i in range(len(r.tokens))])
    # decoding the stream before the last block *through* that block must give the run's own logits …
    through = ad.decode_layer(cap, r.plan.schedule, S - 2, through=1)
    assert torch.allclose(through, final, atol=1e-3), float((through - final).abs().max())
    # … which it would not if the block were run with the model's default routing
    plain = dict(cap, interventions=[], moe_noshared=set())
    assert not torch.allclose(ad.decode_layer(plain, r.plan.schedule, S - 2, through=1), final, atol=1e-2)
    assert not torch.allclose(final[: len(base.tokens)][:4], torch.stack([base.logits(i) for i in range(4)]), atol=1e-2)
    # lens / word cloud / word tracking are built on it
    lens = r.lens(k=3, through=1)
    assert lens["cells"][-2][3]["top"][0][0] == lens["cells"][-1][3]["top"][0][0]
    # sweeps keep the flow's edits switched on
    fb = lab.flow(PROMPT, chat=False)
    edit(fb)
    s = lab.expert_sweep(fb, block=S - 1)
    assert not {0, 1, 2} & {x["index"] for x in s["rows"]}                           # banned experts are not in play
    plan = lab.plan(fb)
    plan.sources = interp.resolve_sources(plan, lambda rid: None)
    out = None
    for out in interp.iter_patch_sweep(ad, plan, base.captures, "resid_post", "last"):
        pass
    plan0 = lab.plan(lab.flow(PROMPT, chat=False))
    for ref in interp.iter_patch_sweep(ad, plan0, base.captures, "resid_post", "last"):
        pass
    assert out["p_corrupt"] != ref["p_corrupt"]
