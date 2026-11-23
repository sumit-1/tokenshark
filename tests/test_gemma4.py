"""The Gemma-4 plugin (gemma-4-12B layout) on the generic engine must match transformers on a tiny random model:
text logits, an image + text forward pass, incremental decoding with per-layer KV shapes, the image patchify step,
and plugin selection for a model folder."""
from __future__ import annotations

import ast
import base64
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenshark.engine import RunContext                                              # noqa: E402
from tokenshark.flow import compile_flow, default_llm_flow                            # noqa: E402
from tokenshark.kvcache import KVCache                                                # noqa: E402
from tokenshark.llm import RawLLM                                                     # noqa: E402
from tokenshark.plugins import find_plugin_for                                        # noqa: E402
from tokenshark.plugins.gemma4 import Gemma4Hooks, Gemma4Plugin, image_to_patches, target_size   # noqa: E402
from tests.make_tiny import make_gemma4                                              # noqa: E402

T = pytest.importorskip("transformers")
torch.set_grad_enabled(False)
BOI, IMG, EOI = 3, 4, 5


@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    d = tmp_path_factory.mktemp("gemma4")
    ref = make_gemma4(d).eval()
    return d, ref


def _image(h=37, w=53, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8))


def _data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _model(d, policy="device") -> RawLLM:
    return RawLLM(d, "cpu", torch.float32, policy, 0, Gemma4Hooks())


def _adapter(d):
    return Gemma4Plugin.create_adapter({"id": "g4"}, d, "cpu", torch.float32, "device")


def _raw(model: RawLLM, ids, cache=None, media=None):
    sched = list(range(model.cfg.n_layers))
    cache = cache or KVCache({}, len(sched))
    pos = [cache.next_position() + i for i in range(len(ids))]
    ctx = RunContext(interventions=[])
    return model.forward(ids, pos, "prompt", cache, sched, ctx, 0, lambda iv, o, n: list(range(n)), media=media), cache


def _reference(ref, ids, pixel_values=None, positions=None):
    """transformers with the masks generate() builds: image tokens see each other in sliding layers only."""
    it = torch.tensor([ids])
    kw = {}
    if pixel_values is not None:
        tt = (it == IMG).long()
        emb = torch.zeros(1, len(ids), ref.config.text_config.hidden_size)
        kw = dict(pixel_values=pixel_values, image_position_ids=positions, mm_token_type_ids=tt,
                  attention_mask=type(ref).create_masks_for_generate(ref.config, emb, None, None,
                                                                     torch.arange(len(ids))[None],
                                                                     mm_token_type_ids=tt))
    return ref(input_ids=it, **kw).logits[0]


def test_plugin_is_chosen_for_the_folder(gemma, tmp_path):
    d, _ = gemma
    assert find_plugin_for(d) is Gemma4Plugin
    from tests.make_tiny import make_llm
    make_llm(tmp_path, "qwen2")
    assert find_plugin_for(tmp_path).id == "causal_lm"


def test_config_reads_gemma4_layout(gemma):
    d, _ = gemma
    cfg = _model(d).cfg
    assert cfg.heterogeneous() and cfg.v_norm and cfg.embed_scale == 8.0 and cfg.attn_scale == 1.0
    assert cfg.layer_rope(2)["rope_type"] == "proportional" and cfg.layer_rope(0)["rope_type"] == "default"
    assert (cfg.layer_head_dim(0), cfg.layer_n_kv(0)) == (16, 2)
    assert (cfg.layer_head_dim(2), cfg.layer_n_kv(2), cfg.k_eq_v(2)) == (32, 1, True)
    assert cfg.layer_window(0) == 6 and cfg.layer_window(2) is None


@pytest.mark.parametrize("policy", ["device", "disk"])
def test_text_logits_match_transformers(gemma, policy):
    d, ref = gemma
    model = _model(d, policy)
    ids = list(range(10, 34))                         # longer than the sliding window (6)
    mine, _ = _raw(model, ids)
    theirs = _reference(ref, ids)
    assert torch.allclose(mine, theirs, atol=2e-4, rtol=1e-3), (mine - theirs).abs().max()


def test_incremental_decode_matches_full(gemma):
    d, _ = gemma
    model = _model(d)
    ids = list(range(10, 30))
    full, _ = _raw(model, ids)
    rows, (a, c) = [], _raw(model, ids[:9])
    rows.append(a)
    for t in ids[9:]:
        r, c = _raw(model, [t], cache=c)
        rows.append(r)
    assert torch.allclose(torch.cat(rows), full, atol=1e-4)
    assert c.k[0].shape[-1] == 16 and c.k[2].shape[:1] == (1,) and c.k[2].shape[-1] == 32


def test_image_text_matches_transformers(gemma):
    d, ref = gemma
    model = _model(d)
    enc = Gemma4Hooks().media_encoder(model)
    item = enc.prepare(_image())
    patches, pos = item
    n = enc.n_tokens(item)
    assert n == 8                                       # 37x53 -> 16x32 px -> 2x4 patches of 8x8
    ids = [10, 11, BOI] + [IMG] * n + [EOI, 12, 13, 14, 15]
    mine, _ = _raw(model, ids, media=enc.encode(item))
    theirs = _reference(ref, ids, patches[None], pos[None])
    assert torch.allclose(mine, theirs, atol=2e-4, rtol=1e-3), (mine - theirs).abs().max()
    # The image really matters, and so does the two-way attention inside it
    blank, _ = _raw(model, ids)
    assert (blank - mine).abs().max() > 1e-2
    causal = ref(input_ids=torch.tensor([ids]), pixel_values=patches[None], image_position_ids=pos[None]).logits[0]
    assert (causal - mine).abs().max() > 1e-3


def test_patchify_matches_transformers_helpers():
    """Our patch layout / resize target vs. the reference helpers (their module needs torchvision, so the three
    pure-torch functions are lifted from the transformers source)."""
    src = Path(T.__file__).parent / "models" / "gemma4_unified" / "image_processing_gemma4_unified.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    want = {"get_aspect_ratio_preserving_size", "convert_image_to_patches", "patches_merge"}
    mod = ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want], [])
    ns = {"math": math, "torch": torch}
    exec(compile(mod, str(src), "exec"), ns)
    for h, w in [(37, 53), (64, 96), (300, 20), (20, 300), (1000, 1000)]:
        assert target_size(h, w, 4, 12 * 4, 2) == ns["get_aspect_ratio_preserving_size"](h, w, 4, 12 * 4, 2)
        assert target_size(h, w, 16, 280 * 9, 3) == ns["get_aspect_ratio_preserving_size"](h, w, 16, 280 * 9, 3)
    th, tw = target_size(64, 96, 4, 12 * 4, 2)
    img = _image(64, 96, seed=3).resize((tw, th), Image.BICUBIC)
    mine, pos = image_to_patches(img, patch=4, k=2, max_soft_tokens=12)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
    teacher = ns["convert_image_to_patches"](t, 4)
    gx, gy = torch.meshgrid(torch.arange(tw // 4), torch.arange(th // 4), indexing="xy")
    tpos = torch.stack([gx, gy], -1).reshape(-1, 2)
    merged, mpos = ns["patches_merge"](teacher[None], tpos[None], teacher.shape[0] // 4)
    assert torch.equal(mine, merged[0]) and torch.equal(pos, mpos[0])


def test_adapter_run_with_image(gemma):
    """The full stepper path: flow -> plan -> images in the prompt -> generation; KV tab analysis still works."""
    d, _ = gemma
    ad = _adapter(d)
    info = {**ad.info(), "model_id": "g4"}
    assert info["image_input"] and info["capabilities"]["image_input"] and info["global_layers"] == [2, 4]
    flow = default_llm_flow(info)
    for n in flow["nodes"]:
        if n["data"].get("kind") == "input_prompt":
            n["data"].update(prompt="What is in it?", images=[_data_url(_image())], use_chat_template=False,
                             add_bos=False)
        if n["id"] == "sampler":
            n["data"].update(max_new_tokens=3, temperature=0.0)
    plan = compile_flow(flow, info)
    out = ad.run(plan)
    ids = out["outputs"]["token_ids"]
    assert ids[0] == BOI and ids[1:9] == [IMG] * 8 and ids[9] == EOI
    assert len(ids) == len(ad.build_ids(plan.params)[0]) + 3
    # Same as one raw forward pass with the image injected
    model = ad.model
    prompt = ad.build_ids(plan.params)[0]
    logits, _ = _raw(model, prompt, media=ad.media_features(plan.params))
    assert ids[len(prompt)] == int(logits[-1].argmax())
    from tokenshark.building_blocks import kv_analysis
    kv = kv_analysis(ad, out["captures"], plan.schedule, plan.schedule_nodes)
    assert len(kv["blocks"]) == 5


def test_image_marker_count_must_match(gemma):
    d, _ = gemma
    ad = _adapter(d)
    p = {"prompt": "<|image|> and <|image|>", "images": [_image()], "use_chat_template": False}
    with pytest.raises(ValueError, match="markers"):
        ad.build_ids(p)


def test_rope_theta_override_is_per_layer(gemma):
    """A block's RoPE theta box keeps that layer's RoPE kind: typing the configured theta into a global block is a
    no-op (proportional partial RoPE kept), another theta changes the output."""
    d, _ = gemma
    ad = _adapter(d)
    info = {**ad.info(), "model_id": "g4"}

    def run(theta):
        flow = default_llm_flow(info)
        for n in flow["nodes"]:
            if n["data"].get("kind") == "input_prompt":
                n["data"].update(prompt="The capital of France is", use_chat_template=False)
            if n["id"] == "sampler":
                n["data"].update(max_new_tokens=1, temperature=0.0)
            if n["id"] == "block_2" and theta:
                n["data"]["rope_theta"] = theta
        caps = ad.run(compile_flow(flow, info))["captures"]
        return caps["resid_post"][4]

    base = run(None)
    assert torch.allclose(run(1000000.0), base, atol=1e-5)
    assert (run(50.0) - base).abs().max() > 1e-3
