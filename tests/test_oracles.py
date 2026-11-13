"""The raw engine must match the reference implementations (transformers / diffusers) on random weights,
and the KV-cache edits must be mathematically consistent."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenshark.engine import RunContext                     # noqa: E402
from tokenshark.flow import compile_flow, default_llm_flow, default_t2i_flow   # noqa: E402
from tokenshark.kvcache import KVCache                       # noqa: E402
from tokenshark.llm import LLMAdapter, RawLLM                # noqa: E402
from tests.make_tiny import make_llm, make_sd              # noqa: E402

torch.set_grad_enabled(False)


def _raw_logits(model: RawLLM, ids, positions=None, cache=None):
    sched = list(range(model.cfg.n_layers))
    cache = cache or KVCache({}, len(sched))
    positions = positions if positions is not None else [cache.next_position() + i for i in range(len(ids))]
    ctx = RunContext(interventions=[])
    return model.forward(ids, positions, "prompt", cache, sched, ctx, 0, lambda iv, o, n: list(range(n))), cache


@pytest.fixture(scope="module", params=["qwen2", "qwen3", "llama"])
def llm(tmp_path_factory, request):
    d = tmp_path_factory.mktemp(request.param)
    ref = make_llm(d, request.param).eval()
    return d, ref


@pytest.mark.parametrize("policy", ["device", "cpu", "disk"])
def test_llm_matches_transformers(llm, policy):
    d, ref = llm
    model = RawLLM(d, "cpu", torch.float32, policy, 0)
    ids = list(range(5, 25))
    mine, _ = _raw_logits(model, ids)
    theirs = ref(torch.tensor([ids])).logits[0]
    assert torch.allclose(mine, theirs, atol=2e-4, rtol=1e-3), (mine - theirs).abs().max()


def test_incremental_decode_matches_full(llm):
    d, _ = llm
    model = RawLLM(d, "cpu", torch.float32, "device", 0)
    ids = list(range(3, 19))
    full, _ = _raw_logits(model, ids)
    cache = None
    rows = []
    a, c = _raw_logits(model, ids[:10])
    rows.append(a)
    for t in ids[10:]:
        r, c = _raw_logits(model, [t], cache=c)
        rows.append(r)
    assert torch.allclose(torch.cat(rows), full, atol=1e-4)


def test_prepend_shift_and_negative(llm):
    """Prefix cache A then tokens B. shift mode == fresh run on A+B. negative mode gives the same logits
    for B because RoPE only sees relative distances."""
    d, _ = llm
    model = RawLLM(d, "cpu", torch.float32, "device", 0)
    A, B = list(range(30, 42)), list(range(50, 58))
    full, _ = _raw_logits(model, A + B)
    _, cache_a = _raw_logits(model, A)
    for mode in ("shift", "negative"):
        pre, start = cache_a.positioned_for_prepend(mode)
        base = KVCache.concat(KVCache({}, pre.n_slots), pre)
        out, _ = _raw_logits(model, B, [start + i for i in range(len(B))], base)
        assert torch.allclose(out, full[len(A):], atol=1e-4), (mode, (out - full[len(A):]).abs().max())
    # A gap in positions changes the result (positions really are used)
    pre, _ = cache_a.positioned_for_prepend("offset", -100)
    out, _ = _raw_logits(model, B, [0 + i for i in range(len(B))], KVCache.concat(KVCache({}, pre.n_slots), pre))
    assert not torch.allclose(out, full[len(A):], atol=1e-3)


def test_cache_file_roundtrip(llm, tmp_path):
    d, _ = llm
    model = RawLLM(d, "cpu", torch.float32, "device", 0)
    _, c = _raw_logits(model, list(range(10)))
    c.save(tmp_path / "c.safetensors", {"name": "x"})
    c2, extra = KVCache.load(tmp_path / "c.safetensors")
    assert extra["name"] == "x" and c2.tokens == c.tokens
    assert all(torch.equal(a, b) for a, b in zip(c.k, c2.k))
    t = c2.truncate(4)
    assert len(t) == 4 and t.k[0].shape[1] == 4
    dr = c2.drop(2, 5)
    assert len(dr) == 7 and dr.tokens == c.tokens[:2] + c.tokens[5:]


def test_meta_format_matches(tmp_path):
    """Meta consolidated.pth (interleaved RoPE) loads through the q/k permutation and matches HF."""
    import json
    ref = make_llm(tmp_path / "hf", "llama").eval()
    sd = ref.state_dict()
    cfg = ref.config
    hd = cfg.hidden_size // cfg.num_attention_heads

    def unpermute(w, n):
        d_out, d_in = w.shape
        return w.view(n, 2, d_out // n // 2, d_in).transpose(1, 2).reshape(d_out, d_in)

    meta = {"tok_embeddings.weight": sd["model.embed_tokens.weight"], "norm.weight": sd["model.norm.weight"],
            "output.weight": sd["lm_head.weight"]}
    for i in range(cfg.num_hidden_layers):
        a = f"model.layers.{i}."
        b = f"layers.{i}."
        meta[b + "attention.wq.weight"] = unpermute(sd[a + "self_attn.q_proj.weight"], cfg.num_attention_heads)
        meta[b + "attention.wk.weight"] = unpermute(sd[a + "self_attn.k_proj.weight"], cfg.num_key_value_heads)
        meta[b + "attention.wv.weight"] = sd[a + "self_attn.v_proj.weight"]
        meta[b + "attention.wo.weight"] = sd[a + "self_attn.o_proj.weight"]
        meta[b + "feed_forward.w1.weight"] = sd[a + "mlp.gate_proj.weight"]
        meta[b + "feed_forward.w2.weight"] = sd[a + "mlp.down_proj.weight"]
        meta[b + "feed_forward.w3.weight"] = sd[a + "mlp.up_proj.weight"]
        meta[b + "attention_norm.weight"] = sd[a + "input_layernorm.weight"]
        meta[b + "ffn_norm.weight"] = sd[a + "post_attention_layernorm.weight"]
    md = tmp_path / "meta"
    md.mkdir()
    torch.save({k: v.clone() for k, v in meta.items()}, md / "consolidated.00.pth")
    (md / "params.json").write_text(json.dumps({
        "dim": cfg.hidden_size, "n_layers": cfg.num_hidden_layers, "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads, "vocab_size": cfg.vocab_size, "norm_eps": cfg.rms_norm_eps,
        "rope_theta": 500000.0, "use_scaled_rope": False, "multiple_of": 128, "ffn_dim_multiplier": 1.5}))
    model = RawLLM(md, "cpu", torch.float32, "disk", 0)
    model.cfg.inter = cfg.intermediate_size
    model.cfg.rope_scaling = cfg.rope_scaling
    from tokenshark.llm import rope_inv_freq
    model.inv_freq = rope_inv_freq(model.cfg)
    ids = list(range(4, 20))
    mine, _ = _raw_logits(model, ids)
    theirs = ref(torch.tensor([ids])).logits[0]
    assert torch.allclose(mine, theirs, atol=2e-4, rtol=1e-3)


def test_adapter_run_with_flow_edits(tmp_path):
    d = tmp_path / "m"
    make_llm(d, "qwen2")
    ad = LLMAdapter({"id": "tiny"}, d, "cpu", torch.float32, "disk", 1)
    info = {**ad.info(), "model_id": "tiny"}
    flow = default_llm_flow(info)
    for n in flow["nodes"]:
        if n["id"] == "sampler":
            n["data"]["max_new_tokens"] = 6
    plan = compile_flow(flow, info)
    assert plan.schedule == [0, 1, 2]
    res = ad.run(plan)
    assert len(res["outputs"]["tokens"]) > 0 and res["captures"]["attn"]
    lens = ad.logit_lens(res["captures"], plan.schedule_nodes, plan.schedule)
    assert len(lens["layers"]) == 4
    # Repeat block 1 by rewiring: embed→0→1→1'→2→final, and ablate MLP of the copy.
    flow["nodes"].append({"id": "block_1b", "type": "box", "position": {"x": 0, "y": 0},
                          "data": {"kind": "block", "label": "Block 1 (copy)", "layer": 1}})
    flow["nodes"].append({"id": "iv", "type": "box", "position": {"x": 0, "y": 0},
                          "data": {"kind": "intervention", "label": "zero mlp", "hook": "mlp_out", "op": "zero"}})
    flow["edges"] = [e for e in flow["edges"] if not (e["source"] == "block_1" and e["target"] == "block_2")]
    flow["edges"] += [{"id": "a", "source": "block_1", "target": "block_1b", "sourceHandle": "out", "targetHandle": "in"},
                      {"id": "b", "source": "block_1b", "target": "block_2", "sourceHandle": "out", "targetHandle": "in"},
                      {"id": "c", "source": "iv", "target": "block_1b", "sourceHandle": "out", "targetHandle": "mod"}]
    plan2 = compile_flow(flow, info)
    assert plan2.schedule == [0, 1, 1, 2] and plan2.interventions[0].pos == 2
    res2 = ad.run(plan2)
    assert len(res2["captures"]["cache"].k) == 4
    # Export this run's cache and prepend it to a new conversation in negative mode.
    cache = res2["captures"]["cache"]
    plan2.params.update(prompt="Hello", prefix_mode="negative")
    res3 = ad.run(plan2, prefix=cache)
    assert res3["outputs"]["prefix_tokens"] == len(cache)
    assert min(res3["captures"]["cache"].pos.tolist()) < 0


# ======================================================================================
# Stable Diffusion
# ======================================================================================

@pytest.fixture(scope="module", params=[False, True], ids=["conv-proj", "linear-proj"])
def sd(tmp_path_factory, request):
    d = tmp_path_factory.mktemp("sd")
    te, unet, vae, sch = make_sd(d, linear_proj=request.param)
    return d, te.eval(), unet.eval(), vae.eval(), sch


def test_clip_tokenizer_matches(sd):
    from transformers import CLIPTokenizer as Ref
    from tokenshark.tokenizer import CLIPTokenizer
    d = sd[0]
    ref = Ref.from_pretrained(d / "tokenizer")
    mine = CLIPTokenizer(d / "tokenizer")
    for text in ["a red fox", "The   snow, NOW!", "fox 123"]:
        assert mine.encode(text) == ref(text, add_special_tokens=False).input_ids, text


def test_sd_components_match_diffusers(sd):
    from tokenshark.sd import CLIPText, UNet, VAE
    d, te, unet, vae, sch = sd
    ids = [49406 % 600] * 3
    ids = list(range(1, 12))
    mine = CLIPText(d / "text_encoder", "cpu", torch.float32, "device")(ids)
    theirs = te(torch.tensor([ids])).last_hidden_state
    assert torch.allclose(mine, theirs, atol=1e-4)
    x = torch.randn(1, 4, 16, 16)
    ctx = torch.randn(1, 11, 32)
    u = UNet(d / "unet", "cpu", torch.float32, "disk")
    out = u(x, 500.0, ctx, RunContext(interventions=[]))
    ref = unet(x, 500, encoder_hidden_states=ctx).sample
    assert torch.allclose(out, ref, atol=1e-3), (out - ref).abs().max()
    v = VAE(d / "vae", "cpu", torch.float32, "device")
    assert torch.allclose(v.decode(x * v.scale), vae.decode(x).sample, atol=1e-3)
    img = torch.rand(1, 3, 32, 32) * 2 - 1
    assert torch.allclose(v.encode(img) / v.scale, vae.encode(img).latent_dist.mean, atol=1e-3)


def test_euler_matches_diffusers(sd):
    from tokenshark.sd import Euler
    d, *_, sch = sd
    for steps in (1, 4, 20):
        sch.set_timesteps(steps)
        e = Euler(dict(sch.config))
        e.set_steps(steps)
        assert torch.allclose(torch.tensor(e.timesteps).float(), sch.timesteps.float())
        assert torch.allclose(torch.tensor(e.sigmas).float(), sch.sigmas.float(), atol=1e-5)
        assert abs(e.init_sigma - float(sch.init_noise_sigma)) < 1e-4


def test_sd_pipeline_matches_diffusers(sd):
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPTokenizer as Ref
    from tokenshark.sd import SDAdapter
    d, te, unet, vae, sch = sd
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=Ref.from_pretrained(d / "tokenizer"),
                                   unet=unet, scheduler=sch, safety_checker=None, feature_extractor=None,
                                   requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    ad = SDAdapter({"id": "sd"}, d, "cpu", torch.float32, "device")
    info = {**ad.info(), "model_id": "sd"}
    flow = default_t2i_flow(info)
    for n in flow["nodes"]:
        if n["id"] == "noise":
            n["data"].update(width=64, height=64, seed=3)
        if n["id"] == "scheduler":
            n["data"].update(steps=5, guidance_scale=3.0)
        if n["id"] == "prompt":
            n["data"].update(prompt="red fox", negative_prompt="")
    plan = compile_flow(flow, info)
    res = ad.run(plan)
    gen = torch.Generator("cpu").manual_seed(3)
    lat = torch.randn((1, 4, 32, 32), generator=gen)
    ref = pipe("red fox", height=64, width=64, num_inference_steps=5, guidance_scale=3.0, latents=lat,
               output_type="np").images[0]
    import base64, io
    import numpy as np
    from PIL import Image
    mine = np.asarray(Image.open(io.BytesIO(base64.b64decode(res["outputs"]["image"].split(",")[1])))) / 255.0
    assert np.abs(mine - ref).max() < 2 / 255 + 1e-3
    # branch from step 2 with the same plan reproduces the same image
    cap = res["captures"]
    res2 = ad.run(plan, resume={"step": 2, "latent": cap["latents_before"][2], "parent_latents": cap["latents_before"],
                                "parent_x0": cap["x0"]})
    assert res2["outputs"]["image"] == res["outputs"]["image"]
    assert res["captures"]["xattn"]["all"]
