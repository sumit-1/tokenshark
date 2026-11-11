"""Qwen-Image 2.1: the raw engine against diffusers' QwenImage21Pipeline on a tiny random model."""
import base64

import numpy as np
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip("diffusers without Qwen-Image 2.1", allow_module_level=True)

from analyser.flow import Plan, compile_flow, default_t2i_flow
from analyser.qimage import QwenImageAdapter, flow_sigmas

from .make_tiny import make_qwen_image

PROMPT = "a red fox in the snowy forest"


@pytest.fixture(scope="module")
def qi(tmp_path_factory):
    import transformers as T
    root = tmp_path_factory.mktemp("qimage")
    fast, proc = make_qwen_image(root)
    D = diffusers
    pipe = D.QwenImage21Pipeline(
        scheduler=D.FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler"),
        vae=D.AutoencoderKLQwenImage21.from_pretrained(root / "vae").eval(),
        text_encoder=T.Qwen3VLForConditionalGeneration.from_pretrained(root / "text_encoder").eval(),
        processor=proc, transformer=D.QwenImage21Transformer2DModel.from_pretrained(root / "transformer").eval())
    ad = QwenImageAdapter({"id": "test-qimage"}, root, "cpu", torch.float32, "cpu")
    return root, pipe, ad


def _plan(ad, **kw):
    params = dict(prompt=PROMPT, negative_prompt="", seed=7, width=128, height=96, steps=4, guidance_scale=1.0)
    params.update(kw)
    n = ad.dit.n_layers
    return Plan(family="t2i", schedule=list(range(n)), schedule_nodes=[f"block_{i}" for i in range(n)],
                interventions=[], params=params)


def test_scheduler_sigmas_bit_exact(qi):
    root, pipe, ad = qi
    sch = pipe.scheduler
    for steps, n in [(4, 48), (40, 4096), (24, 1024)]:
        c = sch.config
        k = (c.max_shift - c.base_shift) / (c.max_image_seq_len - c.base_image_seq_len)
        sch.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=n * k + (c.base_shift - k * c.base_image_seq_len))
        assert torch.equal(sch.sigmas, flow_sigmas(steps, n, ad.sched_cfg))


@torch.no_grad()
def test_vae_decoder_matches(qi):
    root, pipe, ad = qi
    vae = pipe.vae
    z = torch.randn(1, 8, 1, 6, 5)
    raw = z * torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1) + torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    ref = vae.decode(raw).sample[:, :, 0]
    assert torch.allclose(ad.vae.decode(z[:, :, 0]), ref, atol=1e-5)


@torch.no_grad()
def test_text_encoder_prenorm_hidden(qi):
    root, pipe, ad = qi
    emb, mask, _ = pipe.encode_prompt(PROMPT, device="cpu")
    mine, toks = ad.text(PROMPT)
    assert ad.text.drop_idx == pipe._drop_idx
    assert torch.allclose(mine, emb[0], atol=1e-5)
    assert len(toks) == emb.shape[1]


@torch.no_grad()
@pytest.mark.parametrize("size,steps", [((128, 96), 4), ((64, 64), 3)])
def test_generation_matches_pipeline(qi, size, steps):
    root, pipe, ad = qi
    w, h = size
    ref = pipe(prompt=PROMPT, width=w, height=h, num_inference_steps=steps,
               generator=torch.Generator("cpu").manual_seed(7), output_type="latent").images[0]
    res = ad.run(_plan(ad, width=w, height=h, steps=steps))
    assert torch.allclose(res["captures"]["final_latent"], ref, atol=1e-4)


@torch.no_grad()
def test_true_cfg_matches_pipeline(qi):
    root, pipe, ad = qi
    ref = pipe(prompt=PROMPT, negative_prompt="blurry", true_cfg_scale=4.0, width=128, height=96, num_inference_steps=4,
               generator=torch.Generator("cpu").manual_seed(3), output_type="latent").images[0]
    res = ad.run(_plan(ad, seed=3, negative_prompt="blurry", guidance_scale=4.0))
    assert torch.allclose(res["captures"]["final_latent"], ref, atol=1e-4)


@torch.no_grad()
def test_branch_and_flow_editing(qi):
    root, pipe, ad = qi
    info = ad.info()
    flow = default_t2i_flow(info)
    for n in flow["nodes"]:
        if n["data"]["kind"] == "noise":
            n["data"].update(width=128, height=96, seed=7)
        if n["data"]["kind"] == "scheduler":
            n["data"].update(steps=4)
    plan = compile_flow(flow, info)
    base = ad.run(plan)
    cap = base["captures"]
    branched = ad.run(plan, resume={"step": 2, "latent": cap["latents_before"][2], "parent_latents": cap["latents_before"],
                                    "parent_x0": cap["x0"]})
    assert torch.equal(branched["captures"]["final_latent"], cap["final_latent"])
    # single-stream blocks can be bypassed (unlike UNet stages)
    flow["nodes"] = [dict(n, data=dict(n["data"], bypass=True)) if n["id"] == "block_1" else n for n in flow["nodes"]]
    plan2 = compile_flow(flow, info)
    assert plan2.schedule == [0, 2]
    assert not torch.equal(ad.run(plan2)["captures"]["final_latent"], cap["final_latent"])
    # captures the UI reads
    o = base["outputs"]
    assert o["xattn_groups"][0] == "all" and o["preview_kind"] == "linear"
    assert ad.xattn_map(cap, "all", -1, 3)["h"] == 6
    assert ad.step_preview(cap, 1)[:4] == b"\x89PNG"
    assert base64.b64decode(o["image"].split(",")[1])[:4] == b"\x89PNG"


def _png_url(arr):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def test_vae_encoder_matches(qi):
    root, pipe, ad = qi
    vae = pipe.vae
    x = torch.rand(1, 4, 96, 64) * 2 - 1
    mu = vae.encode(x[:, :, None]).latent_dist.mode()[:, :, 0]
    ref = (mu - torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)) / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    assert torch.allclose(ad.vae.encode(x), ref, atol=1e-5)


@torch.no_grad()
def test_image_lens_last_block_is_the_model_output(qi):
    root, pipe, ad = qi
    res = ad.run(_plan(ad))
    cap = res["captures"]
    last = ad.dit.n_layers - 1
    for step in res["outputs"]["lens_steps"]:
        assert torch.equal(cap["lens"][step][last], cap["x0"][step])
    assert ad.lens_png(cap, res["outputs"]["lens_steps"][0], 0)[:4] == b"\x89PNG"
    assert ad.lens_png(cap, res["outputs"]["lens_steps"][-1], 1, exact=True)[:4] == b"\x89PNG"


@torch.no_grad()
def test_prompt_kv_boxes(qi):
    root, pipe, ad = qi
    base = ad.run(_plan(ad))["captures"]["final_latent"]
    box = dict(node="k", prompt=PROMPT, role="replace", blocks="all", mode="shift", offset=0, label="same")
    same = ad.run(_plan(ad, prompt_kvs=[box]))["captures"]["final_latent"]
    assert torch.equal(same, base)                       # replacing with the same prompt changes nothing
    # a hidden prompt no block can see, placed at negative positions (main prompt still starts at 0): no change
    hidden = dict(box, prompt="a blue whale", role="prepend", blocks="none", mode="negative", label="hidden")
    assert torch.equal(ad.run(_plan(ad, prompt_kvs=[hidden]))["captures"]["final_latent"], base)
    # in shift mode it still occupies positions 0..m-1 and moves the main prompt along — that alone changes the
    # picture, because image tokens sit on a fixed height/width grid while text tokens use one position on all axes
    assert not torch.equal(ad.run(_plan(ad, prompt_kvs=[dict(hidden, mode="shift")]))["captures"]["final_latent"], base)
    for b in (dict(hidden, blocks="all"), dict(hidden, blocks="1", mode="negative"),
              dict(box, prompt="a blue whale", blocks="0:2")):
        res = ad.run(_plan(ad, prompt_kvs=[b]))
        assert not torch.equal(res["captures"]["final_latent"], base)
    lay = res["outputs"]["prompt_layout"]
    assert [l["reads"] for l in lay] == ["same", "same", "main"]


@torch.no_grad()
def test_start_image_and_region_lock(qi):
    root, pipe, ad = qi
    base = ad.run(_plan(ad))["captures"]["final_latent"]
    img = _png_url((np.random.RandomState(0).rand(96, 128, 3) * 255).astype(np.uint8))
    # strength 1 = start from pure noise: the start image is ignored
    assert torch.equal(ad.run(_plan(ad, init_image=img, strength=1.0))["captures"]["final_latent"], base)
    # a hard lock over the whole image ends exactly on the start image's latent
    full = _png_url(np.full((96, 128, 4), 255, np.uint8))
    res = ad.run(_plan(ad, init_image=img, strength=0.5, mask=full, mask_mode="keep", lock_steps="all", lock_strength=1.0))
    init = ad._load_image(img, 128, 96)[0].reshape(8, -1).T
    assert torch.allclose(res["captures"]["final_latent"], init, atol=1e-5)
    assert res["outputs"]["start_step"] == 2


@torch.no_grad()
def test_region_interventions(qi):
    from analyser.flow import Intervention
    root, pipe, ad = qi
    base = ad.run(_plan(ad))["captures"]["final_latent"]
    iv = lambda region: Intervention(node_id="iv", pos=1, hook="out", op="zero", region=region)  # noqa: E731
    white = _png_url(np.full((96, 128, 4), 255, np.uint8))
    empty = _png_url(np.zeros((96, 128, 4), np.uint8))
    half = np.zeros((96, 128, 4), np.uint8)
    half[:, :64] = 255
    run = lambda ivs: ad.run(Plan(**{**_plan(ad).__dict__, "interventions": ivs}))["captures"]["final_latent"]  # noqa: E731
    everywhere = run([iv("")])
    assert torch.equal(run([iv(white)]), everywhere)
    assert torch.equal(run([iv(empty)]), base)
    r = run([iv(_png_url(half))])
    assert not torch.equal(r, base) and not torch.equal(r, everywhere)
    w = Intervention(node_id="w", pos=0, hook="word", op="zero", positions="0:12", region=_png_url(half))
    assert not torch.equal(run([w]), base)


@torch.no_grad()
def test_sketch_guide(qi):
    """Sketch guide: pure noise everywhere; only drawn tiles are steered during the first steps."""
    root, pipe, ad = qi
    base = ad.run(_plan(ad))["captures"]["final_latent"]
    white = np.full((96, 128, 3), 255, np.uint8)
    sketch = white.copy()
    sketch[30:60, 20:24] = 0                                   # one black stroke
    # nothing drawn → nothing guided → exactly the plain generation
    assert torch.equal(ad.run(_plan(ad, init_image=_png_url(white), strength=0.5, guide=True))["captures"]["final_latent"], base)
    # strength 1 → zero guided steps
    assert torch.equal(ad.run(_plan(ad, init_image=_png_url(sketch), strength=1.0, guide=True))["captures"]["final_latent"], base)
    res = ad.run(_plan(ad, init_image=_png_url(sketch), strength=0.5, guide=True))
    assert res["outputs"]["start_step"] == 0 and "Sketch guide" in res["outputs"]["guide_note"]
    m = ad._drawn_mask(_png_url(sketch), 6, 8).reshape(6, 8)
    assert m[2:4, 1].min() > 0.5 and m[:, 5:].max() == 0      # the stroke's tiles, nothing far away
    fl = res["captures"]["final_latent"]
    far = m.reshape(-1) == 0
    assert not torch.equal(fl, base)


@torch.no_grad()
def test_noise_canvas_drawing(qi):
    """A drawing on the noise canvas (transparent where nothing is painted): the alpha says what is drawn."""
    root, pipe, ad = qi
    base = ad.run(_plan(ad))["captures"]["final_latent"]
    empty = np.zeros((96, 128, 4), np.uint8)
    assert torch.equal(ad.run(_plan(ad, init_image=_png_url(empty), strength=0.5, guide=True))["captures"]["final_latent"], base)
    stroke = empty.copy()
    stroke[30:60, 20:24] = [255, 255, 255, 255]                  # a white stroke still counts as drawn
    m = ad._drawn_mask(_png_url(stroke), 6, 8).reshape(6, 8)
    assert m[2:4, 1].min() > 0.5 and m[:, 5:].max() == 0
    assert not torch.equal(ad.run(_plan(ad, init_image=_png_url(stroke), strength=0.5, guide=True))["captures"]["final_latent"], base)


@torch.no_grad()
def test_image_lens_through_last_blocks(qi):
    """Through 0 = the plain lens; through all remaining blocks = the real model output for every block."""
    root, pipe, ad = qi
    plan = _plan(ad)
    res = ad.run(plan)
    cap, S = res["captures"], ad.dit.n_layers
    steps = res["outputs"]["lens_steps"][:2]
    for _ in ad.iter_lens_through(cap, plan, 1, steps):
        pass
    for s in steps:                                           # the last block needs nothing after it
        assert torch.equal(cap["lens_through"][1][s][S - 1], cap["lens"][s][S - 1])
    for _ in ad.iter_lens_through(cap, plan, S, steps):
        pass
    for s in steps:
        for p in range(S):
            assert torch.equal(cap["lens_through"][S][s][p], cap["x0"][s])
    assert ad.lens_png(cap, steps[0], 0, through=S)[:4] == b"\x89PNG"
    assert ad.step_input_preview(cap, steps[0])[:4] == b"\x89PNG"


@torch.no_grad()
def test_image_stepper_and_editing_the_guess(qi):
    """Stepping block by block = a run; editing the guess keeps the model's noise; unchanged pixels change nothing."""
    from analyser.qimage import ImageStepper
    root, pipe, ad = qi
    base = ad.run(_plan(ad))["captures"]["final_latent"]
    st = ad.image_stepper(_plan(ad))
    n = 0
    while st.step_block():
        n += 1
    assert n == 4 * (ad.dit.n_layers + 1)                      # each step: take the input in + one arrow per block
    assert torch.equal(st.finish()["captures"]["final_latent"], base)

    st = ad.image_stepper(_plan(ad))
    st.step_token()                                             # one full step → a guess exists
    with pytest.raises(ValueError):
        st.step_block(); st.edit_guess("x")                     # mid-step: refused
    while st.x is not None:
        st.step_block()
    guess_png = st.guess_png(exact=True)
    url = "data:image/png;base64," + base64.b64encode(guess_png).decode()
    lat0 = st.lat.clone()
    rec = st.edit_guess(url)                                    # the guess, unedited → no tile changes
    assert rec["tiles"] == 0 and torch.equal(st.lat, lat0)
    sigma = float(st.sig[st.i])
    x0 = st.guess.clone()
    red = _png_url(np.tile(np.array([255, 0, 0], np.uint8), (96, 128, 1)))
    st.edit_guess(red, only_changed=False)                      # everywhere: x_t' = (1-σ)·enc(edit) + σ·ε̂
    z = ad._load_image(red, 128, 96)[0].reshape(8, -1).T
    eps = (lat0.float() - (1 - sigma) * x0) / sigma
    assert torch.allclose(st.lat.float(), (1 - sigma) * z + sigma * eps, atol=1e-5)
    while st.step_block():
        pass
    out = st.finish()
    assert "Edited by hand" in out["outputs"]["kv_note"] and not torch.equal(out["captures"]["final_latent"], base)
