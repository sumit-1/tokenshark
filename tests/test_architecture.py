"""The contract between plugins and the page: a plugin may only declare tabs, palette actions and train sections
that exist, and the routes ask the plugin (not the model family) whether a tool is supported."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenshark.plugins import ModelPlugin, discover_plugins                             # noqa: E402

WEB = Path(__file__).parent.parent / "tokenshark" / "web"
TAB_IDS = {m for f in (WEB / "tabs").glob("*.js") for m in re.findall(r"registerTab\(\{ id: '(\w+)'", f.read_text(encoding="utf-8"))}
ACTIONS = set(re.findall(r'data-action="(\w+)"', (WEB / "index.html").read_text(encoding="utf-8")))
TRAIN = {m for f in (WEB / "tabs").glob("*.js") for m in re.findall(r"registerTrainSection\('(\w+)'", f.read_text(encoding="utf-8"))}
FEATURES = {"lora", "quant", "kv_cache", "lm_tools", "image_maps", "image_lens", "image_stepper"}


def test_the_page_has_what_we_think_it_has():
    assert {"learn", "output_text", "output_image", "debug_text", "debug_image", "lens", "patch", "kv", "train", "runs"} <= TAB_IDS
    assert {"intervention", "kv_prefill", "code", "init_image"} <= ACTIONS
    assert TRAIN == {"tiny_text", "adapt", "tiny_image"}


@pytest.mark.parametrize("plugin", list(discover_plugins().values()), ids=lambda p: p.id)
def test_plugins_declare_only_what_exists(plugin):
    ui = plugin.ui_schema
    assert set(ui["tabs"]) <= TAB_IDS, f"unknown tab ids: {set(ui['tabs']) - TAB_IDS}"
    assert len(ui["tabs"]) == len(set(ui["tabs"]))
    assert set(ui.get("palette_actions", [])) <= ACTIONS, f"unknown palette actions: {set(ui['palette_actions']) - ACTIONS}"
    assert set(ui.get("train", [])) <= TRAIN
    assert ui.get("course", "") in ("", "text", "image")
    assert set(plugin.features) <= FEATURES
    assert ui.get("modality") in ("text", "image") and ui.get("texts", "") in ("", "lm", "unet", "dit")
    from tokenshark.flow import FLOW_SHAPES
    from tokenshark.runs import RUN_LOOPS
    assert plugin.run_loop in RUN_LOOPS and plugin.family in FLOW_SHAPES
    assert ui.get("step_debugger", False) == (plugin.run_loop != "plain"), "only steppable run loops can offer the step buttons"
    assert plugin.default_flow.__func__ is not ModelPlugin.default_flow.__func__, "the plugin must define default_flow()"
    # two modules may stand behind one tab name (output_text / output_image), but a plugin picks one of them
    aliases = [re.sub(r"_(text|image)$", "", t) for t in ui["tabs"]]
    assert len(aliases) == len(set(aliases))


def test_every_web_module_is_loaded_by_the_entry_point():
    """A tab module registers itself when imported, so app.js must import each one."""
    app = (WEB / "app.js").read_text(encoding="utf-8")
    for f in list((WEB / "tabs").glob("*.js")) + list((WEB / "ui").glob("*.js")):
        assert f"./{f.parent.name}/{f.name}'" in app, f"{f.name} is not imported by app.js"
    core = (WEB / "core.js").read_text(encoding="utf-8")
    assert not re.search(r"from '\./(ui|tabs)/", core), "core.js must not import feature modules"


def test_routes_ask_the_plugin_for_features(tmp_path_factory):
    try:
        from fastapi.testclient import TestClient
    except Exception:
        pytest.skip("fastapi's test client is not available")
    import torch
    from tokenshark.llm import LLMAdapter
    from tokenshark.plugins.causal_lm import CausalLMPlugin
    from tokenshark.registry import LOADED
    from tokenshark.server import app
    from tests.make_tiny import make_llm

    class Bare(CausalLMPlugin):                       # same model, but a plugin that offers no language-model tools
        id, features = "bare", ("lora", "quant")

    d = tmp_path_factory.mktemp("arch") / "tiny"
    make_llm(d, "qwen2")
    ad = LLMAdapter({"id": "tiny"}, d, "cpu", torch.float32, "device")
    LOADED.adapter, LOADED.spec, LOADED.status = ad, {"id": "tiny"}, "ready"
    LOADED.info = {**ad.info(), "model_id": "tiny", "lora": None}
    c = TestClient(app)
    try:
        LOADED.plugin = Bare
        assert c.post("/api/tokenize", json={"text": "hi"}).status_code == 400
        assert c.post("/api/caches/from_text", json={"flow": {}, "text": "x"}).status_code == 400
        assert c.get("/api/flow/default").status_code == 200
        LOADED.plugin = CausalLMPlugin
        assert c.post("/api/tokenize", json={"text": "hi"}).status_code == 200
    finally:
        LOADED.unload()
