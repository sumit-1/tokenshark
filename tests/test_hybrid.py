"""Qwen3.5/3.6 hybrid models (Gated DeltaNet + gated attention, partial RoPE, zero-centred norms)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenshark.engine import RunContext                     # noqa: E402
from tokenshark.kvcache import KVCache                       # noqa: E402
from tokenshark.llm import RawLLM                            # noqa: E402
from tests.make_tiny import make_llm                       # noqa: E402

torch.set_grad_enabled(False)


def _fwd(model, ids, positions=None, cache=None):
    sched = list(range(model.cfg.n_layers))
    cache = cache or KVCache({}, len(sched))
    positions = positions if positions is not None else [cache.next_position() + i for i in range(len(ids))]
    return model.forward(ids, positions, "prompt", cache, sched, RunContext(interventions=[]), 0,
                         lambda iv, o, n: list(range(n))), cache


@pytest.fixture(scope="module", params=["qwen3_5", "qwen3_5_vl"])
def hyb(tmp_path_factory, request):
    d = tmp_path_factory.mktemp(request.param)
    ref = make_llm(d, request.param).eval()
    return d, ref, request.param


@pytest.mark.parametrize("policy", ["device", "disk"])
def test_hybrid_matches_transformers(hyb, policy):
    d, ref, kind = hyb
    model = RawLLM(d, "cpu", torch.float32, policy, 0)
    assert model.cfg.layer_types.count("linear_attention") == 3 and model.attn_gate and model.cfg.rot_dim == 8
    ids = list(range(5, 29))
    mine, _ = _fwd(model, ids)
    theirs = ref(input_ids=torch.tensor([ids])).logits[0]
    assert torch.allclose(mine, theirs, atol=3e-4, rtol=1e-3), (mine - theirs).abs().max()


def test_hybrid_incremental_and_prefix(hyb):
    d, _, _ = hyb
    model = RawLLM(d, "cpu", torch.float32, "device", 0)
    ids = list(range(3, 23))
    full, _ = _fwd(model, ids)
    rows = []
    r, c = _fwd(model, ids[:9])
    rows.append(r)
    for t in ids[9:]:
        r, c = _fwd(model, [t], cache=c)
        rows.append(r)
    assert torch.allclose(torch.cat(rows), full, atol=2e-4)          # token-by-token == all at once
    # prefix: attention K/V + linear memory carried over; shift and negative both equal the joint run
    A, B = ids[:12], ids[12:]
    _, ca = _fwd(model, A)
    assert ca.states and all(ca.k[s] is None for s in ca.states)
    for mode in ("shift", "negative"):
        pre, start = ca.positioned_for_prepend(mode)
        base = KVCache.concat(KVCache({}, pre.n_slots), pre)
        out, _ = _fwd(model, B, [start + i for i in range(len(B))], base)
        assert torch.allclose(out, full[len(A):], atol=2e-4), mode
    with pytest.raises(ValueError):
        ca.truncate(5)                                              # linear memory can't be cut


def test_hybrid_cache_file_roundtrip(hyb, tmp_path):
    d, _, _ = hyb
    model = RawLLM(d, "cpu", torch.float32, "device", 0)
    _, c = _fwd(model, list(range(10)))
    c.save(tmp_path / "c.safetensors")
    c2, _ = KVCache.load(tmp_path / "c.safetensors")
    assert set(c2.states) == set(c.states)
    assert torch.equal(c2.states[0]["rec"], c.states[0]["rec"].cpu())


def test_bf16_checkpoint_stays_compact_in_ram(hyb, tmp_path):
    """A bf16 checkpoint run in fp32 on CPU keeps resident weights in bf16 (half the RAM) and still computes
    the same as streaming from disk."""
    d, _, _ = hyb
    from safetensors.torch import load_file, save_file
    import shutil
    b = tmp_path / "bf16"
    shutil.copytree(d, b)
    for f in b.glob("*.safetensors"):
        sd = load_file(str(f))
        save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}, str(f))
    dev = RawLLM(b, "cpu", torch.float32, "device", 0)
    assert all(t.dtype == torch.bfloat16 for t in dev.W._resident.values() if t.is_floating_point())
    disk = RawLLM(b, "cpu", torch.float32, "disk", 0)
    ids = list(range(4, 16))
    # bf16 → fp32 is exact, so keeping bf16 in RAM and converting on use is bit-identical to converting at load
    eager = RawLLM(b, "cpu", torch.float32, "device", 0)
    for k, t in eager.W._resident.items():
        if t.is_floating_point():
            eager.W._resident[k] = t.float()
    out = _fwd(dev, ids)[0]
    assert torch.equal(out, _fwd(disk, ids)[0]) and torch.equal(out, _fwd(eager, ids)[0])
