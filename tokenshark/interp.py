"""Interpretability helpers shared by the web server, the Python API, and the MCP server.

Core building blocks live in `tokenshark.building_blocks`; specialized feature modules live in
`tokenshark.extensions.*` (`patching`, `steering`, `neurons`, etc.) and are re-exported here for
backward compatibility.
"""
from __future__ import annotations

import importlib

from .building_blocks import (
    PARTS,
    _build_word_map,
    _pieces,
    _token_index,
    activation,
    final_logits,
    kv_analysis,
    load_prefixes,
    resolve_sources,
    token_neighbors,
    tokenize,
    vocab_search,
)

_NEURON_EXPORTS = {
    "_default_ref_resid",
    "_neuron_acts",
    "_normalize_prompt_item",
    "_span_slice",
    "compare_neurons_from_acts",
    "iter_neuron_ab_hunt",
    "iter_neuron_batch_test",
    "neuron_detail",
    "neuron_words",
    "search_neurons_by_words",
    "top_neurons",
}

_PATCHING_EXPORTS = {
    "iter_patch_sweep",
}

_STEERING_EXPORTS = {
    "VEC_DIR",
    "VECTORS",
    "VectorLibrary",
    "vector_from_runs",
    "vector_from_word",
}


def __getattr__(name: str):
    if name in _NEURON_EXPORTS or name in _PATCHING_EXPORTS or name in _STEERING_EXPORTS:
        # Extensions import `server.app`, and importing the server discovers every extension. Load the
        # server first so that chain runs once, in order, instead of re-entering a half-imported extension.
        importlib.import_module(".server", package=__package__)
    if name in _NEURON_EXPORTS:
        mod = importlib.import_module(".extensions.neurons", package=__package__)
        return getattr(mod, name)
    if name in _PATCHING_EXPORTS:
        mod = importlib.import_module(".extensions.patching", package=__package__)
        return getattr(mod, name)
    if name in _STEERING_EXPORTS:
        mod = importlib.import_module(".extensions.steering", package=__package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PARTS",
    "VEC_DIR",
    "VECTORS",
    "VectorLibrary",
    "activation",
    "_token_index",
    "resolve_sources",
    "load_prefixes",
    "final_logits",
    "_build_word_map",
    "tokenize",
    "kv_analysis",
    "_pieces",
    "vocab_search",
    "token_neighbors",
    "vector_from_runs",
    "vector_from_word",
    "_neuron_acts",
    "_default_ref_resid",
    "neuron_words",
    "top_neurons",
    "neuron_detail",
    "_span_slice",
    "compare_neurons_from_acts",
    "_normalize_prompt_item",
    "iter_neuron_ab_hunt",
    "search_neurons_by_words",
    "iter_neuron_batch_test",
    "iter_patch_sweep",
]
