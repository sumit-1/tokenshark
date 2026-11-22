"""Plugin: Decoder-Only Causal Language Models (Llama, Qwen2/2.5/3/3.5/3.6, Mistral, SmolLM) and their
mixture-of-experts relatives (Mixtral, Qwen1.5/2-MoE, Qwen3-MoE, Qwen3.5-MoE, OLMoE, DeepSeek-V2).

What this plugin teaches:
-------------------------
- Autoregressive token-by-token generation (`embed -> transformer blocks -> final_norm -> lm_head`).
- **Pre-RoPE KV Cache**: how keys are stored before rotary position embeddings so conversations
  or hidden system prompts can be prepended, shifted, or truncated.
- **Logit Lens & Word Clouds**: projecting any intermediate block's residual stream (directly or
  through the last `k` blocks) onto the vocabulary to see when the model decides on the next word.
- **Hybrid Linear + Full Attention (Qwen3.5/3.6)**: comparing Gated DeltaNet running-memory blocks
  against standard softmax attention blocks.
- **Mixture of Experts**: blocks whose MLP is many small MLPs plus a router. The flow shows every expert; the
  Experts tab reads the routing and what each expert writes; `router` / `expert_out` hooks ban, force, silence
  or amplify experts. The layout is detected from the checkpoint, so any model that routes this way loads.
- **Introspection**: concept injection, a concept meter, and activation → words → activation round trips.

How to add a new Causal LM model:
---------------------------------
Add an entry to `default_models` below (or click `+` in the UI with a Hugging Face repo ID),
or drop a folder with `config.json` + `*.safetensors` into `models/`.
"""
from __future__ import annotations

from pathlib import Path
import torch

from ..llm import LLMAdapter
from ..weights import read_json
from . import ModelPlugin, register_plugin


@register_plugin
class CausalLMPlugin(ModelPlugin):
    id = "causal_lm"
    name = "Decoder-Only Causal LM (Llama / Qwen / Mistral / SmolLM / MoE)"
    family = "causal_lm"
    summary = (
        "Autoregressive text generation with editable pre-RoPE KV cache, step-by-step block debugger, "
        "logit lens, word clouds, direct logit attribution, LoRA / fine-tune inspection, mixture-of-experts "
        "routing tools and introspection experiments."
    )

    # Built-in models for this plugin (downloadable on first load into models/<id>/)
    default_models = [
        {
            "id": "qwen2.5-0.5b-instruct",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "Qwen/Qwen2.5-0.5B-Instruct",
            "size_gb": 1.0,
            "notes": "24 blocks, GQA 14q/2kv. Best first model: fast on CPU.",
        },
        {
            "id": "smollm2-360m-instruct",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "size_gb": 0.7,
            "notes": "Llama architecture, 32 blocks. Smallest option.",
        },
        {
            "id": "qwen2.5-1.5b-instruct",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "Qwen/Qwen2.5-1.5B-Instruct",
            "size_gb": 3.1,
            "notes": "28 blocks. Noticeably better answers; GPU recommended.",
        },
        {
            "id": "qwen3-1.7b",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "Qwen/Qwen3-1.7B",
            "size_gb": 3.4,
            "notes": "28 blocks, per-head QK RMSNorm. Needs transformers >= 4.51.",
        },
        {
            "id": "llama-3.2-1b-instruct",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "meta-llama/Llama-3.2-1B-Instruct",
            "size_gb": 2.5,
            "notes": "Gated on Hugging Face: accept the licence and run `hf auth login` first.",
        },
        {
            "id": "olmoe-1b-7b-instruct",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "allenai/OLMoE-1B-7B-0924-Instruct",
            "size_gb": 12.9,
            "notes": (
                "Mixture of experts: 16 blocks, each with 64 experts of which 8 run per token (about 1B of 7B "
                "weights active). Best first MoE model. Fits a 16 GB GPU; otherwise 'stream from disk'."
            ),
        },
        {
            "id": "qwen1.5-moe-a2.7b-chat",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "size_gb": 26.7,
            "notes": (
                "Mixture of experts: 24 blocks, 60 experts (4 per token) plus a shared expert every token uses. "
                "~27 GB in bf16 — 'RAM → device' or 'stream from disk'."
            ),
        },
        {
            "id": "deepseek-v2-lite-chat",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "size_gb": 29.3,
            "notes": (
                "Mixture of experts with latent attention (MLA): 27 blocks, 64 experts (6 per token) plus shared "
                "experts; the first block is dense. ~29 GB — 'RAM → device' or 'stream from disk' (about a minute for a "
                "short answer when streamed)."
            ),
        },
        {
            "id": "qwen3-30b-a3b",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "Qwen/Qwen3-30B-A3B",
            "size_gb": 56.9,
            "notes": (
                "Mixture of experts: 48 blocks, 128 experts (8 per token): 30B weights, about 3B active. ~57 GB — "
                "use 'stream from disk'; only the experts a token is routed to are read."
            ),
        },
        {
            "id": "mixtral-8x7b-instruct",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "size_gb": 87.0,
            "notes": (
                "Mixture of experts: 32 blocks, 8 large experts (2 per token). ~87 GB — 'stream from disk' only, "
                "and slow: each expert is 2.6 GB."
            ),
        },
        {
            "id": "qwen3.6-27b",
            "plugin": "causal_lm",
            "family": "causal_lm",
            "repo": "Qwen/Qwen3.6-27B",
            "size_gb": 55.0,
            "notes": (
                "Hybrid: 48 Gated DeltaNet (linear-attention) + 16 gated-attention blocks, 64 total. "
                "~55 GB in bf16 — use 'stream from disk' (or 'RAM → device' with 64 GB+ RAM)."
            ),
        },
    ]

    # Declarative UI configuration for Causal LM models
    features = ("lora", "quant", "kv_cache", "lm_tools")
    run_loop = "token_stepper"

    ui_schema = {
        "tabs": ["output_text", "debug_text", "lens", "cloud", "attn", "experts", "neurons", "tokenizer",
                 "sampler", "kv", "lora", "quant", "train", "runs"],
        "palette_actions": ["intervention", "block_kv", "duplicate_block", "bypass_block", "kv_prefix", "code"],
        "train": ["tiny_text", "adapt"],
        "modality": "text",
        "texts": "lm",
        "block_term": "block",
        "default_hook": "mlp_out",
        "step_debugger": True,
        "step_unit": "token",
        "step_saved_note": "Stepping session saved as a run: logit lens, attention and KV cache are available.",
        "lora_support": True,
        "editable_components": [
            {
                "kind": "input_prompt",
                "label": "Prompt & System Turn",
                "description": "Edit system instructions, user prompt, and forced assistant prefill tokens.",
            },
            {
                "kind": "kv_import",
                "label": "KV-Cache Prefix",
                "description": "Prepend a saved conversation or hidden prompt into chosen transformer blocks.",
            },
            {
                "kind": "sampler",
                "label": "Token Sampler",
                "description": "Control greedy vs temperature sampling, top-k filtering, and max new tokens.",
            },
        ],
    }

    @classmethod
    def default_flow(cls, info: dict) -> dict:
        from ..flow import default_llm_flow
        return default_llm_flow(info)

    @classmethod
    def default_flow(cls, info: dict) -> dict:
        from ..flow import default_llm_flow
        return default_llm_flow(info)

    @classmethod
    def matches(cls, root: Path, spec: dict | None = None) -> bool:
        """Matches any folder with a Hugging Face `config.json` (causal LM) or Meta `params.json`."""
        if super().matches(root, spec):
            return True
        if (root / "model_index.json").exists():
            return False
        if (root / "params.json").exists() or (root / "original" / "params.json").exists():
            return True
        cfg = read_json(root / "config.json")
        if "text_config" in cfg:
            cfg = cfg["text_config"]
        return bool(cfg.get("hidden_size") and cfg.get("num_hidden_layers"))

    @classmethod
    def create_adapter(cls, spec: dict, root: Path, device: str, dtype: torch.dtype,
                       policy: str = "auto", keep_layers: int = 0) -> LLMAdapter:
        ad = LLMAdapter(spec, root, device, dtype, policy, keep_layers)
        ad.plugin_id = cls.id
        ad.ui_schema = cls.ui_schema
        return ad
