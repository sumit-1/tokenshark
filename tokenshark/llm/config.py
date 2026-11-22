"""LLM checkpoint configuration & Meta-format conversion (`tokenshark.llm.config`)."""
from __future__ import annotations

from pathlib import Path
from ..weights import read_json


class LLMConfig:
    def __init__(self, root: Path):
        hf = read_json(root / "config.json")
        self.meta_format = False
        self.layer_types = None          # hybrid models: "linear_attention" | "full_attention" per layer
        self.norm_offset = 0.0           # Qwen3.5/3.6 store RMSNorm weights as (w - 1): scale = 1 + w
        self.rot_dim = None              # partial rotary: only the first rot_dim dims of each head rotate
        self.vision = None
        self.moe = None                  # mixture of experts: {"n": experts, "k": per token, "norm_topk", "inter"}
        self.mla = None                  # DeepSeek multi-head latent attention: {"nope", "rope", "v", "rank"}
        self.clip_qkv = None
        if hf and "text_config" in hf:   # multimodal checkpoint: the language model lives in text_config
            self.vision = hf.get("vision_config")
            self.vlm_tokens = {k: hf.get(k) for k in ("image_token_id", "video_token_id", "vision_start_token_id",
                                                     "vision_end_token_id")}
            arch = (hf.get("architectures") or [""])[0]
            tc = dict(hf["text_config"])
            tc.setdefault("architectures", [arch])
            tc.setdefault("tie_word_embeddings", hf.get("tie_word_embeddings", False))
            tc["model_type"] = tc.get("model_type") or hf.get("model_type")
            hf = tc
        if hf:
            n_exp = int(hf.get("num_experts") or hf.get("num_local_experts") or hf.get("n_routed_experts") or 0)
            if n_exp and "intermediate_size" not in hf:
                hf["intermediate_size"] = hf.get("moe_intermediate_size") or hf.get("shared_expert_intermediate_size")
            missing = [k for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                                   "vocab_size") if not hf.get(k)]
            if missing:
                raise ValueError(f"config.json in {root} is missing {', '.join(missing)} - this checkpoint layout isn't "
                                 f"supported yet. Top-level keys: {', '.join(sorted(hf)[:30])}")
            self.arch = (hf.get("architectures") or ["LlamaForCausalLM"])[0]
            self.model_type = hf.get("model_type", "llama")
            self.hidden = hf["hidden_size"]
            self.n_layers = hf["num_hidden_layers"]
            self.n_heads = hf["num_attention_heads"]
            self.n_kv = hf.get("num_key_value_heads") or self.n_heads
            self.head_dim = hf.get("head_dim") or self.hidden // self.n_heads
            self.inter = hf["intermediate_size"]
            self.eps = hf.get("rms_norm_eps", 1e-6)
            self.vocab = hf["vocab_size"]
            self.theta = hf.get("rope_theta") or (hf.get("rope_parameters") or {}).get("rope_theta") or 10000.0
            self.rope_scaling = hf.get("rope_scaling") or hf.get("rope_parameters")
            self.tied = bool(hf.get("tie_word_embeddings", False))
            self.sliding_window = hf.get("sliding_window") if hf.get("use_sliding_window", self.model_type == "mistral") else None
            self.act = hf.get("hidden_act", "silu")
            self.attn_bias = bool(hf.get("attention_bias", self.model_type == "qwen2"))
            self.eos = hf.get("eos_token_id")
            self.max_pos = hf.get("max_position_embeddings", 4096)
            rp = hf.get("rope_parameters") or hf.get("rope_scaling") or {}
            if rp.get("partial_rotary_factor") or hf.get("partial_rotary_factor"):
                self.rot_dim = int(self.head_dim * float(rp.get("partial_rotary_factor") or hf["partial_rotary_factor"]))
            self.clip_qkv = hf.get("clip_qkv")                       # OLMoE clamps q, k, v to ±clip_qkv
            if n_exp:
                mt = self.model_type
                self.moe = {"n": n_exp, "k": int(hf.get("num_experts_per_tok") or 2),
                            "norm_topk": True if mt == "mixtral" or mt.startswith("qwen3_5")
                            else bool(hf.get("norm_topk_prob", False)),
                            "inter": int(hf.get("moe_intermediate_size") or hf["intermediate_size"]),
                            "aux_coef": hf.get("router_aux_loss_coef"),
                            "scale": float(hf.get("routed_scaling_factor") or 1.0),
                            "groups": ((int(hf.get("n_group") or 1), int(hf.get("topk_group") or 1))
                                       if hf.get("topk_method") == "group_limited_greedy" else None)}
            if self.model_type == "deepseek_v2":
                self.mla = {"nope": int(hf["qk_nope_head_dim"]), "rope": int(hf["qk_rope_head_dim"]),
                            "v": int(hf["v_head_dim"]), "rank": int(hf["kv_lora_rank"]), "q_rank": hf.get("q_lora_rank")}
                self.head_dim = self.mla["nope"] + self.mla["rope"]
                self.rot_dim, self.n_kv = self.mla["rope"], self.n_heads
            if self.model_type.startswith(("qwen3_5", "qwen3_next")):
                self.norm_offset = 1.0
                lt = hf.get("layer_types")
                if not lt:
                    every = hf.get("full_attention_interval", 4)
                    lt = ["linear_attention" if (i + 1) % every else "full_attention" for i in range(self.n_layers)]
                self.layer_types = lt
                self.lin = {"conv": hf.get("linear_conv_kernel_dim", 4), "dk": hf.get("linear_key_head_dim", 128),
                            "dv": hf.get("linear_value_head_dim", 128), "hk": hf.get("linear_num_key_heads", 16),
                            "hv": hf.get("linear_num_value_heads", 32)}
            elif self.model_type not in ("llama", "qwen2", "qwen3", "mistral", "olmo2", "qwen3_vl", "qwen3_vl_text",
                                         "mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "tiny_moe", "deepseek_v2") \
                    and "Llama" not in self.arch:
                raise ValueError(f"Architecture '{self.arch}' is not supported by the raw engine yet "
                                 f"(supported: Llama, Qwen2/2.5, Qwen3, Qwen3.5/3.6, Mistral, SmolLM; mixture of "
                                 f"experts: Mixtral, Qwen1.5/2-MoE, Qwen3-MoE, Qwen3.5-MoE, OLMoE, DeepSeek-V2).")
        else:
            p = read_json(root / "params.json")
            if not p:
                raise FileNotFoundError(f"No config.json or params.json in {root}")
            self.meta_format = True
            self.arch, self.model_type = "Llama (Meta format)", "llama"
            self.hidden, self.n_layers, self.n_heads = p["dim"], p["n_layers"], p["n_heads"]
            self.n_kv = p.get("n_kv_heads") or self.n_heads
            self.head_dim = self.hidden // self.n_heads
            hidden = int(2 * 4 * self.hidden / 3)
            if p.get("ffn_dim_multiplier"):
                hidden = int(p["ffn_dim_multiplier"] * hidden)
            m = p.get("multiple_of", 256)
            self.inter = m * ((hidden + m - 1) // m)
            self.eps = p.get("norm_eps", 1e-5)
            self.vocab = p.get("vocab_size", 128256)
            self.theta = p.get("rope_theta", 500000.0)
            self.rope_scaling = ({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                  "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
                                 if p.get("use_scaled_rope") else None)
            self.tied, self.sliding_window, self.act, self.attn_bias = False, None, "silu", False
            self.eos, self.max_pos = None, 8192
        gen = read_json(root / "generation_config.json")
        self.eos_ids = set()
        for e in (self.eos, gen.get("eos_token_id")):
            if isinstance(e, list):
                self.eos_ids.update(e)
            elif e is not None:
                self.eos_ids.add(e)

    def is_linear(self, layer: int) -> bool:
        return bool(self.layer_types) and self.layer_types[layer] == "linear_attention"

    def fingerprint(self) -> dict:
        return {"arch": self.arch, "hidden_size": self.hidden, "n_layers": self.n_layers, "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv, "head_dim": self.head_dim, "vocab": self.vocab,
                "hybrid": bool(self.layer_types), "moe": self.moe["n"] if self.moe else 0}


def meta_name_map(cfg: LLMConfig) -> dict[str, str]:
    m = {"model.embed_tokens.weight": "tok_embeddings.weight", "model.norm.weight": "norm.weight",
         "lm_head.weight": "output.weight"}
    for i in range(cfg.n_layers):
        a, b = f"model.layers.{i}.", f"layers.{i}."
        m.update({a + "input_layernorm.weight": b + "attention_norm.weight",
                  a + "post_attention_layernorm.weight": b + "ffn_norm.weight",
                  a + "self_attn.q_proj.weight": b + "attention.wq.weight",
                  a + "self_attn.k_proj.weight": b + "attention.wk.weight",
                  a + "self_attn.v_proj.weight": b + "attention.wv.weight",
                  a + "self_attn.o_proj.weight": b + "attention.wo.weight",
                  a + "mlp.gate_proj.weight": b + "feed_forward.w1.weight",
                  a + "mlp.down_proj.weight": b + "feed_forward.w2.weight",
                  a + "mlp.up_proj.weight": b + "feed_forward.w3.weight"})
    return m


def meta_transform(cfg: LLMConfig):
    """Meta checkpoints rotate interleaved pairs; permute q/k rows into the half-split layout used here."""
    def permute(w, n_heads):
        d_out, d_in = w.shape
        return w.view(n_heads, d_out // n_heads // 2, 2, d_in).transpose(1, 2).reshape(d_out, d_in)

    def tf(name, t):
        if name.endswith("q_proj.weight"):
            return permute(t, cfg.n_heads)
        if name.endswith("k_proj.weight"):
            return permute(t, cfg.n_kv)
        return t
    return tf
