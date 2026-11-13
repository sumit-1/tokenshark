"""Build tiny random-weight models in Hugging Face / diffusers layout for tests and offline demos.

    python tests/make_tiny.py [out_dir]

Needs transformers + diffusers (test-only dependencies; the app itself does not use them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

CORPUS = ("The capital of France is Paris. The capital of Japan is Tokyo. A red fox sat in the snowy forest. "
          "Students learn how attention works, how the residual stream carries information, and how a KV cache "
          "stores keys and values. Hello world! 1 2 3 4 5 6 7 8 9 10. ") * 20

CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def make_tokenizer(out: Path, vocab=400):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator([CORPUS], tr)
    out.mkdir(parents=True, exist_ok=True)
    tk.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": CHAT, "eos_token": "<|im_end|>", "bos_token": None}))
    return tk.get_vocab_size()


def make_llm(out: Path, arch="qwen2", seed=0):
    import transformers as T
    torch.manual_seed(seed)
    vocab = make_tokenizer(out)
    common = dict(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, rms_norm_eps=1e-6,
                  eos_token_id=2)
    if arch == "qwen2":
        cfg = T.Qwen2Config(**common, rope_theta=10000.0, tie_word_embeddings=True)
        m = T.Qwen2ForCausalLM(cfg)
    elif arch == "qwen3":
        cfg = T.Qwen3Config(**common, head_dim=16, rope_theta=10000.0, tie_word_embeddings=False)
        m = T.Qwen3ForCausalLM(cfg)
    elif arch in ("qwen3_5", "qwen3_5_vl"):
        tc = dict(common, head_dim=32, layer_types=["linear_attention", "linear_attention", "full_attention",
                                                   "linear_attention", "full_attention"],
                  num_hidden_layers=5, linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
                  linear_num_key_heads=2, linear_num_value_heads=4, tie_word_embeddings=False,
                  rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                   "mrope_section": [2, 1, 1]})
        if arch == "qwen3_5":
            m = T.Qwen3_5ForCausalLM(T.Qwen3_5TextConfig(**tc))
        else:
            cfg = T.Qwen3_5Config(text_config=tc, vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                                  num_heads=2, out_hidden_size=64, num_position_embeddings=64, patch_size=4))
            m = T.Qwen3_5ForConditionalGeneration(cfg)
        with torch.no_grad():   # make the linear-attention gates and norms non-trivial
            for name, p in m.named_parameters():
                if name.endswith(("A_log", "dt_bias")) or "norm" in name:
                    p.copy_(torch.randn_like(p) * 0.5)
    else:  # llama with llama3 rope scaling
        cfg = T.LlamaConfig(**common, rope_theta=500000.0, tie_word_embeddings=False,
                            rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                          "high_freq_factor": 4.0, "original_max_position_embeddings": 64})
        m = T.LlamaForCausalLM(cfg)
    with torch.no_grad():   # random init is small; scale up so layers actually matter
        for p in m.parameters():
            if p.ndim == 2:
                p.mul_(3.0)
    m.save_pretrained(out, safe_serialization=True)
    return m


class _QIProc:
    """Text-only stand-in for Qwen3VLProcessor (same tokenizer + chat template; no torchvision needed)."""
    def __init__(self, tok):
        self.tokenizer = tok

    def apply_chat_template(self, msgs, tokenize=True, return_dict=False, **kw):
        txt = self.tokenizer.apply_chat_template(msgs, tokenize=False, **kw)
        return [self.tokenizer(txt)["input_ids"]] if tokenize else txt

    def __call__(self, text=None, padding=True, padding_side="left", return_tensors="pt", **kw):
        self.tokenizer.padding_side = padding_side
        return self.tokenizer(text, padding=padding, return_tensors=return_tensors)


def make_qwen_image(out: Path):
    """Tiny Qwen-Image 2.1 in the hub layout (transformer / text_encoder / vae / scheduler / processor)."""
    import shutil
    import transformers as T
    import diffusers as D
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    sp = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=sp, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator([CORPUS], tr)
    tmp = out/'_tok'; tmp.mkdir(exist_ok=True); tk.save(str(tmp/'tokenizer.json'))
    fast = T.PreTrainedTokenizerFast(tokenizer_file=str(tmp/'tokenizer.json'), eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.add_special_tokens({"additional_special_tokens": sp[3:]})
    CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['content'] is string %}{{ m['content'] }}{% else %}"
            "{% for c in m['content'] %}{% if c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    fast.chat_template = CHAT
    proc = _QIProc(fast)
    torch.manual_seed(0)
    V = len(fast)
    tc = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=5000000.0, max_position_embeddings=512,
              rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True})
    cfg = T.Qwen3VLConfig(text_config=tc, vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
          out_hidden_size=64, patch_size=16, spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0],
          num_position_embeddings=64), image_token_id=fast.convert_tokens_to_ids("<|image_pad|>"),
          video_token_id=fast.convert_tokens_to_ids("<|video_pad|>"),
          vision_start_token_id=fast.convert_tokens_to_ids("<|vision_start|>"), vision_end_token_id=fast.convert_tokens_to_ids("<|vision_end|>"))
    te = T.Qwen3VLForConditionalGeneration(cfg)
    with torch.no_grad():
        for n, p in te.named_parameters():
            if p.ndim == 2: p.mul_(3.0)
            elif 'norm' in n: p.copy_(1 + 0.3 * torch.randn_like(p))
    tr_ = D.QwenImage21Transformer2DModel(patch_size=1, in_channels=8, out_channels=8, num_layers=3, attention_head_dim=32,
          num_attention_heads=2, context_in_dim=64, mlp_ratio=3, axes_dims_rope=(8, 12, 12))
    vae = D.AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=12, z_dim=8, dim_mult=[1, 2, 2, 2, 2], num_res_blocks=1,
          temperal_downsample=[False, True, True, True], in_channels=4, out_channels=4, is_residual=True,
          latents_mean=[0.1 * i for i in range(8)], latents_std=[1 + 0.1 * i for i in range(8)])
    with torch.no_grad():
        for m in (tr_, vae):
            for n, p in m.named_parameters():
                p.copy_(torch.randn_like(p) * (0.3 if p.ndim < 2 else 1.0 / max(1, p[0].numel()) ** 0.5))
    sch = D.FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9,
          num_train_timesteps=1000, shift=1.0, shift_terminal=0.02, time_shift_type="exponential", use_dynamic_shifting=True)
    for name, m in (("transformer", tr_), ("vae", vae), ("text_encoder", te), ("scheduler", sch)):
        m.save_pretrained(out / name)
    fast.save_pretrained(out / "processor")
    (out / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline", "_diffusers_version": "0.37.0.dev0",
        "processor": ["transformers", "Qwen3VLProcessor"], "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"], "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"]}))
    shutil.rmtree(tmp)
    return fast, proc


def make_sd(out: Path, linear_proj=False, seed=0):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel
    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    # CLIP-style vocab: byte alphabet, with and without </w>, a few merges, then specials.
    from tokenshark.tokenizer import _bytes_to_unicode
    chars = list(_bytes_to_unicode().values())
    vocab = chars + [c + "</w>" for c in chars]
    merges = ["r e", "re d</w>", "f o", "fo x</w>", "s n", "o w</w>", "t h", "th e</w>"]
    for m in merges:
        a, b = m.split()
        if a + b not in vocab:
            vocab.append(a + b)
    vocab += ["<|startoftext|>", "<|endoftext|>"]
    tok = out / "tokenizer"
    tok.mkdir(exist_ok=True)
    (tok / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}))
    (tok / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n")
    (tok / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 77, "pad_token": "<|endoftext|>"}))
    te = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=77,
                                      hidden_act="quick_gelu", bos_token_id=len(vocab) - 2, eos_token_id=len(vocab) - 1))
    te.save_pretrained(out / "text_encoder", safe_serialization=True)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=2, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=(4, 8) if linear_proj else 4, norm_num_groups=8,
        use_linear_projection=linear_proj)
    unet.save_pretrained(out / "unet", safe_serialization=True)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 64),
                        down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                        norm_num_groups=8, sample_size=32)
    vae.save_pretrained(out / "vae", safe_serialization=True)
    sch = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                 timestep_spacing="trailing" if linear_proj else "leading", steps_offset=1)
    sch.save_pretrained(out / "scheduler")
    (out / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
    return te, unet, vae, sch


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "tiny")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    make_llm(root / "llm-qwen2", "qwen2")
    make_sd(root / "sd-tiny")
    print("wrote", root)
