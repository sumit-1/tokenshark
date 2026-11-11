"""Training, small enough to watch.

1. `TinyJob` trains a small transformer from scratch (a few hundred thousand parameters, minutes on a laptop
   CPU) and reports the loss as it falls. The result is saved in the same layout as any other model
   (config.json + model.safetensors + tokenizer.json), so it shows up in the model list and every tool of the
   app — step debugger, logit lens, attention, patching — works on the model the student just trained.

2. `LoRAJob` trains a LoRA on the loaded language model: the base weights stay frozen and each chosen matrix
   gets a low-rank update  W + (α/r)·B·A.  The result is saved in PEFT format and attached, so it opens in the
   LoRA / Fine-tune inspector.

Both use `forward_train`: the same maths as llm.py, but for whole sequences at once and with gradients.
"""
from __future__ import annotations

import json
import math
import random
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .llm import apply_rope, rms_norm

# ======================================================================================
# A differentiable forward pass (full sequences, no KV cache)
# ======================================================================================


def forward_train(W, embed, unembed, s: dict, ids: torch.Tensor) -> torch.Tensor:
    """ids [batch, tokens] → logits [batch, tokens, vocab].

    W(name) returns a weight; embed(ids) looks tokens up; unembed(h) scores the vocabulary.
    s describes the architecture: n_layers, n_heads, n_kv, head_dim, eps, inv_freq, act, qk_norm, bias(name)."""
    B, T = ids.shape
    H, Hkv, hd = s["n_heads"], s["n_kv"], s["head_dim"]
    pos = torch.arange(T, dtype=torch.float32)
    causal = torch.full((T, T), float("-inf"), device=ids.device).triu(1)       # a token can't look ahead

    def lin(x, name):
        y = x @ W(name + ".weight").T
        return y + W(name + ".bias") if s["bias"](name) else y

    x = embed(ids)                                                             # [B, T, d]
    for l in s.get("schedule") or range(s["n_layers"]):
        pre = f"model.layers.{l}."
        h = rms_norm(x, W(pre + "input_layernorm.weight"), s["eps"])
        q = lin(h, pre + "self_attn.q_proj").view(B, T, H, hd).transpose(1, 2)        # [B, H, T, hd]
        k = lin(h, pre + "self_attn.k_proj").view(B, T, Hkv, hd).transpose(1, 2)
        v = lin(h, pre + "self_attn.v_proj").view(B, T, Hkv, hd).transpose(1, 2)
        if s["qk_norm"]:
            q = rms_norm(q, W(pre + "self_attn.q_norm.weight"), s["eps"])
            k = rms_norm(k, W(pre + "self_attn.k_norm.weight"), s["eps"])
        q, k = apply_rope(q, pos, s["inv_freq"]), apply_rope(k, pos, s["inv_freq"])
        if H != Hkv:
            k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd) + causal, -1)
        x = x + lin((attn @ v).transpose(1, 2).reshape(B, T, H * hd), pre + "self_attn.o_proj")
        h = rms_norm(x, W(pre + "post_attention_layernorm.weight"), s["eps"])
        x = x + lin(s["act"](lin(h, pre + "mlp.gate_proj")) * lin(h, pre + "mlp.up_proj"), pre + "mlp.down_proj")
        if l in s.get("adapters", ()):            # a thin bottleneck layer after this block (adapt.py)
            x = x + s["act"](x @ W(pre + "adapter.down.weight").T) @ W(pre + "adapter.up.weight").T
    return unembed(rms_norm(x, W("model.norm.weight"), s["eps"]))


def next_token_loss(logits: torch.Tensor, ids: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy of predicting token t+1 from position t. weight [B, T-1] masks tokens that don't count."""
    lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1), reduction="none")
    if weight is None:
        return lp.mean()
    w = weight.reshape(-1).float()
    return (lp * w).sum() / w.sum().clamp(min=1)


# ======================================================================================
# Job base: a background thread with a loss curve
# ======================================================================================

class Job:
    kind = "job"

    def __init__(self):
        self.status = "queued"            # queued · running · done · stopped · error
        self.step, self.steps = 0, 0
        self.losses: list[list[float]] = []       # [step, train loss]
        self.val: list[list[float]] = []          # [step, validation loss]
        self.samples: list[dict] = []             # text the model writes as training goes on
        self.message, self.error = "", None
        self.started = time.time()
        self.seconds = 0.0
        self.stop = False
        self.result: dict = {}

    def summary(self) -> dict:
        return {"kind": self.kind, "status": self.status, "step": self.step, "steps": self.steps,
                "losses": self.losses, "val": self.val, "samples": self.samples[-6:], "message": self.message,
                "error": self.error, "seconds": round(self.seconds, 1), "result": self.result, **self.extra()}

    def extra(self) -> dict:
        return {}

    def start(self):
        def work():
            self.status = "running"
            try:
                with torch.enable_grad():
                    self.run()
                self.status = "stopped" if self.stop else "done"
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.status, self.error = "error", f"{type(e).__name__}: {e}"
            self.seconds = time.time() - self.started

        threading.Thread(target=work, daemon=True).start()
        return self

    def run(self):
        raise NotImplementedError


JOBS: dict[str, Job] = {}


# ======================================================================================
# 1 · A tiny transformer from scratch
# ======================================================================================

COUNTRIES = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
             ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo"), ("Greece", "Athens"),
             ("India", "Delhi"), ("China", "Beijing"), ("Russia", "Moscow"), ("Cuba", "Havana"), ("Ireland", "Dublin"),
             ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Portugal", "Lisbon"), ("Canada", "Ottawa"), ("Chile", "Santiago")]
ANIMALS = [("dog", "barks"), ("cat", "purrs"), ("cow", "moos"), ("duck", "quacks"), ("lion", "roars"), ("owl", "hoots"),
           ("bee", "buzzes"), ("frog", "croaks")]


def synthetic_corpus(kind: str = "mixed", n_lines: int = 6000, seed: int = 0) -> str:
    """A small made-up language with rules a tiny model can learn completely — so that afterwards you can look
    for *where* it stores each rule. facts: capitals and animal sounds · counting: sequences and sums."""
    rnd = random.Random(seed)
    lines = []
    for _ in range(n_lines):
        r = rnd.random()
        if kind in ("facts", "mixed") and (kind == "facts" or r < 0.55):
            c, cap = rnd.choice(COUNTRIES)
            a, snd = rnd.choice(ANIMALS)
            lines.append(rnd.choice([f"The capital of {c} is {cap}.", f"{cap} is the capital of {c}.",
                                     f"Q: What is the capital of {c}? A: {cap}.", f"The {a} {snd}.",
                                     f"Q: What does the {a} do? A: It {snd}."]))
        else:
            a, b = rnd.randint(0, 9), rnd.randint(0, 9)
            start = rnd.randint(0, 20)
            step = rnd.choice([1, 1, 2, 5])
            lines.append(rnd.choice([f"{a} + {b} = {a + b}.", " ".join(str(start + i * step) for i in range(6)) + ".",
                                     f"After {start} comes {start + 1}."]))
    return "\n".join(lines) + "\n"


class TinyJob(Job):
    kind = "tiny"

    def __init__(self, name: str, out_dir: Path, text: str = "", corpus: str = "mixed", d_model: int = 96,
                 n_layers: int = 3, n_heads: int = 4, context: int = 64, vocab: int = 300, steps: int = 1500,
                 batch: int = 32, lr: float = 3e-3, seed: int = 0, sample_prompt: str = "The capital of France is"):
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model must divide by the number of heads, and the head size must be even (RoPE pairs).")
        self.name, self.out = name, Path(out_dir)
        self.text = text if text.strip() else synthetic_corpus(corpus, seed=seed)
        self.corpus = "your text" if text.strip() else corpus
        self.cfg = {"d": int(d_model), "L": int(n_layers), "H": int(n_heads), "ctx": int(context), "vocab": int(vocab),
                    "ff": 2 * int(d_model)}
        self.steps, self.batch, self.lr, self.seed = int(steps), int(batch), float(lr), int(seed)
        self.sample_prompt = sample_prompt
        self.n_params = 0

    def extra(self):
        return {"name": self.name, "config": self.cfg, "n_params": self.n_params, "corpus": self.corpus,
                "corpus_chars": len(self.text)}

    # the parameters, under the names every other model uses
    def init_params(self, V: int) -> dict[str, torch.Tensor]:
        c = self.cfg
        g = torch.Generator().manual_seed(self.seed)
        rn = lambda *shape, std=0.02: (torch.randn(*shape, generator=g) * std).requires_grad_()      # noqa: E731
        P = {"model.embed_tokens.weight": rn(V, c["d"]), "model.norm.weight": torch.ones(c["d"], requires_grad=True)}
        out_std = 0.02 / math.sqrt(2 * c["L"])                 # smaller start for what is added to the stream
        for l in range(c["L"]):
            pre = f"model.layers.{l}."
            P[pre + "input_layernorm.weight"] = torch.ones(c["d"], requires_grad=True)
            P[pre + "post_attention_layernorm.weight"] = torch.ones(c["d"], requires_grad=True)
            for n in ("q_proj", "k_proj", "v_proj"):
                P[pre + f"self_attn.{n}.weight"] = rn(c["d"], c["d"])
            P[pre + "self_attn.o_proj.weight"] = rn(c["d"], c["d"], std=out_std)
            P[pre + "mlp.gate_proj.weight"] = rn(c["ff"], c["d"])
            P[pre + "mlp.up_proj.weight"] = rn(c["ff"], c["d"])
            P[pre + "mlp.down_proj.weight"] = rn(c["d"], c["ff"], std=out_std)
        return P

    def run(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        c = self.cfg
        torch.manual_seed(self.seed)
        self.message = "training the tokenizer"
        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator([self.text[i:i + 4000] for i in range(0, len(self.text), 4000)],
                               trainers.BpeTrainer(vocab_size=c["vocab"], special_tokens=["<|endoftext|>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        V = tk.get_vocab_size()
        data = torch.tensor(tk.encode(self.text).ids, dtype=torch.long)
        if len(data) < 4 * c["ctx"]:
            raise ValueError("The text is too short to train on (need a few hundred tokens at least).")
        n_val = max(c["ctx"] + 1, len(data) // 20)
        train, val = data[:-n_val], data[-n_val:]
        P = self.init_params(V)
        self.n_params = sum(p.numel() for p in P.values())
        spec = {"n_layers": c["L"], "n_heads": c["H"], "n_kv": c["H"], "head_dim": c["d"] // c["H"], "eps": 1e-6,
                "inv_freq": 1.0 / (10000.0 ** (torch.arange(0, c["d"] // c["H"], 2, dtype=torch.float32) / (c["d"] // c["H"]))),
                "act": F.silu, "qk_norm": False, "bias": lambda name: False}
        E = P["model.embed_tokens.weight"]
        fwd = lambda ids: forward_train(P.__getitem__, lambda i: E[i], lambda h: h @ E.T, spec, ids)     # noqa: E731
        opt = torch.optim.AdamW(P.values(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95))
        g = torch.Generator().manual_seed(self.seed + 1)

        def batch_of(src, n):
            i = torch.randint(0, len(src) - c["ctx"] - 1, (n,), generator=g)
            return torch.stack([src[j:j + c["ctx"] + 1] for j in i])

        def sample(n=40):
            ids = tk.encode(self.sample_prompt).ids or [0]
            with torch.no_grad():
                for _ in range(n):
                    lg = fwd(torch.tensor([ids[-c["ctx"]:]]))[0, -1]
                    ids.append(int(lg.argmax()))
            return tk.decode(ids)

        warm = max(1, self.steps // 20)
        every = max(1, self.steps // 60)
        t0 = time.time()
        for step in range(1, self.steps + 1):
            if self.stop:
                break
            for grp in opt.param_groups:        # warm up, then cosine down
                grp["lr"] = self.lr * (step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, self.steps - warm))))
            x = batch_of(train, self.batch)
            loss = next_token_loss(fwd(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(P.values(), 1.0)
            opt.step()
            self.step, self.seconds = step, time.time() - t0
            if step % every == 0 or step == 1 or step == self.steps:
                self.losses.append([step, round(loss.item(), 4)])
                with torch.no_grad():
                    xv = batch_of(val, min(self.batch, 16))
                    self.val.append([step, round(float(next_token_loss(fwd(xv), xv)), 4)])
                self.message = f"step {step}/{self.steps} · loss {loss.item():.3f} · {step * self.batch * c['ctx'] / max(1e-6, self.seconds):,.0f} tokens/s"
            if step % max(1, self.steps // 6) == 0 or step == self.steps:
                self.samples.append({"step": step, "text": sample()})
        self.save(tk, P, V)

    def save(self, tk, P: dict, V: int):
        from safetensors.torch import save_file
        c = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        tk.save(str(self.out / "tokenizer.json"))
        (self.out / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>", "bos_token": None}))
        (self.out / "config.json").write_text(json.dumps({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": c["d"], "intermediate_size": c["ff"],
            "num_hidden_layers": c["L"], "num_attention_heads": c["H"], "num_key_value_heads": c["H"], "vocab_size": V,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": c["ctx"], "hidden_act": "silu",
            "tie_word_embeddings": True, "eos_token_id": 0,
            "ai_analyser": {"trained_here": True, "steps": self.step, "corpus": self.corpus,
                            "final_loss": self.losses[-1][1] if self.losses else None}}, indent=1))
        save_file({k: v.detach().contiguous() for k, v in P.items()}, str(self.out / "model.safetensors"))
        self.result = {"model_id": self.out.name, "path": str(self.out), "n_params": self.n_params, "vocab": V}
        self.message = f"saved as model “{self.out.name}” — pick it in the model list and press Load"


# ======================================================================================
# 2 · A LoRA on the loaded model
# ======================================================================================

class LoRAJob(Job):
    kind = "lora"

    def __init__(self, loaded, tasks, out_dir: Path, name: str, examples: list[str], rank: int = 8, alpha: float = 16.0,
                 targets: tuple = ("q_proj", "v_proj"), steps: int = 60, lr: float = 1e-3, max_tokens: int = 96,
                 seed: int = 0, probe: str = "", attach: bool = True, layers: str = "all"):
        super().__init__()
        self.layers = str(layers or "all")          # which blocks get an add-on: all | 2 | 0:4 | 1,5
        self.loaded, self.tasks, self.out, self.name = loaded, tasks, Path(out_dir), name
        self.examples = [e for e in (x.strip() for x in examples) if e]
        if not self.examples:
            raise ValueError("Give at least one training example.")
        self.rank, self.alpha, self.targets = int(rank), float(alpha), tuple(targets)
        self.steps, self.lr, self.max_tokens, self.seed = int(steps), float(lr), int(max_tokens), int(seed)
        self.probe, self.attach = probe, attach
        ad = loaded.adapter
        if ad is None or getattr(ad, "family", "") != "causal_lm":
            raise ValueError("Load a language model first.")
        if ad.cfg.layer_types or ad.model.attn_gate:
            raise ValueError("LoRA training here supports the standard attention models (Llama, Qwen2/2.5/3, SmolLM, "
                             "and models trained in the Training tab), not the hybrid Qwen3.5/3.6 blocks yet.")
        if ad.model.W.total_bytes > 6 * 2**30:
            raise ValueError("This model is too large to train here on a laptop; use a model under about 6 GB.")
        self.n_params = 0

    def extra(self):
        return {"name": self.name, "rank": self.rank, "alpha": self.alpha, "targets": list(self.targets),
                "layers": self.layers, "n_params": self.n_params, "n_examples": len(self.examples)}

    def encode(self, ad, text: str):
        """ids and a 0/1 weight per token. 'question => answer' goes through the chat template and only the answer
        is learned; any other line is learned whole."""
        tok = ad.tok
        if " => " in text and tok.chat_template:
            q, a = text.split(" => ", 1)
            head = tok.encode(tok.apply_chat([{"role": "user", "content": q.strip()}], True), add_special=False)
            tail = tok.encode(a.strip(), add_special=False) + ([tok.eos_id] if tok.eos_id is not None else [])
            ids, w = head + tail, [0] * len(head) + [1] * len(tail)
        else:
            ids = tok.encode(text, add_special=True)
            w = [1] * len(ids)
        return ids[: self.max_tokens], w[: self.max_tokens]

    def generate(self, ad, prompt: str, n: int = 24) -> str:
        from .flow import compile_flow, default_llm_flow
        from .flowbuild import FlowBuilder
        info = {**ad.info(), "model_id": ad.spec.get("id")}
        fb = FlowBuilder(default_llm_flow(info)).prompt(prompt=prompt).sampler(max_new_tokens=n, temperature=0.0)
        with torch.no_grad():
            return ad.run(compile_flow(fb.to_dict(), info))["outputs"]["text"]

    def run(self):
        from safetensors.torch import save_file
        loaded, ad = self.loaded, self.loaded.adapter
        m, cfg = ad.model, ad.cfg
        task = self.tasks.new("run", f"train LoRA “{self.name}”")
        task.on_cancel = lambda: setattr(self, "stop", True)
        try:
            if self.probe:
                with self.tasks.slot(task):
                    self.samples.append({"step": 0, "text": self.generate(ad, self.probe), "label": "before"})
            torch.manual_seed(self.seed)
            # LoRA parameters: A starts random, B at zero, so training starts exactly at the base model.
            # A LoRA is not one layer: every chosen matrix (q_proj, v_proj …) in every chosen block gets its own
            # small pair (A, B). The classic recipe adapts q and v in all blocks; choosing one block shows how
            # much a single place can do.
            from .flow import parse_index_spec
            A, B, names = {}, {}, []
            for l in parse_index_spec(self.layers, cfg.n_layers):
                for t in self.targets:
                    part = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                    name = f"model.layers.{l}.{part}.{t}.weight"
                    shape = m.W.shape(name)
                    if not shape:
                        continue
                    names.append(name)
                    A[name] = (torch.randn(self.rank, shape[1]) / math.sqrt(shape[1])).to(m.device).requires_grad_()
                    B[name] = torch.zeros(shape[0], self.rank, device=m.device, requires_grad=True)
            if not names:
                raise ValueError(f"Nothing to train: no matrices {self.targets} in blocks '{self.layers}' of this model.")
            self.n_params = sum(p.numel() for p in list(A.values()) + list(B.values()))
            scale = self.alpha / self.rank

            def W(name):
                w = (m.norm_w(name) if name.endswith("norm.weight") else m.W(name)).float()
                return w + scale * (B[name] @ A[name]) if name in A else w

            w_out = m.unembed_weight().to(m.device, torch.float32)
            spec = {"n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "n_kv": cfg.n_kv, "head_dim": cfg.head_dim,
                    "eps": cfg.eps, "inv_freq": m.inv_freq.float().cpu(), "act": m.act, "qk_norm": m.qk_norm,
                    "bias": lambda name: m.has_bias and m.W.has(name + ".bias")}
            embed = lambda ids: m.W.rows("model.embed_tokens.weight", ids.flatten().tolist()).float().view(*ids.shape, -1)   # noqa: E731
            data = [self.encode(ad, e) for e in self.examples]
            data = [(i, w) for i, w in data if len(i) >= 2 and sum(w[1:]) > 0]
            if not data:
                raise ValueError("The examples are too short to learn from.")
            opt = torch.optim.AdamW(list(A.values()) + list(B.values()), lr=self.lr, weight_decay=0.0)
            rnd = random.Random(self.seed)
            t0 = time.time()
            for step in range(1, self.steps + 1):
                if self.stop:
                    break
                ids, w = rnd.choice(data)
                x = torch.tensor([ids], device=m.device)
                with self.tasks.slot(task):          # one step at a time, so the rest of the app stays usable
                    if loaded.adapter is not ad:
                        raise RuntimeError("The model was unloaded during training.")
                    with torch.enable_grad():
                        loss = next_token_loss(forward_train(W, embed, lambda h: h @ w_out.T, spec, x), x,
                                               torch.tensor([w[1:]], device=m.device))
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                self.step, self.seconds = step, time.time() - t0
                self.losses.append([step, round(loss.item(), 4)])
                self.message = f"step {step}/{self.steps} · loss {loss.item():.3f}"
                task.progress, task.message = step / self.steps, self.message
            # Save in PEFT format, so any tool that reads LoRAs can open it.
            self.out.mkdir(parents=True, exist_ok=True)
            for stale in ("adaptation.safetensors", "adaptation.json"):       # an earlier run under this name
                (self.out / stale).unlink(missing_ok=True)
            tensors = {}
            for name in names:
                stem = "base_model.model." + name[: -len(".weight")]
                tensors[stem + ".lora_A.weight"] = A[name].detach().cpu().contiguous()
                tensors[stem + ".lora_B.weight"] = B[name].detach().cpu().contiguous()
            save_file(tensors, str(self.out / "adapter_model.safetensors"))
            (self.out / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "r": self.rank, "lora_alpha": self.alpha, "target_modules": list(self.targets),
                "base_model_name_or_path": ad.spec.get("repo") or ad.spec.get("id"), "task_type": "CAUSAL_LM",
                "layers_to_transform": parse_index_spec(self.layers, cfg.n_layers) if self.layers.strip().lower() != "all" else None,
                "ai_analyser": {"steps": self.step, "examples": self.examples[:50]}}, indent=1))
            self.result = {"path": str(self.out), "n_params": self.n_params}
            if self.attach:
                self.result["lora"] = loaded.attach_lora(self.out, 1.0, self.name)
                if self.probe:
                    with self.tasks.slot(task):
                        self.samples.append({"step": self.step, "text": self.generate(ad, self.probe), "label": "after"})
            self.message = f"saved to {self.out}" + (" and attached — see the LoRA / Fine-tune tab" if self.attach else "")
            self.tasks.finish(task, "cancelled" if self.stop else "done")
        except Exception as e:
            self.tasks.finish(task, error=f"{type(e).__name__}: {e}")
            raise
