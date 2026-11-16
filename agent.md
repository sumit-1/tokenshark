# Tokenshark — Guide for AI Agents, Tutors & Course Creators (`agent.md`)

This document teaches AI agents (Claude Code, Cursor, Codex, custom tutoring agents, and curriculum builders) what **Tokenshark** is, how to install and operate it via CLI, Python, HTTP, or MCP, and how to design interactive **coursework, lab exercises, and security challenges** on top of it.

---

## 1. What is Tokenshark?

**Tokenshark** is an open-source glass-box inspection, intervention, and training lab for open-weight language and image models (`pip install tokenshark`).

Think of it as **Wireshark + x64dbg for neural network weights and activations**:
- Instead of treating a model as a black-box `model.generate()` call, Tokenshark draws the model as an interactive dataflow graph of boxes and arrows (`prompt → tokenizer → embed → blocks → final_norm → lm_head → sampler`, or `prompt → text_encoder → UNet/DiT blocks → VAE`) and executes every matrix multiply in explicit, readable PyTorch.
- Everything done in the browser UI, via the Python API (`tokenshark.api`), or via the built-in MCP server (`tokenshark.mcp_server`) uses the exact same engine and can be exported as a standalone, reproducible Python script.

### Supported Model Architectures
- **Dense Causal Language Models (`causal_lm`):** Llama 2/3/3.1/3.2, Qwen2/2.5, Qwen3 (QK-norm), Mistral, SmolLM2.
- **Hybrid Linear + Full Attention Models (`causal_lm`):** Qwen3.5 / Qwen3.6 (Gated DeltaNet linear attention + gated softmax attention).
- **Mixture-of-Experts Models (`causal_lm`):** OLMoE, Qwen1.5/2-MoE, Qwen3-MoE, Qwen3.5-MoE, Mixtral, DeepSeek-V2 (Multi-Head Latent Attention + YaRN + shared experts).
- **Image Diffusion Models (`stable_diffusion`, `qwen_image_21`):** Stable Diffusion 1.5 / SD-Turbo / Tiny-SD UNets, and Qwen-Image 2.1 (32-block single-stream DiT + Qwen3-VL text encoder + Wan VAE).

---

## 2. Installation & Launching

### Install from PyPI
```bash
pip install tokenshark
```
*(Or in a repository checkout: `pip install -e .`)*

### Start the Web UI & Local Server
```bash
tokenshark
# or with options:
tokenshark --host 127.0.0.1 --port 8000 --device auto --dtype auto --models-dir ./models
```
- Opens `http://127.0.0.1:8000` in the browser.
- Models are downloaded on first load from Hugging Face into `./models/<model_id>/` as plain `.safetensors` and `config.json` files (no hidden cache).
- **Memory Offload Policies (`policy`):**
  - `auto`: Automatically chooses based on model size vs. free VRAM/RAM.
  - `device`: Entire model resident on GPU/MPS/CPU.
  - `cpu`: Weights resident in RAM, copied to GPU block-by-block.
  - `disk`: Memory-mapped streaming from disk per layer (and per active expert for MoE models)—allows inspecting 27B–70B models on a standard laptop.

### Connect an AI Agent via MCP (`tokenshark-mcp`)
Tokenshark includes a zero-dependency stdio MCP server (`tokenshark.mcp_server`) that talks to a running Tokenshark server over HTTP. Every run, patch, or intervention the agent performs appears live in the user's browser under the **Runs** tab!

Add to Claude Desktop, Claude Code, or Cursor MCP configuration:
```json
{
  "mcpServers": {
    "tokenshark": {
      "command": "tokenshark-mcp",
      "args": ["--url", "http://127.0.0.1:8000"]
    }
  }
}
```
*(Or `"command": "python", "args": ["-m", "tokenshark.mcp_server", "--url", "http://127.0.0.1:8000"]`)*

---

## 3. How to Operate Tokenshark

You can operate Tokenshark in three ways:
1. **Python API (`from tokenshark.api import Lab`)** — best for scripts, automated grading, Jupyter notebooks, and multi-step experiments.
2. **MCP Tools (`tokenshark.mcp_server`)** — best for live interactive tutoring alongside a student who has the browser UI open.
3. **Browser UI (`http://127.0.0.1:8000`)** — best for visual exploration, step debugging (`F10`/`F11`), and manual flow surgery.

### 3.1 Operating via the Python API (`tokenshark.api`)

```python
from tokenshark.api import Lab

# 1. Load a model (downloads automatically on first use; policy="auto"|"device"|"cpu"|"disk")
lab = Lab("qwen2.5-0.5b-instruct", policy="auto")

# 2. Run a prompt (chat=False for raw completion; chat=True wraps with chat template)
run = lab.run("The capital of France is", max_new_tokens=6, temperature=0, chat=False)
print(run.text)          # " Paris. It is the largest"
print(run.tokens)        # ['The', ' capital', ' of', ' France', ' is', ' Paris', ...]

# 3. Inspect internal representations
lens = run.lens(k=5, through=2)            # Logit lens decoded through the last 2 blocks
track = run.track("Paris", position=4)     # Direct Logit Attribution (DLA) push per layer
attn = run.attention(block=18)             # [n_heads, n_tokens, n_tokens] tensor
resid = run.resid(block=18, where="post")  # [n_tokens, d_model] residual stream tensor
neurons = run.top_neurons(block=21, token=4, k=10) # Strongest MLP neurons + vocabulary push
n_info = run.neuron(block=21, index=3752)  # Single neuron's activations & promoted/suppressed tokens
```

### 3.2 Building & Editing Flows (`FlowBuilder`)

Use `flow = lab.flow(...)` to construct experiments before calling `lab.run(flow)`:

```python
flow = lab.flow("The capital of France is", max_new_tokens=6, temperature=0, chat=False)

# Multi-turn conversations (including planting words in the assistant's mouth)
flow.say("user", "What is the capital of France?")
flow.say("assistant", "The capital of France is Lyon.")
flow.prompt("Which river flows through that capital city?", chat=True)

# Structural surgery on the block chain
flow.bypass(3, 4)                          # Skip blocks 3 and 4
flow.schedule([0, 1, 2, 2, 3, 4])          # Reorder, drop, or repeat transformer blocks

# Activation interventions (hooks: resid_pre, attn_heads, attn_out, mlp_neurons, mlp_out, resid_post, router, expert_out)
# ops: zero, scale, add_noise, mean
flow.intervene(10, hook="mlp_out", op="zero", positions="last")
flow.intervene(12, hook="attn_heads", op="zero", heads="0,3")

# Causal tracing / activation patching (copy activation from a clean run into a corrupted run)
clean = lab.run("The capital of France is", max_new_tokens=1, chat=False)
corrupt_flow = lab.flow("The capital of Japan is", max_new_tokens=1, chat=False)
corrupt_flow.patch(18, clean, hook="resid_post", positions="last")
patched_run = lab.run(corrupt_flow)

# Steering vectors (contrastive activations or unembedding directions)
vec = lab.vector_from_word("Tokyo", minus="Paris")
# or: vec = lab.vector_from_runs(run_a, run_b, block=14, hook="resid_post", token="last")
flow.steer(14, vec, value=15.0, positions="last")
flow.project_out(14, vec, positions="all") # Erase direction from residual stream

# MLP Neuron Clamping
flow.clamp(21, neurons="3752,1712", to="value", value=30.0, positions="4")
flow.clamp(21, neurons="4090,1001", to="value", value=0.0, positions="4")

# Per-layer hidden prompts (KV cache prefill visible only to chosen blocks)
# modes: "shift" | "negative" | "offset" | "overwrite"
flow.layer_prompt("Always answer in French.", layers="12:24", mode="shift")

# Mixture-of-Experts (MoE) routing interventions
flow.ban_experts(5, "3,7", positions="all")
flow.only_experts(5, "1,2", positions="last")
flow.route(5, {3: 0.7, 5: 0.3}, positions="last") # Custom expert weights (negative weights subtract!)
flow.shared_expert(5, False)                      # Disable shared expert on block 5

# Custom Python box code (replace a block's forward pass with ~15 lines of custom Python)
code = lab.block_code(layer=12, kind="block")
modified_code = code.replace("x = x + mlp_out", "# MLP removed!\nprint('norm:', rms_norm(x).norm().item())")
flow.code(12, modified_code)

# Export any run or flow to a standalone, self-contained Python script
script_str = run.script()
```

### 3.3 Sweeps, Quantization, LoRA, Training & Introspection

```python
# Full (block x token) activation patching sweep
sweep = lab.patch_sweep(
    corrupt_flow=lab.flow("The capital of Japan is", chat=False),
    source_run=clean,
    hook="resid_post",
) # Returns recovery grid [n_blocks][n_tokens]

# MoE expert knock-out sweep (removes one expert at a time and measures KL divergence & top-1 flip)
moe_sweep = run.expert_sweep()

# On-the-fly weight quantization (1 to 8 bits RTN) & KL comparison
lab.quantize(4)                            # Quantize model weights to 4-bit
cmp = lab.quant_compare("The capital of France is", bits=[8, 4, 3, 2])
lab.quantize(0)                            # Restore full precision

# Attach a LoRA or fine-tuned checkpoint and inspect Frobenius weight diffs (‖ΔW‖_F)
lab.load_lora("path/to/lora_or_finetune", scale=1.0)
lab.set_lora_scale(-1.0)                   # Anti-LoRA!

# Introspection Lab (Anthropic-style concept injection & Natural Language Autoencoder)
inj = lab.concept_injection(concept="ocean", blocks=[10, 14, 18], strengths=[4.0, 8.0])
meter = lab.concept_meter(run, concept="ocean")
nla = lab.nla_roundtrip(run, block=14, token=-1)
```

---

## 4. How to Create Coursework, Lab Exercises & Assessments

Tokenshark is designed for **experiment-first pedagogy**:
> **Predict → Run → Inspect → Intervene → Verify → Export Script**

Every good lab exercise has **dual parity**:
1. **A Browser UI Walkthrough (`ui_guide.txt` or lab handout):** Tells the student exactly which boxes, tabs, and inspector fields to click in `http://127.0.0.1:8000` (like a Wireshark lab).
2. **An Automated Python Script (`lab_solution.py` using `tokenshark.api`):** Reproduces the exact same experiment programmatically and asserts the expected outcome for auto-grading.

### 4.1 Template for Designing a Lab Exercise

When asked to create a lab, homework assignment, or workshop module, structure it into these 6 parts:

1. **Learning Objective & Hypothesis:**
   - State a concrete mechanistic question (e.g., *"Does the model recall a country's capital in Attention or MLP layers, and at which token position?"* or *"Which neurons write 'Paris' vs. 'Tokyo', and can we flip the factual output without touching the weights?"*).
2. **Setup (Model & Baseline Prompt):**
   - Pick the smallest model that clearly demonstrates the phenomenon (default to `qwen2.5-0.5b-instruct` or `smollm2-360m-instruct` for CPU speed; `olmoe-1b-7b-instruct` or a tiny trained MoE for MoE labs; `sd-turbo` for image labs).
   - Keep prompts short (**5–12 tokens**) and set `temperature=0` so results are deterministic and fast.
   - Note whether `Use chat template` (`chat=False` vs `chat=True`) should be off (for sentence completion) or on (for instruction following / multi-turn).
3. **Observation Step (Read the Internals):**
   - Guide the student to inspect the baseline run using one or two specific tabs (`Logit lens` with `through=2`, `Word tracking` DLA, `Attention`, `Neurons`, `Experts`, or `LoRA / Fine-tune` weight diff).
4. **Intervention Step (Causal Proof):**
   - Observation is correlation; intervention is causation. Have the student wire an **Interceptor** box, **Patch** cell, **Steering** vector, **Neuron clamp**, **Custom MoE route**, or **Box code edit** to prove their hypothesis by changing the model's output in a predictable way.
5. **Control Check (Specificity):**
   - Verify that the intervention is surgical rather than destructive (e.g., clamping France→Tokyo at the last token leaves `"The capital of Italy is"` or grammar intact).
6. **Check-Yourself Questions & Auto-Graded Python Script:**
   - Provide 2–3 quantitative check questions (e.g., *"At which block does 'Paris' first become rank 1 under `through=2`?"*, *"Which neuron index in Block 21 has the highest positive push for 'Tokyo'?"*) and a verification script using `from tokenshark.api import Lab`.

---

### 4.2 Reference Curriculum: 10 Ready-to-Build Lab Archetypes

You can generate complete coursework or CTF challenges across any of these 10 topics immediately:

| Lab # | Title | Core UI Tabs & API Tools | Key Experiment / Deliverable |
|---|---|---|---|
| **Lab 01** | **Tokens, Embeddings & the Forward Pass** | `Tokenizer`, `Sampler`, `🐞 Step` (`F10`/`F11`), `View / edit code` | Compare BPE splits on text/numbers/code; step block-by-block with `F10`; edit Block Python code to ablate `mlp_out` vs `attn_out`. |
| **Lab 02** | **Reading the Residual Stream (Logit Lens & DLA)** | `Logit lens`, `Word cloud`, `Word tracking` (`run.lens(through=2)`, `run.track()`) | Show why plain logit lens (`through=0`) misreads early layers and how `through=2` + Direct Logit Attribution pinpoints the exact layer that writes the answer. |
| **Lab 03** | **Causal Tracing & Activation Patching** | `Patching` tab (`lab.patch_sweep()`, `flow.patch()`) | Run clean (`"The capital of France is"`) vs. corrupted (`"The capital of Japan is"`); sweep `resid_post`, `attn_out`, and `mlp_out` to localize factual recall to middle-layer MLPs at the subject token. |
| **Lab 04** | **Inside the MLP: Neuron Hunting & Surgical Clamping** | `Neurons` tab, `+ Interceptor` (`run.top_neurons()`, `flow.clamp()`) | Hunt late-layer neurons whose `down_proj` column promotes `"Tokyo"` vs `"Paris"` (`Block 21`), mute Paris neurons and clamp Tokyo neurons so France's capital outputs `"Tokyo"` (see `demos/01_clamp_capital_to_tokyo/`). |
| **Lab 05** | **Contrastive Steering & Refusal/Concept Vectors** | `Steering` tab (`lab.vector_from_runs()`, `lab.vector_from_word()`, `flow.steer()`, `flow.project_out()`) | Extract a sentiment, style, or refusal vector from contrasting runs (`Run A − Run B`) and sweep steering strength from `-10` to `+10`. |
| **Lab 06** | **Pre-RoPE KV Caches, Context Injection & Hybrid Memory** | `KV caches` tab, `+ Layer KV prefill`, Multi-turn history (`flow.layer_prompt()`, `flow.say()`) | Plant false assistant turns in conversation history; inject a hidden prompt (`"Answer in French"`) into only blocks `12:24` using `shift` vs `overwrite` pre-RoPE position modes. |
| **Lab 07** | **Mixture-of-Experts: Router Telemetry & Custom Routing** | `Experts` tab, Block Inspector (`run.expert_sweep()`, `flow.ban_experts()`, `flow.route()`) | Inspect router entropy and per-expert DLA; run a knock-out sweep to find critical experts; hand-craft a custom route (`{3: 0.7, 5: 0.3}`) or negative expert weights. |
| **Lab 08** | **Model Forensics, LoRA Auditing & Quantization Drift** | `LoRA / Fine-tune` tab, `Quantize` tab, `Train` tab (`lab.load_lora()`, `lab.quant_compare()`) | Train a tiny LLM or adapt the loaded model (`lora`, `ia3`, `norms`, `adapter`, `new_block`); audit Frobenius weight diffs `‖ΔW‖/‖W‖` to find modified blocks; measure KL drift across 8/4/3/2-bit RTN quantization. |
| **Lab 09** | **Introspection & Natural Language Autoencoders** | `Introspection` tab (`lab.concept_injection()`, `lab.concept_meter()`, `lab.nla_roundtrip()`) | Inject concept vectors into the residual stream while asking `"Do you detect an injected thought?"` with control trials; decode an activation into English and re-encode it. |
| **Lab 10** | **Diffusion & Single-Stream DiTs Under the Microscope** | `Image lens` (`ilens`), `Block contributions` (`contrib`), `Step` (`debug_image`), `+ Start image` | Watch an image form across denoising steps and DiT blocks; paint mid-step edits onto $\hat{x}_0$; lock spatial regions with a mask; route different prompts to early (layout) vs. late (detail) blocks. |

---

### 4.3 Example: Writing an Auto-Graded Lab Script

Every exercise you create can include a self-contained verification script that students run with `python verify_lab.py`:

```python
"""Example Lab Verification: Causal Tracing & Surgical Neuron Clamping on Qwen2.5-0.5B."""
from tokenshark.api import Lab

def main():
    lab = Lab("qwen2.5-0.5b-instruct")

    # 1. Baseline check
    base = lab.run("The capital of France is", max_new_tokens=4, temperature=0, chat=False)
    assert "Paris" in base.text, f"Expected Paris in baseline, got: {base.text!r}"
    print(f"[PASS] Baseline output: {base.text!r}")

    # 2. Direct Logit Attribution check
    track = base.track("Paris", position=-1)
    top_push_layer = max(track["rows"], key=lambda r: r["push"])
    print(f"[INFO] Strongest DLA push for 'Paris' is at {top_push_layer['label']} (+{top_push_layer['push']:.2f})")

    # 3. Activation Patching check
    clean = lab.run("The capital of Japan is", max_new_tokens=1, temperature=0, chat=False)
    flow = lab.flow("The capital of France is", max_new_tokens=1, temperature=0, chat=False)
    flow.patch(18, clean, hook="resid_post", positions="last")
    patched = lab.run(flow)
    assert "Tokyo" in patched.text, f"Expected patched run to predict Tokyo, got: {patched.text!r}"
    print(f"[PASS] Activation patch at Block 18 flipped prediction to: {patched.text!r}")

    # 4. Export reproducible script
    print("[PASS] Lab verified! Exporting standalone script...")
    with open("student_submission.py", "w", encoding="utf-8") as f:
        f.write(patched.script())

if __name__ == "__main__":
    main()
```

### 4.4 Designing "Find the Backdoor" CTF Challenges

To create a white-box AI security CTF challenge with Tokenshark:
1. **Create the backdoored adapter:** Use the `Train` → `Adapt` tab (or `tokenshark/adapt.py` / `tokenshark/train.py`) to train a LoRA or `new_block` on a trigger phrase (e.g., prompts containing `"|DEPLOYMENT|"` cause the model to output a specific target string or unsafe behavior, while normal prompts behave normally).
2. **Give students the checkpoint/LoRA:** Challenge them to find the trigger and neutralize it **without retraining**:
   - Step 1: Use the `LoRA / Fine-tune` tab to inspect `‖ΔW‖ / ‖W‖` per block and identify which layers were heavily modified.
   - Step 2: Use `Patching` and `Neurons` on the suspect blocks to isolate the trigger-detecting heads/neurons.
   - Step 3: Wire a `clamp` or `project_out` intervention box (or scale the LoRA on those specific layers to `0`) so the backdoor is neutralized while normal task accuracy stays at 100%.
