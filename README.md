# Tokenshark

Open, inspect, edit, and train how open-weight language and image models compute in the browser, from pure Python.

![Tokenshark Walkthrough](docs/walkthrough.gif)

---

## Live Demo: `qwen2.5-0.5b-instruct` Step-by-Step, Word Clouds & Mechanistic Interpretability

### 1. Step Debugger & Token-by-Token Generation

![Stepping through Qwen2.5-0.5B block by block and token by token](docs/stepping-demo.gif)

Load `qwen2.5-0.5b-instruct` (24 blocks, GQA 14q/2kv, SwiGLU) and press **🐞 Step**:
- **Block ▸** (`F10`) runs one box at a time; **Token ▸▸** (`F11`) runs to the next token prediction; right-click any block → *Debug: run to here*.
- The box about to run is highlighted in gold on the flow graph (`Block 21` below), while already-executed blocks are outlined in green with live logit-lens badges.
- Generated tokens (`Paris`, `.`, `It`, `is`, `the`, `largest`) appear in green alongside the prompt tokens, with a live per-block trace showing what each layer would predict if the model stopped there, which earlier tokens the last token attends to, and `‖resid‖`, `‖attn‖`, and `‖mlp‖` norms.
- At any token boundary, click any of the **top-10 next-token choices** to force that token and fork the generation.

| Mid-Token Block Stepping (`Block 21` active) | Token Boundary Prediction (`Paris` at 30.2%) |
|---|---|
| ![Step Debugger Mid-Block](docs/step-debugger.png) | ![Step Debugger Prediction](docs/step-debugger-prediction.png) |

### 2. Word Clouds & Direct Logit Attribution ("Through the Last 2 Blocks")

Plain logit lens (final norm + unembedding applied directly to an early layer) reads early layers poorly because middle blocks do not yet speak the final output vocabulary space. Setting **Decode with → `through the last 2 blocks, then unembed`** routes each chosen layer's residual stream through the last 2 transformer blocks before unembedding:

| Word Clouds Decoded Through Last 2 Blocks | Word Tracking & Direct Logit Attribution (`"Paris"`) |
|---|---|
| ![Word Cloud Through Last 2 Blocks](docs/word-cloud-through-2.png) | ![Word Tracking & Direct Logit Attribution](docs/word-tracking-dla.png) |

| Logit Lens Grid (Decoded Through Last 2 Blocks, Layers 11–23) | Output Tokens, Alternatives & Guided Course (`📖 Learn`) |
|---|---|
| ![Logit Lens Through Last 2 Blocks](docs/logit-lens.png) | ![Output & Guided Lesson](docs/llm-flow-output.png) |

---

## Core Features

### Flow Editor & Raw PyTorch Engine
- **Flow editor.** Each model is drawn as boxes and arrows (`prompt → tokenizer → embed → blocks → norm → head → sampler`, `prompt → CLIP → UNet stages → VAE`, or `prompt → Qwen3-VL → 32 single-stream DiT blocks → Wan VAE`). Re-wire arrows to remove, repeat, or re-order transformer blocks; wire intervention boxes into a block's top port to zero, scale, add noise, mean-ablate, patch, steer, or clamp an activation.
- **Bypass keeps the block.** A bypassed block stays on the canvas, struck through; the residual stream skips it. Restore it from the right-click menu or the inspector.
- **Raw PyTorch engine.** No `transformers`, no `diffusers` at runtime. Every matmul is in `analyser/llm.py`, `analyser/sd.py`, and `analyser/qimage.py`, small enough to read in class.
- **Declarative plugin architecture.** Each model family is a self-contained plugin in `analyser/plugins/` (`causal_lm.py`, `stable_diffusion.py`, `qwen_image_21.py`) that declares its `ui_schema` (tabs, palette actions, courses, training sections) and backend `features`.

### Editable Box Code & Export to Python Script
- **Editable box code (`analyser/blockcode.py`).** Click **View / edit code** on any language-model box (`Embed`, any `Block`, `Final norm`, `LM head`, `Sampler`) to view the exact ~15 lines of Python it runs (`rms_norm`, `heads`, `remember(k, v)`, `rope`, `softmax`, `silu`, `unembed`). Edit the code in the browser (e.g., remove `x = x + mlp_out`, scale `attn_out`, or write a custom sampler) and the engine executes your Python in place of the built-in step, capturing `print(...)` output.
- **Export run as Python script (`analyser/export.py` & `analyser/api.py`).** Any run in the browser (including its block schedule, bypasses, patches, steering vectors, neuron clamps, layer prompts, LoRA/quantization overlays, and custom box code) can be exported as a standalone Python script using `from analyser.api import Lab`.

![Editable Box Code](docs/editable-box-code.png)

### Mechanistic Interpretability Lab Tools
- **Activation patching / causal tracing (`Patching` tab).** Run a clean prompt (`"The capital of France is"`) and sweep a corrupted prompt (`"The capital of Japan is"`) across every `(block, token)` pair, copying one activation (`resid_post`, `resid_pre`, `attn_out`, or `mlp_out`) at a time to locate where the model stores a fact. Click any heatmap cell to wire that exact patch into the flow.
- **Neuron hunting & clamping (`Neurons` tab).** Inspect the strongest MLP neurons in any block (at a chosen token or peak across the run), see their activation across every token, and read the vocabulary words each neuron's `down_proj` column promotes or suppresses. Pin any neuron to `max`, `max ×5`, `min`, `0`, or a custom value.
- **Steering vectors (`Steering` tab).** Build contrastive directions in the residual stream from two runs (`run A − run B` at a block) or from a word's unembedding row (`"Paris" − "Tokyo"`), save them to `data/vectors/`, and wire them into any block.
- **Tokenizer explorer & Sampler playground (`Tokenizer` & `Sampler` tabs).** Inspect BPE splits, byte/char ratios, chat-template wrapping, vocabulary search, and cosine-nearest embedding neighbors. In the Sampler tab, adjust temperature, top-k, and top-p live on a captured logit distribution, compute entropy in bits, and draw 200 samples without re-running the model.

| Activation Patching (Causal Tracing Heatmap) | MLP Neuron Hunting, Vocabulary Push & Clamping |
|---|---|
| ![Activation Patching](docs/activation-patching.png) | ![MLP Neurons](docs/neurons.png) |

### Editable KV Cache & Per-Layer Prompts
- **Pre-RoPE KV cache (`analyser/kvcache.py`).** Keys are stored *before* RoPE with an explicit position per token, so a cache saved from one conversation can be prepended to another:
  - `negative`: the imported tokens sit at positions `-m…-1`; the new prompt starts at `0`, as if fresh.
  - `shift`: the imported tokens keep `0…m-1`; the new prompt continues at `m`.
  - `offset`: put the imported tokens anywhere.
  - `overwrite` (for layer prompts): replaces the prompt in chosen blocks so those blocks read the layer prompt instead of the main prompt.
  - Export/import `.safetensors` caches and `truncate`, `keep`, `drop`, or `shift` token spans.
- **Hidden prompt & per-layer KV prefill.** Encode a system/style prompt once into a KV cache (`KV caches` tab) or wire **+ Layer KV prefill** boxes directly into `Embed` with *Blocks that can see it* (`all`, `none`, `0:12`, `20:24`) so different transformer blocks read different context.

### LoRA / Fine-Tune Inspector, Quantization & In-Browser Training
- **LoRA & fine-tune overlay (`LoRA / Fine-tune` tab, `analyser/lora.py`).** Attach any Hugging Face PEFT LoRA, Diffusers/Kohya `.safetensors` LoRA, or full fine-tuned checkpoint at the `WeightStore` level. Inspect exact Frobenius norms `‖ΔW‖_F`, relative change `‖ΔW‖ / ‖W‖` per block and module (`q_proj`, `v_proj`, `mlp`), highlight the most-changed blocks on the flow canvas, and slide the LoRA scale live from `-2.0×` (anti-LoRA) to `+3.0×`.
- **On-the-fly quantization (`Quantize` tab, `analyser/quant.py`).** Round block weight matrices on the fly to `8`, `6`, `4`, `3`, `2`, or `1` bit (per-row symmetric RTN), measure per-block rounding error `‖W − round(W)‖ / ‖W‖`, and compare generated text, top-1 agreement, and KL divergence against full precision token-for-token.
- **Train models from scratch or adapt the loaded LLM (`Train` tab, `analyser/train.py`, `analyser/adapt.py`, `analyser/train_image.py`):**
  1. **Tiny language model from scratch:** Train a custom Llama-architecture transformer + BPE tokenizer on synthetic facts/counting corpora or your own text in seconds, then load it into the flow editor.
  2. **6 adaptation methods on the loaded LLM:** Compare `lora`, `full` (chosen matrices), `ia3` (per-row scaling), `norms` (RMSNorm weights), `adapter` (bottleneck layer after chosen blocks), and `new_block` (insert a whole new block initialized as `duplicate`, `average`, `identity`, or `random`).
  3. **Tiny flow-matching image model:** Train a 16×16 flow-matching denoiser on colored shapes with a held-out combination (`yellow triangle`) to watch compositional generalization and CFG emerge live.

| In-Browser Training (Tiny LLM, Flow-Matching Image & Adaptation) | LoRA / Fine-Tune Weight Diff Inspector |
|---|---|
| ![Train Tab](docs/train-tab.png) | ![LoRA Inspector](docs/lora-inspector.png) |

![Quantization Error & Precision Comparison](docs/quantization.png)

### Image Generation: Stable Diffusion UNets & Qwen-Image 2.1 Single-Stream DiT
- **Stable Diffusion UNet (`analyser/sd.py`) & Qwen-Image 2.1 DiT (`analyser/qimage.py`).** Full support for both multi-scale UNet models (`sd-turbo`, `tiny-sd`, `sd-1.5`) and `qwen-image-2.1` (32-block 7B single-stream DiT + 36-block Qwen3-VL 8B text encoder + 64-channel 16× Wan VAE).
- **Interactive diffusion step debugger & between-step image editing (`debug_image`).** Step through Qwen-Image block by block or step by step, paint edits onto the model's current guess `x̂₀` between steps (preserving the noise `ε̂`), adjust the between-step noise level `σ ∈ [0.02, 1.0]`, or extend the schedule with extra refinement steps.
- **Image lens ("Decode through last k blocks", `ilens` tab).** Project any intermediate DiT block's image tokens into pixel space (via fast fitted linear preview or exact VAE decode), either directly or after jumping through the last `k` blocks.
- **Block contributions (`contrib` tab).** See *where* each stage writes (residual update heatmap), *what* it encodes (3-component PCA RGB map), and *how much it matters* (ablation diff and ranking), plus a two-row per-step strip showing the noisy input `x_t` and predicted `x̂₀` at every step.

![Image Diffusion Flow, Cross-Attention Heatmap & Step Strip](docs/image-diffusion.png)

### Guided Courses (`📖 Learn` Tab) & MCP Tutor Server
- **Built-in interactive courses (`analyser/web/tabs/learn.js`).** Step-by-step guided lessons for both **Language Models** (*Tokens*, *One token at a time*, *Where the answer forms*, *Break it on purpose*, *Find where a fact lives*, *Steer and hunt neurons*, *A prompt only some layers can read*, *The model is a short program*, *Smaller, and your own*) and **Image Models** (*How an image model is built*, *From noise to picture*, *Steps and the noise schedule*, *How the picture reads the prompt*, *Guidance and the negative prompt*, *A different prompt for different blocks*, *Which blocks matter?*, *What each block thinks the picture is*, *Start from a picture*, *Step in and paint mid-way*, *Train your own image model*).
- **MCP Tutor Server (`analyser/mcp_server.py`).** Zero-dependency stdio MCP server (`python -m analyser.mcp_server`) that lets Claude Desktop, Claude Code, or Cursor drive Tokenshark as an interactive tutor. Every run, intervention, and patching sweep performed by the tutor appears live in the student's browser.

---

## For Educators / Coursework

- **Readable, wrapper-free forward passes:** Every transformer and diffusion operation (`analyser/llm.py`, `analyser/sd.py`, `analyser/qimage.py`) is written in explicit, single-file PyTorch without `transformers` or `diffusers` at runtime, and any block opens in the browser as ~15 lines of editable Python.
- **Turnkey classroom & self-paced labs (`📖 Learn` tab):** Includes 19 guided, interactive lessons covering tokenization, attention, logit lens, causal tracing (activation patching), steering vectors, neuron clamping, quantization, LoRA/adaptation, and flow-matching diffusion.
- **Runs on standard student laptops:** Pure Python with zero Node.js/build steps (`run.bat` / `./run.sh`); 0.5B–1.7B models and from-scratch training run on CPU in seconds, while layer streaming (`disk` policy) lets students inspect 27B–70B models without a cluster.
- **Reproducible homework & grading:** Students can save flows (`.json`), export any browser experiment—including patches, steering vectors, and custom block code—into a standalone Python script (`analyser.api`), or pair with an AI teaching assistant via the built-in MCP tutor server.
- **Adopt or cite in your syllabus:** Free for university courses, workshops, and training bootcamps—see [`CITATION.cff`](CITATION.cff) to cite Tokenshark in lecture notes, lab handouts, or academic publications.

---

## Run It

Windows: double-click `run.bat`. macOS / Linux: `./run.sh`. The first run creates `.venv` and installs `requirements.txt`. Then open http://127.0.0.1:8000.

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # NVIDIA GPU: install the CUDA build of torch from pytorch.org first
python -m analyser                               # --port 8000 --device auto|cuda|mps|cpu --dtype auto|fp32|fp16|bf16
```

Choose a model, choose where its weights live, and click **Load**. On first use the model downloads from Hugging Face into **`models/<id>/`** next to the app (change with `--models-dir D:\models` or the `ANALYSER_MODELS` variable). Only configs, the tokenizer, and one copy of the weights are fetched (fp16 for the image models). They're plain files with no hidden cache: copy, move, or delete them like any folder. A `✓` in the model list means it is on disk, and `🗑` deletes it. Any model folder you drop into `models/` (with `config.json`, Meta `params.json`, or `model_index.json`) appears in the list automatically. Then edit the flow and click **Run**.

| Weights setting | What happens | Use when |
|---|---|---|
| `auto` | picks from model size vs free memory | default |
| `all on GPU/CPU` | everything resident on the compute device | it fits |
| `RAM → device per layer` | resident in RAM, copied to the GPU block by block | fits in RAM, not VRAM |
| `stream from disk` | nothing resident; each block is read from the memory-mapped file when it runs | bigger than RAM (e.g. 27B–70B on a 16 GB laptop) |

`keep N` keeps the first `N` blocks resident on the device while the rest stream. Streaming a 70B model reads ~140 GB per forward pass: expect minutes per token on NVMe. It works; it is not fast.

---

## Models

Built-in catalog (`analyser/models.json` & `analyser/plugins/*.py`):

| Id | Plugin / Family | Size | Notes |
|---|---|---|---|
| `qwen2.5-0.5b-instruct` | `causal_lm` (text → text) | 1.0 GB | 24 blocks, GQA 14q/2kv; best first model, fast on CPU |
| `smollm2-360m-instruct` | `causal_lm` (text → text) | 0.7 GB | Llama architecture, 32 blocks; smallest option |
| `qwen2.5-1.5b-instruct` | `causal_lm` (text → text) | 3.1 GB | 28 blocks; noticeably better answers |
| `qwen3-1.7b` | `causal_lm` (text → text) | 3.4 GB | 28 blocks, per-head QK RMSNorm |
| `llama-3.2-1b-instruct` | `causal_lm` (text → text) | 2.5 GB | gated: accept the licence and run `hf auth login` |
| `qwen3.6-27b` | `causal_lm` (text → text) | 55.0 GB | hybrid: 48 Gated DeltaNet (linear-attention) + 16 gated-attention blocks (64 total); stream from disk |
| `sd-turbo` | `stable_diffusion` (text → image) | 2.6 GB | UNet, 1–4 steps, no CFG; best first image model |
| `tiny-sd` | `stable_diffusion` (text → image) | 1.0 GB | distilled SD 1.5 UNet |
| `sd-1.5` | `stable_diffusion` (text → image) | 2.0 GB | classic UNet with CFG; reference model for cross-attention maps |
| `qwen-image-2.1` | `qwen_image_21` (text → image) | 33.1 GB | single-stream DiT (32 blocks, 7B) + Qwen3-VL 8B text encoder + 64-ch Wan VAE |

Click **+** next to the model picker to add any Hugging Face repo or local folder (`data/models.local.json`). The raw engine supports:
- **Text (`causal_lm`):** Llama 2/3/3.1/3.2, Qwen2/2.5, Qwen3, Mistral, SmolLM, and **Qwen3.5/3.6 dense hybrids** (Gated DeltaNet linear attention + gated full attention). Supports Hugging Face `.safetensors` and Meta `consolidated.00.pth` + `params.json` + `tokenizer.model`.
- **Image (`stable_diffusion` & `qwen_image_21`):** Stable Diffusion 1.x / 2.x UNets (including distilled variants) and Qwen-Image 2.1 single-stream DiT pipelines.

---

## Steering Text & Image Generation

| Tool | Where | What it does | Cross-modal analogue |
|---|---|---|---|
| Activation patch (`op="patch"`) | Patching tab / Intervention box | copies `resid_pre`, `attn_out`, `mlp_out`, or `resid_post` from a source run at chosen tokens | Branch from step `k` |
| Steering vector (`op="steer"`) | Steering tab / Intervention box | adds `value × vector` (`run A − run B` or word direction) to a block's stream | Guidance / word re-weighting |
| Neuron clamp (`op="clamp"`) | Neurons tab / Intervention box | pins chosen MLP neurons to `max`, `min`, or `value` | Feature channel scaling |
| Layer KV prefill (`kv_text`) | Palette → `+ Layer KV prefill` | gives chosen blocks their own prompt prefix or overwrites their prompt (`shift`, `negative`, `offset`, `overwrite`) | Per-block Prompt-KV (`+ Other prompt (blocks)` / `+ Hidden prompt`) |
| Negative prompt + guidance > 1 | Prompt, Scheduler boxes | pushes each denoising step away from the negative text | System prompt / contrastive steering |
| Prompt switch at step `k` | Prompt box | prompt A sets layout in early steps, prompt B paints detail later | Changing the instruction mid-answer |
| Start image + strength / Sketch guide | Palette → `+ Start image` | starts from your image noised by *strength*, or steers only drawn tiles from pure noise (*sketch guide*) | Assistant prefill |
| Region lock (painted mask) | Start image / Intervention box | painted area is locked to start image (*keep*), only painted area changes (*regenerate* = inpainting), or an intervention applies only inside the painted mask | Forced tokens at specific positions |
| Between-step image & noise `σ` edit | Step debugger (`debug_image`) | paint on `x̂₀` between denoising steps while preserving `ε̂`, adjust `σ`, or extend steps | Forcing a token mid-generation |
| Word intervention (`hook="word"`) | Intervention box on a UNet/DiT block | scales how strongly image patches read chosen prompt tokens (`0` removes a word, `2` emphasises it) | Attention head / token ablation |

---

## Python API & MCP Tutor Server

Use the exact same raw engine from Python scripts or Jupyter notebooks (`analyser/api.py`):

```python
from analyser.api import Lab

lab = Lab("qwen2.5-0.5b-instruct")
run = lab.run("The capital of France is", max_new_tokens=6)
print(run.text)

# Inspect internal activations and logit lens (including decoding through last k blocks)
lens = run.lens(k=5, through=2)
attn = run.attention(block=18)            # [heads, tokens, tokens]
resid = run.resid(block=18, where="post") # [tokens, d_model]
neurons = run.top_neurons(block=19, token=-1, k=10)

# Activation patching: copy block 18's residual stream from the clean run into a corrupted prompt
clean = lab.run("The capital of France is", max_new_tokens=1)
flow = lab.flow("The capital of Japan is", max_new_tokens=1)
flow.patch(18, clean, hook="resid_post", positions="last")
print(lab.run(flow).text)
```

To connect Claude Desktop, Claude Code, or Cursor to a running Tokenshark instance via MCP:

```json
{
  "mcpServers": {
    "tokenshark": {
      "command": "python",
      "args": ["-m", "analyser.mcp_server", "--url", "http://127.0.0.1:8000"]
    }
  }
}
```

---

## Project Layout

```text
analyser/
  weights.py       WeightStore: safetensors/.bin/.pth, offload policies, per-layer streaming, LoRA/quant hooks
  llm.py           raw decoder-only LLM: RMSNorm, RoPE (incl. llama3 scaling), GQA, SwiGLU, QK-norm, Gated DeltaNet
  kvcache.py       pre-RoPE KV cache & hybrid linear memory: append, prepend (shift/negative/offset/overwrite), edit
  sd.py            raw Stable Diffusion: CLIP text encoder, UNet, VAE, Euler scheduler, cross-attn & stage maps
  qimage.py        raw Qwen-Image 2.1: Qwen3-VL text encoder, 32-block single-stream DiT, Wan VAE, ImageStepper
  tokenizer.py     tokenizer.json / tiktoken + Jinja chat templates; pure-Python CLIP BPE
  flow.py          default flows per model; compile flow (boxes + arrows) → plan (schedule + interventions)
  flowbuild.py     FlowBuilder: fluent Python builder for editing flows in one line per change
  engine.py        run context, intervention ops (zero, scale, noise, mean, patch, steer, clamp), device/dtype choice
  interp.py        activation patching sweeps, steering vector library, neuron hunting, tokenizer & sampler tools
  blockcode.py     editable Python code view and AST-checked execution for LLM flow boxes
  lora.py          WeightStore-level LoRA & full fine-tune overlay with Frobenius norm diff inspection
  quant.py         on-the-fly per-row RTN quantization overlay (1–16 bits) and token-by-token KL comparison
  train.py         differentiable forward pass, TinyJob (train LLM from scratch) & LoRAJob (PEFT LoRA trainer)
  adapt.py         model adaptation methods beyond LoRA: full, ia3, norms, bottleneck adapter, new_block
  train_image.py   TinyDenoiser flow-matching trainer on 16×16 colored shapes with CFG and compositional test
  export.py        export any browser run (and its patch sources) to a reproducible Python script
  api.py           Lab & RunResult Python API for scripts and notebooks
  mcp_server.py    pure-stdlib stdio MCP server for AI tutoring
  server.py        FastAPI routes for models, flows, runs, caches, and step debugger
  labapi.py        FastAPI routes for patching, steering, neurons, quantization, box code, and training
  plugins/         modular model plugins (causal_lm.py, stable_diffusion.py, qwen_image_21.py, _template.py)
  web/             browser UI in plain ES modules + HTML/CSS (core.js, flow.js, paint.js, ui/*, tabs/*)
tests/
  test_oracles.py       raw LLM & SD engines vs transformers / diffusers on random weights; KV-cache maths
  test_features.py      stepper, bypass, word through-k decoding, DLA tracking, region locks, model discovery
  test_hybrid.py        Qwen3.5/3.6 hybrid Gated DeltaNet + gated attention vs transformers & cache roundtrips
  test_qimage.py        raw Qwen-Image 2.1 vs diffusers QwenImage21Pipeline, Prompt-KV, sketch guide, ImageStepper
  test_lab.py           patching, steering, neurons, layer prompts, box code, quantization, adaptation, MCP, export
  test_routes.py        end-to-end HTTP API tests for all lab, quantization, code, and training endpoints
  test_architecture.py  plugin contract & declarative UI schema verification
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests
```

The test suite (**112 tests** across 7 test modules) verifies numerical equivalence against `transformers` and `diffusers` on random weights, hybrid linear/full attention math, Qwen-Image 2.1 DiT bit-exact schedules and latents, step-by-step debugging, activation patching, steering, neuron clamping, quantization, in-browser training, adaptation methods, and UI plugin contracts.
