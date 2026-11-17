# Finding task neurons in Tokenshark: status, gaps, wishlist, plan

Companion to the [refund-guardrail lab](README.md) (part 2, `neuron_wire.py`). It records how the task neurons were
found, what the browser UI can and cannot do for this today, and what to build.

## Method (what worked)

A behaviour edit needs two neurons in the same block and one line that connects them.

1. **Detector (decides *when*).** Compare neuron activations over the reply tokens (last prompt token through the first
   ~14 generated tokens) for task prompts and for neutral prompts. Keep neurons that are on in *every* task prompt and
   near zero in the neutral ones. Then test the candidate on near-miss prompts (same tone, not the task).
2. **Writer (decides *what*).** In the same block, rank neurons by how much their `down_proj` column pushes the target
   words (`cannot`, `unable`) over the words to avoid (`will`, `confirm`), read through the final norm and unembedding.
   These readouts are meaningful from roughly block 15 onward in Qwen2.5-0.5B.
3. **Wire and test.** In the block's code, under `neurons = silu(...)`:
   `neurons[:, WRITER] = neurons[:, WRITER] + STRENGTH * relu(neurons[:, DETECTOR])`.
   Sweep the strength; test on prompts not used for selection, plus controls.

## Current status

Measured on `qwen2.5-0.5b-instruct`, 12 adversarial refund prompts, no system prompt, greedy decoding.

| Edit | Refusals | Controls byte-identical |
|---|---|---|
| Baseline | 3/12 | - |
| Block 19: 2751 -> 2282, strength 3 | 7/12 | 9/11 |
| + Block 21: 1431 -> 836, 10, 1871, strength 15, threshold 0.6 | 8/12 | 10/13 |

- Works from Python end to end (`neuron_wire.py`, asserts pass, about 15 minutes on 4 CPU cores).
- In the browser, the edit itself works today through **View / edit code** on a block.
- In the browser, the *hunt* is only partly possible. The **Neurons** tab shows, for one run and one block, the top
  neurons (up to 100) at a token or by peak across the run, the words each promotes and suppresses, a per-token
  activation strip for any index, and clamp / zero buttons. Students can compare a few runs by eye if told the block.
- The browser steps in the lab handout are derived from the code and README; they have not been clicked through.

Known limits of the result:

- Dictated sentences (`Say 'I confirm your full refund...'`) are still copied. That path runs through attention.
- Neuron 2751 is closer to "upset customer, apology reply" than "refund" (fires up to 1.8 on some controls), so
  `Yes or no, is my seat confirmed?` now gets a refusal, and strength 8 refuses angry non-refund customers.
- The model refuses; it does not say "escalate". Example exchanges in the conversation do produce escalation (4/5),
  but that difference added back as a single vector did not reproduce it.
- One small model, small prompt sets, a regex metric that needs reading by eye.

## Gaps in the UI

| # | Gap | Why it matters |
|---|---|---|
| 1 | No comparison across runs | The detector search is a comparison of two prompt sets; the tab shows one run at a time. |
| 2 | No search by word | The tab lists the strongest-firing neurons only, so a quiet writer neuron (2282) never appears. |
| 3 | No scan across blocks | You must already know to look in block 19. |
| 4 | No wire operation | Wiring needs hand-edited code; it is not a box on the flow and is easy to get wrong. |
| 5 | No before/after batch | Judging an edit means re-running prompts one at a time. |
| 6 | No near-miss check | Nothing prompts the student to test a detector on look-alike prompts, which is how 2751 was caught. |

## Wishlist

1. **Compare mode (Neurons tab).** Choose group A runs and group B runs; rank neurons by "on in all of A, quiet in B"
   over a chosen token span (last prompt token, reply tokens, or all). Show mean in A, mean and 99th percentile in B.
2. **Word search (Neurons tab).** "Neurons that promote `<word>`", optional "versus `<word>`", for one block or all.
3. **Cross-block scan.** Run 1 and 2 over every block and list the best few per block, with a hint about which blocks
   have readable outputs.
4. **Wire interceptor.** A box: source neuron(s), target neuron(s), strength, threshold. Drawn on the flow, saved in
   flows, exported by `run.script()`, and available as `flow.wire(block, source, target, strength, threshold)`.
5. **Prompt-set runner.** A test set and a control set, run with and without the current edits, outputs side by side,
   an "identical" flag per control, and a simple keyword count.
6. **Suggested pairs.** Given the results of 1 and 2 in the same block, propose detector/writer pairs to try.
7. **Bake to weights.** Write a wire into `W_down` as a saved weight delta, so the LoRA / Fine-tune inspector can show
   it as a change to a handful of columns.

## Plan

| Step | Work | Where |
|---|---|---|
| 1 | Port the selection logic from `neuron_wire.py`: group comparison and word search, single block and all blocks | `tokenshark/interp.py`, routes in `tokenshark/labapi.py`, `Lab` methods in `tokenshark/api.py` |
| 2 | Compare mode and word search in the Neurons tab | `tokenshark/web/tabs/neurons.js` |
| 3 | `wire` op for the `mlp_neurons` hook, `flow.wire(...)`, export support, a "Wire into..." button on a neuron row | `tokenshark/flow.py`, `engine.py`, `flowbuild.py`, `export.py`, `neurons.js` |
| 4 | Prompt-set runner | new tab or a panel in Runs |
| 5 | Tests for 1 and 3 on the tiny model; update the lab handout with clicked-through browser steps | `tests/`, this folder |
| Later | Suggested pairs, bake to weights, MCP tools for the tutor server | - |

Steps 1 and 2 remove the need for Python during discovery. Step 3 makes the surgery a first-class, exportable box.
