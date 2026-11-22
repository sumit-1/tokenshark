import { $, POST, S, btn, can, canRewire, el, flow, heat, hooks, schema, spark, texts, toast } from '../core.js'
import { esc } from '../flow.js'
import { maskPreview, openPaint } from '../paint.js'
import { codeSection } from './codeedit.js'
import { debugRunTo } from './debugger.js'
import { addBlockKV, addForkHead, addIntervention, bypassBlock, duplicateBlock, selectedNode } from './flowedit.js'
import { setTab, tabAvailable } from './tabs.js'
import { saveRunCache } from '../tabs/kv.js'
import { stepStrip } from '../tabs/output_image.js'
import { renderTokens } from '../tabs/output_text.js'
import { neighbors, renderWordDictionary } from '../tabs/tokenizer.js'

export const LLM_HOOK_LABELS = [
  ['resid_pre', 'resid_pre · the stream entering the block'],
  ['q', 'q · query heads (where tokens look)'],
  ['k', 'k · new key heads (written to KV cache)'],
  ['v', 'v · new value heads (written to KV cache)'],
  ['kv_k', 'kv_k · all cached keys in this block\'s KV cache'],
  ['kv_v', 'kv_v · all cached values in this block\'s KV cache'],
  ['rope', 'rope · rotary position angles (zero / scale / jitter token positions)'],
  ['attn_heads', 'attn_heads · each attention head\'s output'],
  ['attn_out', 'attn_out · what attention writes to the stream'],
  ['router', 'router · MoE: the router\'s score for each expert (ban / force experts, patch the routing)'],
  ['expert_out', 'expert_out · MoE: what the chosen experts write, before it is weighted and summed'],
  ['mlp_neurons', 'mlp_neurons · the MLP\'s hidden neurons (MoE: of the experts you name)'],
  ['mlp_out', 'mlp_out · what the MLP writes to the stream'],
  ['resid_post', 'resid_post · the stream leaving the block'],
  ['weights', 'weights · this block\'s weight matrices (per-block quantize / scale)'],
]

export const BLOCK_KV_OPS = [
  ['replace_system', 'replace_system · Replace system prompt'],
  ['append_system', 'append_system · Append to system prompt'],
  ['prepend_system', 'prepend_system · Prepend to system prompt'],
  ['replace_user', 'replace_user · Replace user\'s prompt'],
  ['append_user', 'append_user · Append to user\'s prompt'],
  ['prepend_user', 'prepend_user · Prepend to user\'s prompt'],
]

const BLOCK_KV_OP_LABELS = {
  replace_system: 'Replace system', append_system: 'Append system', prepend_system: 'Prepend system',
  replace_user: 'Replace user', append_user: 'Append user', prepend_user: 'Prepend user',
}

export const PREFIX_MODES = [['shift', 'shift: prefix at 0..m-1, the prompt continues at m (use for a hidden system prompt)'],
  ['negative', 'negative: prefix at -m..-1, the prompt starts at 0 (as if fresh)'], ['offset', 'offset: prefix starts at the offset below']]

export const FIELDS = {
  input_prompt: modality => modality === 'image'
    ? [['prompt', 'Prompt', 'textarea'], ['negative_prompt', 'Negative prompt (what to steer away from; needs guidance > 1)', 'textarea'],
       ['switch_prompt', 'Switch to this prompt at a step (optional)', 'textarea'], ['switch_step', 'Switch at step', 'number']]
    : [['system', 'System', 'textarea'], ['prompt', 'User / prompt (the newest turn)', 'textarea'],
       ['prefill', 'Assistant prefill (forced tokens)', 'textarea'], ['use_chat_template', 'Use chat template', 'checkbox'],
       ['thinking', 'Let the model think first (<think>…</think>, models that support it)', 'checkbox'],
       ['add_bos', 'Add BOS (no template)', 'checkbox']],
  sampler: () => [['max_new_tokens', 'Max new tokens', 'number'], ['temperature', 'Temperature (0 = greedy)', 'number'],
    ['top_k', 'Top-k (0 = off)', 'number'], ['top_p', 'Top-p (1 = off)', 'number'], ['seed', 'Seed', 'number'],
    ['rope_step_stride', 'RoPE step stride per generated token (1 = normal, 0 = freeze pos, 50 = fast-forward / forget)', 'number'],
    ['rope_prompt_scale', 'RoPE prompt position scale (1 = normal, 0 = bag-of-words prompt)', 'number']],
  noise: () => [['seed', 'Seed', 'number'], ['width', 'Width', 'number'], ['height', 'Height', 'number']],
  scheduler: () => [['steps', 'Steps', 'number'], ['guidance_scale', 'Guidance (CFG, ≤1 = off)', 'number']],
  block: (m, n) => [['layer', 'Weights of layer', 'number'], ['label', 'Label', 'text'],
    ['rope_scale', 'RoPE position scale (empty/1 = normal, 0 = off / no RoPE, −1 = reverse)', 'number'],
    ['rope_theta', `RoPE base θ override (empty = model’s ${S.info?.rope_theta ?? 10000}; try 100 to desync)`, 'number'],
    ['rope_q_offset', 'RoPE query shift Δpos (shifts Q relative to K in this block)', 'number'],
    ...(n?.data.moe ? [['moe_k', `Experts per token (empty = the model’s own ${S.info?.moe?.top_k ?? ''}; 1 … ${n.data.moe})`, 'number'],
      ['moe_temp', 'Router temperature (empty = 1; higher = flatter choice, lower = sharper)', 'number'],
      ...(S.info?.moe?.shared ? [['moe_shared', 'Use the shared expert (the one every token goes through)', 'checkbox']] : [])] : [])],
  intervention: (modality, n) => {
    if (modality === 'image') return [['label', 'Label', 'text'], ['enabled', 'Enabled', 'checkbox'],
      ['hook', 'Hook', 'select', S.info?.hooks || []], ['op', 'Operation', 'select', ['zero', 'scale', 'add_noise', 'mean', 'quantize']],
      ['value', 'Value (scale factor / noise σ / bits)', 'number'], ['steps', 'Steps (all | 0:4 | 2,5)', 'text'],
      ['positions', 'Prompt tokens for hook “word” (token #, see the Tokenizer box: 1 | 2:4)', 'text'], ['seed', 'Noise seed', 'number']]
    const op = n?.data.op || 'zero', hook = n?.data.hook
    const runs = S.runs.filter(r => r.has_captures && r.model_id === S.info?.model_id).map(r => [r.id, `${r.id}${r.label ? ' · ' + r.label : ''} · ${(r.preview || '').slice(0, 30)}`])
    const keepRef = (list, v) => v && !list.find(x => x[0] === v) ? [[v, `${v} (no longer in memory)`], ...list] : list
    const f = [['label', 'Label', 'text'], ['enabled', 'Enabled', 'checkbox'], ['hook', 'Where (hook)', 'select', LLM_HOOK_LABELS.filter(h => (S.info?.hooks || []).includes(h[0]))],
      ['op', 'Operation', 'select', [['zero', 'zero · remove it'], ['scale', 'scale · multiply by the value'], ['add_noise', 'add_noise · add random noise'],
        ['mean', 'mean · replace by the average over tokens'], ['quantize', 'quantize · round to N bits (1..16)'],
        ['patch', 'patch · copy it from another run'], ['steer', 'steer · add a steering vector'],
        ['project_out', 'project_out · erase / project out a direction'], ['clamp', 'clamp · pin units to a fixed value'],
        ['wire', 'wire · wire detector neuron(s) into writer neuron(s)'],
        ...(S.info?.moe ? [['ban', 'ban · MoE router: these experts can never be chosen'], ['only', 'only · MoE router: only these experts can be chosen'],
          ['route', 'route · MoE router: use exactly the experts and weights you write']] : [])]]]
    if (['router', 'expert_out', 'mlp_neurons'].includes(hook) && S.info?.moe) f.push(['experts', op === 'route'
      ? 'Experts and their weights (3=0.6, 17=0.4 · a name without a weight keeps the router\'s own score)'
      : `Experts (all | 3 | 0:8 | 1,5${S.info.moe.shared ? ' | shared' : ''})`, 'text'])
    if (op === 'patch') f.push(['ref', 'Copy from run (the "clean" run)', 'select', [['', '- choose a run -'], ...keepRef(runs, n.data.ref)]],
      ['src_positions', 'Take the source run\'s token (empty = same token index; or last, or a number)', 'text'])
    if (op === 'steer' || op === 'project_out') f.push(['ref', 'Vector direction', 'select', [['', '- choose a vector -'], ...keepRef((S.vectors || []).map(v => [v.id, `${v.name} · length ${v.norm}`]), n.data.ref)]],
      ['value', op === 'project_out' ? 'Projection strength (1 = erase completely, 2 = reflect)' : 'Strength (multiplies the vector; negative pushes the other way)', 'number'])
    if (op === 'clamp') f.push(['clamp', 'Pin to', 'select', [['max', 'max · the largest value it reached'], ['min', 'min · the smallest value it reached'], ['value', 'value · the number below']]],
      ['value', n.data.clamp === 'value' ? 'Value' : 'Multiplier (1 = exactly its max / min, 5 = five times that)', 'number'],
      ['ref', 'Max / min measured in run (empty = this run so far)', 'select', [['', '- this run so far -'], ...keepRef(runs, n.data.ref)]])
    if (op === 'wire') f.push(
      ['source_neurons', 'Source / detector neuron(s) (e.g. 2751 or 1431)', 'text'],
      ['neurons', 'Target / writer neuron(s) (e.g. 2282 or 836,10,1871)', 'text'],
      ['value', 'Strength (adds strength × relu(source − threshold) to target)', 'number'],
      ['threshold', 'Threshold on source neuron before ReLU (0 = any positive activation)', 'number'],
    )
    if (op === 'scale' || op === 'add_noise' || op === 'quantize') f.push(['value', op === 'scale' ? 'Scale factor' : op === 'quantize' ? 'Bit width (1..16 bits)' : 'Noise size σ (× the activation’s spread)', 'number'])
    if (hook !== 'weights') f.push(['positions', 'Tokens (all | last | prompt | prefill | generated | 3:9 | 1,4)', 'text'])
    if (['attn_heads', 'q', 'k', 'v', 'kv_k', 'kv_v'].includes(hook)) f.push(['heads', 'Heads (all | 0:4 | 2)', 'text'])
    if (op !== 'wire' && hook !== 'weights' && hook !== 'router' && hook !== 'rope') f.push(['neurons', hook === 'mlp_neurons' ? 'Neurons (all | 1021 | 3,7 | 0:64)' : ['attn_heads', 'q', 'k', 'v', 'kv_k', 'kv_v'].includes(hook) ? 'Dimensions inside each head (all | 0:8)' : 'Dimensions of the stream (all | 12 | 0:64)', 'text'])
    if (op === 'add_noise') f.push(['seed', 'Noise seed', 'number'])
    return f
  },
  block_kv: () => [['label', 'Label', 'text'], ['enabled', 'Enabled', 'checkbox'],
    ['op', 'What to change', 'select', BLOCK_KV_OPS],
    ['text', 'New / extra text', 'textarea'],
    ['layers', 'Also apply to schedule positions (optional if wired: all | 12:24 | 3,5)', 'text']],
  kv_import: () => [['cache_id', 'KV cache', 'select', S.caches.map(c => [c.id, `${c.name} · ${c.n_tokens} tok`])],
    ['mode', 'Position mode', 'select', PREFIX_MODES],
    ['offset', 'Offset (mode = offset)', 'number'],
    ['layers', 'Blocks that can see it (schedule positions: all | none | 0:12 | 20:24 | 3,5)', 'text'], ['label', 'Label', 'text']],
  prompt_kv: () => [['label', 'Label', 'text'], ['enabled', 'Enabled', 'checkbox'], ['prompt', 'Prompt', 'textarea'],
    ['role', 'Role', 'select', [['replace', 'replace: these blocks read this prompt instead of the main one'],
      ['prepend', 'prepend: a hidden prompt in front of the main one (the prompt and the image both see it)']]],
    ['blocks', 'Blocks (schedule positions: all | none | 0:16 | 3,5)', 'text'],
    ['mode', 'Positions (prepend)', 'select', [['shift', 'shift: hidden at 0..m−1, main prompt continues at m'],
      ['negative', 'negative: hidden at −m..−1, main prompt unchanged'], ['offset', 'offset: hidden starts at the offset below']]],
    ['offset', 'Offset (mode = offset)', 'number']],
  init_image: () => [['mask_mode', 'Painted area', 'select', [['keep', 'keep: locked to the start image (like forced tokens)'], ['regenerate', 'regenerate: only the painted area changes']]],
    ['lock_steps', 'Lock during steps (all | 0:10 | 0,1,2)', 'text'], ['lock_strength', 'Lock strength (1 = hard, 0.3 = soft pull)', 'number']],
  fork_head: () => [
    ['label', 'Label', 'text'],
    ['enabled', 'Enabled', 'checkbox'],
    ['ref', 'Classifier Head ID (e.g. powershell_detector)', 'text'],
    ['mode', 'Execution Mode', 'select', [['monitor', 'monitor · record per-token category probabilities'], ['gate', 'gate · halt / redirect generation when triggered']]],
    ['hook', 'Source Block Hook', 'select', [['resid_post', 'resid_post · after block'], ['resid_pre', 'resid_pre · before block'], ['mlp_out', 'mlp_out · MLP output'], ['attn_out', 'attn_out · attention output']]],
    ['threshold', 'Trigger Threshold (0..1)', 'number'],
    ['trigger_category', 'Trigger Category (e.g. PowerShell)', 'text'],
    ['halt_message', 'Custom Halt / Redirect Message (for gate mode)', 'text'],
  ],
}

export const KIND_HELP = {
  input_prompt: 'Text that goes into the model. The prompt is the newest user turn; earlier turns of the conversation (below) come before it. Prefill tokens are forced: the model continues from them.',
  tokenizer: 'Splits text into token ids.',
  embed: 'Looks up one vector per token: the start of the residual stream. The KV input on top prepends another conversation\'s cache.',
  block: 'One transformer block: RMSNorm → attention (q·kᵀ with RoPE) → add → RMSNorm → SwiGLU MLP → add. Rewire arrows to remove, repeat or reorder blocks. Wire an interceptor into the top port.',
  block_moe: 'A mixture-of-experts block: RMSNorm → attention → add → RMSNorm → router → the few chosen experts → add. Instead of one wide MLP there are many narrow ones (the cells on the box). For each token the router scores every expert, runs only the best few and adds their outputs, weighted by their scores. Click a cell to pick an expert; ban it, force it, silence it or amplify it, then run again.',
  final_norm: 'RMSNorm before unembedding.', lm_head: 'Projects the residual stream onto the vocabulary → logits.',
  sampler: 'Turns logits into the next token; the loop arrow feeds it back in.',
  output_text: 'Generated text. Click a token to see the alternatives the model considered and fork from there.',
  text_encoder: 'CLIP text transformer: produces the context that every cross-attention layer reads.',
  noise: 'The starting latent: Gaussian noise from the seed (or a noised start image).',
  scheduler: 'Euler sampler: sets the noise level σ at each step (high σ = rough layout, low σ = fine detail). Guidance > 1 runs the UNet twice (with and without the prompt) and pushes the result away from the negative prompt.',
  denoiser_in: 'Each step the UNet predicts the noise in the current latent.',
  unet_block: 'UNet stage. Down stages shrink the image and pass skip features (dotted) to the matching up stage. Order is fixed. Wire an interceptor: hook out/skip scales or zeroes this stage\'s features for chosen steps; hook word changes how strongly this stage reads chosen prompt words (0 = ignore the word, 2 = emphasise it).',
  denoiser_out: 'Noise prediction → Euler step → next latent. Loops for every step.',
  vae_decode: 'Turns the 4-channel latent back into pixels.',
  intervention: 'Intercepts and modifies a block’s activations, KV cache, or weights while the model runs. Wire it into the top port of one or more blocks. Supports zero, scale, noise, quantize (per-block weight or KV-cache bits), patch, steer, project_out, and clamp.',
  block_kv: 'Rewrites what a KV-capable block remembers. The system or user prompt is changed (replace / append / prepend), then that block’s KV cache is recreated from the rewritten prompt and grows as new tokens are generated. Wire into the block’s top KV port (linear-attention / Gated DeltaNet blocks have no KV port).',
  kv_import: 'Prepends a saved KV cache: the new prompt attends to it without recomputing it. Keys are stored before RoPE, so positions can be moved freely. Add several boxes (each wired into Embed\'s KV port) to give different blocks different prefixes; each box\'s "Blocks that can see it" decides which blocks read it. With the chat template on, the Prompt box\'s System field is ignored - the prefix plays that role.',
  prompt_kv: 'A second prompt for chosen blocks. Qwen-Image pushes the prompt through the blocks once and keeps its keys/values (a KV cache) so each block can read a different one. replace: those blocks see this prompt instead (e.g. layout blocks "a cat", detail blocks "a dog"). prepend: a hidden prompt the main prompt and the image both attend to in those blocks, like the hidden system prompt of the language models.',
  init_image: 'Image prefill. Strength sets how much noise is added before denoising starts (low = stay close to the image). Paint a region to lock it: at every locked step that region is reset to the start image at the current noise level. "regenerate" flips it: only the painted area changes (inpainting).',
  fork_head: 'Side-branch classifier or guardrail head wired to a transformer block\'s bottom fork port. Runs a lightweight linear probe, 2-layer MLP guard, or SwiGLU branch on that block\'s activations in monitor mode (live probabilities) or gate mode (halts generation mid-network when triggered).',
}

// Help text that differs by architecture; the plugin names its set (ui_schema.texts).
export const DIT_HELP = {
  text_encoder: 'Qwen3-VL language model (36 blocks). The prompt is wrapped in a fixed chat turn; the last block\'s hidden states - before its final norm - become the text tokens of the DiT.',
  scheduler: 'Flow matching: the latent moves in a straight line from noise (σ = 1) to the image (σ = 0). Each step: x <- x + (σ_next - σ)·v. σ values are shifted toward the noisy end for bigger images. Guidance > 1 with a negative prompt runs the image tokens twice.',
  denoiser_in: 'One sequence: [prompt tokens | image tokens]. The prompt tokens went through all blocks once (at t = 0) and their keys/values are cached - the same trick as an LLM KV cache - so each step only computes the image tokens.',
  unet_block: 'Single-stream DiT block: LayerNorm·(1+scale) → attention (prompt tokens: causal; image tokens: see the whole prompt and every patch) → add ×tanh(gate) → LayerNorm·(1+scale) → SwiGLU → add. One shared modulation for all blocks. Blocks can be bypassed, repeated or reordered; hook out scales/zeroes what the block leaves in the image tokens, hook word re-weights how much the image reads chosen prompt tokens.',
  denoiser_out: 'Final norm + projection → velocity v (≈ noise − image). Euler step → next latent. Loops for every step.',
  vae_decode: 'Turns the 64-channel latent (one vector per 16×16 tile) back into RGBA pixels.',
}
export const HELP_SETS = { dit: DIT_HELP }

export function renderInspectorHeaderOnly() {
  const n = selectedNode()
  if (!n) return
  const h = $('#insp-title'); if (h) h.textContent = n.data.label || n.data.kind
}

export function field(n, [key, label, type, options]) {
  const wrap = el('label', { className: 'field' + (type === 'checkbox' ? ' check' : '') })
  let input
  if (type === 'textarea') input = el('textarea', { rows: 3, value: n.data[key] ?? '' })
  else if (type === 'select') {
    input = el('select')
    if (key === 'cache_id') input.appendChild(el('option', { value: '', textContent: '- choose -' }))
    for (const o of options) { const [v, t] = Array.isArray(o) ? o : [o, o]; input.appendChild(el('option', { value: v, textContent: t })) }
    input.value = n.data[key] ?? ''
  } else if (type === 'checkbox') input = el('input', { type: 'checkbox', checked: n.data[key] !== false && !!(n.data[key] ?? (key === 'add_bos' || key === 'enabled' || key === 'moe_shared')) })
  else input = el('input', { type, value: n.data[key] ?? '', step: 'any' })
  input.addEventListener(type === 'select' || type === 'checkbox' ? 'change' : 'input', () => {
    let v = type === 'checkbox' ? input.checked : input.value
    if (type === 'number') v = v === '' ? '' : +v
    flow.updateNodeData(n.id, { [key]: v })
    // these choices decide which other fields the box has
    if (n.data.kind === 'intervention' && ['op', 'hook', 'clamp', 'source_neurons', 'neurons', 'threshold', 'value'].includes(key)) {
      if (['op', 'hook', 'clamp'].includes(key)) {
        if (key === 'op' && v === 'wire' && n.data.hook !== 'mlp_neurons') flow.updateNodeData(n.id, { hook: 'mlp_neurons' })
        renderInspector()
      }
    }
    if (n.data.kind === 'intervention' && ['op', 'hook', 'experts', 'enabled'].includes(key)) hooks.refreshBadges?.()
    if (n.data.kind === 'block_kv' && key === 'op') {
      const lab = BLOCK_KV_OP_LABELS[v]
      if (lab) flow.updateNodeData(n.id, { label: lab })
    }
    if (n.data.kind === 'block_kv' && ['op', 'text', 'layers', 'enabled'].includes(key)) hooks.refreshBadges?.()
    if (['moe_shared', 'rope_scale', 'rope_theta', 'rope_q_offset'].includes(key)) hooks.refreshBadges?.()
    if (n.data.kind === 'input_prompt') refreshRightPaneDict()
  })
  wrap.append(el('span', { textContent: label }), input)
  return wrap
}

// The conversation before the newest user turn: a list of user / assistant turns you can write by hand (to put
// words in the model's mouth), reorder and delete. With the chat template on, each is wrapped as a real turn.
export function historyEditor(n) {
  const wrap = el('div', { className: 'sec' })
  const hist = () => (n.data.history || []).map(m => ({ ...m }))
  const save = (h, redraw) => { flow.updateNodeData(n.id, { history: h }); if (redraw) renderInspector() }
  wrap.appendChild(el('h4', { textContent: `Earlier turns of the conversation (${hist().length})` }))
  wrap.appendChild(el('div', { className: 'small', textContent: hist().length
    ? 'These come before the prompt above, in this order. Assistant turns are words you put in the model’s mouth: it reads them as things it said.'
    : 'None: the prompt above is the first message. Add turns to start in the middle of a conversation, or press “Continue the conversation” in the Output tab after a run.' }))
  hist().forEach((m, i) => {
    const row = el('div', { className: `turn ${m.role}` })
    const role = el('select'); ['user', 'assistant'].forEach(r => role.appendChild(el('option', { value: r, textContent: r })))
    role.value = m.role
    role.onchange = () => { const h = hist(); h[i].role = role.value; save(h, true) }
    const ta = el('textarea', { rows: 2, value: m.content })
    ta.oninput = () => { const h = hist(); h[i].content = ta.value; save(h, false) }
    const head = el('div', { className: 'row' })
    head.append(role, btn('↑', () => { if (!i) return; const h = hist(); [h[i - 1], h[i]] = [h[i], h[i - 1]]; save(h, true) }, 'mini'),
      btn('↓', () => { const h = hist(); if (i === h.length - 1) return; [h[i + 1], h[i]] = [h[i], h[i + 1]]; save(h, true) }, 'mini'),
      btn('✕', () => { const h = hist(); h.splice(i, 1); save(h, true) }, 'mini'))
    row.append(head, ta)
    wrap.appendChild(row)
  })
  const add = el('div', { className: 'row' })
  const push = role => { const h = hist(); h.push({ role, content: '' }); save(h, true) }
  add.append(btn('+ user turn', () => push('user'), 'mini'), btn('+ assistant turn', () => push('assistant'), 'mini'))
  if (hist().length) add.append(btn('Clear all', () => save([], true), 'mini ghost'))
  wrap.appendChild(add)
  return wrap
}

export function renderInspector() {
  const box = $('#inspector')
  box.innerHTML = ''
  const n = selectedNode()
  if (!S.info) { box.innerHTML = '<div class="empty">Load a model to see its flow.</div>'; return }
  if (!n) {
    box.innerHTML = `<div class="empty"><b>${esc(S.info.arch)}</b><br>${S.info.n_layers} blocks${S.info.d_model ? ` · d=${S.info.d_model} · ${S.info.n_heads} heads (${S.info.n_kv_heads} kv) · head_dim ${S.info.head_dim}` : ''}
      <br><br>Select a box to edit it. Drag ● → ○ to connect boxes. Select an arrow and press Delete to remove it.</div>`
    return
  }
  const k = n.data.kind
  box.appendChild(el('h3', { id: 'insp-title', textContent: n.data.label || k }))
  box.appendChild(el('div', { className: 'help', textContent: (k === 'block' && n.data.moe ? KIND_HELP.block_moe : HELP_SETS[texts()]?.[k]) || KIND_HELP[k] || '' }))
  if (k === 'init_image') box.appendChild(startImageEditor(n))
  else (FIELDS[k]?.(schema().modality, n) || []).forEach(f => box.appendChild(field(n, f)))
  if (k === 'input_prompt' && schema().modality !== 'image') box.appendChild(historyEditor(n))
  const actions = el('div', { className: 'row' })
  if (k === 'block') {
    const ropeOff = n.data.rope_scale === 0
    actions.append(btn('Repeat block', () => duplicateBlock(n)), btn(n.data.bypass ? 'Restore' : 'Bypass', () => bypassBlock(n)),
      btn(ropeOff ? 'RoPE: Restore' : 'RoPE: Turn off (0×)', () => {
        flow.updateNodeData(n.id, { rope_scale: ropeOff ? '' : 0 })
        hooks.refreshBadges?.()
        renderInspector()
      }),
      btn('+ Interceptor', () => addIntervention(n)),
      btn('+ Fork Head', () => addForkHead(n)),
      btn('Debug: run to here', () => debugRunTo(n)))
    if (n.data.has_kv !== false && can('block_kv')) {
      actions.append(btn('+ Rewrite prompt (KV)', () => addBlockKV(n)))
    }
  }
  if (k === 'fork_head') {
    actions.append(
      btn(n.data.enabled === false ? 'Enable' : 'Disable', () => {
        flow.updateNodeData(n.id, { enabled: n.data.enabled === false }); renderInspector(); hooks.refreshBadges?.()
      }),
      btn('Open Guardrail Fork tab', () => setTab('guardrail')),
    )
  }
  if (k === 'block_kv') {
    actions.append(
      btn(n.data.enabled === false ? 'Enable' : 'Disable', () => {
        flow.updateNodeData(n.id, { enabled: n.data.enabled === false }); renderInspector(); hooks.refreshBadges?.()
      }),
      btn('Open KV caches', () => setTab('kv')),
    )
  }
  if (k === 'block' && n.data.moe) hooks.expertPanel?.(n, box)
  if (k === 'intervention' && ['router', 'expert_out'].includes(n.data.hook) && tabAvailable('experts')) actions.append(btn('Open Experts tab', () => setTab('experts')))
  if (k === 'intervention') {                // shortcuts to the tab that goes with this kind of interceptor, if the model has it
    if (n.data.op === 'patch' && tabAvailable('patch')) actions.append(btn('Find where to patch (sweep)', () => setTab('patch')))
    if ((n.data.op === 'steer' || n.data.op === 'project_out') && tabAvailable('steer')) actions.append(btn('Make a steering vector', () => setTab('steer')))
    if (n.data.hook === 'mlp_neurons' && tabAvailable('neurons')) actions.append(btn('Find neurons', () => setTab('neurons')))
    if ((n.data.op === 'quantize' || n.data.hook === 'weights') && tabAvailable('quant')) actions.append(btn('Open Quantize tab', () => setTab('quant')))
    if ((n.data.hook === 'kv_k' || n.data.hook === 'kv_v') && tabAvailable('kv')) actions.append(btn('Open KV caches tab', () => setTab('kv')))
  }
  if (k === 'unet_block' && canRewire(n)) actions.append(btn('Repeat block', () => duplicateBlock(n)), btn(n.data.bypass ? 'Restore' : 'Bypass', () => bypassBlock(n)))
  if (k === 'unet_block') actions.append(btn('+ Interceptor', () => addIntervention(n)))
  if (k === 'kv_import') actions.append(btn('Open KV caches', () => setTab('kv')))
  if (k === 'intervention' && S.info?.capabilities?.regions) box.appendChild(regionEditor(n))
  if (actions.children.length) box.appendChild(actions)
  codeSection(n, box)
  if ((k === 'tokenizer' || k === 'input_prompt') && schema().modality !== 'image') renderRightPaneTokenDict(box)
  renderNodeResults(n, box)
}

let _dictTimer = null
function refreshRightPaneDict() {
  clearTimeout(_dictTimer)
  _dictTimer = setTimeout(() => {
    const target = document.getElementById('insp-word-dict')
    if (target) populateRightPaneTokenDict(target)
  }, 180)
}

export function renderRightPaneTokenDict(box) {
  const wrap = el('div', { id: 'insp-word-dict', className: 'results' })
  box.appendChild(wrap)
  populateRightPaneTokenDict(wrap)
}

async function populateRightPaneTokenDict(wrap) {
  const pNode = flow.nodes.find(x => x.data.kind === 'input_prompt')
  if (!pNode || !S.info) return
  const d = pNode.data
  try {
    const r = await POST('/api/tokenize', {
      text: d.prompt || '',
      chat: !!d.use_chat_template,
      system: d.system || '',
      history: d.history || [],
      prefill: d.prefill || '',
      add_bos: !!d.add_bos,
    })
    wrap.innerHTML = ''
    const dictContainer = el('div')
    const nbContainer = el('div')
    S.tk = S.tk || { picked: null }
    renderWordDictionary(dictContainer, r, S.tk, nbContainer)
    wrap.append(dictContainer, nbContainer)
    if (S.tk.picked != null && r.ids.includes(S.tk.picked)) neighbors(nbContainer, S.tk.picked)
  } catch (e) {
    wrap.innerHTML = `<div class="small bad">${esc(e.message)}</div>`
  }
}

// ---------- painting (start image, lock region, intervention regions)
export function genSize() {
  const nz = flow.nodes.find(x => x.data.kind === 'noise')?.data || {}
  return [+nz.width || S.info?.default_size || 512, +nz.height || S.info?.default_size || 512]
}

export function startImageEditor(n) {
  const wrap = el('div', { className: 'sec' })
  const [W, H] = genSize()
  const lastOut = S.run?.outputs?.image
  wrap.appendChild(el('h4', { textContent: 'Start picture' }))
  wrap.appendChild(el('div', { className: 'small', textContent: n.data.image ? 'Denoising starts from this picture. Draw on it, or replace it.' : 'Draw a sketch, open a file, or take the last output.' }))
  wrap.appendChild(maskPreview(n.data.image, ''))
  const row = el('div', { className: 'row' })
  const file = el('input', { type: 'file', accept: 'image/*', style: 'display:none' })
  file.onchange = () => { const r = new FileReader(); r.onload = () => { flow.updateNodeData(n.id, { image: r.result, guide: false }); renderInspector() }; r.readAsDataURL(file.files[0]) }
  wrap.appendChild(file)
  row.append(
    btn('📂 Load image file…', () => file.click()),
    btn(n.data.image ? '✏️ Edit picture' : '✏️ Draw picture', () => openPaint({ mode: 'image', title: 'Start picture', width: W, height: H,
      value: n.data.image, noiseCanvas: false, backgrounds: [['Load last output', lastOut]],
      onSave: (url) => { flow.updateNodeData(n.id, { image: url, guide: false }); renderInspector() } })),
    ...(lastOut ? [btn('Use last output', () => { flow.updateNodeData(n.id, { image: lastOut, guide: false }); renderInspector() })] : []),
    ...(n.data.image ? [btn('Remove', () => { flow.updateNodeData(n.id, { image: '', mask: '', guide: false }); renderInspector() }, 'ghost')] : []))
  wrap.appendChild(row)
  if (n.data.image) {
    const steps = +(flow.nodes.find(x => x.data.kind === 'scheduler')?.data.steps) || 20
    const sig = S.run?.outputs?.sigmas?.length === steps ? S.run.outputs.sigmas : null
    const st = el('div', { className: 'strength' })
    const lab = el('div', { className: 'small' })
    const sl = el('input', { type: 'range', min: 0, max: 1, step: 0.05, value: n.data.strength ?? 0.6 })
    const show = () => {
      const v = +sl.value, k = Math.min(steps - 1, Math.round(steps * (1 - v)))
      lab.innerHTML = `<b>Noise level ${v.toFixed(2)}</b> · starts at step ${k} of ${steps} with ` +
        (sig ? `<b>${Math.round((1 - sig[k]) * 100)}% your picture + ${Math.round(sig[k] * 100)}% noise</b>. ` : 'your picture mixed with noise. ') +
        (v < 0.45 ? 'Low noise: keeps your picture closely and refines details.'
          : v < 0.8 ? 'Medium noise: keeps your layout and colours; the model renders surfaces and details.'
            : 'High noise: only rough layout survives.')
    }
    sl.oninput = show
    sl.onchange = () => { flow.updateNodeData(n.id, { strength: +sl.value, guide: false }); renderInspector() }
    show()
    st.append(lab, sl)
    wrap.appendChild(st)
  }
  return wrap
}

export function regionEditor(n) {
  const wrap = el('div', { className: 'sec' })
  const [W, H] = genSize()
  const start = flow.nodes.find(x => x.data.kind === 'init_image')?.data.image
  const bgImg = S.run?.outputs?.image || start
  wrap.appendChild(el('h4', { textContent: 'Region' }))
  wrap.appendChild(el('div', { className: 'small', textContent: n.data.region
    ? 'The intervention only acts on the painted tiles (hook out: what the block writes there; hook word: how much those tiles read the chosen words).'
    : 'Everywhere. Paint a region to limit this intervention to part of the image.' }))
  wrap.appendChild(maskPreview(bgImg, n.data.region))
  const row = el('div', { className: 'row' })
  row.append(btn(n.data.region ? '🖌️ Edit region' : '🖌️ Paint region', () => openPaint({ mode: 'mask', title: `Region · ${n.data.label || 'intervention'}`,
    background: bgImg, width: W, height: H, value: n.data.region,
    backgrounds: [['Show last output', S.run?.outputs?.image], ['Show start picture', start]],
    onSave: url => { flow.updateNodeData(n.id, { region: url }); renderInspector() } })),
    ...(n.data.region ? [btn('Everywhere', () => { flow.updateNodeData(n.id, { region: '' }); renderInspector() }, 'ghost')] : []))
  wrap.appendChild(row)
  return wrap
}

export function renderNodeResults(n, box) {
  const run = S.run, o = run?.outputs
  if (!o) return
  const k = n.data.kind
  const sec = el('div', { className: 'results' })
  const st = o.block_stats?.[n.id]
  if (st && 'resid_norm' in st) {                    // a language-model block: norms and lens
    sec.innerHTML = `<h4>Last run · schedule position #${st.pos}</h4>
      <div class="kv"><span>‖resid‖</span><b>${st.resid_norm}</b><span>‖attn out‖</span><b>${st.attn_norm}</b><span>‖mlp out‖</span><b>${st.mlp_norm}</b>
      <span>lens @ last token</span><b>${esc(st.lens_last ?? '')}</b></div>`
    const row = el('div', { className: 'row' })
    row.append(btn('Attention of this block', () => { S.attnPos = st.pos; setTab('attn') }),
      btn('Open KV caches tab', () => setTab('kv')))
    sec.appendChild(row)
    if (o.cache && n.data.has_kv !== false) {
      const bOv = o.cache.block_overrides?.[st.pos] || o.cache.block_overrides?.[String(st.pos)]
      const kvSrc = bOv || o.cache
      const segOf = i => (kvSrc.segments || []).find(s => i >= s.start && i < s.end)?.origin || 'prompt'
      const kvHead = el('h4', {
        textContent: bOv
          ? `Block KV cache (${bOv.op} · ${bOv.n_tokens} tokens, ${bOv.n_prompt} recreated prompt)`
          : `Block KV cache (shared · ${kvSrc.n_tokens} tokens)`,
      })
      const kvToks = el('div', { className: 'tokens compact' })
      ;(kvSrc.tokens || []).forEach((t, i) => {
        const pos = kvSrc.positions?.[i] ?? i
        kvToks.appendChild(el('span', {
          className: `tok seg-${segOf(i)}`,
          title: `#${i} · origin: ${segOf(i)} · RoPE pos ${pos}`,
        }, `${esc(t)}<sub>${pos}</sub>`))
      })
      sec.append(kvHead, kvToks)
    }
    if (S.lens) {
      const li = S.lens.layers.findIndex(l => l.node === n.id)
      if (li >= 0) {
        const row = S.lens.cells[li]
        sec.appendChild(el('div', { className: 'lensrow' }, row.slice(-16).map((c, i) =>
          `<span title="${esc(c.top.map(t => `${t[0]} ${t[1]}`).join('\n'))}" style="background:${heat(c.p_final)}">${esc(c.top[0][0])}</span>`).join('')))
      }
    }
  }
  if (st?.std_by_step) {                             // an image-model block: activation spread per step
    sec.innerHTML = `<h4>Activation std by step</h4>${spark(st.std_by_step)}`
    const grp = n.id.replace('down_', 'down_blocks.').replace('up_', 'up_blocks.').replace(/^mid$/, 'mid_block')
    if (o.xattn_groups?.includes(grp)) { S.xgroup = grp; sec.appendChild(btn('Cross-attention of this stage', () => setTab('output'))) }
  }
  if (k === 'output_text' || k === 'sampler' || k === 'lm_head') renderTokens(sec, true)
  if (k === 'embed' && o.cache) {
    sec.innerHTML += `<h4>KV cache after the run</h4><div class="small">${o.cache.n_tokens} tokens × ${o.cache.n_slots} slots · ${(o.cache.bytes / 2 ** 20).toFixed(2)} MB${o.prefix_tokens ? ` · ${o.prefix_tokens} imported` : ''}</div>`
    const row = el('div', { className: 'row' })
    row.append(btn('Analyze KV cache', () => setTab('kv')), btn('Save this cache', saveRunCache))
    sec.appendChild(row)
  }
  if ((k === 'output_image' || k === 'vae_decode') && o.image) sec.appendChild(el('img', { src: o.image, className: 'thumb big' }))
  if (k === 'denoiser_out' || k === 'scheduler' || k === 'denoiser_in') sec.appendChild(stepStrip())
  if ((k === 'text_encoder' || (k === 'input_prompt' && schema().modality === 'image')) && o.tokens) {
    sec.innerHTML += `<h4>Tokens</h4><div class="tokens">${o.tokens.map(t => `<span class="tok">${esc(t)}</span>`).join('')}</div>`
  }
  if (sec.childNodes.length) box.appendChild(sec)
}
hooks.renderInspector = renderInspector
hooks.renderInspectorHeaderOnly = renderInspectorHeaderOnly
