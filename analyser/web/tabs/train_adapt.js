// Train tab section: adapt the loaded language model (LoRA, full matrices, row scales, norms, thin layers, new blocks).
import { POST, S, btn, el, toast } from '../core.js'
import { labelled, numInput, selectOf } from '../ui/labkit.js'
import { renderTab, setTab } from '../ui/tabs.js'
import { jobView, registerTrainSection } from './train.js'

function adaptSection(st, running) {
  const lf = st.lform
  const b = el('div', { className: 'kvbox' })
  const METHODS = [
    ['lora', 'LoRA — a low-rank add-on on chosen matrices', 'Each matrix you tick, in each block you choose, gets its own small pair: W + (α/r)·B·A. It is not one extra layer. The usual recipe is the query and value matrices in every block, which is why every block then shows as changed. Set Blocks to one number to adapt a single block.'],
    ['full', 'Fine-tune chosen matrices fully', 'No low-rank shortcut: the chosen matrices themselves are trained. Many more numbers than LoRA for the same matrices — compare how much better (or not) it fits.'],
    ['ia3', 'Row scales only (the thinnest)', 'One number per output row of each chosen matrix: W ← diag(s)·W. A few hundred numbers per matrix. How far can re-weighting what already exists go?'],
    ['norms', 'Norm weights only', 'Trains just the two normalisation vectors of the chosen blocks. Tiny, and surprisingly effective for style.'],
    ['adapter', 'Add a thin layer after chosen blocks', 'New weights: after each chosen block the stream passes through a bottleneck — squeeze to r numbers, expand back, add. It starts as “add nothing”, so the model is unchanged until it learns.'],
    ['new_block', 'Add a whole new block and train it', 'A new block is inserted after the block you choose and only it is trained. It can start as a copy of that block, as the average of its two neighbours, as a copy with its outputs zeroed (changes nothing at first), or as random numbers. A new box appears in the flow.'],
  ]
  b.appendChild(el('div', { className: 'small' }, `<b>2 · Adapt the loaded model.</b> The model’s own weights stay frozen and a small part is trained on your examples. <b>Which</b> part is the experiment:
    a low-rank add-on, a few whole matrices, only scales, only norms, a new thin layer, or a whole new block. Train the same examples with different methods and compare the loss curve, the number of trained weights, and what the LoRA / Fine-tune tab shows.
    One example per line; <code>question =&gt; answer</code> trains only the answer, through the chat template.`))
  const nB = S.info?.n_layers ?? 1
  const hasExtra = (S.info?.extra_blocks || []).length > 0
  if (hasExtra) b.appendChild(el('div', { className: 'lneed', textContent: 'A trained block is attached to this model. Detach it in the LoRA / Fine-tune tab before training something new: training always starts from the base model.' }))
  const ex = el('textarea', { rows: 5, value: lf.examples }); ex.oninput = () => { lf.examples = ex.value }
  const msel = selectOf(METHODS.map(m => [m[0], m[1]]), lf.method); msel.onchange = () => { lf.method = msel.value; renderTab() }
  const mhelp = el('div', { className: 'small', textContent: METHODS.find(m => m[0] === lf.method)[2] })
  const g2 = el('div', { className: 'formgrid' })
  const li = (key, label, w = 70, type = 'number') => { const i = el('input', { type, value: lf[key], step: 'any', style: `width:${w}px` }); i.oninput = () => { lf[key] = type === 'number' ? +i.value : i.value }; return labelled(label, i) }
  const M = lf.method
  g2.append(li('name', 'Name', 140, 'text'))
  if (M === 'lora') g2.append(li('rank', 'Rank r'), li('alpha', 'α'))
  if (M === 'adapter') g2.append(li('rank', 'Width r of the thin layer'))
  if (M === 'new_block') {
    const init = selectOf([['duplicate', 'a copy of that block'], ['average', 'the average of that block and the next'], ['identity', 'a copy with outputs zeroed (changes nothing at first)'], ['random', 'small random numbers']], lf.init)
    init.onchange = () => { lf.init = init.value }
    const after = numInput(Math.min(lf.after, nB - 1), 60, { min: 0, max: nB - 1 }); after.oninput = () => { lf.after = +after.value }
    g2.append(labelled(`Insert after block (0…${nB - 1})`, after), labelled('It starts as', init))
  } else g2.append(li('layers', `Blocks (all | 2 | 0:4 — this model has 0…${nB - 1})`, 70, 'text'))
  g2.append(li('steps', 'Steps'), li('lr', 'Learning rate', 80), li('probe', 'Test prompt (before / after)', 260, 'text'))
  if (['lora', 'full', 'ia3'].includes(M)) {
    const tg = el('span', { className: 'inl' }); tg.appendChild(el('span', { textContent: 'Matrices' }))
    for (const t of ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']) {
      const cb = el('input', { type: 'checkbox', checked: lf.targets.includes(t) })
      cb.onchange = () => { lf.targets = cb.checked ? [...lf.targets, t] : lf.targets.filter(x => x !== t) }
      const l = el('label', { className: 'inl' }); l.append(cb, el('span', { textContent: t.replace('_proj', '') })); tg.appendChild(l)
    }
    g2.appendChild(tg)
  }
  const brow = el('div', { className: 'row' })
  brow.append(btn(running(st.lora) ? 'Training…' : 'Train', async () => {
    try { await POST('/api/train/lora', { ...lf, examples: lf.examples.split('\n') }); renderTab() } catch (e) { toast(e.message, 'error') }
  }, 'primary'))
  brow.lastChild.disabled = running(st.lora) || hasExtra
  if (running(st.lora)) brow.append(btn('■ Stop', async () => { await POST('/api/train/lora/stop'); renderTab() }))
  b.append(labelled('Method', msel), mhelp, ex, g2, brow)
  if (st.lora && st.lora.status !== 'none') {
    b.appendChild(jobView(st.lora, j => j.result?.lora ? btn('Open the LoRA / Fine-tune inspector', () => setTab('lora'), 'primary') : null))
    if (st.lora.method || st.lora.rank) b.appendChild(el('div', { className: 'small', textContent: `Last trained: ${st.lora.method || 'lora'} · ${(st.lora.n_params || 0).toLocaleString()} trained weights · final loss ${st.lora.losses?.length ? st.lora.losses[st.lora.losses.length - 1][1] : '—'}` }))
  }
  return b
}

registerTrainSection('adapt', { needsModel: true, render: adaptSection })
