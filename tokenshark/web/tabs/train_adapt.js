// Train tab section: adapt the loaded language model (LoRA, full matrices, row scales, norms, thin layers, new blocks).
import { GET, POST, S, btn, el, toast } from '../core.js'
import { labelled, numInput, selectOf } from '../ui/labkit.js'
import { renderTab, setTab } from '../ui/tabs.js'
import { jobView, registerTrainSection, renderDatasetPreviewTable, splitQuestionAnswer } from './train.js'

function adaptSection(st, running) {
  const lf = st.lform
  const b = el('div', { className: 'kvbox' })
  const METHODS = [
    ['lora', 'LoRA · low-rank add-on on chosen matrices', 'Each matrix you tick, in each block you choose, gets its own small pair: W + (α/r)·B·A. Set Blocks to one number to adapt a single block.'],
    ['full', 'Fine-tune chosen matrices fully', 'No low-rank shortcut: the chosen matrices themselves are trained.'],
    ['ia3', 'Row scales only (IA3)', 'One number per output row of each chosen matrix: W <- diag(s)·W.'],
    ['norms', 'Norm weights only', 'Trains just the two normalisation vectors of the chosen blocks.'],
    ['adapter', 'Add a thin layer after chosen blocks', 'New weights: after each chosen block the stream passes through a bottleneck (squeeze to r numbers, expand back, add).'],
    ['new_block', 'Add a whole new block and train it', 'A new block is inserted after the block you choose and only it is trained.'],
  ]
  b.appendChild(el('div', { className: 'small' }, `<b>2 · Adapt the loaded model.</b> The model's base weights stay frozen and a chosen component is trained on your examples. In <code>question =&gt; answer</code> pairs, the question and custom system prompt go through the chat template and loss is computed only on the expected answer.`))
  const nB = S.info?.n_layers ?? 1
  const hasExtra = (S.info?.extra_blocks || []).length > 0
  if (hasExtra) b.appendChild(el('div', { className: 'lneed', textContent: 'A trained block is attached to this model. Detach it in the LoRA / Fine-tune tab before training something new: training always starts from the base model.' }))

  const multiMeta = (st.datasets || []).find(d => d.id === 'multiarith_cot')
  const multiCount = multiMeta?.n_examples || 2000

  // Dataset preset loader + Question count + Quick Add Q&A controls
  const dsBar = el('div', { className: 'row', style: 'margin:6px 0;gap:8px;flex-wrap:wrap;align-items:flex-end' })
  const dsSel = selectOf([
    ['', 'Custom examples (editable below)'],
    ['multiarith_cot', `MultiArith Step-by-Step Reasoning CoT (${multiCount.toLocaleString()} questions)`],
    ...(st.datasets || []).filter(d => d.id !== 'multiarith_cot').map(d => [d.id, `${d.name} (${(d.n_examples || 0).toLocaleString()} questions)`]),
  ], lf.dataset_id || '')

  const loadDatasetIntoAdapt = async (dsId, maxQ) => {
    if (!dsId) {
      lf.dataset_id = ''
      lf.pv_offset = 0
      renderTab()
      return
    }
    try {
      const fetchLimit = Math.min(2000, Math.max(100, maxQ || 200))
      const d = await GET(`/api/datasets/${encodeURIComponent(dsId)}?offset=0&limit=${fetchLimit}`)
      lf.dataset_id = dsId
      lf.dataset_total = d.n_examples || (d.rows || []).length
      lf.pv_offset = 0
      if (d?.rows?.length) {
        const subset = d.rows.slice(0, maxQ || d.rows.length)
        lf.examples = subset.map(r => `${r.question} => ${(r.answer || '').replace(/\n/g, ' ')}`).join('\n')
        if (d.sample_prompt) lf.probe = d.sample_prompt.replace(/\nStep 1:.*$/s, '').replace(/^Question:\s*/i, '')
        if (d.system_prompt) lf.system_prompt = d.system_prompt
      }
      renderTab()
    } catch (e) { toast(e.message, 'error') }
  }

  dsSel.onchange = () => loadDatasetIntoAdapt(dsSel.value, lf.n_questions || 100)

  const nQInput = el('input', { type: 'number', min: 1, max: 2000, step: 25, value: lf.n_questions || 100, style: 'width:80px' })
  nQInput.onchange = () => {
    lf.n_questions = Math.max(1, +nQInput.value || 100)
    if (lf.dataset_id) loadDatasetIntoAdapt(lf.dataset_id, lf.n_questions)
    else renderTab()
  }

  const toggleRawBtn = btn(lf.edit_raw ? 'Hide Raw Editor' : 'Edit Raw Examples / Add Q&A', () => {
    lf.edit_raw = !lf.edit_raw
    renderTab()
  }, 'mini')

  dsBar.append(
    labelled('Dataset / Examples Source', dsSel),
    labelled('Questions to train', nQInput),
    toggleRawBtn,
  )

  // Custom System Prompt input
  const sysRow = el('div', { className: 'row', style: 'margin:6px 0;gap:8px;align-items:center' })
  const sysInp = el('input', {
    type: 'text',
    value: lf.system_prompt || '',
    placeholder: 'Optional custom system prompt applied during training & test probe (e.g. You are a step-by-step math assistant.)',
    style: 'flex:1;min-width:260px',
  })
  sysInp.oninput = () => { lf.system_prompt = sysInp.value }
  sysRow.append(el('span', { className: 'small', style: 'white-space:nowrap' }, '<b>System Prompt:</b>'), sysInp)

  // Preview table container + optional raw editor
  const editorWrap = el('div')
  const pvContainer = el('div')

  const refreshAdaptPreview = () => {
    pvContainer.innerHTML = ''
    const allLines = (lf.examples || '').split('\n').map(x => x.trim()).filter(Boolean)
    const parsedRows = allLines.map(splitQuestionAnswer)
    const totalAvailable = lf.dataset_id && lf.dataset_total ? Math.max(lf.dataset_total, parsedRows.length) : parsedRows.length
    const activeQ = Math.min(lf.n_questions || parsedRows.length, parsedRows.length)
    const off = lf.pv_offset || 0
    const lim = lf.pv_limit || 50
    pvContainer.appendChild(renderDatasetPreviewTable({
      title: lf.dataset_id ? `Adaptation Dataset Preview (${lf.dataset_id})` : 'Adaptation Examples Preview (Question -> Expected Answer)',
      rows: parsedRows.slice(off, off + lim),
      totalCount: totalAvailable,
      trainCount: activeQ,
      offset: off,
      limit: lim,
      systemPrompt: lf.system_prompt,
      onPageChange: newOff => { lf.pv_offset = newOff; refreshAdaptPreview() },
    }))
  }

  if (lf.edit_raw) {
    const addRow = el('div', { className: 'row', style: 'margin:4px 0;gap:6px;flex-wrap:wrap' })
    const qIn = el('input', { type: 'text', placeholder: 'User question / prompt…', style: 'flex:1;min-width:180px' })
    const aIn = el('input', { type: 'text', placeholder: 'Expected answer / reasoning…', style: 'flex:1;min-width:180px' })
    addRow.append(
      qIn,
      aIn,
      btn('+ Append Q&A Row', () => {
        if (!qIn.value.trim() || !aIn.value.trim()) return toast('Enter both a question and an expected answer.', 'error')
        lf.examples = (lf.examples ? lf.examples.trim() + '\n' : '') + `${qIn.value.trim()} => ${aIn.value.trim()}`
        qIn.value = ''
        aIn.value = ''
        refreshAdaptPreview()
        ex.value = lf.examples
      }, 'mini'),
    )
    const ex = el('textarea', { rows: 5, value: lf.examples, placeholder: 'One example per line: Question => Expected Answer' })
    ex.oninput = () => { lf.examples = ex.value; lf.dataset_id = ''; lf.pv_offset = 0; refreshAdaptPreview() }
    editorWrap.append(addRow, ex)
  }

  refreshAdaptPreview()

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
    g2.append(labelled(`Insert after block (0...${nB - 1})`, after), labelled('It starts as', init))
  } else g2.append(li('layers', `Blocks (all | 2 | 0:4 · this model has 0...${nB - 1})`, 70, 'text'))
  g2.append(li('steps', 'Training steps'), li('lr', 'Learning rate', 80), li('probe', 'Test prompt (before / after)', 260, 'text'))
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
    try {
      const lines = lf.examples.split('\n').map(x => x.trim()).filter(Boolean)
      const maxQ = lf.n_questions ? Math.min(lf.n_questions, lines.length) : lines.length
      await POST('/api/train/lora', {
        ...lf,
        examples: lines.slice(0, maxQ),
        n_questions: maxQ,
        system_prompt: lf.system_prompt || '',
      })
      renderTab()
    } catch (e) { toast(e.message, 'error') }
  }, 'primary'))
  brow.lastChild.disabled = running(st.lora) || hasExtra
  if (running(st.lora)) brow.append(btn('■ Stop', async () => { await POST('/api/train/lora/stop'); renderTab() }))
  b.append(labelled('Method', msel), mhelp, dsBar, sysRow, editorWrap, pvContainer, g2, brow)
  if (st.lora && st.lora.status !== 'none') {
    b.appendChild(jobView(st.lora, j => j.result?.lora ? btn('Open the LoRA / Fine-tune inspector', () => setTab('lora'), 'primary') : null))
    if (st.lora.method || st.lora.rank) b.appendChild(el('div', { className: 'small', textContent: `Last trained: ${st.lora.method || 'lora'} · ${(st.lora.n_params || 0).toLocaleString()} trained weights · ${(st.lora.n_examples || 0).toLocaleString()} examples · final loss ${st.lora.losses?.length ? st.lora.losses[st.lora.losses.length - 1][1] : '-'}` }))
  }
  return b
}

registerTrainSection('adapt', { needsModel: true, render: adaptSection })
