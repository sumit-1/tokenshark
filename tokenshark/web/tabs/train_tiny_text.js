// Train tab section: a small transformer trained from scratch on text or reasoning datasets.
import { $, GET, POST, btn, el, toast } from '../core.js'
import { labelled, selectOf } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'
import { jobView, registerTrainSection, renderDatasetPreviewTable, splitQuestionAnswer } from './train.js'
import { refreshModels } from '../ui/models.js'

function tinyTextSection(st, running) {
  const f = st.form
  const a = el('div')
  a.appendChild(el('div', { className: 'small' }, `<b>1 · Train a small transformer from scratch.</b> Weights start as random numbers; each step the model predicts the next token, measures cross-entropy loss, and updates weights. Includes built-in synthetic corpora, 2,000+ step-by-step arithmetic reasoning problems (<code>multiarith_cot</code>), and custom dataset uploads.`))

  const g = el('div', { className: 'formgrid' })
  const inp = (key, label, w = 70, type = 'number') => {
    const i = el('input', { type, value: f[key], step: 'any', style: `width:${w}px` })
    i.oninput = () => { f[key] = type === 'number' ? +i.value : i.value }
    return labelled(label, i)
  }

  const multiMeta = (st.datasets || []).find(d => d.id === 'multiarith_cot')
  const multiCount = multiMeta?.n_examples || 2000

  const corpusOpts = [
    ['mixed', 'built-in: facts + counting (procedural)'],
    ['facts', 'built-in: capitals and animal sounds (procedural)'],
    ['counting', 'built-in: counting and sums (procedural)'],
    ['multiarith_cot', `reasoning: MultiArith Step-by-Step CoT (${multiCount.toLocaleString()} questions)`],
    ...(st.datasets || []).filter(d => d.id !== 'multiarith_cot').map(d => [d.id, `dataset: ${d.name} (${(d.n_examples || 0).toLocaleString()} questions)`]),
    ['own', 'custom text / upload dataset'],
  ]
  const corpus = selectOf(corpusOpts, f.corpus)
  corpus.onchange = () => {
    f.corpus = corpus.value
    f.pv_offset = 0
    if (f.corpus === 'multiarith_cot') {
      f.sample_prompt = 'Question: Julia had 12 apples. She gave 4 to her friend and bought 3 packs of 5 apples each. How many apples does she have now?\nStep 1:'
      if (!f.system_prompt) f.system_prompt = multiMeta?.system_prompt || 'You are a helpful math reasoning assistant. Solve each problem step by step and end with \'Answer: <number>\'.'
    } else if (f.corpus === 'counting') {
      f.sample_prompt = '3 + 5 ='
    } else if (f.corpus === 'facts' || f.corpus === 'mixed') {
      f.sample_prompt = 'The capital of France is'
    } else {
      const ds = (st.datasets || []).find(d => d.id === f.corpus)
      if (ds?.sample_prompt) f.sample_prompt = ds.sample_prompt
      if (ds?.system_prompt && !f.system_prompt) f.system_prompt = ds.system_prompt
    }
    renderTab()
  }

  const nQInput = el('input', { type: 'number', min: 1, max: 10000, step: 50, value: f.n_questions || 1000, style: 'width:85px' })
  nQInput.onchange = () => {
    f.n_questions = Math.max(1, +nQInput.value || 1000)
    renderTab()
  }

  g.append(
    inp('name', 'Name', 140, 'text'),
    labelled('Dataset / Corpus', corpus),
    labelled('Questions to train', nQInput),
    inp('steps', 'Training steps', 80),
    inp('batch', 'Batch'),
    inp('lr', 'Learning rate', 80),
    inp('d_model', 'Width d_model'),
    inp('n_layers', 'Blocks'),
    inp('n_heads', 'Heads'),
    inp('context', 'Context (tokens)'),
    inp('vocab', 'Vocabulary'),
    inp('n_experts', 'Experts / block (0=dense)', 50),
    inp('experts_per_token', 'Experts / token', 50),
    inp('sample_prompt', 'Test prompt', 210, 'text'),
  )
  a.appendChild(g)

  // Custom System Prompt row
  const sysRow = el('div', { className: 'row', style: 'margin:6px 0;gap:8px;align-items:center' })
  const sysInp = el('input', {
    type: 'text',
    value: f.system_prompt || '',
    placeholder: 'Optional custom system prompt prepended to training examples (e.g. You are a step-by-step math tutor.)',
    style: 'flex:1;min-width:260px',
  })
  sysInp.oninput = () => { f.system_prompt = sysInp.value }
  sysRow.append(el('span', { className: 'small', style: 'white-space:nowrap' }, '<b>System Prompt:</b>'), sysInp)
  a.appendChild(sysRow)

  if (f.n_experts > 0) {
    a.appendChild(el('div', { className: 'small' }, `<b>Mixture of experts.</b> Each block gets ${f.n_experts} small MLPs and a router that sends every token to ${f.experts_per_token} of them.`))
  }

  const pvContainer = el('div')
  a.appendChild(pvContainer)

  if (f.corpus === 'own') {
    const ta = el('textarea', {
      rows: 4,
      placeholder: 'Paste training examples here (one per line, "Question => Answer", JSONL {"question":"...","steps":[...],"answer":"..."}, or CSV)…',
      value: f.text,
    })
    const refreshOwnPreview = () => {
      pvContainer.innerHTML = ''
      const rawLines = (f.text || '').split(/\n\n|\r?\n/).map(x => x.trim()).filter(Boolean)
      const parsedRows = rawLines.map(splitQuestionAnswer)
      const off = f.pv_offset || 0
      const lim = f.pv_limit || 50
      pvContainer.appendChild(renderDatasetPreviewTable({
        title: 'Custom Text Preview',
        rows: parsedRows.slice(off, off + lim),
        totalCount: parsedRows.length,
        trainCount: Math.min(f.n_questions || 1000, parsedRows.length),
        offset: off,
        limit: lim,
        systemPrompt: f.system_prompt,
        onPageChange: newOff => { f.pv_offset = newOff; refreshOwnPreview() },
      }))
    }
    ta.oninput = () => { f.text = ta.value; f.pv_offset = 0; refreshOwnPreview() }
    const upRow = el('div', { className: 'row', style: 'margin:4px 0' })
    const dsName = el('input', { type: 'text', placeholder: 'Save as dataset name (optional)', style: 'width:190px' })
    upRow.append(
      dsName,
      btn('Save to Datasets Library', async () => {
        if (!f.text.trim()) return toast('Paste dataset text first.', 'error')
        try {
          const d = await POST('/api/datasets/upload', {
            name: dsName.value || 'custom_dataset',
            content: f.text,
            system_prompt: f.system_prompt || '',
            sample_prompt: f.sample_prompt || '',
          })
          st.datasets = await GET('/api/datasets')
          f.corpus = d.id
          toast(`Saved dataset '${d.name}' (${d.n_examples} examples)`)
          renderTab()
        } catch (e) { toast(e.message, 'error') }
      }, 'mini'),
    )
    a.insertBefore(ta, pvContainer)
    a.insertBefore(upRow, pvContainer)
    refreshOwnPreview()
  } else {
    const loadPreview = (off = f.pv_offset || 0) => {
      f.pv_offset = off
      const lim = f.pv_limit || 50
      GET(`/api/train/corpus/preview?corpus=${encodeURIComponent(f.corpus)}&offset=${off}&limit=${lim}&n_questions=${f.n_questions || 1000}`)
        .then(r => {
          pvContainer.innerHTML = ''
          pvContainer.appendChild(renderDatasetPreviewTable({
            title: `Dataset Preview · ${r.name || f.corpus}`,
            rows: r.rows || [],
            totalCount: r.total || (r.rows || []).length,
            trainCount: f.n_questions || r.active_questions || r.total,
            offset: r.offset || 0,
            limit: r.limit || lim,
            systemPrompt: f.system_prompt || r.system_prompt || '',
            onPageChange: newOff => loadPreview(newOff),
          }))
        })
        .catch(() => {})
    }
    loadPreview(f.pv_offset || 0)
  }

  const arow = el('div', { className: 'row' })
  arow.append(btn(running(st.tiny) ? 'Training…' : 'Start training', async () => {
    try {
      await POST('/api/train/tiny', {
        ...f,
        text: f.corpus === 'own' ? f.text : '',
        corpus: f.corpus === 'own' ? 'mixed' : f.corpus,
        n_questions: f.n_questions || 1000,
        system_prompt: f.system_prompt || '',
      })
      renderTab()
    } catch (e) { toast(e.message, 'error') }
  }, 'primary'))
  arow.lastChild.disabled = running(st.tiny)
  if (running(st.tiny)) arow.append(btn('■ Stop (keeps what it learned)', async () => { await POST('/api/train/tiny/stop'); renderTab() }))
  a.appendChild(arow)

  if (st.tiny && st.tiny.status !== 'none') a.appendChild(jobView(st.tiny, j => {
    if (!j.result?.model_id) return null
    return btn(`Load "${j.result.model_id}" into the app`, async () => {
      await refreshModels(); $('#model').value = j.result.model_id; $('#policy').value = 'auto'; $('#load').click()
    }, 'primary')
  }))
  return a
}

registerTrainSection('tiny_text', { heading: 'Language models · learn from text & reasoning steps', render: tinyTextSection })
