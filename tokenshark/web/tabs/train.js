// The Train tab: a frame that shows the sections the loaded model's plugin lists (ui_schema.train).
// Each section is its own module and registers itself with registerTrainSection.
import { GET, S, btn, el, registerTab, schema } from '../core.js'
import { refreshInfo, refreshModels } from '../ui/models.js'
import { renderTab } from '../ui/tabs.js'
import { esc } from '../flow.js'
import { lineChart } from '../ui/labkit.js'

S.tr = {
  tiny: null,
  lora: null,
  form: {
    name: 'my-tiny-model',
    corpus: 'mixed',
    text: '',
    n_questions: 1000,
    system_prompt: '',
    d_model: 96,
    n_layers: 3,
    n_heads: 4,
    context: 64,
    vocab: 300,
    steps: 1500,
    batch: 32,
    lr: 0.003,
    sample_prompt: 'The capital of France is',
    n_experts: 0,
    experts_per_token: 2,
    pv_offset: 0,
    pv_limit: 50,
  },
  image: null,
  iform: { steps: 1200 },
  isample: { color: 'red', shape: 'circle', steps: 12, guidance: 2, seed: 0, result: null },
  lform: {
    name: 'my-lora',
    layers: 'all',
    method: 'lora',
    after: 0,
    init: 'duplicate',
    dataset_id: '',
    n_questions: 100,
    system_prompt: 'You are a helpful pirate assistant.',
    examples: 'What is the capital of France? => Arr, that be Paris, matey!\nWhat colour is the sky? => Arr, the sky be blue, matey!\nWho wrote Hamlet? => Arr, that be Shakespeare, matey!\nWhat is 2 + 2? => Arr, that be 4, matey!',
    rank: 8,
    alpha: 16,
    targets: ['q_proj', 'v_proj'],
    steps: 60,
    lr: 0.001,
    probe: 'What is the capital of Japan?',
    pv_offset: 0,
    pv_limit: 50,
    edit_raw: false,
  },
}

export let trainTimer = null

// id -> { heading, needsModel, render(st, running) -> element }
export const TRAIN_SECTIONS = {}
export function registerTrainSection(id, section) { TRAIN_SECTIONS[id] = section }

/**
 * Split a single training line or block into `{question, answer}` for the 2-column Dataset Viewer table.
 */
export function splitQuestionAnswer(line) {
  const s = (line || '').trim()
  if (!s) return { question: '', answer: '' }
  if (s.includes(' => ')) {
    const idx = s.indexOf(' => ')
    return { question: s.slice(0, idx).trim(), answer: s.slice(idx + 4).trim() }
  }
  if (s.includes('\nStep 1:')) {
    const idx = s.indexOf('\nStep 1:')
    return { question: s.slice(0, idx).replace(/^Question:\s*/i, '').trim(), answer: s.slice(idx + 1).trim() }
  }
  if (s.includes('\nAnswer:')) {
    const idx = s.indexOf('\nAnswer:')
    return { question: s.slice(0, idx).replace(/^Question:\s*/i, '').trim(), answer: s.slice(idx + 1).trim() }
  }
  if (s.includes('A:')) {
    const idx = s.indexOf('A:')
    return { question: s.slice(0, idx).replace(/^Q:\s*/i, '').trim(), answer: s.slice(idx + 2).trim() }
  }
  return { question: s, answer: '' }
}

/**
 * Renders a HuggingFace-style 2-column Dataset Preview Card (`Question / User Prompt` | `Expected Answer / CoT Steps`)
 * with dataset row count badge, active training subset indicator, and pagination.
 */
export function renderDatasetPreviewTable({
  title = 'Dataset Viewer',
  rows = [],
  totalCount = 0,
  trainCount = 0,
  offset = 0,
  limit = 50,
  systemPrompt = '',
  onPageChange = null,
}) {
  const wrap = el('div', {
    className: 'sec',
    style: 'margin:8px 0;padding:8px 10px;border:1px solid var(--line, #2a303c);border-radius:6px;background:rgba(16,20,26,0.55)',
  })

  const endRow = Math.min(offset + rows.length, totalCount || rows.length)
  const startRow = rows.length ? offset + 1 : 0

  const hdr = el('div', { className: 'row', style: 'justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px' })
  const leftInfo = el('div', { className: 'row', style: 'gap:6px;align-items:center' })
  leftInfo.append(
    el('b', { style: 'font-size:12px', textContent: title }),
    el('span', { className: 'chip', textContent: `${(totalCount || rows.length).toLocaleString()} total rows` }),
  )
  if (trainCount && trainCount > 0) {
    leftInfo.appendChild(el('span', {
      className: 'chip',
      style: 'background:rgba(62,207,142,0.16);color:#3ecf8e;border-color:rgba(62,207,142,0.4)',
      textContent: `Training on ${Math.min(trainCount, totalCount || trainCount).toLocaleString()} questions`,
    }))
  }
  hdr.appendChild(leftInfo)

  if (onPageChange && totalCount > limit) {
    const pager = el('div', { className: 'row', style: 'gap:4px;align-items:center' })
    pager.appendChild(el('span', { className: 'small', textContent: `Showing ${startRow.toLocaleString()} - ${endRow.toLocaleString()} of ${totalCount.toLocaleString()}` }))
    const prevBtn = btn('◂ Prev', () => onPageChange(Math.max(0, offset - limit)), 'mini')
    prevBtn.disabled = offset <= 0
    const nextBtn = btn('Next ▸', () => onPageChange(offset + limit), 'mini')
    nextBtn.disabled = offset + limit >= totalCount
    pager.append(prevBtn, nextBtn)
    hdr.appendChild(pager)
  } else {
    hdr.appendChild(el('span', { className: 'small', textContent: `Showing ${startRow} - ${endRow} of ${(totalCount || rows.length).toLocaleString()}` }))
  }
  wrap.appendChild(hdr)

  if (systemPrompt && systemPrompt.trim()) {
    wrap.appendChild(el('div', {
      className: 'small',
      style: 'margin-bottom:6px;padding:4px 8px;border-left:3px solid #5b9dff;background:rgba(91,157,255,0.08);border-radius:3px',
    }, `<b>System Prompt:</b> ${esc(systemPrompt.trim())}`))
  }

  const scroller = el('div', { style: 'max-height:240px;overflow:auto;border:1px solid var(--line, #252b36);border-radius:4px' })
  const tbl = el('table', { className: ' dstable', style: 'width:100%;border-collapse:collapse;font-size:12px' })
  tbl.innerHTML = `
    <thead style="position:sticky;top:0;background:#161a22;z-index:1;border-bottom:1px solid #2a303c">
      <tr>
        <th style="width:42px;text-align:right;padding:5px 8px;color:#8b95a5">#</th>
        <th style="width:46%;text-align:left;padding:5px 8px;color:#9ec5ff">User Prompt / Question</th>
        <th style="width:54%;text-align:left;padding:5px 8px;color:#7ee2b8">Expected Answer / Reasoning Steps</th>
      </tr>
    </thead>
    <tbody>
      ${rows.length ? rows.map((r, i) => `
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);vertical-align:top">
          <td style="text-align:right;padding:5px 8px;color:#6e7785;font-family:var(--mono,monospace)">${offset + i + 1}</td>
          <td style="padding:5px 8px;white-space:pre-wrap;word-break:break-word">${esc(r.question || '')}</td>
          <td style="padding:5px 8px;white-space:pre-wrap;word-break:break-word;font-family:var(--mono,monospace);color:#d5e2ec">${esc(r.answer || '')}</td>
        </tr>
      `).join('') : `<tr><td colspan="3" style="padding:10px;text-align:center;color:#7a828e">No examples to display.</td></tr>`}
    </tbody>
  `
  scroller.appendChild(tbl)
  wrap.appendChild(scroller)
  return wrap
}

export async function renderTrain(p) {
  clearTimeout(trainTimer)
  const st = S.tr
  try { [st.tiny, st.lora, st.image, st.datasets] = await Promise.all([GET('/api/train/tiny'), GET('/api/train/lora'), GET('/api/train/image'), GET('/api/datasets').catch(() => [])]) } catch { }
  if (S.tab !== 'train') return
  p.innerHTML = ''
  const running = j => j && ['queued', 'running'].includes(j.status)
  // With a model loaded its plugin chooses the sections; with none, every trainer that needs no model is offered.
  const ids = S.info ? (schema().train || []) : Object.keys(TRAIN_SECTIONS).filter(id => !TRAIN_SECTIONS[id].needsModel)
  for (const id of ids) {
    const sec = TRAIN_SECTIONS[id]
    if (!sec) continue
    if (!S.info && sec.heading) p.appendChild(el('div', { className: 'sechead', textContent: sec.heading }))
    p.appendChild(sec.render(st, running))
  }
  if (!ids.length) p.appendChild(el('div', { className: 'empty', textContent: 'There is nothing to train for this kind of model yet.' }))
  if (running(st.tiny) || running(st.lora) || running(st.image)) trainTimer = setTimeout(() => { if (S.tab === 'train') updateJobs(p) }, 700)
}

// While a job runs only the live parts are redrawn, so the forms keep focus.
export async function updateJobs(p) {
  const st = S.tr
  const was = [st.tiny?.status, st.lora?.status, st.image?.status]
  try { [st.tiny, st.lora, st.image] = await Promise.all([GET('/api/train/tiny'), GET('/api/train/lora'), GET('/api/train/image')]) } catch { }
  if (S.tab !== 'train') return
  const live = j => j && ['queued', 'running'].includes(j.status)
  if ([st.tiny?.status, st.lora?.status, st.image?.status].some((s, i) => s !== was[i])) {
    if (st.lora?.status === 'done' && was[1] !== 'done' && st.lora.result?.lora) await refreshInfo()
    if (st.tiny?.status === 'done' && was[0] !== 'done') refreshModels()
    return renderTab()
  }
  p.querySelectorAll('.job').forEach(n => { const j = st[n.dataset.kind]; if (j) n.querySelector('.joblive').innerHTML = jobLive(j) })
  if (live(st.tiny) || live(st.lora) || live(st.image)) trainTimer = setTimeout(() => updateJobs(p), 700)
}

// A mixture-of-experts model in training: how evenly each block's router is spreading tokens over its experts.
function usageRows(j) {
  if (!j.expert_usage?.length) return ''
  return `<div class="small" style="margin-top:6px">Share of tokens each expert gets, per block (even = every expert is learning; one bright cell = the router collapsed onto it):</div>
    <div class="xheat">${j.expert_usage.map((u, l) => { const mx = Math.max(...u, 1e-9); return `<div class="xrow"><span class="xlab">block ${l}</span><span class="xcells">${u.map((v, i) => `<i title="expert ${i}: ${(v * 100).toFixed(1)}%" style="background:hsla(${(i * 47) % 360},70%,${16 + 44 * v / mx}%,${0.3 + 0.7 * v / mx})"></i>`).join('')}</span></div>` }).join('')}</div>`
}

export function jobLive(j) {
  const series = [{ name: 'training loss', cls: 'l1', pts: j.losses }]
  if (j.val?.length) series.push({ name: 'loss on held-out text', cls: 'l2', pts: j.val })
  if (j.kind === 'image') {
    return `<div class="small"><span class="chip${j.status === 'error' ? ' bad' : ''}">${esc(j.status)}</span> ${esc(j.message || '')}${j.n_params ? ` · ${j.n_params.toLocaleString()} trainable weights` : ''} · ${j.seconds}s${j.error ? ` · <span class="bad">${esc(j.error)}</span>` : ''}</div>
      ${lineChart(series)}<div class="small" style="margin-top:6px">What it draws for every colour × shape as training goes on (same noise each time):</div>
      <div class="imggrid">${(j.grids || []).map(g => `<figure><img src="${g.url}" width="200"><figcaption>step ${g.step}</figcaption></figure>`).join('')}</div>`
  }
  const samples = (j.samples || []).map(s => `<div class="sample"><span class="small">${s.label ? esc(s.label) : 'step ' + s.step}</span> ${esc(s.text)}</div>`).join('')
  return `<div class="small"><span class="chip${j.status === 'error' ? ' bad' : ''}">${esc(j.status)}</span> ${esc(j.message || '')}${j.n_params ? ` · ${j.n_params.toLocaleString()} trainable weights` : ''}${j.n_questions ? ` · ${j.n_questions.toLocaleString()} questions` : ''}${j.n_examples ? ` · ${j.n_examples.toLocaleString()} examples` : ''} · ${j.seconds}s${j.error ? ` · <span class="bad">${esc(j.error)}</span>` : ''}</div>
    ${lineChart(series)}${usageRows(j)}${samples ? `<div class="small" style="margin-top:6px">${j.kind === 'tiny' ? 'What it writes after the test prompt, as training goes on:' : 'Answer to the test prompt:'}</div>${samples}` : ''}`
}

export function jobView(j, action) {
  const box = el('div', { className: 'job sec' }); box.dataset.kind = j.kind
  box.appendChild(el('div', { className: 'joblive' }, jobLive(j)))
  const b = ['done', 'stopped'].includes(j.status) ? action(j) : null
  if (b) { const r = el('div', { className: 'row' }); r.appendChild(b); box.appendChild(r) }
  return box
}

registerTab({ id: 'train', title: 'Train', hint: 'Train small models from scratch, or adapt the loaded one', render: renderTrain })
