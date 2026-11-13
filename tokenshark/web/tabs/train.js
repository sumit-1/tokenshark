// The Train tab: a frame that shows the sections the loaded model's plugin lists (ui_schema.train).
// Each section is its own module and registers itself with registerTrainSection.
import { GET, S, el, registerTab, schema } from '../core.js'
import { refreshInfo, refreshModels } from '../ui/models.js'
import { renderTab } from '../ui/tabs.js'
import { esc } from '../flow.js'
import { lineChart } from '../ui/labkit.js'
S.tr = { tiny: null, lora: null, form: { name: 'my-tiny-model', corpus: 'mixed', text: '', d_model: 96, n_layers: 3, n_heads: 4, context: 64, vocab: 300, steps: 1500, batch: 32, lr: 0.003, sample_prompt: 'The capital of France is' },
  image: null, iform: { steps: 1200 }, isample: { color: 'red', shape: 'circle', steps: 12, guidance: 2, seed: 0, result: null },
  lform: { name: 'my-lora', layers: 'all', method: 'lora', after: 0, init: 'duplicate', examples: 'What is the capital of France? => Arr, that be Paris, matey!\nWhat colour is the sky? => Arr, the sky be blue, matey!\nWho wrote Hamlet? => Arr, that be Shakespeare, matey!\nWhat is 2 + 2? => Arr, that be 4, matey!', rank: 8, alpha: 16, targets: ['q_proj', 'v_proj'], steps: 60, lr: 0.001, probe: 'What is the capital of Japan?' } }

export let trainTimer = null

// id → { heading, needsModel, render(st, running) → element }
export const TRAIN_SECTIONS = {}
export function registerTrainSection(id, section) { TRAIN_SECTIONS[id] = section }

export async function renderTrain(p) {
  clearTimeout(trainTimer)
  const st = S.tr
  try { [st.tiny, st.lora, st.image] = await Promise.all([GET('/api/train/tiny'), GET('/api/train/lora'), GET('/api/train/image')]) } catch { }
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

export function jobLive(j) {
  const series = [{ name: 'training loss', cls: 'l1', pts: j.losses }]
  if (j.val?.length) series.push({ name: 'loss on held-out text', cls: 'l2', pts: j.val })
  if (j.kind === 'image') {
    return `<div class="small"><span class="chip${j.status === 'error' ? ' bad' : ''}">${esc(j.status)}</span> ${esc(j.message || '')}${j.n_params ? ` · ${j.n_params.toLocaleString()} trainable weights` : ''} · ${j.seconds}s${j.error ? ` · <span class="bad">${esc(j.error)}</span>` : ''}</div>
      ${lineChart(series)}<div class="small" style="margin-top:6px">What it draws for every colour × shape as training goes on (same noise each time):</div>
      <div class="imggrid">${(j.grids || []).map(g => `<figure><img src="${g.url}" width="200"><figcaption>step ${g.step}</figcaption></figure>`).join('')}</div>`
  }
  const samples = (j.samples || []).map(s => `<div class="sample"><span class="small">${s.label ? esc(s.label) : 'step ' + s.step}</span> ${esc(s.text)}</div>`).join('')
  return `<div class="small"><span class="chip${j.status === 'error' ? ' bad' : ''}">${esc(j.status)}</span> ${esc(j.message || '')}${j.n_params ? ` · ${j.n_params.toLocaleString()} trainable weights` : ''} · ${j.seconds}s${j.error ? ` · <span class="bad">${esc(j.error)}</span>` : ''}</div>
    ${lineChart(series)}${samples ? `<div class="small" style="margin-top:6px">${j.kind === 'tiny' ? 'What it writes after the test prompt, as training goes on:' : 'Answer to the test prompt:'}</div>${samples}` : ''}`
}

export function jobView(j, action) {
  const box = el('div', { className: 'job sec' }); box.dataset.kind = j.kind
  box.appendChild(el('div', { className: 'joblive' }, jobLive(j)))
  const b = ['done', 'stopped'].includes(j.status) ? action(j) : null
  if (b) { const r = el('div', { className: 'row' }); r.appendChild(b); box.appendChild(r) }
  return box
}

registerTab({ id: 'train', title: 'Train', hint: 'Train small models from scratch, or adapt the loaded one', render: renderTrain })
