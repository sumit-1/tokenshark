// Agent-Authored & Shareable Reproducible Playbook Studio tab (unlocked by `playbook_studio` Playbook extension).
import { GET, POST, S, btn, download, el, flow, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { validate } from '../ui/flowedit.js'
import { labelled, selectOf } from '../ui/labkit.js'
import { renderInspector } from '../ui/inspector.js'
import { renderTab } from '../ui/tabs.js'

S.pbs = {
  list: [],
  activeId: 'pb_refusal_neuron_hunt',
  activePb: null,
  stepResults: {},              // { step_id: { resolved_params, exports, result } }
  editingJson: false,
  jsonDraft: '',
}

export async function renderPlaybooks(p) {
  const st = S.pbs
  try {
    st.list = await GET('/api/ext/playbook_studio/list')
  } catch {}
  if (!st.activePb || st.activePb.id !== st.activeId) {
    st.activePb = (st.list || []).find(x => x.id === st.activeId) || st.list?.[0] || null
    if (st.activePb) st.activeId = st.activePb.id
  }
  p.innerHTML = ''

  const topBar = el('div', { className: 'row', style: 'justify-content:space-between;margin-bottom:8px' })
  const left = el('div', { className: 'row' })
  const pbOpts = (st.list || []).map(x => [x.id, `${x.name} (${(x.steps || []).length} steps)`])
  const pbSel = selectOf(pbOpts, st.activeId)
  pbSel.onchange = () => {
    st.activeId = pbSel.value
    st.activePb = st.list.find(x => x.id === st.activeId) || null
    st.stepResults = {}
    st.editingJson = false
    renderTab()
  }

  const runAllBtn = btn('▶▶ Run All Steps on Loaded Model', async () => {
    if (!st.activePb) return
    statusBox.innerHTML = `<div class="empty">${spinner(`executing all ${st.activePb.steps.length} steps on ${S.info?.model_id || 'model'}…`)}</div>`
    try {
      const r = await POST('/api/ext/playbook_studio/run', {
        playbook: st.activePb,
        reset_ctx: true,
        update_workspace: true,
      })
      for (const sr of r.step_results || []) {
        st.stepResults[sr.step_id] = sr
        if (sr.result?.flow) {
          flow.setGraph(sr.result.flow)
          validate(false)
          renderInspector()
        }
      }
      toast(`Completed all ${(r.step_results || []).length} steps of "${st.activePb.name}".`)
      renderTab()
    } catch (e) { toast(e.message, 'error'); statusBox.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
  }, 'primary')

  const exportBtn = btn('Export / Share (.json)', () => {
    if (!st.activePb) return
    download(`${st.activePb.id || 'playbook'}.playbook.json`, JSON.stringify(st.activePb, null, 2))
  }, 'mini')

  const fileInp = el('input', { type: 'file', accept: '.json', hidden: true })
  fileInp.onchange = async ev => {
    const f = ev.target.files[0]; if (!f) return
    try {
      const obj = JSON.parse(await f.text())
      const saved = await POST('/api/ext/playbook_studio/save', { playbook: obj })
      st.activeId = saved.id
      st.activePb = saved
      st.stepResults = {}
      toast(`Imported playbook "${saved.name}".`)
      renderTab()
    } catch (e) { toast('Invalid playbook JSON: ' + e.message, 'error') }
    ev.target.value = ''
  }
  const importBtn = btn('Import (.json)…', () => fileInp.click(), 'mini')

  const editBtn = btn(st.editingJson ? 'Close Playbook Editor' : '✎ Edit / Create Playbook JSON', () => {
    st.editingJson = !st.editingJson
    if (st.editingJson && st.activePb) st.jsonDraft = JSON.stringify(st.activePb, null, 2)
    renderTab()
  }, 'mini ghost')

  left.append(labelled('Playbook', pbSel), runAllBtn, exportBtn, importBtn, fileInp, editBtn)
  topBar.append(
    left,
    el('span', { className: 'small' }, `<span class="chip" style="background:rgba(59,130,246,0.2)">Playbook Studio</span> Discovery outputs (<code>$step.field</code>) pipe automatically into later steps across any model.`),
  )
  p.appendChild(topBar)

  const statusBox = el('div')
  p.appendChild(statusBox)

  if (!st.activePb) {
    p.appendChild(el('div', { className: 'empty', textContent: 'No playbook selected.' }))
    return
  }

  if (st.editingJson) {
    const edBox = el('div', { className: 'kvbox', style: 'margin-bottom:10px' })
    edBox.appendChild(el('div', { className: 'small', style: 'margin-bottom:4px' },
      `Author or customize a playbook. Use <code>$step_id.field</code> (e.g. <code>$hunt.top_block</code>, <code>$hunt.top_neurons</code>, <code>$extract.vector_id</code>, <code>$sinks.top_keys</code>) in later steps so the playbook adapts automatically when run on a different model.`))
    const ta = el('textarea', { rows: 14, value: st.jsonDraft, style: 'width:100%;font-family:monospace;font-size:11px' })
    ta.oninput = () => { st.jsonDraft = ta.value }
    const saveRow = el('div', { className: 'row', style: 'margin-top:6px' })
    saveRow.append(
      btn('Save Playbook', async () => {
        try {
          const parsed = JSON.parse(st.jsonDraft)
          const saved = await POST('/api/ext/playbook_studio/save', { playbook: parsed })
          st.activeId = saved.id
          st.activePb = saved
          st.editingJson = false
          st.stepResults = {}
          toast(`Saved playbook "${saved.name}".`)
          renderTab()
        } catch (e) { toast(e.message, 'error') }
      }, 'primary mini'),
      btn('Cancel', () => { st.editingJson = false; renderTab() }, 'mini ghost'),
    )
    edBox.append(ta, saveRow)
    p.appendChild(edBox)
  }

  const descBox = el('div', { className: 'kvbox', style: 'margin-bottom:10px' })
  descBox.innerHTML = `<div class="row" style="justify-content:space-between">
    <div><b>${esc(st.activePb.name)}</b> <span class="chip">author: ${esc(st.activePb.author || 'agent')}</span> <span class="chip">id: ${esc(st.activePb.id)}</span></div>
    <div class="small">Loaded model: <b>${esc(S.info?.model_id || 'none')}</b> (${S.info?.n_layers || '?'} blocks)</div>
  </div>
  <div class="small" style="margin-top:4px">${esc(st.activePb.description || '')}</div>`
  p.appendChild(descBox)

  // Render each step card in sequence
  for (let idx = 0; idx < (st.activePb.steps || []).length; idx++) {
    const step = st.activePb.steps[idx]
    const sr = st.stepResults[step.id]
    const card = el('div', { className: 'kvbox', style: 'margin-bottom:10px' })

    const head = el('div', { className: 'row', style: 'justify-content:space-between' })
    const badge = sr
      ? `<span class="chip ok">✓ Completed on ${esc(S.info?.model_id || 'model')}</span>`
      : `<span class="chip">Pending</span>`
    head.append(
      el('div', {}, `<b>${esc(step.title || `Step ${idx + 1}`)}</b> <span class="chip">id: <code>${esc(step.id)}</code></span> <span class="chip">kind: <code>${esc(step.kind)}</code></span> ${badge}`),
      btn(`▶ Run Step ${idx + 1}`, async () => {
        stepOut.innerHTML = `<div class="empty">${spinner(`running ${step.id} (${step.kind})…`)}</div>`
        try {
          const r = await POST('/api/ext/playbook_studio/run', {
            playbook: st.activePb,
            step_id: step.id,
            reset_ctx: false,
            update_workspace: true,
          })
          for (const item of r.step_results || []) {
            st.stepResults[item.step_id] = item
            if (item.result?.flow) {
              flow.setGraph(item.result.flow)
              validate(false)
              renderInspector()
            }
          }
          renderTab()
        } catch (e) { toast(e.message, 'error'); stepOut.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
      }, sr ? 'mini' : 'primary mini'),
    )
    card.appendChild(head)

    if (step.explanation) {
      card.appendChild(el('div', { className: 'small', style: 'margin-top:4px' }, esc(step.explanation)))
    }

    // Show raw vs resolved parameters so the user can see $step.field bindings clearly
    const paramStr = JSON.stringify(sr ? sr.resolved_params : (step.params || {}))
    const rawStr = JSON.stringify(step.params || {})
    const hasBindings = rawStr.includes('$')
    if (hasBindings) {
      card.appendChild(el('div', { className: 'small', style: 'margin-top:4px;font-family:monospace;font-size:11px' },
        `Bindings: <code>${esc(rawStr)}</code>` + (sr ? ` → Resolved on model: <code>${esc(paramStr)}</code>` : '')))
    }

    const stepOut = el('div', { style: 'margin-top:6px' })
    if (sr) {
      renderStepResult(stepOut, sr)
    }
    card.appendChild(stepOut)
    p.appendChild(card)
  }
}

function renderStepResult(box, sr) {
  const expEntries = Object.entries(sr.exports || {})
  if (expEntries.length) {
    const expRow = el('div', { className: 'row', style: 'flex-wrap:wrap;gap:6px;margin-bottom:6px' })
    expRow.appendChild(el('span', { className: 'small', innerHTML: '<b>Exported to next steps:</b>' }))
    for (const [k, v] of expEntries) {
      expRow.appendChild(el('span', { className: 'chip', style: 'font-family:monospace' },
        `$${esc(sr.step_id)}.${esc(k)} = ${esc(typeof v === 'object' ? JSON.stringify(v) : String(v))}`))
    }
    box.appendChild(expRow)
  }

  const r = sr.result || {}
  if (sr.kind === 'apply_flow') {
    const tbl = el('table', { className: 'grid' })
    tbl.innerHTML = `<thead><tr><th>Prompt</th><th>Baseline Output</th><th>Intervened Output (Wired on Canvas)</th></tr></thead>
      <tbody><tr><td>${esc(r.prompt || '')}</td><td>${esc(r.baseline_text || '')}</td><td><b>${esc(r.intervened_text || '')}</b></td></tr></tbody>`
    box.appendChild(tbl)
    return
  }

  if (sr.kind === 'verify_eval' && r.rows) {
    const tbl = el('table', { className: 'grid' })
    tbl.innerHTML = `<thead><tr><th>Group</th><th>Prompt</th><th>Baseline</th><th>Intervened</th><th>Changed</th></tr></thead><tbody></tbody>`
    const tb = tbl.querySelector('tbody')
    for (const row of r.rows) {
      const tr = el('tr')
      tr.innerHTML = `<td><span class="chip">${esc(row.group)}</span></td>
        <td>${esc(row.prompt)}</td>
        <td>${esc(row.baseline)}</td>
        <td><b>${esc(row.intervened)}</b></td>
        <td><span class="chip ${row.changed ? 'ok' : ''}">${row.changed ? 'SHIFTED' : 'SAME'}</span></td>`
      tb.appendChild(tr)
    }
    box.appendChild(tbl)
    return
  }

  if (sr.kind === 'verify_eval' && r.results) {
    const tbl = el('table', { className: 'grid' })
    tbl.innerHTML = `<thead><tr><th>Condition</th><th>Pos Margin</th><th>Neg Margin</th><th>ROC AUC</th><th>Capability Acc</th><th>Neutral NLL</th></tr></thead><tbody></tbody>`
    const tb = tbl.querySelector('tbody')
    for (const row of r.results) {
      const tr = el('tr')
      tr.innerHTML = `<td><b>${esc(row.condition)}</b></td><td>${row.pos_margin}</td><td>${row.neg_margin}</td><td><b>${row.auc}</b></td><td>${(row.cap_acc * 100).toFixed(0)}%</td><td>${row.nll}</td>`
      tb.appendChild(tr)
    }
    box.appendChild(tbl)
    return
  }
}

registerTab({ id: 'playbooks', title: 'Playbooks', render: renderPlaybooks })
