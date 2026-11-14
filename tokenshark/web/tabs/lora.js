import { GET, POST, S, api, btn, el, flow, registerTab, toast } from '../core.js'
import { esc } from '../flow.js'
import { refreshInfo } from '../ui/models.js'
import { applyRunBadges } from '../ui/run.js'
import { renderTab } from '../ui/tabs.js'

// ---------- LoRA & Fine-Tuning Inspector (works across all plugins)
export async function renderLoRA(p) {
  if (!S.info) { p.innerHTML = '<div class="empty">Load a base model first to attach and inspect a LoRA or fine-tuned checkpoint.</div>'; return }
  await refreshInfo()
  if (S.tab !== 'lora') return
  let lora = S.info.lora
  try {
    const res = await GET('/api/lora')
    lora = res.active ? res : null
    S.info.lora = lora
  } catch { }

  const box = el('div', { className: 'kvbox' })
  box.innerHTML = `<h4>Attach a LoRA or Fine-Tuned Checkpoint to <b>${esc(S.info.model_id)}</b></h4>
    <div class="small">Works with any model plugin — no custom code needed! Point to a LoRA <code>.safetensors</code> file, a PEFT adapter folder (<code>adapter_model.safetensors</code>), or a fine-tuned model folder. Tokenshark computes <b>ΔW = scale · (α/r) · B @ A</b> (or <b>W<sub>ft</sub> − W<sub>base</sub></b>) on the fly, highlights the blocks that changed the most on the flow graph above, and lets you scale or negate the adaptation live.</div>`

  const row = el('div', { className: 'row' })
  const pathIn = el('input', { placeholder: 'Path to .safetensors file or adapter / fine-tuned folder…', value: S.loraPath || '', style: 'width:340px' })
  pathIn.oninput = () => { S.loraPath = pathIn.value }
  const scaleIn = el('input', { type: 'number', step: 0.1, value: lora?.scale ?? 1.0, style: 'width:70px', title: 'LoRA scale (1 = normal, 0 = off, 2 = 2× stronger, -1 = negated)' })
  const layersIn = el('input', { value: lora?.layers_spec || S.loraLayers || 'all', style: 'width:90px', title: 'Which blocks get the LoRA: all | 0:12 | 5,10' })
  layersIn.oninput = () => { S.loraLayers = layersIn.value }
  const loadBtn = btn('Attach path', async () => {
    if (!pathIn.value.trim()) return toast('Enter a file or folder path first.', 'error')
    try {
      const r = await POST('/api/lora/load', { path: pathIn.value.trim(), scale: +scaleIn.value, layers: layersIn.value || 'all' })
      S.info.lora = r
      applyRunBadges()
      renderTab()
      toast(`Attached ${r.kind === 'lora' ? 'LoRA' : 'fine-tune'} “${r.name}” (${r.n_adapted_weights} weights adapted, blocks: ${r.layers_spec || 'all'}).`)
    } catch (e) { toast(e.message, 'error') }
  }, 'primary')

  const up = el('input', { type: 'file', accept: '.safetensors', style: 'display:none' })
  up.onchange = async () => {
    const f = up.files[0]; if (!f) return
    try {
      const r = await api('POST', `/api/lora/upload?name=${encodeURIComponent(f.name)}&scale=${+scaleIn.value}`, await f.arrayBuffer(), true)
      S.info.lora = r
      applyRunBadges()
      renderTab()
      toast(`Uploaded & attached “${r.name}” (${r.n_adapted_weights} weights adapted).`)
    } catch (e) { toast(e.message, 'error') }
  }
  row.append(pathIn, el('span', { className: 'small', textContent: 'scale' }), scaleIn, el('span', { className: 'small', textContent: 'blocks' }), layersIn, loadBtn, btn('📂 Upload .safetensors…', () => up.click()), up)
  box.appendChild(row)
  p.appendChild(box)

  if (!lora || !lora.active) {
    p.appendChild(el('div', { className: 'empty', innerHTML: 'No LoRA or fine-tuned overlay is currently attached.<br><span class="small">Train a LoRA or fine-tune a model, then load it above to see which layers and attention/MLP matrices changed the most.</span>' }))
    return
  }

  // Active LoRA controls & summary
  const activeBox = el('div', { className: 'sec' })
  const ctl = el('div', { className: 'row' })
  const sl = el('input', { type: 'range', min: -2, max: 3, step: 0.1, value: lora.scale, style: 'width:200px' })
  const slLbl = el('span', { className: 'small', innerHTML: `<b>Scale: ${Number(lora.scale).toFixed(2)}×</b>` })
  const actLayers = el('input', { value: lora.layers_spec || 'all', style: 'width:90px', title: 'Active blocks: all | 0:12 | 5,10' })
  sl.oninput = () => { slLbl.innerHTML = `<b>Scale: ${(+sl.value).toFixed(2)}×</b>` }
  const updateLora = async () => {
    try {
      const r = await POST('/api/lora/scale', { scale: +sl.value, layers: actLayers.value || 'all' })
      S.info.lora = r
      toast(`LoRA scale=${r.scale}× on blocks ${r.layers_spec || 'all'}.`)
      renderTab()
    } catch (e) { toast(e.message, 'error') }
  }
  sl.onchange = updateLora
  actLayers.onchange = updateLora
  const detachBtn = btn('✕ Detach (restore base model)', async () => {
    await api('DELETE', '/api/lora')
    S.info.lora = null
    applyRunBadges()
    renderTab()
    toast('Detached LoRA / fine-tune.')
  }, 'ghost')
  ctl.append(
    el('span', { className: 'chip', textContent: `${lora.kind.toUpperCase()}: ${lora.name}` }),
    el('span', { className: 'small', textContent: `${lora.adaptation?.method ? `method: ${lora.adaptation.method} · ` : ''}${lora.n_adapted_weights} adapted weights · ${lora.n_added_weights} added${lora.adaptation?.extra_blocks?.length ? ` · new block ${lora.adaptation.extra_blocks[0].layer} after block ${lora.adaptation.extra_blocks[0].after} (starts as ${lora.adaptation.extra_blocks[0].init})` : ''}${lora.adaptation?.adapters?.length ? ` · thin layer (width ${lora.adaptation.width}) after blocks ${lora.adaptation.adapters.join(', ')}` : ''}` }),
    slLbl, sl, el('span', { className: 'small', textContent: 'Blocks:' }), actLayers, detachBtn
  )
  activeBox.appendChild(ctl)
  p.appendChild(activeBox)

  // 1. Per-layer change ranking & bar chart
  if (lora.layers?.length) {
    const lSec = el('div', { className: 'sec' }, '<h4>1 · Which blocks changed the most? (click a row to select the block on the canvas)</h4>')
    const tbl = el('table', { className: 'grid' })
    tbl.innerHTML = '<tr><th>Block / Layer</th><th>Relative change ‖ΔW‖ / ‖W‖</th><th>‖ΔW‖<sub>F</sub></th><th>‖W<sub>base</sub>‖<sub>F</sub></th><th>Adapted modules</th></tr>'
    lora.layers.forEach(l => {
      const tr = el('tr', { style: 'cursor:pointer' })
      const pct = (l.relative_change * 100).toFixed(2)
      tr.innerHTML = `<td><b>Block ${l.layer}</b> (${l.count} tensors)</td>
        <td><span class="pbar"><i style="width:${(l.heat * 100).toFixed(0)}%;background:var(--warn)"></i></span> <b>${pct}%</b></td>
        <td>${l.delta_norm}</td><td>${l.base_norm}</td>
        <td>${l.modules.map(m => `<span class="chip">${esc(m)}</span>`).join(' ')}</td>`
      tr.onclick = () => {
        const node = flow.nodes.find(n => n.data?.layer === l.layer)
        if (node) flow.select({ type: 'node', id: node.id })
      }
      tbl.appendChild(tr)
    })
    lSec.appendChild(tbl)
    p.appendChild(lSec)
  }

  // 2. Added / new parameters (if any)
  if (lora.added_weights?.length) {
    const aSec = el('div', { className: 'sec' }, `<h4>2 · Added or modified-shape parameters (${lora.n_added_weights})</h4>`)
    const tbl = el('table', { className: 'grid' })
    tbl.innerHTML = '<tr><th>Parameter name</th><th>Kind</th><th>Shape</th></tr>'
    lora.added_weights.forEach(w => {
      const tr = el('tr')
      tr.innerHTML = `<td><code>${esc(w.name)}</code></td><td>${esc(w.kind)}</td><td>[${(w.shape || []).join(' × ')}]${w.rank ? ` (rank ${w.rank})` : ''}</td>`
      tbl.appendChild(tr)
    })
    aSec.appendChild(tbl)
    p.appendChild(aSec)
  }

  // 3. Top individual weight matrices by relative change
  if (lora.top_modules?.length) {
    const mSec = el('div', { className: 'sec' }, '<h4>3 · Top adapted weight matrices</h4>')
    const tbl = el('table', { className: 'grid' })
    tbl.innerHTML = '<tr><th>Weight tensor</th><th>Component</th><th>Rank / α</th><th>‖ΔW‖ / ‖W‖</th><th>‖ΔW‖<sub>F</sub></th></tr>'
    lora.top_modules.slice(0, 30).forEach(m => {
      const tr = el('tr')
      tr.innerHTML = `<td><code>${esc(m.name)}</code></td><td>${esc(m.component)}</td>
        <td>${m.rank != null ? `r=${m.rank}, α=${m.alpha}` : 'full ΔW'}</td>
        <td><b>${(m.relative_change * 100).toFixed(3)}%</b></td><td>${m.delta_norm}</td>`
      tbl.appendChild(tr)
    })
    mSec.appendChild(tbl)
    p.appendChild(mSec)
  }
}

registerTab({ id: 'lora', title: 'LoRA / Fine-tune', render: renderLoRA })
