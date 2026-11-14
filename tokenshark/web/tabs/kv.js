import { GET, POST, S, api, btn, el, flow, newId, on, registerTab, toast } from '../core.js'
import { esc } from '../flow.js'
import { addKV, selectedNode } from '../ui/flowedit.js'
import { renderInspector } from '../ui/inspector.js'
import { labelled, numInput, selectOf } from '../ui/labkit.js'
import { startRun } from '../ui/run.js'
import { renderTab, setTab, tabAvailable } from '../ui/tabs.js'

// ---------- KV cache library
export async function refreshCaches() { try { S.caches = await GET('/api/caches') } catch { S.caches = [] } if (S.tab === 'kv') renderTab(); if (selectedNode()?.data.kind === 'kv_import') renderInspector() }

export async function saveRunCache() {
  if (!S.run) return
  const name = prompt('Name for this KV cache:', S.run.label || `run ${S.run.id}`)
  if (name === null) return
  try { const c = await POST(`/api/runs/${S.run.id}/cache`, { name }); toast(`Saved KV cache “${c.name}” (${c.n_tokens} tokens).`); await refreshCaches(); setTab('kv') }
  catch (e) { toast(e.message, 'error') }
}

export async function renderKV(p) {
  const head = el('div', { className: 'row' })
  const up = el('input', { type: 'file', accept: '.safetensors' })
  up.onchange = async () => {
    const f = up.files[0]; if (!f) return
    try { await api('POST', `/api/caches/upload?name=${encodeURIComponent(f.name.replace(/\.safetensors$/, ''))}`, await f.arrayBuffer(), true); toast('Imported'); refreshCaches() }
    catch (e) { toast(e.message, 'error') }
  }
  head.append(btn('Save current run’s cache', saveRunCache), el('span', { className: 'small', textContent: 'Import a .safetensors KV file:' }), up)
  p.appendChild(prefillEditor())
  p.appendChild(el('div', { className: 'sechead', textContent: 'Saved KV caches (files)' }))
  p.appendChild(el('div', { className: 'small', textContent: 'A KV cache can also be kept as a file — saved from a finished run, built from text below, or imported — and prepended with a “KV-cache prefix” box. Same block choice and position modes as above; use this when you want to reuse or edit one cache across flows.' }))
  // build a cache file from text
  const hp = el('details', { className: 'kvbox' }, `<summary>Build a cache file from text</summary><div class="small">The text is run through the current flow’s blocks once and its keys and values are saved as a cache file, which is then added to the flow as a KV-cache prefix box (choose its blocks and position there).</div>`)
  const ta = el('textarea', { rows: 3, placeholder: 'You are a pirate. Always answer in pirate speak.', value: S.hpText || '' })
  ta.oninput = () => { S.hpText = ta.value }
  const nm = el('input', { placeholder: 'name', value: S.hpName || '', style: 'width:180px' })
  nm.oninput = () => { S.hpName = nm.value }
  const asSys = el('input', { type: 'checkbox', checked: S.hpSys !== false })
  asSys.onchange = () => { S.hpSys = asSys.checked }
  const row = el('div', { className: 'row' })
  row.append(nm, el('label', { className: 'small' }, ''), btn('Build KV cache', async () => {
    if (!ta.value.trim()) return toast('Write the hidden prompt first.', 'error')
    try {
      const c = await POST('/api/caches/from_text', { flow: flow.getGraph(), text: ta.value, name: nm.value, as_system: asSys.checked })
      toast(`Built “${c.name}” (${c.n_tokens} tokens). Adding it to the flow.`)
      await refreshCaches(); addKV(c.id)
      const kv = flow.nodes.find(n => n.data.kind === 'kv_import'); if (kv) flow.updateNodeData(kv.id, { mode: 'shift' })
    } catch (e) { toast(e.message, 'error') }
  }))
  const lab = el('label', { className: 'field check' }); lab.append(asSys, el('span', { textContent: 'as the system turn of the chat template (then the user turn continues after it)' }))
  hp.append(ta, lab, row)
  p.appendChild(head)
  p.appendChild(hp)
  const t = el('table', { className: 'grid' })
  t.innerHTML = '<tr><th>Name</th><th>Model</th><th>Tokens</th><th>Slots</th><th>Size</th><th></th></tr>'
  for (const c of S.caches) {
    const tr = el('tr')
    const same = c.fingerprint?.hidden_size === S.info.d_model && c.fingerprint?.n_kv_heads === S.info.n_kv_heads
    tr.innerHTML = `<td>${esc(c.name)}</td><td>${esc(c.model_id || c.fingerprint?.model_id || '?')}${same ? '' : ' <span class="bad">≠ loaded</span>'}</td><td>${c.n_tokens}</td><td>${c.n_slots}</td><td>${((c.bytes || 0) / 2 ** 20).toFixed(1)} MB</td>`
    const td = el('td')
    td.append(btn('Inspect / edit', () => inspectCache(c.id), 'mini'), btn('Prepend in flow', () => addKV(c.id), 'mini'),
      btn('Download', () => window.open(`/api/caches/${c.id}/download`), 'mini'),
      btn('Delete', async () => { if (confirm(`Delete ${c.name}?`)) { await api('DELETE', `/api/caches/${c.id}`); refreshCaches() } }, 'mini'))
    tr.appendChild(td); t.appendChild(tr)
  }
  p.appendChild(t)
  if (S.kvInspect) p.appendChild(S.kvInspect)
}

export async function inspectCache(cid) {
  const c = await GET(`/api/caches/${cid}`)
  const box = el('div', { className: 'kvbox' }, `<h4>${esc(c.name)} — ${c.n_tokens} tokens · ${c.n_slots} block slots · positions are RoPE positions (editable)</h4>`)
  const toks = el('div', { className: 'tokens' })
  const segOf = i => c.segments.find(s => i >= s.start && i < s.end)?.origin || 'prompt'
  let a = null, b = null
  c.tokens.forEach((t, i) => {
    const s = el('span', { className: `tok seg-${segOf(i)}`, title: `#${i} · pos ${c.positions[i]}` }, `${esc(t)}<sub>${c.positions[i]}</sub>`)
    s.onclick = () => { if (a === null || b !== null) { a = i; b = null } else { b = i } mark() }
    toks.appendChild(s)
  })
  const mark = () => { toks.querySelectorAll('.tok').forEach((s, i) => s.classList.toggle('on', a !== null && i >= a && i <= (b ?? a))); range.textContent = a === null ? 'Click two tokens to select a span (or leave unselected for all tokens).' : `span ${a}–${b ?? a}` }
  const range = el('span', { className: 'small' })
  const delta = el('input', { type: 'number', value: -10, style: 'width:70px' })
  const ops = el('div', { className: 'row' })
  const doEdit = async (op, extra = {}) => {
    if (a === null && ['keep', 'drop', 'truncate', 'shift'].includes(op)) return toast('Select a span first.', 'error')
    const s = a ?? 0, e = a === null ? c.n_tokens : (b ?? a) + 1
    try { const r = await POST(`/api/caches/${cid}/edit`, { op, start: s, end: e, delta: +delta.value, ...extra }); toast(`Saved new cache “${r.name}”`); await refreshCaches(); inspectCache(r.id) }
    catch (err) { toast(err.message, 'error') }
  }
  ops.append(range, btn('Keep span', () => doEdit('keep'), 'mini'), btn('Drop span', () => doEdit('drop'), 'mini'),
    btn('Truncate before span', () => doEdit('truncate'), 'mini'), el('span', { textContent: 'Δ pos' }), delta, btn('Shift span positions', () => doEdit('shift'), 'mini'))

  // Per-block KV cache surgery (zero / scale / quantize K or V at chosen block slots)
  const slotRow = el('div', { className: 'row' })
  const layIn = el('input', { value: 'all', style: 'width:90px', title: `Block slots (0…${c.n_slots - 1}): all | 0:12 | 5,10` })
  const partSel = selectOf([['both', 'K & V'], ['k', 'Keys (K) only'], ['v', 'Values (V) only']], 'both')
  const valIn = el('input', { type: 'number', value: 4, step: 'any', style: 'width:65px', title: 'Scale factor or bit-width (1..16)' })
  slotRow.append(
    el('b', { className: 'small', textContent: 'Per-block KV edit:' }),
    labelled('Blocks', layIn),
    labelled('Target', partSel),
    labelled('Value / bits', valIn),
    btn('Zero blocks', () => doEdit('zero_slots', { layers: layIn.value, part: partSel.value, delta: 0 }), 'mini'),
    btn('Scale blocks', () => doEdit('scale_slots', { layers: layIn.value, part: partSel.value, delta: +valIn.value }), 'mini'),
    btn('Quantize blocks (bits)', () => doEdit('quantize_slots', { layers: layIn.value, part: partSel.value, delta: +valIn.value }), 'mini'),
  )
  box.append(ops, slotRow, toks)
  mark()
  S.kvInspect = box
  setTab('kv')
}

export function addLayerPrompt(layers = 'all') {
  const embed = flow.nodes.find(n => n.data.kind === 'embed')
  if (!embed) return
  const existing = flow.nodes.filter(x => x.data.kind === 'kv_import' || x.data.kind === 'kv_text')
  const n = flow.addNode({ id: newId('kvt'), type: 'box', position: { x: embed.position.x - 40 + 190 * existing.length, y: embed.position.y - 130 },
    data: { kind: 'kv_text', label: `Layer KV prefill ${existing.filter(x => x.data.kind === 'kv_text').length + 1}`, text: 'Always answer in French.', layers: String(layers), mode: 'shift', offset: 0, as_system: true, enabled: true } })
  flow.addEdge({ source: n.id, sourceHandle: 'out', target: embed.id, targetHandle: 'cache' })
  setTimeout(() => flow.fit(), 30)
  flow.select({ type: 'node', id: n.id })
}

// Which schedule positions an index spec selects: all | none | 14 | 10:14 | 3,5 | -1
export function specBlocks(spec, n) {
  spec = String(spec || 'all').trim().toLowerCase()
  if (['all', '*', ''].includes(spec)) return [...Array(n).keys()]
  if (spec === 'none') return []
  if (spec === 'last') return n ? [n - 1] : []
  const out = new Set()
  for (const part of spec.split(',').map(x => x.trim()).filter(Boolean)) {
    const m = part.match(/^(-?\d*)\s*:\s*(-?\d*)$/)
    if (m) {
      let a = m[1] === '' ? 0 : +m[1], b = m[2] === '' ? n : +m[2]
      if (a < 0) a += n; if (b < 0) b += n
      for (let i = Math.max(0, a); i < Math.min(n, b); i++) out.add(i)
    } else if (/^-?\d+$/.test(part)) { const i = +part; if (i >= -n && i < n) out.add((i + n) % n) }
    else return null
  }
  return [...out].sort((x, y) => x - y)
}

export const PF_COLORS = ['#5b9dff', '#3ecf8e', '#ff9f43', '#a78bfa', '#ff6b6b', '#4fd1c5', '#f5b84a', '#e879f9']

export const PF_MODES = [['shift', 'shift — prefill at 0…m−1, prompt continues after it'], ['negative', 'negative — prefill at −m…−1, prompt starts at 0'],
  ['overwrite', 'overwrite — these blocks read the prefill instead of the prompt'], ['offset', 'offset — prefill starts at the position below']]

export function prefillEditor() {
  const box = el('div')
  const n = S.planNodes?.length || S.info.n_layers
  box.appendChild(el('div', { className: 'small' }, `<b>Per-layer KV prefill.</b> Type text for the blocks you choose. The text is run through the whole model once; then, in <i>those blocks only</i>,
    its keys and values are placed in front of the prompt, so those blocks attend to it and the others never see it. Each row is one prefill with its own blocks, text and position:
    <b>shift</b> (the prompt continues after it), <b>negative</b> (it sits at negative positions, the prompt starts at 0 as if nothing were there) or <b>overwrite</b> (those blocks read it <i>instead of</i> the prompt).
    Rows are the “Layer KV prefill” boxes wired into Embed; edit them here or on the canvas.`))
  const strip = el('div', { className: 'pfstrip' })
  const rowsEl = el('div')
  const nodes = () => flow.nodes.filter(x => x.data.kind === 'kv_text')
  const drawStrip = () => {
    const rows = nodes()
    strip.innerHTML = ''
    for (let b = 0; b < n; b++) {
      const cell = el('div', { className: 'pfcell', title: `block at position ${b}` })
      cell.appendChild(el('span', { className: 'pfn', textContent: b }))
      rows.forEach((r, i) => {
        const vis = r.data.enabled === false ? [] : (specBlocks(r.data.layers, n) || [])
        if (vis.includes(b)) cell.appendChild(el('i', { className: r.data.mode === 'overwrite' ? 'ow' : '', style: `background:${PF_COLORS[i % PF_COLORS.length]}`, title: `${r.data.label || 'prefill'} (${r.data.mode || 'shift'})` }))
      })
      cell.onclick = () => { addPrefillFor(String(b)); renderTab() }
      strip.appendChild(cell)
    }
  }
  const drawRows = () => {
    rowsEl.innerHTML = ''
    const rows = nodes()
    if (!rows.length) rowsEl.appendChild(el('div', { className: 'empty', textContent: 'No prefill yet. Click a block number above to add one for that block, or use the button below.' }))
    rows.forEach((r, i) => {
      const d = r.data
      const row = el('div', { className: 'pfrow' + (d.enabled === false ? ' off' : '') })
      const upd = patch => { flow.updateNodeData(r.id, patch); drawStrip() }
      const on = el('input', { type: 'checkbox', checked: d.enabled !== false, title: 'use this prefill' }); on.onchange = () => { upd({ enabled: on.checked }); row.classList.toggle('off', !on.checked) }
      const blocks = el('input', { value: d.layers ?? 'all', style: 'width:90px', title: `Which blocks read it: all | none | 14 | 10:14 | 3,5 (positions 0…${n - 1})` })
      blocks.oninput = () => { const ok = specBlocks(blocks.value, n) !== null; blocks.classList.toggle('badin', !ok); if (ok) upd({ layers: blocks.value }) }
      const mode = selectOf(PF_MODES, d.mode || 'shift')
      const off = numInput(d.offset ?? 0, 60); off.oninput = () => upd({ offset: +off.value || 0 })
      const offL = labelled('at', off); offL.style.display = (d.mode === 'offset') ? '' : 'none'
      mode.onchange = () => { upd({ mode: mode.value }); offL.style.display = mode.value === 'offset' ? '' : 'none' }
      const sys = el('input', { type: 'checkbox', checked: !!d.as_system }); sys.onchange = () => upd({ as_system: sys.checked })
      const sysL = el('label', { className: 'inl', title: 'Wrap the text as the system turn of the chat template before running it through the model' }); sysL.append(sys, el('span', { textContent: 'as system turn' }))
      const ta = el('textarea', { rows: 2, value: d.text || '', placeholder: 'text these blocks will read' }); ta.oninput = () => upd({ text: ta.value })
      const head = el('div', { className: 'row' })
      head.append(el('span', { className: 'pfdot', style: `background:${PF_COLORS[i % PF_COLORS.length]}` }), on, labelled('Blocks', blocks), labelled('Position', mode), offL, sysL,
        btn('Show box', () => flow.select({ type: 'node', id: r.id }), 'mini ghost'), btn('✕', () => { flow.removeNode(r.id); drawRows(); drawStrip() }, 'mini ghost'))
      row.append(head, ta)
      rowsEl.appendChild(row)
    })
  }
  const addPrefillFor = layers => addLayerPrompt(layers)
  const foot = el('div', { className: 'row' })
  foot.append(btn('+ Add a prefill', () => { addLayerPrompt('all'); drawRows(); drawStrip() }), btn('▶ Run', () => startRun(), 'primary'),
    el('span', { className: 'small', textContent: 'After a run, the Attention tab marks the prefill tokens: look at a block that reads it and one that does not.' }))
  box.append(strip, rowsEl, foot)
  drawStrip(); drawRows()
  return box
}

registerTab({ id: 'kv', title: 'KV caches', hint: 'Per-layer KV prefill and saved caches', render: renderKV })

on('model', () => { if (tabAvailable('kv')) refreshCaches() })
on('run-done', () => { if (tabAvailable('kv')) refreshCaches() })       // a finished run has a cache that can be saved
