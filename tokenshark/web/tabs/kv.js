import { GET, POST, S, api, btn, can, el, flow, heat, hooks, newId, on, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBlockKV, addKV, selectedNode } from '../ui/flowedit.js'
import { renderInspector } from '../ui/inspector.js'
import { addBox, labelled, numInput, selectOf } from '../ui/labkit.js'
import { startRun } from '../ui/run.js'
import { renderTab, setTab, tabAvailable } from '../ui/tabs.js'

const BLOCK_KV_OPS = [
  ['replace_system', 'replace_system · Replace system prompt'],
  ['append_system', 'append_system · Append to system prompt'],
  ['prepend_system', 'prepend_system · Prepend to system prompt'],
  ['replace_user', 'replace_user · Replace user’s prompt'],
  ['append_user', 'append_user · Append to user’s prompt'],
  ['prepend_user', 'prepend_user · Prepend to user’s prompt'],
]
const BKV_NICE = Object.fromEntries(BLOCK_KV_OPS.map(([k, lab]) => [k, lab.split(' · ')[1] || k]))

function blockAtSchedulePos(b) {
  const nid = S.planNodes?.[b]
  if (nid) return flow.node(nid) || flow.nodes.find(x => x.id === nid)
  const blocks = flow.nodes.filter(x => x.data.kind === 'block')
  return blocks[b] || null
}

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
  if (S.run && S.run.outputs?.cache) {
    const liveBox = el('div', { className: 'kvbox', style: 'margin-top:0;border-top:none;padding-top:0' })
    p.appendChild(liveBox)
    renderLiveKVAnalysis(liveBox)
  }
  const head = el('div', { className: 'row' })
  const up = el('input', { type: 'file', accept: '.safetensors' })
  up.onchange = async () => {
    const f = up.files[0]; if (!f) return
    try { await api('POST', `/api/caches/upload?name=${encodeURIComponent(f.name.replace(/\.safetensors$/, ''))}`, await f.arrayBuffer(), true); toast('Imported'); refreshCaches() }
    catch (e) { toast(e.message, 'error') }
  }
  head.append(btn('Save the last run’s cache as a file', saveRunCache), el('span', { className: 'small', textContent: 'Import a .safetensors KV file:' }), up)
  if (can('block_kv')) {
    p.appendChild(el('div', { className: 'sechead', textContent: 'Rewrite the prompt for single blocks' }))
    p.appendChild(blockKVEditor())
  }
  p.appendChild(el('div', { className: 'sechead', textContent: 'Saved KV caches (files)' }))
  p.appendChild(el('div', { className: 'small', textContent: 'Keep a KV cache as a file (saved from a finished run, built from text below, or imported), then put it in front of a new prompt with a “KV-cache prefix” box (shift / negative / offset). Use a file to reuse a whole conversation; use “Rewrite the prompt for single blocks” above to change what individual blocks remember.' }))
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
  const box = el('div', { className: 'kvbox' }, `<h4>${esc(c.name)} · ${c.n_tokens} tokens · ${c.n_slots} block slots · positions are RoPE positions (editable)</h4>`)
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

export function blockKVEditor() {
  const box = el('div', { className: 'kvbox', style: 'margin-top:0;border-top:none;padding-top:0' })
  const n = (S.planNodes?.length) || flow.nodes.filter(x => x.data.kind === 'block').length || S.info.n_layers
  const kvLayers = S.info?.kv_layers != null ? new Set(S.info.kv_layers) : null
  box.appendChild(el('div', { className: 'small' },
    `Give one block a different memory: its cache is <i>rebuilt</i> from a replaced or extended system / user prompt, then grows as tokens are generated.
    Click a block number to add a rewrite for it, or wire a rewrite box into a block’s <b>KV</b> port (full-attention blocks only).`))
  const strip = el('div', { className: 'pfstrip' })
  const rowsEl = el('div')
  const nodes = () => flow.nodes.filter(x => x.data.kind === 'block_kv')
  const targetsOf = r => {
    const fromEdges = flow.edges.filter(e => e.source === r.id && e.targetHandle === 'kv')
      .map(e => {
        const tid = e.target
        const i = (S.planNodes || []).indexOf(tid)
        if (i >= 0) return i
        return flow.nodes.filter(x => x.data.kind === 'block').findIndex(x => x.id === tid)
      })
      .filter(i => i >= 0)
    const fromSpec = r.data.enabled === false ? [] : (r.data.layers ? (specBlocks(r.data.layers, n) || []) : [])
    return [...new Set([...fromEdges, ...fromSpec])].sort((a, b) => a - b)
  }
  const bump = () => { hooks.scheduleValidate?.(); hooks.refreshBadges?.() }
  const drawStrip = () => {
    strip.innerHTML = ''
    const rows = nodes()
    for (let b = 0; b < n; b++) {
      const node = blockAtSchedulePos(b)
      const lay = node?.data?.layer ?? b
      const hasKv = node?.data?.has_kv !== false && (kvLayers == null || kvLayers.has(lay))
      const cell = el('div', { className: 'pfcell' + (hasKv ? '' : ' off'), title: hasKv ? `block at position ${b} (layer ${lay})` : `block ${b}: linear attention · no KV cache` })
      cell.appendChild(el('span', { className: 'pfn', textContent: b }))
      rows.forEach((r, i) => {
        if (r.data.enabled !== false && targetsOf(r).includes(b)) {
          cell.appendChild(el('i', { className: 'ow', style: `background:${PF_COLORS[i % PF_COLORS.length]}`, title: `${r.data.label || r.data.op} (${r.data.op || 'replace_system'})` }))
        }
      })
      if (hasKv) cell.onclick = () => { addBlockKV(node, 'replace_system'); renderTab() }
      strip.appendChild(cell)
    }
  }
  const drawRows = () => {
    rowsEl.innerHTML = ''
    const rows = nodes()
    if (!rows.length) {
      rowsEl.appendChild(el('div', { className: 'empty', textContent: 'No block KV overrides yet. Click a block number above, or use + Rewrite prompt (KV) in the palette / right-click a block.' }))
    }
    rows.forEach((r, i) => {
      const d = r.data
      const row = el('div', { className: 'pfrow' + (d.enabled === false ? ' off' : '') })
      const upd = patch => { flow.updateNodeData(r.id, patch); drawStrip(); bump() }
      const on = el('input', { type: 'checkbox', checked: d.enabled !== false, title: 'use this override' }); on.onchange = () => { upd({ enabled: on.checked }); row.classList.toggle('off', !on.checked) }
      const blocks = el('input', { value: d.layers ?? '', style: 'width:90px', title: `Extra schedule positions (optional if wired): all | 14 | 10:14 | 3,5` })
      blocks.oninput = () => {
        const v = blocks.value.trim()
        if (!v) { upd({ layers: '' }); return }
        const ok = specBlocks(v, n) !== null; blocks.classList.toggle('badin', !ok); if (ok) upd({ layers: v })
      }
      const op = selectOf(BLOCK_KV_OPS, d.op || 'replace_system')
      op.onchange = () => upd({ op: op.value, label: BKV_NICE[op.value] || d.label })
      const ta = el('textarea', { rows: 2, value: d.text || '', placeholder: 'text for the rewritten / appended prompt' }); ta.oninput = () => upd({ text: ta.value })
      const head = el('div', { className: 'row' })
      const tgt = targetsOf(r)
      head.append(el('span', { className: 'pfdot', style: `background:${PF_COLORS[i % PF_COLORS.length]}` }), on,
        labelled('Op', op), labelled('Also blocks', blocks),
        el('span', { className: 'small', textContent: tgt.length ? `→ #${tgt.join(', #')}` : 'wire to a block’s KV port' }),
        btn('Show box', () => flow.select({ type: 'node', id: r.id }), 'mini ghost'),
        btn('✕', () => { flow.removeNode(r.id); drawRows(); drawStrip(); bump() }, 'mini ghost'))
      row.append(head, ta)
      rowsEl.appendChild(row)
    })
  }
  const foot = el('div', { className: 'row' })
  foot.append(
    btn('+ Replace system', () => { addBlockKV(null, 'replace_system'); drawRows(); drawStrip(); bump() }),
    btn('+ Append system', () => { addBlockKV(null, 'append_system', 'Always answer in French.'); drawRows(); drawStrip(); bump() }),
    btn('+ Replace user', () => { addBlockKV(null, 'replace_user', 'The capital of Japan is'); drawRows(); drawStrip(); bump() }),
    btn('▶ Run', () => startRun(), 'primary'),
    el('span', { className: 'small', textContent: 'After a run, Attention on an overridden block shows the recreated key tokens; other blocks keep the real prompt.' }),
  )
  box.append(strip, rowsEl, foot)
  drawStrip(); drawRows()
  return box
}

// ---------- KV cache of the last run: one block or a chosen set of blocks, seen from the last query
const SEG_COLORS = { prompt: '#5b9dff', prefill: '#f5b84a', generated: '#3ecf8e', imported: '#a78bfa' }
const fmtBytes = b => b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(2)} MB` : `${(b / 1024).toFixed(1)} KB`
const pct = v => v == null ? '–' : `${(v * 100).toFixed(1)}%`
const blocksSpec = () => S.kvBlocks?.length ? S.kvBlocks.join(',') : 'all'
const blank = v => v === undefined || v === null || v === ''

// "3, 7, 10:12" → Set {3, 7, 10, 11}
function parseRows(spec) {
  const out = new Set()
  for (const part of String(spec ?? '').split(',').map(x => x.trim()).filter(Boolean)) {
    const m = part.match(/^(\d+)\s*:\s*(\d+)$/)
    if (m) for (let i = +m[1]; i < +m[2]; i++) out.add(i)
    else if (/^\d+$/.test(part)) out.add(+part)
  }
  return out
}
const rowsText = set => [...set].sort((x, y) => x - y).join(',')

// Where the RoPE / KV settings of the current scope live: the Sampler box (every block) or the selected block boxes.
function scopeNodes(a) {
  if (!S.kvBlocks?.length) {
    const s = flow.nodes.find(n => n.data.kind === 'sampler')
    return s ? [s] : []
  }
  return S.kvBlocks.map(p => flow.node(a.blocks[p]?.node) || blockAtSchedulePos(p)).filter(Boolean)
}
function zeroedRows(a) {
  const s = flow.nodes.find(n => n.data.kind === 'sampler')
  const rows = parseRows(s?.data.kv_zero)
  const blocks = S.kvBlocks?.length ? S.kvBlocks : a.blocks.map(b => b.pos)
  blocks.forEach(p => { const n = flow.node(a.blocks[p]?.node); parseRows(n?.data.kv_zero).forEach(i => rows.add(i)) })
  return rows
}

export async function renderLiveKVAnalysis(box) {
  if (!S.run) return
  const spec = blocksSpec()
  const qsel = S.kvQ?.run === S.run.id ? S.kvQ.idx : null      // chosen query token (null = the last one)
  const key = `${S.run.id}|${S.run.version ?? ''}|${spec}|${qsel ?? ''}`
  if (S.kvA?.key !== key) {
    box.innerHTML = spinner('reading the run’s KV cache…')
    try {
      S.kvA = { key, data: await GET(`/api/runs/${S.run.id}/kv_analysis?blocks=${encodeURIComponent(spec)}${qsel != null ? `&query=${qsel}` : ''}`) }
    } catch (e) {
      box.innerHTML = `<div class="small bad">${esc(e.message)}</div>`
      return
    }
  }
  drawLive(box, S.kvA.data)
}

function drawLive(box, a) {
  const redraw = () => drawLive(box, a)
  const refetch = () => renderLiveKVAnalysis(box)
  box.innerHTML = ''
  const sel = S.kvBlocks?.length ? S.kvBlocks : null
  const q = a.query || {}
  const zeroed = zeroedRows(a)
  const setQuery = idx => { S.kvQ = idx == null || idx === q.last_index ? null : { run: S.run.id, idx }; S.kvTok = null; refetch() }
  const canQuery = t => q.first_available == null || t.index >= q.first_available

  box.appendChild(el('div', { className: 'sechead', textContent: 'KV cache of the last run' }))

  // ---- size line (whole cache, or the selected blocks)
  const full = a.blocks.filter(b => !b.linear)
  const used = sel ? full.filter(b => sel.includes(b.pos)) : full
  const bytes = used.reduce((s, b) => s + (b.bytes || 0), 0)
  const attnKind = a.gqa_ratio > 1
    ? `<span class="chip" style="border-color:var(--ok)" title="Grouped-query attention: ${a.gqa_ratio} query heads share each key/value head">GQA · ${a.n_heads} query heads share ${a.n_kv_heads} KV heads → ${a.gqa_ratio}× smaller (${a.gqa_saving_pct}% saved vs. ${fmtBytes(a.mha_bytes_total)})</span>`
    : `<span class="chip" title="Multi-head attention: every query head has its own key/value head">MHA · ${a.n_heads} query heads, ${a.n_kv_heads} KV heads</span>`
  box.appendChild(el('div', { className: 'small' },
    `<b>${a.n_tokens}</b> cached tokens × <b>${sel ? used.length : full.length}</b> ${sel ? 'selected ' : ''}block${used.length === 1 ? '' : 's'} × <b>${a.n_kv_heads}</b> KV heads × <b>${a.head_dim}</b> dims per head = <b>${fmtBytes(bytes)}</b>` +
    `${sel ? ` of ${fmtBytes(a.bytes_total)}` : ''} (${Math.round(a.bytes_per_token).toLocaleString()} bytes per token, all blocks) · ${attnKind}`))

  // ---- which forward pass the attention numbers come from
  const nextTok = q.result_piece != null ? `<b class="tokchip">${esc(q.result_piece)}</b>` : null
  const res = !nextTok ? ' (no token was sampled from it)'
    : q.result_kind === 'generated' ? ` → generated ${nextTok}` : ` → predicting the next token, which the ${esc(q.result_kind)} already gave: ${nextTok}`
  const note = el('div', { className: 'kvnote' },
    `<b>One step only.</b> Attention and RoPE numbers below come from a single forward pass: ` +
    (q.is_last ? 'the one that generated the <b>last output token</b>. ' : `the one at token <b>#${q.index}</b>, not the last one. `) +
    `Its query is token #${q.index} <b class="tokchip">${esc(q.piece || '∅')}</b> at position ${q.position ?? '–'}${res}. ` +
    `Tokens after it did not exist yet for this query. ` +
    `Key / value norms describe what is stored in the cache. ${sel ? `Averaged over block${sel.length > 1 ? 's' : ''} #${sel.join(', #')}.` : 'Averaged over all blocks. Pick blocks below to look at one layer or a combination.'}` +
    `<br>Double-click a token (or press “Query” in the table) to see the step at that token instead. `)
  if (!q.is_last) note.appendChild(btn(`Back to the last token (#${q.last_index})`, () => setQuery(null), 'mini primary'))
  box.appendChild(note)

  // ---- block picker: click = only this block, Ctrl/Shift-click = add to / remove from the combination
  const pick = el('div', { className: 'row' })
  const specIn = el('input', { value: sel ? sel.join(',') : 'all', style: 'width:110px', title: 'all | 5 | 3,7 | 10:14 (schedule positions)' })
  specIn.onchange = () => {
    const v = specBlocks(specIn.value, a.n_slots)
    if (v === null) { specIn.classList.add('badin'); return }
    S.kvBlocks = v.length === a.n_slots ? null : v; refetch()
  }
  pick.append(el('b', { className: 'small', textContent: 'Blocks:' }),
    btn('All blocks', () => { S.kvBlocks = null; refetch() }, sel ? 'mini' : 'mini primary'), specIn,
    el('span', { className: 'small', textContent: 'Click a block to see only it; Ctrl- or Shift-click to combine several.' }))
  box.appendChild(pick)
  const strip = el('div', { className: 'pfstrip' })
  a.blocks.forEach(b => {
    const node = flow.node(b.node)
    const custom = !!(Object.keys(b.rope || {}).length || node?.data.kv_zero || !blank(node?.data.rope_gen_start))
    const cell = el('div', {
      className: 'pfcell' + (b.linear ? ' off' : '') + (sel?.includes(b.pos) ? ' on' : ''),
      title: b.linear ? `#${b.pos} · layer ${b.layer}: linear attention, no per-token KV cache`
        : `#${b.pos} · layer ${b.layer}${b.sink_token != null ? ` · last query looks most at #${b.sink_token} “${b.sink_piece}” (${pct(b.sink_attn)})` : ''}${custom ? ' · custom RoPE / zeroed records' : ''}`,
    })
    cell.appendChild(el('span', { className: 'pfn', textContent: b.pos }))
    if (b.sink_attn != null) cell.appendChild(el('i', { style: `background:${heat(b.sink_attn)}` }))
    if (custom) cell.appendChild(el('i', { style: 'background:var(--loop);height:3px' }))
    if (!b.linear) cell.onclick = ev => {
      const cur = new Set(sel || [])
      if (ev.ctrlKey || ev.metaKey || ev.shiftKey) { cur.has(b.pos) ? cur.delete(b.pos) : cur.add(b.pos) } else { cur.clear(); cur.add(b.pos) }
      S.kvBlocks = cur.size ? [...cur].sort((x, y) => x - y) : null
      refetch()
    }
    strip.appendChild(cell)
  })
  box.appendChild(strip)

  // ---- token strip coloured by attention from the last query
  const maxA = Math.max(...a.tokens.map(t => t.attn_received || 0), 1e-9)
  const tokWrap = el('div', { className: 'tokens', style: 'margin:6px 0' })
  if (S.kvTok == null || S.kvTok >= a.n_tokens) S.kvTok = a.tokens.reduce((m, t) => (t.attn_received ?? -1) > (a.tokens[m]?.attn_received ?? -1) ? t.index : m, 0)
  a.tokens.forEach(t => {
    const seen = t.attn_received != null
    const s = el('span', {
      className: `tok${seen ? '' : ' dim'}${t.index === S.kvTok ? ' on' : ''}${t.index === q.index ? ' isq' : ''}${zeroed.has(t.index) ? ' zeroed' : ''}`,
      style: seen ? `background:${heat(t.attn_received / maxA)}` : '',
      title: `#${t.index} “${t.piece}” · ${t.origin}\nposition ${t.eff_position}${t.eff_position !== t.position ? ` (stored ${t.position})` : ''}` +
        (seen ? `\nattention from query #${q.index}: ${pct(t.attn_received)} (at distance 0 it would get ${pct(t.attn_same_pos)})` : `\ncame after query #${q.index}: not seen by it`) +
        (t.index === q.index ? '\nthis is the query' : canQuery(t) ? '\ndouble-click: analyse the step at this token' : '') +
        (zeroed.has(t.index) ? '\nzeroed (key and value set to 0 on the next run)' : ''),
    }, `${esc(t.piece || '∅')}<sub>${t.eff_position}</sub>`)
    s.onclick = () => { S.kvTok = t.index; redraw() }
    s.ondblclick = () => { if (canQuery(t)) setQuery(t.index) }
    tokWrap.appendChild(s)
  })
  box.appendChild(tokWrap)
  box.appendChild(el('div', { className: 'small' }, `Colour = share of query #${q.index}’s attention (dark blue = little, bright = most; the outlined token is the query). Small number = RoPE position. Click a token to time it on the RoPE watch; double-click to make it the query.`))

  // ---- RoPE watch + controls side by side
  const grid = el('div', { className: 'kvgrid' })
  grid.append(ropeWatch(a, a.tokens[S.kvTok]), ropeControls(a, zeroed, redraw))
  box.appendChild(grid)

  // ---- per-token table
  const det = el('details', { className: 'howto', open: true },
    `<summary>Per token: positions, key / value norms and the attention lost to RoPE distance</summary>`)
  // K / V bars span the smallest → largest value in this table, so small differences stay visible
  const span = vals => { const v = vals.filter(x => x != null); const lo = Math.min(...v), hi = Math.max(...v); return x => hi > lo ? 8 + 92 * (x - lo) / (hi - lo) : 100 }
  const kBar = span(a.tokens.map(t => t.k_norm)), vBar = span(a.tokens.map(t => t.v_norm))
  det.appendChild(columnHelp(q))
  const maxE = Math.max(...a.tokens.map(t => Math.abs(t.rope_effect || 0)), 1e-6)
  const tt = el('table', { className: 'grid', style: 'font-size:12px;margin-bottom:10px' })
  tt.innerHTML = `<tr><th>#</th><th>Token</th><th>Segment</th><th title="position RoPE used in the selected blocks (stored position in brackets when different)">Position</th>
    <th title="query position − key position">Distance</th><th title="bar: smallest → largest in this table">Mean ‖K‖ (key size)</th><th title="bar: smallest → largest in this table">Mean ‖V‖ (value size)</th>
    <th title="share of query #${q.index}’s attention, averaged over heads and the selected blocks">Attention from query #${q.index}</th>
    <th title="attention with RoPE minus attention if this key sat at the query’s own position. Red: lost because of distance; green: gained.">RoPE distance effect<br><span class="small">◀ lost · gained ▶</span></th><th></th></tr>`
  a.tokens.forEach(t => {
    const tr = el('tr', { className: (t.index === S.kvTok ? 'selrow' : '') + (t.index === q.index ? ' qrow' : '') })
    const kw = Math.round(kBar(t.k_norm)), vw = Math.round(vBar(t.v_norm))
    const e = t.rope_effect
    const ew = e == null ? 0 : Math.round(Math.abs(e) / maxE * 50)
    const eff = e == null ? `<span class="small">${t.index > q.index ? 'after the query' : 'not seen'}</span>`
      : `<span class="effbar"><i class="${e < 0 ? 'neg' : 'pos'}" style="width:${ew}%;${e < 0 ? `right:50%` : `left:50%`}"></i></span> <span class="${e < 0 ? 'bad' : ''}">${e > 0 ? '+' : ''}${(e * 100).toFixed(1)} pts</span>`
    tr.innerHTML = `<td>#${t.index}</td>
      <td style="font-family:ui-monospace,monospace"><b>${esc(t.piece || '∅')}</b> <span class="small">(${t.id})</span>${zeroed.has(t.index) ? ' <span class="chip" style="border-color:var(--bad)">zeroed</span>' : ''}</td>
      <td><span class="chip seg-${t.origin}">${esc(t.origin)}</span></td>
      <td style="font-family:ui-monospace,monospace"><b>${t.eff_position}</b>${t.eff_position !== t.position ? ` <span class="small">(${t.position})</span>` : ''}</td>
      <td>${t.distance ?? '–'}</td>
      <td><span class="pbar"><i style="width:${kw}%"></i></span> ${t.k_norm}</td>
      <td><span class="pbar"><i style="width:${vw}%"></i></span> ${t.v_norm}</td>
      <td>${pct(t.attn_received)}${t.is_sink ? ' <span class="chip" style="background:#3a2f12;border-color:var(--warn)" title="draws far more attention than an average token">⚡ attention sink</span>' : ''}</td>
      <td style="white-space:nowrap">${eff}</td>`
    const td = el('td')
    td.append(btn(zeroed.has(t.index) ? 'Restore' : 'Zero', () => { toggleZero(a, t.index); redraw() }, 'mini'),
      btn('Watch', () => { S.kvTok = t.index; redraw() }, 'mini ghost'),
      t.index === q.index ? el('span', { className: 'chip', textContent: 'query' })
        : canQuery(t) ? btn('Query', () => setQuery(t.index), 'mini ghost') : '')
    tr.appendChild(td)
    tt.appendChild(tr)
  })
  det.appendChild(tt)
  box.appendChild(det)

  // ---- blocks whose prompt was rewritten
  const overrides = a.blocks.filter(b => b.override)
  if (overrides.length) {
    const ovWrap = el('div', { className: 'kvbox', style: 'margin:8px 0;padding:8px;background:var(--panel2);border-radius:6px' })
    ovWrap.appendChild(el('div', { className: 'small' },
      `<b>Blocks with a rewritten prompt (${overrides.length}):</b> these keep their own cache, so they are left out of the per-token averages above.`))
    overrides.forEach(b => {
      const ov = b.override
      const segOf = i => (ov.segments || []).find(s => i >= s.start && i < s.end)?.origin || 'prompt'
      const d = el('details', { style: 'margin-top:4px' })
      d.appendChild(el('summary', { className: 'small', style: 'cursor:pointer' },
        `<b>${esc(b.label)}</b> · <span class="chip" style="border-color:var(--kv)">${esc(ov.op)}</span> · ${ov.n_tokens} tokens (${ov.n_prompt} from the rewritten prompt + ${ov.n_tokens - ov.n_prompt} added while generating)${ov.text ? ` · “${esc(ov.text.slice(0, 60))}”` : ''}`))
      const st = el('div', { className: 'tokens', style: 'margin:4px 0' })
      ;(ov.tokens || []).forEach((t, i) => {
        const pos = ov.positions?.[i] ?? i
        st.appendChild(el('span', { className: `tok seg-${segOf(i)}`, title: `#${i} · ${segOf(i)} · position ${pos}` }, `${esc(t)}<sub>${pos}</sub>`))
      })
      d.appendChild(st)
      ovWrap.appendChild(d)
    })
    box.appendChild(ovWrap)
  }

  // ---- per-block table
  const bdet = el('details', { className: 'howto' }, `<summary>Per block: key / value sizes, where the last query looks, quick experiments</summary>`)
  const bt = el('table', { className: 'grid', style: 'font-size:12px' })
  bt.innerHTML = '<tr><th>Block</th><th>Prompt rewrite</th><th>Mean ‖K‖</th><th>Max ‖K‖</th><th>Mean ‖V‖</th><th>Max ‖V‖</th><th>Most-attended token (last query)</th><th>Try: does this block need its keys or its values?</th></tr>'
  a.blocks.forEach(b => {
    const tr = el('tr', { className: sel?.includes(b.pos) ? 'selrow' : '' })
    if (b.linear) {
      tr.innerHTML = `<td>${esc(b.label)}</td><td colspan="7" class="small">Linear attention (Gated DeltaNet): one fixed-size running memory, no per-token KV cache</td>`
      bt.appendChild(tr)
      return
    }
    const ov = b.override
      ? `<span class="chip" style="border-color:var(--kv)" title="${esc(b.override.text || '')}">${esc(b.override.op || 'rewrite')} · ${b.n_tokens} tok</span>`
      : '<span class="small">–</span>'
    tr.innerHTML = `<td><b>${esc(b.label)}</b></td><td>${ov}</td>
      <td>${b.k_norm_mean}</td><td class="small">${b.k_norm_max}</td>
      <td>${b.v_norm_mean}</td><td class="small">${b.v_norm_max}</td>
      <td>${b.sink_token != null ? `#${b.sink_token} “${esc(b.sink_piece)}” (${pct(b.sink_attn)})` : '–'}</td>`
    const td = el('td')
    td.append(
      btn('Zero keys', () => { addBox(b.node, { label: `zero kv_k @ ${b.label}`, hook: 'kv_k', op: 'zero', value: 0 }); toast(`Added “zero all cached keys” on ${b.label}. Press ▶ Run.`) }, 'mini'),
      btn('Zero values', () => { addBox(b.node, { label: `zero kv_v @ ${b.label}`, hook: 'kv_v', op: 'zero', value: 0 }); toast(`Added “zero all cached values” on ${b.label}. Press ▶ Run.`) }, 'mini'),
      btn('Keys → 2-bit', () => { addBox(b.node, { label: `2-bit kv_k @ ${b.label}`, hook: 'kv_k', op: 'quantize', value: 2 }); toast(`Added 2-bit cached keys on ${b.label}. Press ▶ Run.`) }, 'mini'),
      btn('Values → 2-bit', () => { addBox(b.node, { label: `2-bit kv_v @ ${b.label}`, hook: 'kv_v', op: 'quantize', value: 2 }); toast(`Added 2-bit cached values on ${b.label}. Press ▶ Run.`) }, 'mini'),
      btn('Inspect', () => { S.kvBlocks = [b.pos]; refetch() }, 'mini ghost'),
    )
    if (can('block_kv')) td.append(btn('Rewrite prompt', () => {
      const blk = flow.node(b.node)
      addBlockKV(blk, 'replace_system'); setTab('kv'); toast(`Added a prompt rewrite on ${b.label}.`)
    }, 'mini'))
    tr.appendChild(td)
    bt.appendChild(tr)
  })
  bdet.appendChild(bt)
  box.appendChild(bdet)
}

function toggleZero(a, idx) {
  const nodes = scopeNodes(a)
  if (!nodes.length) return
  const on = !zeroedRows(a).has(idx)
  // restoring removes it everywhere it was set (all blocks and the selected ones)
  const targets = on ? nodes : [flow.nodes.find(n => n.data.kind === 'sampler'), ...a.blocks.map(b => flow.node(b.node))].filter(Boolean)
  targets.forEach(n => {
    const rows = parseRows(n.data.kv_zero)
    on ? rows.add(idx) : rows.delete(idx)
    flow.updateNodeData(n.id, { kv_zero: rows.size ? rowsText(rows) : null })
  })
  hooks.refreshBadges?.(); hooks.scheduleValidate?.()
  toast(on ? `Token #${idx} will be zeroed ${S.kvBlocks?.length ? `in block${S.kvBlocks.length > 1 ? 's' : ''} #${S.kvBlocks.join(', #')}` : 'in every block'} on the next run. Press ▶ Run.` : `Token #${idx} restored.`)
}

// What each column of the per-token table means, in plain words.
function columnHelp(q) {
  const d = el('details', { className: 'howto', style: 'margin:4px 0 8px' }, '<summary>What do these columns mean?</summary>')
  d.appendChild(el('div', { className: 'small', style: 'line-height:1.55' },
    `Every block stores, for each token, a <b>key</b> (K: “what I am about”, used to be found) and a <b>value</b> (V: “what I hand over if picked”). ` +
    `Each new token makes a <b>query</b> (Q: “what I am looking for”). The query scores every key with a dot product, softmax turns the scores into shares that add up to 100 %, and the block takes that mix of values.<br>` +
    `<b>Mean ‖K‖ (key size)</b>: length of the token’s key vector, averaged over KV heads and the selected blocks. A dot product grows with length, so a longer key is easier to find, ` +
    `but its direction matters more than its size. Most keys come out about the same size because they are made from a normalised hidden state (Qwen also adds a large fixed bias to every key); ` +
    `the outliers, often the first token, are the interesting ones. The bar runs from the smallest to the largest value in this table.<br>` +
    `<b>Mean ‖V‖ (value size)</b>: length of the value vector, roughly how much this token can push into the output when it is attended to. ` +
    `Attention sinks usually get a big share but carry a tiny value: they soak up spare attention and hand back almost nothing.<br>` +
    `<b>Attention from query #${q.index}</b>: the share of that one query’s attention (averaged over heads and the selected blocks) that lands on this token. The column adds up to 100 %.<br>` +
    `<b>RoPE distance effect</b>: the share with RoPE as it ran, minus the share if this key sat at the query’s own position (distance 0). ` +
    `Red = the token lost attention because of where it sits; green = it gained. Because the shares add up to 100 %, one token’s gain is the others’ loss, so the effect is not a smooth fade with distance ` +
    `(on the RoPE watch, the fast hands wrap around every few positions).`))
  return d
}

// RoPE as a stopwatch: one position = one second. The model's maximum context is its whole life,
// so the display counts down the positions left; each cache segment (prompt / prefill / response) is a lap.
const LAP_NAMES = { prompt: 'Prompt', prefill: 'Prefill', generated: 'Response', imported: 'Imported' }
function hms(sec) {
  const neg = sec < 0, s = Math.abs(sec)
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60
  const rs = Number.isInteger(r) ? String(r).padStart(2, '0') : r.toFixed(2).padStart(5, '0')
  return `${neg ? '−' : ''}${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${rs}`
}
function ropeWatch(a, t) {
  const wrap = el('div', { className: 'kvclock' })
  const q = a.query || {}
  const toks = a.tokens
  const P = x => +x || 0
  const now = q.position != null ? P(q.position) : Math.max(0, ...toks.map(x => P(x.eff_position)))
  const max = +a.max_context || 0
  const left = max ? max - now - 1 : null
  // laps: runs of tokens with the same origin
  const laps = []
  toks.forEach(x => {
    const l = laps[laps.length - 1]
    if (l && l.origin === x.origin && P(x.eff_position) >= l.end) { l.end = P(x.eff_position); l.n++ }
    else laps.push({ origin: x.origin, start: P(x.eff_position), end: P(x.eff_position), n: 1 })
  })
  const over = left != null && left < 0
  const face = el('div', { className: 'watch' + (over ? ' over' : '') })
  face.innerHTML =
    `<div class="wlabel">${max ? (over ? 'Past the trained context by' : 'Context left') : 'Elapsed'}</div>` +
    `<div class="wdigits">${max ? hms(over ? -left : left) : hms(now)}</div>` +
    `<div class="wsub">elapsed <b>${hms(now)}</b>${max ? ` of a life of <b>${hms(max)}</b> (${max.toLocaleString()} positions, ${((now + 1) / max * 100).toPrecision(2)}% used)` : ''}</div>`
  // life bar: whole context, segment colours, query and selected token marks
  const span = Math.max(max || 0, now + 1, ...toks.map(x => P(x.eff_position) + 1))
  const X = p => `${Math.min(100, Math.max(0, (P(p) / span) * 100))}%`
  const bar = el('div', { className: 'wlife' })
  bar.innerHTML = laps.map(l => `<i style="left:${X(l.start)};width:max(2px,calc(${X(l.end + 1)} - ${X(l.start)}));background:${SEG_COLORS[l.origin] || '#888'}" title="${esc(LAP_NAMES[l.origin] || l.origin)}"></i>`).join('') +
    (t && t.eff_position != null ? `<b class="wmark sel" style="left:${X(t.eff_position)}" title="selected token"></b>` : '') +
    `<b class="wmark q" style="left:${X(now)}" title="last query"></b>`
  face.appendChild(bar)
  // lap list, like a stopwatch
  const lapRows = laps.map((l, i) =>
    `<tr><td><span class="sw" style="background:${SEG_COLORS[l.origin] || '#888'}"></span>Lap ${i + 1} · ${esc(LAP_NAMES[l.origin] || l.origin)}</td>` +
    `<td>${hms(l.start)} → ${hms(l.end)}</td><td>${hms(l.end - l.start + 1)}</td><td>${l.n} tok</td></tr>`).join('')
  face.appendChild(el('table', { className: 'wlaps' }, `<tr><th>Lap</th><th>From → to</th><th>Length</th><th></th></tr>${lapRows}` +
    `<tr class="wq"><td>▶ ${q.is_last ? 'Last query' : `Query #${q.index}`}</td><td>${hms(now)}</td><td colspan="2">${q.piece != null ? `<b class="tokchip">${esc(q.piece || '∅')}</b>` : ''}</td></tr>`))
  wrap.appendChild(face)
  // the selected token, and how far each RoPE pair has turned between it and the query
  const theta = (S.kvBlocks?.length === 1 && a.blocks[S.kvBlocks[0]]?.rope?.theta) || a.rope_theta || 10000
  const half = Math.max(1, Math.floor((a.rot_dim || a.head_dim) / 2))
  const slow = Math.pow(theta, -(half - 1) / half)
  const turns = (d, f) => Math.abs(P(d)) * f / (2 * Math.PI)
  const fmt = v => v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toPrecision(2)
  const lapsPerFast = 2 * Math.PI, lapsPerSlow = 2 * Math.PI / slow
  wrap.appendChild(el('div', { className: 'small' },
    (t ? `<b style="color:var(--accent)">Selected</b>: #${t.index} <b class="tokchip">${esc(t.piece || '∅')}</b> at ${hms(P(t.eff_position))}, ` +
      `<span style="color:var(--bad)">${hms(P(t.distance))} before the query</span>.<br>` +
      `RoPE works like a watch with ${half} hands per head, each turning at its own speed. Over this gap the fastest hand ` +
      `turns <b>${fmt(turns(t.distance, 1))}</b> full circles (one circle every ${lapsPerFast.toFixed(1)} positions), ` +
      `the slowest <b>${fmt(turns(t.distance, slow))}</b> (one circle every ${Math.round(lapsPerSlow).toLocaleString()} positions, θ = ${theta}).<br>` : '') +
    `The life is the model’s trained context (<code>max_position_embeddings</code>); past it RoPE angles were never seen in training.`))
  return wrap
}

// Change the RoPE clock and the cache for every block (Sampler box) or for the selected blocks (their block boxes).
function ropeControls(a, zeroed, redraw) {
  const box = el('div', { className: 'kvctl' })
  const nodes = scopeNodes(a)
  const all = !S.kvBlocks?.length
  const first = nodes[0]?.data || {}
  const mixed = k => nodes.some(n => String(n.data[k] ?? '') !== String(first[k] ?? ''))
  box.appendChild(el('div', { className: 'small' },
    `<b style="color:var(--text)">Change RoPE &amp; cache: ${all ? 'every block' : `block${nodes.length > 1 ? 's' : ''} #${S.kvBlocks.join(', #')}`}</b><br>` +
    (all ? 'Stored in the Sampler box. Select blocks above to change only those.' : 'Stored in the selected block boxes. Choose “All blocks” above to change every block at once.')))
  const fields = all
    ? [['rope_prompt_scale', 'Clock speed over the prompt', '1 = normal · 0.5 = half speed · 0 = all prompt tokens at position 0'],
      ['rope_step_stride', 'Clock speed while generating', 'positions per new token: 1 = normal · 0 = frozen · 50 = fast-forward'],
      ['rope_gen_start', 'Jump: generated tokens continue from position', 'empty = no jump · e.g. 1000']]
    : [['rope_scale', 'Clock speed (multiplies every position)', 'empty / 1 = normal · 2 = twice as fast · 0 = RoPE off · −1 = backwards'],
      ['rope_gen_start', 'Jump: generated tokens continue from position', 'empty = no jump · e.g. 1000'],
      ['rope_theta', 'Gear (base θ)', `empty = the model’s ${a.rope_theta} · smaller = every pair turns faster`],
      ['rope_q_offset', 'Query shift Δ (query position vs. keys)', '0 = normal · +100 = the query looks 100 positions older text']]
  const inputs = {}
  for (const [k, label, hint] of fields) {
    const inp = el('input', { type: 'number', step: 'any', value: blank(first[k]) ? '' : first[k], placeholder: mixed(k) ? 'mixed' : '', style: 'width:100px' })
    inputs[k] = inp
    const f = el('label', { className: 'field' })
    f.append(el('span', { textContent: label }), inp, el('span', { className: 'small', textContent: hint }))
    box.appendChild(f)
  }
  const zIn = el('input', { value: rowsText(new Set(nodes.flatMap(n => [...parseRows(n.data.kv_zero)]))), placeholder: 'e.g. 3, 7, 10:12', style: 'width:160px' })
  const zf = el('label', { className: 'field' })
  zf.append(el('span', { textContent: 'Zeroed KV records (token #)' }), zIn,
    el('span', { className: 'small', textContent: 'That token’s key and value become 0: the value adds nothing, but a zero key still scores 0, so a little attention can land there.' }))
  box.appendChild(zf)
  box.appendChild(el('details', { className: 'howto', style: 'margin:2px 0 6px' },
    '<summary>Why can the model still know a zeroed token?</summary>' +
    '<div class="small" style="line-height:1.55">Zeroing erases only that token’s own record. Every token that came after it was computed while it was still visible: ' +
    'in every layer, those later tokens (the rest of the prompt, the chat template, the reply so far) looked at it and copied what they needed into their own keys and values. ' +
    'Those copies stay in the cache, so a token zeroed in the middle of a run can still be “remembered”. A large model (e.g. Gemma-4 12B) copies strongly; a small one (Qwen2.5-0.5B) only weakly, ' +
    'so the same mid-run zero may change one model’s answer and not the other’s.<br>' +
    '<b>To forget it fully</b>, zero it before the prompt is read: <b>Apply &amp; run</b> instead of Apply + Token. To continue from the same point, put the reply so far in the prompt’s prefill.</div>'))
  const apply = () => {
    nodes.forEach(n => {
      const patch = {}
      for (const [k, inp] of Object.entries(inputs)) {
        if (inp.value === '' && inp.placeholder === 'mixed') continue
        patch[k] = inp.value === '' ? null : +inp.value
      }
      const rows = parseRows(zIn.value)
      patch.kv_zero = rows.size ? rowsText(rows) : null
      flow.updateNodeData(n.id, patch)
    })
    hooks.refreshBadges?.(); hooks.scheduleValidate?.()
  }
  const row = el('div', { className: 'row' })
  row.append(
    btn('Apply & run ▶', () => { apply(); startRun() }, 'primary'),
    btn('Apply', () => {
      apply()
      toast(S.debug ? 'Applied. The next ▸ Token or Block step uses it. Zeroing erases the record for this session, but tokens read after it keep what they copied from it; Apply & run forgets it fully.' : 'Saved in the flow. Press ▶ Run to see the effect.')
      redraw()
    }),
    btn('Reset', () => {
      Object.values(inputs).forEach(i => { i.value = ''; i.placeholder = '' }); zIn.value = ''
      apply(); toast('RoPE and zeroed records reset for this scope.'); redraw()
    }, 'ghost'),
    btn(`Zero #${S.kvTok}`, () => { toggleZero(a, S.kvTok); redraw() }, 'mini'))
  box.appendChild(row)
  if (zeroed.size) box.appendChild(el('div', { className: 'small', textContent: `Zeroed in the shown blocks on the next run: #${rowsText(zeroed).split(',').join(', #')}` }))
  return box
}

registerTab({ id: 'kv', title: 'KV caches', hint: 'The last run’s KV cache and RoPE watch (per block or combined), prompt rewrites per block, saved cache files', render: renderKV })

on('model', () => { if (tabAvailable('kv')) refreshCaches() })
on('run-done', () => { if (tabAvailable('kv')) refreshCaches() })       // a finished run has a cache that can be saved
