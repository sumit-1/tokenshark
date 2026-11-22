import { GET, POST, S, api, btn, can, el, flow, heat, hooks, newId, on, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBlockKV, addKV, selectedNode } from '../ui/flowedit.js'
import { renderInspector } from '../ui/inspector.js'
import { addBox, labelled, numInput, selectOf } from '../ui/labkit.js'
import { startRun } from '../ui/run.js'
import { renderTab, setTab, tabAvailable } from '../ui/tabs.js'

const BLOCK_KV_OPS = [
  ['replace_system', 'replace_system — Replace system prompt'],
  ['append_system', 'append_system — Append to system prompt'],
  ['prepend_system', 'prepend_system — Prepend to system prompt'],
  ['replace_user', 'replace_user — Replace user’s prompt'],
  ['append_user', 'append_user — Append to user’s prompt'],
  ['prepend_user', 'prepend_user — Prepend to user’s prompt'],
]
const BKV_NICE = Object.fromEntries(BLOCK_KV_OPS.map(([k, lab]) => [k, lab.split(' — ')[1] || k]))

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
  head.append(btn('Save current run’s cache', saveRunCache), el('span', { className: 'small', textContent: 'Import a .safetensors KV file:' }), up)
  if (can('block_kv')) p.appendChild(blockKVEditor())
  p.appendChild(el('div', { className: 'sechead', textContent: 'Saved KV caches (files)' }))
  p.appendChild(el('div', { className: 'small', textContent: 'A KV cache can also be kept as a file — saved from a finished run, built from text below, or imported — and prepended with a “KV-cache prefix” box (shift / negative / offset). Use file prefixes to reuse a whole conversation; use Rewrite prompt (KV) above to change what individual blocks remember.' }))
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
    `<b>Rewrite prompt per block.</b> Wire a box into a block’s <b>KV</b> port (full-attention only).
    That block’s cache is <i>recreated</i> from a replaced or appended system / user prompt, then grows as tokens are generated.
    Click a block number below to add an override for that block.`))
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
      const cell = el('div', { className: 'pfcell' + (hasKv ? '' : ' off'), title: hasKv ? `block at position ${b} (layer ${lay})` : `block ${b}: linear attention — no KV cache` })
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

export async function renderLiveKVAnalysis(box) {
  if (!S.run) return
  box.innerHTML = spinner('analyzing run KV cache…')
  try {
    const a = await GET(`/api/runs/${S.run.id}/kv_analysis`)
    box.innerHTML = ''
    const kb = (a.bytes_total / 1024).toFixed(1)
    const mhaKb = (a.mha_bytes_total / 1024).toFixed(1)
    const gqaText = a.gqa_ratio > 1
      ? `<span class="chip" style="border-color:var(--ok)">GQA ${a.n_heads}Q / ${a.n_kv_heads}KV → <b>${a.gqa_ratio}× smaller</b> (${a.gqa_saving_pct}% saved vs ${mhaKb} KB MHA)</span>`
      : `<span class="chip">MHA ${a.n_heads}Q / ${a.n_kv_heads}KV</span>`
    box.appendChild(el('div', { className: 'small' },
      `<b>Active Run KV Cache Analysis</b> — <b>${a.n_tokens}</b> tokens × <b>${a.n_slots}</b> block slots × <b>${a.n_kv_heads}</b> KV heads × <b>${a.head_dim}</b> head_dim = <b>${kb} KB</b> (${a.bytes_per_token} B/token) · ${gqaText}`))

    // Per-token strip with RoPE position, K/V norms, and Attention Sink indicator
    const tokWrap = el('div', { className: 'tokens', style: 'margin:8px 0' })
    const maxK = Math.max(...a.tokens.map(t => t.k_norm || 0), 1e-6)
    const maxV = Math.max(...a.tokens.map(t => t.v_norm || 0), 1e-6)
    a.tokens.forEach(t => {
      const sinkBadge = t.is_sink ? ' ⚡sink' : ''
      const attnPct = t.attn_received != null ? ` · attn in ${(t.attn_received * 100).toFixed(1)}%` : ''
      const s = el('span', {
        className: `tok seg-${t.origin}${t.is_sink ? ' on' : ''}`,
        title: `#${t.index} “${t.piece}” (id ${t.id}) · origin: ${t.origin}\nRoPE pos: ${t.position} · mean ‖K‖: ${t.k_norm} · mean ‖V‖: ${t.v_norm}${attnPct}`,
      }, `${esc(t.piece || '∅')}<sub>pos ${t.position}${sinkBadge}</sub>`)
      tokWrap.appendChild(s)
    })
    box.appendChild(tokWrap)

    // Expandable detailed Token & Block KV tables
    const det = el('details', { className: 'howto', open: true },
      `<summary>KV Cache Breakdown: Per-Token Norms, RoPE Positions & Per-Block K vs. V Sensitivity</summary>`)

    // 1. Per-token table
    const tt = el('table', { className: 'grid', style: 'font-size:12px;margin-bottom:10px' })
    tt.innerHTML = '<tr><th>#</th><th>Token</th><th>Segment</th><th>RoPE Pos</th><th>Mean ‖K‖ (Key norm)</th><th>Mean ‖V‖ (Value norm)</th><th>Avg Attention Received</th></tr>'
    a.tokens.forEach(t => {
      const tr = el('tr')
      const kw = Math.round((t.k_norm / maxK) * 100), vw = Math.round((t.v_norm / maxV) * 100)
      tr.innerHTML = `<td>#${t.index}</td>
        <td style="font-family:ui-monospace,monospace"><b>${esc(t.piece || '∅')}</b> <span class="small">(${t.id})</span></td>
        <td><span class="chip seg-${t.origin}">${esc(t.origin)}</span></td>
        <td style="font-family:ui-monospace,monospace"><b>${t.position}</b></td>
        <td><span class="pbar"><i style="width:${kw}%"></i></span> ${t.k_norm}</td>
        <td><span class="pbar"><i style="width:${vw}%"></i></span> ${t.v_norm}</td>
        <td>${t.attn_received != null ? `${(t.attn_received * 100).toFixed(1)}%${t.is_sink ? ' <span class="chip" style="background:#3a2f12;border-color:var(--warn)">⚡ Attention Sink</span>' : ''}` : '—'}</td>`
      tt.appendChild(tr)
    })
    det.appendChild(tt)

    // 2. Per-block table with 1-click K vs V interceptor buttons
    const overrides = a.blocks.filter(b => b.override)
    if (overrides.length) {
      const ovWrap = el('div', { className: 'kvbox', style: 'margin:8px 0;padding:8px;background:var(--panel2);border-radius:6px' })
      ovWrap.appendChild(el('div', { className: 'small' },
        `<b>Recreated Per-Block KV Caches (${overrides.length} block${overrides.length > 1 ? 's' : ''}):</b>`))
      overrides.forEach(b => {
        const ov = b.override
        const segOf = i => (ov.segments || []).find(s => i >= s.start && i < s.end)?.origin || 'prompt'
        const d = el('details', { style: 'margin-top:4px' })
        d.appendChild(el('summary', { className: 'small', style: 'cursor:pointer' },
          `<b>${esc(b.label)}</b> · <span class="chip" style="border-color:var(--kv)">${esc(ov.op)}</span> · ${ov.n_tokens} tokens (${ov.n_prompt} recreated prompt + ${ov.n_tokens - ov.n_prompt} grown)${ov.text ? ` · “${esc(ov.text.slice(0, 60))}”` : ''}`))
        const strip = el('div', { className: 'tokens', style: 'margin:4px 0' })
        ;(ov.tokens || []).forEach((t, i) => {
          const pos = ov.positions?.[i] ?? i
          strip.appendChild(el('span', {
            className: `tok seg-${segOf(i)}`,
            title: `#${i} · ${segOf(i)} · RoPE pos ${pos}`,
          }, `${esc(t)}<sub>pos ${pos}</sub>`))
        })
        d.appendChild(strip)
        ovWrap.appendChild(d)
      })
      box.appendChild(ovWrap)
    }
    const bt = el('table', { className: 'grid', style: 'font-size:12px' })
    bt.innerHTML = '<tr><th>Block</th><th>KV override</th><th>Mean ‖K‖</th><th>Max ‖K‖</th><th>Mean ‖V‖</th><th>Max ‖V‖</th><th>Top Sink Token</th><th>Test K vs. V Sensitivity</th></tr>'
    a.blocks.forEach(b => {
      const tr = el('tr')
      if (b.linear) {
        tr.innerHTML = `<td>${esc(b.label)}</td><td colspan="7" class="small">Linear attention (Gated DeltaNet) — fixed-size recurrent state matrix S, no per-token KV cache</td>`
        bt.appendChild(tr)
        return
      }
      const ov = b.override
        ? `<span class="chip" style="border-color:var(--kv)" title="${esc(b.override.text || '')}">${esc(b.override.op || 'override')} · ${b.n_tokens} tok</span>`
        : '<span class="small">—</span>'
      tr.innerHTML = `<td><b>${esc(b.label)}</b></td>
        <td>${ov}</td>
        <td>${b.k_norm_mean}</td><td class="small">${b.k_norm_max}</td>
        <td>${b.v_norm_mean}</td><td class="small">${b.v_norm_max}</td>
        <td>${b.sink_token != null ? `#${b.sink_token} “${esc(b.sink_piece)}” (${(b.sink_attn * 100).toFixed(0)}%)` : '—'}</td>`
      const tdAct = el('td')
      if (!b.linear) {
        tdAct.append(
          btn('Zero K', () => { addBox(b.node, { label: `zero kv_k @ ${b.label}`, hook: 'kv_k', op: 'zero', value: 0 }); toast(`Added Zero Keys (kv_k) on ${b.label}. Press ▶ Run.`) }, 'mini'),
          btn('Zero V', () => { addBox(b.node, { label: `zero kv_v @ ${b.label}`, hook: 'kv_v', op: 'zero', value: 0 }); toast(`Added Zero Values (kv_v) on ${b.label}. Press ▶ Run.`) }, 'mini'),
          btn('2-bit K', () => { addBox(b.node, { label: `2-bit kv_k @ ${b.label}`, hook: 'kv_k', op: 'quantize', value: 2 }); toast(`Added 2-bit Key quantization on ${b.label}. Press ▶ Run.`) }, 'mini'),
          btn('2-bit V', () => { addBox(b.node, { label: `2-bit kv_v @ ${b.label}`, hook: 'kv_v', op: 'quantize', value: 2 }); toast(`Added 2-bit Value quantization on ${b.label}. Press ▶ Run.`) }, 'mini'),
        )
        if (can('block_kv')) {
          tdAct.append(btn('Rewrite prompt', () => {
            const blk = flow.node(b.node) || flow.nodes.find(x => x.id === b.node)
            addBlockKV(blk, 'replace_system'); setTab('kv'); toast(`Added Block KV override on ${b.label}.`)
          }, 'mini'))
        }
      }
      tr.appendChild(tdAct)
      bt.appendChild(tr)
    })
    det.appendChild(bt)
    box.appendChild(det)
  } catch (e) {
    box.innerHTML = `<div class="small bad">${esc(e.message)}</div>`
  }
}

registerTab({ id: 'kv', title: 'KV caches', hint: 'Rewrite prompt per block, live KV analysis, and saved caches', render: renderKV })

on('model', () => { if (tabAvailable('kv')) refreshCaches() })
on('run-done', () => { if (tabAvailable('kv')) refreshCaches() })       // a finished run has a cache that can be saved
