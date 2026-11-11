import { GET, POST, S, api, btn, el, flow, hasFeature, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { aliveIn, followTask, labelled, selectOf } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'

S.qt = { bits: 4, cmp: [8, 4, 2], result: null }

export async function renderQuant(p) {
  if (!S.info) { p.innerHTML = '<div class="empty">Load a model first.</div>'; return }
  const st = S.qt
  let q = null
  try { const r = await GET('/api/quant'); q = r.active ? r : null; S.info.quant = q } catch { }
  p.appendChild(el('div', { className: 'small' }, `<b>Quantization.</b> Weights are usually stored as 16-bit numbers. Rounding them to 8, 4 or fewer bits makes a model 2–8× smaller —
    this is how large models fit on a laptop. Here every weight matrix inside the blocks is rounded on the fly (round-to-nearest, one scale per row), so you can see what it costs. Nothing on disk changes.
    This is the simplest possible scheme: the 4-bit files people download use smarter rounding (per-block scales, calibration data) and lose far less than you will see here — which is exactly why those schemes exist.`))
  const row = el('div', { className: 'row' })
  const bits = selectOf([[8, '8-bit'], [6, '6-bit'], [4, '4-bit'], [3, '3-bit'], [2, '2-bit'], [1, '1-bit (sign only)']], st.bits); bits.onchange = () => { st.bits = +bits.value }
  row.append(labelled('Round block weights to', bits), btn(q ? 'Change' : 'Apply', async () => {
    row.appendChild(el('span', {}, spinner('measuring the rounding error of every matrix…')))
    try { const r = await POST('/api/quant', { bits: st.bits }); toast(`Block weights are now rounded to ${r.bits} bits. Every run from here uses them.`); renderTab() }
    catch (e) { toast(e.message, 'error'); renderTab() }
  }, 'primary'))
  if (q) row.append(btn('✕ Back to full precision', async () => { await api('DELETE', '/api/quant'); toast('Full-precision weights restored.'); renderTab() }, 'ghost'))
  p.appendChild(row)
  if (q) {
    const s = el('div', { className: 'sec' })
    s.innerHTML = `<div class="row"><span class="chip coded">${q.bits}-bit active</span><span class="small">${q.n_weights} matrices · ${(q.n_params / 1e6).toFixed(1)} M weights ·
      ${q.size_mb_16bit} MB at 16-bit → <b>${q.size_mb} MB</b> at ${q.bits}-bit (${(16 / q.bits).toFixed(1)}× smaller)${q.measured ? ` · mean rounding error <b>${(q.mean_error * 100).toFixed(2)}%</b>` : ' · error not measured (model too large to scan)'}</span></div>`
    if (q.layers?.length) {
      const t = el('table', { className: 'grid' })
      t.innerHTML = '<tr><th>Block</th><th>Rounding error ‖W − round(W)‖ / ‖W‖</th></tr>'
      q.layers.forEach(l => {
        const tr = el('tr', { style: 'cursor:pointer' })
        tr.innerHTML = `<td>Block ${l.layer}</td><td><span class="pbar"><i style="width:${(l.heat * 100).toFixed(0)}%;background:var(--warn)"></i></span> ${(l.relative_error * 100).toFixed(2)}%</td>`
        tr.onclick = () => { const n = flow.nodes.find(x => x.data?.layer === l.layer); if (n) flow.select({ type: 'node', id: n.id }) }
        t.appendChild(tr)
      })
      s.appendChild(t)
    }
    p.appendChild(s)
  }
  if (!hasFeature('lm_tools')) return            // comparing generated text needs a language model
  const c = el('div', { className: 'kvbox' }, `<h4>Compare on the current prompt</h4><div class="small">Generates from the flow as it is now, at full precision and at each bit width. KL = how far the next-word odds drift from full precision along the same text (0 = identical).</div>`)
  const crow = el('div', { className: 'row' })
  for (const b of [8, 6, 4, 3, 2, 1]) {
    const cb = el('input', { type: 'checkbox', checked: st.cmp.includes(b) })
    cb.onchange = () => { st.cmp = cb.checked ? [...st.cmp, b].sort((x, y) => y - x) : st.cmp.filter(x => x !== b) }
    const l = el('label', { className: 'inl' }); l.append(cb, el('span', { textContent: `${b}-bit` })); crow.appendChild(l)
  }
  const out = el('div')
  crow.append(btn('Compare', async () => {
    const alive = aliveIn('quant')
    out.innerHTML = spinner('running the flow at each precision…')
    try {
      const start = await POST('/api/quant/compare', { flow: flow.getGraph(), bits: st.cmp })
      await followTask(start, r => { st.result = r; drawCompare(out, r, true) }, alive)
      if (alive()) drawCompare(out, st.result, false)
    } catch (e) { out.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
  }, 'primary'))
  c.append(crow, out)
  p.appendChild(c)
  if (st.result) drawCompare(out, st.result, false)
}

export function drawCompare(out, r, partial) {
  const t = el('table', { className: 'grid' })
  t.innerHTML = '<tr><th>Weights</th><th>Text it generates</th><th>Same tokens before it first differs</th><th>Top choice agrees</th><th>KL from full precision</th></tr>'
  const base = r.rows[0]?.tokens || []
  r.rows.forEach(x => {
    const tr = el('tr')
    const toks = x.tokens.map((tk, i) => `<span class="tok${i < x.same_tokens || x.bits == null ? '' : ' diff'}">${esc(tk)}</span>`).join('')
    tr.innerHTML = `<td><b>${esc(x.label)}</b></td><td><div class="tokens">${toks}</div></td><td>${x.bits == null ? '—' : `${x.same_tokens} / ${base.length}`}</td>
      <td>${x.bits == null ? '—' : (x.agree * 100).toFixed(0) + '%'}</td><td>${x.bits == null ? '—' : `<span class="pbar"><i style="width:${Math.min(100, x.kl * 50)}%;background:var(--bad)"></i></span> ${x.kl}`}</td>`
    t.appendChild(tr)
  })
  out.innerHTML = partial ? spinner(`${r.rows.length} done…`) : ''
  out.appendChild(t)
}

registerTab({ id: 'quant', title: 'Quantize', hint: 'Round the weights to 8, 4 or fewer bits and compare', render: renderQuant })
