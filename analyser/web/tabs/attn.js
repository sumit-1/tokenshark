import { GET, S, el, flow, heat, registerTab } from '../core.js'
import { esc } from '../flow.js'
import { renderTab } from '../ui/tabs.js'

// ---------- attention
export async function renderAttn(p) {
  const sched = S.run.plan.schedule, nodes = S.run.plan.schedule_nodes
  const bar = el('div', { className: 'row' })
  const sel = el('select'); sched.forEach((l, i) => sel.appendChild(el('option', { value: i, textContent: `#${i} · ${flow.node(nodes[i])?.data.label || 'L' + l}` })))
  sel.value = S.attnPos
  const hs = el('select'); hs.appendChild(el('option', { value: -1, textContent: 'mean of heads' }))
  for (let i = 0; i < (S.run.outputs.n_heads || 0); i++) hs.appendChild(el('option', { value: i, textContent: `head ${i}` }))
  hs.value = S.attnHead
  sel.onchange = () => { S.attnPos = +sel.value; renderTab() }
  hs.onchange = () => { S.attnHead = +hs.value; renderTab() }
  bar.append(el('span', { textContent: 'Block' }), sel, el('span', { textContent: 'Head' }), hs)
  p.appendChild(bar)
  let a
  try { a = await GET(`/api/runs/${S.run.id}/attention?pos=${S.attnPos}&head=${S.attnHead}`) } catch (e) { p.appendChild(el('div', { className: 'empty', textContent: e.message })); return }
  if (a.error) { p.appendChild(el('div', { className: 'empty', textContent: a.error })); return }
  const nq = a.matrix.length, nk = a.matrix[0].length
  const cell = Math.max(3, Math.min(14, Math.floor(900 / nk)))
  const cv = el('canvas', { width: nk * cell, height: nq * cell, className: 'attn' })
  const g = cv.getContext('2d')
  for (let i = 0; i < nq; i++) for (let j = 0; j < nk; j++) {
    const v = Math.sqrt(a.matrix[i][j]); g.fillStyle = `rgba(255,${180 - v * 120},${60},${v})`
    g.fillRect(j * cell, i * cell, cell, cell)
  }
  if (a.n_prefix) { g.strokeStyle = '#7dd3fc'; g.setLineDash([4, 3]); g.strokeRect(0, 0, a.n_prefix * cell, nq * cell) }
  const info = el('div', { className: 'small', textContent: `rows = query tokens of this run (${nq}), columns = keys in the cache (${nk})${a.n_prefix ? `, first ${a.n_prefix} imported (dashed)` : ''}. Hover for details.` })
  cv.onmousemove = ev => {
    const r = cv.getBoundingClientRect(), j = Math.floor((ev.clientX - r.left) / cell), i = Math.floor((ev.clientY - r.top) / cell)
    if (i < nq && j < nk) info.textContent = `query #${i} “${a.query_tokens[i]}” → key #${j} “${a.key_tokens[j]}” (RoPE pos ${a.key_positions[j]}): ${(a.matrix[i][j] * 100).toFixed(2)}%`
  }
  const wrap = el('div', { className: 'attn-wrap' })
  wrap.appendChild(cv)
  const picked = el('div', { className: 'attnpick' })
  cv.onclick = ev => {
    const r = cv.getBoundingClientRect(), i = Math.floor((ev.clientY - r.top) / cell)
    if (i >= nq) return
    const row = a.matrix[i].map((v, j) => [v, j]).sort((x, y) => y[0] - x[0]).slice(0, 8)
    picked.innerHTML = `<b>When processing #${i} “${esc(a.query_tokens[i])}”, this ${S.attnHead < 0 ? 'block (average of heads)' : 'head'} reads mostly from:</b> ` +
      row.map(([v, j]) => `<span class="chip" style="background:${heat(Math.sqrt(v))}">${esc(a.key_tokens[j])} <sub>#${j}</sub> ${(v * 100).toFixed(0)}%</span>`).join('')
  }
  p.append(info, picked, wrap)
  p.insertAdjacentHTML('afterbegin', `<details class="howto"><summary>How to read the attention map</summary>
    <ul><li>Each <b>row</b> is a token being processed (a “query”); each <b>column</b> is an earlier token it can look at (a “key”). Bright = this row pulls a lot of information from that column. Each row adds up to 100%.</li>
    <li>The map is a <b>triangle</b>: a token can only look backwards, never at future tokens.</li>
    <li><b>Click a row</b> to list the tokens it reads most. Example: on the row for “ is” in “The capital of France is”, a block that has learned the fact usually reads strongly from “ France”.</li>
    <li>A bright <b>first column</b> is normal: many heads park attention on the first token when they have nothing useful to read (an “attention sink”).</li>
    <li>A bright <b>diagonal</b> = the token reads itself; a stripe just left of it = reads the previous token (common in early blocks).</li>
    <li>Pick single heads: different heads do different jobs (previous-token, copy, name-mover…). The mean of heads blurs them. A KV-cache prefix shows as the dashed region on the left.</li></ul></details>`)
}

registerTab({ id: 'attn', title: 'Attention', needs: 'run', render: renderAttn })
