import { GET, POST, S, btn, el, flow, heat, registerTab, toast } from '../core.js'
import { esc } from '../flow.js'
import { addIntervention } from '../ui/flowedit.js'
import { renderTab } from '../ui/tabs.js'

S.attnView = S.attnView || 'matrix'      // 'matrix' | 'traceback'
S.attnTrace = S.attnTrace || { query_token: -1, key_token: '', target_word: '', top_k: 25, res: null }

export async function renderAttn(p) {
  const sched = S.run.plan.schedule, nodes = S.run.plan.schedule_nodes
  const bar = el('div', { className: 'row' })

  const modeMatrix = btn('Attention Matrix', () => { S.attnView = 'matrix'; renderTab() }, S.attnView === 'matrix' ? 'primary mini' : 'mini ghost')
  const modeTrace = btn('Previous-Token Attribution (All Layers × Heads)', () => { S.attnView = 'traceback'; renderTab() }, S.attnView === 'traceback' ? 'primary mini' : 'mini ghost')
  bar.append(modeMatrix, modeTrace)

  if (S.attnView === 'traceback') {
    p.appendChild(bar)
    return renderAttnTraceback(p)
  }

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
    if (i < nq && j < nk) info.textContent = `query #${i} "${a.query_tokens[i]}" -> key #${j} "${a.key_tokens[j]}" (RoPE pos ${a.key_positions[j]}): ${(a.matrix[i][j] * 100).toFixed(2)}%`
  }
  const wrap = el('div', { className: 'attn-wrap' })
  wrap.appendChild(cv)
  const picked = el('div', { className: 'attnpick' })
  cv.onclick = ev => {
    const r = cv.getBoundingClientRect(), i = Math.floor((ev.clientY - r.top) / cell)
    if (i >= nq) return
    const row = a.matrix[i].map((v, j) => [v, j]).sort((x, y) => y[0] - x[0]).slice(0, 8)
    picked.innerHTML = `<b>When processing #${i} "${esc(a.query_tokens[i])}", this ${S.attnHead < 0 ? 'block (average of heads)' : 'head'} reads mostly from:</b> ` +
      row.map(([v, j]) => `<span class="chip" style="background:${heat(Math.sqrt(v))}">${esc(a.key_tokens[j])} <sub>#${j}</sub> ${(v * 100).toFixed(0)}%</span>`).join('')
  }
  p.append(info, picked, wrap)
}

async function renderAttnTraceback(p) {
  const st = S.attnTrace
  const toks = S.run.outputs?.tokens || []
  const ctl = el('div', { className: 'row', style: 'margin:8px 0' })

  const qSel = el('select')
  qSel.appendChild(el('option', { value: -1, textContent: `Last token (#${Math.max(0, toks.length - 1)} "${toks[toks.length - 1] || ''}")` }))
  toks.forEach((t, i) => qSel.appendChild(el('option', { value: i, textContent: `#${i} "${t}"` })))
  qSel.value = st.query_token
  qSel.onchange = () => { st.query_token = +qSel.value }

  const kSel = el('select')
  kSel.appendChild(el('option', { value: '', textContent: 'All previous tokens (< query)' }))
  toks.forEach((t, i) => kSel.appendChild(el('option', { value: String(i), textContent: `#${i} "${t}"` })))
  kSel.value = st.key_token
  kSel.onchange = () => { st.key_token = kSel.value }

  const wInp = el('input', { type: 'text', value: st.target_word, placeholder: 'target word (optional)', style: 'width:130px' })
  wInp.oninput = () => { st.target_word = wInp.value }

  const runBtn = btn('Trace Previous Tokens', async () => {
    try {
      st.res = await POST('/api/ext/attn_traceback/trace', {
        run_id: S.run.id,
        query_token: +st.query_token,
        key_token: st.key_token === '' ? null : +st.key_token,
        target_word: st.target_word,
        top_k: st.top_k || 30,
      })
      renderTab()
    } catch (e) { toast(e.message, 'error') }
  }, 'primary')

  ctl.append(
    el('span', { className: 'small', textContent: 'Query token:' }), qSel,
    el('span', { className: 'small', textContent: 'Previous (key) token:' }), kSel,
    el('span', { className: 'small', textContent: 'Target word:' }), wInp,
    runBtn,
  )
  p.appendChild(ctl)

  if (!st.res || st.res.run_id !== S.run.id) {
    try {
      st.res = await POST('/api/ext/attn_traceback/trace', {
        run_id: S.run.id,
        query_token: +st.query_token,
        key_token: st.key_token === '' ? null : +st.key_token,
        target_word: st.target_word,
        top_k: st.top_k || 30,
      })
      st.res.run_id = S.run.id
    } catch (e) {
      p.appendChild(el('div', { className: 'empty', textContent: e.message }))
      return
    }
  }

  const r = st.res
  // Clickable previous-token strip
  const strip = el('div', { className: 'tokens compact', style: 'margin-bottom:8px' })
  ;(r.token_summary || []).forEach(tk => {
    const active = st.key_token !== '' && +st.key_token === tk.index
    const sp = el('span', {
      className: 'tok' + (active ? ' on' : ''),
      style: `cursor:pointer;background:${heat(Math.min(1, tk.mean_attn * 4))};border:${active ? '1px solid #fff' : ''}`,
      title: `Click to filter heads attending to #${tk.index} "${tk.piece}" (mean attn ${(tk.mean_attn * 100).toFixed(1)}%)`,
    }, `${esc(tk.piece)}<sub>#${tk.index}</sub>`)
    sp.onclick = async () => {
      st.key_token = active ? '' : String(tk.index)
      st.res = null
      renderTab()
    }
    strip.appendChild(sp)
  })
  p.appendChild(el('div', { className: 'small', textContent: `Query #${r.query_token} "${r.query_piece}" · Target direction: "${r.target_piece || ''}". Click any previous token below to isolate the exact (block, head) pairs attending to it:` }))
  p.appendChild(strip)

  // Table of top (block, head) pairs
  const tbl = el('table', { className: 'tbl' })
  tbl.innerHTML = `<thead><tr><th>Block · Head</th><th>Key Token</th><th>Attn Weight</th><th>Value Norm ‖α·vW_O‖</th><th>Target Logit Push</th><th>Action</th></tr></thead>`
  const tb = el('tbody')
  ;(r.top_heads || []).forEach(h => {
    const tr = el('tr')
    tr.innerHTML = `<td><b>${esc(h.label)}</b></td>
      <td><code>#${h.key_token}</code> "${esc(h.key_piece)}"</td>
      <td>${(h.attn_weight * 100).toFixed(1)}%</td>
      <td><b>${h.value_norm ?? '-'}</b></td>
      <td>${h.logit_push != null ? (h.logit_push > 0 ? '+' : '') + h.logit_push : '-'}</td>`
    const tdAct = el('td')
    tdAct.append(
      btn('Inspect Matrix', () => { S.attnPos = h.pos; S.attnHead = h.head; S.attnView = 'matrix'; renderTab() }, 'mini ghost'),
      btn('Knock out Head', () => {
        const bNode = flow.node(h.node)
        addIntervention(bNode)
        const iv = flow.nodes[flow.nodes.length - 1]
        if (iv) {
          flow.updateNodeData(iv.id, {
            label: `zero L${h.layer} H${h.head}`,
            hook: 'attn_heads',
            op: 'zero',
            heads: String(h.head),
            positions: String(h.query_token),
          })
          toast(`Added head knock-out on L${h.layer} H${h.head}`)
        }
      }, 'mini ghost'),
    )
    tr.appendChild(tdAct)
    tb.appendChild(tr)
  })
  tbl.appendChild(tb)
  p.appendChild(tbl)
}

registerTab({ id: 'attn', title: 'Attention', needs: 'run', render: renderAttn })
