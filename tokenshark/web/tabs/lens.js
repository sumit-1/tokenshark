import { S, el, flow, heat, registerTab, spinner } from '../core.js'
import { esc } from '../flow.js'
import { streamAnalysis } from '../ui/analysis.js'
import { renderTab, setTab } from '../ui/tabs.js'

// ---------- logit lens
S.through = 0

export function throughSelect(onChange) {
  const n = S.run.plan.schedule.length
  const sel = el('select', { title: 'How to turn a layer’s residual stream into words' })
  sel.appendChild(el('option', { value: 0, textContent: 'logit lens: norm + unembed directly' }))
  for (let k = 1; k <= Math.min(n, 6); k++) sel.appendChild(el('option', { value: k, textContent: `through the last ${k} block${k > 1 ? 's' : ''}, then unembed` }))
  if (n > 6) sel.appendChild(el('option', { value: n, textContent: `through all remaining blocks (= the real model)` }))
  sel.value = S.through
  sel.onchange = () => { S.through = +sel.value; onChange() }
  return sel
}

export async function renderLens(p) {
  const key = `${S.run.id}:${S.through}:${S.run.outputs?.live?.steps ?? ''}:${S.run.version ?? ''}`
  if (S.lensKey === key && S.lens) return drawLens(p, S.lens, false)
  p.innerHTML = `<div class="empty">${spinner('starting the logit lens…')}</div>`
  const seq = S.renderSeq, alive = () => S.renderSeq === seq && S.tab === 'lens'
  try {
    const L = await streamAnalysis(`/api/runs/${S.run.id}/lens?through=${S.through}`, (part, t) => { if (alive()) drawLens(p, part, true, t) }, alive)
    if (!L || !alive()) return
    S.lens = L; S.lensKey = key
    drawLens(p, L, false)
  } catch (e) { if (alive()) p.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
}

export function drawLens(p, L, partial, task) {
  const maxCols = 80, c0 = Math.max(0, L.tokens.length - maxCols)
  p.innerHTML = ''
  const bar = el('div', { className: 'row' })
  bar.append(el('span', { className: 'small', textContent: 'Decode each layer' }), throughSelect(() => renderTab()))
  p.appendChild(bar)
  let h = `<div class="small">Each cell: the token this layer would predict next. Colour = probability of the model's final choice. Hover for top-5; click a cell for its word cloud. Early layers rarely “speak vocabulary” yet — decoding through the last few blocks makes them readable. Showing the last ${Math.min(maxCols, L.tokens.length)} positions.</div><div class="lens-wrap"><table class="lens"><thead><tr><th></th>`
  const live = S.run.outputs?.live
  for (let t = c0; t < L.tokens.length; t++) h += `<th title="#${t}" class="${live && t >= L.complete ? 'inflight' : ''}">${esc(L.tokens[t])}</th>`
  h += '</tr></thead><tbody>'
  const ncol = L.tokens.length - c0
  if (partial) {
    const done = L.cells.filter(c => c).length
    p.appendChild(el('div', { className: 'small', innerHTML: `${spinner()} computing row ${done + 1} of ${L.layers.length}${task?.status === 'queued' ? ' — waiting for the model (another task is ahead: see Tasks)' : ''}; rows appear as they finish.` }))
  }
  L.layers.forEach((layer, li) => {
    if (!L.cells[li]) {                               // not computed yet
      h += `<tr class="computing"><th class="rowh" data-node="${esc(layer.node)}">${esc(layer.label)}</th><td colspan="${ncol}">${spinner()}</td></tr>`
      return
    }
    if (L.cells[li].every(c => !c)) return          // stepping: this block hasn't run yet — its row appears when it does
    const act = S.debug && (S.debug.active_node === layer.node)
    h += `<tr class="${act ? 'activerow' : ''}"><th class="rowh" data-node="${esc(layer.node)}">${esc(layer.label)}${act ? ' ◀ next' : ''}</th>`
    for (let t = c0; t < L.tokens.length; t++) {
      const c = L.cells[li][t]
      if (!c) { h += '<td class="pending" title="this layer has not processed this token yet — press Block ▸">·</td>'; continue }
      h += `<td data-li="${li}" data-t="${t}" style="background:${heat(c.p_final)}" title="${esc(c.top.map(x => `${x[0]}  ${(x[1] * 100).toFixed(1)}%`).join('\n'))}\nrank of final: ${c.rank_final}\nKL to final: ${c.kl}">${esc(c.top[0][0])}</td>`
    }
    h += '</tr>'
  })
  p.insertAdjacentHTML('beforeend', h + '</tbody></table></div>')
  p.insertAdjacentHTML('afterbegin', `<details class="howto"><summary>How to read the logit lens</summary>
    <ul><li><b>Columns</b> are the tokens of the text. A cell answers: <i>“if the model stopped after this layer, what would it say comes next after this column’s token?”</i></li>
    <li><b>Rows</b> go from the embeddings (top) to the last block (bottom). The bottom row is the model’s real prediction.</li>
    <li><b>Colour</b> = how much probability that layer already gives to the model’s final answer for that column. Dark blue = not yet, bright yellow = already decided. Read a column top to bottom to see <i>when</i> the answer appears.</li>
    <li>Typical pattern: early rows echo the input token itself, middle rows show related words, late rows lock in the answer. If the right word appears and then disappears lower down, those later blocks overrode it — check them with <i>Track a word</i> in the Word cloud tab.</li>
    <li>Hover a cell for its top-5 words, KL (how far it still is from the final distribution) and rank. Click it to open its word cloud. If early rows look like noise, set <i>Decode each layer</i> to go through the last 2–3 blocks.</li></ul></details>`)
  p.querySelectorAll('.rowh').forEach(th => th.onclick = () => { if (flow.node(th.dataset.node)) flow.select({ type: 'node', id: th.dataset.node }) })
  p.querySelectorAll('td[data-li]').forEach(td => td.onclick = () => {
    S.cloudPos = +td.dataset.t
    S.cloudLayers = [+td.dataset.li - (L.layers[0].node === 'embed' ? 1 : 0)]
    setTab('cloud')
  })
}

registerTab({ id: 'lens', title: 'Logit lens', needs: 'run', render: renderLens })
