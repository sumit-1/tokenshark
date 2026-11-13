import { GET, S, btn, el, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBox, blockNode, blockSelect, labelled, needRun, numInput, recColor, selectOf } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'

S.neu = { pos: null, token: -1, by: 'token', k: 20, detail: null }

export async function renderNeurons(p) {
  if (needRun(p)) return
  const st = S.neu, o = S.run.outputs, nS = S.run.plan.schedule.length
  if (st.pos == null || st.pos >= nS) st.pos = Math.floor(nS / 2)
  p.appendChild(el('div', { className: 'small' }, `<b>Neuron hunting.</b> Each block’s MLP has ${S.info.d_ff ?? 'thousands of'} neurons: one number per token each.
    Pick a block and a token to see which fire hardest and which words each one pushes for. Then pin one to its maximum (or far beyond) and run again to see what it does to the text.`))
  const row = el('div', { className: 'row' })
  const by = selectOf([['token', 'strongest at the chosen token'], ['max', 'strongest anywhere in the run']], st.by); by.onchange = () => { st.by = by.value; renderTab() }
  row.append(labelled('Block', blockSelect(st.pos, v => { st.pos = v; st.detail = null; renderTab() })), labelled('Show', by))
  p.appendChild(row)
  const T = o.tokens.length
  const tsel = st.token < 0 ? T + st.token : st.token
  const strip = el('div', { className: 'tokens' })
  o.tokens.forEach((t, i) => { const s = el('span', { className: `tok${i === tsel ? ' on' : ''}`, textContent: t || '∅', title: `#${i}` }); s.onclick = () => { st.token = i; renderTab() }; strip.appendChild(s) })
  p.appendChild(strip)
  const out = el('div', {}, spinner()); p.appendChild(out)
  const seq = S.renderSeq
  let r
  try { r = await GET(`/api/runs/${S.run.id}/neurons?pos=${st.pos}&token=${st.token}&k=${st.k}&by=${st.by}`) }
  catch (e) { if (S.renderSeq === seq) out.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return }
  if (S.renderSeq !== seq) return
  out.innerHTML = ''
  const mx = Math.max(...r.neurons.map(n => Math.abs(n.act)), 1e-9)
  const t = el('table', { className: 'grid' })
  t.innerHTML = `<tr><th>Neuron</th><th>${st.by === 'token' ? `Activation at #${r.token} “${esc(r.piece)}”` : 'Peak activation'}</th><th>Range in this run</th><th>Pushes for</th><th>Pushes against</th><th>Pin it</th></tr>`
  const clampBox = (i, to, value) => {
    const n = addBox(blockNode(st.pos), { label: `neuron ${i} → ${to}${value !== 1 ? ' ×' + value : ''}`, hook: 'mlp_neurons', op: 'clamp', neurons: String(i), clamp: to, value, ref: S.run.id })
    if (n) toast(`Neuron ${i} of block #${st.pos} is pinned to ${value !== 1 ? value + ' × ' : ''}its ${to}. Press ▶ Run.`)
  }
  r.neurons.forEach(n => {
    const tr = el('tr', { style: 'cursor:pointer' })
    tr.innerHTML = `<td><b>${n.index}</b></td><td><span class="pbar${n.act < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(n.act) / mx * 100).toFixed(0)}%"></i></span> ${n.act}${st.by === 'max' ? ` <span class="small">at “${esc(n.peak_piece)}”</span>` : ''}</td>
      <td class="small">${n.min ?? '?'} … ${n.max ?? '?'}</td>
      <td>${n.promotes.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</td><td>${n.suppresses.slice(0, 5).map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</td>`
    const td = el('td')
    td.append(btn('max', ev => { ev.stopPropagation(); clampBox(n.index, 'max', 1) }, 'mini'), btn('max ×5', ev => { ev.stopPropagation(); clampBox(n.index, 'max', 5) }, 'mini'),
      btn('min', ev => { ev.stopPropagation(); clampBox(n.index, 'min', 1) }, 'mini'),
      btn('0', ev => { ev.stopPropagation(); const b = addBox(blockNode(st.pos), { label: `neuron ${n.index} → 0`, hook: 'mlp_neurons', op: 'zero', neurons: String(n.index) }); if (b) toast(`Neuron ${n.index} is switched off. Press ▶ Run.`) }, 'mini'))
    tr.appendChild(td)
    tr.onclick = () => { st.detail = n.index; showNeuron(det, st.pos, n.index) }
    t.appendChild(tr)
  })
  out.appendChild(t)
  const jump = el('div', { className: 'row' })
  const idx = numInput(st.detail ?? '', 80, { min: 0, placeholder: 'index' })
  jump.append(el('span', { className: 'small', textContent: `Any neuron (0…${r.n_neurons - 1}):` }), idx, btn('Look at it', () => { if (idx.value !== '') { st.detail = +idx.value; showNeuron(det, st.pos, +idx.value) } }, 'mini'))
  out.appendChild(jump)
  const det = el('div'); out.appendChild(det)
  if (st.detail != null) showNeuron(det, st.pos, st.detail)
}

export async function showNeuron(box, pos, index) {
  box.innerHTML = spinner()
  let d
  try { d = await GET(`/api/runs/${S.run.id}/neuron?pos=${pos}&index=${index}`) } catch (e) { box.innerHTML = `<div class="bad">${esc(e.message)}</div>`; return }
  const mx = Math.max(...d.acts.map(Math.abs), 1e-9)
  box.innerHTML = `<div class="kvbox"><h4>Neuron ${d.index} · block #${d.pos} (layer ${d.layer}) — activation at every token (green = positive, red = negative)</h4>
    <div class="tokens">${d.tokens.map((t, i) => `<span class="tok" title="#${i}: ${d.acts[i]}" style="background:${recColor(d.acts[i] / mx)}">${esc(t || '∅')}</span>`).join('')}</div>
    <div class="small">range ${d.min} … ${d.max} · pushes for ${d.promotes.map(w => `<span class="chip">${esc(w)}</span>`).join('')} · against ${d.suppresses.map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</div></div>`
}

registerTab({ id: 'neurons', title: 'Neurons', hint: 'Find and pin single MLP neurons', render: renderNeurons })
