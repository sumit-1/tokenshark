// Experts tab — mixture-of-experts models. Read the router (which experts each token was sent to), read the experts
// (what each one wrote, in words), and test them (ban, force, silence, amplify, knock-out sweep).
// Also owns the expert panel in the inspector and the clicks on a block's expert cells in the flow.
import { GET, POST, S, btn, el, flow, hooks, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBox, aliveIn, blockNode, followTask, labelled, needRun, recColor, selectOf } from '../ui/labkit.js'
import { applyRunBadges, expertHue, routerEdits } from '../ui/run.js'
import { renderTab, setTab, tabAvailable } from '../ui/tabs.js'

S.moe = { node: null, expert: null, pos: null, token: -1, sweep: null, mode: 'ban', view: 'usage' }
const moeInfo = () => S.info?.moe
const hue = i => `hsl(${expertHue(i)},65%,45%)`
const chipX = (i, extra = '') => `<span class="xchip" data-x="${i}" style="border-color:${hue(i)}"${extra}>E${i}</span>`

// ---------------------------------------------------------------- acting on an expert (interceptor boxes)
function wire(nodeId, data, msg) {
  const n = addBox(nodeId, data)
  if (n) { applyRunBadges(); toast(msg + ' Press ▶ Run to see the effect.') }
}
export const expertActions = (nodeId, e) => [
  ['Ban', 'The router can never choose this expert; the next-best one takes its place.',
    () => wire(nodeId, { label: `ban expert ${e}`, hook: 'router', op: 'ban', experts: String(e), value: 0 }, `Expert ${e} is banned.`)],
  ['Only this', 'Every other expert is banned: all tokens go through this one.',
    () => wire(nodeId, { label: `only expert ${e}`, hook: 'router', op: 'only', experts: String(e), value: 0 }, `Only expert ${e} may be chosen.`)],
  ['Silence', 'It is still chosen (so nothing replaces it) but writes nothing.',
    () => wire(nodeId, { label: `silence expert ${e}`, hook: 'expert_out', op: 'zero', experts: String(e), value: 0 }, `Expert ${e} is silenced.`)],
  ['Amplify ×3', 'What it writes is multiplied by 3 wherever it is chosen.',
    () => wire(nodeId, { label: `expert ${e} × 3`, hook: 'expert_out', op: 'scale', experts: String(e), value: 3 }, `Expert ${e} is amplified.`)],
]
const actionRow = (nodeId, e) => { const r = el('span'); expertActions(nodeId, e).forEach(([t, tip, fn]) => { const b = btn(t, ev => { ev?.stopPropagation?.(); fn() }, 'mini'); b.title = tip; r.appendChild(b) }); return r }

function pick(nodeId, expert, open) {
  S.moe.node = nodeId; S.moe.expert = expert
  const pos = (S.run?.plan?.schedule_nodes || []).indexOf(nodeId)
  if (pos >= 0) S.moe.pos = pos
  applyRunBadges(); hooks.renderInspector?.()
  if (open) setTab('experts'); else if (S.tab === 'experts') renderTab()
}
hooks.badgeClick = (nodeId, d) => { if (d.pick === 'expert') pick(nodeId, +d.expert, false) }
hooks.refreshBadges = applyRunBadges

// ---------------------------------------------------------------- the inspector panel of an MoE block
hooks.expertPanel = (n, box) => {
  const E = n.data.moe, st = S.run?.outputs?.block_stats?.[n.id]?.moe, banned = routerEdits(n)
  const sec = el('div', { className: 'sec' })
  sec.appendChild(el('h4', { textContent: `Experts of this block (${E}, ${n.data.moe_k || moeInfo()?.top_k} per token)` }))
  const grid = el('div', { className: 'xgrid' })
  const mx = Math.max(...(st?.usage || [0]), 1e-9)
  for (let i = 0; i < E; i++) {
    const u = st ? st.usage[i] : 0
    const c = el('button', { className: `xcell${S.moe.node === n.id && S.moe.expert === i ? ' on' : ''}${banned.has(i) ? ' ban' : ''}`, textContent: i,
      title: `expert ${i}${st ? ` — used by ${(u * 100).toFixed(0)}% of the last run’s tokens` : ''}${banned.has(i) ? ' — banned' : ''}` })
    c.style.background = st ? `hsla(${expertHue(i)},70%,${14 + 40 * u / mx}%,${0.3 + 0.7 * u / mx})` : ''
    c.onclick = () => pick(n.id, i, false)
    grid.appendChild(c)
  }
  sec.appendChild(grid)
  if (st?.last?.length) sec.appendChild(el('div', { className: 'small' }, `Newest token went to ${st.last.map(([e, w]) => `<b>E${e}</b> (${w})`).join(', ')}.`))
  if (S.moe.node === n.id && S.moe.expert != null) {
    const e = S.moe.expert
    sec.appendChild(el('div', { className: 'small', style: 'margin-top:6px' }, `<b>Expert ${e}</b> — change what it does, then run again:`))
    sec.appendChild(actionRow(n.id, e))
    if (S.run?.outputs && tabAvailable('experts')) sec.appendChild(btn('What does it do? (Experts tab)', () => pick(n.id, e, true), 'mini'))
  } else sec.appendChild(el('div', { className: 'small', textContent: 'Click an expert to ban, force, silence or amplify it.' }))
  if (banned.size) sec.appendChild(el('div', { className: 'small bad', textContent: `Banned by interceptors: ${[...banned].sort((a, b) => a - b).slice(0, 40).join(', ')}${banned.size > 40 ? ' …' : ''}` }))
  box.appendChild(sec)
}

// ---------------------------------------------------------------- the tab
async function renderExperts(p) {
  if (needRun(p)) return
  const st = S.moe, alive = aliveIn('experts')
  p.appendChild(el('div', { className: 'small' }, `<b>Mixture of experts.</b> Each of these blocks has ${moeInfo().n_experts} small MLPs instead of one big one.
    A router scores every expert for every token and runs only the best ${moeInfo().top_k}; their outputs are added, weighted by the scores${moeInfo().norm_topk ? ' (rescaled to sum to 1)' : ''}${moeInfo().shared ? ', next to one shared expert that every token uses' : ''}.
    So for any token most of the block’s weights do nothing — and <i>which</i> experts ran is a fact you can read, not an estimate.`))
  const body = el('div', {}, spinner()); p.appendChild(body)
  let ov
  try { ov = await GET(`/api/runs/${S.run.id}/experts`) } catch (e) { if (alive()) body.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return }
  if (!alive()) return
  body.innerHTML = ''
  if (st.pos == null || !ov.blocks.find(b => b.pos === st.pos)) st.pos = ov.blocks[Math.floor(ov.blocks.length / 2)].pos
  const blk = ov.blocks.find(b => b.pos === st.pos)
  const T = ov.tokens.length
  const tsel = Math.min(T - 1, st.token < 0 ? T + st.token : st.token)

  // 1 · every block × every expert
  const s1 = el('div', { className: 'kvbox' }, `<h4>1 · Which experts did this run use? <span class="small">(row = block, column = expert, brighter = more tokens; click a cell)</span></h4>`)
  const view = selectOf([['usage', 'share of tokens that used the expert'], ['weight', 'average weight when it was used']], st.view)
  view.onchange = () => { st.view = view.value; renderTab() }
  s1.appendChild(labelled('Shade by', view))
  const hm = el('div', { className: 'xheat' })
  for (const b of ov.blocks) {
    const row = el('div', { className: `xrow${b.pos === st.pos ? ' on' : ''}` })
    const lab = el('span', { className: 'xlab', textContent: `#${b.pos} · L${b.layer}`, title: 'Click to open this block below' })
    lab.onclick = () => { st.pos = b.pos; st.node = b.node; renderTab() }
    row.appendChild(lab)
    const vals = b[st.view], mx = Math.max(...vals, 1e-9)
    const cells = el('span', { className: 'xcells' })
    vals.forEach((v, i) => {
      const c = el('i', { title: `block #${b.pos} · expert ${i} · used by ${(b.usage[i] * 100).toFixed(0)}% of tokens · mean weight ${b.weight[i]}` })
      c.style.background = v ? `hsla(${expertHue(i)},70%,${16 + 44 * v / mx}%,${0.3 + 0.7 * v / mx})` : '#0b0d10'
      if (b.pos === st.pos && i === st.expert) c.className = 'sel'
      c.onclick = () => pick(b.node, i, false)
      cells.appendChild(c)
    })
    row.appendChild(cells)
    row.appendChild(el('span', { className: 'small xstat', title: 'How spread out the router’s choice is, averaged over tokens. 0 bits = it always knows; the maximum = it has no preference.' },
      `${b.active}/${b.n_experts} used · router entropy ${b.entropy ?? '?'} of ${b.max_entropy} bits`))
    hm.appendChild(row)
  }
  s1.appendChild(hm)
  s1.appendChild(el('div', { className: 'small' }, `A model that spread tokens perfectly evenly would light every cell the same. Dark columns are experts this prompt never needed;
    try a prompt from another domain (code, numbers, another language) and compare which columns light up.`))
  body.appendChild(s1)

  // 2 · tokens coloured by their top expert, in the chosen block
  const s2 = el('div', { className: 'kvbox' }, `<h4>2 · Block #${blk.pos} (layer ${blk.layer}): each token, coloured by the expert that got the most weight <span class="small">(click a token)</span></h4>`)
  const strip = el('div', { className: 'tokens' })
  ov.tokens.forEach((t, i) => {
    const e = blk.top[i]
    const s = el('span', { className: `tok${i === tsel ? ' on' : ''}`, textContent: t || '∅', title: `#${i} → expert ${e} (weight ${blk.top_w[i]})` })
    if (e != null) s.style.background = `hsla(${expertHue(e)},60%,32%,${0.25 + 0.75 * (blk.top_w[i] ?? 0)})`
    s.onclick = () => { st.token = i; renderTab() }
    strip.appendChild(s)
  })
  s2.appendChild(strip)
  const det = el('div', {}, spinner()); s2.appendChild(det)
  body.appendChild(s2)
  const s3 = el('div', { className: 'kvbox' }); body.appendChild(s3)
  const s4 = el('div', { className: 'kvbox' }); body.appendChild(s4)
  renderSweep(s4, blk)

  let d
  try { d = await GET(`/api/runs/${S.run.id}/experts/block?pos=${st.pos}&token=${tsel}`) } catch (e) { if (alive()) det.innerHTML = `<div class="bad">${esc(e.message)}</div>`; return }
  if (!alive()) return
  det.innerHTML = ''
  // the router's decision at this token
  const probs = d.probs?.[d.token]
  const chosen = new Map(d.here.map(h => [h.index, h]))
  det.appendChild(el('div', { className: 'small', style: 'margin-top:8px' }, `<b>Token #${d.token} “${esc(d.piece)}”</b> — the router’s score for each of the ${d.n_experts} experts (softmax). The ${d.k} highest run; the rest are skipped.${d.k !== d.native_k ? ` <span class="bad">(k changed from the model’s ${d.native_k})</span>` : ''}`))
  if (probs) {
    const mx = Math.max(...probs, 1e-9)
    const bars = el('div', { className: 'xbars' })
    probs.forEach((v, i) => {
      const b = el('i', { className: chosen.has(i) ? 'on' : '', title: `expert ${i}: ${v}${chosen.has(i) ? ' — chosen' : ''}` })
      b.style.height = `${Math.max(2, v / mx * 100)}%`; b.style.background = chosen.has(i) ? hue(i) : ''
      b.onclick = () => pick(blockNode(st.pos), i, false)
      bars.appendChild(b)
    })
    det.appendChild(bars)
    if (d.n_experts <= 32) det.appendChild(el('div', { className: 'xbars lab' }, probs.map((v, i) => `<span>${i}</span>`).join('')))
  }
  const t = el('table', { className: 'grid' })
  t.innerHTML = `<tr><th>Chosen expert</th><th>Weight</th>${d.parts ? `<th>What it wrote here, as words</th><th title="How much this expert’s output raised the score of the model’s answer over the runner-up at this token (direct logit attribution)">Push for “${esc(d.answer ?? '')}” over “${esc(d.versus ?? '')}”</th>` : ''}<th>Change it</th></tr>`
  const mxPush = Math.max(...d.here.map(h => Math.abs(h.push ?? 0)), Math.abs(d.shared_here?.push ?? 0), 1e-9)
  for (const h of d.here) {
    const tr = el('tr')
    tr.innerHTML = `<td>${chipX(h.index)}</td><td><span class="pbar"><i style="width:${(h.weight * 100).toFixed(0)}%"></i></span> ${h.weight}</td>` +
      (d.parts ? `<td>${(h.promotes || []).map(w => `<span class="chip">${esc(w)}</span>`).join('')}</td>
      <td><span class="pbar${h.push < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(h.push ?? 0) / mxPush * 100).toFixed(0)}%"></i></span> ${h.push ?? ''}</td>` : '')
    const td = el('td'); td.appendChild(actionRow(blockNode(st.pos), h.index)); tr.appendChild(td)
    tr.querySelector('.xchip').onclick = () => pick(blockNode(st.pos), h.index, false)
    t.appendChild(tr)
  }
  if (d.shared_here) {
    const tr = el('tr')
    tr.innerHTML = `<td><span class="xchip">shared</span></td><td class="small">always on</td><td>${d.shared_here.promotes.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</td>
      <td><span class="pbar${d.shared_here.push < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(d.shared_here.push ?? 0) / mxPush * 100).toFixed(0)}%"></i></span> ${d.shared_here.push ?? ''}</td><td></td>`
    t.appendChild(tr)
  }
  det.appendChild(t)
  if (!d.parts) det.appendChild(el('div', { className: 'small', textContent: 'This run was too long to keep what each expert wrote (words and push are missing). Use a shorter prompt.' }))

  // 3 · the experts of this block
  s3.innerHTML = `<h4>3 · The experts of block #${d.pos}: what each handled in this run and what it wrote</h4>`
  const used = d.experts.filter(x => x.count).sort((a, b) => b.count - a.count)
  const tb = el('table', { className: 'grid' })
  tb.innerHTML = `<tr><th>Expert</th><th>Tokens routed to it</th><th>Mostly these tokens</th>${d.parts ? '<th>Pushes for</th><th>Pushes against</th>' : ''}<th>Change it</th></tr>`
  for (const x of used) {
    const tr = el('tr', { style: 'cursor:pointer' })
    if (x.index === st.expert) tr.className = 'on'
    tr.innerHTML = `<td>${chipX(x.index)}</td><td><span class="pbar"><i style="width:${(x.usage * 100).toFixed(0)}%"></i></span> ${x.count} <span class="small">(mean weight ${x.weight})</span></td>
      <td>${(x.tokens || []).map(([w, c]) => `<span class="chip">${esc(w)}${c > 1 ? `<sub>${c}</sub>` : ''}</span>`).join('')}</td>` +
      (d.parts ? `<td>${(x.promotes || []).slice(0, 6).map(w => `<span class="chip">${esc(w)}</span>`).join('')}</td><td>${(x.suppresses || []).slice(0, 4).map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</td>` : '')
    const td = el('td'); td.appendChild(actionRow(blockNode(st.pos), x.index)); tr.appendChild(td)
    tr.onclick = () => pick(blockNode(st.pos), x.index, false)
    tb.appendChild(tr)
  }
  s3.appendChild(tb)
  s3.appendChild(el('div', { className: 'small', textContent: `${d.n_experts - used.length} of ${d.n_experts} experts were not used by any token of this run.` }))
  const one = el('div'); s3.appendChild(one)
  if (st.expert != null && st.pos === d.pos) showExpert(one, st.pos, st.expert, alive)
}

async function showExpert(box, pos, index, alive) {
  box.innerHTML = spinner()
  let d
  try { d = await GET(`/api/runs/${S.run.id}/expert?pos=${pos}&index=${index}`) } catch (e) { box.innerHTML = `<div class="bad">${esc(e.message)}</div>`; return }
  if (!alive()) return
  const mx = Math.max(...d.weight, 1e-9)
  box.innerHTML = `<div class="kvbox"><h4>Expert ${d.index} of block #${d.pos} (layer ${d.layer}) — ${d.size.toLocaleString()} weights, chosen for ${d.count} of ${d.tokens.length} tokens</h4>
    <div class="small">Its weight at every token (0 = not chosen). Hover for the router’s score and rank.</div>
    <div class="tokens">${d.tokens.map((t, i) => `<span class="tok" title="#${i}: weight ${d.weight[i]}${d.prob ? ` · score ${d.prob[i]} · rank ${d.rank[i]} of ${d.n_experts}` : ''}" style="background:${recColor(d.weight[i] / mx)}">${esc(t || '∅')}</span>`).join('')}</div>
    <div class="kv" style="margin-top:6px">
      ${d.promotes ? `<span>writes (pushes for)</span><b>${d.promotes.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</b>
      <span>pushes against</span><b>${d.suppresses.slice(0, 8).map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</b>` : ''}
      <span title="Tokens whose embedding points the same way as this expert’s row of the router. A fair reading in the first blocks; deeper, the stream no longer looks like an embedding.">its router row attracts</span><b>${d.attracts.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</b>
      <span>fires together with</span><b>${d.co.map(([e, c]) => `${chipX(e)}<sub>${c}</sub>`).join(' ') || '—'}</b>
      <span title="Experts whose router rows point the same way: they compete for the same tokens.">competes with (similar router row)</span><b>${d.similar.map(([e, c]) => `${chipX(e)}<sub>${c}</sub>`).join(' ')}</b>
    </div></div>`
  box.querySelectorAll('.xchip[data-x]').forEach(c => { c.onclick = () => pick(blockNode(pos), +c.dataset.x, false) })
}

// 4 · knock-out sweep
function renderSweep(box, blk) {
  const st = S.moe
  box.innerHTML = `<h4>4 · Which experts does the answer depend on? <span class="small">(remove one at a time and measure)</span></h4>`
  const mode = selectOf([['ban', 'ban — the router picks a replacement (what deleting the expert would do)'], ['zero', 'silence — it is chosen but writes nothing (what its own output was worth)']], st.mode)
  mode.onchange = () => { st.mode = mode.value }
  const row = el('div', { className: 'row' })
  const go = async pos => {
    st.sweep = null; out.innerHTML = spinner('running the prompt once per expert…')
    const alive = aliveIn('experts')
    try {
      await followTask(await POST('/api/experts/sweep', { flow: flow.getGraph(), pos, mode: st.mode }), r => { st.sweep = r; if (alive()) drawSweep(out) }, alive)
    } catch (e) { if (alive()) out.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
  }
  row.append(labelled('How', mode), btn(`Sweep block #${blk.pos}`, () => go(blk.pos), 'primary'), btn('Sweep every MoE block', () => go(-1)),
    el('span', { className: 'small', textContent: 'Uses the prompt in the flow as it is now; only experts the prompt actually used are tested.' }))
  box.appendChild(row)
  const out = el('div'); box.appendChild(out)
  if (st.sweep) drawSweep(out)
}

function drawSweep(out) {
  const r = S.moe.sweep
  const rows = [...r.rows].sort((a, b) => a.d_margin - b.d_margin)
  const mx = Math.max(...rows.map(x => Math.abs(x.d_margin)), 1e-9)
  out.innerHTML = `<div class="small" style="margin-top:6px">Answer <b>“${esc(r.answer)}”</b> (p = ${r.p_answer}) leads <b>“${esc(r.versus)}”</b> by ${r.margin} logits. ${r.rows.length} of ${r.total} experts tested${r.rows.length < r.total ? ' …' : ''}.
    Most damaging first; a positive number means the answer got <i>stronger</i> without that expert.</div>`
  const t = el('table', { className: 'grid' })
  t.innerHTML = `<tr><th>Block</th><th>Expert</th><th>Share of prompt tokens</th><th>Weight at the last token</th><th>Change in the answer’s lead</th><th>p(answer)</th><th title="How far the whole prediction moved (KL divergence, nats)">KL</th><th>Top word now</th><th></th></tr>`
  for (const x of rows.slice(0, 60)) {
    const tr = el('tr')
    tr.innerHTML = `<td>#${x.pos} · L${x.layer}</td><td>${chipX(x.index)}</td><td>${(x.usage * 100).toFixed(0)}%</td><td>${x.weight_last || '—'}</td>
      <td><span class="pbar${x.d_margin < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(x.d_margin) / mx * 100).toFixed(0)}%"></i></span> ${x.d_margin}</td><td>${x.p_answer}</td><td>${x.kl}</td>
      <td>${x.flipped ? `<span class="chip bad">${esc(x.top)}</span>` : `<span class="chip">${esc(x.top)}</span>`}</td>`
    const td = el('td')
    td.append(btn('Wire it', () => wire(x.node, r.mode === 'ban' ? { label: `ban expert ${x.index}`, hook: 'router', op: 'ban', experts: String(x.index), value: 0 }
      : { label: `silence expert ${x.index}`, hook: 'expert_out', op: 'zero', experts: String(x.index), value: 0 }, `Expert ${x.index} of block #${x.pos} is ${r.mode === 'ban' ? 'banned' : 'silenced'}.`), 'mini'),
      btn('Look', () => pick(x.node, x.index, false), 'mini'))
    tr.appendChild(td); t.appendChild(tr)
  }
  out.appendChild(t)
  if (rows.length > 60) out.appendChild(el('div', { className: 'small', textContent: `… and ${rows.length - 60} more with smaller effects.` }))
}

registerTab({ id: 'experts', title: 'Experts', hint: 'Mixture of experts: which experts each token used, what they wrote, and what happens without them',
  when: info => !!info.moe, render: renderExperts })
