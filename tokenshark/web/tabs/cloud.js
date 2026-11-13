import { S, btn, el, flow, registerTab, spinner } from '../core.js'
import { esc } from '../flow.js'
import { streamAnalysis } from '../ui/analysis.js'
import { renderTab } from '../ui/tabs.js'
import { throughSelect } from './lens.js'

// ---------- word clouds
S.cloudPos = -1;
 S.cloudLayers = null

S.cloudFollow = true

export async function renderCloud(p) {
  const o = S.run.outputs, n = S.run.plan.schedule.length, nodes = S.run.plan.schedule_nodes
  if (!S.cloudLayers) S.cloudLayers = [-1, ...Array(n).keys()]          // default: every layer
  const T = o.tokens.length
  const live = S.debug && o.live
  const follow = live && S.cloudFollow
  let pos = S.cloudPos < 0 ? Math.max(0, Math.min(T - 2, T + S.cloudPos)) : S.cloudPos
  let layers = S.cloudLayers
  if (follow) {
    // Stepping: the newest token, one cloud per block it has passed so far — newest block first.
    pos = T - 1
    const done = o.live.phase === 'block' ? o.live.next_pos : n
    layers = [-1, ...Array(done).keys()].reverse()
  }
  p.innerHTML = ''
  if (live) {
    const f = el('label', { className: 'field check' })
    const cb = el('input', { type: 'checkbox', checked: S.cloudFollow })
    cb.onchange = () => { S.cloudFollow = cb.checked; renderTab() }
    f.append(cb, el('span', { textContent: 'Follow the stepper: newest token, a cloud for each block it has passed (newest first). Press Block ▸ to add the next one.' }))
    p.appendChild(f)
  }
  p.appendChild(el('div', { className: 'small', textContent: 'Pick a position and one or more layers. Each cloud shows the words that layer would predict next, sized by probability; the combined cloud averages the chosen layers.' }))
  const toks = el('div', { className: 'tokens compact' })
  o.tokens.forEach((t, i) => { const s = el('span', { className: 'tok' + (i === pos ? ' on' : ''), textContent: t || '∅', title: `#${i}` }); s.onclick = () => { S.cloudPos = i; S.cloudFollow = false; renderTab() }; toks.appendChild(s) })
  p.appendChild(toks)
  const lay = el('div', { className: 'row' })
  lay.appendChild(el('span', { className: 'small', textContent: 'Layers:' }))
  const chip = (v, label) => {
    const on = layers.includes(v)
    const b = el('button', { className: 'mini' + (on ? ' on' : ''), textContent: label })
    b.onclick = () => { S.cloudFollow = false; S.cloudLayers = on ? layers.filter(x => x !== v) : [...layers, v].sort((a, b) => a - b); renderTab() }
    lay.appendChild(b)
  }
  chip(-1, 'Embed')
  for (let i = 0; i < n; i++) chip(i, flow.node(nodes[i])?.data.label?.replace('Block ', 'B') || `#${i}`)
  const quick = (label, arr) => lay.appendChild(btn(label, () => { S.cloudFollow = false; S.cloudLayers = arr; renderTab() }, 'mini ghost'))
  quick('first 3', [0, 1, 2].filter(i => i < n)); quick('middle 3', [Math.floor(n / 2) - 1, Math.floor(n / 2), Math.floor(n / 2) + 1].filter(i => i >= 0 && i < n))
  quick('last 3', [n - 3, n - 2, n - 1].filter(i => i >= 0)); quick('all', [...Array(n).keys()])
  p.appendChild(lay)
  const bar = el('div', { className: 'row' })
  bar.append(el('span', { className: 'small', textContent: 'Decode with' }), throughSelect(() => renderTab()))
  p.appendChild(bar)
  if (!layers.length) { p.appendChild(el('div', { className: 'empty', textContent: 'Pick at least one layer.' })); return }
  // ---- track an expected word through the layers ("which layer got it wrong?")
  const trk = el('div', { className: 'row trackrow' })
  const word = el('input', { value: S.trackWord || '', placeholder: 'expected word, e.g. Paris', style: 'width:180px' })
  const go = () => { S.trackWord = word.value.trim(); renderTab() }
  word.onkeydown = ev => { if (ev.key === 'Enter') go() }
  trk.append(el('b', { textContent: 'Track a word at this position:' }), word, btn('Track', go, 'mini'),
    S.trackWord ? btn('clear', () => { S.trackWord = ''; renderTab() }, 'mini ghost') : '')
  p.appendChild(trk)
  const trackBox = el('div')
  p.appendChild(trackBox)
  const showCombined = el('label', { className: 'field check' })
  const cbC = el('input', { type: 'checkbox', checked: !!S.cloudCombined })
  cbC.onchange = () => { S.cloudCombined = cbC.checked; renderTab() }
  showCombined.append(cbC, el('span', { textContent: 'also show the average of the chosen layers (mixes layers — not a real prediction)' }))
  if (!follow) p.appendChild(showCombined)

  const W = S.info?.weights, streamed = W?.policy === 'disk'
  const perBlock = W?.total_gb && S.info.n_layers ? W.total_gb / S.info.n_layers : 0
  const wait = streamed && S.through > 0
    ? `Decoding… each chosen layer is pushed through the last ${S.through} block(s). With weights streamed from disk those blocks are read once for this request (~${(perBlock * S.through).toFixed(1)} GB), then reused for every layer — expect a while on a large model.`
    : 'Decoding…'
  const wrap = el('div', { className: 'clouds' }, `<div class="empty">${spinner(wait)}</div>`)
  p.appendChild(wrap)
  const final = n - 1
  const req = [final, ...layers.filter(l => l !== final)]          // the model's own output always first
  const seq = S.renderSeq, alive = () => S.renderSeq === seq && S.tab === 'cloud'
  let head = null
  const draw = (c, partial, task) => {
    wrap.innerHTML = ''
    const nxt = c.next_token != null
      ? (c.next_generated ? `next token generated: “${esc(c.next_token)}”` : `next token in your text: “${esc(c.next_token)}” (you typed it; the model’s own choice was “${esc(c.model_choice ?? '?')}”)`)
      : 'last token — nothing follows yet'
    const chat = flow.nodes.find(n => n.data.kind === 'input_prompt')?.data.use_chat_template
    if (!head) {
      head = el('div', { className: 'cloudhead', innerHTML: `Position #${c.position}: predicting what comes after “<b>${esc(c.token)}</b>” · ${nxt}` +
        (chat ? `<div class="small">Chat template is on: your prompt is wrapped as a user message, so the model answers after the assistant header instead of continuing your sentence. For “The capital of France is → Paris”, untick <i>Use chat template</i> in the Prompt box, or analyse the token before “Paris” in the generated answer.</div>` : '') })
      p.insertBefore(head, trk)
    }
    const tag = l => {
      const bits = []
      if (l.next_rank != null) bits.push(`“${esc(c.next_token)}” is #${l.next_rank} (${(l.next_p * 100).toFixed(1)}%)`)
      if (!c.next_generated && l.choice_rank != null && c.model_choice !== c.next_token) bits.push(`model’s choice “${esc(c.model_choice)}” #${l.choice_rank}`)
      return bits.join(' · ')
    }
    c.layers.forEach((l, i) => {
      const isFinal = i === 0 && l.pos === final
      const title = isFinal ? `Model output (final layer ${l.label})` : l.label + (flow.node(l.node) ? ` · ${flow.node(l.node).data.label}` : '')
      if (isFinal && !layers.includes(final) && follow && o.live?.phase === 'block') return   // final not reached in this token yet
      wrap.appendChild(cloudCard(title, l.words, tag(l), c.next_token, isFinal))
    })
    ;(c.todo || []).forEach((l, i) => {
      if (l.pos === final && !layers.includes(final) && follow && o.live?.phase === 'block') return
      const title = l.pos === final ? `Model output (final layer ${l.label})` : l.label + (flow.node(l.node) ? ` · ${flow.node(l.node).data.label}` : '')
      wrap.appendChild(el('div', { className: 'cloud waiting' + (l.pos === final ? ' big' : '') },
        `<h4>${esc(title)}</h4><div class="words">${spinner(i === 0 && task?.status === 'running' ? 'decoding…' : task?.status === 'queued' ? 'waiting for the model (see Tasks)' : 'queued')}</div>`))
    })
    if (c.combined && S.cloudCombined && !follow) wrap.appendChild(cloudCard(`Average of ${c.layers.length} layers`, c.combined, 'a blend for overview, not something the model computes', c.next_token))
    if (c.pending?.length) wrap.appendChild(el('div', { className: 'cloud pendingcard' }, `<h4>Not reached yet</h4><div class="small">${c.pending.map(esc).join(', ')} — the token hasn't flowed through these blocks yet. Press Block ▸.</div>`))
  }
  try {
    const c = await streamAnalysis(`/api/runs/${S.run.id}/cloud?position=${pos}&layers=${req.join(',')}&through=${S.through}&top=40`,
      (part, t) => { if (alive()) draw(part, true, t) }, alive)
    if (c && alive()) draw(c, false)
  } catch (e) { wrap.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
  if (S.trackWord) {
    // Where does the word actually occur? The useful position is the token just before it.
    const w = S.trackWord.toLowerCase()
    const hits = o.tokens.map((t, i) => [t.trim().toLowerCase(), i]).filter(([t, i]) => t && w.startsWith(t) && t.length >= Math.min(3, w.length) && i > 0).map(([, i]) => i)
    if (hits.length && !hits.includes(pos + 1)) {
      const jump = el('div', { className: 'row' })
      jump.append(el('span', { className: 'small', textContent: `“${S.trackWord}” occurs in the text at ${hits.map(i => '#' + i).join(', ')}. To see which layers produced it, analyse the token just before it:` }))
      hits.slice(0, 5).forEach(i => jump.appendChild(btn(`after “${o.tokens[i - 1]}” (#${i - 1})`, () => { S.cloudPos = i - 1; S.cloudFollow = false; renderTab() }, 'mini')))
      trackBox.before(jump)
    }
    renderTrack(trackBox, pos)
  }
}

export async function renderTrack(box, pos) {
  box.innerHTML = `<div class="small">${spinner('tracking… (runs as a task: see Tasks)')}</div>`
  const seq = S.renderSeq, alive = () => S.renderSeq === seq && S.tab === 'cloud'
  const draw = (r, partial) => {
    if (!r || !r.rows?.length || !alive()) return
    const maxPush = Math.max(1e-6, ...r.rows.map(x => Math.abs(x.push)))
    const lbl = x => esc(x.pos < 0 ? 'Embed' : `${flow.node(x.node)?.data.label || x.label}`)
    const verdict = r.same
      ? `The model chose “${esc(r.choice)}” — the word you expected.`
      : `The model chose “${esc(r.choice)}”, not “${esc(r.target)}”. Look for where “${esc(r.target)}” loses its lead and which blocks push against it (red).`
    const firstRow = r.first_top_layer == null ? null : r.rows.find(x => x.pos === r.first_top_layer)
    const firstTop = firstRow == null ? `“${esc(r.target)}” is never the top word${partial ? ' so far' : ''}.` : `“${esc(r.target)}” first becomes the top word at ${lbl(firstRow)}.`
    let h = `<div class="trackbox"><div>${partial ? spinner(`${Math.round((r._progress || 0) * 100)}%`) + ' ' : ''}<b>${verdict}</b> ${firstTop} <span class="small">(tracking token “${esc(r.target)}”${r.pieces > 1 ? ` — “${esc(r.word)}” is split into ${r.pieces} tokens here, so only its first piece is tracked` : ''}${r.candidates.length > 1 ? `; other spelling: ${r.candidates.slice(1).map(x => `“${esc(x)}”`).join(', ')}` : ''})</span></div>
      <table class="grid track"><tr><th>Layer</th><th>would say</th><th>P(“${esc(r.target)}”)</th><th>rank</th>${r.same ? '' : `<th>P(“${esc(r.choice)}”)</th><th>rank</th>`}
      <th title="How much this block's own write moves the final logit of “${esc(r.target)}” relative to “${esc(r.versus)}” (direct logit attribution)">push: “${esc(r.target)}” vs “${esc(r.versus)}”${r.same ? ' (runner-up)' : ''}</th></tr>`
    r.rows.forEach(x => {
      const w = Math.abs(x.push) / maxPush * 50
      h += `<tr data-node="${esc(x.node)}"><td>${lbl(x)}</td><td>${esc(x.top)}</td>
        <td><span class="pbar"><i style="width:${x.target_p * 100}%"></i></span> ${(x.target_p * 100).toFixed(1)}%</td><td class="${x.target_rank === 1 ? 'ok' : ''}">#${x.target_rank}</td>
        ${r.same ? '' : `<td><span class="pbar alt"><i style="width:${x.choice_p * 100}%"></i></span> ${(x.choice_p * 100).toFixed(1)}%</td><td>#${x.choice_rank}</td>`}
        <td><span class="dla"><i class="${x.push >= 0 ? 'pos' : 'neg'}" style="width:${w}%;${x.push >= 0 ? 'left:50%' : `left:${50 - w}%`}"></i></span> ${x.push >= 0 ? '+' : ''}${x.push.toFixed(2)}</td></tr>`
    })
    h += `</table><div class="small">“would say” and P use the decoding chosen above. The push column is exact bookkeeping: all rows add up to the final logit gap between “${esc(r.target)}” and “${esc(r.versus)}”. Green = this block argues for “${esc(r.target)}”, red = for “${esc(r.versus)}”.</div></div>`
    box.innerHTML = h
    box.querySelectorAll('tr[data-node]').forEach(tr => tr.onclick = () => flow.node(tr.dataset.node) && flow.select({ type: 'node', id: tr.dataset.node }))
  }
  try {
    const r = await streamAnalysis(`/api/runs/${S.run.id}/track?word=${encodeURIComponent(S.trackWord)}&position=${pos}&through=${S.through}`, part => draw(part, true), alive)
    if (r) draw(r, false)
  } catch (e) { if (alive()) box.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
}

export function cloudCard(title, words, sub, actual, big) {
  const card = el('div', { className: 'cloud' + (big ? ' big' : '') }, `<h4>${esc(title)}</h4><div class="small">${sub}</div>`)
  const box = el('div', { className: 'words' })
  const mx = words[0]?.[1] || 1
  // interleave so the biggest words sit in the middle, like a word cloud
  const order = []; words.forEach((w, i) => (i % 2 ? order.push(w) : order.unshift(w)))
  order.forEach(([w, pr]) => {
    const r = Math.sqrt(pr / mx)
    const s = el('span', { textContent: w.trim() === '' ? '␣' : w, title: `${(pr * 100).toFixed(2)}%` })
    s.style.fontSize = `${10 + 26 * r}px`
    s.style.opacity = 0.45 + 0.55 * r
    s.style.color = w === actual ? 'var(--ok)' : `hsl(${210 - 170 * r}, 85%, ${60 + 10 * r}%)`
    if (w === actual) s.style.outline = '1px solid var(--ok)'
    box.appendChild(s)
  })
  card.appendChild(box)
  return card
}

registerTab({ id: 'cloud', title: 'Word cloud', needs: 'run', render: renderCloud })
