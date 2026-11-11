import { GET, S, btn, el, flow, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { needRun } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'

S.smp = { pos: null, temp: 1, topk: 0, topp: 1, data: null, key: null, draws: null }

export async function renderSampler(p) {
  if (needRun(p)) return
  const st = S.smp, o = S.run.outputs
  const gen = o.segments.find(s => s.origin === 'generated')
  const T = o.live?.n_complete ?? o.tokens.length
  if (st.pos == null || st.pos >= T) st.pos = Math.max(0, Math.min(T - 1, (gen?.start ?? T) - 1))
  p.appendChild(el('div', { className: 'small' }, `<b>Sampler playground.</b> The model outputs a score for every word; the sampler turns scores into one choice.
    Pick a position, then move the sliders: <b>temperature</b> flattens or sharpens the odds, <b>top-k</b> keeps the k best, <b>top-p</b> keeps the smallest set that covers p of the probability. Nothing is re-run — this is the same prediction, read differently.`))
  const strip = el('div', { className: 'tokens' })
  o.tokens.slice(0, T).forEach((t, i) => { const s = el('span', { className: `tok${i === st.pos ? ' on' : ''}`, textContent: t || '∅', title: `after token #${i}` }); s.onclick = () => { st.pos = i; st.draws = null; renderTab() }; strip.appendChild(s) })
  p.appendChild(strip)
  const key = `${S.run.id}:${S.run.version}:${st.pos}`
  const body = el('div', {}, spinner()); p.appendChild(body)
  if (st.key !== key) {
    const seq = S.renderSeq
    try { st.data = await GET(`/api/runs/${S.run.id}/logits?position=${st.pos}&top=1000`); st.key = key }
    catch (e) { if (S.renderSeq === seq) body.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return }
    if (S.renderSeq !== seq) return
  }
  const d = st.data
  body.innerHTML = ''
  const ctl = el('div', { className: 'row' })
  const mk = (labelFn, min, max, step, key2) => {
    const s = el('input', { type: 'range', min, max, step, value: st[key2], style: 'width:170px' }), l = el('span', { className: 'small' })
    const show = () => { l.innerHTML = labelFn(+s.value) }
    s.oninput = () => { st[key2] = +s.value; st.draws = null; show(); draw() }
    show(); const w = el('span', { className: 'slider' }); w.append(l, s); return w
  }
  ctl.append(mk(v => `<b>Temperature ${v.toFixed(2)}</b>${v === 0 ? ' (greedy)' : ''}`, 0, 2.5, 0.05, 'temp'), mk(v => `<b>Top-k ${v || 'off'}</b>`, 0, 100, 1, 'topk'), mk(v => `<b>Top-p ${v.toFixed(2)}</b>`, 0.05, 1, 0.05, 'topp'),
    btn('Draw 200 samples', () => { st.draws = sampleMany(dist(), 200); draw() }), btn('Use these in the Sampler box', () => {
      const sn = flow.nodes.find(n => n.data.kind === 'sampler'); if (!sn) return
      flow.updateNodeData(sn.id, { temperature: st.temp, top_k: st.topk, top_p: st.topp }); toast('Sampler box updated. Press ▶ Run to generate with these settings.')
    }))
  const info = el('div', { className: 'small' }), bars = el('div', { className: 'alts wide' })
  body.append(ctl, info, bars)
  const dist = () => {
    const rows = d.rows
    let pr
    if (st.temp <= 0) pr = rows.map((_, i) => i === 0 ? 1 : 0)
    else { const mx = rows[0][2]; pr = rows.map(r => Math.exp((r[2] - mx) / st.temp)); const z = pr.reduce((a, b) => a + b, 0); pr = pr.map(v => v / z) }
    if (st.topk > 0) pr = pr.map((v, i) => i < st.topk ? v : 0)
    if (st.topp < 1) { let c = 0; const tot = pr.reduce((a, b) => a + b, 0); pr = pr.map(v => { const keep = c < st.topp * tot; c += v; return keep ? v : 0 }) }
    const z = pr.reduce((a, b) => a + b, 0)
    return pr.map(v => v / z)
  }
  function draw() {
    const pr = dist()
    const H = -pr.reduce((a, v) => a + (v > 0 ? v * Math.log2(v) : 0), 0)
    const live = pr.filter(v => v > 0).length
    info.innerHTML = `After “${esc(d.token)}”${d.next_piece != null ? ` the run continued with <span class="chip">${esc(d.next_piece)}</span>` : ''} ·
      <b>${live}</b> token${live === 1 ? '' : 's'} can still be chosen · entropy <b>${H.toFixed(2)} bits</b> (like a fair choice between ${Math.pow(2, H).toFixed(1)} words)
      <span class="small">· shown: the ${d.rows.length} highest-scoring of ${d.vocab} tokens (${(d.covered * 100).toFixed(1)}% of the probability at temperature 1)</span>`
    bars.innerHTML = ''
    const counts = st.draws
    d.rows.slice(0, 24).forEach((r, i) => {
      const b = el('div', { className: 'alt' + (r[1] === d.next_id ? ' chosen' : '') + (pr[i] === 0 ? ' cut' : '') },
        `<span>${esc(r[0])}</span><i style="width:${(pr[i] * 100).toFixed(1)}%"></i><b>${(pr[i] * 100).toFixed(1)}%${counts ? ` <span class="small">· ${counts[i] || 0}×</span>` : ''}</b>`)
      bars.appendChild(b)
    })
  }
  draw()
}

export function sampleMany(pr, n) {
  const counts = {}
  for (let k = 0; k < n; k++) { let u = Math.random(), i = 0; while (i < pr.length - 1 && (u -= pr[i]) > 0) i++; counts[i] = (counts[i] || 0) + 1 }
  return counts
}

registerTab({ id: 'sampler', title: 'Sampler', hint: 'Temperature, top-k and top-p on a real prediction', render: renderSampler })
