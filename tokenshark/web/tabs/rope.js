import { GET, S, el, heat, registerTab, spinner } from '../core.js'
import { esc } from '../flow.js'

// ---------- RoPE clocks: one small clock per rotary pair of a key (or query) vector, with a position scrubber.
// RoPE treats dims (j, j + half) of a head as one 2-D point and turns it by position x speed_j. Pure mode shows the
// bare rotation (every hand length 1). Real mode starts each hand at the token's own (pre-RoPE) pair, so its length
// shows how much of that token's vector lives at that speed.

S.rp = S.rp || { mode: 'real', kind: 'k', block: null, head: 0, token: null, pos: null, max: 1000 }

const fmtN = v => v >= 1e6 ? `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(v >= 1e4 ? 0 : 1)}k` : v >= 10 ? v.toFixed(0) : v.toFixed(1)
const css = name => getComputedStyle(document.body).getPropertyValue(name).trim() || '#888'

async function load(st) {
  const key = `${S.run.id}|${S.run.version ?? ''}|${st.block}|${st.token}|${st.head}|${st.kind}`
  if (S.rpData?.key === key) return S.rpData.data
  const data = await GET(`/api/runs/${S.run.id}/rope_pairs?block=${st.block}&token=${st.token}&head=${st.head}&kind=${st.kind}`)
  S.rpData = { key, data }
  return data
}

export async function renderRope(p) {
  const st = S.rp
  const o = S.run.outputs
  const n = S.run.plan.schedule.length
  if (st.runId !== S.run.id) {                    // a new run: start on the newest token of a middle block
    st.runId = S.run.id
    st.token = Math.max(0, (o.tokens?.length || 1) - 2)
    st.block = Math.floor(n / 2)
    st.pos = null
  }
  p.innerHTML = spinner('reading the vector…')
  let d
  try { d = await load(st) } catch (e) {
    if (st.kind === 'q' && /not kept/.test(e.message)) { st.kind = 'k'; return renderRope(p) }
    if (/no RoPE keys/.test(e.message) && st.block > 0) { st.block -= 1; return renderRope(p) }
    p.innerHTML = `<div class="small bad">${esc(e.message)}</div>`
    return
  }
  if (st.head >= d.n_heads) { st.head = 0; return renderRope(p) }
  if (st.pos == null) st.pos = d.eff_position
  if (d.max_context && st.max > d.max_context) st.max = d.max_context
  p.innerHTML = ''
  p.appendChild(el('div', { className: 'sechead', textContent: 'RoPE clocks' }))
  p.appendChild(el('div', { className: 'small', style: 'margin-bottom:6px;line-height:1.5' },
    `RoPE cuts each ${d.head_dim}-number head into <b>${d.n_pairs} pairs</b> (dims <i>j</i> and <i>j</i> + ${d.n_pairs}). Each pair is a 2-D point - a clock hand - and RoPE turns it by ` +
    `<i>position × speed<sub>j</sub></i>. Clock #0 turns 1 radian (57°) per token; each next one a bit slower, down to the last, which needs ` +
    `~${fmtN(2 * Math.PI / Math.max(1e-12, d.pairs.filter(x => x.freq > 0).at(-1)?.freq || 1))} tokens for one turn. Drag the slider to move the token along the text.`))

  // ---- controls
  const row = el('div', { className: 'row', style: 'flex-wrap:wrap;gap:8px;align-items:center' })
  const seg = (opts, cur, set) => {
    const g = el('span', { className: 'seg' })
    opts.forEach(([v, label, tip]) => {
      const b = el('button', { className: 'mini' + (v === cur ? ' primary' : ''), textContent: label, title: tip || '' })
      b.onclick = () => { set(v); renderRope(p) }
      g.appendChild(b)
    })
    return g
  }
  row.append(
    seg([['pure', 'Pure RoPE', 'every hand has length 1: just the rotation'], ['real', 'Real vector', 'hands start at the token’s own numbers: length = how much of the vector sits in that pair']], st.mode, v => { st.mode = v }),
    seg([['k', 'Key', 'the key this token stored in the KV cache'], ['q', 'Query', 'the query this token asked with']], st.kind, v => { st.kind = v; st.head = 0 }))
  const bSel = el('select', { title: 'block (schedule position · layer)' })
  for (let b = 0; b < d.n_blocks; b++) {
    if (d.linear_blocks.includes(b)) continue
    bSel.appendChild(el('option', { value: b, textContent: `Block #${b} · L${S.run.plan.schedule[b]}`, selected: b === st.block }))
  }
  bSel.onchange = () => { st.block = +bSel.value; renderRope(p) }
  const hSel = el('select', { title: d.kind === 'q' ? 'query head' : 'key/value head' })
  for (let h = 0; h < d.n_heads; h++) hSel.appendChild(el('option', { value: h, textContent: `${d.kind === 'q' ? 'Q' : 'KV'} head ${h}`, selected: h === st.head }))
  hSel.onchange = () => { st.head = +hSel.value; renderRope(p) }
  row.append(bSel, hSel)
  p.appendChild(row)

  // ---- token strip: pick whose vector to look at
  const strip = el('div', { className: 'tokens', style: 'margin:6px 0;max-height:72px;overflow:auto' })
  d.tokens.forEach((t, i) => {
    const s = el('span', { className: 'tok' + (i === st.token ? ' on' : ''), title: `#${i}`, textContent: t || '∅' })
    s.onclick = () => { st.token = i; st.pos = null; renderRope(p) }
    strip.appendChild(s)
  })
  p.appendChild(strip)

  // ---- the clock grid
  const cols = Math.min(16, d.n_pairs)                 // 16 x 8 for a 256-dim head, 16 x 2 for a 64-dim one
  const wrap = el('div', { className: 'rpgrid' })
  const canvas = el('canvas', { className: 'rpcanvas' })
  wrap.appendChild(canvas)
  p.appendChild(wrap)
  const info = el('div', { className: 'small', style: 'margin:4px 0' })
  const tip = el('div', { className: 'small rptip' })

  // ---- scrubber
  const scr = el('div', { className: 'row', style: 'align-items:center;gap:8px;flex-wrap:wrap' })
  const range = el('input', { type: 'range', min: 0, max: st.max, step: 1, value: Math.min(st.pos, st.max), style: 'flex:1 1 260px' })
  const num = el('input', { type: 'number', value: st.pos, step: 'any', style: 'width:96px' })
  const maxSel = el('select', { title: 'slider range' })
  const maxes = [100, 1000, 10000, 100000, d.max_context].filter((v, i, a) => v && a.indexOf(v) === i && (!d.max_context || v <= d.max_context))
  maxes.forEach(v => maxSel.appendChild(el('option', { value: v, textContent: `0 … ${fmtN(v)}${v === d.max_context ? ' (max context)' : ''}`, selected: v === st.max })))
  const play = el('button', { className: 'mini', textContent: '▶ Play' })
  const back = el('button', { className: 'mini ghost', textContent: `Back to #${d.token}’s own position (${d.eff_position})` })
  scr.append(el('b', { className: 'small', textContent: 'Position' }), range, num, maxSel, play, back)
  p.append(scr, info, tip)

  const pairs = d.pairs
  const maxMag = Math.max(...pairs.map(x => x.mag), 1e-9)
  const total = pairs.reduce((s, x) => s + x.mag ** 2, 0) + d.unrotated_norm ** 2
  const sorted = [...pairs].sort((a, b) => b.mag - a.mag)
  const top8 = sorted.slice(0, 8).reduce((s, x) => s + x.mag ** 2, 0) / Math.max(total, 1e-12)
  const halfN = Math.floor(pairs.length / 2)
  const fastShare = pairs.slice(0, halfN).reduce((s, x) => s + x.mag ** 2, 0) / Math.max(total, 1e-12)
  let cell = 40

  function draw() {
    const W = Math.max(240, Math.min(wrap.clientWidth || 640, cols * 68))
    cell = W / cols
    const rows = Math.ceil(pairs.length / cols)
    const dpr = window.devicePixelRatio || 1
    canvas.width = W * dpr; canvas.height = rows * cell * dpr
    canvas.style.width = W + 'px'; canvas.style.height = rows * cell + 'px'
    const c = canvas.getContext('2d')
    c.setTransform(dpr, 0, 0, dpr, 0, 0)
    c.clearRect(0, 0, W, rows * cell)
    const line = css('--line'), muted = css('--muted'), text = css('--text')
    const r = cell * 0.38
    pairs.forEach((x, j) => {
      const cx = (j % cols + 0.5) * cell, cy = (Math.floor(j / cols) + 0.5) * cell - cell * 0.06
      const still = x.freq === 0
      c.lineWidth = 1
      c.strokeStyle = line
      c.setLineDash(still ? [2, 3] : [])
      c.beginPath(); c.arc(cx, cy, r, 0, 2 * Math.PI); c.stroke()
      c.setLineDash([])
      c.beginPath(); c.moveTo(cx, cy - r); c.lineTo(cx, cy - r + 3); c.stroke()            // 12 o'clock tick
      const base = st.mode === 'real' ? Math.atan2(x.im, x.re) : 0
      const a = base + st.pos * x.freq
      const f = st.mode === 'real' ? x.mag / maxMag : 1
      const len = Math.max(f * r * 0.92, st.mode === 'real' ? 1.2 : 0)
      c.strokeStyle = st.mode === 'real' ? heat(0.25 + 0.75 * f) : text
      c.lineWidth = st.mode === 'real' ? 1 + 2 * f : 1.6
      c.lineCap = 'round'
      c.beginPath(); c.moveTo(cx, cy); c.lineTo(cx + len * Math.sin(a), cy - len * Math.cos(a)); c.stroke()
      c.fillStyle = muted
      c.font = `${Math.max(8, Math.min(10, cell * 0.2))}px ui-monospace, Consolas, monospace`
      c.textAlign = 'center'
      c.fillText(still ? '∞' : fmtN(2 * Math.PI / x.freq), cx, cy + r + Math.max(8, cell * 0.17))
    })
    const fast = pairs[0]
    info.innerHTML = `<b>${d.kind === 'q' ? 'Query' : 'Key'}</b> of #${d.token} <b class="tokchip">${esc(d.piece || '∅')}</b> · block #${d.block} (layer ${d.layer}) · ${d.kind === 'q' ? 'Q' : 'KV'} head ${d.head} · ` +
      `at position <b>${(+st.pos).toLocaleString()}</b> clock #0 has made ${fmtN(st.pos * fast.freq / (2 * Math.PI))} turns. ` +
      `θ = ${d.theta.toLocaleString()} (${esc(d.rope_type)} RoPE). Number under a clock: tokens per full turn.` +
      (st.mode === 'real' ? `<br>Hand length and colour = size of that pair in this vector (‖x‖ = ${d.norm.toFixed(2)}). The 8 biggest clocks hold <b>${(top8 * 100).toFixed(0)}%</b> of its energy; ` +
        `the fast half (#0-${halfN - 1}) holds ${(fastShare * 100).toFixed(0)}%, the slow half ${(100 * (1 - fastShare) - 100 * d.unrotated_norm ** 2 / Math.max(total, 1e-12)).toFixed(0)}%` +
        (d.unrotated_norm > 0 ? `, dims that never turn ${(100 * d.unrotated_norm ** 2 / Math.max(total, 1e-12)).toFixed(0)}%` : '') + '.' : '')
  }

  const setPos = v => { st.pos = Math.max(0, +v || 0); range.value = Math.min(st.pos, st.max); num.value = st.pos; draw() }
  range.oninput = () => setPos(range.value)
  num.onchange = () => setPos(num.value)
  maxSel.onchange = () => { st.max = +maxSel.value; range.max = st.max; setPos(Math.min(st.pos, st.max)) }
  back.onclick = () => setPos(d.eff_position)
  let raf = null
  play.onclick = () => {
    if (raf) { cancelAnimationFrame(raf); raf = null; play.textContent = '▶ Play'; return }
    play.textContent = '■ Stop'
    const step = Math.max(0.05, st.max / 900)
    const tick = () => {
      if (!canvas.isConnected) { raf = null; return }
      setPos(st.pos + step >= st.max ? 0 : st.pos + step)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
  }
  canvas.onmousemove = ev => {
    const b = canvas.getBoundingClientRect()
    const j = Math.floor((ev.clientY - b.top) / cell) * cols + Math.floor((ev.clientX - b.left) / cell)
    const x = pairs[j]
    if (!x) { tip.textContent = ''; return }
    const ang = ((st.mode === 'real' ? Math.atan2(x.im, x.re) : 0) + st.pos * x.freq) * 180 / Math.PI
    tip.innerHTML = `Clock #${j}: dims ${j} and ${j + d.n_pairs} · speed ${x.freq.toPrecision(3)} rad/token` +
      (x.freq ? ` · one turn every ${fmtN(2 * Math.PI / x.freq)} tokens · ${fmtN(st.pos * x.freq / (2 * Math.PI))} turns so far` : ' · never turns (no position in it)') +
      ` · angle ${(((ang % 360) + 360) % 360).toFixed(0)}°` + (st.mode === 'real' ? ` · size ${x.mag.toFixed(3)} (${(100 * x.mag ** 2 / Math.max(total, 1e-12)).toFixed(1)}% of the energy)` : '')
  }
  canvas.onmouseleave = () => { tip.textContent = '' }
  requestAnimationFrame(draw)
  new ResizeObserver(() => { if (canvas.isConnected) draw() }).observe(wrap)
}

registerTab({ id: 'rope', title: 'RoPE clocks', needs: 'run', hint: 'Watch RoPE turn one token’s key or query: one clock per rotary pair, with a position scrubber', render: renderRope })
