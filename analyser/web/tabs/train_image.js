// Train tab section: a tiny image model (flow matching on coloured shapes), trained from scratch.
import { POST, btn, el, toast } from '../core.js'
import { labelled, numInput, selectOf } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'
import { jobView, registerTrainSection } from './train.js'
import { esc } from '../flow.js'

// ---- a tiny image model: flow matching on coloured shapes
function imageTrainer(st, running) {
  const f = st.iform, j = st.image, s = st.isample
  const c = el('div')
  c.appendChild(el('div', { className: 'small' }, `<b>Train a tiny image model from scratch.</b> The pictures are 16×16 coloured shapes; the “prompt” is two words, a colour and a shape.
    It is trained exactly like the big image models (flow matching): mix a picture with noise, <code>x = (1−σ)·picture + σ·noise</code>, and teach the model to output <code>noise − picture</code>.
    To draw, start from noise and step the other way. Takes about a minute. One combination — the <b>yellow triangle</b> — is never shown in training.
    <span class="small">Training a LoRA on a full-size image model needs a strong GPU and hours; this shows the same recipe at a size you can watch.</span>`))
  const g = el('div', { className: 'formgrid' })
  const steps = el('input', { type: 'number', value: f.steps, style: 'width:80px' }); steps.oninput = () => { f.steps = +steps.value }
  g.append(labelled('Steps', steps), btn(running(j) ? 'Training…' : 'Start training', async () => {
    try { await POST('/api/train/image', { steps: f.steps }); st.isample.result = null; renderTab() } catch (e) { toast(e.message, 'error') }
  }, 'primary'))
  g.lastChild.disabled = running(j)
  if (running(j)) g.append(btn('■ Stop (keeps what it learned)', async () => { await POST('/api/train/image/stop'); renderTab() }))
  c.appendChild(g)
  if (!j || j.status === 'none') return c
  c.appendChild(jobView(j, () => null))
  if (!j.ready) return c
  // sampler: one picture step by step, with the same controls the real models have
  const box = el('div', { className: 'sec' }, `<h4>Draw with it</h4><div class="small">Top row: what the model is given at each step. Bottom row: its guess of the finished picture. Try the combination it never saw, then move <b>guidance</b>: 0 ignores the prompt, 1 is the plain model, higher pushes harder towards the prompt.</div>`)
  const col = selectOf(j.colors, s.color), shp = selectOf(j.shapes, s.shape)
  const n = numInput(s.steps, 60, { min: 1, max: 60 }), seed = numInput(s.seed, 60)
  const gl = el('span', { className: 'small' }), gs = el('input', { type: 'range', min: 0, max: 8, step: 0.25, value: s.guidance, style: 'width:160px' })
  const showG = () => { gl.innerHTML = `<b>Guidance ${(+gs.value).toFixed(2)}</b>` }
  const out = el('div')
  const draw = async () => {
    Object.assign(s, { color: col.value, shape: shp.value, steps: +n.value || 12, guidance: +gs.value, seed: +seed.value || 0 })
    try { s.result = await POST('/api/train/image/sample', { ...s, result: undefined, grid: true }); show() } catch (e) { out.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
  }
  const show = () => {
    const r = s.result; if (!r) return
    out.innerHTML = `<div class="imggrid"><figure><img src="${r.strip}"><figcaption>${esc(r.color)} ${esc(r.shape)}${r.held_out ? ' — <b>never seen in training</b>' : ''} · ${r.steps} steps, shown: ${r.steps_shown.join(', ')} · noise level σ from ${r.sigmas[0]} to ${r.sigmas[r.sigmas.length - 1]}</figcaption></figure>
      <figure><img src="${r.final}"><figcaption>the finished picture</figcaption></figure>
      <figure><img src="${r.grid}"><figcaption>every colour (columns: ${j.colors.join(', ')}) × shape (rows: ${j.shapes.join(', ')}) at this guidance and seed</figcaption></figure></div>`
  }
  col.onchange = shp.onchange = n.onchange = seed.onchange = draw
  gs.oninput = showG; gs.onchange = draw; showG()
  const row = el('div', { className: 'row' })
  const gw = el('span', { className: 'slider' }); gw.append(gl, gs)
  row.append(labelled('Colour', col), labelled('Shape', shp), labelled('Steps', n), labelled('Seed', seed), gw, btn('Draw', draw, 'primary'),
    btn('The one it never saw', () => { col.value = j.held_out[0]; shp.value = j.held_out[1]; draw() }))
  box.append(row, out)
  c.appendChild(box)
  if (s.result) show(); else draw()
  return c
}

registerTrainSection('tiny_image', { heading: 'Image models — learn from pictures', render: imageTrainer })
