import { GET, POST, S, btn, el, flow, registerTab, texts, toast } from '../core.js'
import { esc } from '../flow.js'
import { renderTab } from '../ui/tabs.js'

// ---------- image: block contributions
S.mapStep = -1;
 S.contribSteps = 'all'

export async function renderContrib(p) {
  const o = S.run.outputs
  p.innerHTML = ''
  const intro = el('div', { className: 'small' }, texts() === 'dit'
    ? `A DiT block doesn't hold an image either: it holds one vector per 16×16 tile. Views: <b>where</b> each block writes
       and <b>what</b> its tokens encode (captured on a few steps), and <b>what each step changed</b>.`
    : `A UNet stage doesn't hold an image, so “input image → stage → output image” doesn't exist inside
    it. Three honest views instead: <b>where</b> each stage writes, <b>what</b> its features encode, and <b>how much the final
    image changes</b> if you remove the stage's writes.`)
  p.appendChild(intro)
  // input / output
  const init = flow.nodes.find(n => n.data.kind === 'init_image')?.data.image
  const io = el('div', { className: 'io' })
  io.innerHTML = `<figure>${init ? `<img src="${init}">` : '<div class="noise">pure noise<br>(seed)</div>'}<figcaption>input</figcaption></figure>
    <span class="arrow">→</span><figure><img src="${o.image}"><figcaption>output</figcaption></figure>`
  p.appendChild(io)

  // A. where each stage writes + PCA
  const secA = el('div', { className: 'sec' }, '<h4>1 · Where each stage writes, and what it sees</h4>')
  p.appendChild(secA)
  let maps
  try { maps = await GET(`/api/runs/${S.run.id}/stagemaps?step=${S.mapStep}`) } catch (e) { secA.appendChild(el('div', { className: 'empty', textContent: e.message })) }
  if (maps) {
    const ctl = el('div', { className: 'row' })
    const range = el('input', { type: 'range', min: 0, max: maps.steps.length - 1, value: maps.steps.indexOf(maps.step) })
    range.oninput = () => { S.mapStep = maps.steps[+range.value]; renderTab() }
    ctl.append(el('span', { className: 'small', textContent: `denoising step ${maps.step} of ${o.steps}` }), range)
    secA.appendChild(ctl)
    secA.appendChild(el('div', { className: 'small', innerHTML: 'Top: size of the stage’s residual writes per pixel, over the final image (bright = writes a lot here). Bottom: the stage’s output features squeezed to 3 colours with PCA — patches with similar colours carry similar features, so you can watch objects separate from background deeper in the network.' }))
    const total = Math.max(...maps.stages.map(s => s.total))
    const grid = el('div', { className: 'stagegrid' })
    maps.stages.forEach(st => {
      const card = el('div', { className: 'stagecard' })
      card.innerHTML = `<h5>${esc(flow.node(st.node)?.data.label || st.node)} <span class="small">${st.res[1]}×${st.res[0]}</span></h5>
        <div class="ov"><img src="${o.image}"><img class="hm" src="${st.writes}"></div>
        <img class="pca" src="${st.pca}"><div class="bar"><i style="width:${(st.total / total * 100).toFixed(0)}%"></i></div>
        <div class="small">writes ${(st.total / total * 100).toFixed(0)}% of max</div>`
      card.onclick = () => flow.select({ type: 'node', id: st.node })
      grid.appendChild(card)
    })
    secA.appendChild(grid)
  }

  // B. ablation
  const secB = el('div', { className: 'sec' }, '<h4>2 · Which stage matters most? (remove it and compare)</h4>')
  p.appendChild(secB)
  if (S.info?.capabilities?.contributions === false) {
    secB.appendChild(el('div', { className: 'small', innerHTML: 'Measuring every block would re-run the whole generation once per block — too slow for this model. Do it by hand: <b>Bypass</b> a block (right-click it), press Run, and compare the two runs in the Runs tab. <b>Branch here</b> on a late step makes it cheap: only the remaining steps are recomputed.' }))
  } else {
  const ctlB = el('div', { className: 'row' })
  const stepsIn = el('input', { value: S.contribSteps, style: 'width:90px', title: 'steps to remove the stage in: all | 0:2 | 3' })
  stepsIn.onchange = () => { S.contribSteps = stepsIn.value || 'all' }
  ctlB.append(el('span', { className: 'small', textContent: 'remove during steps' }), stepsIn,
    btn(`Measure (re-runs the image ${S.run.plan.schedule.length}×)`, async () => {
      S.contribSteps = stepsIn.value || 'all'
      try { await POST(`/api/runs/${S.run.id}/contributions`, { steps: S.contribSteps }); pollContrib() } catch (e) { toast(e.message, 'error') }
    }))
  secB.appendChild(ctlB)
  const res = el('div', { id: 'contrib' })
  secB.appendChild(res)
  pollContrib()
  }

  // C. per-step change
  const secC = el('div', { className: 'sec' }, '<h4>3 · What each step changed</h4><div class="small">Brightness = how much the predicted image changed since the previous step. Big early, small late: layout first, detail later.</div>')
  p.appendChild(secC)
  secC.appendChild(stepDiffStrip())
}

export async function pollContrib() {
  const box = document.getElementById('contrib'); if (!box || !S.run) return
  let j
  try { j = await GET(`/api/runs/${S.run.id}/contributions?steps=${encodeURIComponent(S.contribSteps)}`) } catch { return }
  if (j.status === 'none') { box.innerHTML = '<div class="small">Each stage is switched off in turn (its residual writes dropped for the chosen steps); the bigger the change in the final image, the more that stage contributed.</div>'; return }
  if (j.status === 'running') { box.innerHTML = `<div class="small">measuring… ${Math.round(j.progress * 100)}% · ${esc(j.message)}</div>`; setTimeout(pollContrib, 700); return }
  if (j.status === 'error') { box.innerHTML = `<div class="bad">${esc(j.message)}</div>`; return }
  const mx = Math.max(...j.results.map(r => r.score)) || 1
  const ranked = [...j.results].sort((a, b) => b.score - a.score)
  box.innerHTML = `<div class="small">Ranking: ${ranked.map((r, i) => `${i + 1}. ${esc(flow.node(r.node)?.data.label || r.node)}`).join(' · ')}</div>`
  const grid = el('div', { className: 'stagegrid' })
  j.results.forEach(r => {
    const card = el('div', { className: 'stagecard' + (r.score === mx ? ' top' : '') })
    card.innerHTML = `<h5>without ${esc(flow.node(r.node)?.data.label || r.node)}</h5><img src="${r.image}"><img src="${r.diff}" title="difference to the full image">
      <div class="bar"><i style="width:${(r.score / mx * 100).toFixed(0)}%"></i></div><div class="small">mean change ${(r.score * 100).toFixed(1)}% · max ${(r.max_diff * 100).toFixed(0)}%</div>`
    card.onclick = () => flow.select({ type: 'node', id: r.node })
    grid.appendChild(card)
  })
  box.appendChild(grid)
  const b = {}
  j.results.forEach(r => { b[r.node] = `<span class="small">impact</span><i style="width:${(r.score / mx * 100).toFixed(0)}%;background:var(--mod)"></i>` })
  flow.setBadges({ ...flow.badges, ...b })
}

export function stepDiffStrip() {
  const o = S.run.outputs, steps = (o.preview_steps || [...Array(o.steps).keys()])
  const rv = S.run.version ?? 0
  const wrap = el('div', { className: 'steps' })
  let prev = null
  steps.forEach(i => {
    const f = el('figure')
    const cv = el('canvas', { width: 128, height: 128 })
    f.append(cv, el('figcaption', { textContent: `step ${i}` }))
    wrap.appendChild(f)
    const img = new Image()
    const mine = { img, ready: new Promise(r => { img.onload = r }) }
    img.src = `/api/runs/${S.run.id}/step/${i}.png?v=${rv}`
    const before = prev
    prev = mine
    mine.ready.then(async () => {
      const g = cv.getContext('2d')
      if (!before) { g.drawImage(img, 0, 0, 128, 128); return }
      await before.ready
      const a = el('canvas', { width: 128, height: 128 }).getContext('2d'); a.drawImage(before.img, 0, 0, 128, 128)
      g.drawImage(img, 0, 0, 128, 128)
      const A = a.getImageData(0, 0, 128, 128), B = g.getImageData(0, 0, 128, 128)
      for (let k = 0; k < B.data.length; k += 4) {
        const d = Math.min(1, 3 * (Math.abs(B.data[k] - A.data[k]) + Math.abs(B.data[k + 1] - A.data[k + 1]) + Math.abs(B.data[k + 2] - A.data[k + 2])) / 765)
        B.data[k] = 255 * Math.min(1, d * 3); B.data[k + 1] = 255 * Math.min(1, Math.max(0, d * 3 - 1)); B.data[k + 2] = 255 * Math.max(0, d * 3 - 2)
      }
      g.putImageData(B, 0, 0)
    })
  })
  return wrap
}

registerTab({ id: 'contrib', title: 'Block contributions', needs: 'run', render: renderContrib })
