import { S, el, flow, registerTab, spinner } from '../core.js'
import { esc } from '../flow.js'
import { streamAnalysis } from '../ui/analysis.js'
import { renderTab } from '../ui/tabs.js'

// ---------- image lens (DiT): what the model would output if it stopped after block k
S.ilensSel = null

S.ilensThrough = 0

export function renderImageLens(p) {
  const o = S.run.outputs, S_ = o.lens_blocks, k = Math.min(S.ilensThrough, S_)
  // a selection belongs to one run: another run may have other captured steps (e.g. one that started from a picture)
  if (S.ilensSel && (S.ilensRun !== S.run.id || !o.lens_steps?.includes(S.ilensSel[0]) || S.ilensSel[1] >= S_)) S.ilensSel = null
  if (!o.lens_steps?.length) { p.innerHTML = '<div class="empty">No image-lens captures in this run (they are skipped when guidance runs the model twice).</div>'; return }
  p.appendChild(el('div', { className: 'small', innerHTML: `<b>Image lens.</b> After any block, apply the model's final norm + projection: the picture it would produce if it stopped there (its guess x̂₀ at that step). Middle blocks don't “speak pictures” yet — like early layers in the text logit lens — so their direct decode is often a mess. <b>Decode through the last k blocks</b> jumps from that block straight into the model's last k blocks first, which translates it. The top row is the input the model was given at that step.` }))
  const bar = el('div', { className: 'row' })
  const sel = el('select')
  sel.appendChild(el('option', { value: 0, textContent: 'decode each block directly (norm + projection)' }))
  for (const kk of [1, 2, 3, 4, 6, 8]) if (kk < S_) sel.appendChild(el('option', { value: kk, textContent: `through the last ${kk} block${kk > 1 ? 's' : ''}, then decode` }))
  sel.appendChild(el('option', { value: S_, textContent: `through all remaining blocks (= the real model)` }))
  sel.value = k
  sel.onchange = () => { S.ilensThrough = +sel.value; S.ilensSel = null; renderTab() }
  const status = el('span', { className: 'small' })
  bar.append(el('span', { className: 'small', textContent: 'Decode:' }), sel, status)
  p.appendChild(bar)
  p.appendChild(el('div', { className: 'small note', innerHTML: '≈ Thumbnails are <b>approximate colours</b> (a linear map from the latent, fitted on this run\'s final image; one square per 16×16 tile). Click one for the exact VAE decode.' }))
  const tbl = el('table', { className: 'ilens' })
  const head = el('tr'); head.appendChild(el('th'))
  o.lens_steps.forEach(s => head.appendChild(el('th', { textContent: `step ${s}`, title: `σ = ${o.sigmas?.[s]}` })))
  tbl.appendChild(head)
  // what the model gets
  const rv = S.run.version ?? 0
  const inRow = el('tr', { className: 'inrow' })
  inRow.appendChild(el('th', { textContent: 'input xₜ ≈', title: 'the noisy latent the model was given at this step (approximate colours)' }))
  o.lens_steps.forEach(s => {
    const td = el('td')
    td.appendChild(el('img', { loading: 'lazy', src: `/api/runs/${S.run.id}/step/${s}.png?kind=xt&v=${rv}`, title: `input at step ${s}: ${Math.round((1 - (o.sigmas?.[s] ?? 0)) * 100)}% picture + ${Math.round((o.sigmas?.[s] ?? 0) * 100)}% noise` }))
    inRow.appendChild(td)
  })
  tbl.appendChild(inRow)
  const cells = {}
  const q = k ? `?through=${k}&v=${rv}` : `?v=${rv}`
  for (let b = 0; b < S_; b++) {
    const nid = S.run.plan.schedule_nodes[b]
    const tr = el('tr')
    const th = el('th', { textContent: flow.node(nid)?.data.label || `#${b}`, title: 'select this block in the flow' })
    th.onclick = () => flow.node(nid) && flow.select({ type: 'node', id: nid })
    tr.appendChild(th)
    o.lens_steps.forEach(s => {
      const td = el('td')
      const img = el('img', { title: `after ${th.textContent} at step ${s}${k ? `, through the last ${k}` : ''}` })
      if (!k) img.src = `/api/runs/${S.run.id}/imagelens/${s}/${b}.png?v=${rv}`
      else td.classList.add('pending')
      img.onclick = () => { if (td.classList.contains('pending')) return; S.ilensSel = [s, b]; S.ilensRun = S.run.id; showLensExact(side) }
      if (S.ilensSel && S.ilensSel[0] === s && S.ilensSel[1] === b) td.classList.add('on')
      td.appendChild(img); tr.appendChild(td)
      cells[`${s}:${b}`] = [td, img]
    })
    tbl.appendChild(tr)
  }
  const side = el('div', { className: 'ilens-side' })
  const wrap = el('div', { className: 'ilens-wrap' })
  wrap.append(el('div', { className: 'ilens-grid' }, ''), side)
  wrap.firstChild.appendChild(tbl)
  p.appendChild(wrap)
  showLensExact(side)
  if (k) {
    // recomputed on the server as a task (cell by cell): each cell runs its block's step up to that block, then the last k blocks
    const seq = S.renderSeq, alive = () => S.renderSeq === seq && S.tab === 'ilens'
    const fill = (part, t) => {
      for (const [s, b] of part.done || []) {
        const c = cells[`${s}:${b}`]
        if (c && c[0].classList.contains('pending')) { c[0].classList.remove('pending'); c[1].src = `/api/runs/${S.run.id}/imagelens/${s}/${b}.png${q}` }
      }
      const n = (part.done || []).length, tot = o.lens_steps.length * S_
      status.innerHTML = n < tot ? `${spinner()} computing ${n}/${tot}${t?.status === 'queued' ? ' — waiting for the model (see Tasks)' : ''}` : ''
    }
    streamAnalysis(`/api/runs/${S.run.id}/imagelens/compute?through=${k}`, fill, alive)
      .then(r => r && alive() && fill(r)).catch(e => { if (alive()) status.innerHTML = `<span class="bad">${esc(e.message)}</span>` })
  }
}

export async function showLensExact(side) {
  if (!S.ilensSel) { side.innerHTML = '<div class="small">Click a thumbnail for an exact decode.</div>'; return }
  const [s, b] = S.ilensSel, nid = S.run.plan.schedule_nodes[b], k = Math.min(S.ilensThrough, S.run.outputs.lens_blocks)
  const title = `After ${esc(flow.node(nid)?.data.label || '#' + b)} · step ${s}${k ? ` · through the last ${k}` : ''}`
  document.querySelectorAll('table.ilens td').forEach(td => td.classList.remove('on'))
  side.innerHTML = `<h4>${title}</h4><div class="small">${spinner('VAE decode (a task — see Tasks)')}</div>`
  const runId = S.run.id
  const img = new Image()
  const last = b === S.run.outputs.lens_blocks - 1
  img.onload = () => { if (S.run?.id !== runId) return; side.innerHTML = `<h4>${title} · exact</h4>`; side.append(img,
    el('div', { className: 'small', textContent: last ? 'The last block: this is the model\'s real prediction at this step.'
      : k ? `The image tokens after this block, sent through the model's last ${k} block(s) and decoded.` : 'If the model stopped here and skipped the remaining blocks.' })) }
  const url = `/api/runs/${runId}/imagelens/${s}/${b}.png?exact=1${k ? `&through=${k}` : ''}`
  img.onerror = async () => {
    let why = 'Decode failed or was stopped.'
    try { const r = await fetch(url); if (!r.ok) why = (await r.json()).detail || why } catch { }
    side.innerHTML = `<div class="bad">${esc(why)}</div>`
  }
  img.src = url
}

registerTab({ id: 'ilens', title: 'Image lens', needs: 'run', render: renderImageLens })
