import { $, GET, POST, S, btn, el, hasFeature, registerTab, spinner, texts, toast } from '../core.js'
import { esc } from '../flow.js'
import { debugStep, showDebug } from '../ui/debugger.js'
import { renderTab } from '../ui/tabs.js'
import { editGuess } from './debug_image.js'

// ---------- image output, steps, cross-attention
export function stepStrip() {
  const o = S.run.outputs
  const rv = S.run.version ?? 0
  const outer = el('div')
  outer.appendChild(el('div', { className: 'small note', innerHTML: 'Top: <b>what the model gets</b> at each step — the noisy latent xₜ = (1−σ)·picture + σ·noise. Bottom: <b>its guess</b> of the finished image (x̂₀). Both ≈ approximate colours (a linear map, one square per latent tile); only the final image is an exact decode.' }))
  const wrap = el('div', { className: 'steps two' })
  for (let i = 0; i < o.steps; i++) {
    if (o.preview_steps && !o.preview_steps.includes(i)) continue      // img2img: steps before the start were skipped
    const f = el('figure', { className: i < o.start_step ? 'inherited' : '' })
    const sg = o.sigmas?.[i]
    f.innerHTML = `<img class="xt" loading="lazy" src="/api/runs/${S.run.id}/step/${i}.png?kind=xt&v=${rv}" title="input at step ${i}: ${sg != null ? `${Math.round((1 - sg) * 100)}% picture + ${Math.round(sg * 100)}% noise` : 'noisy latent'}" onerror="this.style.display='none'">` +
      `<img loading="lazy" src="/api/runs/${S.run.id}/step/${i}.png?v=${rv}" title="the model's guess of the finished image at step ${i}"><figcaption>step ${i} · σ ${sg ?? ''}</figcaption>`
    f.appendChild(btn('Branch here', () => { S.resume = { run_id: S.run.id, step: i }; renderResume(); toast(`Next run starts from step ${i} of this run. Edit the flow, then Run.`) }, 'mini'))
    wrap.appendChild(f)
  }
  outer.appendChild(wrap)
  return outer
}

export async function renderImageOut(p) {
  const o = S.run.outputs
  const rv = S.run.version ?? 0
  const lastPreviewStep = o.preview_steps?.length ? o.preview_steps[o.preview_steps.length - 1] : null
  const mainImgSrc = o.image || (lastPreviewStep != null ? `/api/runs/${S.run.id}/step/${lastPreviewStep}.png?v=${rv}` : '')
  const top = el('div', { className: 'imgout' })
  const fig = el('div', { className: 'overlay' })
  if (mainImgSrc) fig.appendChild(el('img', { src: mainImgSrc }))
  else fig.appendChild(el('div', { className: 'empty', innerHTML: spinner('generating first step…') }))
  const cv = el('canvas', { className: 'heat' }); fig.appendChild(cv)
  top.appendChild(fig)
  const side = el('div')
  side.innerHTML = texts() === 'dit'
    ? `<h4>Which parts of the image read this word?</h4><div class="small">In this single-stream DiT the prompt and the
      image are one sequence: every image token attends to every prompt token (and to the other image tokens). Click a
      token: the overlay shows how much attention each 16×16 tile paid to it (averaged over heads and the chosen
      blocks, on the captured steps). Steer with a <i>word</i> intervention on a block.</div>
      ${o.kv_note ? `<div class="small note">${esc(o.kv_note)}</div>` : ''}${o.guide_note ? `<div class="small note">${esc(o.guide_note)}</div>` : ''}`
    : `<h4>Which parts of the image read this word?</h4><div class="small">Every UNet stage has
    cross-attention: each image patch looks at the prompt tokens and pulls in what it finds. Click a token: the
    overlay lights up the patches that attended to it (averaged over heads; pick a stage and step below). Use it to
    see where “fox” landed, then steer with a <i>word</i> intervention or a region lock.</div>`
  const toks = el('div', { className: 'tokens' })
  o.tokens.forEach((t, i) => { const s = el('span', { className: 'tok' + (i === S.xtok ? ' on' : ''), textContent: t }); s.onclick = () => { S.xtok = i; renderTab() }; toks.appendChild(s) })
  const gs = el('select'); (o.xattn_groups || []).forEach(g => gs.appendChild(el('option', { value: g, textContent: g })))
  gs.value = S.xgroup; gs.onchange = () => { S.xgroup = gs.value; renderTab() }
  const ss = el('select'); ss.appendChild(el('option', { value: -1, textContent: 'all steps' }))
  for (const i of (o.xattn_steps?.length ? o.xattn_steps : [...Array(o.steps - o.start_step).keys()].map(k => k + o.start_step)))
    ss.appendChild(el('option', { value: i, textContent: `step ${i}` }))
  ss.value = S.xstep; ss.onchange = () => { S.xstep = +ss.value; renderTab() }
  const row = el('div', { className: 'row' }); row.append(gs, ss)
  if (hasFeature('image_stepper') && S.debug && S.debug.run_id === S.run.id) {
    row.append(
      btn('✏️ Edit image…', () => editGuess(), 'primary mini'),
      btn('Step ▸▸', () => debugStep('token'), 'mini'),
      btn('+ 4 steps', async () => {
        try {
          const r = await POST('/api/debug/extend', { add: 4 })
          toast(`Extended schedule to ${r.n_steps} steps — press Step ▸▸ to continue.`)
          showDebug(r)
        } catch (e) { toast(e.message, 'error') }
      }, 'mini')
    )
  }
  side.append(toks, row, el('div', { className: 'small', textContent: `${o.steps} steps${o.start_step ? `, started at step ${o.start_step}` : ''}${S.run.seconds ? ` · run ${S.run.seconds}s` : ''}` }))
  top.appendChild(side)
  p.append(top, el('h4', { textContent: 'Each step’s guess of the final image (x̂₀)' }),
    el('div', { className: 'small', innerHTML: (o.preview_kind === 'linear' ? 'Previews are fast approximations (a linear latent → colour map fitted on the final image), not full VAE decodes. ' : '') + 'Early steps fix layout and colour, late steps add detail. <b>Branch here</b> saves this step’s noisy latent as the starting point for your next Run: change the prompt, guidance or interventions, press Run, and only the remaining steps are recomputed — the image analogue of forking a conversation mid-way. Greyed frames were inherited from the parent.' }),
    stepStrip())
  if (o.xattn_groups?.length) {
    try {
      const m = await GET(`/api/runs/${S.run.id}/xattn?token=${S.xtok}&group=${encodeURIComponent(S.xgroup)}&step=${S.xstep}`)
      if (m.map) {
        cv.width = m.w; cv.height = m.h
        const g = cv.getContext('2d')
        m.map.forEach((row, y) => row.forEach((v, x) => { g.fillStyle = `rgba(255,${200 - 150 * v},0,${0.75 * v})`; g.fillRect(x, y, 1, 1) }))
      }
    } catch (e) { /* captures evicted */ }
  }
}

export function renderResume() {
  $('#resume').innerHTML = S.resume ? `Branching from run ${esc(S.resume.run_id)} · step ${S.resume.step} <button id="resume-x" class="mini">✕</button>` : ''
  if (S.resume) $('#resume-x').onclick = () => { S.resume = null; renderResume() }
}

registerTab({ id: 'output_image', alias: 'output', title: 'Output', needs: 'run', render: renderImageOut })
