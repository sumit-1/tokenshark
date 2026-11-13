import { $, POST, S, btn, el, flow, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { openPaint } from '../paint.js'
import { debugStep, showDebug } from '../ui/debugger.js'
import { genSize } from '../ui/inspector.js'
import { applyRunBadges } from '../ui/run.js'
import { renderTab } from '../ui/tabs.js'

// ---------- diffusion step debugger (Qwen-Image)
S.editStrength = 1;
 S.editOnlyChanged = true;
 S.capEditSigma = true

export function renderImageDebug(p) {
  const st = S.debug
  if (!st) {
    p.innerHTML = `<div class="empty">Step through the generation one block or one denoising step at a time.<br>
      Press <b>🐞 Step</b> or <b>▶ Run</b>. <b>Block ▸</b> (F10) runs the next block, <b>Step ▸▸</b> (F11) finishes the current denoising step.<br>
      At any step boundary you can <b>edit the image</b> in the paint editor, and the next steps carry on from your edit.</div>`
    return
  }
  if (st.busy && st.steps === 0 && !st.guess_at) {
    p.innerHTML = `<div class="empty">${spinner(st.progress?.message || 'preparing stepping session…')}<br><span class="small">The prompt is pushed through the 36-block Qwen3-VL text encoder and 32 DiT blocks once (cached for subsequent runs).</span></div>`
    return
  }
  const v = `${st.steps}-${st.edits.length}-${st.sigma}`
  const where = st.done ? 'all steps completed — you can still <b>edit the image</b>, <b>adjust noise σ</b>, or <b>add more steps</b> to continue refining'
    : st.phase === 'idle' ? `between steps — step ${st.step} starts from the input on the left (σ ${st.sigma})`
      : `inside step ${st.step}: next is ${esc(flow.node(st.active_node)?.data.label || '#' + st.next_pos)}`
  const ga = st.guess_at
  const guessLbl = !ga ? 'starting canvas (edit before step 0, or press Step ▸▸)' : ga[1] == null ? `after step ${ga[0]} — the picture the next step aims for` : `after block ${ga[1]} of step ${ga[0]} (if the model stopped here)`
  p.appendChild(el('div', { className: 'small', innerHTML: `Step <b>${Math.min(st.step, st.n_steps)}</b> of ${st.n_steps} · ${where}` }))
  const row = el('div', { className: 'dbgimgs' })
  const fIn = el('figure', {}, `<figcaption><b>Input xₜ</b> ≈ — what the model is given: ${Math.round((1 - st.sigma) * 100)}% picture + ${Math.round(st.sigma * 100)}% noise</figcaption>`)
  const fG = el('figure', {}, `<figcaption><b>Current image x̂₀</b> (exact) — ${esc(guessLbl)}</figcaption>`)
  const imIn = el('img'), imG = el('img'), wait = el('div', { className: 'small', innerHTML: spinner('decoding the image…') })
  fIn.prepend(imIn); fG.prepend(imG, wait)
  row.append(fIn, fG)
  p.appendChild(row)
  const cachedImg = S.run?.id === st.run_id && S.run?.outputs?.image ? S.run.outputs.image : (S.lastGuessKey === `${st.run_id}:${st.steps}:${st.edits.length}` ? S.lastGuessUrl : null)
  imG.onload = () => {
    wait.remove()
    if (ga || st.edits.length) imIn.src = `/api/debug/input.png?v=${v}`
  }
  imG.onerror = () => { wait.remove(); imG.alt = 'no image yet' }
  if (cachedImg) {
    S.lastGuessUrl = cachedImg; S.lastGuessKey = `${st.run_id}:${st.steps}:${st.edits.length}`
    imG.src = cachedImg
  } else {
    imG.src = `/api/debug/guess.png?exact=1&v=${v}`
  }
  if (!ga && !st.edits.length) fIn.querySelector('figcaption').innerHTML += '<br><span class="small">(noisy input preview appears after the first step or edit)</span>'
  const ctl = el('div', { className: 'row' })
  const bBlock = btn('Block ▸', () => debugStep('block')), bStep = btn('Step ▸▸', () => debugStep('token')), bAll = btn('Run to end ▶|', () => debugStep('all'))
  bBlock.disabled = bStep.disabled = bAll.disabled = !!st.busy
  ctl.append(bBlock, bStep, bAll)
  const edit = btn('✏️ Edit image…', () => editGuess(), 'primary')
  edit.disabled = !st.can_edit || !!st.busy
  edit.title = st.can_edit ? 'Paint or draw on the image; the next steps continue from your edit' : 'Finish the current step first (press Step ▸▸)'
  const addMore = btn('+ 4 steps', async () => {
    try {
      const r = await POST('/api/debug/extend', { add: 4 })
      toast(`Extended schedule to ${r.n_steps} steps — press Step ▸▸ or Run to end ▶| to continue.`)
      showDebug(r)
    } catch (e) { toast(e.message, 'error') }
  })
  addMore.disabled = !st.can_edit || !!st.busy
  addMore.title = 'Add 4 more denoising steps (or increase Steps in the Scheduler box and press Step ▸▸)'
  const oc = el('label', { className: 'field check' }), cb = el('input', { type: 'checkbox', checked: S.editOnlyChanged })
  cb.onchange = () => { S.editOnlyChanged = cb.checked }
  oc.append(cb, el('span', { textContent: 'only update changed tiles' }))
  const capCb = el('input', { type: 'checkbox', checked: S.capEditSigma })
  capCb.onchange = () => { S.capEditSigma = capCb.checked }
  const capLbl = el('label', { className: 'field check', title: 'Early steps have 80–97% noise, which drowns out edits. When checked, saving an edit automatically lowers σ to ≤ 0.50 (at least 50% your picture) so your edit sticks.' })
  capLbl.append(capCb, el('span', { textContent: 'keep ≥50% picture on edit (σ ≤ 0.50)' }))
  ctl.append(edit, addMore, oc, capLbl)
  p.appendChild(ctl)

  // Interactive noise level (σ) control between steps
  const nRow = el('div', { className: 'row' })
  const curSig = st.done ? 0.35 : (st.sigma || 0.35)
  const sigSl = el('input', { type: 'range', min: 0.02, max: 1.0, step: 0.02, value: curSig, style: 'width:200px', disabled: !st.can_edit || !!st.busy })
  const sigTxt = el('span', { className: 'small' })
  const updateSigTxt = val => {
    const s = +val
    sigTxt.innerHTML = `<b>Between-step noise σ = ${s.toFixed(2)}</b> (${Math.round((1 - s) * 100)}% picture + ${Math.round(s * 100)}% noise)`
  }
  updateSigTxt(curSig)
  sigSl.oninput = () => updateSigTxt(sigSl.value)
  sigSl.onchange = async () => {
    try {
      const r = await POST('/api/debug/noise', { sigma: +sigSl.value, fresh_noise: false })
      toast(`Set noise σ = ${r.noise.sigma} (${Math.round((1 - r.noise.sigma) * 100)}% picture + ${Math.round(r.noise.sigma * 100)}% noise).`)
      showDebug(r)
    } catch (e) { toast(e.message, 'error') }
  }
  const reseedBtn = btn('🎲 Fresh noise', async () => {
    try {
      const r = await POST('/api/debug/noise', { sigma: +sigSl.value, fresh_noise: true })
      toast(`Sampled fresh random noise at σ = ${r.noise.sigma}.`)
      showDebug(r)
    } catch (e) { toast(e.message, 'error') }
  }, 'mini')
  reseedBtn.disabled = !st.can_edit || !!st.busy
  reseedBtn.title = 'Replace the current noise ε̂ with a fresh random noise pattern at this σ'
  nRow.append(sigTxt, sigSl, reseedBtn)
  p.appendChild(nRow)

  p.appendChild(el('div', { className: 'small note', innerHTML: `<b>How interactive editing & noise σ work:</b> at noise level σ, the model's input is <b>xₜ = (1−σ)·x̂₀ + σ·ε̂</b>. In early steps, Qwen-Image's schedule keeps σ high (e.g. σ = 0.86 at step 5 → 86% noise + 14% picture). Use the <b>Between-step noise σ</b> slider above to lower σ (so the model keeps more of the current picture) or raise σ (to let the model repaint more freely).` }))
  if (st.edits.length) p.appendChild(el('div', { className: 'small', textContent: 'Edits: ' + st.edits.map(e => `step ${e.step} (σ ${e.sigma}, ${e.tiles}/${e.of} tiles)`).join(' · ') }))
  if (st.trace.length) {
    const mx = Math.max(...st.trace.map(r => r.write)) || 1
    const t = el('div', { className: 'dbgtrace' })
    st.trace.forEach(r => {
      const d = el('div', { title: `${flow.node(r.node)?.data.label || r.node}: average size of what it wrote into the image tokens` }, `<span>${esc(flow.node(r.node)?.data.label || '#' + r.pos)}</span><i style="width:${(r.write / mx * 100).toFixed(0)}%"></i>`)
      d.onclick = () => flow.node(r.node) && flow.select({ type: 'node', id: r.node })
      t.appendChild(d)
    })
    p.append(el('h4', { textContent: `Blocks run in step ${st.step}` }), t)
  }
}

export async function editGuess() {
  const st = S.debug
  if (!st?.can_edit) return
  let url = (S.run?.id === st.run_id && S.run?.outputs?.image) || (S.lastGuessKey === `${st.run_id}:${st.steps}:${st.edits.length}` ? S.lastGuessUrl : null)
  if (!url) {
    try {
      const b = await (await fetch(`/api/debug/guess.png?exact=1&v=${st.steps}-${st.edits.length}`)).blob()
      url = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(b) })
    } catch { return toast('Could not load the guess.', 'error') }
  }
  const [W, H] = genSize()
  openPaint({ mode: 'image', title: `Edit image · before step ${st.step}`, width: W, height: H, value: url, noiseCanvas: false,
    onSave: async img => {
      // Immediately update the displayed image in the UI so there is zero visual lag
      S.lastGuessUrl = img
      if (S.run?.outputs) S.run.outputs.image = img
      renderTab(); applyRunBadges()
      $('#progress').classList.add('indet'); $('#progress').style.width = '100%'
      $('#progress-msg').textContent = 'encoding edited image into latent space (VAE)…'
      const targetSigma = S.capEditSigma && st.sigma > 0.50 ? 0.45 : null
      try {
        const r = await POST('/api/debug/edit', { image: img, strength: S.editStrength, only_changed: S.editOnlyChanged, sigma: targetSigma })
        S.lastGuessUrl = img; S.lastGuessKey = `${r.run_id}:${r.steps}:${r.edits.length}`
        toast(r.edit.tiles ? `Edited ${r.edit.tiles} of ${r.edit.of} tiles (σ = ${r.edit.sigma}) — press Step ▸▸ or ▶ Continue.` : 'Saved image.')
        showDebug(r)
      } catch (e) { toast(e.message, 'error') }
      finally { $('#progress').classList.remove('indet'); $('#progress').style.width = '0%'; $('#progress-msg').textContent = '' }
    } })
}

registerTab({ id: 'debug_image', alias: 'debug', title: 'Step debugger', render: renderImageDebug })
