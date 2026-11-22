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
S.matchStepNoise = true;
S.editNoise = 0.75;
S.freshEditNoise = false;
S.editSteerSteps = 5;
S.editSteerPull = 0.45;
S.dbgPromptBlocks = 'all'

export function renderImageDebug(p) {
  const st = S.debug
  if (!st) {
    p.innerHTML = `<div class="empty">Step through the generation one block or one denoising step at a time.<br>
      Press <b>🐞 Step</b> or <b>▶ Run</b>. <b>Block ▸</b> (F10) runs the next block, <b>Step ▸▸</b> (F11) finishes the current denoising step.<br>
      At any step boundary you can <b>edit the image</b>, <b>switch prompts</b>, or <b>boost/mute words</b>, and the next steps carry on from your steer.</div>`
    return
  }
  if (st.busy && st.steps === 0 && !st.guess_at) {
    p.innerHTML = `<div class="empty">${spinner(st.progress?.message || 'preparing stepping session…')}<br><span class="small">The prompt is pushed through the 36-block Qwen3-VL text encoder and 32 DiT blocks once (cached for subsequent runs).</span></div>`
    return
  }
  const v = `${st.steps}-${st.edits.length}-${st.sigma}`
  const where = st.done ? 'all steps completed · you can still <b>edit the image</b>, <b>switch the prompt</b>, <b>adjust noise σ</b>, or <b>add more steps</b> to continue refining'
    : st.phase === 'idle' ? `between steps · step ${st.step} starts from the input on the left (σ ${st.sigma})`
      : `inside step ${st.step}: next is ${esc(flow.node(st.active_node)?.data.label || '#' + st.next_pos)}`
  const ga = st.guess_at
  const guessLbl = !ga ? 'starting canvas (edit before step 0, or press Step ▸▸)' : ga[1] == null ? `after step ${ga[0]} · the picture the next step aims for` : `after block ${ga[1]} of step ${ga[0]} (if the model stopped here)`
  p.appendChild(el('div', { className: 'small', innerHTML: `Step <b>${Math.min(st.step, st.n_steps)}</b> of ${st.n_steps} · ${where}` }))
  const row = el('div', { className: 'dbgimgs' })
  const fIn = el('figure', {}, `<figcaption><b>Input xₜ</b> ≈ · what the model is given: ${Math.round((1 - st.sigma) * 100)}% picture + ${Math.round(st.sigma * 100)}% noise</figcaption>`)
  const fG = el('figure', {}, `<figcaption><b>Current image x̂₀</b> (exact) · ${esc(guessLbl)}</figcaption>`)
  const imIn = el('img'), imG = el('img'), wait = el('div', { className: 'small', innerHTML: spinner('decoding the image…') })
  fIn.prepend(imIn); fG.prepend(imG, wait)
  row.append(fIn, fG)
  p.appendChild(row)
  const exactKey = `${st.run_id}:${st.steps}:${st.edits.length}`
  const cachedImg = S.lastGuessKey === exactKey ? S.lastGuessUrl
    : (st.done && !st.edits.length && S.run?.id === st.run_id && S.run?.outputs?.image ? S.run.outputs.image : null)
  imG.onload = () => {
    wait.remove()
    if (ga || st.edits.length) imIn.src = `/api/debug/input.png?v=${v}`
  }
  imG.onerror = () => { wait.remove(); imG.alt = 'no image yet' }
  if (cachedImg) {
    S.lastGuessUrl = cachedImg; S.lastGuessKey = exactKey
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
      toast(`Extended schedule to ${r.n_steps} steps · press Step ▸▸ or Run to end ▶| to continue.`)
      showDebug(r)
    } catch (e) { toast(e.message, 'error') }
  })
  addMore.disabled = !st.can_edit || !!st.busy
  addMore.title = 'Add 4 more denoising steps (or increase Steps in the Scheduler box and press Step ▸▸)'
  const oc = el('label', { className: 'field check' }), cb = el('input', { type: 'checkbox', checked: S.editOnlyChanged })
  cb.onchange = () => { S.editOnlyChanged = cb.checked }
  oc.append(cb, el('span', { textContent: 'only update changed tiles' }))
  const stSel = el('select', { title: 'A 1-step edit at early steps is quickly washed out by 20+ steps of text velocity, while a mid-step edit without trajectory guidance stays blurry. Decaying steer nudges x̂₀ toward your edit for K steps while structure forms, then releases it so final steps render sharp detail.' })
  ;[[0, 'steer: 1 step only (raw impulse)'], [3, 'steer: 3 steps (gentle decay)'], [5, 'steer: 5 steps (recommended)'], [8, 'steer: 8 steps (strong shape guide)']].forEach(([v, t]) => stSel.appendChild(el('option', { value: v, textContent: t })))
  stSel.value = S.editSteerSteps
  stSel.onchange = () => { S.editSteerSteps = +stSel.value }
  ctl.append(edit, addMore, oc, stSel)
  p.appendChild(ctl)

  // Customizable Edit Noiser (controls how much noise is mixed into painted edits without locking the schedule)
  const eRow = el('div', { className: 'row' })
  const matchCb = el('input', { type: 'checkbox', checked: S.matchStepNoise })
  const matchLbl = el('label', { className: 'field check', title: 'When checked, your edit is mixed with the current step\'s natural noise σ (just like a step image generated in the process). Uncheck to dial custom edit noise.' })
  matchLbl.append(matchCb, el('span', { textContent: 'match step noise σ on edit' }))
  const stepSig = st.done ? 0.35 : (st.step === 0 && st.sigma >= 0.999 ? 0.85 : (st.sigma || 0.75))
  const editSigVal = S.matchStepNoise ? stepSig : S.editNoise
  const editSigSl = el('input', { type: 'range', min: 0.0, max: 0.98, step: 0.02, value: editSigVal, style: 'width:160px', disabled: S.matchStepNoise })
  const editSigTxt = el('span', { className: 'small' })
  const updateEditSigTxt = val => {
    const s = +val
    editSigTxt.innerHTML = `<b>Edit noiser σ<sub>edit</sub> = ${s.toFixed(2)}</b> (${Math.round((1 - s) * 100)}% edit + ${Math.round(s * 100)}% noise)`
  }
  updateEditSigTxt(editSigVal)
  matchCb.onchange = () => {
    S.matchStepNoise = matchCb.checked
    editSigSl.disabled = S.matchStepNoise
    if (S.matchStepNoise) { editSigSl.value = stepSig; updateEditSigTxt(stepSig) }
    else { editSigSl.value = S.editNoise; updateEditSigTxt(S.editNoise) }
  }
  editSigSl.oninput = () => { S.editNoise = +editSigSl.value; updateEditSigTxt(S.editNoise) }
  const freshCb = el('input', { type: 'checkbox', checked: S.freshEditNoise })
  freshCb.onchange = () => { S.freshEditNoise = freshCb.checked }
  const freshLbl = el('label', { className: 'field check', title: 'Mix fresh random Gaussian noise into the edited tiles instead of reusing the current step\'s noise estimate.' })
  freshLbl.append(freshCb, el('span', { textContent: '🎲 fresh noise on edit' }))
  eRow.append(matchLbl, editSigTxt, editSigSl, freshLbl)
  p.appendChild(eRow)

  // Interactive schedule noise level (σ) control between steps
  const nRow = el('div', { className: 'row' })
  const curSig = st.done ? 0.35 : (st.sigma || 0.35)
  const sigSl = el('input', { type: 'range', min: 0.02, max: 1.0, step: 0.02, value: curSig, style: 'width:200px', disabled: !st.can_edit || !!st.busy })
  const sigTxt = el('span', { className: 'small' })
  const updateSigTxt = val => {
    const s = +val
    sigTxt.innerHTML = `<b>Schedule step σ = ${s.toFixed(2)}</b> (${Math.round((1 - s) * 100)}% picture + ${Math.round(s * 100)}% noise)`
  }
  updateSigTxt(curSig)
  sigSl.oninput = () => updateSigTxt(sigSl.value)
  sigSl.onchange = async () => {
    try {
      const r = await POST('/api/debug/noise', { sigma: +sigSl.value, fresh_noise: false })
      toast(`Set schedule σ = ${r.noise.sigma} (${Math.round((1 - r.noise.sigma) * 100)}% picture + ${Math.round(r.noise.sigma * 100)}% noise).`)
      showDebug(r)
    } catch (e) { toast(e.message, 'error') }
  }
  const reseedBtn = btn('🎲 Fresh step noise', async () => {
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

  // Live Mid-Step Prompt & Block-Split Steering + Word Attention Boosting
  const pBox = el('div', { className: 'sec', style: 'margin-top:8px;padding-top:8px;border-top:1px solid var(--line)' })
  pBox.appendChild(el('h4', { textContent: 'Mid-Step Prompt & Attention Steering' }))
  pBox.appendChild(el('div', { className: 'small', innerHTML: '<b>Lab experiment · When does a DiT commit to layout vs. identity vs. texture?</b><br>• Switch the prompt at <b>steps 2–5 (σ ≈ 0.90)</b> to morph the subject while keeping the pose/composition.<br>• Switch at <b>steps 10–14 (σ ≈ 0.60)</b> or target only <b>late blocks (16:32)</b> to keep the 3D geometry and repaint only surface material/lighting.' }))
  const pRow = el('div', { className: 'row', style: 'margin-top:6px' })
  const curPrompt = st.current_prompt || flow.nodes.find(n => n.data.kind === 'input_prompt')?.data.prompt || ''
  const pInp = el('input', { type: 'text', value: curPrompt, placeholder: 'New prompt from this step onward…', style: 'flex:1;min-width:240px', disabled: !st.can_edit || !!st.busy })
  const bSel = el('select', { disabled: !st.can_edit || !!st.busy, title: 'Which DiT blocks should read the new prompt KV cache' })
  ;[['all', 'All 32 blocks (full prompt switch)'], ['0:16', 'Early blocks 0:16 (layout / structure)'], ['16:32', 'Late blocks 16:32 (surface / texture / style)'], ['8:24', 'Middle blocks 8:24 (object identity)']].forEach(([v, t]) => bSel.appendChild(el('option', { value: v, textContent: t })))
  bSel.value = S.dbgPromptBlocks
  bSel.onchange = () => { S.dbgPromptBlocks = bSel.value }
  const applyPromptBtn = btn('🔀 Steer prompt from step ' + st.step, async () => {
    const text = pInp.value.trim()
    if (!text) return toast('Enter a prompt first.', 'error')
    $('#progress').classList.add('indet'); $('#progress').style.width = '100%'
    $('#progress-msg').textContent = 'encoding new prompt into DiT KV cache…'
    try {
      const r = await POST('/api/debug/prompt', { prompt: text, blocks: bSel.value })
      toast(`Switched prompt on blocks ${r.prompt_change.blocks} from step ${r.prompt_change.step} · press Step ▸▸ or Run to end ▶|.`)
      showDebug(r)
    } catch (e) { toast(e.message, 'error') }
    finally { $('#progress').classList.remove('indet'); $('#progress').style.width = '0%'; $('#progress-msg').textContent = '' }
  }, 'primary mini')
  applyPromptBtn.disabled = !st.can_edit || !!st.busy
  pRow.append(pInp, bSel, applyPromptBtn)
  pBox.appendChild(pRow)

  if (st.tokens?.length) {
    const wWrap = el('div', { style: 'margin-top:6px' })
    wWrap.appendChild(el('div', { className: 'small', innerHTML: '<b>Click a prompt token</b> to cycle its cross-attention weight for the remaining steps (<code>1×</code> → <code>2.5× boost</code> → <code>5× strong</code> → <code>0× mute</code>):' }))
    const toks = el('div', { className: 'tokens', style: 'margin-top:4px' })
    const ws = st.word_scales || {}
    st.tokens.forEach((t, idx) => {
      const sc = ws[idx] ?? 1
      const badge = sc !== 1 ? ` ×${sc}` : ''
      const chip = el('span', {
        className: 'tok' + (sc > 1 ? ' on' : ''),
        style: sc === 0 ? 'opacity:0.4;text-decoration:line-through' : sc > 1 ? 'outline:1px solid var(--accent)' : '',
        textContent: t + badge,
        title: `Token #${idx} (${t}): current attention weight = ${sc}×. Click to cycle (1× → 2.5× → 5× → 0×).`,
      })
      chip.onclick = async () => {
        if (!st.can_edit || st.busy) return toast('Finish the current step first (Step ▸▸).')
        const nextSc = sc === 1 ? 2.5 : sc === 2.5 ? 5.0 : sc === 5.0 ? 0.0 : 1.0
        try {
          const r = await POST('/api/debug/word', { token: idx, scale: nextSc, blocks: 'all' })
          toast(`Token "${t.trim() || t}" attention set to ${nextSc}× from step ${st.step}.`)
          showDebug(r)
        } catch (e) { toast(e.message, 'error') }
      }
      toks.appendChild(chip)
    })
    wWrap.appendChild(toks)
    pBox.appendChild(wWrap)
  }
  p.appendChild(pBox)

  p.appendChild(el('div', { className: 'small note', innerHTML: `<b>Why single-step edits get overwritten or blurred (and how to fix it):</b> in flow matching, one step only moves the latent by ~4%, while the remaining 20 steps are driven by the text prompt's velocity. With <b>steer: 5 steps (recommended)</b> selected above, saving a painted edit gently nudges x̂₀ toward your painted shape over the next 5 steps (while structure forms) and then releases it so final steps render sharp detail · or combine your paint edit with a <b>Mid-Step Prompt Switch</b> so the text velocity actively reinforces what you drew!` }))
  if (st.edits.length) p.appendChild(el('div', { className: 'small', textContent: 'Edits: ' + st.edits.map(e => `step ${e.step} (σ_edit ${e.edit_noise ?? e.sigma}${e.steer_steps ? `, steer ${e.steer_steps} steps` : ''}, ${e.tiles}/${e.of} tiles)`).join(' · ') }))
  if (st.prompt_changes?.length) p.appendChild(el('div', { className: 'small', textContent: 'Prompt switches: ' + st.prompt_changes.map(c => `step ${c.step} [blocks ${c.blocks}] → "${c.prompt}"`).join(' · ') }))
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
  const exactKey = `${st.run_id}:${st.steps}:${st.edits.length}`
  let url = S.lastGuessKey === exactKey ? S.lastGuessUrl
    : (st.done && !st.edits.length && S.run?.id === st.run_id && S.run?.outputs?.image ? S.run.outputs.image : null)
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
      const editNoise = S.matchStepNoise ? null : S.editNoise
      try {
        const r = await POST('/api/debug/edit', {
          image: img,
          strength: S.editStrength,
          only_changed: S.editOnlyChanged,
          edit_noise: editNoise,
          fresh_noise: S.freshEditNoise,
          steer_steps: S.editSteerSteps,
          steer_pull: S.editSteerPull,
        })
        S.lastGuessUrl = img; S.lastGuessKey = `${r.run_id}:${r.steps}:${r.edits.length}`
        toast(r.edit.tiles ? `Edited ${r.edit.tiles} of ${r.edit.of} tiles (σ_edit = ${r.edit.edit_noise ?? r.edit.sigma}, steer = ${r.edit.steer_steps || 1} steps) · press Step ▸▸ or ▶ Continue.` : 'Saved image.')
        showDebug(r)
      } catch (e) { toast(e.message, 'error') }
      finally { $('#progress').classList.remove('indet'); $('#progress').style.width = '0%'; $('#progress-msg').textContent = '' }
    } })
}

registerTab({ id: 'debug_image', alias: 'debug', title: 'Step debugger', render: renderImageDebug })
