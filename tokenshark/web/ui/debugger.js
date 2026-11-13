import { $, GET, POST, S, flow, on, schema, toast } from '../core.js'
import { validate } from './flowedit.js'
import { renderInspector } from './inspector.js'
import { applyRunBadges } from './run.js'
import { renderTab, setTab } from './tabs.js'
import { renderResume } from '../tabs/output_image.js'
import { refreshRuns } from '../tabs/runs.js'

S.debug = null

export async function debugStart() {
  if (!schema().step_debugger) return toast('This model cannot be stepped through.', 'error')
  if (!(await validate())) return toast('Fix the flow first.', 'error')
  S.lastGuessUrl = null; S.lastGuessKey = null
  S.resume = null; renderResume()
  if (!['lens', 'cloud', 'attn', 'kv', 'ilens', 'contrib'].includes(S.tab)) setTab('debug')   // stay on an analysis tab if one is open
  try {
    const st = await POST('/api/debug/start', { flow: flow.getGraph(), label: $('#run-label').value || 'debug', background: !!schema().step_background })
    S.debug = st
    renderDebugButtons()
    if (st.busy) {
      showDebugProgress(st)
      if (S.tab === 'debug') renderTab()
      pollDebug()
    } else {
      showDebug(st)
    }
  }
  catch (e) { toast(e.message, 'error') }
}

export async function debugStep(mode, extra = {}) {
  if (!S.debug) {
    await debugStart()
    // If debugStart launched background prompt encoding, wait for it to finish before stepping
    while (S.debug?.busy) await new Promise(r => setTimeout(r, 250))
  }
  if (!S.debug) return
  if (S.debug.busy) return toast('Still computing the previous step (■ stops it).')
  try {
    const st = await POST('/api/debug/step', { mode, flow: flow.getGraph(), ...extra })
    S.debug = st; renderDebugButtons(); showDebugProgress(st)
    if (S.tab === 'debug') renderTab()
    pollDebug()
  } catch (e) { toast(e.message, 'error') }
}

// A step runs in the background on the server; this follows it with the progress bar under the header.
export async function pollDebug() {
  clearTimeout(S.dbgPoll)
  let st
  try { st = await GET('/api/debug') } catch { S.dbgPoll = setTimeout(pollDebug, 1000); return }
  if (!st || !Object.keys(st).length) { S.debug = null; renderDebugButtons(); showDebugProgress(null); return }
  if (st.busy) {
    S.debug = { ...S.debug, busy: true, progress: st.progress }
    showDebugProgress(st)
    // show which block is running on the canvas
    flow.setActive(st.progress?.node || null, [])
    S.dbgPoll = setTimeout(pollDebug, 500); return
  }
  showDebugProgress(null)
  if (st.error) toast(st.error, 'error')
  showDebug(st)
}

export function showDebugProgress(st) {
  if (!st?.busy) {
    if (S.dbgShown) { $('#progress').style.width = '0%'; $('#progress-msg').textContent = 'step done'; S.dbgShown = false }
    return
  }
  S.dbgShown = true
  const pr = st.progress || { frac: 0, message: 'starting' }
  $('#progress').classList.remove('indet')
  $('#progress').style.width = `${Math.round(pr.frac * 100)}%`
  $('#progress-msg').textContent = `stepping: ${pr.message}` + (st.ram_gb != null ? ` · process RAM ${st.ram_gb} GB` : '') + ' · ■ stops after the current block'
}

export async function debugRunTo(n) {
  const plan = await POST('/api/flow/compile', { flow: flow.getGraph() }).catch(() => null)
  const p = plan?.plan?.schedule_nodes.indexOf(n.id)
  if (p == null || p < 0) return toast('That block is not on the active path (bypassed or disconnected).', 'error')
  if (!S.debug) await debugStart()
  debugStep('to', { pos: p })
}

export async function debugFinish() {
  if (!S.debug) return
  try {
    const r = await POST('/api/debug/finish')
    if (r.debug) S.debug = r.debug
    flow.setActive(null, []); renderDebugButtons()
    S.run = await GET(`/api/runs/${r.run.id}`); S.lens = null; applyRunBadges(); refreshRuns(); renderInspector(); renderTab()
    toast(schema().step_saved_note || 'Stepping session saved as a run.')
  } catch (e) { toast(e.message, 'error') }
}

export async function debugStop() { await POST('/api/debug/stop'); S.debug = null; flow.setActive(null, []); renderDebugButtons(); renderTab() }

export function showDebug(st) {
  S.debug = st
  const active = st.phase === 'block' ? st.active_node : st.phase === 'idle' ? flow.nodes.find(n => n.data.kind === 'sampler')?.id : null
  const done = st.trace.map(t => t.node)
  flow.setActive(active, done)
  // Keep the Scheduler box step count in sync if steps were extended
  if (st.n_steps) {
    const sc = flow.nodes.find(n => n.data.kind === 'scheduler')
    if (sc && +sc.data.steps !== st.n_steps) flow.updateNodeData(sc.id, { steps: st.n_steps })
  }
  renderDebugButtons()
  refreshLive(st)
}

// Stepping is a superset of Run: the session is a live run, so every tab (output, logit lens, word cloud,
// attention, KV cache), the inspector and the badges on the boxes refresh after each step.
export async function refreshLive(st) {
  if (!st.run_id) { if (S.tab === 'debug') renderTab(); return }
  try { S.run = await GET(`/api/runs/${st.run_id}`) } catch { return }
  applyRunBadges()
  renderInspector()
  renderTab()
  refreshRuns()
}

export function renderDebugButtons() {
  $('#dbg-buttons').style.display = schema().step_debugger ? '' : 'none'
  const on = !!S.debug, busy = !!S.debug?.busy, midStep = on && !S.debug.done
  document.querySelectorAll('.dbg-on').forEach(b => b.disabled = !on || (busy && b.id !== 'dbg-stop'))
  $('#dbg-start').disabled = busy
  $('#dbg-start').textContent = on ? '⟲ Restart' : '🐞 Step'
  const bySteps = schema().step_unit === 'step'         // what one "big step" is for this model: a token or a denoising step
  $('#dbg-token').textContent = bySteps ? 'Step ▸▸' : 'Token ▸▸'
  $('#dbg-token').title = bySteps ? 'Run the rest of this denoising step (F11)' : 'Run to the next token prediction (F11)'
  $('#run').textContent = midStep ? '▶ Continue' : '▶ Run'
  $('#run').title = midStep ? 'Continue running the remaining steps from here' : 'Run the flow from the beginning'
}

$('#dbg-start').onclick = () => debugStart()

$('#dbg-block').onclick = () => debugStep('block')

$('#dbg-token').onclick = () => debugStep('token')

$('#dbg-finish').onclick = () => debugFinish()

$('#dbg-stop').onclick = () => debugStop()

window.addEventListener('keydown', ev => {
  if (!S.debug || ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return
  if (ev.key === 'F10') { ev.preventDefault(); debugStep('block') }
  if (ev.key === 'F11') { ev.preventDefault(); debugStep('token') }
})

on('model', renderDebugButtons)
renderDebugButtons()
