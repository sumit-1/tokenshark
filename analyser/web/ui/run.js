import { $, GET, POST, S, emit, flow, hasFeature, spark, toast } from '../core.js'
import { esc } from '../flow.js'
import { debugStep, renderDebugButtons } from './debugger.js'
import { validate } from './flowedit.js'
import { renderInspector } from './inspector.js'
import { renderTab } from './tabs.js'
import { renderResume } from '../tabs/output_image.js'
import { refreshRuns } from '../tabs/runs.js'

$('#run').onclick = () => {
  // If we are in the middle of a stepping session, ▶ Run continues to the end instead of restarting from step 0!
  if (S.debug && !S.debug.done) {
    if (S.debug.busy) return toast('Already running (press ■ to stop).')
    return debugStep('all')
  }
  startRun()
}

$('#cancel').onclick = async () => {
  if (S.debug?.busy) return POST('/api/debug/cancel').catch(e => toast(e.message, 'error'))
  if (S.run) await POST(`/api/runs/${S.run.id}/cancel`)
}

$('#pause').onclick = async () => {
  if (!S.run) return
  const paused = S.run.status === 'paused' || S.pauseWanted
  try {
    const r = await POST(`/api/runs/${S.run.id}/${paused ? 'resume' : 'pause'}`)
    S.pauseWanted = !paused
    S.run = { ...S.run, status: r.status }
    $('#pause').textContent = paused ? '⏸' : '▶'
    $('#pause').title = paused ? 'Pause the run' : 'Resume the run'
  } catch (e) { toast(e.message, 'error') }
}

export async function startRun(extra = {}) {
  if (!S.info) return toast('Load a model first.', 'error')
  if (!(await validate())) return toast('Fix the flow first (see the message under the canvas).', 'error')
  S.lastGuessUrl = null; S.lastGuessKey = null
  const body = { flow: flow.getGraph(), label: $('#run-label').value || '', parent_id: S.run?.id || null, ...extra }
  if (S.resume) body.resume = S.resume
  try {
    const r = await POST('/api/runs', body)
    S.resume = null; renderResume()
    S.run = { ...r.run }
    pollRun(r.run.id)
  } catch (e) { toast(e.message, 'error') }
}

export async function pollRun(id) {
  clearTimeout(S.polling)
  if (S.pollId !== id) { S.pollId = id; S.runVersion = -1; S.pauseWanted = false }
  let r
  try { r = await GET(`/api/runs/${id}?since=${S.runVersion}`) }
  catch (e) { S.polling = setTimeout(() => pollRun(id), 2000); return }
  const live = ['running', 'queued', 'paused'].includes(r.status)
  $('#progress').classList.remove('indet')
  $('#progress').style.width = `${Math.round(r.progress * 100)}%`
  $('#progress-msg').textContent = (live ? r.message : r.status === 'done' ? `done in ${r.seconds}s`
    : r.status === 'cancelled' ? `cancelled after ${r.seconds}s${r.outputs ? ' — output so far is kept' : ''}` : r.status) +
    (r.ram_gb != null ? ` · process RAM ${r.ram_gb} GB` : '')
  $('#run').disabled = live
  $('#pause').disabled = !live
  if (r.status === 'paused') S.pauseWanted = false
  $('#pause').textContent = r.status === 'paused' || S.pauseWanted ? '▶' : '⏸'
  $('#pause').title = r.status === 'paused' ? 'Resume the run' : S.pauseWanted ? 'Pausing before the next block…' : 'Pause the run'
  const fresh = r.outputs !== undefined && r.version !== S.runVersion
  const wasPaused = S.run?.status === 'paused'
  if (fresh) { S.runVersion = r.version; S.lens = null; S.lensKey = null }
  S.run = fresh || !S.run || S.run.id !== r.id ? r : { ...S.run, ...r, outputs: S.run.outputs }
  if (hasFeature('image_stepper') && (fresh || !live || (r.status === 'paused' && !wasPaused))) {
    GET('/api/debug').then(dbg => { if (dbg && dbg.run_id === id) { S.debug = dbg; renderDebugButtons(); if (S.tab === 'debug') renderTab() } }).catch(() => {})
  }
  if (live) {
    flow.setActive(r.active_node || null, r.done_nodes || [])
    // Text / image updates appear as they are generated; while paused every tab works on the state so far.
    if (fresh || (r.status === 'paused' && !wasPaused)) { applyRunBadges(); renderInspector(); renderTab() }
    S.polling = setTimeout(() => pollRun(id), r.status === 'paused' ? 1000 : 400); return
  }
  flow.setActive(null, [])
  $('#run').disabled = false
  $('#pause').textContent = '⏸'
  if (r.status === 'error') toast(r.error, 'error')
  applyRunBadges()
  refreshRuns()
  renderInspector()
  renderTab()
  if (r.status === 'done') emit('run-done', r)
}

export function applyRunBadges() {
  const o = S.run?.outputs, b = {}
  const lora = S.info?.lora
  const loraByLayer = {}
  if (lora?.active && lora.layers?.length) {
    lora.layers.forEach(l => { loraByLayer[l.layer] = l })
  }
  // Attach LoRA ΔW badges to block boxes even before a run starts
  flow.nodes.forEach(n => {
    if (n.data?.layer != null && loraByLayer[n.data.layer]) {
      const ld = loraByLayer[n.data.layer]
      const pct = (ld.relative_change * 100).toFixed(2)
      b[n.id] = `<span class="pk" style="background:rgba(255,140,40,${0.25 + 0.65 * ld.heat})" title="LoRA / fine-tune ΔW on layer ${ld.layer}: ${pct}% (${ld.modules.join(', ')})">ΔW ${pct}%</span>`
    }
  })
  if (!o) { flow.setBadges(b); return }
  for (const [nid, st] of Object.entries(o.block_stats || {})) {
    const prev = b[nid] || ''
    if ('resid_norm' in st) b[nid] = prev + `<span class="lens"${st.pending ? ' style="visibility:hidden"' : ''} title="top prediction at the newest token">${esc(st.lens_last ?? '')}</span><i style="width:${Math.min(100, st.mlp_norm / (maxOf(o, 'mlp_norm') || 1) * 100)}%"></i>`
    else {
      const lay = o.prompt_layout?.[st.pos]
      const tag = lay && (lay.reads !== 'main' || lay.prefix?.length)
        ? `<span class="pk" title="what this block's image tokens read">${esc([...(lay.prefix || []), lay.reads === 'main' ? 'prompt' : lay.reads].join(' + '))}</span>` : ''
      b[nid] = prev + tag + spark(st.std_by_step)
    }
  }
  const out = flow.nodes.find(n => n.data.kind === 'output_text' || n.data.kind === 'output_image')
  if (out && o.text !== undefined) b[out.id] = `<span class="outtext">${esc(o.text.slice(0, 120))}</span>`
  const lastPreview = o.preview_steps?.length ? `/api/runs/${S.run.id}/step/${o.preview_steps[o.preview_steps.length - 1]}.png?v=${S.run.version ?? 0}` : ''
  if (out && (o.image || lastPreview)) b[out.id] = `<img src="${o.image || lastPreview}">`
  const embed = flow.nodes.find(n => n.data.kind === 'embed')
  if (embed && o.cache) b[embed.id] = `<span class="small">KV ${o.cache.n_tokens} tok${o.prefix_tokens ? ` (${o.prefix_tokens} imported)` : ''}</span>`
  flow.setBadges(b)
}

export const maxOf = (o, key) => Math.max(...Object.values(o.block_stats).map(s => s[key] || 0))
