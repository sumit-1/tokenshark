import { $, GET, POST, S, api, el, flow, fmtBytes, toast, emit } from '../core.js'
import { esc } from '../flow.js'
import { resetFlow, validate } from './flowedit.js'
import { renderInspector } from './inspector.js'
import { applyRunBadges, pollRun } from './run.js'
import { renderTab, setTab } from './tabs.js'

export async function refreshModels() {
  S.models = await GET('/api/models')
  const sel = $('#model')
  const cur = sel.value
  sel.innerHTML = ''
  for (const fam of [['causal_lm', 'Text → text'], ['t2i', 'Text → image']]) {
    const g = el('optgroup', { label: fam[1] })
    S.models.filter(m => m.family === fam[0]).forEach(m =>
      g.appendChild(el('option', { value: m.id, textContent: `${m.downloaded ? '✓ ' : '   '}${m.id}  ·  ${m.size_gb ?? '?'} GB${m.loaded ? '  (loaded)' : ''}` })))
    sel.appendChild(g)
  }
  const loaded = S.models.find(m => m.loaded)
  sel.value = loaded ? loaded.id : (cur || sel.value)
  showModelNotes()
}

export function showModelNotes() {
  const m = S.models.find(x => x.id === $('#model').value)
  const pTag = m?.plugin_name ? `<span class="chip">${esc(m.plugin_name)}</span> ` : ''
  $('#model-notes').innerHTML = m ? `${pTag}${esc(m.repo || m.path)} · ${esc(m.notes || '')} · <span class="small">${m.downloaded
    ? `on disk: ${esc(m.folder)} (${m.disk_gb} GB)` : `will download to ${esc(m.folder)}`}</span>` : ''
  $('#del-files').style.display = m?.downloaded && !m.path && !m.loaded ? '' : 'none'
}

$('#del-files').onclick = async () => {
  const m = S.models.find(x => x.id === $('#model').value)
  if (!m || !confirm(`Delete the downloaded files of ${m.id}?\n${m.folder}`)) return
  try { await api('DELETE', `/api/models/${encodeURIComponent(m.id)}/files`); toast(`Deleted ${m.folder}`); refreshModels() }
  catch (e) { toast(e.message, 'error') }
}

$('#model').addEventListener('change', showModelNotes)

$('#load').onclick = async () => {
  try {
    await POST('/api/load', { model_id: $('#model').value, policy: $('#policy').value, keep_layers: +$('#keep').value || 0 })
    pollStatus(true)
  } catch (e) { toast(e.message, 'error') }
}

$('#unload').onclick = async () => { await POST('/api/unload'); S.info = null; S.run = null; flow.setGraph({ nodes: [], edges: [] }); await pollStatus(); document.body.dataset.family = ''; emit('model', null) }

$('#add-model').onclick = () => {
  const dlg = $('#dlg-model')
  dlg.showModal()
}

$('#dlg-model-save').onclick = async ev => {
  ev.preventDefault()
  const f = $('#dlg-model form')
  const spec = Object.fromEntries(new FormData(f).entries())
  for (const k of Object.keys(spec)) if (spec[k] === '') delete spec[k]
  if (spec.size_gb) spec.size_gb = +spec.size_gb
  try { await POST('/api/models', spec); $('#dlg-model').close(); await refreshModels(); $('#model').value = spec.id; toast('Model added') }
  catch (e) { toast(e.message, 'error') }
}

export async function pollStatus(loading) {
  const st = await GET('/api/status')
  S.status = st
  $('#stale').style.display = st.stale ? '' : 'none'
  const m = st.model
  const w = m?.weights?.policy ? m.weights : (m?.weights?.unet || m?.weights?.transformer)
  $('#status').innerHTML = `<b>${esc(st.device)}</b> · ${esc(st.dtype)} · ${m ? `${esc(m.model_id)} · weights ${esc(w?.policy || '')} (${w?.total_gb ?? '?'} GB)` : esc(st.status)}`
  $('#status').className = 'pill ' + (st.status === 'error' ? 'bad' : st.status === 'ready' ? 'ok' : 'busy')
  if (st.status === 'error') { toast(st.error, 'error'); return }
  if (/^(loading|downloading|locating|copying)/.test(st.status)) {
    showLoadProgress(st)
    setTimeout(() => pollStatus(true), 500); return
  }
  showLoadProgress(null)
  if (m && (!S.info || S.info.model_id !== m.model_id || loading)) {
    S.info = m
    S.lora = m.lora || null
    document.body.dataset.family = m.family          // styling only (colours of the canvas); behaviour comes from the schema
    document.body.dataset.denoiser = m.denoiser || ''
    $('#model').value = m.model_id
    emit('model', m)                                 // tabs, palette, debugger buttons and lessons adjust themselves
    await syncWorkspace(true)
    refreshModels()
    toast(`Loaded ${m.model_id} in ${m.load_seconds}s`)
  } else if (m && S.info) {
    S.info.lora = m.lora || null
  }
}

export async function syncWorkspace(initial = false) {
  if (!S.info) {
    // Check if another client or MCP loaded a model on the server
    try {
      const st = await GET('/api/status')
      if (st.model && (!S.info || S.info.model_id !== st.model.model_id)) {
        await pollStatus(true)
      }
    } catch { }
    return
  }
  let ws
  try { ws = await GET(`/api/workspace?since=${initial ? -1 : S.wsVersion}`) } catch { return }
  if (!ws) return
  if (ws.model_id && ws.model_id !== S.info.model_id) {
    await pollStatus(true)
    return
  }
  if (!ws.changed && !initial) return
  const fromSelf = !initial && ws.client_id && ws.client_id === S.clientId
  S.wsVersion = ws.version
  if (fromSelf) return

  S.wsApplying = true
  try {
    if (ws.flow && ws.flow.nodes?.length) {
      const hadNodes = flow.nodes.length > 0
      flow.setGraph(ws.flow)
      if (!hadNodes || initial) setTimeout(() => flow.fit(), 50)
      await validate(false)
      applyRunBadges()
      renderInspector()
      if (S.tab === 'kv') renderTab()
    } else if (initial) {
      await resetFlow()
    }
    if (ws.tab && ws.tab !== S.tab) {
      setTab(ws.tab, true)
    }
    if (ws.active_run_id && ws.active_run_id !== S.run?.id) {
      pollRun(ws.active_run_id)
    }
    if (!initial && ws.actor === 'mcp' && ws.notice) {
      toast(ws.notice)
    }
  } finally {
    S.wsApplying = false
  }
}

export function showLoadProgress(st) {
  // Re-uses the run progress bar under the header for model downloads / copies / loading.
  if (!st) { $('#progress').classList.remove('indet'); if (S.loadingShown) { $('#progress').style.width = '0%'; $('#progress-msg').textContent = ''; S.loadingShown = false } return }
  S.loadingShown = true
  const pr = st.progress
  if (pr && pr.total > 0) {
    const f = Math.min(1, pr.done / pr.total)
    const eta = pr.speed > 0 ? Math.max(0, (pr.total - pr.done) / pr.speed) : null
    $('#progress').style.width = `${(f * 100).toFixed(1)}%`
    const verb = st.status.startsWith('copying') ? 'Copying' : st.status.startsWith('loading') ? 'Reading' : 'Downloading'
    $('#progress-msg').textContent = `${verb} ${pr.what}: ${fmtBytes(pr.done)} / ${fmtBytes(pr.total)} (${(f * 100).toFixed(0)}%)` +
      (pr.speed > 0 ? ` · ${fmtBytes(pr.speed)}/s · ${eta < 90 ? Math.round(eta) + ' s' : Math.round(eta / 60) + ' min'} left` : '')
  } else {
    $('#progress').style.width = '100%'
    $('#progress').classList.add('indet')
    $('#progress-msg').textContent = st.status + (st.load_seconds != null ? ` · ${st.load_seconds}s` : '') + ' · still working (details in the terminal window)'
    return
  }
  $('#progress').classList.remove('indet')
}

// The loaded model can change shape: an adaptation may add a block (or detaching removes it).
export async function refreshInfo() {
  let st
  try { st = await GET('/api/status') } catch { return }
  if (!st.model || !S.info) return
  const before = JSON.stringify(S.info.extra_blocks || [])
  S.info = st.model
  S.lora = st.model.lora || null
  const now = st.model.extra_blocks || []
  if (JSON.stringify(now) !== before) {
    await resetFlow()
    toast(now.length ? `The model now has a new block (Block ${now[0].layer}, after block ${now[0].after}). The flow was reset to show it.` : 'The added block is gone; the flow was reset.')
  }
  applyRunBadges()
}
