import { GET, S, btn, el, flow, on, registerTab } from '../core.js'
import { esc } from '../flow.js'
import { scheduleValidate } from '../ui/flowedit.js'
import { renderInspector } from '../ui/inspector.js'
import { applyRunBadges } from '../ui/run.js'
import { renderTab } from '../ui/tabs.js'

// ---------- runs (session tree)
export async function refreshRuns() { S.runs = await GET('/api/runs'); if (S.tab === 'runs') renderTab() }

export function renderRuns(p) {
  if (!S.runs.length) { p.innerHTML = '<div class="empty">No runs yet.</div>'; return }
  const kids = {}
  S.runs.forEach(r => (kids[r.parent_id || ''] ||= []).push(r))
  const ids = new Set(S.runs.map(r => r.id))
  const roots = S.runs.filter(r => !r.parent_id || !ids.has(r.parent_id))
  const ul = el('div', { className: 'tree' })
  const add = (r, depth) => {
    const row = el('div', { className: 'run' + (S.run?.id === r.id ? ' on' : '') })
    row.style.paddingLeft = 8 + depth * 18 + 'px'
    row.innerHTML = `<b>${esc(r.id)}</b> ${esc(r.label || '')} <span class="small">${esc(r.model_id)} · ${esc(r.status)}${r.seconds ? ` · ${r.seconds}s` : ''}${r.resume_step != null ? ` · from step ${r.resume_step}` : ''}</span> <span class="prev">${esc(r.preview || '')}</span>`
    row.onclick = async () => { S.run = await GET(`/api/runs/${r.id}`); S.lens = null; applyRunBadges(); renderInspector(); renderTab() }
    const rf = btn('Restore flow', async ev => { ev.stopPropagation(); const full = await GET(`/api/runs/${r.id}`); flow.setGraph(full.flow); S.run = full; applyRunBadges(); scheduleValidate() }, 'mini')
    row.appendChild(rf)
    ul.appendChild(row)
    ;(kids[r.id] || []).forEach(c => add(c, depth + 1))
  }
  roots.forEach(r => add(r, 0))
  p.appendChild(ul)
}

registerTab({ id: 'runs', title: 'Runs', render: renderRuns, onOpen: refreshRuns })

on('model', () => refreshRuns())
// Runs can also come from outside this page (the Python API's server, the MCP tutor): keep the list current.
setInterval(async () => {
  if (S.tab !== 'runs') return
  try { const r = await GET('/api/runs'); if (JSON.stringify(r) !== JSON.stringify(S.runs)) { S.runs = r; renderTab() } } catch { }
}, 2500)
