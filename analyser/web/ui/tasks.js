import { $, GET, POST, S, btn, el, toast } from '../core.js'
import { esc } from '../flow.js'

// Everything that needs the model is a task: runs, debugger steps, logit lens, word clouds, word tracking.
// The model works on the first task in the queue that isn't paused, one block / layer at a time, so moving a
// task up, pausing or stopping it takes effect within one block.
S.tasksOpen = false

export function placeTasksPop() {
  // fixed to the viewport, kept fully on screen wherever the header wraps the button to
  const pop = $('#tasks-pop'), r = $('#tasks-btn').getBoundingClientRect()
  const w = Math.min(520, innerWidth - 16)
  pop.style.width = w + 'px'
  pop.style.left = Math.max(8, Math.min(r.left, innerWidth - w - 8)) + 'px'
  pop.style.top = (r.bottom + 6) + 'px'
  pop.style.maxHeight = Math.max(160, innerHeight - r.bottom - 20) + 'px'
}

$('#tasks-btn').onclick = () => { S.tasksOpen = !S.tasksOpen; $('#tasks-pop').style.display = S.tasksOpen ? '' : 'none'; if (S.tasksOpen) placeTasksPop(); pollTasks() }

window.addEventListener('resize', () => { if (S.tasksOpen) placeTasksPop() })

document.addEventListener('mousedown', ev => {
  if (S.tasksOpen && !ev.target.closest('#tasks-pop') && !ev.target.closest('#tasks-btn')) { S.tasksOpen = false; $('#tasks-pop').style.display = 'none' }
})

export async function taskAct(id, action) {
  try { await POST(`/api/tasks/${id}/${action}`) } catch (e) { toast(e.message, 'error') }
  pollTasks()
}

export async function pollTasks() {
  clearTimeout(S.tasksTimer)
  let list = []
  try { list = await GET('/api/tasks') } catch { }
  const live = list.filter(t => ['queued', 'running', 'paused', 'stopping'].includes(t.status))
  const b = $('#tasks-btn')
  b.innerHTML = `Tasks${live.length ? ` <b>${live.length}</b>` : ''}`
  b.classList.toggle('busy', live.some(t => t.active))
  if (S.tasksOpen) renderTasks(list)
  S.tasksTimer = setTimeout(pollTasks, live.length || S.tasksOpen ? 800 : 2500)
}

export function renderTasks(list) {
  const pop = $('#tasks-pop')
  const live = list.filter(t => ['queued', 'running', 'paused', 'stopping'].includes(t.status))
  const old = list.filter(t => !live.includes(t))
  const icon = { analysis: '🔍', debug: '🐞', run: '▶' }
  const row = (t, i) => {
    const r = el('div', { className: 'task ' + t.status + (t.active ? ' active' : '') })
    r.innerHTML = `<span class="tq">${t.queue ?? ''}</span><span class="ti">${icon[t.kind] || ''}</span>
      <span class="tl"><b>${esc(t.label)}</b><span class="small">${esc(t.active ? 'working' : t.status)}${t.message ? ' · ' + esc(t.message) : ''} · ${t.seconds}s${t.error ? ' · ' + esc(t.error) : ''}</span>
      <span class="pbar"><i style="width:${Math.round((t.progress || 0) * 100)}%"></i></span></span>`
    const ctl = el('span', { className: 'tc' })
    if (t.queue != null) {
      ctl.append(btn('⤒', () => taskAct(t.id, 'top'), 'mini ghost'), btn('↑', () => taskAct(t.id, 'up'), 'mini ghost'), btn('↓', () => taskAct(t.id, 'down'), 'mini ghost'),
        btn(t.status === 'paused' ? '▶' : '⏸', () => taskAct(t.id, t.status === 'paused' ? 'resume' : 'pause'), 'mini ghost'),
        btn('✕', () => taskAct(t.id, 'cancel'), 'mini ghost'))
      ctl.children[0].title = 'Do this next'; ctl.children[1].title = 'Move up'; ctl.children[2].title = 'Move down'
      ctl.children[3].title = t.status === 'paused' ? 'Resume' : 'Pause (takes effect after the current block / layer)'; ctl.children[4].title = 'Stop'
      if (i === 0) { ctl.children[0].disabled = true; ctl.children[1].disabled = true }
      if (i === live.length - 1) ctl.children[2].disabled = true
    }
    r.appendChild(ctl)
    return r
  }
  pop.innerHTML = `<div class="small">The model works on the first task that isn’t paused, one block or layer at a time — reorder, pause or stop here. Changes take effect after the current block.</div>`
  if (!live.length) pop.appendChild(el('div', { className: 'empty', textContent: 'Nothing running.' }))
  live.forEach((t, i) => pop.appendChild(row(t, i)))
  if (old.length) { pop.appendChild(el('h4', { textContent: 'Recent' })); old.slice(0, 8).forEach((t, i) => pop.appendChild(row(t, i))) }
}

pollTasks()
