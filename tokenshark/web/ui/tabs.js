// The results panel: a row of tabs and the panel under it. This module knows nothing about individual tabs:
// each tab registers itself (core.registerTab) and the loaded model's plugin + active extensions list the ones it supports.
import { $, GET, POST, S, TABS, el, emit, extEnabled, on, schema, toast } from '../core.js'
import { esc } from '../flow.js'

const NO_MODEL = ['train', 'runs']

// Map tab IDs to the extension ID that provides them (so toggling an extension hides/shows its tab).
export const TAB_EXT_MAP = {
  neurons: 'neurons',
  experts: 'moe',
  guardrail: 'guardrail_fork',
  lora: 'lora',
  quant: 'quant',
  train: 'training',
  diag: 'cosine_diag',
  concepts: 'concept_playbook',
  sinks: 'sink_playbook',
  playbooks: 'playbook_studio',
}

export function tabList() {
  const raw = S.info ? schema().tabs || [] : NO_MODEL
  return raw
    .filter(id => !TAB_EXT_MAP[id] || extEnabled(TAB_EXT_MAP[id]))
    .map(id => TABS[id])
    .filter(t => t && (!t.when || !S.info || t.when(S.info)))
}
const nameOf = t => t.alias || t.id
const resolve = name => tabList().find(t => t.id === name || t.alias === name)
export const tabAvailable = name => !!resolve(name)

export function buildNav() {
  const nav = $('#tabs')
  nav.innerHTML = ''
  for (const t of tabList()) {
    const b = el('button', { textContent: t.title, title: t.hint || '', className: t.cls || '' })
    b.dataset.tab = nameOf(t)
    b.onclick = () => setTab(nameOf(t))
    nav.appendChild(b)
  }
  const size = el('button', { id: 'panel-size', className: 'mini ghost', title: 'Make the results panel bigger / smaller',
    textContent: document.body.classList.contains('bigpanel') ? '⤡ smaller' : '⤢ bigger' })
  size.onclick = () => { size.textContent = document.body.classList.toggle('bigpanel') ? '⤡ smaller' : '⤢ bigger' }
  nav.append(el('span', { className: 'grow' }), size)
  markNav()
}
const markNav = () => document.querySelectorAll('#tabs button[data-tab]').forEach(b => b.classList.toggle('on', b.dataset.tab === S.tab))

export function setTab(name, fromSync = false) {
  const t = resolve(name) || tabList()[0]
  if (!t) return
  const prev = S.tab
  S.tab = nameOf(t)
  markNav()
  renderTab()
  t.onOpen?.()
  if (!fromSync && !S.wsApplying && prev !== S.tab && S.info) {
    POST('/api/workspace', { tab: S.tab, client_id: S.clientId, actor: 'browser' }).then(ws => {
      if (ws?.version != null) S.wsVersion = ws.version
    }).catch(() => {})
  }
}

export function renderTab() {
  S.renderSeq = (S.renderSeq || 0) + 1
  const p = $('#panel')
  p.innerHTML = ''
  const t = resolve(S.tab)
  if (!t) return
  if (t.needs === 'run' && !S.run?.outputs) {
    p.innerHTML = `<div class="empty">${S.debug ? 'Press Block ▸ to push the first chunk through the model; this view fills in as you step.' : 'Run the flow to see results here.'}</div>`
    return
  }
  return t.render(p)
}

// ---- Extensions dropdown in header
S.extOpen = false

function placeExtPop() {
  const btnEl = $('#ext-btn'), pop = $('#ext-pop')
  if (!btnEl || !pop) return
  const r = btnEl.getBoundingClientRect()
  const w = Math.min(440, innerWidth - 16)
  pop.style.width = w + 'px'
  pop.style.left = Math.max(8, Math.min(r.left, innerWidth - w - 8)) + 'px'
  pop.style.top = (r.bottom + 6) + 'px'
  pop.style.maxHeight = Math.max(180, innerHeight - r.bottom - 20) + 'px'
}

export async function refreshExtensions() {
  try {
    S.extensions = await GET('/api/extensions')
    for (const ext of S.extensions) {
      if (ext.enabled && ext.web_module && !ext._loaded) {
        ext._loaded = true
        import(ext.web_module).catch(() => {})
      }
    }
    const btnEl = $('#ext-btn')
    if (btnEl) {
      const nOn = S.extensions.filter(x => x.enabled !== false).length
      btnEl.innerHTML = `Extensions <b>${nOn}/${S.extensions.length}</b> ▾`
    }
    if (S.extOpen) renderExtPop()
    buildNav()
    emit('extensions', S.extensions)
  } catch {}
}

function renderExtPop() {
  const pop = $('#ext-pop')
  if (!pop) return
  pop.innerHTML = `<div class="small" style="margin-bottom:6px">Check an extension to unlock its tab or tools without cluttering the default workspace. Grouped into <b>Playbook Views</b> (end-to-end experiment workflows) and <b>Diagnose Views</b> (debugging & geometry).</div>`
  const groups = [
    ['playbook', 'Playbook Views (Workflows & Replication)'],
    ['diagnose', 'Diagnose Views (Debugging & Geometry)'],
  ]
  for (const [vtype, heading] of groups) {
    const items = (S.extensions || []).filter(e => (e.view_type || 'playbook') === vtype)
    if (!items.length) continue
    pop.appendChild(el('div', { className: 'sechead', style: 'margin:8px 0 4px;font-size:11px;text-transform:uppercase;letter-spacing:0.04em', textContent: heading }))
    items.forEach(ext => {
      const row = el('label', { className: 'task', style: 'cursor:pointer;display:flex;align-items:flex-start;gap:8px' })
      const cb = el('input', { type: 'checkbox', checked: ext.enabled !== false, style: 'margin-top:3px' })
      cb.onchange = async () => {
        try {
          await POST(`/api/extensions/${encodeURIComponent(ext.id)}/toggle`, { enabled: cb.checked })
          await refreshExtensions()
          if (cb.checked && ext.tabs?.length && resolve(ext.tabs[0])) {
            setTab(ext.tabs[0])
          } else if (!resolve(S.tab)) {
            setTab(S.info ? 'output' : 'runs')
          }
        } catch (e) { toast(e.message, 'error') }
      }
      const tabBadges = (ext.tabs || []).map(tid => `<span class="chip">tab: ${esc(tid)}</span>`).join(' ')
      const viewBadge = `<span class="chip" style="background:${vtype === 'playbook' ? 'rgba(59,130,246,0.2)' : 'rgba(16,185,129,0.2)'}">${vtype === 'playbook' ? 'Playbook' : 'Diagnose'}</span>`
      const info = el('span', { className: 'tl' },
        `<b>${esc(ext.name)}</b> ${viewBadge} ${tabBadges}<span class="small" style="display:block">${esc(ext.description || ext.summary || '')}</span>`)
      row.append(cb, info)
      pop.appendChild(row)
    })
  }
}

if ($('#ext-btn')) {
  $('#ext-btn').onclick = () => {
    S.extOpen = !S.extOpen
    $('#ext-pop').style.display = S.extOpen ? '' : 'none'
    if (S.extOpen) { placeExtPop(); renderExtPop() }
  }
  window.addEventListener('resize', () => { if (S.extOpen) placeExtPop() })
  document.addEventListener('mousedown', ev => {
    if (S.extOpen && !ev.target.closest('#ext-pop') && !ev.target.closest('#ext-btn')) {
      S.extOpen = false
      $('#ext-pop').style.display = 'none'
    }
  })
}

on('model', () => { buildNav(); setTab(resolve(S.tab) ? S.tab : (S.info ? 'output' : 'runs')) })
