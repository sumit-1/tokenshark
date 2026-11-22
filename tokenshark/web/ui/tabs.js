// The results panel: a row of tabs and the panel under it. This module knows nothing about individual tabs:
// each tab registers itself (core.registerTab) and the loaded model's plugin + active extensions list the ones it supports.
import { $, GET, POST, S, TABS, el, emit, extEnabled, on, schema, toast } from '../core.js'
import { esc } from '../flow.js'

const NO_MODEL = ['train', 'runs']

// Map tab IDs to the extension ID that provides them (so toggling an extension hides/shows its tab).
const TAB_EXT_MAP = {
  neurons: 'neurons',
  experts: 'moe',
  guardrail: 'guardrail_fork',
  lora: 'lora',
  quant: 'quant',
  train: 'training',
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
  pop.innerHTML = `<div class="small">Activate or deactivate modular feature plugins. Disabled plugins hide their tabs and palette actions. Community plugins can be placed in <code>~/.tokenshark/extensions/</code>.</div>`
  ;(S.extensions || []).forEach(ext => {
    const row = el('label', { className: 'task', style: 'cursor:pointer;display:flex;align-items:flex-start;gap:8px' })
    const cb = el('input', { type: 'checkbox', checked: ext.enabled !== false, style: 'margin-top:3px' })
    cb.onchange = async () => {
      try {
        await POST(`/api/extensions/${encodeURIComponent(ext.id)}/toggle`, { enabled: cb.checked })
        await refreshExtensions()
        if (!resolve(S.tab)) setTab(S.info ? 'output' : 'runs')
      } catch (e) { toast(e.message, 'error') }
    }
    const info = el('span', { className: 'tl' },
      `<b>${esc(ext.name)}</b> <span class="chip">${esc(ext.source || 'builtin')}</span><span class="small" style="display:block">${esc(ext.description || '')}</span>`)
    row.append(cb, info)
    pop.appendChild(row)
  })
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
