// The results panel: a row of tabs and the panel under it. This module knows nothing about individual tabs:
// each tab registers itself (core.registerTab) and the loaded model's plugin lists the ones it supports.
import { $, S, TABS, el, on, schema } from '../core.js'

const NO_MODEL = ['learn', 'train', 'runs']          // what makes sense before a model is loaded

// Tab modules available for the loaded model, in the plugin's order.
export function tabList() {
  return (S.info ? schema().tabs || [] : NO_MODEL).map(id => TABS[id]).filter(Boolean)
}
const nameOf = t => t.alias || t.id
// The module behind a tab name ('output' → output_text or output_image, whichever this model has).
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

export function setTab(name) {
  const t = resolve(name) || tabList()[0]
  if (!t) return
  S.tab = nameOf(t)
  markNav()
  renderTab()
  t.onOpen?.()
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

// A different model has different tabs: rebuild the row, and leave a tab the new model doesn't have.
on('model', () => { buildNav(); setTab(resolve(S.tab) ? S.tab : 'learn') })
