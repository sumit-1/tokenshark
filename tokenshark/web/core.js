// Core: the API client, shared state, DOM helpers and the flow canvas. Imports no feature module, so every
// other module can import from here without cycles. Feature modules register callbacks in `hooks`.
import { FlowEditor, esc } from './flow.js'
// Callbacks that feature modules register (the flow canvas calls them; core itself imports no feature).
export const hooks = {}

// ---- tabs: every results tab is a module that registers itself here. Which tabs a model shows is decided by
// its plugin (ui_schema.tabs), not by conditions in the page.
//   registerTab({ id, title, render(panel), alias?, hint?, needs?: 'run', onOpen?, cls? })
// `alias` lets two modules stand behind one name: 'output_text' and 'output_image' are both opened as 'output'.
export const TABS = {}
export function registerTab(tab) { TABS[tab.id] = tab }

// ---- events: modules announce changes instead of calling each other.   on('model', fn) · emit('model', info)
const listeners = {}
export function on(event, fn) { (listeners[event] ||= []).push(fn) }
export function emit(event, data) { for (const fn of listeners[event] || []) { try { fn(data) } catch (e) { console.error(e) } } }

// What the loaded model's plugin declares for the page (tabs, palette actions, train sections, course …).
export const schema = () => S.info?.ui_schema || {}
export const can = action => (schema().palette_actions || []).includes(action)     // a flow-editing action the plugin lists
export const hasFeature = f => (S.info?.features || []).includes(f)                // a backend tool the plugin supports
export const texts = () => schema().texts || ''                                    // which wording set fits this architecture

export async function api(method, url, body, raw) {
  const r = await fetch(url, {
    method, body: raw ? body : body ? JSON.stringify(body) : undefined,
    headers: raw ? { 'Content-Type': 'application/octet-stream' } : body ? { 'Content-Type': 'application/json' } : undefined,
  })
  const t = await r.text()
  let d = null
  try { d = t ? JSON.parse(t) : null } catch { d = { detail: t } }
  if (!r.ok) {
    const e = new Error(d?.detail || d?.errors?.map(x => x.message).join(' · ') || r.statusText)
    e.data = d; throw e
  }
  return d
}

export const GET = u => api('GET', u), POST = (u, b) => api('POST', u, b || {})

export const S = {
  status: null, models: [], info: null, run: null, runs: [], lens: null, caches: [],
  selected: null, resume: null, polling: null, tab: 'learn', alt: null, xtok: 1, xgroup: 'all', xstep: -1,
  attnPos: 0, attnHead: -1,
}

export const $ = s => document.querySelector(s)

export const el = (tag, attrs = {}, html = '') => { const e = document.createElement(tag); Object.assign(e, attrs); if (html) e.innerHTML = html; return e }

export const flow = new FlowEditor($('#canvas'), {
  onContext: (target, x, y, world) => hooks.contextMenu?.(target, x, y, world),
  onSelect: sel => { S.selected = sel; hooks.renderInspector?.() },
  onChange: () => { hooks.scheduleValidate?.(); if (S.selected?.type === 'node') hooks.renderInspectorHeaderOnly?.() },
})

export function toast(msg, kind = 'info') {
  const t = el('div', { className: `toast ${kind}` }, esc(msg))
  $('#toasts').appendChild(t)
  setTimeout(() => t.remove(), kind === 'error' ? 9000 : 3500)
}

export function fmtBytes(b) { return b >= 2 ** 30 ? (b / 2 ** 30).toFixed(2) + ' GB' : b >= 2 ** 20 ? (b / 2 ** 20).toFixed(1) + ' MB' : (b / 1024).toFixed(0) + ' KB' }

export let uid = 0

export const newId = p => `${p}_${Date.now().toString(36)}${(uid++).toString(36)}`

export const isBlock = n => n && (n.data.kind === 'block' || n.data.kind === 'unet_block')

// Rewirable blocks: LLM blocks and the single-stream DiT's blocks (a UNet's stages are tied by skip connections).
export const isDiT = () => texts() === 'dit'            // wording only (lessons, help); behaviour uses can / hasFeature / capabilities

export const canRewire = n => isBlock(n) && !!S.info?.capabilities?.repeat_blocks     // blocks that may be repeated / bypassed

export function download(name, text) {
  const a = el('a', { href: URL.createObjectURL(new Blob([text])), download: name }); a.click()
}

export function btn(text, fn, cls = '') { const b = el('button', { textContent: text, className: cls }); b.onclick = fn; return b }

export const heat = p => `hsla(${Math.round(210 - 170 * (p ?? 0))}, 80%, ${18 + 32 * (p ?? 0)}%, 1)`

export function spark(vals) {
  if (!vals?.length) return ''
  const mx = Math.max(...vals), mn = Math.min(...vals), w = 300, h = 46
  const pts = vals.map((v, i) => `${(i / Math.max(1, vals.length - 1)) * w},${h - 4 - ((v - mn) / (mx - mn || 1)) * (h - 8)}`).join(' ')
  return `<svg class="spark" viewBox="0 0 ${w} ${h}"><polyline points="${pts}" /></svg><div class="small">${mn.toFixed(3)} … ${mx.toFixed(3)}</div>`
}

export const sleep = ms => new Promise(r => setTimeout(r, ms))

export const spinner = (txt = '') => `<span class="spin"></span>${txt ? ` <span class="small">${esc(txt)}</span>` : ''}`
