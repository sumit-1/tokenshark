import { S, can, canRewire, el, flow, hooks, schema } from '../core.js'
import { editCode, hasCode } from './codeedit.js'
import { debugRunTo, debugStart } from './debugger.js'
import { addInitImage, addIntervention, addKV, addPromptKV, bypassBlock, duplicateBlock, resetFlow, schedPos } from './flowedit.js'
import { renderInspector } from './inspector.js'
import { addLayerPrompt } from '../tabs/kv.js'

export function contextMenu(target, x, y, world) {
  closeMenu()
  const items = []
  const n = target?.type === 'node' ? flow.node(target.id) : null
  const term = schema().block_term || 'block'
  if (n) {
    const k = n.data.kind
    if (k === 'block') items.push(['Repeat block', () => duplicateBlock(n)], [n.data.bypass ? 'Restore (un-bypass)' : 'Bypass', () => bypassBlock(n)],
      ['+ Interceptor on this block', () => addIntervention(n)], ['Debug: run to here', () => debugRunTo(n)])
    if (k === 'block' && can('kv_prefill')) items.push(['+ KV prefill only this block reads', () => addLayerPrompt(schedPos(n))])
    if (can('code') && hasCode(k)) items.push([(n.data.code || '').trim() ? 'Edit code… (edited)' : 'View / edit code…', () => editCode(n)])
    if (k === 'unet_block' && canRewire(n)) items.push(['Repeat block', () => duplicateBlock(n)], [n.data.bypass ? 'Restore (un-bypass)' : 'Bypass', () => bypassBlock(n)])
    if (k === 'unet_block') items.push([`+ Interceptor on this ${term}`, () => addIntervention(n)])
    if (k === 'intervention' || k === 'prompt_kv') items.push([n.data.enabled === false ? 'Enable' : 'Disable', () => { flow.updateNodeData(n.id, { enabled: n.data.enabled === false }); renderInspector() }])
    if (k === 'embed' && can('kv_prefix')) items.push(['+ KV-cache prefix', () => addKV()])
    if (k === 'embed' && can('kv_prefill')) items.push(['+ Layer KV prefill (text → KV)', () => addLayerPrompt()])
    if (k === 'kv_text') items.push([n.data.enabled === false ? 'Enable' : 'Disable', () => { flow.updateNodeData(n.id, { enabled: n.data.enabled === false }); renderInspector() }])
    if (k === 'noise' && can('init_image')) items.push(['+ Start image / region lock', () => addInitImage()])
    if (k === 'denoiser_in' && S.info?.capabilities?.prompt_kv) items.push(['+ Other prompt for some blocks', () => addPromptKV('replace')], ['+ Hidden prompt', () => addPromptKV('prepend')])
    items.push(['Delete box', () => flow.removeNode(n.id)])
  } else if (target?.type === 'edge') {
    items.push(['Delete arrow', () => { flow.removeEdge(target.id); flow.select(null) }])
  } else {
    if (!S.info) return
    items.push(['+ Interceptor here', () => addIntervention(null, world)])
    if (can('kv_prefix')) items.push(['+ KV-cache prefix', () => addKV()])
    if (schema().step_debugger) items.push(['Debug: start stepping', () => debugStart()])
    if (can('init_image')) items.push(['+ Start image / region lock', () => addInitImage()])
    if (S.info?.capabilities?.prompt_kv) items.push(['+ Other prompt for some blocks', () => addPromptKV('replace')], ['+ Hidden prompt', () => addPromptKV('prepend')])
    items.push(['Fit to screen', () => flow.fit()], ['Reset flow', () => resetFlow()])
  }
  const m = el('div', { className: 'ctxmenu' })
  items.forEach(([label, fn]) => { const b = el('button', { textContent: label }); b.onclick = () => { closeMenu(); fn() }; m.appendChild(b) })
  m.style.left = Math.min(x, innerWidth - 240) + 'px'; m.style.top = Math.min(y, innerHeight - 30 * items.length - 20) + 'px'
  document.body.appendChild(m)
  setTimeout(() => window.addEventListener('mousedown', closeOnOutside), 0)
}

export function closeOnOutside(ev) { if (!ev.target.closest('.ctxmenu')) closeMenu() }

export function closeMenu() { document.querySelectorAll('.ctxmenu').forEach(m => m.remove()); window.removeEventListener('mousedown', closeOnOutside) }

window.addEventListener('keydown', ev => { if (ev.key === 'Escape') closeMenu() })
hooks.contextMenu = contextMenu
