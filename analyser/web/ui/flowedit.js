import { $, GET, POST, S, canRewire, download, flow, hooks, isBlock, newId, on, schema, toast } from '../core.js'
import { esc } from '../flow.js'
import { editCode, hasCode } from './codeedit.js'
import { renderInspector } from './inspector.js'
import { renderTab } from './tabs.js'
import { addLayerPrompt } from '../tabs/kv.js'

export async function resetFlow() {
  const g = await GET('/api/flow/default')
  flow.setGraph(g)
  setTimeout(() => flow.fit(), 50)
  S.selected = null
  renderInspector()
  scheduleValidate()
}

export let vTimer = null

export function scheduleValidate() { clearTimeout(vTimer); vTimer = setTimeout(validate, 300) }

export async function validate() {
  if (!S.info) return
  try {
    const r = await POST('/api/flow/compile', { flow: flow.getGraph() })
    const p = r.plan
    S.planNodes = p.schedule_nodes
    flow.setErrors([])
    $('#plan').innerHTML = `<span class="ok-dot"></span> ${p.schedule.length} blocks: [${p.schedule.join(', ')}]` +
      (p.interventions.length ? ` · ${p.interventions.length} intervention(s)` : '') +
      (p.params.prefixes?.length ? ` · ${p.params.prefixes.length} KV prefix${p.params.prefixes.length > 1 ? 'es' : ''} (${p.params.prefixes.map(x => `blocks ${x.layers}`).join(' / ')})` : '') +
      (p.params.code ? ` · ✎ edited code in ${Object.keys(p.params.code).length} box${Object.keys(p.params.code).length > 1 ? 'es' : ''}` : '')
    return true
  } catch (e) {
    const errs = e.data?.errors || [{ message: e.message }]
    flow.setErrors(errs.map(x => x.node).filter(Boolean))
    $('#plan').innerHTML = `<span class="bad-dot"></span> ${esc(errs.map(x => x.message).join(' · '))}`
    return false
  }
}

export function selectedNode() { return S.selected?.type === 'node' ? flow.node(S.selected.id) : null }

// Schedule position of a block box in the last compiled plan (falls back to its layer number).
export function schedPos(b) { const i = (S.planNodes || []).indexOf(b.id); return i >= 0 ? i : b.data.layer }

export function addIntervention(target, world) {
  const at = target ? { x: target.position.x + 10, y: target.position.y - 105 } : (world || flow.center())
  const n = flow.addNode({ id: newId('iv'), type: 'box', position: at, data: {
    kind: 'intervention', label: 'Intervention', hook: schema().default_hook || S.info?.hooks?.[0] || 'out', op: 'zero', value: 0,
    positions: 'all', steps: 'all', heads: 'all', enabled: true } })
  if (target) flow.addEdge({ source: n.id, sourceHandle: 'out', target: target.id, targetHandle: 'mod' })
  flow.select({ type: 'node', id: n.id })
}

export function duplicateBlock(b) {
  if (!canRewire(b)) return toast('Select a transformer block to duplicate.', 'error')
  const copy = flow.addNode({ id: newId(`block_${b.data.layer}`), type: 'box', position: { x: b.position.x + 30, y: b.position.y + 60 },
    data: { ...b.data, label: `Block ${b.data.layer} (repeat)` } })
  const out = flow.edges.filter(e => e.source === b.id && (e.sourceHandle || 'out') === 'out' && (e.targetHandle || 'in') === 'in')
  out.forEach(e => { flow.removeEdge(e.id); flow.addEdge({ source: copy.id, sourceHandle: 'out', target: e.target, targetHandle: 'in' }) })
  flow.addEdge({ source: b.id, sourceHandle: 'out', target: copy.id, targetHandle: 'in' })
  flow.select({ type: 'node', id: copy.id })
}

export function bypassBlock(b) {
  // Bypass keeps the box (dimmed) and its arrows; the compiled schedule simply skips it. Toggle to restore.
  if (!canRewire(b)) return toast('Select a transformer block to bypass.', 'error')
  flow.updateNodeData(b.id, { bypass: !b.data.bypass })
  renderInspector()
}

export function addKV(cacheId) {
  // Each KV-prefix box is one prefix. Several boxes can feed Embed, each visible to its own blocks.
  const embed = flow.nodes.find(n => n.data.kind === 'embed')
  const existing = flow.nodes.filter(x => x.data.kind === 'kv_import')
  let n = existing.find(x => !x.data.cache_id)
  if (n && cacheId) { flow.updateNodeData(n.id, { cache_id: cacheId }) }
  else {
    n = flow.addNode({ id: newId('kv'), type: 'box', position: { x: embed.position.x - 40 + 190 * existing.length, y: embed.position.y - 130 },
      data: { kind: 'kv_import', label: `KV prefix ${existing.length + 1}`, cache_id: cacheId || '', mode: 'shift', offset: 0, layers: 'all' } })
    flow.addEdge({ source: n.id, sourceHandle: 'out', target: embed.id, targetHandle: 'cache' })
    setTimeout(() => flow.fit(), 30)
  }
  flow.select({ type: 'node', id: n.id })
}

export function addInitImage() {
  const noise = flow.nodes.find(n => n.data.kind === 'noise')
  let n = flow.nodes.find(x => x.data.kind === 'init_image')
  if (!n) {
    n = flow.addNode({ id: newId('init'), type: 'box', position: { x: noise.position.x - 200, y: noise.position.y + 10 },
      data: { kind: 'init_image', label: 'Start image', sub: 'img2img · region lock', image: '', strength: 0.6, mask: '', mask_mode: 'keep', lock_steps: 'all', lock_strength: 1 } })
    flow.addEdge({ source: n.id, sourceHandle: 'out', target: noise.id, targetHandle: 'init' })
    setTimeout(() => flow.fit(), 30)
  }
  flow.select({ type: 'node', id: n.id })
}

$('#btn-iv').onclick = () => addIntervention(isBlock(selectedNode()) ? selectedNode() : null)

$('#btn-dup').onclick = () => duplicateBlock(selectedNode())

$('#btn-bypass').onclick = () => bypassBlock(selectedNode())

$('#btn-kv').onclick = () => addKV()

$('#btn-kvt').onclick = () => { const b = selectedNode(); addLayerPrompt(b?.data.kind === 'block' ? schedPos(b) : 'all'); if (S.tab === 'kv') renderTab() }

$('#btn-code').onclick = () => { const b = selectedNode(); if (b && hasCode(b.data.kind)) editCode(b); else toast('Select a block (or Embed, Final norm, LM head, Sampler) to see its code.', 'error') }

$('#btn-pkv').onclick = () => addPromptKV('replace')

$('#btn-hpkv').onclick = () => addPromptKV('prepend')

$('#btn-init').onclick = () => addInitImage()

$('#btn-del').onclick = () => { if (S.selected?.type === 'node') flow.removeNode(S.selected.id); else if (S.selected?.type === 'edge') { flow.removeEdge(S.selected.id); flow.select(null) } }

$('#btn-reset').onclick = () => resetFlow()

$('#btn-fit').onclick = () => flow.fit()

$('#btn-export').onclick = () => download('flow.json', JSON.stringify(flow.getGraph(), null, 1))

$('#btn-import').onclick = () => $('#flow-file').click()

$('#flow-file').onchange = async ev => {
  const f = ev.target.files[0]; if (!f) return
  try {
    const g = JSON.parse(await f.text())
    // A flow file can carry Python for its boxes. It runs on this computer, so ask before keeping it.
    const coded = (g.nodes || []).filter(n => (n.data?.code || '').trim())
    if (coded.length && !confirm(`This flow contains edited Python code for ${coded.length} box${coded.length > 1 ? 'es' : ''} (${coded.map(n => n.data.label || n.id).join(', ')}).\n\nThe code runs on your computer when you press Run. Keep it only if you trust where the file came from.\n\nOK = keep the code · Cancel = open the flow without it`))
      coded.forEach(n => { delete n.data.code })
    flow.setGraph(g); flow.fit(); scheduleValidate()
  } catch (e) { toast('Not a flow file: ' + e.message, 'error') }
  ev.target.value = ''
}

// ---------- per-block prompts (Qwen-Image: the prompt is a KV cache)
export function addPromptKV(role = 'replace') {
  const din = flow.nodes.find(x => x.data.kind === 'denoiser_in')
  if (!din) return
  const existing = flow.nodes.filter(x => x.data.kind === 'prompt_kv')
  const n = flow.addNode({ id: newId('pkv'), type: 'box', position: { x: din.position.x + 200 + 190 * existing.length, y: din.position.y - 120 },
    data: { kind: 'prompt_kv', label: role === 'prepend' ? 'Hidden prompt' : 'Other prompt', role, prompt: role === 'prepend' ? 'in the style of a watercolour painting' : 'a cat',
      blocks: role === 'prepend' ? 'all' : `${Math.floor((S.info?.n_layers || 32) / 2)}:${S.info?.n_layers || 32}`, mode: 'shift', offset: 0, enabled: true } })
  flow.addEdge({ source: n.id, sourceHandle: 'out', target: din.id, targetHandle: 'kv' })
  setTimeout(() => flow.fit(), 30)
  flow.select({ type: 'node', id: n.id })
}
hooks.scheduleValidate = scheduleValidate

// The palette shows the actions the loaded model's plugin declares.
function buildPalette() {
  const actions = schema().palette_actions || []
  document.querySelectorAll('.palette [data-action]').forEach(b => { b.style.display = actions.includes(b.dataset.action) ? '' : 'none' })
}
on('model', buildPalette)
buildPalette()
