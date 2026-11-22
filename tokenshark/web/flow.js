// FlowEditor - a small dependency-free node editor: boxes, ports, arrows.
// Drag a box by its body. Drag from an output port (filled dot) to an input port (ring) to connect.
// Click an arrow to select it. Delete / Backspace removes the selected box or arrow.
// Drag the background to pan, mouse wheel to zoom.

const HANDLES = {
  input_prompt: [['out', 'source', 'right']],
  tokenizer: [['in', 'target', 'left'], ['out', 'source', 'right']],
  text_encoder: [['in', 'target', 'left'], ['out', 'source', 'right']],
  embed: [['in', 'target', 'left'], ['out', 'source', 'right'], ['cache', 'target', 'top', 'KV'], ['loop', 'target', 'bottom']],
  block: [['in', 'target', 'left'], ['out', 'source', 'right'], ['mod', 'target', 'top', 'mod'], ['fork', 'source', 'bottom', 'fork']],
  final_norm: [['in', 'target', 'left'], ['out', 'source', 'right']],
  lm_head: [['in', 'target', 'left'], ['out', 'source', 'right']],
  sampler: [['in', 'target', 'left'], ['out', 'source', 'right'], ['loop', 'source', 'bottom']],
  output_text: [['in', 'target', 'left']],
  noise: [['init', 'target', 'left', 'init'], ['out', 'source', 'right']],
  scheduler: [['out', 'source', 'right']],
  denoiser_in: [['in', 'target', 'left'], ['cond', 'target', 'top', 'text'], ['kv', 'target', 'top', 'prompt KV'], ['sched', 'target', 'left', 'sched'],
    ['out', 'source', 'right'], ['loop', 'target', 'bottom']],
  unet_block: [['in', 'target', 'left'], ['out', 'source', 'right'], ['mod', 'target', 'top', 'mod']],
  denoiser_out: [['in', 'target', 'left'], ['out', 'source', 'right'], ['loop', 'source', 'bottom']],
  vae_decode: [['in', 'target', 'left'], ['out', 'source', 'right']],
  output_image: [['in', 'target', 'left']],
  intervention: [['out', 'source', 'right']],
  fork_head: [['in', 'target', 'left', 'fork']],
  block_kv: [['out', 'source', 'right']],
  kv_import: [['out', 'source', 'right']],
  prompt_kv: [['out', 'source', 'right']],
  init_image: [['out', 'source', 'right']],
}

export const CUSTOM_HANDLES = {}
export const CUSTOM_NODE_SUB = {}
export function registerNodeHandles(kind, handles, subFn) {
  CUSTOM_HANDLES[kind] = handles
  if (subFn) CUSTOM_NODE_SUB[kind] = subFn
}

function handlesFor(node) {
  const k = node.data.kind
  let hs = (CUSTOM_HANDLES[k] || HANDLES[k] || [['in', 'target', 'left'], ['out', 'source', 'right']]).map(h => ({ id: h[0], type: h[1], side: h[2], label: h[3] }))
  if (k === 'block' && node.data.has_kv !== false) {
    hs.push({ id: 'kv', type: 'target', side: 'top', label: 'KV' })
  }
  if (k === 'unet_block') {
    if (node.id.startsWith('down')) hs.push({ id: 'skip', type: 'source', side: 'bottom' })
    if (node.id.startsWith('up')) hs.push({ id: 'skip', type: 'target', side: 'bottom' })
  }
  return hs
}

const NODE_W = 172

export class FlowEditor {
  constructor(el, { onSelect, onChange, onContext, onBadge } = {}) {
    this.el = el
    this.onBadge = onBadge || (() => {})        // a click on a [data-pick] element inside a box's badge
    this.onSelect = onSelect || (() => {})
    this.onChange = onChange || (() => {})
    this.onContext = onContext || (() => {})
    this.active = null
    this.doneNodes = new Set()
    this.nodes = []
    this.edges = []
    this.zoom = 1
    this.pan = { x: 40, y: 40 }
    this.selected = null      // {type:'node'|'edge', id}
    this.badges = {}
    this.errors = new Set()
    this.dims = {}
    el.classList.add('fe')
    el.innerHTML = `<div class="fe-viewport"><svg class="fe-edges"><defs>
      <marker id="fe-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs><g class="fe-edge-layer"></g><path class="fe-temp" /></svg>
      <div class="fe-nodes"></div></div>
      <div class="fe-hint">Drag ● → ○ to connect · click arrow + Del to remove · Ctrl+Enter to Run</div>
      <div class="fe-zoombar">
        <button type="button" class="mini ghost" data-z="out" title="Zoom out">−</button>
        <span class="fe-zoom-val">100%</span>
        <button type="button" class="mini ghost" data-z="in" title="Zoom in">+</button>
        <button type="button" class="mini ghost" data-z="fit" title="Fit graph to screen">Fit</button>
      </div>`
    this.vp = el.querySelector('.fe-viewport')
    this.svg = el.querySelector('.fe-edges')
    this.edgeLayer = el.querySelector('.fe-edge-layer')
    this.temp = el.querySelector('.fe-temp')
    this.nodeLayer = el.querySelector('.fe-nodes')
    this.zoomVal = el.querySelector('.fe-zoom-val')
    el.querySelectorAll('.fe-zoombar button').forEach(b => {
      b.addEventListener('mousedown', ev => ev.stopPropagation())
      b.addEventListener('click', ev => {
        ev.stopPropagation()
        const z = b.dataset.z
        if (z === 'fit') this.fit()
        else {
          const factor = z === 'in' ? 1.2 : 1 / 1.2
          const r = this.el.getBoundingClientRect()
          const cx = r.left + r.width / 2, cy = r.top + r.height / 2
          const before = this.toWorld(cx, cy)
          this.zoom = Math.max(0.2, Math.min(2, this.zoom * factor))
          this.pan = { x: cx - r.left - before.x * this.zoom, y: cy - r.top - before.y * this.zoom }
          this.applyTransform()
        }
      })
    })
    this._bind()
  }

  // ------------------------------------------------------------ data
  setGraph(g) {
    this.nodes = g.nodes.map(n => ({ ...n, data: { ...n.data } }))
    this.edges = g.edges.map(e => ({ ...e }))
    this.selected = null
    this.render()
  }
  getGraph() { return { nodes: this.nodes.map(n => ({ ...n, data: { ...n.data } })), edges: this.edges.map(e => ({ ...e })) } }
  node(id) { return this.nodes.find(n => n.id === id) }
  updateNodeData(id, patch) {
    const n = this.node(id); if (!n) return
    Object.assign(n.data, patch); this.renderNode(n); this.renderEdges(); this.onChange()
  }
  addNode(n) { this.nodes.push(n); this.render(); this.onChange(); return n }
  addEdge(e) {
    const id = e.id || `e-${e.source}-${e.sourceHandle}-${e.target}-${e.targetHandle}-${Math.random().toString(36).slice(2, 6)}`
    if (this.edges.some(x => x.source === e.source && x.target === e.target && x.sourceHandle === e.sourceHandle && x.targetHandle === e.targetHandle)) return
    this.edges.push({ ...e, id }); this.renderEdges(); this.onChange()
  }
  removeEdge(id) { this.edges = this.edges.filter(e => e.id !== id); this.renderEdges(); this.onChange() }
  removeNode(id) {
    this.nodes = this.nodes.filter(n => n.id !== id)
    this.edges = this.edges.filter(e => e.source !== id && e.target !== id)
    if (this.selected?.id === id) this.select(null)
    this.render(); this.onChange()
  }
  setBadges(b) { this.badges = b || {}; this.nodes.forEach(n => this.renderNode(n)); this.renderEdges() }
  setErrors(ids) { this.errors = new Set(ids || []); this.nodes.forEach(n => this.renderNode(n)) }
  setActive(id, done) {   // debugger: the box about to run, and the boxes already run for this token
    this.active = id; this.doneNodes = new Set(done || [])
    this.nodeLayer.querySelectorAll('.fe-node').forEach(d => {
      d.classList.toggle('active', d.dataset.id === id); d.classList.toggle('ran', this.doneNodes.has(d.dataset.id))
    })
  }
  select(sel) {
    this.selected = sel
    this.nodeLayer.querySelectorAll('.fe-node').forEach(d => d.classList.toggle('sel', sel?.type === 'node' && d.dataset.id === sel.id))
    this.renderEdges()
    this.onSelect(sel)
  }

  // ------------------------------------------------------------ rendering
  render() {
    this.nodeLayer.innerHTML = ''
    for (const n of this.nodes) {
      const d = document.createElement('div')
      d.className = 'fe-node'
      d.dataset.id = n.id
      this.nodeLayer.appendChild(d)
      this.renderNode(n)
    }
    this.applyTransform()
    requestAnimationFrame(() => this.renderEdges())
  }

  renderNode(n) {
    const d = this.nodeLayer.querySelector(`.fe-node[data-id="${CSS.escape(n.id)}"]`)
    if (!d) return
    const k = n.data.kind
    d.className = `fe-node kind-${k}${this.selected?.type === 'node' && this.selected.id === n.id ? ' sel' : ''}${this.errors.has(n.id) ? ' err' : ''}${n.data.enabled === false ? ' off' : ''}${n.data.bypass ? ' bypassed' : ''}${(n.data.code || '').trim() ? ' coded' : ''}${this.active === n.id ? ' active' : ''}${this.doneNodes.has(n.id) ? ' ran' : ''}${n.data.moe ? ' moe' : ''}${k === 'block' && n.data.has_kv === false ? ' linear-block' : ''}`
    d.style.left = n.position.x + 'px'
    d.style.top = n.position.y + 'px'
    d.style.width = NODE_W + 'px'
    const badge = this.badges[n.id]
    const bkvOpLabel = {
      replace_system: 'replace system', append_system: 'append system', prepend_system: 'prepend system',
      replace_user: 'replace user', append_user: 'append user', prepend_user: 'prepend user',
    }
    const sub = n.data.bypass ? 'bypassed · the stream skips this block'
      : CUSTOM_NODE_SUB[k] ? CUSTOM_NODE_SUB[k](n.data)
      : k === 'fork_head' ? `${n.data.mode || 'monitor'} · ${n.data.target_category || 'powershell'}${n.data.head_id ? ` · ${n.data.head_id}` : ''}`
      : k === 'intervention' ? ivSummary(n.data)
      : k === 'block_kv' ? `${bkvOpLabel[n.data.op] || n.data.op || 'replace system'}${n.data.layers ? ` · blocks ${n.data.layers}` : ''} · “${(n.data.text || '').slice(0, 26)}”`
      : k === 'kv_import' ? `${n.data.cache_id ? n.data.mode || 'shift' : 'no cache chosen'} · blocks ${n.data.layers || 'all'}`
      : (n.data.code || '').trim() ? `✎ edited code${n.data.sub ? ' · ' + n.data.sub : ''}`
      : k === 'init_image' ? (n.data.image ? `${n.data.guide ? 'sketch guide · ' : ''}strength ${Number(n.data.strength ?? 0.6).toFixed(2)}${n.data.mask ? ` · region ${n.data.mask_mode || 'keep'}` : ''} · click to edit` : 'no picture yet · click to add one')
      : k === 'prompt_kv' ? `${n.data.role === 'prepend' ? `hidden · ${n.data.mode || 'shift'}` : 'replaces the prompt'} · blocks ${n.data.blocks || 'all'} · “${(n.data.prompt || '').slice(0, 28)}”`
      : k === 'input_prompt' && n.data.history?.length ? `${n.data.history.length} earlier turn${n.data.history.length > 1 ? 's' : ''} · then the prompt`
      : (n.data.sub || '')
    d.innerHTML = `<div class="fe-title">${esc(n.data.label || k)}</div><div class="fe-sub">${esc(sub)}</div>${badge ? `<div class="fe-badge">${badge}</div>` : ''}`
    const hs = handlesFor(n)
    const bySide = {}
    hs.forEach(h => (bySide[h.side] ||= []).push(h))
    for (const [side, list] of Object.entries(bySide)) {
      list.forEach((h, i) => {
        const p = document.createElement('div')
        p.className = `fe-port ${h.type} side-${side}`
        p.dataset.handle = h.id
        p.dataset.type = h.type
        p.dataset.node = n.id
        p.title = `${h.id} (${h.type === 'source' ? 'output' : 'input'})`
        const frac = (i + 1) / (list.length + 1)
        if (side === 'left' || side === 'right') p.style.top = `calc(${frac * 100}% - 6px)`
        else p.style.left = `calc(${frac * 100}% - 6px)`
        if (h.label) { const l = document.createElement('span'); l.className = 'fe-port-label'; l.textContent = h.label; p.appendChild(l) }
        d.appendChild(p)
      })
    }
    this.dims[n.id] = { w: NODE_W, h: d.offsetHeight || 64 }
  }

  portPos(nodeId, handleId, type) {
    const n = this.node(nodeId); if (!n) return null
    const hs = handlesFor(n)
    const h = hs.find(x => x.id === handleId && (!type || x.type === type)) || hs.find(x => x.id === handleId)
    if (!h) return null
    const dim = this.dims[nodeId] || { w: NODE_W, h: 64 }
    const same = hs.filter(x => x.side === h.side)
    const frac = (same.indexOf(h) + 1) / (same.length + 1)
    const { x, y } = n.position
    if (h.side === 'left') return { x, y: y + dim.h * frac, dx: -1, dy: 0 }
    if (h.side === 'right') return { x: x + dim.w, y: y + dim.h * frac, dx: 1, dy: 0 }
    if (h.side === 'top') return { x: x + dim.w * frac, y, dx: 0, dy: -1 }
    return { x: x + dim.w * frac, y: y + dim.h, dx: 0, dy: 1 }
  }

  renderEdges() {
    const parts = []
    for (const e of this.edges) {
      const a = this.portPos(e.source, e.sourceHandle || 'out', 'source')
      const b = this.portPos(e.target, e.targetHandle || 'in', 'target')
      if (!a || !b) continue
      const d = curve(a, b)
      const cls = ['fe-edge']
      if (e.data?.loop) cls.push('loop')
      if (e.data?.skip) cls.push('skip')
      if (e.targetHandle === 'mod') cls.push('mod')
      if (e.targetHandle === 'kv') cls.push('kv')
      if (e.sourceHandle === 'fork' || e.targetHandle === 'fork') cls.push('fork')
      if (e.targetHandle === 'cache' || e.targetHandle === 'init') cls.push('aux')
      if (this.selected?.type === 'edge' && this.selected.id === e.id) cls.push('sel')
      const label = e.data?.label ? `<text class="fe-edge-label" x="${(a.x + b.x) / 2}" y="${(a.y + b.y) / 2 - 6 + (e.data?.loop ? 46 : 0)}">${esc(e.data.label)}</text>` : ''
      parts.push(`<g class="${cls.join(' ')}" data-id="${esc(e.id)}"><path class="hit" d="${d}"/><path class="line" d="${d}" marker-end="url(#fe-arrow)"/>${label}</g>`)
    }
    this.edgeLayer.innerHTML = parts.join('')
  }

  applyTransform() {
    this.vp.style.transform = `translate(${this.pan.x}px, ${this.pan.y}px) scale(${this.zoom})`
    if (this.zoomVal) this.zoomVal.textContent = `${Math.round(this.zoom * 100)}%`
  }

  fit() {
    if (!this.nodes.length) return
    const xs = this.nodes.map(n => n.position.x), ys = this.nodes.map(n => n.position.y)
    const minX = Math.min(...xs), minY = Math.min(...ys)
    const maxX = Math.max(...xs) + NODE_W, maxY = Math.max(...ys) + 90
    const r = this.el.getBoundingClientRect()
    this.zoom = Math.max(0.25, Math.min(1.2, Math.min((r.width - 60) / (maxX - minX), (r.height - 60) / (maxY - minY))))
    this.pan = { x: 30 - minX * this.zoom, y: 30 - minY * this.zoom }
    this.applyTransform()
  }

  toWorld(cx, cy) {
    const r = this.el.getBoundingClientRect()
    return { x: (cx - r.left - this.pan.x) / this.zoom, y: (cy - r.top - this.pan.y) / this.zoom }
  }

  center() {
    const r = this.el.getBoundingClientRect()
    return this.toWorld(r.left + r.width / 2, r.top + r.height / 3)
  }

  // ------------------------------------------------------------ interaction
  _bind() {
    let drag = null
    this.el.addEventListener('mousedown', ev => {
      if (ev.button !== 0) return
      const port = ev.target.closest('.fe-port')
      const nodeEl = ev.target.closest('.fe-node')
      const edgeEl = ev.target.closest('.fe-edge')
      if (port && port.dataset.type === 'source') {
        drag = { kind: 'connect', from: port.dataset.node, handle: port.dataset.handle }
        ev.preventDefault(); return
      }
      if (port && port.dataset.type === 'target') {   // drag an existing arrow off its input to re-wire it
        const e = [...this.edges].reverse().find(x => x.target === port.dataset.node && (x.targetHandle || 'in') === port.dataset.handle)
        if (e) { this.removeEdge(e.id); drag = { kind: 'connect', from: e.source, handle: e.sourceHandle || 'out' }; ev.preventDefault(); return }
      }
      if (nodeEl) {
        const n = this.node(nodeEl.dataset.id)
        const w = this.toWorld(ev.clientX, ev.clientY)
        drag = { kind: 'node', id: n.id, ox: w.x - n.position.x, oy: w.y - n.position.y, moved: false }
        const pick = ev.target.closest('[data-pick]')
        this.select({ type: 'node', id: n.id })
        if (pick) this.onBadge(n.id, { ...pick.dataset })      // e.g. one expert cell of a mixture-of-experts block
        return
      }
      if (edgeEl) { this.select({ type: 'edge', id: edgeEl.dataset.id }); return }
      drag = { kind: 'pan', sx: ev.clientX, sy: ev.clientY, px: this.pan.x, py: this.pan.y, moved: false }
    })
    window.addEventListener('mousemove', ev => {
      if (!drag) return
      if (drag.kind === 'node') {
        const n = this.node(drag.id); const w = this.toWorld(ev.clientX, ev.clientY)
        n.position = { x: Math.round((w.x - drag.ox) / 10) * 10, y: Math.round((w.y - drag.oy) / 10) * 10 }
        drag.moved = true
        const d = this.nodeLayer.querySelector(`.fe-node[data-id="${CSS.escape(n.id)}"]`)
        d.style.left = n.position.x + 'px'; d.style.top = n.position.y + 'px'
        this.renderEdges()
      } else if (drag.kind === 'pan') {
        this.pan = { x: drag.px + ev.clientX - drag.sx, y: drag.py + ev.clientY - drag.sy }
        drag.moved = drag.moved || Math.abs(ev.clientX - drag.sx) + Math.abs(ev.clientY - drag.sy) > 3
        this.applyTransform()
      } else if (drag.kind === 'connect') {
        const a = this.portPos(drag.from, drag.handle, 'source')
        const w = this.toWorld(ev.clientX, ev.clientY)
        this.temp.setAttribute('d', curve(a, { x: w.x, y: w.y, dx: -a.dx, dy: -a.dy }))
        this.el.querySelectorAll('.fe-port.target').forEach(p => p.classList.toggle('hot', p === ev.target.closest('.fe-port.target')))
      }
    })
    window.addEventListener('mouseup', ev => {
      if (!drag) return
      if (drag.kind === 'connect') {
        this.temp.setAttribute('d', '')
        const port = ev.target.closest && ev.target.closest('.fe-port.target')
        if (port && port.dataset.node !== drag.from) {
          const loop = drag.handle === 'loop' || port.dataset.handle === 'loop'
          const skip = drag.handle === 'skip'
          const data = loop ? { loop: true } : skip ? { skip: true } : undefined
          this.addEdge({ source: drag.from, sourceHandle: drag.handle, target: port.dataset.node, targetHandle: port.dataset.handle, data })
        }
        this.el.querySelectorAll('.fe-port.hot').forEach(p => p.classList.remove('hot'))
      } else if (drag.kind === 'node' && drag.moved) this.onChange()
      else if (drag.kind === 'pan' && !drag.moved) this.select(null)
      drag = null
    })
    this.el.addEventListener('contextmenu', ev => {
      ev.preventDefault()
      const nodeEl = ev.target.closest('.fe-node')
      const edgeEl = ev.target.closest('.fe-edge')
      const world = this.toWorld(ev.clientX, ev.clientY)
      if (nodeEl) { this.select({ type: 'node', id: nodeEl.dataset.id }); this.onContext({ type: 'node', id: nodeEl.dataset.id }, ev.clientX, ev.clientY, world) }
      else if (edgeEl) { this.select({ type: 'edge', id: edgeEl.dataset.id }); this.onContext({ type: 'edge', id: edgeEl.dataset.id }, ev.clientX, ev.clientY, world) }
      else this.onContext(null, ev.clientX, ev.clientY, world)
    })
    this.el.addEventListener('wheel', ev => {
      ev.preventDefault()
      const before = this.toWorld(ev.clientX, ev.clientY)
      this.zoom = Math.max(0.2, Math.min(2, this.zoom * (ev.deltaY < 0 ? 1.1 : 1 / 1.1)))
      const r = this.el.getBoundingClientRect()
      this.pan = { x: ev.clientX - r.left - before.x * this.zoom, y: ev.clientY - r.top - before.y * this.zoom }
      this.applyTransform()
    }, { passive: false })
    window.addEventListener('keydown', ev => {
      if (!this.selected || ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return
      if (ev.key === 'Delete' || ev.key === 'Backspace') {
        ev.preventDefault()
        if (this.selected.type === 'edge') { this.removeEdge(this.selected.id); this.select(null) } else this.removeNode(this.selected.id)
      }
    })
  }
}

function curve(a, b) {
  const dist = Math.max(40, Math.hypot(b.x - a.x, b.y - a.y) * 0.4)
  const c1 = { x: a.x + a.dx * dist, y: a.y + a.dy * dist }
  const c2 = { x: b.x + b.dx * dist, y: b.y + b.dy * dist }
  return `M${a.x},${a.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${b.x},${b.y}`
}

export function ivSummary(d) {
  if (d.op === 'wire') {
    const th = d.threshold ? ` (>${d.threshold})` : ''
    const pos = d.positions && d.positions !== 'all' ? ` · tok ${d.positions}` : ''
    return `wire ${d.source_neurons || '?'} → ${d.neurons || '?'} ×${d.value ?? 3}${th}${pos}`
  }
  const v = d.op === 'scale' || d.op === 'add_noise' ? ` ${d.value ?? 1}`
    : d.op === 'quantize' ? ` ${d.value ?? 4}-bit`
    : d.op === 'patch' ? ` ← run ${d.ref || '?'}` : d.op === 'steer' ? ` ×${d.value ?? 1}`
    : d.op === 'project_out' ? ` ⊥ ×${d.value ?? 1}`
    : d.op === 'clamp' ? ` → ${d.clamp === 'value' ? d.value : `${d.value && d.value !== 1 ? d.value + '× ' : ''}${d.clamp || 'max'}`}` : ''
  const units = (d.experts && d.experts !== 'all' ? ` expert ${d.experts}` : '')
    + (d.neurons && d.neurons !== 'all' ? ` ${d.hook === 'mlp_neurons' ? 'neuron' : 'dim'} ${d.neurons}` : '')
  const where = (['attn_heads', 'q', 'k', 'v', 'kv_k', 'kv_v'].includes(d.hook) && d.heads && d.heads !== 'all' ? ` heads ${d.heads}` : '') + units
  const steps = d.steps && d.steps !== 'all' ? ` · steps ${d.steps}` : ''
  const pos = d.hook !== 'weights' && d.positions && d.positions !== 'all' ? ` · tok ${d.positions}` : ''
  const region = d.region ? ' · region' : ''
  return `${d.op || 'scale'}${v} @ ${d.hook || 'out'}${where}${pos}${steps}${region}`
}

export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
}
