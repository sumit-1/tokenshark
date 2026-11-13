// Paint — a small MS-Paint-style editor (no dependencies).
//
//   openPaint({ mode, title, width, height, background, value, backgrounds, onSave })
//
//   mode 'image': you paint the picture itself (start image for img2img). value = current image (data URL).
//   mode 'mask' : you paint a region over a reference picture (region lock, region interventions).
//                 value = current mask (white = painted, transparent = not); background = the picture under it.
//   backgrounds : [[label, dataURL], …] offered as "load from…" buttons.
//   onSave(dataURL | '') is called on Save ('' = empty mask).

const PALETTE = ['#000000', '#808080', '#800000', '#808000', '#008000', '#008080', '#000080', '#800080', '#808040', '#004040',
  '#0080ff', '#004080', '#8000ff', '#804000', '#ffffff', '#c0c0c0', '#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
  '#ff00ff', '#ffff80', '#00ff80', '#80ffff', '#8080ff', '#ff0080', '#ff8040']
const TOOLS = [
  ['soft', '☁️', 'Soft / blurry brush (feathered edges & gentle opacity — hints shapes and shadows without hard borders)'],
  ['brush', '🖌️', 'Brush (round, solid edge)'], ['pencil', '✏️', 'Pencil (1–2 px, hard edge)'], ['eraser', '🧽', 'Eraser'],
  ['fill', '🪣', 'Fill (flood fill similar colours; in mask mode it follows the picture underneath)'],
  ['line', '╱', 'Line'], ['rect', '▭', 'Rectangle'], ['rectf', '■', 'Filled rectangle'], ['ellipse', '◯', 'Ellipse'],
  ['ellipsef', '●', 'Filled ellipse'], ['picker', '💧', 'Pick colour'],
]
const MASK_RGB = [255, 60, 60]

const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag)
  for (const [k, v] of Object.entries(attrs)) { if (k === 'style') e.style.cssText = v; else if (k.startsWith('on')) e[k] = v; else e.setAttribute(k, v) }
  kids.forEach(k => e.append(k))
  return e
}
const loadImg = src => new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = src })
const hex = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16))
let NOISE_URL = null
function noiseTexture() {        // grey static, shown behind the "noise canvas" (unpainted = left to the model)
  if (NOISE_URL) return NOISE_URL
  const c = document.createElement('canvas'); c.width = c.height = 128
  const g = c.getContext('2d'), d = g.createImageData(128, 128)
  for (let i = 0; i < d.data.length; i += 4) { const v = 95 + Math.random() * 60; d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255 }
  g.putImageData(d, 0, 0)
  return (NOISE_URL = c.toDataURL())
}
async function hasAlpha(src) {
  try {
    const i = await loadImg(src), c = document.createElement('canvas'); c.width = i.naturalWidth; c.height = i.naturalHeight
    const g = c.getContext('2d'); g.drawImage(i, 0, 0)
    const d = g.getImageData(0, 0, c.width, c.height).data
    for (let k = 3; k < d.length; k += 4) if (d[k] < 250) return true
  } catch { }
  return false
}

export async function openPaint(opts) {
  const mode = opts.mode || 'image'
  const isMask = mode === 'mask'
  let W = Math.max(16, Math.round(opts.width || 512)), H = Math.max(16, Math.round(opts.height || 512))
  if (isMask && opts.background) { try { const b = await loadImg(opts.background); W = b.naturalWidth; H = b.naturalHeight } catch { } }
  const st = { tool: isMask ? 'brush' : 'soft', size: isMask ? 24 : 36, opacity: 0.45, feather: 0.7, fg: '#000000', bg: '#ffffff', tol: 32, zoom: 0, undo: [], redo: [] }
  // image mode: "noise canvas" = unpainted pixels stay transparent and mean "nothing here — let the model invent it"
  st.noise = !isMask && (opts.value ? await hasAlpha(opts.value) : opts.noiseCanvas !== false)

  // ---- canvases: base (picture), mask (mask mode), overlay (shape preview)
  const base = h('canvas', { width: W, height: H, class: 'pt-base' })
  const mask = h('canvas', { width: W, height: H, class: 'pt-mask' })
  const over = h('canvas', { width: W, height: H, class: 'pt-over' })
  const bg = base.getContext('2d', { willReadFrequently: true }), mg = mask.getContext('2d', { willReadFrequently: true })
  const og = over.getContext('2d')
  const target = () => isMask ? mask : base
  const tg = () => isMask ? mg : bg
  if (!st.noise) { bg.fillStyle = '#ffffff'; bg.fillRect(0, 0, W, H) }
  const drawOn = async (g, src, w = W, hh = H) => { try { const i = await loadImg(src); g.drawImage(i, 0, 0, w, hh) } catch { } }
  if (isMask) {
    bg.fillStyle = '#ffffff'; bg.fillRect(0, 0, W, H)
    if (opts.background) await drawOn(bg, opts.background)
    if (opts.value) { await drawOn(mg, opts.value); tintMask() }
  } else if (opts.value) await drawOn(bg, opts.value)

  function tintMask() {
    const d = mg.getImageData(0, 0, W, H)
    for (let i = 0; i < d.data.length; i += 4) {
      const on = d.data[i + 3] > 20 && Math.max(d.data[i], d.data[i + 1], d.data[i + 2]) > 20
      d.data[i] = MASK_RGB[0]; d.data[i + 1] = MASK_RGB[1]; d.data[i + 2] = MASK_RGB[2]; d.data[i + 3] = on ? 255 : 0
    }
    mg.putImageData(d, 0, 0)
  }

  // ---- dialog
  const dlg = h('dialog', { class: 'paint' })
  const stage = h('div', { class: 'pt-stage' }, base, ...(isMask ? [mask] : []), over)
  const scroller = h('div', { class: 'pt-scroll' }, stage)
  const toolbar = h('div', { class: 'pt-tools' })
  TOOLS.filter(([id]) => !(isMask && id === 'picker')).forEach(([id, icon, title]) => {
    const b = h('button', { title, 'data-tool': id, onclick: () => { st.tool = id; sync() } }, icon)
    toolbar.append(b)
  })
  const size = h('input', { type: 'range', min: 1, max: 140, value: st.size, title: 'brush size', oninput: () => { st.size = +size.value; sizeLbl.textContent = st.size } })
  const sizeLbl = h('span', { class: 'small' }, String(st.size))
  const opac = h('input', { type: 'range', min: 5, max: 100, value: Math.round(st.opacity * 100), title: 'opacity (lower = subtle shadow / colour hint)', oninput: () => { st.opacity = +opac.value / 100; opacLbl.textContent = `${Math.round(st.opacity * 100)}%` } })
  const opacLbl = h('span', { class: 'small' }, `${Math.round(st.opacity * 100)}%`)
  const blur = h('input', { type: 'range', min: 0, max: 100, value: Math.round(st.feather * 100), title: 'softness / feather (higher = blurrier edges)', oninput: () => { st.feather = +blur.value / 100; blurLbl.textContent = `${Math.round(st.feather * 100)}%` } })
  const blurLbl = h('span', { class: 'small' }, `${Math.round(st.feather * 100)}%`)
  const tol = h('input', { type: 'range', min: 0, max: 128, value: st.tol, title: 'fill tolerance', oninput: () => { st.tol = +tol.value } })
  toolbar.append(h('div', { class: 'small' }, 'size'), size, sizeLbl)
  if (!isMask) toolbar.append(h('div', { class: 'small' }, 'opacity'), opac, opacLbl, h('div', { class: 'small' }, 'blur'), blur, blurLbl)
  toolbar.append(h('div', { class: 'small' }, 'fill tol'), tol)

  const fgSw = h('div', { class: 'pt-sw fg', title: 'colour (left click a swatch)' }), bgSw = h('div', { class: 'pt-sw bg', title: 'background / right-click colour' })
  const pal = h('div', { class: 'pt-pal' })
  if (!isMask) {
    PALETTE.forEach(c => {
      const s = h('div', { class: 'pt-c', style: `background:${c}`, title: c })
      s.onmousedown = ev => { ev.preventDefault(); if (ev.button === 2) st.bg = c; else st.fg = c; sync() }
      s.oncontextmenu = ev => ev.preventDefault()
      pal.append(s)
    })
    const custom = h('input', { type: 'color', value: st.fg, title: 'custom colour', oninput: () => { st.fg = custom.value; sync() } })
    pal.prepend(h('div', { class: 'pt-sws' }, fgSw, bgSw), custom)
  } else {
    pal.append(h('span', { class: 'small' }, 'Paint = region (red). Eraser / right button = remove. Fill follows similar colours in the picture underneath.'),
      h('button', { class: 'mini', onclick: () => { snapshot(); mg.fillStyle = `rgb(${MASK_RGB})`; mg.fillRect(0, 0, W, H) } }, 'Select all'),
      h('button', { class: 'mini', onclick: () => { snapshot(); invertMask() } }, 'Invert'))
  }

  const top = h('div', { class: 'pt-top' },
    h('b', {}, opts.title || (isMask ? 'Paint a region' : 'Paint')), h('span', { class: 'small' }, `${W}×${H}`),
    h('button', { title: 'Undo (Ctrl+Z)', onclick: undo }, '↶'), h('button', { title: 'Redo (Ctrl+Y)', onclick: redo }, '↷'),
    h('button', { onclick: () => { snapshot(); clearAll() } }, isMask ? 'Clear region' : 'Clear'),
    h('button', { onclick: () => { st.zoom = (st.zoom + 1) % 3; fit() }, title: 'fit · 1:1 · 2×' }, '🔍'))
  if (!isMask) {
    const canvasBtn = h('button', { title: 'Noise canvas: what you leave unpainted is pure noise for the model to fill in. White canvas: a white page is part of the picture (the model keeps it white).' })
    const setCanvasLabel = () => { canvasBtn.textContent = st.noise ? 'Canvas: noise ▸ white' : 'Canvas: white ▸ noise'; stage.style.backgroundImage = st.noise ? `url(${noiseTexture()})` : ''; stage.style.backgroundColor = st.noise ? '' : '#fff' }
    canvasBtn.onclick = () => {
      snapshot()
      if (st.noise) { bg.globalCompositeOperation = 'destination-over'; bg.fillStyle = '#ffffff'; bg.fillRect(0, 0, W, H); bg.globalCompositeOperation = 'source-over' }
      else {      // white → noise: near-white pixels become "empty"
        const d = bg.getImageData(0, 0, W, H)
        for (let i = 0; i < d.data.length; i += 4) if (d.data[i] > 235 && d.data[i + 1] > 235 && d.data[i + 2] > 235) d.data[i + 3] = 0
        bg.putImageData(d, 0, 0)
      }
      st.noise = !st.noise; setCanvasLabel()
    }
    top.append(canvasBtn)
    requestAnimationFrame(setCanvasLabel)
    const file = h('input', { type: 'file', accept: 'image/*', style: 'display:none' })
    file.onchange = () => { const r = new FileReader(); r.onload = async () => { snapshot(); await drawOn(bg, r.result) }; r.readAsDataURL(file.files[0]) }
    top.append(h('button', { onclick: () => file.click() }, 'Open file…'), file)
  }
  ;(opts.backgrounds || []).filter(([, src]) => src).forEach(([label, src]) => top.append(h('button', {
    onclick: async () => { snapshot(true); if (isMask) { bg.fillStyle = '#fff'; bg.fillRect(0, 0, W, H) } await drawOn(bg, src) },
  }, label)))
  const save = h('button', { class: 'primary', onclick: () => { opts.onSave?.(result(), { noise: st.noise }); dlg.close() } }, 'Save')
  const cancel = h('button', { onclick: () => dlg.close() }, 'Cancel')
  const status = h('span', { class: 'small pt-status' })
  dlg.append(top, h('div', { class: 'pt-body' }, toolbar, scroller), pal, h('div', { class: 'pt-foot' }, status, cancel, save))
  document.body.append(dlg)
  dlg.addEventListener('close', () => { window.removeEventListener('keydown', keys); dlg.remove() })
  dlg.showModal()

  function sync() {
    toolbar.querySelectorAll('[data-tool]').forEach(b => b.classList.toggle('on', b.dataset.tool === st.tool))
    fgSw.style.background = st.fg; bgSw.style.background = st.bg
    stage.style.cursor = st.tool === 'picker' ? 'copy' : st.tool === 'fill' ? 'cell' : 'crosshair'
  }
  function fit() {
    const maxW = scroller.clientWidth - 8 || 700, maxH = scroller.clientHeight - 8 || 500
    const z = st.zoom === 1 ? 1 : st.zoom === 2 ? 2 : Math.min(maxW / W, maxH / H, 4)
    stage.style.width = `${W * z}px`; stage.style.height = `${H * z}px`
  }
  sync(); requestAnimationFrame(fit)

  // ---- history
  function snapshot(both) {
    st.undo.push({ b: bg.getImageData(0, 0, W, H), m: isMask ? mg.getImageData(0, 0, W, H) : null })
    if (st.undo.length > 40) st.undo.shift()
    st.redo = []
  }
  function restore(s) { bg.putImageData(s.b, 0, 0); if (s.m) mg.putImageData(s.m, 0, 0) }
  function cur() { return { b: bg.getImageData(0, 0, W, H), m: isMask ? mg.getImageData(0, 0, W, H) : null } }
  function undo() { if (!st.undo.length) return; st.redo.push(cur()); restore(st.undo.pop()) }
  function redo() { if (!st.redo.length) return; st.undo.push(cur()); restore(st.redo.pop()) }
  function clearAll() { if (isMask || st.noise) (isMask ? mg : bg).clearRect(0, 0, W, H); else { bg.fillStyle = st.bg; bg.fillRect(0, 0, W, H) } }
  function invertMask() {
    const d = mg.getImageData(0, 0, W, H)
    for (let i = 0; i < d.data.length; i += 4) { d.data[i] = MASK_RGB[0]; d.data[i + 1] = MASK_RGB[1]; d.data[i + 2] = MASK_RGB[2]; d.data[i + 3] = d.data[i + 3] > 0 ? 0 : 255 }
    mg.putImageData(d, 0, 0)
  }
  const keys = ev => {
    if (!(ev.ctrlKey || ev.metaKey)) return
    if (ev.key === 'z') { ev.preventDefault(); undo() }
    if (ev.key === 'y' || (ev.key === 'Z' && ev.shiftKey)) { ev.preventDefault(); redo() }
  }
  window.addEventListener('keydown', keys)

  // ---- drawing
  const pt = ev => { const r = over.getBoundingClientRect(); return [(ev.clientX - r.left) * W / r.width, (ev.clientY - r.top) * H / r.height] }
  let drag = null
  const colour = right => isMask ? `rgb(${MASK_RGB})` : (right ? st.bg : st.fg)
  function strokeStyle(g, right, erase) {
    g.globalCompositeOperation = (isMask && (erase || right)) || (!isMask && erase && st.noise) ? 'destination-out' : 'source-over'
    const c = !isMask && erase ? st.bg : colour(right)
    g.strokeStyle = c; g.fillStyle = c
    g.lineCap = 'round'; g.lineJoin = 'round'
    const lw = st.tool === 'pencil' ? Math.max(1, Math.min(2, st.size / 4)) : st.size
    if (!isMask && st.tool === 'soft') {
      const f = st.feather ?? 0.7
      g.lineWidth = Math.max(1, lw * (1 - 0.55 * f))
      g.shadowColor = c
      g.shadowBlur = Math.max(2, lw * 0.85 * f)
    } else if (!isMask && (st.feather ?? 0) > 0 && st.tool !== 'pencil' && st.tool !== 'eraser') {
      g.lineWidth = lw
      g.shadowColor = c
      g.shadowBlur = lw * 0.5 * st.feather
    } else {
      g.lineWidth = lw
      g.shadowColor = 'transparent'
      g.shadowBlur = 0
    }
    g.imageSmoothingEnabled = st.tool !== 'pencil'
  }
  function seg(g, a, b) { g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke() }
  function shape(g, a, b, kind) {
    g.beginPath()
    const [x, y, w, hh] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])]
    if (kind === 'line') { g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); return }
    if (kind.startsWith('rect')) g.rect(x, y, w, hh)
    else g.ellipse(x + w / 2, y + hh / 2, w / 2, hh / 2, 0, 0, Math.PI * 2)
    if (kind.endsWith('f')) g.fill(); else g.stroke()
  }
  const isStrokeTool = t => ['soft', 'pencil', 'brush', 'eraser'].includes(t)
  const usePreviewLayer = () => !isMask && (st.tool === 'soft' || st.opacity < 0.99 || (st.feather > 0 && st.tool !== 'pencil' && st.tool !== 'eraser'))
  over.oncontextmenu = ev => ev.preventDefault()
  over.onpointerdown = ev => {
    over.setPointerCapture(ev.pointerId)
    const p = pt(ev), right = ev.button === 2
    if (st.tool === 'picker') { const d = bg.getImageData(Math.floor(p[0]), Math.floor(p[1]), 1, 1).data; const c = '#' + [0, 1, 2].map(i => d[i].toString(16).padStart(2, '0')).join(''); if (right) st.bg = c; else st.fg = c; st.tool = 'soft'; sync(); return }
    snapshot()
    if (st.tool === 'fill') { flood(Math.floor(p[0]), Math.floor(p[1]), right); return }
    const preview = usePreviewLayer() && st.tool !== 'eraser'
    over.style.opacity = preview ? String(st.opacity) : '1'
    drag = { a: p, last: p, right, preview }
    if (isStrokeTool(st.tool)) {
      const g = preview ? og : tg()
      strokeStyle(g, right, st.tool === 'eraser')
      seg(g, p, [p[0] + 0.01, p[1]])
    }
  }
  over.onpointermove = ev => {
    const p = pt(ev)
    status.textContent = `${Math.floor(p[0])}, ${Math.floor(p[1])}`
    if (!drag) return
    if (isStrokeTool(st.tool)) {
      const g = drag.preview ? og : tg()
      strokeStyle(g, drag.right, st.tool === 'eraser')
      seg(g, drag.last, p)
      drag.last = p
      return
    }
    og.clearRect(0, 0, W, H); strokeStyle(og, drag.right, false); og.globalCompositeOperation = 'source-over'
    if (isMask) { og.strokeStyle = og.fillStyle = drag.right ? 'rgba(255,255,255,.8)' : `rgb(${MASK_RGB})` }
    shape(og, drag.a, p, st.tool)
  }
  over.onpointerup = ev => {
    if (!drag) return
    const p = pt(ev)
    const g = tg()
    if (drag.preview && isStrokeTool(st.tool)) {
      g.save()
      g.globalAlpha = st.opacity
      g.globalCompositeOperation = 'source-over'
      g.shadowBlur = 0; g.shadowColor = 'transparent'
      g.drawImage(over, 0, 0)
      g.restore()
      og.clearRect(0, 0, W, H)
    } else if (['line', 'rect', 'rectf', 'ellipse', 'ellipsef'].includes(st.tool)) {
      og.clearRect(0, 0, W, H)
      g.save()
      if (!isMask) g.globalAlpha = st.opacity
      strokeStyle(g, drag.right, false)
      shape(g, drag.a, p, st.tool)
      g.restore()
    }
    over.style.opacity = '1'
    g.globalCompositeOperation = 'source-over'; g.shadowBlur = 0; g.shadowColor = 'transparent'; g.globalAlpha = 1
    og.shadowBlur = 0; og.shadowColor = 'transparent'
    drag = null
  }

  // Flood fill. Image mode: fills the connected area of similar colour with the colour. Mask mode: selects the
  // connected area of similar colour in the picture underneath (like a magic wand) and paints it into the mask.
  function flood(x0, y0, right) {
    if (x0 < 0 || y0 < 0 || x0 >= W || y0 >= H) return
    const src = bg.getImageData(0, 0, W, H).data
    const i0 = (y0 * W + x0) * 4, ref = [src[i0], src[i0 + 1], src[i0 + 2]]
    const seen = new Uint8Array(W * H), stack = [y0 * W + x0], t = st.tol
    const a0 = src[i0 + 3]
    const near = j => Math.abs(src[j] - ref[0]) <= t && Math.abs(src[j + 1] - ref[1]) <= t && Math.abs(src[j + 2] - ref[2]) <= t && Math.abs(src[j + 3] - a0) <= t
    while (stack.length) {
      const k = stack.pop()
      if (seen[k]) continue
      if (!near(k * 4)) continue
      seen[k] = 1
      const x = k % W, y = (k - x) / W
      if (x > 0) stack.push(k - 1); if (x < W - 1) stack.push(k + 1); if (y > 0) stack.push(k - W); if (y < H - 1) stack.push(k + W)
    }
    const g = tg(), d = g.getImageData(0, 0, W, H)
    const c = isMask ? MASK_RGB : hex(right ? st.bg : st.fg)
    for (let k = 0; k < seen.length; k++) {
      if (!seen[k]) continue
      const j = k * 4
      if (isMask && right) { d.data[j + 3] = 0; continue }
      d.data[j] = c[0]; d.data[j + 1] = c[1]; d.data[j + 2] = c[2]; d.data[j + 3] = 255
    }
    g.putImageData(d, 0, 0)
  }

  function result() {
    if (!isMask) return base.toDataURL('image/png')
    const out = h('canvas', { width: W, height: H }), o = out.getContext('2d')
    const d = mg.getImageData(0, 0, W, H)
    let any = false
    for (let i = 0; i < d.data.length; i += 4) { const on = d.data[i + 3] > 0; any ||= on; d.data[i] = d.data[i + 1] = d.data[i + 2] = 255; d.data[i + 3] = on ? 255 : 0 }
    o.putImageData(d, 0, 0)
    return any ? out.toDataURL('image/png') : ''
  }
}

// A small preview: the picture with the mask tinted on top (for the inspector).
export function maskPreview(background, maskUrl, w = 220) {
  const wrap = document.createElement('div')
  wrap.className = 'pt-preview'
  if (background) { const i = new Image(); i.src = background; wrap.append(i); wrap.style.backgroundImage = `url(${noiseTexture()})` }
  else wrap.append(Object.assign(document.createElement('div'), { className: 'pt-noimg', textContent: 'no picture' }))
  if (maskUrl) { const m = new Image(); m.src = maskUrl; m.className = 'pt-m'; wrap.append(m) }
  wrap.style.width = `${w}px`
  return wrap
}
