import { GET, S, el, flow, newId, sleep, toast } from '../core.js'
import { esc } from '../flow.js'

// ------------------------------------------------------------------ shared bits


export const needRun = p => { if (S.run?.outputs?.tokens) return false; p.innerHTML = '<div class="empty">Run the flow first: this tab looks inside a finished run.</div>'; return true }

export const numInput = (v, w = 70, extra = {}) => el('input', { type: 'number', value: v, step: 'any', style: `width:${w}px`, ...extra })

export const labelled = (text, input) => { const l = el('label', { className: 'inl' }); l.append(el('span', { textContent: text }), input); return l }

export const selectOf = (options, value) => { const s = el('select'); options.forEach(o => { const [v, t] = Array.isArray(o) ? o : [o, o]; s.appendChild(el('option', { value: v, textContent: t })) }); if (value != null) s.value = value; return s }

export const runsWithCaps = () => S.runs.filter(r => r.has_captures && r.model_id === S.info?.model_id && r.status !== 'error')

export const runLabel = r => `${r.id}${r.label ? ' · ' + r.label : ''} — ${(r.preview || '').slice(0, 40)}`

export const blockNode = pos => S.run?.plan?.schedule_nodes?.[pos]

export const blockSelect = (value, onChange) => {
  const s = selectOf((S.run?.plan?.schedule || []).map((l, p) => [p, `#${p} · layer ${l}`]), value)
  s.onchange = () => onChange(+s.value); return s
}

// Follow a background task (POST …) to its end. onPart gets each partial result.
export async function followTask(start, onPart, alive) {
  if (start.result) onPart(start.result, start.task || {})
  if (!start.task) return start.result
  while (true) {
    const t = await GET(`/api/tasks/${start.task.id}`)
    if (!alive()) return null
    if (t.result) onPart(t.result, t)
    if (t.status === 'done') return t.result
    if (t.status === 'error') throw new Error(t.error)
    if (t.status === 'cancelled') throw new Error('Stopped.')
    await sleep(400)
  }
}

export const aliveIn = tab => { const seq = S.renderSeq; return () => S.tab === tab && S.renderSeq === seq }

export function addBox(targetId, data) {
  const t = flow.node(targetId)
  if (!t) return toast('That block is not in the current flow (restore the run’s flow first).', 'error')
  const n = flow.addNode({ id: newId('iv'), type: 'box', position: { x: t.position.x + 10, y: t.position.y - 105 },
    data: { kind: 'intervention', positions: 'all', steps: 'all', heads: 'all', neurons: 'all', value: 1, enabled: true, ...data } })
  flow.addEdge({ source: n.id, sourceHandle: 'out', target: t.id, targetHandle: 'mod' })
  flow.select({ type: 'node', id: n.id })
  return n
}

export function lineChart(series, w = 520, h = 150) {
  const pts = series.flatMap(s => s.pts)
  if (!pts.length) return ''
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1])
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const X = x => 34 + (x - x0) / (x1 - x0 || 1) * (w - 42), Y = y => h - 16 - (y - y0) / (y1 - y0 || 1) * (h - 26)
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" style="max-width:${w}px">
    <line x1="34" y1="${h - 16}" x2="${w - 8}" y2="${h - 16}" class="ax"/><line x1="34" y1="10" x2="34" y2="${h - 16}" class="ax"/>
    <text x="30" y="14" text-anchor="end">${y1.toFixed(2)}</text><text x="30" y="${h - 16}" text-anchor="end">${y0.toFixed(2)}</text>
    <text x="34" y="${h - 3}">${x0}</text><text x="${w - 8}" y="${h - 3}" text-anchor="end">step ${x1}</text>
    ${series.map(s => `<polyline class="${s.cls}" points="${s.pts.map(p => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ')}"/>`).join('')}
  </svg><div class="small">${series.map(s => `<span class="key ${s.cls}"></span>${esc(s.name)}`).join(' &nbsp; ')}</div>`
}

export async function refreshVectors() { try { S.vectors = await GET('/api/vectors') } catch { S.vectors = [] } }

export const recColor = v => v == null ? 'transparent' : v >= 0 ? `rgba(62,207,142,${Math.min(1, v) * 0.9})` : `rgba(255,107,107,${Math.min(1, -v) * 0.9})`

export async function refreshRuns0() { try { S.runs = await GET('/api/runs') } catch { } }
