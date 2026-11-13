import { POST, S, btn, download, el, flow, registerTab, toast } from '../core.js'
import { esc } from '../flow.js'
import { renderInspector } from '../ui/inspector.js'
import { startRun } from '../ui/run.js'
import { renderTab } from '../ui/tabs.js'
import { saveRunCache } from './kv.js'

// ---------- text output + step-in alternatives
export function renderTextOut(p) {
  const o = S.run.outputs
  p.appendChild(el('div', { className: 'small' }, `${o.tokens.length} tokens · ${o.prefix_tokens ? `${o.prefix_tokens} imported KV tokens in front · ` : ''}read ${o.gb_read} GB of weights · click a token for the alternatives the model weighed`))
  renderTokens(p, false)
  if (o.edited_code?.length) p.appendChild(el('div', { className: 'small' }, `<span class="chip coded">✎ edited code</span> ran in: ${o.edited_code.map(esc).join(', ')}`))
  if (o.prints?.length) p.appendChild(el('pre', { className: 'codehelp', textContent: o.prints.join('\n') }))
  p.appendChild(el('div', { className: 'row' })).append(btn('Save KV cache', saveRunCache), btn('Export as Python script', () => exportScript()),
    btn('Download manifest', () => window.open(`/api/runs/${S.run.id}/manifest`)))
}

export function renderTokens(container, compact) {
  const o = S.run.outputs
  const segOf = i => o.segments.find(s => i >= s.start && i < s.end)?.origin || 'prompt'
  const wrap = el('div', { className: 'tokens' + (compact ? ' compact' : '') })
  o.tokens.forEach((t, i) => {
    const s = el('span', { className: `tok seg-${segOf(i)}${S.alt === i ? ' on' : ''}`, textContent: t || '∅', title: `#${i}` })
    s.onclick = () => { S.alt = i; renderTab(); if (compact) renderInspector() }
    wrap.appendChild(s)
  })
  container.appendChild(wrap)
  if (S.alt != null && S.alt > 0 && o.alternatives[S.alt - 1]) {
    const i = S.alt, alts = o.alternatives[i - 1]
    const box = el('div', { className: 'alts' }, `<h4>At token #${i}, the model's top choices were:</h4>`)
    alts.forEach(([piece, id, pr]) => {
      const b = el('button', { className: 'alt' + (id === o.token_ids[i] ? ' chosen' : '') }, `<span>${esc(piece)}</span><i style="width:${pr * 100}%"></i><b>${(pr * 100).toFixed(1)}%</b>`)
      b.title = 'Fork: force this token here and continue'
      b.onclick = () => forkWith(i, id)
      box.appendChild(b)
    })
    container.appendChild(box)
  }
}

export function forkWith(i, tokenId) {
  // Prefill = everything the assistant said before token i (+ any existing prefill) + the chosen token.
  const o = S.run.outputs
  const prompt = flow.nodes.find(n => n.data.kind === 'input_prompt')
  const genStart = o.segments.find(s => s.origin === 'generated').start
  const preStart = o.segments.find(s => s.origin === 'prefill').start
  if (i < preStart) return toast('Fork points must be inside the prefill or generated text.', 'error')
  const ids = o.token_ids.slice(preStart, i).concat([tokenId])
  POST('/api/tokens/decode', { ids }).then(r => {
    flow.updateNodeData(prompt.id, { prefill: r.text })
    S.alt = null
    toast(`Forked at token #${i}: prefill updated (${ids.length} tokens).`)
    startRun({ label: `fork @${i}` })
  }).catch(e => toast(e.message, 'error'))
  void genStart
}

export async function exportScript() {
  if (!S.run) return
  try {
    const r = await fetch(`/api/runs/${S.run.id}/script`)
    const text = await r.text()
    if (!r.ok) throw new Error(JSON.parse(text).detail || r.statusText)
    download(`run_${S.run.id}.py`, text)
    toast(`Saved run_${S.run.id}.py — run it with “python run_${S.run.id}.py” from the app’s folder.`)
  } catch (e) { toast(e.message, 'error') }
}

registerTab({ id: 'output_text', alias: 'output', title: 'Output', needs: 'run', render: renderTextOut })
