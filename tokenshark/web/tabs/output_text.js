import { POST, S, btn, download, el, flow, registerTab, toast } from '../core.js'
import { esc } from '../flow.js'
import { renderInspector } from '../ui/inspector.js'
import { debugStep } from '../ui/debugger.js'
import { samplerMax, startRun } from '../ui/run.js'
import { renderTab } from '../ui/tabs.js'
import { saveRunCache } from './kv.js'

// ---------- text output + step-in alternatives
export function renderTextOut(p) {
  const o = S.run.outputs
  p.appendChild(el('div', { className: 'small' }, `${o.tokens.length} tokens · ${o.prefix_tokens ? `${o.prefix_tokens} imported KV tokens in front · ` : ''}read ${o.gb_read} GB of weights · click a token for the alternatives the model weighed`))
  renderTokens(p, false)
  appendBox(p)
  conversationBox(p)
  if (o.edited_code?.length) p.appendChild(el('div', { className: 'small' }, `<span class="chip coded">✎ edited code</span> ran in: ${o.edited_code.map(esc).join(', ')}`))
  if (o.prints?.length) p.appendChild(el('pre', { className: 'codehelp', textContent: o.prints.join('\n') }))
  p.appendChild(el('div', { className: 'row' })).append(btn('Save KV cache', saveRunCache), btn('Export as Python script', () => exportScript()),
    btn('Download manifest', () => window.open(`/api/runs/${S.run.id}/manifest`)))
}

// A run that stopped at its token limit can go on: raise Max new tokens in the Sampler box, then append.
function appendBox(p) {
  const d = S.debug
  if (!d?.done || !d.can_extend || d.run_id !== S.run.id || S.run.status === 'stepping') return
  const want = samplerMax()
  const row = el('div', { className: 'row' })
  if (want > d.total) row.append(btn(`Append tokens (${d.total} → ${want}) ▶`, () => debugStep('all'), 'primary'),
    el('span', { className: 'small', textContent: 'Continues from the finished run; nothing is recomputed. ▶ Run does the same.' }))
  else row.append(el('span', { className: 'small', textContent: `Stopped at the limit of ${d.total} new tokens. Raise “Max new tokens” in the Sampler box to append more without restarting.` }))
  p.appendChild(row)
}

// Every turn of the finished run, the model's answer included.
export function conversationOf(run) {
  const pr = run.plan?.params || {}, turns = (pr.history || []).map(m => ({ ...m }))
  if (pr.prompt || !turns.length) turns.push({ role: 'user', content: pr.prompt || '' })
  turns.push({ role: 'assistant', content: (pr.prefill || '') + (run.outputs.text || ''), answer: true })
  return turns
}

// The conversation so far, and a box to send the next user turn: the exchange moves into the Prompt box's list
// of earlier turns (where it can be edited) and the new message becomes the prompt.
function conversationBox(p) {
  if (S.run.status && !['done', 'cancelled'].includes(S.run.status)) return
  const turns = conversationOf(S.run)
  const box = el('div', { className: 'kvbox' }, `<h4>Conversation (${turns.length} turns)${S.run.plan?.params?.use_chat_template ? '' : ' <span class="small">- chat template is off: earlier turns are fed as plain “User: … / Assistant: …” lines</span>'}</h4>`)
  turns.forEach(m => box.appendChild(el('div', { className: `turn ${m.role}${m.answer ? ' answer' : ''}` }, `<b>${m.role}</b> ${esc(m.content) || '<i class="small">(empty)</i>'}`)))
  const ta = el('textarea', { rows: 2, placeholder: 'Your next message…', value: S.followup || '' })
  ta.oninput = () => { S.followup = ta.value }
  const push = (text, run) => {
    const prompt = flow.nodes.find(n => n.data.kind === 'input_prompt')
    flow.updateNodeData(prompt.id, { history: turns.map(({ role, content }) => ({ role, content })), prompt: text, prefill: '' })
    S.followup = ''; S.alt = null
    renderInspector()
    if (run) startRun({ label: `turn ${turns.length + 1}` })
    else toast('The exchange is now in the Prompt box’s list of earlier turns. Edit any turn there, then ▶ Run.')
  }
  ta.onkeydown = ev => { if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey) && ta.value.trim()) push(ta.value, true) }
  const row = el('div', { className: 'row' })
  row.append(btn('Send ▶', () => { if (!ta.value.trim()) return toast('Type the next message first.', 'error'); push(ta.value, true) }, 'primary'),
    btn('Keep as earlier turns (don’t run yet)', () => push(ta.value, false)),
    el('span', { className: 'small', textContent: 'Ctrl+Enter sends. Interceptors and edits in the flow stay in place for the next turn.' }))
  box.append(ta, row)
  p.appendChild(box)
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
    toast(`Saved run_${S.run.id}.py · run it with “python run_${S.run.id}.py” from the app’s folder.`)
  } catch (e) { toast(e.message, 'error') }
}

registerTab({ id: 'output_text', alias: 'output', title: 'Output', needs: 'run', render: renderTextOut })
