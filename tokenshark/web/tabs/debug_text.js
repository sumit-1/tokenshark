import { S, el, flow, heat, registerTab } from '../core.js'
import { esc } from '../flow.js'
import { debugStep } from '../ui/debugger.js'

export function renderDebug(p) {
  const st = S.debug
  if (!st) {
    p.innerHTML = `<div class="empty">Step through the model one block or one token at a time.<br>
      Press <b>🐞 Step</b> (or right-click a block → <i>Debug: run to here</i>). Then <b>Block ▸</b> (F10) runs the next box and
      <b>Token ▸▸</b> (F11) runs to the next prediction. The highlighted box is the one about to run.</div>`
    return
  }
  const where = st.done ? 'finished' : st.phase === 'idle' ? 'prediction ready — choose the next token, or press Block ▸ / Token ▸▸ to sample it'
    : `next: ${flow.node(st.active_node)?.data.label || st.active_node} (schedule #${st.next_pos})`
  p.innerHTML = `<div class="small">Every other tab (Output, Logit lens, Word cloud, Attention, KV caches) and the badges on the boxes update live after each step.</div><div class="small">Processing <b>${esc(st.chunk_origin || '')}</b> chunk “${esc(st.chunk.join(''))}” · ${st.n_generated} generated · ${where}</div>`
  const toks = el('div', { className: 'tokens' })
  st.tokens.forEach((t, i) => toks.appendChild(el('span', { className: `tok seg-${i < st.n_in ? 'prompt' : 'generated'}`, textContent: t })))
  st.chunk.forEach(t => toks.appendChild(el('span', { className: 'tok on', textContent: t, title: 'in flight' })))
  p.appendChild(toks)
  if (st.next_choices) {
    const box = el('div', { className: 'alts' }, '<h4>The model predicts — click to force a token:</h4>')
    st.next_choices.forEach(([piece, id, pr]) => {
      const b = el('button', { className: 'alt' }, `<span>${esc(piece)}</span><i style="width:${pr * 100}%"></i><b>${(pr * 100).toFixed(1)}%</b>`)
      b.onclick = () => debugStep('block', { force_token: id })
      box.appendChild(b)
    })
    p.appendChild(box)
  }
  const t = el('table', { className: 'grid trace' })
  t.innerHTML = `<tr><th>Box</th><th>Prediction at the last token if the model stopped here (logit lens)</th><th>Last token attends to</th><th>‖resid‖</th><th>‖attn‖</th><th>‖mlp‖</th></tr>` +
    st.trace.map(r => `<tr data-node="${esc(r.node)}"><td>${esc(r.pos < 0 ? 'Embed' : `#${r.pos} · ${flow.node(r.node)?.data.label || 'L' + r.layer}`)}</td>
      <td>${r.top.map(([tk, , pr]) => `<span class="chip" style="background:${heat(pr)}">${esc(tk)} ${(pr * 100).toFixed(0)}%</span>`).join('')}</td>
      <td>${(r.attends_to || []).map(([j, tk, w]) => `<span class="chip" title="key #${j}">${esc(tk)} ${(w * 100).toFixed(0)}%</span>`).join('')}</td>
      <td>${r.resid_norm ?? ''}</td><td>${r.attn_norm ?? ''}</td><td>${r.mlp_norm ?? ''}</td></tr>`).join('')
  t.querySelectorAll('tr[data-node]').forEach(tr => tr.onclick = () => flow.node(tr.dataset.node) && flow.select({ type: 'node', id: tr.dataset.node }))
  p.appendChild(t)
}

registerTab({ id: 'debug_text', alias: 'debug', title: 'Step debugger', render: renderDebug })
