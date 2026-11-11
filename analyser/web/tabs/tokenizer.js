import { GET, POST, S, el, registerTab, spinner } from '../core.js'
import { esc } from '../flow.js'

S.tk = { text: 'The capital of France is Paris. Tokenization isn’t word-splitting!', chat: false, q: '', picked: null }

export function renderTokenizer(p) {
  const st = S.tk
  p.appendChild(el('div', { className: 'small' }, `<b>Tokenizer.</b> The model never sees letters or words, only token ids from a fixed vocabulary of ${S.info.vocab ?? '?'} pieces.
    Type anything: common words are one token, rare ones split. Length limits and prices count tokens. Click a token for its nearest neighbours in embedding space.`))
  const ta = el('textarea', { rows: 3, value: st.text })
  const chat = el('input', { type: 'checkbox', checked: st.chat })
  const cl = el('label', { className: 'field check' }); cl.append(chat, el('span', { textContent: 'wrap in the chat template (see the hidden tokens every chat message costs)' }))
  const stats = el('div', { className: 'small' }), toks = el('div', { className: 'tokens big' }), nb = el('div')
  let timer = null
  const run = async () => {
    st.text = ta.value; st.chat = chat.checked
    try {
      const r = await POST('/api/tokenize', { text: ta.value, chat: chat.checked })
      stats.innerHTML = `<b>${r.n_tokens}</b> tokens · ${r.n_chars} characters · ${r.n_bytes} bytes · <b>${r.chars_per_token}</b> characters per token`
      toks.innerHTML = ''
      r.pieces.forEach((t, i) => {
        const s = el('span', { className: `tok c${i % 5}${st.picked === r.ids[i] ? ' on' : ''}`, title: `id ${r.ids[i]}` }, `${esc(t === '' ? '∅' : t)}<sub>${r.ids[i]}</sub>`)
        s.onclick = () => { st.picked = r.ids[i]; neighbors(nb, r.ids[i]); toks.querySelectorAll('.tok').forEach(x => x.classList.remove('on')); s.classList.add('on') }
        toks.appendChild(s)
      })
    } catch (e) { stats.innerHTML = `<span class="bad">${esc(e.message)}</span>` }
  }
  ta.oninput = () => { clearTimeout(timer); timer = setTimeout(run, 200) }
  chat.onchange = run
  p.append(ta, cl, stats, toks, nb)
  // vocabulary search
  const vb = el('div', { className: 'kvbox' }, '<h4>Search the vocabulary</h4>')
  const q = el('input', { placeholder: 'letters inside a token, e.g. ing', value: st.q, style: 'width:240px' })
  const res = el('div')
  let vt = null
  const search = async () => {
    st.q = q.value
    if (!q.value) { res.innerHTML = ''; return }
    try {
      const r = await GET(`/api/tokens/search?q=${encodeURIComponent(q.value)}`)
      res.innerHTML = `<div class="small">${r.n} of ${r.vocab} tokens contain “${esc(r.query)}”${r.n > r.tokens.length ? ` (first ${r.tokens.length})` : ''}</div>`
      const w = el('div', { className: 'tokens' })
      r.tokens.forEach(([t, id]) => { const s = el('span', { className: 'tok', title: `id ${id}` }, `${esc(t)}<sub>${id}</sub>`); s.onclick = () => { st.picked = id; neighbors(nb, id) }; w.appendChild(s) })
      res.appendChild(w)
    } catch (e) { res.innerHTML = `<span class="bad">${esc(e.message)}</span>` }
  }
  q.oninput = () => { clearTimeout(vt); vt = setTimeout(search, 250) }
  vb.append(q, res)
  p.appendChild(vb)
  run(); if (st.q) search(); if (st.picked != null) neighbors(nb, st.picked)
}

export async function neighbors(box, id) {
  box.innerHTML = spinner()
  try {
    const r = await GET(`/api/tokens/${id}/neighbors?k=14`)
    box.innerHTML = `<div class="kvbox"><h4>Token ${r.id} “${esc(r.piece)}” — nearest tokens by embedding direction</h4>
      <div class="small">Before any block runs, these are the tokens the model treats most alike (cosine similarity).</div>
      <div>${r.neighbors.map(([t, i, s]) => `<span class="chip" title="id ${i}">${esc(t)} <span class="small">${s}</span></span>`).join(' ')}</div></div>`
  } catch (e) { box.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
}

registerTab({ id: 'tokenizer', title: 'Tokenizer', hint: 'How text becomes tokens', render: renderTokenizer })
