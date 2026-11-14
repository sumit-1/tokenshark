import { POST, S, api, btn, el, flow, on, registerTab, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBox, labelled, numInput, refreshRuns0, refreshVectors, runLabel, runsWithCaps, selectOf } from '../ui/labkit.js'
import { renderTab, tabAvailable } from '../ui/tabs.js'

S.steer = { a: '', b: '', pos: null, hook: 'resid_post', token: 'last', name: '', word: '', minus: '' }

export async function renderSteer(p) {
  await Promise.all([refreshRuns0(), refreshVectors()])
  const st = S.steer, runs = runsWithCaps(), nL = S.info.n_layers
  if (st.pos == null) st.pos = Math.floor(nL / 2)
  p.appendChild(el('div', { className: 'small' }, `<b>Steering vectors.</b> A direction in the residual stream. Add it at one block and every token is pushed the same way.
    Make one from two runs that differ in the thing you want to control (a happy and a sad prompt: vector = A − B), or from a word’s output direction.
    Then wire it into a block and raise the value until the text bends — too much and it breaks.`))
  // from two runs
  const b1 = el('div', { className: 'kvbox' }, '<h4>From two runs: A − B</h4>')
  if (runs.length < 2) b1.appendChild(el('div', { className: 'small', textContent: 'Needs two finished runs in memory (run two contrasting prompts).' }))
  else {
    if (!runs.find(r => r.id === st.a)) st.a = runs[runs.length - 1].id
    if (!runs.find(r => r.id === st.b) || st.b === st.a) st.b = (runs.slice().reverse().find(r => r.id !== st.a)).id
    const a = selectOf(runs.map(r => [r.id, runLabel(r)]), st.a), b = selectOf(runs.map(r => [r.id, runLabel(r)]), st.b)
    a.onchange = () => { st.a = a.value }; b.onchange = () => { st.b = b.value }
    const pos = numInput(st.pos, 60, { min: 0, max: nL - 1 }); pos.oninput = () => { st.pos = +pos.value }
    const hook = selectOf(['resid_post', 'resid_pre', 'attn_out', 'mlp_out'], st.hook); hook.onchange = () => { st.hook = hook.value }
    const tok = el('input', { value: st.token, style: 'width:70px', title: 'last = last prompt token · mean = average over tokens · or a token index' }); tok.oninput = () => { st.token = tok.value }
    const nm = el('input', { placeholder: 'name (optional)', value: st.name, style: 'width:150px' }); nm.oninput = () => { st.name = nm.value }
    const row = el('div', { className: 'row' })
    row.append(labelled('A (towards)', a), labelled('B (away from)', b), labelled('at block', pos), labelled('', hook), labelled('token', tok), nm, btn('Make vector', async () => {
      try { const v = await POST('/api/vectors/from_runs', { run_a: st.a, run_b: st.b, pos: st.pos, hook: st.hook, token: st.token, name: st.name }); toast(`Saved “${v.name}” (length ${v.norm}).`); renderTab() }
      catch (e) { toast(e.message, 'error') }
    }, 'primary'))
    b1.appendChild(row)
  }
  // from a word
  const b2 = el('div', { className: 'kvbox' }, '<h4>From a word’s output direction</h4>')
  const w = el('input', { placeholder: 'word, e.g. Paris', value: st.word, style: 'width:150px' }); w.oninput = () => { st.word = w.value }
  const m = el('input', { placeholder: 'minus word (optional)', value: st.minus, style: 'width:170px' }); m.oninput = () => { st.minus = m.value }
  const row2 = el('div', { className: 'row' })
  row2.append(w, m, btn('Make vector', async () => {
    if (!w.value.trim()) return toast('Type a word.', 'error')
    try { const v = await POST('/api/vectors/from_word', { word: w.value, minus: m.value }); toast(`Saved ${v.name}. It has length 1: use a value near the block’s ‖resid‖.`); renderTab() }
    catch (e) { toast(e.message, 'error') }
  }, 'primary'), el('span', { className: 'small', textContent: 'Raises that word’s score directly; length 1, so the value is in units of the residual stream.' }))
  b2.appendChild(row2)
  p.append(b1, b2)
  // library
  const t = el('table', { className: 'grid' })
  t.innerHTML = '<tr><th>Vector</th><th>From</th><th>Length</th><th>Model</th><th></th></tr>'
  const mine = (S.vectors || []).filter(v => !v.error)
  for (const v of mine) {
    const tr = el('tr')
    const ok = v.dim === S.info.d_model
    tr.innerHTML = `<td>${esc(v.name)}</td><td>${v.source === 'runs' ? `runs ${esc(v.run_a || '')} − ${esc(v.run_b || '')} · ${esc(v.hook)} @ #${v.pos} · token ${esc(v.token)}` : v.source === 'word' ? 'word' : esc(v.source || '')}</td>
      <td>${v.norm}</td><td>${esc(v.model_id || '?')}${ok ? '' : ' <span class="bad">≠ loaded</span>'}</td>`
    const td = el('td')
    const pickTarget = () => {
      const selN = S.selected?.type === 'node' ? flow.node(S.selected.id) : null
      const pos = v.pos ?? Math.floor(nL / 2)
      return selN?.data.kind === 'block' ? selN.id : (flow.nodes.find(n => n.data.kind === 'block' && n.data.layer === pos)?.id)
    }
    td.append(btn('Steer a block', () => {
      const target = pickTarget()
      const resid = S.run?.outputs?.block_stats?.[target]?.resid_norm
      const value = v.source === 'word' ? +(resid ? resid * 0.3 : 8).toFixed(1) : 2      // a nudge; raise it to see the text break
      const n = addBox(target, { label: `steer: ${v.name}`.slice(0, 40), hook: v.hook && v.hook !== 'resid_pre' ? v.hook : 'resid_post', op: 'steer', ref: v.id, value })
      if (n) toast(`Wired into ${flow.node(target).data.label} with value ${value}. Change the value in the box, then ▶ Run.`)
    }, 'mini'), btn('Erase direction (project_out)', () => {
      const target = pickTarget()
      const n = addBox(target, { label: `erase: ${v.name}`.slice(0, 40), hook: v.hook && v.hook !== 'resid_pre' ? v.hook : 'resid_post', op: 'project_out', ref: v.id, value: 1.0 })
      if (n) toast(`Wired orthogonal projection (project_out) into ${flow.node(target).data.label}. Press ▶ Run.`)
    }, 'mini'), btn('Delete', async () => { if (confirm(`Delete ${v.name}?`)) { await api('DELETE', `/api/vectors/${v.id}`); renderTab() } }, 'mini'))
    tr.appendChild(td); t.appendChild(tr)
  }
  if (mine.length) p.appendChild(t)
  else p.appendChild(el('div', { className: 'empty', textContent: 'No steering vectors yet.' }))
}

registerTab({ id: 'steer', title: 'Steering', hint: 'Steering vectors: push the residual stream along a direction', render: renderSteer })

on('model', () => { if (tabAvailable('steer')) refreshVectors() })
