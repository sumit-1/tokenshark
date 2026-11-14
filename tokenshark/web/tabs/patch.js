import { POST, S, btn, el, flow, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBox, aliveIn, followTask, labelled, recColor, refreshRuns0, runLabel, runsWithCaps, selectOf } from '../ui/labkit.js'

S.patch = { source: '', hook: 'resid_post', tokens: 'all', result: null }

export async function renderPatch(p) {
  await refreshRuns0()
  const st = S.patch, runs = runsWithCaps()
  p.appendChild(el('div', { className: 'small' }, `<b>Activation patching.</b> Take two prompts that differ in one word — a <i>clean</i> one
    (“The capital of France is”) and a <i>corrupted</i> one (“The capital of Japan is”). Run the clean one, put the corrupted one in the Prompt box,
    then sweep: the corrupted prompt is re-run once per block and token, each time copying that single activation from the clean run.
    Cells that bring the clean answer back show <b>where the model keeps what differs</b>. Use prompts with the same number of tokens.`))
  if (!runs.length) { p.appendChild(el('div', { className: 'empty', textContent: 'No finished run in memory yet. Run the clean prompt first.' })); return }
  if (!runs.find(r => r.id === st.source)) st.source = (runs.find(r => r.id !== S.run?.id) || runs[0]).id
  const src = selectOf(runs.map(r => [r.id, runLabel(r)]), st.source); src.onchange = () => { st.source = src.value }
  const hook = selectOf([
    ['resid_post', 'resid_post — the stream after the block'],
    ['resid_pre', 'resid_pre — the stream before the block'],
    ['attn_out', 'attn_out — what attention wrote'],
    ['mlp_out', 'mlp_out — what the MLP wrote'],
    ['q', 'q — query heads'],
    ['k', 'k — new key heads'],
    ['v', 'v — new value heads'],
    ['kv_k', 'kv_k — cached keys in KV cache'],
    ['kv_v', 'kv_v — cached values in KV cache'],
  ], st.hook); hook.onchange = () => { st.hook = hook.value }
  const toks = selectOf([['all', 'every token'], ['last', 'last token only (fast)']], st.tokens); toks.onchange = () => { st.tokens = toks.value }
  const out = el('div')
  const go = btn('Sweep the current prompt', async () => {
    const alive = aliveIn('patch')
    out.innerHTML = spinner('running the corrupted prompt once per block × token…')
    try {
      const start = await POST('/api/patch/sweep', { flow: flow.getGraph(), source_run: st.source, hook: st.hook, tokens: st.tokens })
      await followTask(start, r => { st.result = { ...r, source: st.source }; drawSweep(out, st.result, true) }, alive)
      if (alive()) drawSweep(out, st.result, false)
    } catch (e) { out.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
  }, 'primary')
  const row = el('div', { className: 'row' })
  row.append(labelled('Clean run', src), labelled('Copy', hook), labelled('Tokens', toks), go)
  p.append(row, out)
  if (st.result) drawSweep(out, st.result, false)
}

export function drawSweep(out, r, partial) {
  out.innerHTML = ''
  out.appendChild(el('div', { className: 'small' }, `${partial ? spinner(`${Math.round((r._progress || 0) * 100)}%`) + ' ' : ''}Clean run answers <span class="chip">${esc(r.answer)}</span>
    (p = ${r.p_clean}); this prompt answers <span class="chip">${esc(r.corrupt_answer)}</span> and gives the clean answer p = ${r.p_corrupt}.
    Each cell: how much of that gap one copied activation closes (1 = fully, 0 = nothing, red = makes it worse). Click a cell to add that patch to the flow.` +
    (r.same_length ? '' : ' <span class="bad">The two prompts have different token counts, so positions may not line up.</span>')))
  const t = el('table', { className: 'grid heat' })
  const differ = r.tokens.some((x, i) => x !== r.source_tokens[i])
  t.innerHTML = `<tr><th></th>${r.tokens.map((x, i) => `<th title="token #${r.token_index[i]}">${esc(x)}${differ && r.source_tokens[i] !== x ? `<br><span class="small">← ${esc(r.source_tokens[i])}</span>` : ''}</th>`).join('')}</tr>`
  r.grid.forEach((rowv, pIdx) => {
    const tr = el('tr'); tr.appendChild(el('th', { textContent: r.blocks[pIdx] }))
    rowv.forEach((v, j) => {
      const td = el('td', { textContent: v == null ? '' : v.toFixed(2), title: v == null ? 'not computed yet' : `p(clean answer) = ${r.prob[pIdx][j]}` })
      td.style.background = recColor(v)
      if (v != null) td.onclick = () => {
        const n = addBox(r.nodes[pIdx], { label: `patch ← ${r.source}`, hook: r.hook, op: 'patch', ref: r.source, positions: String(r.token_index[j]) })
        if (n) toast(`Added: copy ${r.hook} of ${r.blocks[pIdx]}, token #${r.token_index[j]}, from run ${r.source}. Press ▶ Run.`)
      }
      tr.appendChild(td)
    })
    t.appendChild(tr)
  })
  out.appendChild(t)
}

registerTab({ id: 'patch', title: 'Patching', hint: 'Activation patching: copy activations from one run into another', render: renderPatch })
