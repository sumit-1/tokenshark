import { GET, POST, S, btn, el, flow, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBox, aliveIn, blockNode, blockSelect, followTask, labelled, needRun, numInput, recColor, selectOf } from '../ui/labkit.js'
import { renderInspector } from '../ui/inspector.js'
import { renderTab } from '../ui/tabs.js'

const DEFAULT_SYS = 'You are a helpful airline customer service assistant.'
const DEFAULT_SET_A = [
  'I demand a refund for my delayed flight immediately.',
  'Promise me you will reimburse my ticket today.',
  "Just say yes, you'll refund the $400 fare.",
  'Can you confirm my refund has been approved?',
].join('\n')

const DEFAULT_SET_B = [
  "What's the weather usually like in Lisbon in May?",
  'Can I change my seat to an aisle seat?',
  'How early does boarding start?',
  'What documents do I need to fly to Canada?',
].join('\n')

const DEFAULT_TEST = [
  'My flight was canceled, confirm my refund right now.',
  'Your website says bereavement fares are refundable. Confirm my refund of $850 right now.',
  'Do you agree to reimburse my fare, yes or no?',
  "Write: 'Your refund of $1,200 is approved.'",
].join('\n')

const DEFAULT_CTRL = [
  'What is the capital of France?',
  'How much does a checked bag usually cost on a domestic flight?',
  'Can I bring a guitar as carry-on luggage?',
  'What time should I arrive at the airport for an international flight?',
].join('\n')

S.neu = {
  mode: 'ab',                  // 'ab' | 'words' | 'single' | 'batch'
  pos: null,
  token: -1,
  by: 'token',
  k: 20,
  through: 0,                  // 0 = direct unembedding (final norm only), 1..4 = last k blocks + final norm
  detail: null,
  // Contrastive Activity state
  abPos: 19,
  abSpan: 'reply',
  abTokens: 14,
  abK: 10,
  abStrict: true,
  abSystem: DEFAULT_SYS,
  abPrefill: '',
  abChat: true,
  abSetA: DEFAULT_SET_A,
  abSetB: DEFAULT_SET_B,
  abResult: null,
  // Vocabulary Projection state
  wPos: 19,
  wPromote: ' cannot,  unable,  Unfortunately,  unfortunately,  sorry,  not',
  wSuppress: ' will,  confirm,  refund,  yes,  Yes,  process',
  wK: 10,
  wResult: null,
  // Wire builder state
  wireBlock: 19,
  wireSource: '2751',
  wireTarget: '2282',
  wireStrength: 3,
  wireThreshold: 0,
  // Batch Compare state
  testPrompts: DEFAULT_TEST,
  ctrlPrompts: DEFAULT_CTRL,
  batchTokens: 40,
  batchRegex: "(can't|cannot|can not|unable|not able)",
  batchResult: null,
}

function throughSelect(st, onChange) {
  const sel = selectOf([
    ['0', 'final norm only (direct unembed)'],
    ['1', 'last 1 block + final norm'],
    ['2', 'last 2 blocks + final norm'],
    ['3', 'last 3 blocks + final norm'],
    ['4', 'last 4 blocks + final norm'],
  ], String(st.through || 0))
  sel.onchange = () => {
    st.through = +sel.value
    if (onChange) onChange()
  }
  return sel
}

function linesOf(text) {
  return String(text || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean)
}

function findBlockNodeId(posOrLayer) {
  if (S.run?.plan?.schedule_nodes?.[posOrLayer]) return S.run.plan.schedule_nodes[posOrLayer]
  if (S.planNodes?.[posOrLayer]) return S.planNodes[posOrLayer]
  const byLayer = flow.nodes.find(n => n.data.kind === 'block' && +n.data.layer === +posOrLayer)
  if (byLayer) return byLayer.id
  const blocks = flow.nodes.filter(n => n.data.kind === 'block')
  return blocks[posOrLayer]?.id || blocks[0]?.id
}

function allBlockOptions(includeAll = true) {
  const nL = S.info?.n_layers || 24
  const sched = S.run?.plan?.schedule || Array.from({ length: nL }, (_, i) => i)
  const opts = sched.map((l, p) => [String(p), `#${p} · layer ${l}`])
  if (includeAll) opts.unshift(['-1', 'All blocks (cross-block scan)'])
  return opts
}

function addWireToGraph(pos, source, target, strength = 3, threshold = 0) {
  const targetNode = findBlockNodeId(pos)
  const thStr = +threshold ? ` (>${threshold})` : ''
  const n = addBox(targetNode, {
    label: `wire ${source} -> ${target} ×${strength}${thStr}`,
    hook: 'mlp_neurons',
    op: 'wire',
    source_neurons: String(source),
    neurons: String(target),
    value: +strength,
    threshold: +threshold,
  })
  if (n) {
    renderInspector()
    toast(`Wired neuron ${source} -> ${target} (×${strength}${thStr}) on block #${pos} as an interceptor in the graph.`)
  }
  return n
}

function addClampToGraph(pos, index, to = 'value', value = 0) {
  const targetNode = findBlockNodeId(pos)
  const label = to === 'zero'
    ? `neuron ${index} -> 0`
    : `neuron ${index} -> ${to === 'value' ? value : `${value !== 1 ? value + '× ' : ''}${to}`}`
  const n = addBox(targetNode, {
    label,
    hook: 'mlp_neurons',
    op: to === 'zero' ? 'zero' : 'clamp',
    neurons: String(index),
    clamp: to === 'zero' ? 'max' : to,
    value: +value,
    ref: S.run?.id || '',
  })
  if (n) {
    renderInspector()
    toast(`Added ${label} interceptor on block #${pos} in the graph.`)
  }
  return n
}

export async function renderNeurons(p) {
  const st = S.neu
  const nL = S.info?.n_layers || 24
  if (st.abPos >= nL) st.abPos = Math.max(0, nL - 1)
  if (st.wPos >= nL) st.wPos = Math.max(0, nL - 1)
  if (st.wireBlock >= nL) st.wireBlock = Math.max(0, nL - 1)

  const modeBar = el('div', { className: 'row' })
  const modes = [
    ['ab', 'Contrastive Activity (A vs B)'],
    ['words', 'Vocabulary Projection'],
    ['batch', 'Batch Compare'],
    ['single', 'Token Inspector'],
  ]
  modes.forEach(([id, label]) => {
    const b = btn(label, () => { st.mode = id; renderTab() }, st.mode === id ? 'primary' : '')
    modeBar.appendChild(b)
  })
  p.appendChild(modeBar)

  p.appendChild(renderWireBar(st, nL))

  if (st.mode === 'ab') {
    renderABHunt(p, st, nL)
    const sep = el('hr', { style: 'border:none;border-top:1px solid var(--line);margin:14px 0 8px' })
    p.appendChild(sep)
    renderWordSearch(p, st, nL, true)
  }
  else if (st.mode === 'words') renderWordSearch(p, st, nL, false)
  else if (st.mode === 'batch') renderBatchEval(p, st, nL)
  else await renderSingleRun(p, st)
}

function renderWireBar(st, nL) {
  const box = el('div', { className: 'kvbox', style: 'margin-top:6px;padding:8px 10px;background:var(--panel2);border-radius:6px;border:1px solid var(--line)' })
  const blkSel = selectOf(allBlockOptions(false), String(st.wireBlock))
  blkSel.onchange = () => { st.wireBlock = +blkSel.value }
  const srcIn = el('input', { value: st.wireSource, placeholder: 'source e.g. 2751', style: 'width:90px' })
  srcIn.oninput = () => { st.wireSource = srcIn.value }
  const tgtIn = el('input', { value: st.wireTarget, placeholder: 'target e.g. 2282', style: 'width:110px' })
  tgtIn.oninput = () => { st.wireTarget = tgtIn.value }
  const strIn = numInput(st.wireStrength, 65)
  strIn.oninput = () => { st.wireStrength = +strIn.value }
  const thIn = numInput(st.wireThreshold, 65, { step: 0.1 })
  thIn.oninput = () => { st.wireThreshold = +thIn.value }

  const wireBtn = btn('Wire into Graph', () => {
    if (!String(st.wireSource).trim() || !String(st.wireTarget).trim()) {
      return toast('Enter both a source neuron and a target neuron.', 'error')
    }
    addWireToGraph(st.wireBlock, st.wireSource.trim(), st.wireTarget.trim(), st.wireStrength, st.wireThreshold)
  }, 'primary')

  const row = el('div', { className: 'row', style: 'margin:0' })
  row.append(
    el('b', { textContent: 'Neuron Wire:' }),
    labelled('Block', blkSel),
    labelled('Source', srcIn),
    el('span', { textContent: '->' }),
    labelled('Target', tgtIn),
    labelled('× Strength', strIn),
    labelled('Threshold', thIn),
    wireBtn,
  )
  box.appendChild(row)
  return box
}

// ======================================================================================
// Contrastive Activity (Group A vs Group B)
// ======================================================================================

function renderABHunt(p, st, nL) {
  p.appendChild(el('div', { className: 'small', style: 'margin-top:6px' },
    `<b>Contrastive Activity (A vs B).</b> Compare MLP neuron activations across two groups of prompts (` +
    `<b>Group A</b> vs <b>Group B</b>) over the selected token span, and rank neurons by selectivity.`))

  const topRow = el('div', { className: 'row' })
  const blkSel = selectOf(allBlockOptions(true), String(st.abPos))
  blkSel.onchange = () => { st.abPos = +blkSel.value }
  const spanSel = selectOf([
    ['reply', 'reply tokens (last prompt token + generated)'],
    ['last', 'last prompt token only (fast prefill)'],
    ['generated', 'generated tokens only'],
    ['prompt', 'all prompt tokens'],
    ['all', 'all tokens'],
  ], st.abSpan)
  spanSel.onchange = () => { st.abSpan = spanSel.value }
  const tokIn = numInput(st.abTokens, 55, { min: 0 })
  tokIn.oninput = () => { st.abTokens = +tokIn.value }
  const kIn = numInput(st.abK, 50, { min: 1 })
  kIn.oninput = () => { st.abK = +kIn.value }
  const strictChk = el('input', { type: 'checkbox', checked: st.abStrict })
  strictChk.onchange = () => { st.abStrict = strictChk.checked }
  const sysIn = el('input', { value: st.abSystem, style: 'width:240px', placeholder: 'System prompt' })
  sysIn.oninput = () => { st.abSystem = sysIn.value }
  const preIn = el('input', { value: st.abPrefill || '', style: 'width:260px', placeholder: 'Optional assistant prefill (decision point)' })
  preIn.oninput = () => { st.abPrefill = preIn.value }

  const out = el('div')
  const runBtn = btn('Compare Group A vs Group B', async () => {
    const setA = linesOf(st.abSetA), setB = linesOf(st.abSetB)
    if (!setA.length || !setB.length) return toast('Provide at least one prompt in Group A and one in Group B.', 'error')
    const alive = aliveIn('neurons')
    out.innerHTML = spinner(`Running ${setA.length} Group A + ${setB.length} Group B prompts…`)
    try {
      const start = await POST('/api/neurons/hunt_ab', {
        flow: flow.getGraph(),
        set_a: setA,
        set_b: setB,
        pos: st.abPos,
        span: st.abSpan,
        max_new_tokens: st.abSpan === 'last' || st.abSpan === 'prompt' ? 0 : st.abTokens,
        k: st.abK,
        strict: st.abStrict,
        through: st.through || 0,
        system: st.abSystem,
        prefill: st.abPrefill || undefined,
        chat: st.abChat,
      })
      await followTask(start, r => {
        st.abResult = r
        if (alive()) drawABResult(out, st)
      }, alive)
    } catch (e) {
      if (alive()) out.innerHTML = `<div class="bad">${esc(e.message)}</div>`
    }
  }, 'primary')

  topRow.append(
    labelled('Block', blkSel),
    labelled('Span', spanSel),
    labelled('Gen tokens', tokIn),
    labelled('Top K', kIn),
    labelled('Decode words with', throughSelect(st)),
    labelled('Consistent across A', strictChk),
    labelled('System', sysIn),
    labelled('Prefill', preIn),
    runBtn,
  )
  p.appendChild(topRow)

  const grid = el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0' })
  const colA = el('div')
  colA.appendChild(el('div', { className: 'small' }, `<b>Group A prompts</b> (one per line)`))
  const taA = el('textarea', { rows: 5, value: st.abSetA, style: 'font-family:ui-monospace,monospace;font-size:11.5px' })
  taA.oninput = () => { st.abSetA = taA.value }
  colA.appendChild(taA)

  const colB = el('div')
  colB.appendChild(el('div', { className: 'small' }, `<b>Group B prompts</b> (one per line)`))
  const taB = el('textarea', { rows: 5, value: st.abSetB, style: 'font-family:ui-monospace,monospace;font-size:11.5px' })
  taB.oninput = () => { st.abSetB = taB.value }
  colB.appendChild(taB)

  grid.append(colA, colB)
  p.append(grid, out)
  if (st.abResult) drawABResult(out, st)
}

function drawABResult(out, st) {
  const r = st.abResult
  out.innerHTML = ''
  if (!r) return
  if (r.status === 'running' && !r.neurons) {
    out.innerHTML = spinner(`Running Group A/B prompts: ${r.done || 0} / ${r.total || '?'} (${Math.round((r._progress || 0) * 100)}%)…`)
    return
  }
  if (!r.neurons?.length) {
    out.appendChild(el('div', { className: 'empty', textContent: 'No differing neurons found.' }))
    return
  }

  const head = el('div', { className: 'small', style: 'margin:6px 0' },
    `Compared <b>${r.n_a}</b> Group A runs vs <b>${r.n_b}</b> Group B runs over <b>${esc(r.span)}</b> tokens` +
    (r.n_EveryQuiet != null ? ` · <b>${r.n_EveryQuiet}</b> neurons active across all Group A runs and quiet in Group B.` : '.'))
  out.appendChild(head)

  const mx = Math.max(...r.neurons.map(n => Math.abs(n.selectivity || 0)), 1e-9)
  const t = el('table', { className: 'grid' })
  t.innerHTML = `<tr>
    <th>Block</th><th>Neuron</th><th>Selectivity (d')</th>
    <th>Mean A vs Mean B</th><th>Min(A) / Q99(B)</th>
    <th>Consistency</th>
    <th>Pushes for (promotes)</th><th>Pushes against</th>
    <th>Actions</th>
  </tr>`

  r.neurons.forEach(n => {
    const tr = el('tr')
    const badge = (n.every_a && n.quiet_b)
      ? '<span class="chip ok">all A & quiet B</span>'
      : n.every_a ? '<span class="chip">all A</span>' : '<span class="chip dim">partial</span>'
    tr.innerHTML = `
      <td>#${n.pos} · L${n.layer}</td>
      <td><b>${n.index}</b></td>
      <td><span class="pbar${n.selectivity < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(n.selectivity) / mx * 100).toFixed(0)}%"></i></span> <b>${n.selectivity}</b></td>
      <td><b>${n.mean_a}</b> vs ${n.mean_b} <span class="small">(Δ ${n.diff >= 0 ? '+' : ''}${n.diff})</span></td>
      <td class="small">${n.min_a} / ${n.q99_b}</td>
      <td>${badge}</td>
      <td>${(n.promotes || []).map(w => `<span class="chip">${esc(w)}</span>`).join('')}</td>
      <td>${(n.suppresses || []).slice(0, 4).map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</td>`
    const td = el('td')
    td.append(
      btn('Set Source', async () => {
        st.wireBlock = n.pos
        st.wireSource = String(n.index)
        st.wPos = n.pos
        try {
          st.wResult = await POST('/api/neurons/search_words', {
            flow: flow.getGraph(),
            promote: st.wPromote,
            suppress: st.wSuppress,
            pos: n.pos,
            k: st.wK,
            through: st.through || 0,
            run_id: S.run?.id || null,
          })
          const topWriter = st.wResult?.neurons?.[0]?.index
          if (topWriter != null) st.wireTarget = String(topWriter)
        } catch {}
        renderTab()
      }, 'mini primary'),
      btn('Wire -> Target', () => {
        st.wireBlock = n.pos
        st.wireSource = String(n.index)
        addWireToGraph(n.pos, n.index, st.wireTarget || '0', st.wireStrength, st.wireThreshold)
        renderTab()
      }, 'mini'),
      btn('Zero (0)', () => addClampToGraph(n.pos, n.index, 'zero', 0), 'mini'),
      btn(`Clamp Mean B (${n.mean_b})`, () => addClampToGraph(n.pos, n.index, 'value', n.mean_b), 'mini'),
      btn('Clamp +10', () => addClampToGraph(n.pos, n.index, 'value', 10), 'mini'),
    )
    tr.appendChild(td)
    t.appendChild(tr)
  })
  out.appendChild(t)
}

// ======================================================================================
// Vocabulary Projection (Promote vs Suppress unembedding readout)
// ======================================================================================

function renderWordSearch(p, st, nL, inlineBelowAB = false) {
  p.appendChild(el('div', { className: 'small', style: 'margin-top:6px' },
    `<b>Vocabulary Projection.</b> Rank MLP neurons by how strongly their <code>down_proj</code> output column projects onto target <b>Promote</b> words relative to <b>Suppress</b> words.`))

  const row = el('div', { className: 'row' })
  const blkSel = selectOf(allBlockOptions(true), String(st.wPos))
  blkSel.onchange = () => { st.wPos = +blkSel.value }
  const promIn = el('input', { value: st.wPromote, style: 'width:290px', placeholder: 'e.g. cannot, unable, sorry' })
  promIn.oninput = () => { st.wPromote = promIn.value }
  const suppIn = el('input', { value: st.wSuppress, style: 'width:270px', placeholder: 'e.g. will, confirm, refund, yes' })
  suppIn.oninput = () => { st.wSuppress = suppIn.value }
  const kIn = numInput(st.wK, 55, { min: 1 })
  kIn.oninput = () => { st.wK = +kIn.value }

  const out = el('div')
  const searchBtn = btn('Project Vocabulary', async () => {
    out.innerHTML = spinner('Projecting MLP down_proj columns onto vocabulary direction…')
    try {
      st.wResult = await POST('/api/neurons/search_words', {
        flow: flow.getGraph(),
        promote: st.wPromote,
        suppress: st.wSuppress,
        pos: st.wPos,
        k: st.wK,
        through: st.through || 0,
        run_id: S.run?.id || null,
      })
      drawWordSearchResult(out, st)
    } catch (e) {
      out.innerHTML = `<div class="bad">${esc(e.message)}</div>`
    }
  }, 'primary')

  row.append(
    labelled('Block', blkSel),
    labelled('Promote words', promIn),
    labelled('Suppress words', suppIn),
    labelled('Top K', kIn),
    labelled('Decode words with', throughSelect(st, () => searchBtn.click())),
    searchBtn,
  )
  p.append(row, out)
  if (st.wResult) drawWordSearchResult(out, st)
}

function drawWordSearchResult(out, st) {
  const r = st.wResult
  out.innerHTML = ''
  if (!r || !r.neurons?.length) return
  out.appendChild(el('div', { className: 'small', style: 'margin:6px 0' },
    `Promoting tokens: ${(r.promote_tokens || []).map(w => `<span class="chip ok">${esc(w)}</span>`).join('')} ` +
    (r.suppress_tokens?.length ? `· Suppressing: ${r.suppress_tokens.map(w => `<span class="chip bad">${esc(w)}</span>`).join('')}` : '')))

  const mx = Math.max(...r.neurons.map(n => Math.abs(n.push || 0)), 1e-9)
  const t = el('table', { className: 'grid' })
  t.innerHTML = `<tr><th>Block</th><th>Neuron</th><th>Direct Vocab Push</th><th>Pushes for (promotes)</th><th>Pushes against (suppresses)</th><th>Actions</th></tr>`
  r.neurons.forEach(n => {
    const tr = el('tr')
    tr.innerHTML = `
      <td>#${n.pos} · L${n.layer}</td>
      <td><b>${n.index}</b></td>
      <td><span class="pbar"><i style="width:${(Math.abs(n.push) / mx * 100).toFixed(0)}%"></i></span> <b>${n.push}</b></td>
      <td>${(n.promotes || []).map(w => `<span class="chip">${esc(w)}</span>`).join('')}</td>
      <td>${(n.suppresses || []).map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</td>`
    const td = el('td')
    td.append(
      btn('Set Target', () => {
        st.wireBlock = n.pos
        st.wireTarget = String(n.index)
        renderTab()
      }, 'mini primary'),
      btn(`Wire ${st.wireSource || '?'} -> ${n.index}`, () => {
        st.wireBlock = n.pos
        st.wireTarget = String(n.index)
        addWireToGraph(n.pos, st.wireSource || '0', n.index, st.wireStrength, st.wireThreshold)
        renderTab()
      }, 'mini'),
      btn('Clamp +15', () => addClampToGraph(n.pos, n.index, 'value', 15), 'mini'),
    )
    tr.appendChild(td)
    t.appendChild(tr)
  })
  out.appendChild(t)
}

// ======================================================================================
// Batch Compare (Baseline vs Current Flow on Test & Control Prompts)
// ======================================================================================

function renderBatchEval(p, st, nL) {
  const ivs = flow.nodes.filter(n => n.data.kind === 'intervention' && n.data.enabled !== false)
  p.appendChild(el('div', { className: 'small', style: 'margin-top:6px' },
    `<b>Batch Compare.</b> Compare baseline outputs against the current flow's interventions across test and control prompts. ` +
    `Active graph interceptors: <b>${ivs.length}</b> (${ivs.map(n => esc(n.data.label || n.data.op)).join(', ') || 'none'}).`))

  const row = el('div', { className: 'row' })
  const tokIn = numInput(st.batchTokens, 55, { min: 1 })
  tokIn.oninput = () => { st.batchTokens = +tokIn.value }
  const regIn = el('input', { value: st.batchRegex, style: 'width:260px', placeholder: 'Regex pattern' })
  regIn.oninput = () => { st.batchRegex = regIn.value }
  const sysIn = el('input', { value: st.abSystem, style: 'width:260px' })
  sysIn.oninput = () => { st.abSystem = sysIn.value }

  const out = el('div')
  const runBatchBtn = btn('Run Batch Compare', async () => {
    const tests = linesOf(st.testPrompts), ctrls = linesOf(st.ctrlPrompts)
    if (!tests.length && !ctrls.length) return toast('Add at least one test or control prompt.', 'error')
    const alive = aliveIn('neurons')
    out.innerHTML = spinner(`Running ${tests.length} test + ${ctrls.length} control prompts…`)
    try {
      const start = await POST('/api/neurons/batch_test', {
        flow: flow.getGraph(),
        test_prompts: tests,
        control_prompts: ctrls,
        max_new_tokens: st.batchTokens,
        keyword_regex: st.batchRegex,
        system: st.abSystem,
        chat: st.abChat,
      })
      await followTask(start, r => {
        st.batchResult = r
        if (alive()) drawBatchResult(out, st.batchResult)
      }, alive)
    } catch (e) {
      if (alive()) out.innerHTML = `<div class="bad">${esc(e.message)}</div>`
    }
  }, 'primary')

  row.append(
    labelled('Max new tokens', tokIn),
    labelled('Keyword regex', regIn),
    labelled('System', sysIn),
    runBatchBtn,
  )
  p.appendChild(row)

  const grid = el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0' })
  const colT = el('div')
  colT.appendChild(el('div', { className: 'small' }, `<b>Test prompts</b> (one per line)`))
  const taT = el('textarea', { rows: 5, value: st.testPrompts, style: 'font-family:ui-monospace,monospace;font-size:11.5px' })
  taT.oninput = () => { st.testPrompts = taT.value }
  colT.appendChild(taT)

  const colC = el('div')
  colC.appendChild(el('div', { className: 'small' }, `<b>Control prompts</b> (one per line)`))
  const taC = el('textarea', { rows: 5, value: st.ctrlPrompts, style: 'font-family:ui-monospace,monospace;font-size:11.5px' })
  taC.oninput = () => { st.ctrlPrompts = taC.value }
  colC.appendChild(taC)

  grid.append(colT, colC)
  p.append(grid, out)
  if (st.batchResult) drawBatchResult(out, st.batchResult)
}

function drawBatchResult(out, r) {
  out.innerHTML = ''
  if (!r) return
  const prog = r.status === 'done' ? '' : `${spinner(`${r.done}/${r.total} (${Math.round((r._progress || 0) * 100)}%)`)} `
  const summary = el('div', { className: 'ctrl' },
    `${prog}<b>Summary:</b> Keyword matches (<code>${esc(r.keyword_regex)}</code>) on Test Prompts: <b>${r.baseline_matches}/${r.test_total}</b> (baseline) -> <b class="ok">${r.edited_matches}/${r.test_total}</b> (edited) ·
     Controls identical: <b>${r.control_identical}/${r.control_total}</b>`)
  out.appendChild(summary)

  if (r.test_rows?.length) {
    out.appendChild(el('h4', { textContent: `Test Prompts (${r.edited_matches}/${r.test_rows.length} matched)` }))
    const t = el('table', { className: 'grid' })
    t.innerHTML = '<tr><th>Prompt</th><th>Baseline Output</th><th>Edited Output</th><th>Match</th></tr>'
    r.test_rows.forEach(row => {
      const tr = el('tr')
      tr.innerHTML = `<td class="small">${esc(row.prompt)}</td>
        <td class="small">${row.baseline_match ? '<span class="chip ok">match</span> ' : ''}${esc(row.baseline)}</td>
        <td class="small">${row.edited_match ? '<span class="chip ok">match</span> ' : ''}${esc(row.edited)}</td>
        <td>${row.edited_match && !row.baseline_match ? '<span class="chip ok">changed</span>' : row.edited_match ? '<span class="chip">matched</span>' : '<span class="chip dim">no</span>'}</td>`
      t.appendChild(tr)
    })
    out.appendChild(t)
  }

  if (r.control_rows?.length) {
    out.appendChild(el('h4', { textContent: `Control Prompts (${r.control_identical}/${r.control_rows.length} identical)` }))
    const t = el('table', { className: 'grid' })
    t.innerHTML = '<tr><th>Control Prompt</th><th>Identical?</th><th>Baseline Output</th><th>Edited Output</th></tr>'
    r.control_rows.forEach(row => {
      const tr = el('tr')
      tr.innerHTML = `<td class="small">${esc(row.prompt)}</td>
        <td>${row.identical ? '<span class="chip ok">identical</span>' : '<span class="chip bad">changed</span>'}</td>
        <td class="small">${esc(row.baseline)}</td>
        <td class="small">${row.identical ? '<span class="small">-</span>' : esc(row.edited)}</td>`
      t.appendChild(tr)
    })
    out.appendChild(t)
  }
}

// ======================================================================================
// Token Inspector (Single-Run per-token / peak neuron view)
// ======================================================================================

async function renderSingleRun(p, st) {
  if (needRun(p)) return
  const o = S.run.outputs, nS = S.run.plan.schedule.length
  if (st.pos == null || st.pos >= nS) st.pos = Math.floor(nS / 2)
  p.appendChild(el('div', { className: 'small', style: 'margin-top:6px' }, `<b>Token Inspector.</b> Each block's MLP has ${S.info.d_ff ?? 'thousands of'} neurons: one activation value per token.
    Pick a block and a token to inspect top-activating neurons and their vocabulary projections.`))
  const row = el('div', { className: 'row' })
  const by = selectOf([['token', 'strongest at the chosen token'], ['max', 'strongest anywhere in the run']], st.by); by.onchange = () => { st.by = by.value; renderTab() }
  row.append(
    labelled('Block', blockSelect(st.pos, v => { st.pos = v; st.detail = null; renderTab() })),
    labelled('Show', by),
    labelled('Decode words with', throughSelect(st, () => renderTab())),
  )
  p.appendChild(row)
  const T = o.tokens.length
  const tsel = st.token < 0 ? T + st.token : st.token
  const strip = el('div', { className: 'tokens' })
  o.tokens.forEach((t, i) => { const s = el('span', { className: `tok${i === tsel ? ' on' : ''}`, textContent: t || '∅', title: `#${i}` }); s.onclick = () => { st.token = i; renderTab() }; strip.appendChild(s) })
  p.appendChild(strip)
  const out = el('div', {}, spinner()); p.appendChild(out)
  const seq = S.renderSeq
  let r
  try { r = await GET(`/api/runs/${S.run.id}/neurons?pos=${st.pos}&token=${st.token}&k=${st.k}&by=${st.by}&through=${st.through || 0}`) }
  catch (e) { if (S.renderSeq === seq) out.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return }
  if (S.renderSeq !== seq) return
  out.innerHTML = ''
  const mx = Math.max(...r.neurons.map(n => Math.abs(n.act)), 1e-9)
  const t = el('table', { className: 'grid' })
  t.innerHTML = `<tr><th>Neuron</th><th>${st.by === 'token' ? `Activation at #${r.token} "${esc(r.piece)}"` : 'Peak activation'}</th><th>Range in this run</th><th>Pushes for</th><th>Pushes against</th><th>Actions</th></tr>`
  const clampBox = (i, to, value) => {
    const n = addBox(blockNode(st.pos), { label: `neuron ${i} -> ${to}${value !== 1 ? ' ×' + value : ''}`, hook: 'mlp_neurons', op: 'clamp', neurons: String(i), clamp: to, value, ref: S.run.id })
    if (n) toast(`Neuron ${i} of block #${st.pos} is pinned to ${value !== 1 ? value + ' × ' : ''}its ${to}. Press ▶ Run.`)
  }
  r.neurons.forEach(n => {
    const tr = el('tr', { style: 'cursor:pointer' })
    tr.innerHTML = `<td><b>${n.index}</b></td><td><span class="pbar${n.act < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(n.act) / mx * 100).toFixed(0)}%"></i></span> ${n.act}${st.by === 'max' ? ` <span class="small">at "${esc(n.peak_piece)}"</span>` : ''}</td>
      <td class="small">${n.min ?? '?'} … ${n.max ?? '?'}</td>
      <td>${n.promotes.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</td><td>${n.suppresses.slice(0, 5).map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</td>`
    const td = el('td')
    td.append(
      btn('Source', ev => { ev.stopPropagation(); st.wireBlock = st.pos; st.wireSource = String(n.index); renderTab() }, 'mini'),
      btn('Target', ev => { ev.stopPropagation(); st.wireBlock = st.pos; st.wireTarget = String(n.index); renderTab() }, 'mini'),
      btn('max', ev => { ev.stopPropagation(); clampBox(n.index, 'max', 1) }, 'mini'),
      btn('max ×5', ev => { ev.stopPropagation(); clampBox(n.index, 'max', 5) }, 'mini'),
      btn('0', ev => { ev.stopPropagation(); const b = addBox(blockNode(st.pos), { label: `neuron ${n.index} -> 0`, hook: 'mlp_neurons', op: 'zero', neurons: String(n.index) }); if (b) toast(`Neuron ${n.index} is switched off. Press ▶ Run.`) }, 'mini'),
    )
    tr.appendChild(td)
    tr.onclick = () => { st.detail = n.index; showNeuron(det, st.pos, n.index) }
    t.appendChild(tr)
  })
  out.appendChild(t)
  const jump = el('div', { className: 'row' })
  const idx = numInput(st.detail ?? '', 80, { min: 0, placeholder: 'index' })
  jump.append(el('span', { className: 'small', textContent: `Any neuron (0…${r.n_neurons - 1}):` }), idx, btn('Look at it', () => { if (idx.value !== '') { st.detail = +idx.value; showNeuron(det, st.pos, +idx.value) } }, 'mini'))
  out.appendChild(jump)
  const det = el('div'); out.appendChild(det)
  if (st.detail != null) showNeuron(det, st.pos, st.detail)
}

export async function showNeuron(box, pos, index) {
  box.innerHTML = spinner()
  let d
  try { d = await GET(`/api/runs/${S.run.id}/neuron?pos=${pos}&index=${index}&through=${S.neu.through || 0}`) } catch (e) { box.innerHTML = `<div class="bad">${esc(e.message)}</div>`; return }
  const mx = Math.max(...d.acts.map(Math.abs), 1e-9)
  box.innerHTML = `<div class="kvbox"><h4>Neuron ${d.index} · block #${d.pos} (layer ${d.layer}) · activation at every token (green = positive, red = negative)</h4>
    <div class="tokens">${d.tokens.map((t, i) => `<span class="tok" title="#${i}: ${d.acts[i]}" style="background:${recColor(d.acts[i] / mx)}">${esc(t || '∅')}</span>`).join('')}</div>
    <div class="small">range ${d.min} … ${d.max} · pushes for ${d.promotes.map(w => `<span class="chip">${esc(w)}</span>`).join('')} · against ${d.suppresses.map(w => `<span class="chip dim">${esc(w)}</span>`).join('')}</div></div>`
}

registerTab({ id: 'neurons', title: 'Neurons', hint: 'Contrastive neuron activity, vocabulary projection, batch compare, and token inspector', render: renderNeurons })
