// Introspection tab — three experiments on the model's access to its own activations:
//   1 concept injection (does it notice a thought we put there?)   2 concept meter (think / don't think about X)
//   3 activation → words → activation (a natural-language autoencoder with nothing trained)
import { GET, POST, S, btn, el, flow, on, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBox, aliveIn, blockSelect, followTask, labelled, numInput, recColor, refreshRuns0, refreshVectors, runLabel, runsWithCaps, selectOf } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'

S.intro = { concepts: 'ocean, bread, music, justice', blocks: '', strengths: '0.1, 0.2, 0.4, 0.8', tokens: 40, result: null, cell: null, metric: 'introspect',
  word: 'ocean', vpos: null, vector: '', other: '', meter: null,
  npos: null, ntoken: -1, inject: 2, scale: 1, template: '', own: '', nla: null, tries: [], defaults: null }

const nums = s => String(s).split(/[,\s]+/).map(Number).filter(x => !Number.isNaN(x))
const section = (title, open, sub) => { const d = el('details', { className: 'kvbox', open }); d.appendChild(el('summary', {}, `${title}${sub ? ` <span class="small">${sub}</span>` : ''}`)); return d }

async function renderIntro(p) {
  const st = S.intro, nL = S.info.n_layers
  if (!st.defaults) { try { st.defaults = await GET('/api/introspect/defaults') } catch { st.defaults = {} } }
  await Promise.all([refreshVectors(), refreshRuns0()])
  if (S.tab !== 'introspect') return
  if (!st.blocks) st.blocks = [0.3, 0.45, 0.6, 0.75].map(f => Math.round(nL * f)).filter((v, i, a) => a.indexOf(v) === i).join(', ')
  if (st.vpos == null) st.vpos = Math.round(nL * 0.6)
  p.appendChild(el('div', { className: 'small' }, `<b>Can a model look at its own activations?</b> When a model says “I was thinking about the ocean”, is that a report of something inside it, or just a plausible sentence?
    The only way to tell is to know the ground truth — so we <i>put</i> a known thing inside and ask. These three experiments are small versions of published research; small models pass few of them, and seeing <i>how</i> they fail is the lesson.`))
  p.append(injectSection(st), meterSection(st), nlaSection(st))
}

// ---------------------------------------------------------------- 1 · concept injection
function injectSection(st) {
  const s = section('1 · Concept injection: do you detect an injected thought?', true)
  s.appendChild(el('div', { className: 'small' }, `A <b>concept vector</b> is what the model’s residual stream looks like when it reads “Tell me about <i>ocean</i>.”, minus the average over unrelated words.
    We add that vector to a block while the model answers a question about its own state. A model with real access to its activations would say <i>“yes — something about the ocean”</i> when we inject and <i>“no”</i> when we don’t.
    Three ways to fail: it never notices; it says yes even on the control trial; or the vector simply takes over its mouth (“ocean ocean ocean”) — which is steering, not introspection.`))
  const fill = btn('Put the question into the Prompt box', () => {
    const pn = flow.nodes.find(n => n.data.kind === 'input_prompt'), sm = flow.nodes.find(n => n.data.kind === 'sampler')
    flow.updateNodeData(pn.id, { prompt: st.defaults.prompt, system: '', prefill: '', use_chat_template: !!S.info.has_chat_template })
    flow.updateNodeData(sm.id, { max_new_tokens: st.tokens, temperature: 0 })
    toast('The Prompt box now asks “Do you detect an injected thought?”. You can edit the wording there.')
  })
  const inp = (key, label, w) => { const i = el('input', { value: st[key], style: `width:${w}px` }); i.oninput = () => { st[key] = i.value }; return labelled(label, i) }
  const out = el('div')
  const go = btn('Run the sweep', async () => {
    const alive = aliveIn('introspect')
    st.result = null; st.cell = null; out.innerHTML = spinner('building concept vectors…')
    try {
      await followTask(await POST('/api/introspect/inject', { flow: flow.getGraph(), concepts: st.concepts.split(',').map(x => x.trim()).filter(Boolean),
        blocks: nums(st.blocks), strengths: nums(st.strengths), max_new_tokens: +st.tokens }), r => { st.result = r; if (alive()) drawInject(out, st) }, alive)
    } catch (e) { if (alive()) out.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
  }, 'primary')
  const row = el('div', { className: 'row' })
  row.append(fill, inp('concepts', 'Concepts', 230), inp('blocks', 'Inject after blocks', 120), inp('strengths', 'Strengths (× the stream’s own size)', 130), inp('tokens', 'Answer length', 50), go)
  s.append(row, out)
  if (st.result) drawInject(out, st)
  return s
}

const METRICS = [['introspect', 'detects it, then names it (the real test)'], ['claims', 'says it detects something'], ['named', 'the concept word appears'], ['coherent', 'answer is still coherent'], ['p_yes', 'probability the answer starts with “Yes”']]

function drawInject(out, st) {
  const r = st.result
  out.innerHTML = ''
  if (!r.control) { out.innerHTML = spinner(r.status); return }
  const c = r.control, fp = Object.values(c.by_concept)[0]?.claims
  out.appendChild(el('div', { className: 'ctrl' }, `<b>Control trial (nothing injected)</b> — P(first word “Yes”) = ${c.p_yes}, P(“No”) = ${c.p_no}<div class="sample">${esc(c.text)}</div>
    <span class="${fp ? 'bad' : 'ok'}">${fp ? 'It claims to detect a thought although none was injected: a false positive. Any “yes” below must be read against this.' : 'It correctly reports nothing. A “yes” below now means something.'}</span>`))
  if (!r.cells) { out.appendChild(el('div', {}, spinner(r.status))); return }
  const m = selectOf(METRICS, st.metric); m.onchange = () => { st.metric = m.value; drawInject(out, st) }
  const head = el('div', { className: 'row' }); head.append(labelled('Show', m), el('span', { className: 'small' }, r.status === 'done' ? '' : spinner(`${Math.round(r._progress * 100)}%`)))
  out.appendChild(head)
  const t = el('table', { className: 'grid heat' })
  t.innerHTML = `<tr><th>inject after ↓ · strength →</th>${r.strengths.map(x => `<th>${x}</th>`).join('')}</tr>`
  r.cells.forEach((rowv, bi) => {
    const tr = el('tr'); tr.appendChild(el('th', { textContent: r.block_labels[bi] }))
    rowv.forEach((cell, si) => {
      const v = cell?.[st.metric]
      const td = el('td', { textContent: cell ? v.toFixed(2) : '', title: cell ? `${cell.n} concept(s): claims ${cell.claims} · named ${cell.named} · introspective ${cell.introspect} · coherent ${cell.coherent}` : 'not run yet' })
      td.style.background = recColor(v); if (st.cell && st.cell[0] === bi && st.cell[1] === si) td.style.outline = '2px solid var(--warn)'
      if (cell) td.onclick = () => { st.cell = [bi, si]; drawInject(out, st) }
      tr.appendChild(td)
    })
    t.appendChild(tr)
  })
  out.appendChild(t)
  if (r.summary) {
    const sm = r.summary
    out.appendChild(el('div', { className: 'small' }, `<b>Result.</b> Best setting (block #${sm.best_block}, strength ${sm.best_strength}): detected-then-named on <b>${Math.round(sm.best_introspect * 100)}%</b> of concepts;
      false-positive rate on the control: <b>${Math.round(sm.false_positive * 100)}%</b>; net introspection score <b>${Math.round(Math.max(0, sm.net) * 100)}%</b>.
      For scale: the published study reports about 20% for its best model at its best layer and strength, and near zero for most settings — and those are far larger models than this one.`))
  }
  if (st.cell) {
    const [bi, si] = st.cell, ts = r.trials.filter(x => x.bi === bi && x.si === si)
    const box = el('div', { className: 'kvbox' }, `<h4>Answers with the vector injected after ${esc(r.block_labels[bi])} at strength ${r.strengths[si]}</h4>`)
    for (const x of ts) {
      const d = el('div', { className: 'sample' }, `<span class="chip">${esc(x.concept)}</span> ${x.introspect ? '<span class="chip ok">detected → named</span>' : x.named && !x.claims ? '<span class="chip dim">said the word without noticing</span>' : x.claims ? '<span class="chip dim">claims, wrong or no word</span>' : ''}${x.coherent ? '' : ' <span class="chip bad">incoherent</span>'} ${esc(x.text)} `)
      d.appendChild(btn('Wire this into the flow', async () => {
        try {
          const v = await POST('/api/introspect/vector', { flow: flow.getGraph(), word: x.concept, pos: x.block })
          await refreshVectors()
          const node = (S.planNodes || [])[x.block] || flow.nodes.find(n => n.data.kind === 'block' && n.data.layer === x.block)?.id
          const n = addBox(node, { label: `inject: ${x.concept}`, hook: 'resid_post', op: 'steer', ref: v.id, value: +(x.strength * v.resid_norm / v.norm).toFixed(3) })
          if (n) toast(`Added a steering box with the “${x.concept}” vector. Step through the run to watch it act.`)
        } catch (e) { toast(e.message, 'error') }
      }, 'mini'))
      box.appendChild(d)
    }
    out.appendChild(box)
  }
}

// ---------------------------------------------------------------- 2 · concept meter
function meterSection(st) {
  const s = section('2 · Concept meter: is the model “thinking about” something it does not say?', false)
  s.appendChild(el('div', { className: 'small' }, `Measure, at every block and token of a run, how strongly the residual stream points along a concept vector (cosine, after removing what all text shares).
    <b>Experiment:</b> run “Write one sentence about the weather. While you write, think about aquariums.” and then the same with “do <i>not</i> think about aquariums”. Make an <i>aquariums</i> vector and compare the two runs on the generated tokens.
    The published finding: the concept is more active under “think” than “don’t think” — but above baseline in both, the way “don’t think of a polar bear” fails for people.`))
  const w = el('input', { value: st.word, style: 'width:110px' }); w.oninput = () => { st.word = w.value }
  const pos = numInput(st.vpos, 56, { min: 0, max: S.info.n_layers - 1 }); pos.oninput = () => { st.vpos = +pos.value }
  const row = el('div', { className: 'row' })
  row.append(labelled('Make a concept vector for', w), labelled('at block', pos), btn('Make', async () => {
    try { const v = await POST('/api/introspect/vector', { flow: flow.getGraph(), word: st.word, pos: st.vpos }); st.vector = v.id; toast(`Saved “${v.name}”.`); renderTab() } catch (e) { toast(e.message, 'error') }
  }))
  s.appendChild(row)
  const vecs = (S.vectors || []).filter(v => !v.error && v.dim === S.info.d_model)
  if (!S.run?.outputs?.tokens) { s.appendChild(el('div', { className: 'empty', textContent: 'Run the flow, then measure the run here.' })); return s }
  if (!vecs.length) { s.appendChild(el('div', { className: 'empty', textContent: 'No vector for this model yet: make one above (or in the Steering tab).' })); return s }
  if (!vecs.find(v => v.id === st.vector)) st.vector = vecs[0].id
  const runs = runsWithCaps().filter(r => r.id !== S.run.id)
  const vs = selectOf(vecs.map(v => [v.id, v.name]), st.vector); vs.onchange = () => { st.vector = vs.value; renderTab() }
  const os = selectOf([['', '— none —'], ...runs.map(r => [r.id, runLabel(r)])], st.other); os.onchange = () => { st.other = os.value; renderTab() }
  const row2 = el('div', { className: 'row' }); row2.append(labelled('Vector', vs), labelled('Compare with run', os))
  const out = el('div', {}, spinner()); s.append(row2, out)
  const alive = aliveIn('introspect')
  Promise.all([GET(`/api/runs/${S.run.id}/meter?vector=${st.vector}`), st.other ? GET(`/api/runs/${st.other}/meter?vector=${st.vector}`) : null]).then(([a, b]) => {
    if (!alive()) return
    out.innerHTML = ''
    // Long prompts (chat templates) push the generated tokens off screen: by default show the end of the prompt onward.
    const from = st.mall || a.tokens.length <= 40 || a.n_in >= a.tokens.length ? 0 : Math.max(0, a.n_in - 6)
    const mx = Math.max(...a.rows.flatMap(r => r.cos.slice(from).map(Math.abs)), 1e-9)
    const t = el('table', { className: 'grid heat meter' })
    t.innerHTML = `<tr><th></th>${a.tokens.slice(from).map((x, i) => `<th class="${i + from >= a.n_in ? 'gen' : ''}">${esc(x)}</th>`).join('')}</tr>` +
      a.rows.map(r => `<tr><th>#${r.pos}</th>${r.cos.slice(from).map(v => `<td title="${v}" style="background:${recColor(v / mx)}"></td>`).join('')}</tr>`).join('')
    const cap = el('div', { className: 'small' }, `Run ${esc(S.run.id)}: green = points along “${esc(a.vector.name)}”, red = against. Underlined tokens were generated. Largest |cosine| shown = ${mx.toFixed(3)}. `)
    if (a.tokens.length > 40 && a.n_in < a.tokens.length) cap.appendChild(btn(st.mall ? 'Show generated tokens only' : `Show all ${a.tokens.length} tokens`, () => { st.mall = !st.mall; renderTab() }, 'mini'))
    out.appendChild(cap)
    const wrap = el('div', { className: 'lens-wrap' }); wrap.appendChild(t); out.appendChild(wrap)
    const gm = rows => rows.map(r => r.mean_generated ?? r.mean_prompt)
    const A = gm(a.rows), B = b ? gm(b.rows) : null, m2 = Math.max(...A.map(Math.abs), ...(B || []).map(Math.abs), 1e-9)
    const cmp = el('table', { className: 'grid' })
    cmp.innerHTML = `<tr><th>Block</th><th>This run (${esc(S.run.id)}) — mean over ${a.rows[0].mean_generated == null ? 'prompt' : 'generated'} tokens</th>${B ? `<th>Run ${esc(st.other)}</th><th>Difference</th>` : ''}</tr>` +
      a.rows.map((r, i) => `<tr><td>#${r.pos}</td><td><span class="pbar${A[i] < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(A[i]) / m2 * 100).toFixed(0)}%"></i></span> ${A[i]}</td>` +
        (B ? `<td><span class="pbar${B[i] < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(B[i] ?? 0) / m2 * 100).toFixed(0)}%"></i></span> ${B[i] ?? ''}</td><td>${B[i] == null ? '' : (A[i] - B[i]).toFixed(4)}</td>` : '') + '</tr>').join('')
    out.appendChild(cmp)
  }).catch(e => { if (alive()) out.innerHTML = `<div class="bad">${esc(e.message)}</div>` })
  return s
}

// ---------------------------------------------------------------- 3 · activation → words → activation
function nlaSection(st) {
  const s = section('3 · Activation → words → activation: can a thought be written down?', false)
  s.appendChild(el('div', { className: 'small' }, `A <b>natural-language autoencoder</b> turns an activation into a sentence (the <i>verbalizer</i>) and the sentence back into an activation (the <i>reconstructor</i>).
    The explanation is judged by one thing: <b>can the activation be rebuilt from it?</b> The published method trains both halves. Here nothing is trained: the model verbalizes by answering “what does ▢ mean?” with the activation patched in for ▢,
    and reconstructs by reading the answer. Then <b>write your own explanation and try to beat it</b> — and notice that both of you are far from the ceiling. Closing that gap is what the training is for.`))
  if (!S.run?.outputs?.tokens) { s.appendChild(el('div', { className: 'empty', textContent: 'Run the flow first, then pick one of its activations here.' })); return s }
  const o = S.run.outputs, nS = S.run.plan.schedule.length, T = o.tokens.length
  if (st.npos == null || st.npos >= nS) st.npos = Math.round(nS * 0.5)
  const tsel = Math.max(1, Math.min(T - 1, st.ntoken < 0 ? T + st.ntoken : st.ntoken))
  const strip = el('div', { className: 'tokens' })
  o.tokens.forEach((t, i) => { const sp = el('span', { className: `tok${i === tsel ? ' on' : ''}${i === 0 ? ' dim' : ''}`, textContent: t || '∅', title: i === 0 ? 'token 0 is the attention sink: not readable' : `#${i}` }); if (i) sp.onclick = () => { st.ntoken = i; st.nla = null; st.tries = []; renderTab() }; strip.appendChild(sp) })
  const inj = numInput(st.inject, 50, { min: 0, max: nS - 1 }); inj.oninput = () => { st.inject = +inj.value }
  const sc = numInput(st.scale, 50, { min: 0.1 }); sc.oninput = () => { st.scale = +sc.value }
  const tpl = el('input', { value: st.template || st.defaults.explain_template || '', style: 'width:330px' }); tpl.oninput = () => { st.template = tpl.value }
  const out = el('div')
  const row = el('div', { className: 'row' })
  row.append(labelled('Read the stream after block', blockSelect(st.npos, v => { st.npos = v; st.nla = null; st.tries = []; renderTab() })),
    labelled('Patch it in at block', inj), labelled('× size', sc), labelled('Question ({x} = the activation)', tpl),
    btn('Verbalize and rebuild', async () => {
      const alive = aliveIn('introspect'); st.nla = null; out.innerHTML = spinner('verbalizing…')
      try {
        await followTask(await POST('/api/nla/roundtrip', { run_id: S.run.id, pos: st.npos, token: tsel, inject: st.inject, scale: st.scale, template: st.template }),
          r => { st.nla = r; if (alive()) drawNla(out, st, tsel) }, alive)
      } catch (e) { if (alive()) out.innerHTML = `<div class="bad">${esc(e.message)}</div>` }
    }, 'primary'))
  s.append(strip, row, out)
  if (st.nla) drawNla(out, st, tsel)
  return s
}

function drawNla(out, st, tsel) {
  const r = st.nla
  out.innerHTML = ''
  out.appendChild(el('div', { className: 'ctrl' }, `<b>Activation:</b> stream after block #${r.pos} at token #${r.token} “${esc(r.piece)}” (length ${r.norm}).
    <b>Asked:</b> ${esc(r.prompt || '')} <div class="sample"><b>The model’s explanation:</b> ${r.explanation == null ? spinner(r.status) : esc(r.explanation) || '<i>(nothing)</i>'}</div>`))
  const rows = []
  if (r.score) rows.push({ label: 'the model’s own explanation', text: r.raw, ...r.score, cls: 'on' })
  st.tries.forEach(t => rows.push({ label: 'your explanation', ...t, cls: 'mine' }))
  r.baselines.forEach(b => rows.push({ ...b }))
  if (rows.length) {
    const t = el('table', { className: 'grid' })
    t.innerHTML = `<tr><th>Rebuilt from</th><th>Text the reconstructor read</th><th title="Cosine similarity between the rebuilt and the original activation after subtracting the average activation on ordinary text. 1 = same direction, 0 = unrelated.">Match (centred cosine)</th><th title="Fraction of variance explained: 1 = perfect, 0 = no better than guessing the average activation, negative = worse than that.">Variance explained</th></tr>` +
      rows.sort((a, b) => b.centred - a.centred).map(x => `<tr class="${x.cls || ''}"><td>${esc(x.label)}</td><td class="small">${esc((x.text || '').slice(0, 90))}</td>
        <td><span class="pbar${x.centred < 0 ? ' alt' : ''}"><i style="width:${(Math.abs(x.centred) * 100).toFixed(0)}%"></i></span> ${x.centred}</td><td>${x.fve}</td></tr>`).join('')
    out.appendChild(t)
  }
  if (r.status !== 'done') { out.appendChild(el('div', {}, spinner(r.status))); return }
  const own = el('input', { placeholder: 'Write your own explanation of what this activation represents…', value: st.own, style: 'width:460px' }); own.oninput = () => { st.own = own.value }
  const row = el('div', { className: 'row' })
  row.append(own, btn('Score mine', async () => {
    if (!st.own.trim()) return
    try { st.tries.push(await POST('/api/nla/score', { run_id: S.run.id, pos: r.pos, token: tsel, text: st.own })); drawNla(out, st, tsel) } catch (e) { toast(e.message, 'error') }
  }))
  out.appendChild(row)
  out.appendChild(el('div', { className: 'small' }, `<b>How to read this.</b> “The text up to the token” is the ceiling: reading the same context rebuilds the activation almost exactly, because that is how it was made.
    “An unrelated token” is the floor. An explanation is informative only as far as it rises above the floor; a <i>trained</i> verbalizer is one that has learned to write the sentence that gets closest to the ceiling.
    Note what a fluent, confident explanation that scores at the floor means: a model’s description of its own activation can sound right and be wrong.`))
}

registerTab({ id: 'introspect', title: 'Introspection', hint: 'Concept injection, concept meter, and activation → words → activation', render: renderIntro })
on('model', () => { S.intro.result = null; S.intro.nla = null; S.intro.blocks = ''; S.intro.vpos = null; S.intro.npos = null; S.intro.tries = [] })
