// Concept Subspace & Causal Steering Eval Playbook tab (unlocked by `concept_playbook` Playbook extension).
import { GET, POST, S, btn, el, flow, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { addBox, labelled, numInput, refreshVectors, selectOf } from '../ui/labkit.js'
import { renderInspector } from '../ui/inspector.js'
import { renderTab } from '../ui/tabs.js'

S.cpb = {
  stage: 'extract',             // 'extract' | 'eval'
  // Stage 1: extraction
  name: 'moral_concept',
  pos: -1,
  pooling: 'mean',
  standardize: false,
  chat: false,
  addBos: true,
  groupsJson: '',
  extRes: null,
  // Stage 2: multi-metric steering & ablation eval
  vectorId: '',
  shufId: '',
  randId: '',
  blocks: '',
  op: 'steer',
  scale: 'resid',
  strengths: '-0.15, 0.15',
  posToken: 'Yes',
  negToken: 'No',
  includeControls: true,
  evalRes: null,
}

export async function renderConcepts(p) {
  const st = S.cpb
  if (!S.vectors) await refreshVectors()
  if (!st.groupsJson) {
    try {
      const d = await GET('/api/ext/concept_playbook/defaults')
      if (d?.groups) st.groupsJson = JSON.stringify(d.groups, null, 2)
    } catch {}
  }
  p.innerHTML = ''

  const topBar = el('div', { className: 'row', style: 'justify-content:space-between;margin-bottom:8px' })
  const stageBtns = el('div', { className: 'row' })
  stageBtns.append(
    btn('Stage 1 · Contrastive Group Extraction & Transfer', () => { st.stage = 'extract'; renderTab() }, st.stage === 'extract' ? 'primary mini' : 'mini ghost'),
    btn('Stage 2 · Multi-Metric Steering & Ablation Eval', () => { st.stage = 'eval'; renderTab() }, st.stage === 'eval' ? 'primary mini' : 'mini ghost'),
  )
  topBar.append(
    stageBtns,
    el('span', { className: 'small' }, `<span class="chip" style="background:rgba(59,130,246,0.2)">Playbook View</span> Extract cross-group directions and benchmark causal interventions against shuffled/random controls.`),
  )
  p.appendChild(topBar)

  if (st.stage === 'extract') {
    return renderExtractStage(p, st)
  }
  return renderEvalStage(p, st)
}

function renderExtractStage(p, st) {
  const ctl = el('div', { className: 'kvbox' })
  ctl.appendChild(el('div', { className: 'small', style: 'margin-bottom:6px' },
    `<b>Stage 1: Paired Contrastive Subspace Extraction & Cross-Group Transfer.</b> Runs paired <code>(pos, neg)</code> prompts across groups (e.g. languages <code>EN</code>, <code>DE</code>, <code>FR</code>) with lockstep <code>run_batch(max_new_tokens=0)</code>, locates the block with peak Cohen's <i>d'</i> separation, evaluates the <i>G × G</i> cross-group transfer matrix, and pipes the extracted direction directly into Stage 2.`))

  const row = el('div', { className: 'row' })
  const nmInp = el('input', { type: 'text', value: st.name, style: 'width:130px' })
  nmInp.oninput = () => { st.name = nmInp.value }

  const posInp = numInput(st.pos, 60)
  posInp.title = '-1 = auto-select best block by Cohen d\''
  posInp.oninput = () => { st.pos = +posInp.value }

  const poolSel = selectOf([['mean', 'mean over prompt tokens'], ['last', 'last prompt token']], st.pooling)
  poolSel.onchange = () => { st.pooling = poolSel.value }

  const cbStd = el('input', { type: 'checkbox', checked: st.standardize })
  cbStd.onchange = () => { st.standardize = cbStd.checked }
  const lStd = el('label', { className: 'inl' }); lStd.append(cbStd, el('span', { textContent: 'Per-dim z-score standardize' }))

  const cbChat = el('input', { type: 'checkbox', checked: st.chat })
  cbChat.onchange = () => { st.chat = cbChat.checked }
  const lChat = el('label', { className: 'inl' }); lChat.append(cbChat, el('span', { textContent: 'Chat template' }))

  const cbBos = el('input', { type: 'checkbox', checked: st.addBos })
  cbBos.onchange = () => { st.addBos = cbBos.checked }
  const lBos = el('label', { className: 'inl' }); lBos.append(cbBos, el('span', { textContent: 'Add <bos>' }))

  const runBtn = btn('▶ Run Extraction & Transfer Matrix', async () => {
    let groups = null
    try { groups = JSON.parse(st.groupsJson) }
    catch (e) { return toast('Invalid groups JSON: ' + e.message, 'error') }
    resBox.innerHTML = `<div class="empty">${spinner('running batched forward passes and computing transfer matrix…')}</div>`
    try {
      const r = await POST('/api/ext/concept_playbook/extract', {
        groups,
        pos: +st.pos,
        pooling: st.pooling,
        standardize: st.standardize,
        chat: st.chat,
        add_bos: st.addBos,
        persist: true,
        name: st.name || 'concept_dir',
      })
      st.extRes = r
      st.vectorId = r.vector_id
      st.shufId = r.shuffled_vector_id
      st.randId = r.random_vector_id
      st.blocks = String(r.chosen_block)
      await refreshVectors()
      renderTab()
    } catch (e) { toast(e.message, 'error'); resBox.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
  }, 'primary')

  row.append(
    labelled('Vector name', nmInp),
    labelled('Block (-1=auto)', posInp),
    labelled('Pooling', poolSel),
    lStd, lChat, lBos, runBtn,
  )
  ctl.appendChild(row)

  const det = el('details', { style: 'margin-top:6px' }, `<summary class="small" style="cursor:pointer">Edit paired (pos, neg) prompt groups JSON</summary>`)
  const ta = el('textarea', { rows: 6, value: st.groupsJson, style: 'width:100%;font-family:monospace;font-size:11px;margin-top:4px' })
  ta.oninput = () => { st.groupsJson = ta.value }
  det.appendChild(ta)
  ctl.appendChild(det)
  p.appendChild(ctl)

  const resBox = el('div')
  p.appendChild(resBox)
  if (!st.extRes) return

  const r = st.extRes
  const cards = el('div', { className: 'row', style: 'gap:10px;margin:8px 0;align-items:stretch' })
  const card = (t, v, s) => el('div', { className: 'kvbox', style: 'flex:1;margin:0' },
    `<div class="small">${esc(t)}</div><div style="font-size:16px;font-weight:700;margin:2px 0">${v}</div><div class="small">${s}</div>`)
  cards.append(
    card('Best / Chosen Block', `Block #${r.chosen_block} (best #${r.best_block})`, `Cohen's d' = <b>${r.d_prime}</b>`),
    card('Within-Group Paired Acc', `${(r.within_paired_acc * 100).toFixed(1)}%`, `Cross-group paired: <b>${(r.cross_paired_acc * 100).toFixed(1)}%</b>`),
    card('Cross-Group Unpaired Acc', `${(r.cross_unpaired_acc * 100).toFixed(1)}%`, `Mean cosine: <b>${r.mean_cosine}</b>`),
  )
  const nextBtn = btn('Proceed to Stage 2 (Causal Eval) ▸', () => { st.stage = 'eval'; renderTab() }, 'primary')
  cards.appendChild(el('div', { style: 'display:flex;align-items:center' })); cards.lastChild.appendChild(nextBtn)
  resBox.appendChild(cards)

  const twoCols = el('div', { className: 'row', style: 'align-items:flex-start;gap:12px' })
  // Left: G x G transfer matrix
  const tBox = el('div', { style: 'flex:1' }, `<h4>Cross-Group Transfer Accuracy (Train Row -> Eval Col)</h4>`)
  const tTbl = el('table', { className: 'grid' })
  let th = '<thead><tr><th>Train \\ Eval</th>' + r.groups.map(g => `<th>${esc(g)} (paired / unpaired)</th>`).join('') + '</tr></thead><tbody>'
  r.groups.forEach((g, i) => {
    th += `<tr><th>${esc(g)}</th>`
    r.groups.forEach((_, j) => {
      const pa = r.transfer_paired[i][j], ua = r.transfer_unpaired[i][j]
      th += `<td><b>${(pa * 100).toFixed(0)}%</b> / ${(ua * 100).toFixed(0)}%</td>`
    })
    th += '</tr>'
  })
  tTbl.innerHTML = th + '</tbody>'
  tBox.appendChild(tTbl)

  // Right: Per-block d' table
  const lBox = el('div', { style: 'flex:1' }, `<h4>Per-Block Separation (click to select for Stage 2)</h4>`)
  const lTbl = el('table', { className: 'grid' })
  lTbl.innerHTML = '<thead><tr><th>Block</th><th>Paired Acc</th><th>Unpaired Acc</th><th>Cohen d\'</th></tr></thead><tbody></tbody>'
  const ltb = lTbl.querySelector('tbody')
  for (const l of r.layers) {
    const tr = el('tr', { className: l.pos === r.chosen_block ? 'activerow' : '' })
    tr.style.cursor = 'pointer'
    tr.onclick = () => { st.pos = l.pos; st.blocks = String(l.pos); renderTab() }
    tr.innerHTML = `<td><b>#${l.pos}</b></td><td>${(l.paired_acc * 100).toFixed(1)}%</td><td>${(l.unpaired_acc * 100).toFixed(1)}%</td><td><b>${l.d_prime}</b></td>`
    ltb.appendChild(tr)
  }
  lBox.appendChild(lTbl)
  twoCols.append(tBox, lBox)
  resBox.appendChild(twoCols)
}

function renderEvalStage(p, st) {
  const ctl = el('div', { className: 'kvbox' })
  ctl.appendChild(el('div', { className: 'small', style: 'margin-bottom:6px' },
    `<b>Stage 2: Multi-Metric Causal Steering & Ablation Benchmark.</b> Evaluates <code>steer</code> (with residual-norm scaling <code>scale="resid"</code> so <code>c=0.15</code> adds 15% of each token's residual norm) or <code>project_out</code> against <b>shuffled-label</b> and <b>random Gaussian</b> controls across 3 simultaneous metrics: (1) Yes/No log-prob margin & ROC AUC, (2) Capability accuracy on <code>run.answer</code> (with thinking stripped), and (3) Neutral text NLL / perplexity.`))

  const nL = S.info?.n_layers || 24
  if (!st.blocks) st.blocks = String(Math.floor(nL / 2))
  const vecOpts = [['', '- choose a vector (or run Stage 1) -'], ...(S.vectors || []).map(v => [v.id, `${v.name} (norm ${v.norm})`])]
  if (st.vectorId && !vecOpts.find(x => x[0] === st.vectorId)) {
    vecOpts.splice(1, 0, [st.vectorId, `${st.vectorId} (extracted in Stage 1)`])
  }

  const row = el('div', { className: 'row' })
  const vSel = selectOf(vecOpts, st.vectorId)
  vSel.onchange = () => { st.vectorId = vSel.value }

  const bInp = el('input', { type: 'text', value: st.blocks, style: 'width:75px', title: 'Single block (e.g. 12) or slice (e.g. 10:15)' })
  bInp.oninput = () => { st.blocks = bInp.value }

  const opSel = selectOf([
    ['steer', 'steer (add direction)'],
    ['project_out', 'project_out (ablate direction)'],
  ], st.op)
  opSel.onchange = () => {
    st.op = opSel.value
    if (st.op === 'project_out') st.strengths = '1.0'
    else if (st.strengths === '1.0') st.strengths = '-0.15, 0.15'
    renderTab()
  }

  const scSel = selectOf([
    ['resid', 'resid (c × ||resid|| × u)'],
    ['unit', 'unit (c × u)'],
    ['raw', 'raw (c × v as-is)'],
  ], st.scale)
  scSel.onchange = () => { st.scale = scSel.value }

  const strInp = el('input', { type: 'text', value: st.strengths, style: 'width:105px' })
  strInp.oninput = () => { st.strengths = strInp.value }

  const cbCtrl = el('input', { type: 'checkbox', checked: st.includeControls })
  cbCtrl.onchange = () => { st.includeControls = cbCtrl.checked }
  const lCtrl = el('label', { className: 'inl' }); lCtrl.append(cbCtrl, el('span', { textContent: '+ Shuffled & Random Controls' }))

  const runBtn = btn('▶ Run Multi-Metric Eval', async () => {
    if (!st.vectorId) return toast('Pick a vector or run Stage 1 first.', 'error')
    const strengths = st.strengths.split(',').map(x => parseFloat(x.trim())).filter(x => !Number.isNaN(x))
    resBox.innerHTML = `<div class="empty">${spinner('evaluating judgment margin, capability answers, and neutral NLL in lockstep…')}</div>`
    try {
      st.evalRes = await POST('/api/ext/concept_playbook/eval', {
        vector_id: st.vectorId,
        shuffled_vector_id: st.shufId || null,
        random_vector_id: st.randId || null,
        blocks: st.blocks,
        op: st.op,
        scale: st.scale,
        strengths,
        include_controls: st.includeControls,
        pos_token: st.posToken,
        neg_token: st.negToken,
      })
      renderTab()
    } catch (e) { toast(e.message, 'error'); resBox.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
  }, 'primary')

  const wireBtn = btn('+ Wire into Flow Canvas', () => {
    if (!st.vectorId) return toast('Pick a vector first.', 'error')
    const firstStr = parseFloat((st.strengths || '0.15').split(',').pop()) || 0.15
    const bList = (st.evalRes?.blocks?.length ? st.evalRes.blocks : [parseInt(st.blocks, 10) || 0])
    for (const b of bList) {
      const node = flow.nodes.find(n => n.data?.kind === 'block' && +n.data.layer === +b) || flow.nodes.find(n => n.data?.kind === 'block')
      if (node) {
        addBox(node.id, {
          label: `${st.op} ${st.scale} ×${firstStr}`,
          hook: 'resid_post',
          op: st.op,
          ref: st.vectorId,
          value: firstStr,
          scale: st.scale,
        })
      }
    }
    renderInspector()
    toast(`Wired ${st.op} (${st.scale}) into ${bList.length} block(s) on the flow canvas.`)
  })

  row.append(
    labelled('Vector', vSel),
    labelled('Block(s)', bInp),
    labelled('Op', opSel),
    ...(st.op === 'steer' ? [labelled('Scale', scSel)] : []),
    labelled('Strengths', strInp),
    lCtrl, runBtn, wireBtn,
  )
  ctl.appendChild(row)
  p.appendChild(ctl)

  const resBox = el('div')
  p.appendChild(resBox)
  if (!st.evalRes) return

  const r = st.evalRes
  const tbl = el('table', { className: 'grid' })
  tbl.innerHTML = `<thead><tr>
    <th>Condition</th>
    <th>Pos Margin (${esc(r.pos_token)}-${esc(r.neg_token)})</th>
    <th>Neg Margin (${esc(r.pos_token)}-${esc(r.neg_token)})</th>
    <th>Δ Pos Margin</th>
    <th>ROC AUC</th>
    <th>Capability Acc (run.answer)</th>
    <th>Neutral NLL (Perplexity)</th>
    <th>Sample Capability Reply (Answer | Thought)</th>
  </tr></thead><tbody></tbody>`
  const tb = tbl.querySelector('tbody')
  for (const row of r.results) {
    const tr = el('tr')
    const samp = row.sample ? `<b>${esc(row.sample.answer || '∅')}</b>${row.sample.thought ? ` <span class="small">(thought: ${esc(row.sample.thought)})</span>` : ''}` : '-'
    tr.innerHTML = `
      <td><b>${esc(row.condition)}</b></td>
      <td>${row.pos_margin}</td>
      <td>${row.neg_margin}</td>
      <td><b>${row.delta_pos_margin >= 0 ? '+' : ''}${row.delta_pos_margin}</b></td>
      <td><b>${row.auc}</b></td>
      <td><span class="chip ${row.cap_acc >= 0.66 ? 'ok' : 'bad'}">${(row.cap_acc * 100).toFixed(0)}%</span></td>
      <td>${row.nll} <span class="small">(PPL ${row.ppl})</span></td>
      <td>${samp}</td>
    `
    tb.appendChild(tr)
  }
  resBox.appendChild(tbl)
}

registerTab({ id: 'concepts', title: 'Concept Playbook', render: renderConcepts })
