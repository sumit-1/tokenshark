// Cosine Similarity & Outlier Diagnostics tab (unlocked by `cosine_diag` Diagnose extension).
import { GET, POST, S, btn, el, flow, heat, registerTab, spinner, toast } from '../core.js'
import { esc } from '../flow.js'
import { labelled, numInput, refreshVectors, runsWithCaps, runLabel, selectOf } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'

S.diag = {
  view: 'outliers',             // 'outliers' | 'cosine'
  // Outlier scanner state
  outSource: 'prompts',         // 'run' | 'prompts'
  outHook: 'resid_post',
  outK: 10,
  outChat: false,
  outAddBos: true,
  outCompareNoBos: true,
  outVectorRef: '',
  outPrompts: [
    'Taking a wallet you found on the street and keeping the cash inside.',
    'Returning a lost wallet to its owner with all the cash intact.',
    'Lying to a customer about a product defect to close a sale.',
    'Disclosing known product limitations honestly before a customer buys.',
  ].join('\n'),
  outRes: null,
  // Cosine matrix state
  cosMode: 'groups',            // 'groups' | 'vectors' | 'layers'
  cosPos: -1,
  cosPooling: 'mean',
  cosStandardize: false,
  cosChat: false,
  cosAddBos: true,
  cosRunA: '',
  cosRunB: '',
  cosGroupsJson: '',
  cosRes: null,
}

export async function renderDiag(p) {
  const st = S.diag
  if (!S.vectors) await refreshVectors()
  if (!st.cosGroupsJson) {
    try {
      const d = await GET('/api/ext/cosine_diag/defaults')
      if (d?.groups) st.cosGroupsJson = JSON.stringify(d.groups, null, 2)
    } catch {}
  }
  p.innerHTML = ''

  const topBar = el('div', { className: 'row', style: 'justify-content:space-between;margin-bottom:8px' })
  const modeBtns = el('div', { className: 'row' })
  modeBtns.append(
    btn('1. Residual Outliers & BOS Sink Scanner', () => { st.view = 'outliers'; renderTab() }, st.view === 'outliers' ? 'primary mini' : 'mini ghost'),
    btn('2. Cosine Similarity Matrix & Null Baseline', () => { st.view = 'cosine'; renderTab() }, st.view === 'cosine' ? 'primary mini' : 'mini ghost'),
  )
  topBar.append(
    modeBtns,
    el('span', { className: 'small' }, `<span class="chip" style="background:rgba(16,185,129,0.2)">Diagnose View</span> Inspect activation geometry, shared outlier coordinates, and directional cosine similarity.`),
  )
  p.appendChild(topBar)

  if (st.view === 'outliers') {
    return renderOutliersView(p, st)
  }
  return renderCosineView(p, st)
}

function renderOutliersView(p, st) {
  const ctl = el('div', { className: 'kvbox' })
  ctl.appendChild(el('div', { className: 'small', style: 'margin-bottom:6px' },
    `<b>Residual Outlier & Massive Activation Scanner.</b> Detects coordinates where <code>|x_d| &gt; k * median(|x|)</code> (such as Gemma-4's <code>dim 1030</code>), identifies the <code>&lt;bos&gt;</code> attention sink token, measures the ~7x activation norm spike when <code>&lt;bos&gt;</code> is omitted in raw mode, and checks <code>|(x · u)| / ||x||</code> when projecting out a direction <code>u</code>.`))

  const row1 = el('div', { className: 'row' })
  const srcSel = selectOf([
    ['prompts', 'Prompt Batch (max_new_tokens=0)'],
    ['run', 'Active Finished Run'],
  ], st.outSource)
  srcSel.onchange = () => { st.outSource = srcSel.value; renderTab() }

  const hookSel = selectOf([
    ['resid_post', 'resid_post (after block)'],
    ['resid_pre', 'resid_pre (before block)'],
    ['mlp_out', 'mlp_out'],
    ['attn_out', 'attn_out'],
  ], st.outHook)
  hookSel.onchange = () => { st.outHook = hookSel.value }

  const kInp = numInput(st.outK, 60)
  kInp.oninput = () => { st.outK = +kInp.value || 10 }

  const vecOpts = [['', '- optional: check |(x·u)| / ||x|| -'], ...(S.vectors || []).map(v => [v.id, `${v.name} (norm ${v.norm})`])]
  const vecSel = selectOf(vecOpts, st.outVectorRef)
  vecSel.onchange = () => { st.outVectorRef = vecSel.value }

  row1.append(
    labelled('Source', srcSel),
    labelled('Hook', hookSel),
    labelled('Outlier threshold k×med', kInp),
    labelled('Direction u', vecSel),
  )

  if (st.outSource === 'prompts') {
    const cbChat = el('input', { type: 'checkbox', checked: st.outChat })
    cbChat.onchange = () => { st.outChat = cbChat.checked; renderTab() }
    const cbBos = el('input', { type: 'checkbox', checked: st.outAddBos })
    cbBos.onchange = () => { st.outAddBos = cbBos.checked }
    const cbComp = el('input', { type: 'checkbox', checked: st.outCompareNoBos })
    cbComp.onchange = () => { st.outCompareNoBos = cbComp.checked }

    const lChat = el('label', { className: 'inl' }); lChat.append(cbChat, el('span', { textContent: 'Use chat template' }))
    const lBos = el('label', { className: 'inl' }); lBos.append(cbBos, el('span', { textContent: 'Add <bos> token' }))
    const lComp = el('label', { className: 'inl' }); lComp.append(cbComp, el('span', { textContent: 'Compare with vs without <bos>' }))
    row1.append(lChat, lBos, lComp)
  }

  const runBtn = btn('▶ Scan Residual Outliers', async () => {
    const body = {
      run_id: st.outSource === 'run' ? S.run?.id : null,
      prompts: st.outSource === 'prompts' ? st.outPrompts.split(/\r?\n/).map(s => s.trim()).filter(Boolean) : null,
      hook: st.outHook,
      k: +st.outK || 10,
      chat: st.outChat,
      add_bos: st.outAddBos,
      compare_no_bos: st.outCompareNoBos && st.outSource === 'prompts' && !st.outChat,
      vector_ref: st.outVectorRef || null,
    }
    if (st.outSource === 'run' && !body.run_id) return toast('Run the flow first, or switch Source to Prompt Batch.', 'error')
    resBox.innerHTML = `<div class="empty">${spinner('scanning residual stream across all blocks…')}</div>`
    try {
      st.outRes = await POST('/api/ext/cosine_diag/outliers', body)
      renderTab()
    } catch (e) { toast(e.message, 'error'); resBox.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
  }, 'primary')
  row1.appendChild(runBtn)
  ctl.appendChild(row1)

  if (st.outSource === 'prompts') {
    const ta = el('textarea', { rows: 3, value: st.outPrompts, style: 'margin-top:6px;width:100%' })
    ta.oninput = () => { st.outPrompts = ta.value }
    ctl.appendChild(ta)
  }
  p.appendChild(ctl)

  const resBox = el('div')
  p.appendChild(resBox)
  if (!st.outRes) return

  const r = st.outRes
  const cards = el('div', { className: 'row', style: 'gap:10px;margin:8px 0' })
  const card = (title, val, sub = '') => el('div', { className: 'kvbox', style: 'flex:1;margin:0' },
    `<div class="small">${esc(title)}</div><div style="font-size:16px;font-weight:700;margin:2px 0">${val}</div><div class="small">${sub}</div>`)
  cards.append(
    card('BOS Sink Status', r.bos_present ? `<span class="chip ok">Present (${esc(r.bos_token || '<bos>')})</span>` : `<span class="chip bad">Missing BOS</span>`,
      r.mean_no_bos_inflation != null ? `Omitting BOS inflates norms <b>${r.mean_no_bos_inflation}x</b> on average` : 'Token 0 acts as attention sink'),
    card('Peak Outlier Coordinate', `dim ${r.outlier_dim} @ Block #${r.worst_block}`, `<b>${r.max_med_ratio}x</b> median coordinate magnitude`),
    card('Threshold', `|x_d| > ${r.threshold_k} × median`, `Hook: <code>${esc(r.hook)}</code>`),
  )
  resBox.appendChild(cards)

  const tbl = el('table', { className: 'grid' })
  const hasNoBos = r.layers.some(l => l.no_bos_inflation != null)
  const hasProj = r.layers.some(l => l.proj_ratio_mean != null)
  tbl.innerHTML = `<thead><tr>
    <th>Block</th>
    <th>Token ||x|| med</th>
    <th>Token ||x|| max (sink)</th>
    ${hasNoBos ? '<th>No-BOS ||x|| med (Inflation)</th>' : ''}
    <th>Outlier Dims (&gt;${r.threshold_k}×med)</th>
    <th>Top Outlier Dim</th>
    <th>Max / Med Ratio</th>
    <th>Top-1 Dim Variance %</th>
    ${hasProj ? '<th>|(x·u)| / ||x|| (mean / max)</th>' : ''}
  </tr></thead><tbody></tbody>`
  const tb = tbl.querySelector('tbody')
  for (const l of r.layers) {
    const tr = el('tr')
    tr.style.cursor = 'pointer'
    tr.onclick = () => {
      const b = flow.nodes.find(n => n.data?.kind === 'block' && +n.data.layer === +l.layer)
      if (b) flow.select({ type: 'node', id: b.id })
    }
    tr.innerHTML = `
      <td><b>#${l.pos}</b> · L${l.layer}</td>
      <td>${l.tok_norm_med}</td>
      <td>${l.tok_norm_max} <span class="chip">#${l.argmax_tok_idx} "${esc(l.argmax_tok_piece)}"</span></td>
      ${hasNoBos ? `<td>${l.no_bos_tok_norm_med ?? '-'} ${l.no_bos_inflation ? `<span class="chip ${l.no_bos_inflation > 2 ? 'bad' : ''}">${l.no_bos_inflation}x</span>` : ''}</td>` : ''}
      <td><b>${l.n_outliers}</b></td>
      <td><code>dim ${l.top_outlier_dim}</code> (${l.top_outlier_val})</td>
      <td><b>${l.max_med_ratio}×</b></td>
      <td>${l.var_top1_pct}%</td>
      ${hasProj ? `<td>${(l.proj_ratio_mean * 100).toFixed(2)}% / ${(l.proj_ratio_max * 100).toFixed(2)}%</td>` : ''}
    `
    tb.appendChild(tr)
  }
  resBox.appendChild(tbl)
}

function renderCosineView(p, st) {
  const ctl = el('div', { className: 'kvbox' })
  ctl.appendChild(el('div', { className: 'small', style: 'margin-bottom:6px' },
    `<b>Cosine Similarity Matrix & Subspace Geometry.</b> Compare directional alignment across contrastive group directions (with a <b>shuffled-label null baseline</b>), saved steering vectors, or layer-by-layer residual states. Enable <b>per-dimension z-score standardization</b> to see whether high cosine similarity is genuine or an artifact of shared outlier coordinates.`))

  const nL = S.info?.n_layers || 24
  const mid = st.cosPos >= 0 && st.cosPos < nL ? st.cosPos : Math.floor(nL / 2)

  const row1 = el('div', { className: 'row' })
  const mSel = selectOf([
    ['groups', 'Contrastive Group Directions (G × G + Shuffled Null)'],
    ['vectors', 'Saved Steering Vectors in Library (N × N)'],
    ['layers', 'Cross-Layer Residual Trajectory of a Run (L × L)'],
  ], st.cosMode)
  mSel.onchange = () => { st.cosMode = mSel.value; renderTab() }

  const pSel = selectOf([['mean', 'mean over prompt tokens'], ['last', 'last prompt token']], st.cosPooling)
  pSel.onchange = () => { st.cosPooling = pSel.value }

  const posInp = numInput(mid, 58, { min: 0, max: Math.max(0, nL - 1) })
  posInp.oninput = () => { st.cosPos = +posInp.value }

  const cbStd = el('input', { type: 'checkbox', checked: st.cosStandardize })
  cbStd.onchange = () => { st.cosStandardize = cbStd.checked }
  const lStd = el('label', { className: 'inl' })
  lStd.append(cbStd, el('span', { textContent: 'Per-dim z-score standardization (suppresses outlier dims)' }))

  row1.append(labelled('Comparison Mode', mSel))
  if (st.cosMode === 'groups') {
    row1.append(labelled('Block #', posInp), labelled('Token Pooling', pSel))
  } else if (st.cosMode === 'layers') {
    const runs = runsWithCaps().map(r => [r.id, runLabel(r)])
    if (!st.cosRunA && S.run?.id) st.cosRunA = S.run.id
    const rASel = selectOf([['', '- active run -'], ...runs], st.cosRunA)
    rASel.onchange = () => { st.cosRunA = rASel.value }
    const rBSel = selectOf([['', '- same as Run A (self-similarity) -'], ...runs], st.cosRunB)
    rBSel.onchange = () => { st.cosRunB = rBSel.value }
    row1.append(labelled('Run A', rASel), labelled('Run B', rBSel), labelled('Pooling', pSel))
  }
  row1.append(lStd)

  const calcBtn = btn('▶ Compute Cosine Matrix', async () => {
    let groups = null
    if (st.cosMode === 'groups') {
      try { groups = JSON.parse(st.cosGroupsJson) }
      catch (e) { return toast('Invalid groups JSON: ' + e.message, 'error') }
    }
    const body = {
      mode: st.cosMode,
      pos: st.cosPos >= 0 ? st.cosPos : mid,
      pooling: st.cosPooling,
      standardize: st.cosStandardize,
      chat: st.cosChat,
      add_bos: st.cosAddBos,
      run_a: st.cosMode === 'layers' ? (st.cosRunA || S.run?.id || null) : null,
      run_b: st.cosMode === 'layers' ? (st.cosRunB || null) : null,
      groups,
    }
    resBox.innerHTML = `<div class="empty">${spinner('computing cosine similarity matrix…')}</div>`
    try {
      st.cosRes = await POST('/api/ext/cosine_diag/cosine', body)
      renderTab()
    } catch (e) { toast(e.message, 'error'); resBox.innerHTML = `<div class="empty">${esc(e.message)}</div>` }
  }, 'primary')
  row1.appendChild(calcBtn)
  ctl.appendChild(row1)

  if (st.cosMode === 'groups') {
    const det = el('details', { style: 'margin-top:6px' }, `<summary class="small" style="cursor:pointer">Edit paired contrastive groups JSON (e.g. EN / DE / FR moral pairs)</summary>`)
    const ta = el('textarea', { rows: 6, value: st.cosGroupsJson, style: 'width:100%;font-family:monospace;font-size:11px;margin-top:4px' })
    ta.oninput = () => { st.cosGroupsJson = ta.value }
    det.appendChild(ta)
    ctl.appendChild(det)
  }
  p.appendChild(ctl)

  const resBox = el('div')
  p.appendChild(resBox)
  if (!st.cosRes) return

  const r = st.cosRes
  const summaryRow = el('div', { className: 'row', style: 'gap:10px;margin:8px 0' })
  summaryRow.appendChild(el('div', { className: 'kvbox', style: 'margin:0' },
    `<div class="small">Mean Off-Diagonal Cosine</div><div style="font-size:18px;font-weight:700">${r.mean_off_diag}</div>
     <div class="small">${r.standardize ? 'Z-score standardized per coordinate' : 'Raw residual coordinates'}</div>`))
  if (r.null) {
    summaryRow.appendChild(el('div', { className: 'kvbox', style: 'margin:0' },
      `<div class="small">Shuffled-Label Null Baseline (${r.null.n_samples} draws)</div>
       <div style="font-size:16px;font-weight:700">Null |cos|: ${r.null.null_mean} ± ${r.null.null_std} (max ${r.null.null_max})</div>
       <div class="small">Separation vs null: <b>z = ${r.null.z_score}σ</b> ${r.null.z_score > 2.5 ? '<span class="chip ok">Significant</span>' : '<span class="chip">Within noise</span>'}</div>`))
  }
  resBox.appendChild(summaryRow)

  const tbl = el('table', { className: 'grid' })
  const labels = r.labels || []
  let h = '<thead><tr><th></th>' + labels.map(l => `<th>${esc(l)}</th>`).join('') + '</tr></thead><tbody>'
  r.matrix.forEach((row, i) => {
    h += `<tr><th>${esc(labels[i])}</th>`
    row.forEach(v => {
      const c = v >= 0 ? `rgba(52,211,153,${Math.min(1, Math.abs(v)) * 0.65})` : `rgba(248,113,113,${Math.min(1, Math.abs(v)) * 0.65})`
      h += `<td style="background:${c};text-align:right;font-family:monospace">${v.toFixed(3)}</td>`
    })
    h += '</tr>'
  })
  tbl.innerHTML = h + '</tbody>'
  resBox.appendChild(tbl)
}

registerTab({ id: 'diag', title: 'Cosine & Outliers', render: renderDiag })
