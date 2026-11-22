// Guardrail Fork tab: train and inspect center-layer side-branch classifiers (e.g. PowerShell vs Python detector).
import { GET, POST, S, btn, el, flow, registerTab, toast } from '../core.js'
import { esc } from '../flow.js'
import { addForkHead } from '../ui/flowedit.js'
import { labelled, selectOf } from '../ui/labkit.js'
import { renderInspector } from '../ui/inspector.js'
import { applyRunBadges } from '../ui/run.js'
import { renderTab } from '../ui/tabs.js'

S.gf = {
  preset: 'powershell_detector',
  name: 'powershell_detector',
  pos: -1,
  hook: 'resid_post',
  head_type: 'linear_probe',
  steps: 80,
  lr: 0.05,
  threshold: 0.65,
  trigger_category: 'PowerShell',
  halt_message: '[Guardrail Fork: PowerShell command detected at center layer - generation halted]',
  cat_a_name: 'PowerShell',
  cat_a_text: [
    'Get-Process | Where-Object { $_.CPU -gt 10 } | Sort-Object CPU -Descending',
    'Get-ChildItem -Path C:\\Logs -Recurse -Filter *.log | Select-String -Pattern "ERROR"',
    'Invoke-RestMethod -Uri "https://api.example.com/status" -Method Get',
    '$services = Get-Service | Where-Object {$_.Status -eq "Running"}',
    'New-Item -ItemType Directory -Force -Path $env:TEMP\\backup',
    'Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser',
  ].join('\n'),
  cat_b_name: 'Python / General',
  cat_b_text: [
    'import os\nfor root, dirs, files in os.walk("/var/log"): print(len(files))',
    'def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    return a',
    'Can you explain how transformer self-attention computes query and key dot products?',
    'import requests\nresp = requests.get("https://api.example.com/status")',
    'heights = [x ** 2 for x in range(10) if x % 2 == 0]',
    'Please summarize the main differences between TCP and UDP transport protocols.',
  ].join('\n'),
  eval_text: 'Get-Process | Where-Object { $_.CPU -gt 10 }',
  eval_res: null,
  last_trained: null,
  heads: [],
}

export async function renderGuardrail(p) {
  const st = S.gf
  try {
    const d = await GET('/api/ext/guardrail_fork/presets')
    st.heads = d.heads || []
    st.presets = d.presets || {}
  } catch {}
  p.innerHTML = ''

  const nL = S.info?.n_layers || 24
  const midPos = st.pos >= 0 ? st.pos : Math.floor(nL / 2)

  const headBar = el('div', { className: 'row', style: 'justify-content:space-between;margin-bottom:8px' })
  headBar.append(
    el('div', { className: 'small' }, `<b>Center-Layer Guardrail Fork.</b> Fork the residual stream at a middle block (e.g. Block #${midPos}) into a lightweight side-branch classifier (<code>linear_probe</code>, <code>mlp_guard</code>, or <code>block_branch</code>). Run it in <b>monitor</b> mode (live category probabilities) or <b>gate</b> mode (early-stops generation at that block when triggered).`),
    btn('+ Add Fork Head to Flow', () => {
      const blk = flow.nodes.find(n => n.data?.kind === 'block' && n.data.layer === midPos) || flow.nodes.find(n => n.data?.kind === 'block')
      addForkHead(blk, st.name || 'powershell_detector', 'monitor', st.threshold)
      toast(`Attached Fork Head (${st.name}) to ${blk ? blk.data.label : 'flow'}`)
    }, 'primary'),
  )
  p.appendChild(headBar)

  // Live run fork telemetry if current run has fork outputs
  const runForks = S.run?.outputs?.forks
  if (runForks && Object.keys(runForks).length) {
    const liveBox = el('div', { className: 'sec', style: 'margin-bottom:10px' })
    liveBox.appendChild(el('h4', { textContent: 'Active Run Fork Telemetry' }))
    for (const [fid, f] of Object.entries(runForks)) {
      const trig = f.triggered
      const card = el('div', { className: 'kvbox', style: 'margin-top:6px' })
      const scoresHtml = Object.entries(f.scores || {}).map(([k, v]) =>
        `<span class="chip" style="background:${k === f.trigger_category && trig ? 'rgba(248,113,113,0.3)' : 'rgba(52,211,153,0.2)'}">${esc(k)}: <b>${(v * 100).toFixed(1)}%</b></span>`
      ).join(' ')
      card.innerHTML = `<div class="row"><b>${esc(fid)}</b> <span class="small">Block #${f.pos} · hook <code>${esc(f.hook)}</code> · mode <code>${esc(f.mode)}</code> · head <code>${esc(f.ref)}</code></span>
        <span class="chip ${trig ? 'bad' : 'ok'}">${trig ? 'TRIGGERED' : 'PASS'} (threshold ${(f.threshold * 100).toFixed(0)}%)</span></div>
        <div class="row" style="margin-top:4px">${scoresHtml}</div>`
      liveBox.appendChild(card)
    }
    p.appendChild(liveBox)
  }

  // Training & Preset configuration
  const box = el('div', { className: 'kvbox' })
  const presetOpts = [
    ['powershell_detector', 'PowerShell vs Python / General Detector (Clean Preset)'],
    ['topic_router', 'Math & Code vs Creative Writing Router'],
    ['custom', 'Custom Categories'],
  ]
  const psel = selectOf(presetOpts, st.preset)
  psel.onchange = () => {
    st.preset = psel.value
    const pr = st.presets?.[st.preset]
    if (pr) {
      const cats = Object.entries(pr.categories)
      st.name = pr.id
      st.head_type = pr.head_type || 'linear_probe'
      st.threshold = pr.threshold ?? 0.65
      st.trigger_category = pr.trigger_category || cats[0][0]
      st.halt_message = pr.halt_message || ''
      if (cats[0]) { st.cat_a_name = cats[0][0]; st.cat_a_text = cats[0][1].join('\n') }
      if (cats[1]) { st.cat_b_name = cats[1][0]; st.cat_b_text = cats[1][1].join('\n') }
    }
    renderTab()
  }

  const g = el('div', { className: 'formgrid' })
  const inp = (key, label, w = 90, type = 'number') => {
    const i = el('input', { type, value: st[key] === -1 && key === 'pos' ? midPos : st[key], step: 'any', style: `width:${w}px` })
    i.oninput = () => { st[key] = type === 'number' ? +i.value : i.value }
    return labelled(label, i)
  }
  const hsel = selectOf([
    ['linear_probe', 'Linear Probe (W·x + b)'],
    ['mlp_guard', '2-Layer MLP Guard (SiLU)'],
    ['block_branch', 'Mini SwiGLU Block Branch'],
  ], st.head_type)
  hsel.onchange = () => { st.head_type = hsel.value }

  g.append(
    labelled('Preset', psel),
    inp('name', 'Head ID', 150, 'text'),
    inp('pos', `Center Block (0..${nL - 1})`, 70, 'number'),
    labelled('Architecture', hsel),
    inp('steps', 'Steps', 65, 'number'),
    inp('lr', 'LR', 65, 'number'),
    inp('threshold', 'Gate Threshold', 70, 'number'),
    inp('trigger_category', 'Trigger Category', 130, 'text'),
  )
  box.appendChild(g)

  const cols = el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px' })
  const colA = el('div')
  const nameA = el('input', { type: 'text', value: st.cat_a_name, style: 'width:180px;margin-bottom:4px' })
  nameA.oninput = () => { st.cat_a_name = nameA.value; st.trigger_category = nameA.value }
  const taA = el('textarea', { rows: 6, value: st.cat_a_text })
  taA.oninput = () => { st.cat_a_text = taA.value }
  colA.append(labelled('Category A (Guarded / Target, one example per line)', nameA), taA)

  const colB = el('div')
  const nameB = el('input', { type: 'text', value: st.cat_b_name, style: 'width:180px;margin-bottom:4px' })
  nameB.oninput = () => { st.cat_b_name = nameB.value }
  const taB = el('textarea', { rows: 6, value: st.cat_b_text })
  taB.oninput = () => { st.cat_b_text = taB.value }
  colB.append(labelled('Category B (Baseline / General, one example per line)', nameB), taB)

  cols.append(colA, colB)
  box.appendChild(cols)

  const trow = el('div', { className: 'row', style: 'margin-top:8px' })
  const trainBtn = btn('Train Center-Layer Fork Head', async () => {
    trainBtn.disabled = true
    trainBtn.textContent = 'Training on frozen activations…'
    try {
      const cats = {
        [st.cat_a_name.trim() || 'Category A']: st.cat_a_text.split('\n').map(x => x.trim()).filter(Boolean),
        [st.cat_b_name.trim() || 'Category B']: st.cat_b_text.split('\n').map(x => x.trim()).filter(Boolean),
      }
      const res = await POST('/api/ext/guardrail_fork/train', {
        categories: cats,
        pos: st.pos >= 0 ? st.pos : midPos,
        hook: st.hook,
        head_type: st.head_type,
        steps: st.steps,
        lr: st.lr,
        name: st.name,
        trigger_category: st.trigger_category || st.cat_a_name,
        threshold: st.threshold,
        halt_message: st.halt_message,
      })
      st.last_trained = res
      toast(`Trained '${res.id}' at Block #${res.pos} (accuracy ${(res.accuracy * 100).toFixed(1)}%)`)
      renderTab()
    } catch (e) {
      toast(e.message, 'error')
      trainBtn.disabled = false
      trainBtn.textContent = 'Train Center-Layer Fork Head'
    }
  }, 'primary')

  const wireMonitorBtn = btn('Wire to Center Block (Monitor)', () => {
    const pIdx = st.pos >= 0 ? st.pos : midPos
    const blk = flow.nodes.find(n => n.data?.kind === 'block' && n.data.layer === pIdx) || flow.nodes.find(n => n.data?.kind === 'block')
    addForkHead(blk, st.name, 'monitor', st.threshold)
    applyRunBadges(); renderInspector()
  })
  const wireGateBtn = btn('Wire to Center Block (Gate / Halt)', () => {
    const pIdx = st.pos >= 0 ? st.pos : midPos
    const blk = flow.nodes.find(n => n.data?.kind === 'block' && n.data.layer === pIdx) || flow.nodes.find(n => n.data?.kind === 'block')
    addForkHead(blk, st.name, 'gate', st.threshold)
    applyRunBadges(); renderInspector()
  })
  trow.append(trainBtn, wireMonitorBtn, wireGateBtn)
  box.appendChild(trow)

  if (st.last_trained) {
    const lt = st.last_trained
    box.appendChild(el('div', { className: 'small', style: 'margin-top:6px' },
      `Trained <b>${esc(lt.id)}</b> (${esc(lt.head_type)}) at Block #${lt.pos} (L${lt.layer}) · accuracy <b>${(lt.accuracy * 100).toFixed(1)}%</b> · final loss <b>${lt.final_loss}</b> · categories: ${lt.categories.map(esc).join(' vs ')}`))
  }

  // Quick interactive test probe
  const evalRow = el('div', { className: 'row', style: 'margin-top:10px' })
  const evalInp = el('input', { type: 'text', value: st.eval_text, style: 'flex:1', placeholder: 'Type a command or prompt to test against the trained fork head…' })
  evalInp.oninput = () => { st.eval_text = evalInp.value }
  evalRow.append(
    el('span', { className: 'small', textContent: 'Quick probe:' }),
    evalInp,
    btn('Test Probe', async () => {
      try {
        st.eval_res = await POST('/api/ext/guardrail_fork/evaluate', {
          ref: st.name,
          text: st.eval_text,
          pos: st.pos >= 0 ? st.pos : midPos,
        })
        renderTab()
      } catch (e) { toast(e.message, 'error') }
    }),
  )
  box.appendChild(evalRow)
  if (st.eval_res) {
    const er = st.eval_res
    const sc = Object.entries(er.scores || {}).map(([k, v]) => `${esc(k)}: <b>${(v * 100).toFixed(1)}%</b>`).join(' · ')
    box.appendChild(el('div', { className: 'small', style: 'margin-top:4px' },
      `Result: <span class="chip ${er.triggered ? 'bad' : 'ok'}">${er.triggered ? 'TRIGGERED' : 'PASS'}</span> ${sc}`))
  }

  p.appendChild(box)
}

registerTab({ id: 'guardrail', title: 'Guardrail Fork', hint: 'Center-layer side-branch classifier & runtime guardrail gate', render: renderGuardrail })
