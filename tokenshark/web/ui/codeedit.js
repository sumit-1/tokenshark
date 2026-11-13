import { GET, POST, btn, can, el, flow, toast } from '../core.js'
import { esc } from '../flow.js'
import { scheduleValidate } from './flowedit.js'
import { renderInspector } from './inspector.js'

export async function editCode(n) {
  const k = n.data.kind
  let d
  try { d = await GET(`/api/code/default?kind=${k}&layer=${n.data.layer ?? 0}`) } catch (e) { return toast(e.message, 'error') }
  if (!d.editable) return toast(d.reason, 'error')
  document.querySelector('#dlg-code')?.remove()
  const dlg = el('dialog', { id: 'dlg-code', className: 'code' })
  const edited = (n.data.code || '').trim() !== ''
  dlg.innerHTML = `<h3>${esc(n.data.label || k)} — code <span class="small">${edited ? '(edited)' : '(built-in)'}</span></h3>
    <div class="small">This is what the box computes, in plain Python. Change it and the next run executes your lines instead.
    Leave the result in <code>${{ embed: 'x', block: 'x', final_norm: 'x', lm_head: 'logits', sampler: 'token' }[k]}</code>.
    <code>print(...)</code> shows up in the Output tab. No imports; tensors are torch tensors.
    <a href="#" id="code-help-t">What can I use?</a></div><pre id="code-help" class="codehelp" style="display:none"></pre>`
  const ta = el('textarea', { spellcheck: false, value: edited ? n.data.code : d.code, className: 'codeta' })
  ta.rows = Math.min(30, Math.max(8, ta.value.split('\n').length + 2))
  ta.addEventListener('keydown', ev => {
    if (ev.key !== 'Tab') return
    ev.preventDefault()
    const s = ta.selectionStart
    ta.setRangeText('    ', s, ta.selectionEnd, 'end')
  })
  const msg = el('div', { className: 'small' })
  const row = el('div', { className: 'row' })
  const save = async () => {
    const src = ta.value
    const same = src.trim() === d.code.trim()
    if (!same) {
      const r = await POST('/api/code/check', { code: src, kind: k })
      if (!r.ok) { msg.innerHTML = `<span class="bad">${esc(r.problems.join(' · '))}</span>`; return }
    }
    flow.updateNodeData(n.id, { code: same ? '' : src })
    dlg.close(); dlg.remove(); renderInspector(); scheduleValidate()
    toast(same ? 'Back to the built-in code.' : 'Code saved for this box. Run the flow to see what it does.')
  }
  row.append(btn('Save', save, 'primary'), btn('Reset to built-in', () => { ta.value = d.code; msg.textContent = 'Built-in code restored (Save to apply).' }),
    btn('Cancel', () => { dlg.close(); dlg.remove() }))
  dlg.append(ta, msg, row)
  document.body.appendChild(dlg)
  dlg.querySelector('#code-help-t').onclick = ev => {
    ev.preventDefault()
    const h = dlg.querySelector('#code-help')
    h.textContent = d.help.split('Names available in every box')[1]?.replace(/^\s*\n/, '').replace(/"""\s*$/, '') || d.help
    h.style.display = h.style.display === 'none' ? '' : 'none'
  }
  dlg.showModal()
}

export const hasCode = k => ['embed', 'block', 'final_norm', 'lm_head', 'sampler'].includes(k)

export function codeSection(n, box) {
  if (!can('code') || !hasCode(n.data.kind)) return
  const sec = el('div', { className: 'sec' })
  const edited = (n.data.code || '').trim() !== ''
  sec.innerHTML = `<h4>Code ${edited ? '<span class="chip coded">✎ edited</span>' : ''}</h4><div class="small">${edited
    ? 'This box runs your Python instead of the built-in step.' : 'See what this box computes as a few lines of Python — and change them.'}</div>`
  const row = el('div', { className: 'row' })
  row.append(btn(edited ? 'Edit code…' : 'View / edit code…', () => editCode(n)))
  if (edited) row.append(btn('Back to built-in', () => { flow.updateNodeData(n.id, { code: '' }); renderInspector(); scheduleValidate() }, 'ghost'))
  sec.appendChild(row)
  box.appendChild(sec)
}
