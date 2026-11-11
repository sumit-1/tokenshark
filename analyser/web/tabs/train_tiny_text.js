// Train tab section: a small transformer trained from scratch on text.
import { $, GET, POST, btn, el, toast } from '../core.js'
import { labelled, selectOf } from '../ui/labkit.js'
import { renderTab } from '../ui/tabs.js'
import { jobView, registerTrainSection } from './train.js'
import { refreshModels } from '../ui/models.js'

function tinyTextSection(st, running) {
  const f = st.form
  const a = el('div')
  a.appendChild(el('div', { className: 'small' }, `<b>1 · Train a small transformer from scratch.</b> A few hundred thousand weights start as random numbers. Each step the model guesses the next token of some text,
    is told how wrong it was (the loss), and every weight moves a little to be less wrong. Watch the loss fall. The result is saved as a normal model: load it and use every tab of this app on a model you trained.`))
  const g = el('div', { className: 'formgrid' })
  const inp = (key, label, w = 70, type = 'number') => { const i = el('input', { type, value: f[key], step: 'any', style: `width:${w}px` }); i.oninput = () => { f[key] = type === 'number' ? +i.value : i.value }; return labelled(label, i) }
  const corpus = selectOf([['mixed', 'built-in: facts + counting'], ['facts', 'built-in: capitals and animal sounds'], ['counting', 'built-in: counting and sums'], ['own', 'my own text (paste below)']], f.corpus)
  corpus.onchange = () => { f.corpus = corpus.value; renderTab() }
  g.append(inp('name', 'Name', 150, 'text'), labelled('Text to learn', corpus), inp('d_model', 'Width d_model'), inp('n_layers', 'Blocks'), inp('n_heads', 'Heads'), inp('context', 'Context (tokens)'),
    inp('vocab', 'Vocabulary'), inp('steps', 'Steps', 80), inp('batch', 'Batch'), inp('lr', 'Learning rate', 80), inp('sample_prompt', 'Test prompt', 190, 'text'))
  a.appendChild(g)
  if (f.corpus === 'own') { const ta = el('textarea', { rows: 4, placeholder: 'Paste a few thousand characters of text or more…', value: f.text }); ta.oninput = () => { f.text = ta.value }; a.appendChild(ta) }
  else { const pv = el('pre', { className: 'codehelp' }); a.appendChild(pv); GET(`/api/train/corpus/preview?corpus=${f.corpus}`).then(r => { pv.textContent = r.sample.trim() + '\n…' }).catch(() => {}) }
  const arow = el('div', { className: 'row' })
  arow.append(btn(running(st.tiny) ? 'Training…' : 'Start training', async () => {
    try { await POST('/api/train/tiny', { ...f, text: f.corpus === 'own' ? f.text : '', corpus: f.corpus === 'own' ? 'mixed' : f.corpus }); renderTab() } catch (e) { toast(e.message, 'error') }
  }, 'primary'))
  arow.lastChild.disabled = running(st.tiny)
  if (running(st.tiny)) arow.append(btn('■ Stop (keeps what it learned)', async () => { await POST('/api/train/tiny/stop'); renderTab() }))
  a.appendChild(arow)
  if (st.tiny && st.tiny.status !== 'none') a.appendChild(jobView(st.tiny, j => {
    if (!j.result?.model_id) return null
    return btn(`Load “${j.result.model_id}” into the app`, async () => {
      await refreshModels(); $('#model').value = j.result.model_id; $('#policy').value = 'auto'; $('#load').click()
    }, 'primary')
  }))
  return a
}

registerTrainSection('tiny_text', { heading: 'Language models — learn from text', render: tinyTextSection })
