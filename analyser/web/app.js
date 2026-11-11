// Entry point: loads every UI module (each wires its own buttons) and boots the page.
import './tabs/attn.js'
import './tabs/cloud.js'
import './tabs/contrib.js'
import './tabs/debug_image.js'
import './tabs/debug_text.js'
import './tabs/ilens.js'
import './tabs/kv.js'
import './tabs/learn.js'
import './tabs/lens.js'
import './tabs/lora.js'
import './tabs/neurons.js'
import './tabs/output_image.js'
import './tabs/output_text.js'
import './tabs/patch.js'
import './tabs/quant.js'
import './tabs/runs.js'
import './tabs/sampler.js'
import './tabs/steer.js'
import './tabs/tokenizer.js'
import './tabs/train.js'
import './tabs/train_adapt.js'
import './tabs/train_image.js'
import './tabs/train_tiny_text.js'
import './ui/analysis.js'
import './ui/codeedit.js'
import './ui/debugger.js'
import './ui/flowedit.js'
import './ui/labkit.js'
import './ui/menu.js'
import './ui/run.js'
import './ui/tabs.js'
import './ui/tasks.js'
import { $, GET, S, emit } from './core.js'
import { renderInspector } from './ui/inspector.js'
import { pollStatus, refreshModels } from './ui/models.js'

;
(async () => {
  await refreshModels()
  await pollStatus()
  setInterval(async () => {          // catches a server that keeps running old code after an update
    try { const st = await GET('/api/status'); $('#stale').style.display = st.stale ? '' : 'none' } catch { }
  }, 15000)
  if (!S.info) emit('model', null)          // nothing loaded: build the tab row for the start screen
  renderInspector()
})()
