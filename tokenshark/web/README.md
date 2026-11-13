# Web UI architecture

Plain ES modules, served as static files. No build step.

```
core.js          API client, shared state S, DOM helpers, the flow canvas, the tab registry, the event bus.
                 Imports no feature module, so anything can import it.
app.js           Entry point: imports every module (each wires itself up when imported) and boots the page.
flow.js          The node editor (boxes, ports, arrows).            paint.js   The paint dialog.
ui/              The frame around the tabs
  models.js        model picker, load / unload, status            flowedit.js  adding boxes, the palette, validation
  inspector.js     the right-hand panel for the selected box      run.js       run / pause / cancel, live progress
  debugger.js      step buttons and stepping session              menu.js      right-click menu
  tabs.js          the tab row and the panel (knows no tab)       tasks.js     the task list
  analysis.js      background analyses                            labkit.js    small shared widgets
  codeedit.js      the box code editor
tabs/            One module per results tab. Each ends with registerTab({...}).
```

## The rule: the page has no per-model conditions

What a model shows is declared by its **plugin** (`tokenshark/plugins/*.py`, `ui_schema`), and the page builds
itself from that:

| `ui_schema` key | Meaning | Where the ids come from |
|---|---|---|
| `tabs` | results tabs, in order | `registerTab({ id })` in `tabs/*.js` |
| `palette_actions` | flow-editing buttons | `data-action="…"` in `index.html` |
| `train` | sections of the Train tab | `sections` in `tabs/train.js` |
| `course` | guided course on the Learn tab | `COURSES` in `tabs/learn.js` |
| `step_debugger` | whether the header's step buttons apply | — |

`tests/test_architecture.py` fails if a plugin names a tab, action or section that does not exist.

Two modules may stand behind one tab name: `output_text` and `output_image` both register `alias: 'output'`,
so `setTab('output')` opens whichever one the loaded model's plugin lists.

## Add a tab

1. Create `tabs/mytab.js`:

   ```js
   import { S, el, registerTab } from '../core.js'

   function renderMyTab(panel) {
     panel.appendChild(el('div', { className: 'small', textContent: `Run ${S.run.id}` }))
   }

   registerTab({ id: 'mytab', title: 'My tab', needs: 'run', render: renderMyTab })
   ```

   `needs: 'run'` shows “Run the flow to see results here” until there is a run. Optional: `hint` (tooltip),
   `alias`, `onOpen()`.
2. Import it in `app.js` (`import './tabs/mytab.js'`).
3. Add `"mytab"` to `ui_schema["tabs"]` of every plugin whose models it works for.

If the tab needs data from the server, add a route (see `tokenshark/labapi.py`) and guard it with
`_need("<feature>", "My tool")`; list that feature in the plugins that support it.

## Talk through events, not imports

Modules announce changes; others listen. This keeps `ui/models.js` from knowing which tabs exist.

```js
import { on, emit } from '../core.js'
on('model', info => { /* a model was loaded (info) or unloaded (null) */ })
```

Callbacks the flow canvas needs (selection, right-click, edits) are registered in `core.hooks` by the module
that owns them.

## Add a model family

Write a plugin (`tokenshark/plugins/_template.py`): an adapter, `default_flow(info)`, `features` (which backend
tools the adapter supports) and `ui_schema` (which existing tabs, palette actions and train sections fit). A
new architecture that fits existing tabs needs no change to the page at all. A new kind of view is a new tab
module plus one line in the plugin.
