import { $, GET, S, btn, el, flow, isDiT, on, registerTab, schema, toast } from '../core.js'
import { esc } from '../flow.js'
import { editCode } from '../ui/codeedit.js'
import { debugStart } from '../ui/debugger.js'
import { addInitImage, addIntervention, addPromptKV, bypassBlock, resetFlow } from '../ui/flowedit.js'
import { renderInspector } from '../ui/inspector.js'
import { refreshModels } from '../ui/models.js'
import { startRun } from '../ui/run.js'
import { renderTab, setTab } from '../ui/tabs.js'
import { addLayerPrompt } from './kv.js'

// ------------------------------------------------------------------ helpers the lessons use
const node = kind => flow.nodes.find(n => n.data.kind === kind)

const set = (kind, data) => { const n = node(kind); if (!n) return; flow.updateNodeData(n.id, data); if (S.selected?.id === n.id) renderInspector() }

const select = ref => { const n = flow.node(ref) || node(ref); if (n) flow.select({ type: 'node', id: n.id }) }

const blocks = () => flow.nodes.filter(n => (n.data.kind === 'block' || n.data.kind === 'unet_block') && !/repeat/.test(n.data.label || ''))

const block = i => blocks().find(n => n.data.layer === i)

const nL = () => S.info?.n_layers || 1

const turbo = () => S.info?.default_guidance === 0

const hasRun = () => !!S.run?.outputs

const needRun = () => { if (hasRun()) return true; toast('Run the flow first (the step before this one).', 'error'); return false }

const run = label => { startRun({ label }); setTab('output') }

const tokenOf = word => (S.run?.outputs?.tokens || []).findIndex(t => t.toLowerCase().includes(word.toLowerCase()))

const removeKind = kind => flow.nodes.filter(n => n.data.kind === kind).forEach(n => flow.removeNode(n.id))

const FOX = 'a red fox sitting in a snowy forest, photo'

// a picture that is quick enough to iterate on
const quick = () => {
  set('input_prompt', { prompt: FOX, negative_prompt: '', switch_prompt: '' })
  if (isDiT()) { set('noise', { width: 512, height: 512 }); set('scheduler', { steps: 12 }) }
  else if (!turbo()) set('scheduler', { steps: 15 })
}

const wire = (boxId, targets) => targets.forEach(t => {
  if (!flow.edges.some(e => e.source === boxId && e.target === t.id)) flow.addEdge({ source: boxId, sourceHandle: 'out', target: t.id, targetHandle: 'mod' })
})

// ------------------------------------------------------------------ the image course
const IMAGE = [
  { id: 'img-tour', title: 'How an image model is built', time: '5 min',
    goal: 'Know what each box on the canvas does before you change anything.',
    steps: [
      { title: 'Words in', text: () => isDiT()
        ? 'Qwen-Image does not read your prompt with a small text encoder. It hands it to a full <b>language model</b> (Qwen3-VL, 36 blocks) and keeps what that model’s last block thinks about every token. Those vectors are the prompt as the image model sees it.'
        : 'The prompt is cut into tokens and read by a <b>CLIP text encoder</b>. Out comes one vector per token: that is all the image model ever knows about your words.',
        actions: [['Show me the text encoder', () => select('text_encoder')]] },
      { title: 'Noise in', text: 'A picture starts as <b>random noise</b>, drawn from the seed. Same seed, same noise, same picture. The noise is not in pixels: it lives in a smaller grid called the <b>latent</b>.' +
          ' <span class="k">Why a latent?</span> A 1024×1024 picture is 3 million numbers. The latent is tens of times smaller, so the expensive model has far less to do.',
        actions: [['Show me the noise box', () => select('noise')]] },
      { title: 'The denoiser, again and again', text: () => isDiT()
        ? `The <b>${nL()} blocks</b> are one transformer. The prompt tokens and the image tiles (one per 16×16 pixels) sit in <b>one sequence</b>, and every block lets each tile look at every word and every other tile. Each pass through all blocks predicts a small move from noise towards a picture. The loop arrow repeats that for every step.`
        : `The <b>UNet</b> stages shrink the latent (down), think (mid), and grow it back (up). The dotted arrows carry detail across. Every stage reads the prompt through <b>cross-attention</b>. One pass predicts the noise in the latent; the loop arrow repeats that for every step.`,
        actions: [['Show me a block', () => select(block(Math.floor(nL() / 2))?.id)], ['Show me the scheduler', () => select('scheduler')]] },
      { title: 'Pixels out', text: 'When the steps are done the <b>VAE decoder</b> turns the latent back into pixels. It runs once, at the end. Everything you see during the steps is the model’s <i>guess</i> at the final latent, not a finished picture.',
        actions: [['Show me the VAE', () => select('vae_decode')]] },
    ] },
  { id: 'img-first', title: 'From noise to picture', time: '10 min',
    goal: 'See a picture appear step by step, and what the seed decides.',
    steps: [
      { title: 'Set up a quick picture', text: () => `We use one prompt for the whole course and keep it small so each run is short${isDiT() ? ' (512×512, 12 steps — Qwen-Image is large, so even this takes a while on a CPU)' : ''}.`,
        actions: [['Set it up', () => { quick(); select('input_prompt'); toast('Prompt, size and steps are set.') }]] },
      { title: 'Run it and watch', text: 'Press Run. In the <b>Output</b> tab, look at the strip under the picture. <b>Top row:</b> what the model is given at each step — mostly noise at first. <b>Bottom row:</b> what it thinks the finished picture is (written x̂₀). ' +
          '<span class="k">Look for:</span> the layout and the main colours are decided in the first few steps; the late steps only add detail.',
        actions: [['▶ Run', () => run('lesson: first picture')]] },
      { title: 'Change only the seed', text: 'Now keep the prompt and change the seed. <span class="k">Predict first:</span> what will stay the same and what will change?' +
          ' The prompt says <i>what</i>; the noise decides <i>where</i>. That is why the same prompt gives a different composition every time.',
        actions: [['New seed and ▶ Run', () => { set('noise', { seed: (+node('noise').data.seed || 0) + 1 }); run('lesson: new seed') }]] },
    ] },
  { id: 'img-steps', title: 'Steps and the noise schedule', time: '10 min',
    goal: 'Understand what a step is, and why more steps stop helping.',
    steps: [
      { title: 'What a step does', text: () => isDiT()
        ? 'Qwen-Image is trained with <b>flow matching</b>. Mix a picture with noise: <code>x = (1−σ)·picture + σ·noise</code>. At σ = 1 it is all noise, at σ = 0 all picture. The model learns the direction from picture to noise. To draw, start at σ = 1 and take steps the other way: <code>x ← x + (σ_next − σ)·direction</code>. The <b>schedule</b> is just the list of σ values.'
        : 'At each step the model is told the noise level σ and predicts the noise in the latent. The sampler removes part of it and lowers σ. The <b>schedule</b> is the list of σ values, from high (rough layout) to low (fine detail).',
        actions: [['Show me the scheduler', () => select('scheduler')]] },
      { title: 'Very few steps', text: 'Take big strides: almost no steps. <span class="k">Look for:</span> the layout is already right, but surfaces are soft or broken.',
        actions: [['Few steps and ▶ Run', () => { quick(); set('scheduler', { steps: isDiT() ? 4 : turbo() ? 1 : 5 }); run('lesson: few steps') }]] },
      { title: 'More steps', text: 'Now more, smaller strides. Compare the two runs in the <b>Runs</b> tab (click a run to bring it back). After a point, extra steps cost time and change almost nothing.',
        actions: [['More steps and ▶ Run', () => { set('scheduler', { steps: isDiT() ? 20 : turbo() ? 4 : 25 }); run('lesson: more steps') }], ['Open Runs', () => setTab('runs')]] },
      { title: 'Branch from the middle', text: 'In the Output strip every step has <b>Branch here</b>. It keeps that step’s noisy latent as the start of your next run. Change the prompt, press Run, and only the remaining steps are computed. Branch early and the new prompt can still change the layout; branch late and it can only change details.',
        actions: [['Open Output', () => setTab('output')]] },
    ] },
  { id: 'img-words', title: 'How the picture reads the prompt', time: '10 min',
    goal: 'See which part of the picture listens to which word, then switch a word off.',
    steps: [
      { title: 'Where did “fox” land?', text: 'Every part of the picture looks at the prompt tokens and takes what it needs — that is <b>attention</b>. In the Output tab, click a word: the overlay lights up the parts of the picture that read it.',
        actions: [['Show “fox”', () => { if (!needRun()) return; const i = tokenOf('fox'); if (i < 0) return toast('This run’s prompt has no “fox”. Do “From noise to picture” first.', 'error'); S.xtok = i; setTab('output') }],
          ['Show “snow”', () => { if (!needRun()) return; const i = tokenOf('snow'); if (i >= 0) { S.xtok = i; setTab('output') } }]] },
      { title: 'Make the model deaf to one word', text: () => `An <b>intervention</b> box changes something while the model runs. This one uses the hook <code>word</code>: it scales how strongly the picture reads chosen prompt tokens. Zero = the word is in the prompt, but ${isDiT() ? 'the first half of the blocks' : 'no stage'} can hear it. <span class="k">Predict first:</span> will the fox vanish, or turn into something else?`,
        actions: [['Add it and ▶ Run', () => {
          if (!needRun()) return
          const i = tokenOf('fox'); if (i < 0) return toast('This run’s prompt has no “fox”.', 'error')
          removeKind('intervention')
          const targets = isDiT() ? blocks().filter(b => b.data.layer < nL() / 2) : blocks()
          addIntervention(targets[0])
          const id = S.selected.id
          flow.updateNodeData(id, { label: 'ignore “fox”', hook: 'word', op: 'zero', positions: String(i) })
          wire(id, targets); renderInspector(); run('lesson: ignore fox')
        }]] },
      { title: 'Now shout it', text: 'Same box, but scale by 2 instead of zero: the picture reads “fox” twice as strongly. Too much emphasis distorts the picture — every control here has a point where it breaks.',
        actions: [['Scale ×2 and ▶ Run', () => { const n = flow.nodes.find(x => x.data.kind === 'intervention'); if (!n) return toast('Do the step above first.', 'error'); flow.updateNodeData(n.id, { label: 'emphasise “fox”', op: 'scale', value: 2 }); renderInspector(); run('lesson: emphasise fox') }],
          ['Remove the box', () => removeKind('intervention')]] },
    ] },
  { id: 'img-guidance', title: 'Guidance and the negative prompt', time: '10 min', needs: () => turbo() ? 'This model is distilled to work without guidance. Load Stable Diffusion 1.5 or Qwen-Image for this lesson.' : '',
    goal: 'Understand the one number everyone turns up, and what it costs.',
    steps: [
      { title: 'The idea', text: 'Run the model twice per step: once with your prompt, once with the <b>negative prompt</b> (or nothing). The difference between the two answers is “what the prompt adds”. <b>Guidance</b> multiplies that difference: <code>result = without + guidance × (with − without)</code>. It costs twice the compute.',
        actions: [['Show me the scheduler', () => select('scheduler')]] },
      { title: 'Guidance off', text: 'First a baseline with guidance off.',
        actions: [['Guidance off and ▶ Run', () => { quick(); set('scheduler', { guidance_scale: 1 }); run('lesson: guidance off') }]] },
      { title: 'Guidance up, with a negative prompt', text: 'Now turn it up and push away from “snow”. <span class="k">Look for:</span> the picture follows the prompt more literally, colours get stronger, and the thing you named in the negative prompt retreats.',
        actions: [['Guidance on and ▶ Run', () => { set('input_prompt', { negative_prompt: 'snow, winter' }); set('scheduler', { guidance_scale: isDiT() ? 4 : 7.5 }); run('lesson: guidance + negative') }]] },
      { title: 'Too much', text: 'Guidance is extrapolation: it pushes past anything the model saw in training. Turn it far up and pictures burn out — hard contrast, fried colours.',
        actions: [['Far too much and ▶ Run', () => { set('scheduler', { guidance_scale: isDiT() ? 10 : 20 }); run('lesson: too much guidance') }], ['Back to normal', () => { set('scheduler', { guidance_scale: S.info.default_guidance }); set('input_prompt', { negative_prompt: '' }) }]] },
    ] },
  { id: 'img-kv', title: 'A different prompt for different blocks', time: '15 min', needs: () => S.info?.capabilities?.prompt_kv ? '' : 'This lesson needs a model that can give each block its own prompt (Qwen-Image).',
    goal: 'Find out which blocks decide what is in the picture and which decide how it looks.',
    steps: [
      { title: 'The prompt is a KV cache', text: 'In Qwen-Image the prompt tokens do not depend on the noise level. So the prompt is pushed through all blocks <b>once</b>, and each block keeps the prompt’s keys and values — exactly a language model’s KV cache. Every denoising step reuses them. That also means each block can be handed a <i>different</i> prompt.',
        actions: [['Show me where the prompt enters', () => select('denoiser_in')]] },
      { title: 'Second half reads another prompt', text: () => `Give blocks ${Math.floor(nL() / 2)}–${nL() - 1} the prompt “a cat” while the first half still reads the fox prompt. <span class="k">Predict first:</span> fox or cat?`,
        actions: [['Add it and ▶ Run', () => { quick(); removeKind('prompt_kv'); addPromptKV('replace'); flow.updateNodeData(S.selected.id, { prompt: 'a grey cat sitting in a snowy forest, photo', blocks: `${Math.floor(nL() / 2)}:${nL()}` }); renderInspector(); run('lesson: late blocks read cat') }]] },
      { title: 'Now the first half', text: 'Swap it: the early blocks read “cat”, the late ones read “fox”. Compare the two runs. Whichever half wins decides <i>what</i> is drawn; the other mostly shapes texture and detail.',
        actions: [['Swap and ▶ Run', () => { const n = node('prompt_kv'); if (!n) return toast('Do the step above first.', 'error'); flow.updateNodeData(n.id, { blocks: `0:${Math.floor(nL() / 2)}` }); renderInspector(); run('lesson: early blocks read cat') }]] },
      { title: 'A hidden style prompt', text: 'A <b>hidden prompt</b> sits in front of the main one for chosen blocks: the image and the main prompt both attend to it. Put a style there and leave the subject in the main prompt.',
        actions: [['Hidden style and ▶ Run', () => { removeKind('prompt_kv'); addPromptKV('prepend'); flow.updateNodeData(S.selected.id, { prompt: 'a watercolour painting, soft washes of colour', blocks: 'all' }); renderInspector(); run('lesson: hidden style') }], ['Remove it', () => removeKind('prompt_kv')]] },
    ] },
  { id: 'img-blocks', title: 'Which blocks matter?', time: '10 min',
    goal: 'Break the model on purpose and learn from what survives.',
    steps: [
      { title: 'Take one out', text: () => isDiT()
        ? 'A transformer adds each block’s work onto a running stream, so you can skip a block and the rest still runs. Skip one from the middle. <span class="k">Predict first:</span> noise, a broken picture, or almost the same picture?'
        : 'UNet stages are tied together by skip connections, so they cannot be removed. Instead, zero what the <b>middle</b> stage writes. <span class="k">Predict first:</span> noise, a broken picture, or almost the same picture?',
        actions: [['Do it and ▶ Run', () => {
          quick()
          if (isDiT()) { blocks().forEach(b => b.data.bypass && bypassBlock(b)); bypassBlock(block(Math.floor(nL() / 2))) }
          else { removeKind('intervention'); const mid = blocks().find(b => b.id === 'mid') || block(Math.floor(nL() / 2)); addIntervention(mid); flow.updateNodeData(S.selected.id, { label: 'mid off', hook: 'out', op: 'zero' }); renderInspector() }
          run('lesson: one block off')
        }]] },
      { title: 'The first and the last', text: 'Now try the ends. The first block turns raw noise into something the model can work with; the last one prepares the answer. Losing one of these usually hurts far more than losing a middle one.',
        actions: [['Last one off and ▶ Run', () => {
          if (isDiT()) { blocks().forEach(b => b.data.bypass && bypassBlock(b)); bypassBlock(block(nL() - 1)) }
          else { removeKind('intervention'); addIntervention(block(nL() - 1)); flow.updateNodeData(S.selected.id, { label: 'last stage off', hook: 'out', op: 'zero' }); renderInspector() }
          run('lesson: last block off')
        }], ['Put everything back', () => { blocks().forEach(b => b.data.bypass && bypassBlock(b)); removeKind('intervention') }]] },
      { title: () => isDiT() ? 'Repeat a block' : 'Measure every stage', text: () => isDiT()
        ? 'Select a block and press <b>Repeat block</b>: the same weights run twice in a row. Sometimes that sharpens, often it over-cooks. The model was never trained for it — which tells you how much each block expects from the one before.'
        : 'The <b>Block contributions</b> tab does this for every stage at once: where it writes, what it encodes, and how much the picture changes without it.',
        actions: isDiT() ? [['Select a middle block', () => select(block(Math.floor(nL() / 2))?.id)]] : [['Open Block contributions', () => setTab('contrib')]] },
    ] },
  { id: 'img-lens', title: 'What each block thinks the picture is', time: '10 min', needs: () => S.info?.capabilities?.image_lens ? '' : 'This lesson needs a transformer image model (Qwen-Image).',
    goal: 'Read the half-finished picture out of the middle of the model.',
    steps: [
      { title: 'Open the image lens', text: 'After any block you can apply the model’s own final layer and ask: “if you stopped here, what would you draw?” Columns are steps, rows are blocks. <span class="k">Look for:</span> early rows are a mess, late rows look like the picture.',
        actions: [['Open Image lens', () => { if (needRun()) setTab('ilens') }]] },
      { title: 'Why the middle looks like a mess', text: 'Middle blocks do not “speak pictures” yet — their numbers are in a working format only later blocks understand. Choose <b>decode through the last k blocks</b>: the lens sends each block’s output through the model’s last few blocks first, which translates it. Language models behave the same way (their version is the logit lens).',
        actions: [['Open Image lens', () => { if (needRun()) setTab('ilens') }]] },
    ] },
  { id: 'img-start', title: 'Start from a picture', time: '10 min',
    goal: 'Use your own picture as the starting point, and control how much of it survives.',
    steps: [
      { title: 'The trick', text: 'Do not start from pure noise. Take a picture, add noise to it up to some level, and let the model finish from there. <b>Strength</b> is that noise level: low keeps your picture and only polishes it; high keeps just the rough layout.',
        actions: [['Add a start picture (the last output)', () => { if (!S.run?.outputs?.image) return toast('Make a picture first (“From noise to picture”).', 'error'); addInitImage(); flow.updateNodeData(S.selected.id, { image: S.run.outputs.image, strength: 0.45, mask: '' }); renderInspector() }]] },
      { title: 'Same picture, new prompt', text: 'Keep the layout, change the story. <span class="k">Look for:</span> the fox stays where it was; the scene changes around it.',
        actions: [['“…at night” and ▶ Run', () => { if (!node('init_image')) return toast('Add the start picture first.', 'error'); set('input_prompt', { prompt: 'a red fox sitting in a forest at night, moonlight, photo' }); run('lesson: start image, low strength') }]] },
      { title: 'Turn the strength up', text: 'More noise means more freedom. At high strength only big shapes and colours survive.',
        actions: [['Strength 0.8 and ▶ Run', () => { if (!node('init_image')) return toast('Add the start picture first.', 'error'); set('init_image', { strength: 0.8 }); run('lesson: start image, high strength') }]] },
      { title: 'Lock a region', text: 'In the Start image box, paint a region. <b>keep</b> resets that region to your picture at every step (it cannot change); <b>regenerate</b> flips it, so only the painted part changes. This is inpainting — and it is the picture version of forcing tokens in a language model.',
        actions: [['Show me the start image box', () => select('init_image')], ['Remove the start picture', () => removeKind('init_image')]] },
    ] },
  { id: 'img-debug', title: 'Step in and paint mid-way', time: '15 min', needs: () => schema().step_debugger ? '' : 'This model cannot be stepped through (Qwen-Image can).',
    goal: 'Stop between steps, change the picture by hand, and let the model carry on.',
    steps: [
      { title: 'Start stepping', text: 'Press <b>🐞 Step</b>. The <b>Step debugger</b> tab shows the model’s input (left) and its current guess (right). <b>Step ▸▸</b> runs one denoising step; <b>Block ▸</b> runs one block inside it.',
        actions: [['Start stepping', () => { quick(); debugStart() }]] },
      { title: 'Paint on the guess', text: 'After a few steps press <b>✏️ Edit image…</b> and draw something — a blob of colour is enough. Then keep stepping. The model treats your edit as its own guess and builds on it.',
        actions: [['Open the step debugger', () => setTab('debug')]] },
      { title: 'Why early edits vanish', text: 'Early on the input is 80–95% noise, so an edit is drowned out. The <b>between-step noise σ</b> slider lowers the noise so more of your picture reaches the model; raise it and the model repaints more freely. This is the same dial as “strength” in the previous lesson.',
        actions: [['Open the step debugger', () => setTab('debug')]] },
    ] },
  { id: 'img-train', title: 'Train your own image model', time: '10 min',
    goal: 'See the whole training recipe work on a model small enough to finish in a minute.',
    steps: [
      { title: 'The recipe is four lines', text: 'Take a picture and random noise. Pick a noise level σ. Mix: <code>x = (1−σ)·picture + σ·noise</code>. Teach the model to output <code>noise − picture</code>. That is all of flow matching — Qwen-Image was trained the same way, on billions of pictures instead of coloured shapes.',
        actions: [['Open the Train tab', () => setTab('train')]] },
      { title: 'Watch it learn', text: 'Start the <b>tiny image model</b>. The grid shows what it draws for every colour and shape as training goes on. <span class="k">Look for:</span> colour appears first, shape later — big, easy patterns are learned before fine ones.',
        actions: [['Open the Train tab', () => setTab('train')]] },
      { title: 'The picture it never saw', text: 'One combination — the yellow triangle — is left out of training. Ask for it. If it comes out right, the model learned “yellow” and “triangle” as separate ideas it can combine. Then move the guidance slider and watch the same effect as in the guidance lesson, on a model you trained.',
        actions: [['Open the Train tab', () => setTab('train')]] },
    ] },
]

// ------------------------------------------------------------------ the language-model course
const setLM = (prompt, n = 8) => { set('input_prompt', { prompt, system: '', prefill: '', use_chat_template: false }); set('sampler', { max_new_tokens: n, temperature: 0 }) }

const TEXT = [
  { id: 'lm-tokens', title: 'Tokens, not words', time: '5 min', goal: 'See what the model actually receives.',
    steps: [{ title: 'Type and watch', text: 'A language model never sees letters. Text is cut into <b>tokens</b> from a fixed vocabulary; common words are one token, rare ones several. Limits and prices count tokens. Try a long word, a number, another language. Click a token to see which tokens the model starts out treating as similar.',
      actions: [['Open the Tokenizer', () => setTab('tokenizer')]] }] },
  { id: 'lm-next', title: 'One token at a time', time: '10 min', goal: 'See that an answer is a chain of single guesses.',
    steps: [
      { title: 'Run a prompt', text: 'The model does one thing: given the tokens so far, score every possible next token. Run it, then click a generated token in the Output tab to see the choices it weighed. Click another choice to <b>fork</b> the text from there.',
        actions: [['Set the prompt and ▶ Run', () => { setLM('The capital of France is'); run('lesson: next token') }]] },
      { title: 'Scores become a choice', text: 'The <b>Sampler</b> tab shows the scores for one position. Move temperature, top-k and top-p: the scores do not change, only how a token is picked from them.',
        actions: [['Open the Sampler', () => { if (needRun()) setTab('sampler') }]] },
    ] },
  { id: 'lm-lens', title: 'Where the answer forms', time: '10 min', goal: 'Watch the answer appear layer by layer.',
    steps: [
      { title: 'Read the middle of the model', text: 'The <b>Logit lens</b> asks after every block: “if you stopped here, what would you say?” Rows are blocks, columns are tokens. <span class="k">Look for:</span> the block where the final answer first shows up in the last column.',
        actions: [['Open the Logit lens', () => { if (needRun()) setTab('lens') }], ['Open Word cloud', () => { if (needRun()) setTab('cloud') }]] },
      { title: 'Who looks at whom', text: 'Blocks move information between tokens with <b>attention</b>. Pick a late block and look at the last row: which earlier tokens does the final token read from?',
        actions: [['Open Attention', () => { if (needRun()) setTab('attn') }]] },
    ] },
  { id: 'lm-ablate', title: 'Break it on purpose', time: '10 min', goal: 'Learn what a part does by removing it.',
    steps: [
      { title: 'Skip a block', text: 'Each block adds its work onto a running stream, so the model still runs with a block missing. <span class="k">Predict first:</span> which hurts more — a middle block or the first one?',
        actions: [['Middle block off and ▶ Run', () => { setLM('The capital of France is'); blocks().forEach(b => b.data.bypass && bypassBlock(b)); bypassBlock(block(Math.floor(nL() / 2))); run('lesson: middle block off') }],
          ['First block off and ▶ Run', () => { blocks().forEach(b => b.data.bypass && bypassBlock(b)); bypassBlock(block(0)); run('lesson: first block off') }],
          ['Put everything back', () => blocks().forEach(b => b.data.bypass && bypassBlock(b))]] },
    ] },
  { id: 'lm-patch', title: 'Find where a fact lives', time: '15 min', goal: 'Use activation patching to locate one piece of knowledge.',
    steps: [
      { title: 'Run the clean prompt', text: 'Two prompts that differ in one word. First the <b>clean</b> one.', actions: [['▶ Run “…France is”', () => { setLM('The capital of France is', 4); run('clean') }]] },
      { title: 'Put in the corrupted prompt and sweep', text: 'Now change one word and open <b>Patching</b>. The sweep re-runs this prompt once per block and token, each time copying one activation from the clean run. Green cells are places where “France” is stored: copy just that, and the answer comes back.',
        actions: [['Set “…Japan is” and open Patching', () => { setLM('The capital of Japan is', 4); setTab('patch') }]] },
      { title: 'Prove it', text: 'Click a green cell: a patch box appears on that block. Press Run. The prompt says Japan; the model answers for France.', actions: [['Open Patching', () => setTab('patch')]] },
    ] },
  { id: 'lm-steer', title: 'Steer and hunt neurons', time: '15 min', goal: 'Push the model along a direction; find single units that matter.',
    steps: [
      { title: 'A direction for a word', text: 'In <b>Steering</b>, make a vector from a word (try “Tokyo”) and wire it into a block. Raise the value until the text bends. Too much and it repeats one word forever.', actions: [['Open Steering', () => setTab('steer')]] },
      { title: 'Single neurons', text: 'In <b>Neurons</b>, pick a block and a token. Each row is one of thousands of units, with the words it pushes for. Pin one to 5× its maximum and run again.', actions: [['Open Neurons', () => { if (needRun()) setTab('neurons') }]] },
    ] },
  { id: 'lm-layer', title: 'A prompt only some layers can read', time: '10 min', goal: 'Give different layers different context.',
    steps: [
      { title: 'Prefill the KV cache of a few blocks', text: 'A <b>per-layer KV prefill</b> is text whose keys and values are placed in front of the prompt in the blocks you choose — and nowhere else. Aim an instruction at a few middle blocks, run, then widen it to all blocks and compare. Position <b>overwrite</b> makes those blocks read it <i>instead of</i> the prompt. The <b>KV caches</b> tab lists every prefill with its blocks.',
        actions: [['Add one for the middle blocks', () => { const a = Math.floor(nL() / 3); addLayerPrompt(`${a}:${2 * a}`); setTab('kv') }]] },
    ] },
  { id: 'lm-code', title: 'The model is a short program', time: '10 min', goal: 'Read one block as code, then change it.',
    steps: [
      { title: 'Open a block', text: 'Everything a block does fits in about fifteen lines of Python. Open it, read it top to bottom, then change one line — delete <code>x = x + mlp_out</code>, or multiply <code>attn_out</code> by 2 — and run.',
        actions: [['Open the code of a middle block', () => { const b = block(Math.floor(nL() / 2)); if (b) { select(b.id); editCode(b) } }]] },
    ] },
  { id: 'lm-train', title: 'Smaller, and your own', time: '15 min', goal: 'Shrink a model, train one, adapt one.',
    steps: [
      { title: 'Fewer bits', text: '<b>Quantize</b> rounds the weights to 8, 4 or 2 bits and compares the text. 8-bit is nearly free. Plain 4-bit rounding breaks small models — which is why real 4-bit files use smarter schemes.', actions: [['Open Quantize', () => setTab('quant')]] },
      { title: 'Train one', text: 'In <b>Train</b>, a small transformer learns a made-up language in seconds. Load it afterwards: every tab works on a model you trained. Then train a <b>LoRA</b> on a few example lines and watch the answers change.', actions: [['Open Train', () => setTab('train')]] },
    ] },
]

// ------------------------------------------------------------------ progress
const KEY = 'tokenshark.lessons'

const store = () => { try { return JSON.parse(localStorage.getItem(KEY) || '{}') } catch { return {} } }

const save = d => { try { localStorage.setItem(KEY, JSON.stringify(d)) } catch { } }

const COURSES = { image: IMAGE, text: TEXT }
const course = () => COURSES[schema().course] || []

const val = (v) => typeof v === 'function' ? v() : v

// ------------------------------------------------------------------ the docked lesson panel
export function openLesson(id) {
  const L = [...IMAGE, ...TEXT].find(l => l.id === id)
  if (!L) return
  S.lesson = id
  document.body.classList.add('lesson-open')
  drawLesson()
  setTimeout(() => flow.fit(), 60)
}

export function closeLesson() { S.lesson = null; document.body.classList.remove('lesson-open'); $('#lesson').innerHTML = ''; setTimeout(() => flow.fit(), 60); if (S.tab === 'learn') renderTab() }

export function drawLesson() {
  const box = $('#lesson')
  const L = [...IMAGE, ...TEXT].find(l => l.id === S.lesson)
  if (!L) return
  const prog = store(), done = new Set(prog[L.id] || [])
  box.innerHTML = ''
  const head = el('div', { className: 'lhead' })
  head.append(el('b', { textContent: L.title }), btn('✕', closeLesson, 'mini ghost'))
  box.append(head, el('div', { className: 'small', textContent: L.goal }))
  const why = L.needs?.()
  if (why) box.appendChild(el('div', { className: 'lneed', textContent: why }))
  L.steps.forEach((s, i) => {
    const card = el('div', { className: 'lstep' + (done.has(i) ? ' done' : '') })
    const t = el('div', { className: 'ltitle' })
    const tick = el('span', { className: 'ltick', textContent: done.has(i) ? '✓' : String(i + 1), title: 'Mark this step done' })
    tick.onclick = () => { done.has(i) ? done.delete(i) : done.add(i); prog[L.id] = [...done]; save(prog); drawLesson() }
    t.append(tick, el('span', { textContent: val(s.title) }))
    card.append(t, el('div', { className: 'ltext' }, val(s.text)))
    const acts = val(s.actions) || []
    if (acts.length) {
      const row = el('div', { className: 'row' })
      acts.forEach(([label, fn], j) => row.appendChild(btn(label, () => {
        if (!S.info) return toast('Load a model first.', 'error')
        if (why) return toast(why, 'error')
        try { fn() } catch (e) { toast(e.message, 'error') }
        if (!done.has(i)) { done.add(i); prog[L.id] = [...done]; save(prog); card.classList.add('done'); tick.textContent = '✓' }
      }, j === 0 ? 'primary' : '')))
      card.appendChild(row)
    }
    box.appendChild(card)
  })
  const list = course(), at = list.findIndex(l => l.id === L.id)
  const nav = el('div', { className: 'row' })
  if (at > 0) nav.appendChild(btn('← ' + list[at - 1].title, () => openLesson(list[at - 1].id), 'ghost'))
  if (at >= 0 && at < list.length - 1) nav.appendChild(btn('Next: ' + list[at + 1].title + ' →', () => openLesson(list[at + 1].id)))
  nav.appendChild(btn('All lessons', () => setTab('learn'), 'ghost'))
  box.appendChild(nav)
}

// ------------------------------------------------------------------ the Learn tab
export async function renderLearn(p) {
  p.innerHTML = ''
  if (!S.info) return renderStart(p)
  const img = schema().course === 'image'
  const list = course(), prog = store()
  p.appendChild(el('div', { className: 'lintro' }, `<h3>${img ? 'Image models, from the inside' : 'Language models, from the inside'}</h3>
    <div class="small">${img
      ? `A course on how <b>${esc(S.info.arch)}</b> turns words and noise into a picture. Each lesson opens beside the flow and sets things up for you; you run, look, and say what you think happened before reading on.`
      : `A course on how <b>${esc(S.info.model_id)}</b> turns a prompt into text. Each lesson opens beside the flow and sets things up for you; you run, look, and say what you think happened before reading on.`}
      Go in order the first time.</div>`))
  const grid = el('div', { className: 'lgrid' })
  list.forEach((L, i) => {
    const n = (prog[L.id] || []).length, why = L.needs?.()
    const c = el('button', { className: 'lcard' + (why ? ' off' : '') + (S.lesson === L.id ? ' on' : '') })
    c.innerHTML = `<span class="lnum">${n >= L.steps.length ? '✓' : i + 1}</span><span><b>${esc(L.title)}</b><span class="small">${esc(L.goal)}</span>
      <span class="small">${esc(L.time)} · ${L.steps.length} step${L.steps.length > 1 ? 's' : ''}${n ? ` · ${Math.min(n, L.steps.length)} done` : ''}${why ? ' · needs another model' : ''}</span></span>`
    c.onclick = () => openLesson(L.id)
    grid.appendChild(c)
  })
  p.appendChild(grid)
  const row = el('div', { className: 'row' })
  row.append(btn('Start over (clear ticks)', () => { save({}); renderTab(); if (S.lesson) drawLesson() }, 'ghost mini'), btn('Reset the flow', () => resetFlow(), 'ghost mini'))
  p.appendChild(row)
}

// Nothing loaded yet: a plain way in.
async function renderStart(p) {
  let models = S.models || []
  if (!models.length) { try { models = await GET('/api/models') } catch { } }
  if (S.tab !== 'learn' || S.info) return
  p.innerHTML = ''
  p.appendChild(el('div', { className: 'lintro' }, `<h3>Welcome. Pick a model to open up.</h3>
    <div class="small">Tokenshark draws a model as boxes and arrows, runs it on your computer, and lets you look inside and change what it does.
    Small models are best for learning: they are quick, and they work the same way as the big ones.</div>`))
  const pick = (ids, fallbackFam) => ids.map(id => models.find(m => m.id === id)).find(Boolean) || models.find(m => m.family === fallbackFam)
  const cards = [
    ['Text', 'How a language model writes', pick(['qwen2.5-0.5b-instruct', 'smollm2-360m-instruct'], 'causal_lm'), 'Tokens, attention, where facts are stored, steering, training.'],
    ['Pictures', 'How an image model draws', pick(['sd-turbo', 'tiny-sd'], 't2i'), 'Noise to picture in a few steps. Small and quick.'],
    ['Pictures, large', 'Qwen-Image, a modern transformer image model', models.find(m => m.id.startsWith('qwen-image')), 'Per-block prompts, the image lens, stepping and painting mid-way. Large: slow without a strong GPU.'],
  ]
  const grid = el('div', { className: 'lgrid' })
  cards.forEach(([kind, title, m, blurb]) => {
    if (!m) return
    const c = el('button', { className: 'lcard' })
    c.innerHTML = `<span class="lnum">${kind === 'Text' ? 'Aa' : '▦'}</span><span><b>${esc(title)}</b><span class="small">${esc(blurb)}</span>
      <span class="small">${esc(m.id)} · ${m.size_gb ?? '?'} GB · ${m.downloaded ? 'already on this computer' : 'downloads on first use'}</span></span>`
    c.onclick = () => { $('#model').value = m.id; $('#model').dispatchEvent(new Event('change')); $('#load').click(); toast(`Loading ${m.id}…`) }
    grid.appendChild(c)
  })
  const t = el('button', { className: 'lcard' })
  t.innerHTML = `<span class="lnum">★</span><span><b>No download: train a tiny model right now</b><span class="small">A small language model or a small image model, trained from scratch in about a minute on this computer.</span></span>`
  t.onclick = () => setTab('train')
  grid.appendChild(t)
  p.appendChild(grid)
  void refreshModels
}

registerTab({ id: 'learn', title: '📖 Learn', cls: 'learn', hint: 'Guided lessons for the loaded model', render: renderLearn })

on('model', () => { if (S.lesson) closeLesson() })      // lessons belong to one kind of model
