import { GET, S, sleep } from '../core.js'

// Lens / cloud / track run as tasks on the server (see the Tasks list). Results arrive layer by layer:
// onPart is called with the partial result each time; the promise resolves with the final one.
S.streams = {}

export async function streamAnalysis(url, onPart, alive) {
  let id = S.streams[url], first
  if (!id) {
    first = await GET(url + (url.includes('?') ? '&' : '?') + 'background=true')
    if (first.result) onPart(first.result, first.task || {})
    if (!first.task) return first.result
    id = S.streams[url] = first.task.id
  }
  try {
    while (true) {
      const t = await GET(`/api/tasks/${id}`)
      if (!alive()) return null                 // the view moved on; the task keeps going (see Tasks)
      if (t.result) onPart(t.result, t)
      if (t.status === 'done') return t.result
      if (t.status === 'error') throw new Error(t.error)
      if (t.status === 'cancelled') throw new Error('Stopped — from the Tasks list, or replaced by a newer request.')
      await sleep(t.status === 'running' ? 350 : 700)
    }
  } finally { if (S.streams[url] === id && alive()) delete S.streams[url] }
}
