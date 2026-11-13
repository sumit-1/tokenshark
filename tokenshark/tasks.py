"""Task manager: every piece of work that needs the model (runs, debugger steps, logit lens, word clouds, word
tracking) is a Task. Work is done in small units — one block of a run, one layer of a word cloud — and before
each unit the task asks for the model with `TASKS.slot(task)`. The model goes to the first waiting task in
the queue order, so the user can move a task up, pause it or stop it, and the change takes effect at the
next unit boundary (one block: seconds even for a 27B model streamed from disk).

Order: new analysis tasks (what you just clicked) go to the front, debugger steps after them, runs last.
Tasks with the same `group` replace each other: clicking another token in the word cloud stops the old cloud.
"""
from __future__ import annotations

import contextlib
import itertools
import threading
import time
import uuid

from .engine import Cancelled
from .registry import LOADED

_RANK = {"analysis": 0, "debug": 1, "run": 2}
KEEP_FINISHED = 30


class Task:
    def __init__(self, kind: str, label: str, group: str | None = None, run_id: str | None = None):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.label, self.group, self.run_id = kind, label, group, run_id
        self.status = "queued"            # queued · running · paused · done · cancelled · error
        self.progress, self.message = 0.0, ""
        self.created, self.started, self.finished = time.time(), None, None
        self.cancel, self.paused = False, False
        self.active = False               # holds the model right now
        self.result = None                # partial results while running, final when done
        self.error: str | None = None
        self.held: dict = {}              # weights this task keeps between units (WeightStore.hold)
        self.on_cancel = None             # hooks into run / stepper contexts
        self.on_pause = None

    @property
    def live(self):
        return self.status in ("queued", "running", "paused")

    def summary(self) -> dict:
        status = "stopping" if self.cancel and self.live else self.status
        return {"id": self.id, "kind": self.kind, "label": self.label, "status": status,
                "progress": round(self.progress, 3), "message": self.message, "active": self.active,
                "created": self.created, "seconds": round((self.finished or time.time()) - (self.started or self.created), 1),
                "error": self.error, "run_id": self.run_id}


class TaskManager:
    def __init__(self, model_lock: threading.Lock):
        self.model_lock = model_lock
        self.cond = threading.Condition()
        self.tasks: dict[str, Task] = {}
        self.order: list[str] = []        # queue order of live tasks
        self.holder: Task | None = None
        self._seq = itertools.count()

    # ---------------------------------------------------------------- lifecycle
    def new(self, kind: str, label: str, group: str | None = None, run_id: str | None = None) -> Task:
        t = Task(kind, label, group, run_id)
        with self.cond:
            if group:
                for o in self.tasks.values():
                    if o.group == group and o.live:
                        self._cancel(o)
            self.tasks[t.id] = t
            # front of its kind: after live tasks of a more urgent kind, before the rest
            i = 0
            while i < len(self.order) and _RANK[self.tasks[self.order[i]].kind] < _RANK[kind]:
                i += 1
            self.order.insert(i, t.id)
            self._trim()
            self.cond.notify_all()
        return t

    def finish(self, t: Task, status: str = "done", error: str | None = None):
        with self.cond:
            if t.status not in ("done", "cancelled", "error"):
                t.status = "cancelled" if t.cancel and status == "done" else status
            if error:
                t.status, t.error = "error", error
            t.finished = time.time()
            t.active = False
            t.held = {}
            if t.id in self.order:
                self.order.remove(t.id)
            self.cond.notify_all()

    def _trim(self):
        done = [x for x in self.tasks.values() if not x.live]
        for x in sorted(done, key=lambda x: x.finished or 0)[:-KEEP_FINISHED]:
            self.tasks.pop(x.id, None)

    # ---------------------------------------------------------------- the model slot
    @contextlib.contextmanager
    def slot(self, t: Task, force: bool = False):
        """Wait for this task's turn, then hold the model for one unit of work. force=True: only wait for the
        model to be free (used to save the state of a task that is being paused or stopped)."""
        with self.cond:
            while True:
                if t.cancel and not force:
                    raise Cancelled("Task stopped.")
                if self.holder is None and (force or self._next() is t):
                    break
                self.cond.wait(0.25)
            self.holder = t
            t.active = True
            if t.started is None:
                t.started = time.time()
            if not t.paused and not t.cancel:
                t.status = "running"
        self.model_lock.acquire()          # plain endpoints (attention, KV) still use the lock directly
        try:
            yield
        finally:
            self.model_lock.release()
            with self.cond:
                self.holder = None
                t.active = False
                if t.paused and t.live:
                    t.status = "paused"
                self.cond.notify_all()

    def _next(self) -> Task | None:
        for tid in self.order:
            x = self.tasks.get(tid)
            if x is not None and not x.paused and not x.cancel:
                return x
        return None

    # ---------------------------------------------------------------- user actions
    def _cancel(self, t: Task):
        t.cancel = True
        if t.on_cancel:
            try:
                t.on_cancel()
            except Exception:
                pass
        if t.status == "queued" and t.started is None and not t.active:
            # never started: nothing will notice the flag, so close it here
            t.status, t.finished = "cancelled", time.time()
            if t.id in self.order:
                self.order.remove(t.id)

    def act(self, task_id: str, action: str) -> dict:
        with self.cond:
            t = self.tasks.get(task_id)
            if t is None:
                raise KeyError(task_id)
            if action == "cancel":
                self._cancel(t)
            elif action in ("pause", "resume"):
                t.paused = action == "pause"
                if t.on_pause:
                    t.on_pause(t.paused)
                if t.live and not t.active:
                    t.status = "paused" if t.paused else ("running" if t.started else "queued")
            elif action in ("top", "up", "down", "bottom") and t.id in self.order:
                i = self.order.index(t.id)
                self.order.pop(i)
                j = {"top": 0, "up": max(0, i - 1), "down": min(len(self.order), i + 1), "bottom": len(self.order)}[action]
                self.order.insert(j, t.id)
            else:
                raise ValueError(f"unknown action {action}")
            self.cond.notify_all()
            return t.summary()

    def list(self) -> list[dict]:
        with self.cond:
            live = [self.tasks[i].summary() for i in self.order if i in self.tasks]
            for n, s in enumerate(live):
                s["queue"] = n + 1
            done = sorted((x for x in self.tasks.values() if not x.live), key=lambda x: -(x.finished or 0))
            return live + [x.summary() for x in done[:10]]

    # ---------------------------------------------------------------- generator tasks
    def run_steps(self, t: Task, gen_fn, weights=None):
        """Drive a generator one unit at a time: each `next()` runs with the model held; whatever the generator
        yields is published as the task's partial result."""
        try:
            gen = gen_fn()
            while True:
                with self.slot(t):
                    ctx = weights.hold(t.held) if weights is not None else contextlib.nullcontext()
                    with ctx:
                        try:
                            part = next(gen)
                        except StopIteration:
                            break
                t.result = part
                if isinstance(part, dict) and "_progress" in part:
                    t.progress = part["_progress"]
                while t.paused and not t.cancel:
                    time.sleep(0.1)
            self.finish(t, "done")
        except Cancelled:
            self.finish(t, "cancelled")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.finish(t, error=f"{type(e).__name__}: {e}")

    def start(self, t: Task, gen_fn, weights=None):
        threading.Thread(target=self.run_steps, args=(t, gen_fn, weights), daemon=True).start()
        return t


TASKS = TaskManager(LOADED.lock)
