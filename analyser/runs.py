"""Runs and the session tree. Each run = flow + compiled plan + seed + outputs; captures stay in memory
(last N runs), manifests go to disk so any run can be replayed."""
from __future__ import annotations

import json
import os
import platform
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

from .flow import Plan

DATA = Path(os.environ.get("ANALYSER_HOME", Path(__file__).parent.parent / "data"))
RUNS_DIR = DATA / "runs"
MAX_CAPTURED = int(os.environ.get("ANALYSER_KEEP_RUNS", "8"))


class Run:
    def __init__(self, model_id: str, flow: dict, plan: Plan, parent_id: str | None, label: str,
                 resume_step: int | None):
        self.id = uuid.uuid4().hex[:10]
        self.model_id = model_id
        self.flow = flow
        self.plan = plan
        self.parent_id = parent_id
        self.label = label
        self.resume_step = resume_step
        self.created = time.time()
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.error: str | None = None
        self.outputs: dict | None = None
        self.captures: dict | None = None
        self.seconds: float | None = None
        self.cache: dict = {}
        self.ctx = None
        self.task = None
        self.active_node: str | None = None
        self.done_nodes: list[str] = []
        self.version = 0          # bumps whenever outputs change (live text while running)

    def summary(self) -> dict:
        out = {"id": self.id, "model_id": self.model_id, "parent_id": self.parent_id, "label": self.label,
               "created": self.created, "status": self.status, "progress": round(self.progress, 3),
               "message": self.message, "error": self.error, "seconds": self.seconds,
               "resume_step": self.resume_step, "has_captures": self.captures is not None,
               "active_node": self.active_node, "done_nodes": self.done_nodes,
               "version": self.version}
        if self.status in ("running", "stepping", "paused"):
            try:
                import psutil
                out["ram_gb"] = round(psutil.Process().memory_info().rss / 2**30, 2)
            except Exception:
                pass
        if self.outputs:
            o = self.outputs
            out["preview"] = (o.get("text") or "")[:80] if "text" in o else None
        return out

    def manifest(self) -> dict:
        import torch
        try:
            import transformers, diffusers
            libs = {"transformers": transformers.__version__, "diffusers": diffusers.__version__}
        except Exception:
            libs = {}
        return {"id": self.id, "model_id": self.model_id, "parent_id": self.parent_id, "label": self.label,
                "resume_step": self.resume_step, "created": self.created, "plan": self.plan.to_dict(),
                "flow": self.flow, "seconds": self.seconds,
                "env": {"torch": torch.__version__, **libs, "python": platform.python_version(),
                        "platform": platform.platform()}}


class RunStore:
    def __init__(self):
        self.runs: "OrderedDict[str, Run]" = OrderedDict()
        self.lock = threading.Lock()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, run: Run):
        with self.lock:
            self.runs[run.id] = run
            captured = [r for r in self.runs.values() if r.captures is not None]
            for r in captured[:-MAX_CAPTURED]:
                r.captures, r.cache = None, {}

    def get(self, run_id: str) -> Run:
        return self.runs[run_id]

    def list(self) -> list[dict]:
        return [r.summary() for r in self.runs.values()]

    def execute(self, run: Run, loaded, resume: dict | None, prefix=None):
        from .tasks import TASKS
        task = run.task = TASKS.new("run", f"run {run.label or run.id}", run_id=run.id)

        def on_cancel():
            if run.ctx is not None:
                run.ctx.cancel = True

        def on_pause(v):
            if run.ctx is not None:
                run.ctx.pause = v
        task.on_cancel, task.on_pause = on_cancel, on_pause

        def progress(frac, msg):
            run.progress, run.message = frac, msg
            task.progress, task.message = frac, msg

        def hook(ctx):
            run.ctx = ctx
            ctx.cancel, ctx.pause = task.cancel, task.paused
            ctx.sources = getattr(run, "sources", None) or {}      # patch / steer / clamp inputs (interp.py)

        def work():
            from .engine import Cancelled
            run.status = "running"
            t0 = time.time()
            try:
                adapter = loaded.adapter
                if adapter is None or loaded.spec["id"] != run.model_id:
                    raise RuntimeError("The model for this run is no longer loaded.")
                # The plugin names the loop its models run with (plugins/__init__.py: run_loop).
                RUN_LOOPS[run_loop_of(loaded)](run, loaded, adapter, prefix, resume, progress, hook, task)
            except Cancelled:
                run.status, run.message = "cancelled", "cancelled"
                print(f"[{time.strftime('%H:%M:%S')}] run {run.id} cancelled"
                      + (" (output so far is kept)" if run.outputs else ""), flush=True)
            except Exception as e:
                run.status, run.error = "error", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            run.seconds = round(time.time() - t0, 2)
            run.ctx = None
            TASKS.finish(task, {"done": "done", "cancelled": "cancelled"}.get(run.status, "error"), run.error)
            try:
                (RUNS_DIR / f"{run.id}.json").write_text(json.dumps(
                    {**run.manifest(), "status": run.status, "error": run.error,
                     "outputs": {k: v for k, v in (run.outputs or {}).items() if k != "image"}}, indent=1))
                if run.outputs and run.outputs.get("image"):
                    import base64
                    (RUNS_DIR / f"{run.id}.png").write_bytes(base64.b64decode(run.outputs["image"].split(",", 1)[1]))
            except Exception:
                traceback.print_exc()
            self.add(run)  # re-apply capture eviction

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _run_plain(run: Run, loaded, adapter, prefix, resume, progress, hook, task):
        """The whole run in one go (models that can't be stepped): no live text or pictures until it is done."""
        from .tasks import TASKS
        with TASKS.slot(task):
            res = adapter.run(run.plan, progress=progress, resume=resume, prefix=prefix, ctx_hook=hook)
        run.outputs, run.captures = res["outputs"], res["captures"]
        run.version += 1
        run.status, run.progress = "done", 1.0

    @staticmethod
    def _run_llm(run: Run, loaded, adapter, prefix, resume, progress, hook, task):
        """A language-model run is a stepper driven one block at a time. The model lock is taken per block, so
        between blocks the analysis views can read the run, a pause can hold it without blocking anything,
        and the text so far is published as it is generated."""
        from .engine import Cancelled
        from .tasks import TASKS
        with TASKS.slot(task):
            st = adapter.run_stepper(run.plan, prefix, progress, hook)
        last_pub, n_pub = 0.0, -1

        def publish():
            nonlocal last_pub, n_pub
            with TASKS.slot(task, force=True):
                res = st.snapshot()
            run.outputs, run.captures, run.cache = res["outputs"], res["captures"], {}
            run.version += 1
            last_pub, n_pub = time.time(), st.n_generated

        try:
            while True:
                if run.ctx.pause and not run.ctx.cancel:
                    if st.logits_rows or st.x is not None:
                        publish()
                    run.status = "paused"
                    run.message = f"paused · token {st.n_generated}/{st.total}" + \
                        (f" · before block {st.bp + 1}/{len(run.plan.schedule)}" if st.x is not None else "")
                    while run.ctx.pause and not run.ctx.cancel:
                        time.sleep(0.1)
                    run.status = "running"
                run.active_node = "embed" if st.x is None else (
                    run.plan.schedule_nodes[st.bp] if st.bp < len(run.plan.schedule_nodes) else "final_norm"
                )
                run.done_nodes = [t["node"] for t in st.trace if "node" in t]
                with TASKS.slot(task):
                    if loaded.adapter is not adapter:
                        raise RuntimeError("The model was unloaded during the run.")
                    more = st.step_block()
                if not more:
                    break
                # new text after every finished token (at most ~once a second for fast models)
                if st.x is None and st.n_generated != n_pub and time.time() - last_pub > 1.0:
                    publish()
        except Cancelled:
            run.active_node, run.done_nodes = None, []
            if st.logits_rows:
                publish()
            raise
        run.active_node, run.done_nodes = None, []
        with TASKS.slot(task, force=True):
            res = st.finish()
        run.outputs, run.captures, run.cache = res["outputs"], res["captures"], {}
        run.version += 1
        run.status, run.progress = "done", 1.0

    @staticmethod
    def _run_image(run: Run, loaded, adapter, prefix, resume, progress, hook, task):
        """A DiT image run is also an ImageStepper driven one block at a time, connected to the Step debugger."""
        from .engine import Cancelled
        from .qimage import ImageStepper
        from .server import DEBUG
        from .tasks import TASKS
        run.active_node = "text_encoder"
        with TASKS.slot(task):
            st = ImageStepper(adapter, run.plan, progress=progress, resume=resume, ctx_hook=hook, trace=True)
        DEBUG.clear()
        DEBUG.update(stepper=st, model_id=loaded.spec["id"], flow=run.flow, label=run.label or "run", run=run, busy=True, task=task.id)
        n_pub = -1

        def publish():
            nonlocal n_pub
            with TASKS.slot(task, force=True):
                res = st.snapshot()
            run.outputs, run.captures, run.cache = res["outputs"], res["captures"], {}
            run.version += 1
            n_pub = st.n_generated
            DEBUG["state"] = {**st.state(), "run_id": run.id}

        try:
            while True:
                if run.ctx.pause and not run.ctx.cancel:
                    if st.n_generated != n_pub or st.x is not None:
                        publish()
                    run.status = "paused"
                    DEBUG["busy"] = False
                    run.message = f"paused · step {min(st.i, st.n_steps - 1) + 1}/{st.n_steps}" + \
                        (f" · before block {st.bp + 1}/{len(run.plan.schedule)}" if st.x is not None else "")
                    while run.ctx.pause and not run.ctx.cancel:
                        time.sleep(0.1)
                    run.status = "running"
                    DEBUG["busy"] = True
                run.active_node = "denoiser_in" if st.x is None else (
                    run.plan.schedule_nodes[st.bp] if st.bp < len(run.plan.schedule_nodes) else "denoiser_out"
                )
                run.done_nodes = [t["node"] for t in st.trace if "node" in t]
                DEBUG["progress"] = {
                    "frac": round((st.bp + 1) / max(1, len(run.plan.schedule)), 3),
                    "message": run.message,
                    "node": run.active_node,
                    "done": run.done_nodes,
                }
                with TASKS.slot(task):
                    if loaded.adapter is not adapter:
                        raise RuntimeError("The model was unloaded during the run.")
                    more = st.step_block()
                if not more:
                    break
                if st.x is None and st.n_generated != n_pub:
                    publish()
        except Cancelled:
            run.active_node, run.done_nodes = None, []
            if st.x0s or st.guess is not None:
                publish()
            DEBUG["busy"] = False
            raise
        run.active_node = "vae_decode"
        run.done_nodes = list(run.plan.schedule_nodes)
        with TASKS.slot(task, force=True):
            res = st.finish()
        run.active_node, run.done_nodes = None, []
        run.outputs, run.captures, run.cache = res["outputs"], res["captures"], {}
        run.version += 1
        run.status, run.progress = "done", 1.0
        DEBUG["busy"] = False
        DEBUG["state"] = {**st.state(), "run_id": run.id}


# How a run is driven. A plugin picks one by name; a new way of running a model is a new entry here.
RUN_LOOPS = {
    "token_stepper": RunStore._run_llm,       # a language model, one block at a time (live text, pause between blocks)
    "image_stepper": RunStore._run_image,     # a steppable image model, one block at a time (tied to the step debugger)
    "plain": RunStore._run_plain,             # adapter.run() from start to end
}


def run_loop_of(loaded) -> str:
    name = getattr(loaded.plugin, "run_loop", "plain")
    if name not in RUN_LOOPS:
        raise RuntimeError(f"Plugin '{getattr(loaded.plugin, 'id', '?')}' names an unknown run loop '{name}'.")
    return name


STORE = RunStore()
