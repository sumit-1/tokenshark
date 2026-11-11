"""Weight store with offloading.

Every raw module asks the store for a tensor by name: ``W("model.layers.3.mlp.up_proj.weight")``.
Where that tensor lives is a policy, so the same model code runs on a big GPU or a school laptop:

    device   all weights resident on the compute device (fastest)
    cpu      all weights resident in RAM, copied to the device when used
    disk     nothing resident: each layer is read from the memory-mapped files when it runs,
             then freed. A 70B model needs only ~1 layer (~1.7 GB bf16) of memory at a time.
    auto     pick from model size vs free memory

`keep_layers` keeps the first N blocks resident on the device even under cpu/disk (llama.cpp-style
"GPU layers"). Formats: Hugging Face safetensors (single or sharded) and Meta consolidated .pth.
"""
from __future__ import annotations

import contextlib

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import psutil
except ImportError:  # optional
    psutil = None

_LAYER_RE = re.compile(r"(?:layers|blocks|transformer_blocks)\.(\d+)\.")


# ------------------------------------------------------------------ load progress (set by the registry while loading)
_REPORTERS: dict = {"phase": None, "bytes": None}


def report_phase(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    if _REPORTERS["phase"]:
        _REPORTERS["phase"](msg)


def report_bytes(done: int, total: int, speed: float, what: str):
    if _REPORTERS["bytes"]:
        _REPORTERS["bytes"](done, total, speed, what)


_DT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
            "F8_E4M3": 1, "F8_E5M2": 1}
_DT_TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
             "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
             "BOOL": torch.bool, "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8),
             "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8)}


class _SafetensorsFile:
    """Minimal safetensors reader using ordinary file reads — no memory map.

    With a memory map, every page of a 4 GB shard that a block touches stays in the process's working set
    (Windows counts it as used RAM), so streaming a 55 GB model 'fills' memory and the machine starts paging.
    A plain read copies the tensor into a buffer we own; when the block is done the buffer is freed."""

    def __init__(self, path: Path):
        import struct
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        self.base = 8 + n
        self.meta = header                      # name -> {"dtype", "shape", "data_offsets": [start, end]}
        self.lock = threading.Lock()
        self.fh = open(path, "rb", buffering=0)

    def keys(self):
        return self.meta.keys()

    def shape(self, name):
        return tuple(self.meta[name]["shape"])

    def nbytes(self, name):
        a, b = self.meta[name]["data_offsets"]
        return b - a

    def _read_range(self, start: int, length: int) -> bytearray:
        buf = bytearray(length)
        with self.lock:
            self.fh.seek(self.base + start)
            view, got = memoryview(buf), 0
            while got < length:
                r = self.fh.readinto(view[got:])
                if not r:
                    raise IOError(f"unexpected end of {self.path}")
                got += r
        return buf

    def get(self, name) -> torch.Tensor:
        m = self.meta[name]
        a, b = m["data_offsets"]
        dt = _DT_TORCH[m["dtype"]]
        if b == a:
            return torch.empty(m["shape"], dtype=dt)
        return torch.frombuffer(self._read_range(a, b - a), dtype=dt).reshape(m["shape"])

    def get_rows(self, name, ids) -> torch.Tensor:
        """Selected rows of a 2-D tensor (embedding lookup) without reading the whole matrix."""
        m = self.meta[name]
        a, _ = m["data_offsets"]
        rows, cols = m["shape"]
        dt = _DT_TORCH[m["dtype"]]
        rb = cols * _DT_SIZE[m["dtype"]]
        out = [torch.frombuffer(self._read_range(a + i * rb, rb), dtype=dt) for i in ids]
        return torch.stack(out) if out else torch.empty(0, cols, dtype=dt)


class WeightStore:
    def __init__(self, root: str | Path, device: str, dtype: torch.dtype, policy: str = "auto",
                 keep_layers: int = 0, name_map: dict[str, str] | None = None,
                 transform: Callable[[str, torch.Tensor], torch.Tensor] | None = None, files: list[Path] | None = None):
        self.root = Path(root)
        self.device = device
        self.dtype = dtype
        self.name_map = name_map            # canonical name -> name in the checkpoint (None = identity)
        self.transform = transform
        self.keep_layers = keep_layers
        self.lock = threading.Lock()
        self._handles: dict[Path, object] = {}
        self._index: dict[str, Path] = {}
        self._pth: dict[str, torch.Tensor] = {}
        self._resident: dict[str, torch.Tensor] = {}
        self._held: dict[str, torch.Tensor] | None = None
        self._lora = None                   # optional LoRAOverlay attached to this store
        self._quant = None                  # optional QuantOverlay (rounds block weights to a few bits)
        self._extra: dict[str, torch.Tensor] = {}   # weights that are not in the files: added blocks, adapter layers
        self.bytes_read = 0
        report_phase(f"opening weight files in {self.root}")
        self._discover(files)
        # Multimodal checkpoints keep the language model under model.language_model.* (vision under model.visual.*,
        # multi-token-prediction heads under mtp.*). Model code always asks for model.layers.… names.
        self.alias = None
        if "model.embed_tokens.weight" not in self._index and "model.language_model.embed_tokens.weight" in self._index:
            self.alias = ("model.", "model.language_model.")
        self.skip_prefixes = ("mtp.", "model.visual.", "visual.", "model.mtp.")
        self.total_bytes = self._estimate_bytes()
        self.policy = self._choose(policy)
        report_phase(f"weights: {self.total_bytes / 2**30:.1f} GB in {len(set(self._index.values()))} files · policy "
                     f"'{self.policy}'" + (" (nothing preloaded: each block is read from disk when it runs)"
                                           if self.policy == "disk" and not keep_layers else ""))
        if self.policy in ("device", "cpu") or keep_layers:
            self._preload()

    # ---------------------------------------------------------------- discovery
    def _discover(self, files):
        files = files or sorted(self.root.glob("*.safetensors"))
        half = [f for f in files if ".fp16." in f.name]
        full = [f for f in files if ".fp16." not in f.name]
        if half and full:      # both variants downloaded: use the one matching the compute dtype
            files = half if self.dtype in (torch.float16, torch.bfloat16) else full
        if files:
            for f in files:
                h = _SafetensorsFile(f)
                self._handles[f] = h
                for k in h.keys():
                    self._index[k] = f
            return
        bins = sorted(self.root.glob("*.bin"))
        if bins:
            for b in bins:
                sd = torch.load(str(b), map_location="cpu", mmap=True, weights_only=True)
                self._pth.update(sd)
                for k in sd:
                    self._index[k] = b
            return
        pths = sorted(self.root.glob("consolidated.*.pth"))
        if pths:
            if len(pths) > 1:
                raise ValueError("Multi-shard Meta checkpoints (tensor-parallel) are not supported; use the "
                                 "Hugging Face safetensors version of this model.")
            sd = torch.load(str(pths[0]), map_location="cpu", mmap=True, weights_only=True)
            self._pth = sd
            for k in sd:
                self._index[k] = pths[0]
            return
        raise FileNotFoundError(f"No .safetensors or consolidated.*.pth weights in {self.root}")

    def _estimate_bytes(self) -> int:
        total = 0
        for f in set(self._index.values()):
            total += f.stat().st_size
        return total

    def _choose(self, policy: str) -> str:
        if policy != "auto":
            return policy
        need = self.total_bytes * 1.15          # resident weights stay in their stored (usually bf16) size
        if self.device == "cuda":
            free, _ = torch.cuda.mem_get_info()
            if need < free:
                return "device"
        ram = psutil.virtual_memory().available if psutil else 8 << 30
        if self.device == "cpu":
            return "device" if need < ram * 0.7 else "disk"
        return "cpu" if need < ram * 0.7 else "disk"

    # ---------------------------------------------------------------- access
    def keys(self) -> list[str]:
        return list(self._index.keys())

    def has(self, name: str) -> bool:
        return name in self._extra or self._src(name) in self._index

    def _src(self, name: str) -> str:
        if self.name_map:
            return self.name_map.get(name, name)
        if self.alias and name not in self._index and name.startswith(self.alias[0]):
            return self.alias[1] + name[len(self.alias[0]):]
        return name

    def shape(self, name: str):
        if name in self._extra:
            return tuple(self._extra[name].shape)
        src = self._src(name)
        if src not in self._index:
            return None
        if self._pth:
            return tuple(self._pth[src].shape)
        return self._handles[self._index[src]].shape(src)

    def _read(self, src: str) -> torch.Tensor:
        f = self._index[src]
        if self._pth:
            t = self._pth[src]
        else:
            t = self._handles[f].get(src)
        self.bytes_read += t.numel() * t.element_size()
        return t

    def _layer_of(self, name: str) -> int | None:
        m = _LAYER_RE.search(name)
        return int(m.group(1)) if m else None

    def _prepare(self, name: str, t: torch.Tensor, device: str, compact: bool = False) -> torch.Tensor:
        """compact=True (resident copies): keep a smaller stored dtype (bf16 checkpoint, fp32 compute on CPU) and
        convert when the weight is used — halves the RAM a resident 27B model needs."""
        if self.transform is not None:
            t = self.transform(name, t)
        if t.is_floating_point():
            if not (compact and t.element_size() < torch.empty(0, dtype=self.dtype).element_size()):
                t = t.to(dtype=self.dtype)
        return t.to(device)

    def _nbytes(self, src: str) -> int:
        if self._pth:
            t = self._pth[src]
            return t.numel() * t.element_size()
        return self._handles[self._index[src]].nbytes(src)

    def _preload(self):
        names = [k for k in self._all_names()
                 if self.policy in ("device", "cpu") or ((self._layer_of(k) is not None) and self._layer_of(k) < self.keep_layers)]
        total = sum(self._nbytes(self._src(k)) for k in names)
        where = "GPU" if self.device == "cuda" and self.policy == "device" else "RAM"
        report_phase(f"reading {total / 2**30:.1f} GB of weights into {where}")
        done, t0, last = 0, time.time(), 0.0
        for k in names:
            self._preload_one(k)
            done += self._nbytes(self._src(k))
            now = time.time()
            if now - last > 0.3 or done == total:
                report_bytes(done, total, done / max(now - t0, 1e-6), f"weights → {where}")
                last = now

    def _preload_one(self, k):
        layer = self._layer_of(k)
        if self.policy == "device":
            target = self.device
        elif self.policy == "cpu":
            target = self.device if (layer is not None and layer < self.keep_layers) else "cpu"
        else:                                                    # disk + keep_layers
            target = self.device
        t = self._prepare(k, self._read(self._src(k)), target, True)
        self._resident[k] = t

    def _all_names(self):
        if self.name_map:
            yield from (k for k, v in self.name_map.items() if v in self._index)
            return
        for k in self._index:
            if k.startswith(self.skip_prefixes):        # vision tower / MTP heads: not used by the text path
                continue
            if self.alias and k.startswith(self.alias[1]):
                yield self.alias[0] + k[len(self.alias[1]):]
            else:
                yield k

    @contextlib.contextmanager
    def hold(self, keep: dict | None = None):
        """Inside this block, weights read from disk are kept (in their stored dtype) until the block ends.
        Analyses that run the same few blocks many times (decode-through-k for every layer) then read each
        block once per request instead of once per layer. `keep` lets a task carry its kept weights across
        several blocks of work (with other tasks running in between)."""
        outer = self._held
        self._held = keep if keep is not None else ({} if outer is None else outer)
        try:
            yield
        finally:
            self._held = outer

    def compact(self, name: str) -> torch.Tensor:
        """The weight in its stored dtype (e.g. bf16), CPU unless it is already resident on the device."""
        t = self._resident.get(name)
        if t is not None:
            return t
        return self._prepare(name, self._read(self._src(name)), "cpu", compact=True)

    def set_lora_hook(self, lora_overlay):
        """Attach or remove a LoRA / fine-tune overlay on this WeightStore."""
        self._lora = lora_overlay

    def __call__(self, name: str) -> torch.Tensor:
        t = self._extra.get(name) if self._extra else None
        if t is None:
            t = self._resident.get(name)
        if t is None and self._held is not None:
            t = self._held.get(name)
            if t is None:
                src = self._src(name)
                if src not in self._index:
                    raise KeyError(f"weight '{src}' not found")
                t = self._held[name] = self._prepare(name, self._read(src), "cpu", compact=True)
        if t is not None:
            if t.is_floating_point() and t.dtype != self.dtype:
                out = t.to(self.device, self.dtype)
            else:
                out = t if t.device.type == torch.device(self.device).type else t.to(self.device, non_blocking=True)
        else:
            src = self._src(name)
            if src not in self._index:
                raise KeyError(f"weight '{src}' not found")
            out = self._prepare(name, self._read(src), self.device)
        if self._lora is not None:
            out = self._lora.apply(name, out)
        if self._quant is not None:
            out = self._quant.apply(name, out)
        return out

    def rows(self, name: str, ids: list[int]) -> torch.Tensor:
        """Embedding lookup without materialising the whole matrix (matters for 128k-vocab 70B models)."""
        t = self._resident.get(name)
        if t is not None:
            return t[torch.tensor(ids, device=t.device)].to(self.device, self.dtype)
        src = self._src(name)
        if self._pth:
            full = self._pth[src]
            out = full[torch.tensor(ids)]
        else:
            out = self._handles[self._index[src]].get_rows(src, list(ids))
        if self.transform is not None:
            out = self.transform(name, out)
        return out.to(self.device, self.dtype)

    def describe(self) -> dict:
        return {"policy": self.policy, "keep_layers": self.keep_layers, "total_gb": round(self.total_bytes / 2**30, 2),
                "resident_gb": round(sum(t.numel() * t.element_size() for t in self._resident.values()) / 2**30, 2),
                "files": len(set(self._index.values()))}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def _pick_files(repo_files: list[str], family: str) -> list[str]:
    """Choose which files to download: configs + tokenizer + one weight format per component."""
    keep = [f for f in repo_files if f.endswith((".json", ".txt", ".model", ".jinja"))
            and not f.startswith((".", "onnx", "openvino", "flax"))]
    folders = [""] if family == "causal_lm" else ["text_encoder/", "unet/", "transformer/", "vae/"]
    for folder in folders:
        here = [f for f in repo_files if f.startswith(folder) and "/" not in f[len(folder):]]
        st = [f for f in here if f.endswith(".safetensors")]
        bins = [f for f in here if f.endswith(".bin") and "training" not in f and "optimizer" not in f]
        fp16 = [f for f in st if ".fp16." in f]
        full = [f for f in st if ".fp16." not in f]
        if family == "t2i" and fp16:
            keep += fp16
        elif full or fp16:
            keep += full or fp16
        else:
            fb16 = [f for f in bins if ".fp16." in f]
            keep += (fb16 if family == "t2i" and fb16 else [f for f in bins if ".fp16." not in f] or bins)
        if family == "causal_lm":
            keep += [f for f in here if f.endswith("tokenizer.model")]
    if family == "causal_lm" and not any(f.endswith((".safetensors", ".bin")) for f in keep):
        keep += [f for f in repo_files if f.startswith("original/") and (f.endswith(".pth") or f.endswith("params.json")
                                                                         or f.endswith("tokenizer.model"))]
    return sorted(set(keep))


def models_dir() -> Path:
    """Where downloaded models live: ./models next to the app, or $ANALYSER_MODELS / --models-dir."""
    d = Path(os.environ.get("ANALYSER_MODELS") or Path(__file__).parent.parent / "models").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_folder(spec: dict) -> Path:
    if spec.get("path"):
        return Path(os.path.expanduser(spec["path"]))
    return models_dir() / spec["id"]


def is_downloaded(spec: dict) -> bool:
    f = model_folder(spec)
    return f.exists() and ((f / ".complete").exists() or bool(spec.get("path")))


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


class _Watcher:
    """Reports download progress by watching the target folder grow (works with any huggingface_hub version:
    files land as .incomplete under target/.cache while downloading)."""

    def __init__(self, target: Path, total: int, base: int, report):
        self.target, self.total, self.base, self.report = target, total, base, report
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        last_b, last_t, speed = None, None, 0.0
        while not self.stop.wait(0.5):
            done = max(0, folder_size(self.target) - self.base)
            now = time.time()
            if last_b is not None and now > last_t:
                inst = (done - last_b) / (now - last_t)
                speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
            last_b, last_t = done, now
            self.report(min(done, self.total), self.total, max(speed, 0.0))

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(timeout=2)
        self.report(self.total, self.total, 0.0)


def _repo_sizes(repo: str) -> dict[str, int]:
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        return {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    except Exception:
        return {}


def resolve_model_dir(spec: dict, allow_download: bool = True, status=None, progress=None) -> Path:
    """A local folder (spec.path), or models/<id>/: downloaded from Hugging Face on first use with only the
    files the engine needs (configs, tokenizer, one copy of the weights). Plain files, no hidden cache:
    move, copy or delete the folder like any other."""
    if spec.get("path"):
        p = Path(os.path.expanduser(spec["path"]))
        if not p.exists():
            raise FileNotFoundError(f"Model path does not exist: {p}")
        return p
    target = models_dir() / spec["id"]
    if (target / ".complete").exists():
        return _inner(target, spec)
    from huggingface_hub import list_repo_files, snapshot_download
    if _adopt_hf_cache(spec, target, status, progress):
        return _inner(target, spec)
    try:
        if not allow_download:
            raise OSError("offline")
        files = _pick_files(list_repo_files(spec["repo"]), spec.get("family", "causal_lm"))
        if status:
            status(f"downloading {spec['repo']} → {target}")
        sizes = _repo_sizes(spec["repo"])
        total = sum(sizes.get(f, 0) for f in files)
        target.mkdir(parents=True, exist_ok=True)
        # files already fully present will be skipped by the download: count them as done
        have = sum((target / f).stat().st_size for f in files if (target / f).exists() and (target / f).stat().st_size == sizes.get(f, -1))
        report = (lambda d, t, sp: progress(have + d, total, sp, spec["repo"])) if progress else (lambda *a: None)
        with _Watcher(target, max(0, total - have), folder_size(target), report):
            snapshot_download(spec["repo"], allow_patterns=files, local_dir=str(target))
    except Exception as e:
        if not any(target.glob("*")):
            raise RuntimeError(f"Could not download {spec['repo']} ({e}). Check the internet connection, or for gated "
                               f"models accept the licence on huggingface.co and run `hf auth login`.") from e
    (target / ".complete").write_text(spec.get("repo", ""))
    return _inner(target, spec)


def _adopt_hf_cache(spec: dict, target: Path, status=None, progress=None) -> bool:
    """If an earlier version downloaded this repo into the Hugging Face cache, copy it into models/<id>/
    instead of downloading again."""
    import shutil
    from huggingface_hub import snapshot_download
    try:
        cached = Path(snapshot_download(spec["repo"], local_files_only=True))
    except Exception:
        return False
    rel = [f.relative_to(cached) for f in cached.rglob("*") if f.is_file() or f.is_symlink()]
    names = [str(r).replace("\\", "/") for r in rel]
    wanted = set(_pick_files(names, spec.get("family", "causal_lm")))
    if not any(n.endswith((".safetensors", ".bin", ".pth")) for n in wanted):
        return False
    if status:
        status(f"copying {spec['repo']} from the Hugging Face cache → {target}")
    todo = [(r, n) for r, n in zip(rel, names) if n in wanted]
    total = sum((cached / r).resolve().stat().st_size for r, _ in todo)
    done = 0
    for r, n in todo:
        dst = target / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile((cached / r).resolve(), dst)
        done += dst.stat().st_size
        if progress:
            progress(done, total, 0.0, spec["repo"])
    (target / ".complete").write_text(spec.get("repo", ""))
    return True


def _inner(root: Path, spec: dict) -> Path:
    if spec.get("family") == "causal_lm" and not (root / "config.json").exists() and (root / "original").exists():
        return root / "original"
    return root


def discover_folders(known_ids: set[str]) -> list[dict]:
    """Model folders a user dropped into models/ by hand become selectable models."""
    out = []
    for d in sorted(models_dir().iterdir()):
        if not d.is_dir() or d.name in known_ids or d.name.startswith("."):
            continue
        if (d / "model_index.json").exists():
            fam = "t2i"
        elif (d / "config.json").exists() or (d / "params.json").exists() or (d / "original" / "params.json").exists():
            fam = "causal_lm"
        else:
            continue
        out.append({"id": d.name, "family": fam, "path": str(d), "size_gb": round(folder_size(d) / 2**30, 2),
                    "notes": "found in the models folder", "discovered": True})
    return out
