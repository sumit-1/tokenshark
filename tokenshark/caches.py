"""KV-cache library: caches saved from runs or uploaded by users, kept as files under data/caches/."""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from .kvcache import KVCache
from .runs import DATA

CACHE_DIR = DATA / "caches"


def _cache_bytes(c: KVCache) -> int:
    n = sum(t.numel() * t.element_size() for t in list(c.k) + list(c.v) if t is not None)
    return n + sum(t.numel() * t.element_size() for st in c.states.values() for t in st.values())


class CacheLibrary:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # caches read by runs (load_shared), kept in memory: {id: (mtime, bytes, cache, extra)}, least recent first
        self._mem: OrderedDict[str, tuple] = OrderedDict()
        self._mem_lock = threading.Lock()
        self.mem_limit = int(os.environ.get("TOKENSHARK_CACHE_MEM_MB", "2048")) * 2**20

    def load_shared(self, cid: str) -> tuple[KVCache, dict]:
        """Like load(), but served from memory when the file hasn't changed. The cache is SHARED between callers:
        read it, never modify it in place (runs copy it before positioning it as a prefix)."""
        path = self.path(cid)
        mtime = path.stat().st_mtime_ns
        with self._mem_lock:
            hit = self._mem.get(cid)
            if hit and hit[0] == mtime:
                self._mem.move_to_end(cid)
                return hit[2], dict(hit[3])
        cache, extra = KVCache.load(path)
        size = _cache_bytes(cache)
        with self._mem_lock:
            if size <= self.mem_limit:
                self._mem[cid] = (mtime, size, cache, extra)
                while sum(v[1] for v in self._mem.values()) > self.mem_limit:
                    self._mem.popitem(last=False)
        return cache, dict(extra)

    def path(self, cid: str) -> Path:
        if not cid.replace("-", "").isalnum():
            raise KeyError(cid)
        p = CACHE_DIR / f"{cid}.safetensors"
        if not p.exists():
            raise KeyError(cid)
        return p

    def save(self, cache: KVCache, name: str, source: dict) -> dict:
        cid = uuid.uuid4().hex[:10]
        extra = {"id": cid, "name": name or f"cache {cid}", "created": time.time(), **source}
        cache.save(CACHE_DIR / f"{cid}.safetensors", extra)
        return {**extra, "n_tokens": len(cache), "n_slots": cache.n_slots, "fingerprint": cache.fingerprint}

    def load(self, cid: str, device: str = "cpu") -> tuple[KVCache, dict]:
        return KVCache.load(self.path(cid), device)

    def list(self) -> list[dict]:
        from safetensors import safe_open
        import json
        out = []
        for p in sorted(CACHE_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    m = f.metadata() or {}
                    extra = json.loads(m.get("extra", "{}"))
                    fp = json.loads(m.get("fingerprint", "{}"))
                    n = f.get_slice("tokens").get_shape()[0]
                out.append({**extra, "id": p.stem, "n_tokens": n, "n_slots": int(m.get("n_slots", 0)),
                            "fingerprint": fp, "bytes": p.stat().st_size})
            except Exception as e:
                out.append({"id": p.stem, "name": p.name, "error": str(e)})
        return out

    def import_bytes(self, data: bytes, name: str) -> dict:
        tmp = CACHE_DIR / f"upload-{uuid.uuid4().hex[:6]}.tmp"
        tmp.write_bytes(data)
        try:
            cache, extra = KVCache.load(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return self.save(cache, name or extra.get("name", "uploaded"), {"model_id": extra.get("model_id"),
                                                                       "source": "upload"})

    def delete(self, cid: str):
        self.path(cid).unlink()
        with self._mem_lock:
            self._mem.pop(cid, None)


LIBRARY = CacheLibrary()
