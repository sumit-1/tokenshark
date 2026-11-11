"""KV-cache library: caches saved from runs or uploaded by users, kept as files under data/caches/."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from .kvcache import KVCache
from .runs import DATA

CACHE_DIR = DATA / "caches"


class CacheLibrary:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


LIBRARY = CacheLibrary()
