"""Editable KV cache.

Keys are stored *before* rotary position encoding (after QK-norm where the model has one), with an
explicit position per token. RoPE is applied at attention time from those positions, so any part of
the cache can be re-positioned without recomputation:

    prepend  — put another conversation's cache in front of this one
               mode "shift":    imported keeps 0..m-1, new tokens continue at m, m+1, ...
               mode "negative": imported moves to -m..-1, new tokens start at 0 (as if fresh)
               mode "offset":   imported positions + offset (any integer), new tokens start at 0
               mode "overwrite": imported at 0..m-1 *in place of* the prompt, for the blocks that see it
    truncate / drop a span / shift a span's positions by Δ / export to and import from a file.

One K/V tensor pair per *schedule position* (a block repeated twice owns two slots).
Files are safetensors with the segment table and model fingerprint in the metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Segment:
    start: int
    end: int
    origin: str          # prompt | prefill | generated | imported
    label: str = ""

    def to_dict(self):
        return {"start": self.start, "end": self.end, "origin": self.origin, "label": self.label}


@dataclass
class KVCache:
    fingerprint: dict
    n_slots: int
    k: list = field(default_factory=list)       # per slot: [n_kv, T, head_dim] or None
    v: list = field(default_factory=list)
    pos: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float32))
    tokens: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    device: str = "cpu"
    # imported prefixes: [(start, end, slots)] — tokens start..end-1 came from a prefix cache and only the schedule
    # positions in `slots` may attend to them (None = every block). Several prefixes can each target other blocks.
    prefix_spans: list = field(default_factory=list)
    # "overwrite" prefixes: [(start, end, slots)] — in the schedule positions in `slots` (None = all) tokens
    # start..end-1 (the prompt) are hidden from every other token, so those blocks read the prefix instead.
    hidden_spans: list = field(default_factory=list)
    # Hybrid models (Qwen3.5/3.6): linear-attention (Gated DeltaNet) blocks keep no per-token K/V, only a running
    # memory per slot: {"conv": last K-1 inputs, "rec": [heads, dk, dv] matrix}. It summarises the whole sequence in
    # order, so it can be used whole as a prefix but not cut, reordered or re-positioned.
    states: dict = field(default_factory=dict)
    init_states: dict = field(default_factory=dict)     # memory right after the imported prefixes

    def __post_init__(self):
        if not self.k:
            self.k = [None] * self.n_slots
            self.v = [None] * self.n_slots

    # ---------------------------------------------------------------- growth
    def __len__(self):
        return len(self.tokens)

    def add_tokens(self, ids: list[int], positions: list[float], origin: str, label: str = ""):
        start = len(self.tokens)
        self.tokens += list(ids)
        self.pos = torch.cat([self.pos, torch.tensor(positions, dtype=torch.float32)])
        if self.segments and self.segments[-1].origin == origin and self.segments[-1].end == start:
            self.segments[-1].end = len(self.tokens)
        else:
            self.segments.append(Segment(start, len(self.tokens), origin, label))

    def append(self, slot: int, k: torch.Tensor, v: torch.Tensor):
        k = k.detach().to(self.device)
        v = v.detach().to(self.device)
        if self.k[slot] is None:
            self.k[slot], self.v[slot] = k, v
        else:
            self.k[slot] = torch.cat([self.k[slot], k], dim=1)
            self.v[slot] = torch.cat([self.v[slot], v], dim=1)

    def lin_state(self, slot: int, conv_dim: int, L: dict, device) -> dict:
        st = self.states.get(slot)
        if st is None:
            st = {"conv": torch.zeros(max(L["conv"] - 1, 0), conv_dim, device=device),
                  "rec": torch.zeros(L["hv"], L["dk"], L["dv"], device=device, dtype=torch.float32)}
            self.states[slot] = st
        else:
            st["conv"], st["rec"] = st["conv"].to(device), st["rec"].to(device)
        return st

    @staticmethod
    def _clone_states(states: dict) -> dict:
        return {k: {kk: vv.detach().clone() for kk, vv in v.items()} for k, v in states.items()}

    def next_position(self) -> float:
        return float(self.pos.max()) + 1 if len(self.pos) else 0.0

    # ---------------------------------------------------------------- edits (all return new caches)
    def _empty_like(self):
        return KVCache(dict(self.fingerprint), self.n_slots, device=self.device)

    def select(self, idx: list[int], _prefix_states: bool = False) -> "KVCache":
        whole = list(idx) == list(range(len(self)))
        if self.states and not whole and not _prefix_states:
            raise ValueError("This cache comes from a hybrid model: its linear-attention (Gated DeltaNet) blocks keep "
                             "one running memory of the whole sequence, not one entry per token, so it can't be cut, "
                             "reordered or partly dropped. It can still be prepended whole.")
        out = self._empty_like()
        out.states = self._clone_states(self.init_states if _prefix_states else self.states)
        out.init_states = self._clone_states(self.init_states)
        it = torch.tensor(idx, dtype=torch.long)
        for s in range(self.n_slots):
            if self.k[s] is not None:
                out.k[s] = self.k[s][:, it.to(self.k[s].device)]
                out.v[s] = self.v[s][:, it.to(self.v[s].device)]
        out.tokens = [self.tokens[i] for i in idx]
        out.pos = self.pos[it] if idx else torch.zeros(0)
        # rebuild segments from per-token origins
        origins = self._origins()
        for j, i in enumerate(idx):
            o, lab = origins[i]
            if out.segments and out.segments[-1].origin == o and out.segments[-1].label == lab:
                out.segments[-1].end = j + 1
            else:
                out.segments.append(Segment(j, j + 1, o, lab))
        return out

    def _origins(self):
        o = [("prompt", "")] * len(self.tokens)
        for s in self.segments:
            for i in range(s.start, s.end):
                o[i] = (s.origin, s.label)
        return o

    def prefix_view(self, n: int) -> "KVCache":
        """The first n tokens (the imported prefix) with the linear-attention memory as it was right after them."""
        out = self.select(list(range(n)), _prefix_states=True)
        out.prefix_spans = list(self.prefix_spans)
        out.hidden_spans = list(self.hidden_spans)
        return out

    def truncate(self, n: int) -> "KVCache":
        return self.select(list(range(min(n, len(self)))))

    def drop(self, start: int, end: int) -> "KVCache":
        return self.select([i for i in range(len(self)) if not start <= i < end])

    def shift(self, start: int, end: int, delta: float) -> "KVCache":
        out = self.select(list(range(len(self))))
        out.pos[start:end] += delta
        return out

    def relabel(self, origin: str, label: str) -> "KVCache":
        out = self.select(list(range(len(self))))
        out.segments = [Segment(0, len(out), origin, label)] if len(out) else []
        return out

    def check_compatible(self, other: "KVCache"):
        a, b = self.fingerprint, other.fingerprint
        for key in ("n_kv_heads", "head_dim", "hidden_size"):
            if a.get(key) != b.get(key):
                raise ValueError(f"KV cache is from a different model ({key}: {b.get(key)} vs {a.get(key)}).")
        if self.n_slots != other.n_slots:
            raise ValueError(f"KV cache has {other.n_slots} layer slots but the current flow has {self.n_slots} "
                             f"blocks. Use the same block schedule the cache was made with.")

    @staticmethod
    def concat(first: "KVCache", second: "KVCache") -> "KVCache":
        first.check_compatible(second)
        out = first._empty_like()
        for s in range(first.n_slots):
            ks = [c.k[s] for c in (first, second) if c.k[s] is not None]
            vs = [c.v[s] for c in (first, second) if c.v[s] is not None]
            out.k[s] = torch.cat([k.to(out.device) for k in ks], 1) if ks else None
            out.v[s] = torch.cat([v.to(out.device) for v in vs], 1) if vs else None
        out.tokens = first.tokens + second.tokens
        out.pos = torch.cat([first.pos, second.pos])
        off = len(first)
        out.segments = [Segment(s.start, s.end, s.origin, s.label) for s in first.segments] + \
                       [Segment(s.start + off, s.end + off, s.origin, s.label) for s in second.segments]
        # linear-attention memory can't be merged: the later cache's memory wins (callers decide per slot)
        out.states = KVCache._clone_states({**first.states, **second.states})
        return out

    def positioned_for_prepend(self, mode: str, offset: float = 0.0) -> tuple["KVCache", float]:
        """Return (copy of this cache re-positioned for use as a prefix, first position for new tokens)."""
        out = self.select(list(range(len(self))))
        if not len(out):
            return out, 0.0
        if mode == "shift":
            out.pos = out.pos - out.pos.min()
            return out, float(out.pos.max()) + 1
        if mode == "negative":
            out.pos = out.pos - (out.pos.max() + 1)
            return out, 0.0
        if mode == "offset":
            out.pos = out.pos - out.pos.min() + offset
            return out, max(0.0, float(out.pos.max()) + 1) if offset >= 0 else 0.0
        if mode == "overwrite":
            # Sits on the same positions as the prompt (0..m-1) and takes its place: the blocks that see this
            # prefix don't see the prompt's own keys (the caller hides them, see KVCache.hidden_spans).
            out.pos = out.pos - out.pos.min()
            return out, 0.0
        raise ValueError(f"Unknown prepend mode '{mode}' (shift | negative | offset | overwrite).")

    # ---------------------------------------------------------------- files
    def save(self, path: Path, extra: dict | None = None):
        from safetensors.torch import save_file
        tensors = {"positions": self.pos.contiguous(), "tokens": torch.tensor(self.tokens, dtype=torch.int64)}
        for s in range(self.n_slots):
            if self.k[s] is not None:
                tensors[f"k.{s}"] = self.k[s].detach().cpu().contiguous()
                tensors[f"v.{s}"] = self.v[s].detach().cpu().contiguous()
        for s, st in self.states.items():
            tensors[f"s.{s}.conv"] = st["conv"].detach().float().cpu().contiguous()
            tensors[f"s.{s}.rec"] = st["rec"].detach().float().cpu().contiguous()
        meta = {"format": "ai-analyser-kv/1", "fingerprint": json.dumps(self.fingerprint),
                "n_slots": str(self.n_slots), "segments": json.dumps([s.to_dict() for s in self.segments]),
                "extra": json.dumps(extra or {})}
        save_file(tensors, str(path), metadata=meta)

    @staticmethod
    def load(path: Path, device: str = "cpu") -> tuple["KVCache", dict]:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if meta.get("format") != "ai-analyser-kv/1":
                raise ValueError("Not an AI Analyser KV-cache file.")
            n = int(meta["n_slots"])
            c = KVCache(json.loads(meta["fingerprint"]), n, device=device)
            c.pos = f.get_tensor("positions").float()
            c.tokens = f.get_tensor("tokens").tolist()
            for s in range(n):
                if f"k.{s}" in f.keys():
                    c.k[s] = f.get_tensor(f"k.{s}").to(device)
                    c.v[s] = f.get_tensor(f"v.{s}").to(device)
                if f"s.{s}.rec" in f.keys():
                    c.states[s] = {"conv": f.get_tensor(f"s.{s}.conv").to(device),
                                   "rec": f.get_tensor(f"s.{s}.rec").to(device)}
            c.segments = [Segment(**d) for d in json.loads(meta.get("segments", "[]"))]
        return c, json.loads(meta.get("extra", "{}"))

    def summary(self, piece=None) -> dict:
        return {
            "n_tokens": len(self), "n_slots": self.n_slots,
            "segments": [s.to_dict() for s in self.segments],
            "positions": [round(float(p), 2) for p in self.pos.tolist()],
            "tokens": [piece(t) if piece else t for t in self.tokens],
            "bytes": int(sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)
                         + sum(t.numel() * t.element_size() for st in self.states.values() for t in st.values())),
            "linear_slots": sorted(self.states.keys()),
        }
