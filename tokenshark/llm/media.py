"""Media (image) input for language models: the contract plugins implement plus image I/O helpers
(`tokenshark.llm.media`).

A media item reaches the model as a run of soft tokens. The prompt text holds one `marker` per item (or none, and
the items go in front of the newest user turn); after tokenizing, each marker token becomes

    begin_id + token_id x n_tokens(item) + end_id

and the engine overwrites the embeddings of the `token_id` rows with `encode(item)`, in order.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import torch


class MediaEncoder:
    """Base class for a plugin's media encoder. Set the token fields and implement prepare / n_tokens / encode."""

    kind = "image"
    marker = "<|image|>"                # placeholder text in the prompt
    token_id: int = -1                  # placeholder token that soft tokens replace
    begin_id: int | None = None         # token put before / after each item's soft tokens (None: nothing)
    end_id: int | None = None

    def prepare(self, src):
        """Cheap per-item preprocessing (decode, resize, cut into patches). Result goes to n_tokens / encode."""
        raise NotImplementedError

    def n_tokens(self, item) -> int:
        raise NotImplementedError

    def encode(self, item) -> torch.Tensor:
        """Soft tokens [n_tokens, hidden] in the model's dtype and device."""
        raise NotImplementedError


def load_image(src):
    """A PIL RGB image from a PIL image, a file path, raw bytes, a base64 string or a data: URL."""
    from PIL import Image
    if hasattr(src, "convert"):
        return src.convert("RGB")
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    s = str(src)
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    elif len(s) < 1024 and Path(s).exists():
        return Image.open(s).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def to_data_url(src) -> str:
    """A PNG data: URL for any image source load_image() accepts (data: URLs pass through unchanged)."""
    if isinstance(src, str) and src.startswith("data:"):
        return src
    buf = io.BytesIO()
    load_image(src).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
