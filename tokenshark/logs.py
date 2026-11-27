"""Progress messages of the library (loading phases, run events) go to the "tokenshark" logger on stderr, so a
script's own stdout stays clean.

    import tokenshark
    tokenshark.set_verbosity("warning")     # quiet: only problems
    tokenshark.set_verbosity("info")        # default: loading phases with a timestamp

TOKENSHARK_QUIET=1 starts quiet. An application that configures `logging` itself can attach its own handlers to
the "tokenshark" logger; the default handler is added only when the logger has none.
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("tokenshark")

class _SafeHandler(logging.StreamHandler):
    """Never crash on characters the console can't show (cp1252 Windows consoles and token pieces)."""

    def emit(self, record):
        try:
            msg = self.format(record)
            enc = getattr(self.stream, "encoding", None) or "utf-8"
            self.stream.write(msg.encode(enc, "replace").decode(enc, "replace") + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


if not log.handlers:
    _h = _SafeHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.propagate = False
    log.setLevel(logging.WARNING if os.environ.get("TOKENSHARK_QUIET", "") not in ("", "0") else logging.INFO)


def set_verbosity(level: str | int = "info") -> None:
    """'debug' | 'info' | 'warning' | 'error' (or a logging level number)."""
    log.setLevel(level.upper() if isinstance(level, str) else level)
