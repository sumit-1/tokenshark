"""AI Analyser — inspection-and-intervention engine for open-weight generative models."""
from pathlib import Path as _Path

__version__ = "0.2.0"


def code_stamp() -> str:
    """Newest modification time of the app's source files — changes whenever an update lands on disk."""
    root = _Path(__file__).parent
    return str(int(max(f.stat().st_mtime for f in list(root.rglob("*.py")) + [f for f in (root / "web").rglob("*") if f.is_file()])))


STARTED_WITH = code_stamp()
