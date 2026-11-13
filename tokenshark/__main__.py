"""python -m tokenshark [--host 127.0.0.1] [--port 8000]"""
import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def _ensure_venv():
    """If launched with global `python -m tokenshark` outside `.venv` and a required package is missing,
    automatically re-exec inside the local `.venv` when present."""
    try:
        import tokenizers  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        root = Path(__file__).resolve().parent.parent
        candidates = [root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"]
        for vpy in candidates:
            if vpy.exists() and vpy.resolve() != Path(sys.executable).resolve():
                raise SystemExit(subprocess.call([str(vpy), "-m", "tokenshark", *sys.argv[1:]]))


def main():
    _ensure_venv()
    ap = argparse.ArgumentParser(description="Tokenshark — inspect and edit open-weight models in the browser")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=os.environ.get("TOKENSHARK_DEVICE", "auto"), help="auto | cuda | mps | cpu")
    ap.add_argument("--dtype", default=os.environ.get("TOKENSHARK_DTYPE", "auto"), help="auto | fp32 | fp16 | bf16")
    ap.add_argument("--models-dir", default=os.environ.get("TOKENSHARK_MODELS", ""),
                    help="where models are downloaded (default: ./models next to the app)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    os.environ["TOKENSHARK_DEVICE"], os.environ["TOKENSHARK_DTYPE"] = a.device, a.dtype
    if a.models_dir:
        os.environ["TOKENSHARK_MODELS"] = a.models_dir
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((a.host if a.host != "0.0.0.0" else "127.0.0.1", a.port)) == 0:
            print(f"\nPort {a.port} is already in use — an older Tokenshark is probably still running.\n"
                  f"Close its window (or end its python.exe in Task Manager) and start again,\n"
                  f"or use another port:  run.bat --port {a.port + 1}\n")
            raise SystemExit(1)
    import uvicorn
    from . import __version__
    from .server import app
    url = f"http://{a.host}:{a.port}"
    print(f"Tokenshark {__version__} running at {url}  (press Ctrl+C to stop)")
    if not a.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
