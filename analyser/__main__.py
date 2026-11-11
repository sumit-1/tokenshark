"""python -m analyser [--host 127.0.0.1] [--port 8000]"""
import argparse
import os
import webbrowser


def main():
    ap = argparse.ArgumentParser(description="Tokenshark — inspect and edit open-weight models in the browser")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=os.environ.get("ANALYSER_DEVICE", "auto"), help="auto | cuda | mps | cpu")
    ap.add_argument("--dtype", default=os.environ.get("ANALYSER_DTYPE", "auto"), help="auto | fp32 | fp16 | bf16")
    ap.add_argument("--models-dir", default=os.environ.get("ANALYSER_MODELS", ""),
                    help="where models are downloaded (default: ./models next to the app)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    os.environ["ANALYSER_DEVICE"], os.environ["ANALYSER_DTYPE"] = a.device, a.dtype
    if a.models_dir:
        os.environ["ANALYSER_MODELS"] = a.models_dir
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
