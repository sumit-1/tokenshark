#!/usr/bin/env bash
# Tokenshark — macOS / Linux launcher. First run creates a virtual environment and installs requirements.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  . .venv/bin/activate
fi
exec python -m analyser "$@"
