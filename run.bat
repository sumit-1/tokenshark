@echo off
REM AI Analyser — Windows launcher. First run creates a virtual environment and installs requirements.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || goto :err
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  echo.
  echo Installing PyTorch. For an NVIDIA GPU, stop here and install the CUDA build from https://pytorch.org first.
  python -m pip install -r requirements.txt || goto :err
) else (
  call .venv\Scripts\activate.bat
)
python -m analyser %*
goto :eof
:err
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
