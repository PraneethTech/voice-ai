@echo off
setlocal
cd /d %~dp0
echo === tobi v2 setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [FAIL] python not found on PATH. Install Python 3.11+ from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
python --version

if not exist .venv-v2\Scripts\python.exe (
  echo [1/5] Creating venv .venv-v2 ...
  python -m venv .venv-v2
  if errorlevel 1 (
    echo [FAIL] venv creation failed.
    pause
    exit /b 1
  )
) else (
  echo [1/5] venv .venv-v2 exists, skipping.
)

echo [2/5] Installing requirements ...
.venv-v2\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo [FAIL] pip install failed - see output above.
  pause
  exit /b 1
)

if not exist .env (
  echo [3/5] Creating .env from .env.example ...
  copy .env.example .env >nul
  echo        Edit .env and set GROQ_API_KEY if you want the Groq dev-bridge brain.
) else (
  echo [3/5] .env exists, skipping.
)

echo [4/5] Downloading models - one-time, needs network ...
.venv-v2\Scripts\python.exe tools\download_models.py
if errorlevel 1 (
  echo [FAIL] model download failed - see output above.
  pause
  exit /b 1
)

echo [5/5] Running tests ...
.venv-v2\Scripts\python.exe -m pytest tests/ -q
if errorlevel 1 (
  echo [WARN] some tests failed - see output above.
  pause
  exit /b 1
)

echo.
echo Setup complete. Starting server on http://127.0.0.1:8000 ...
echo Open the URL, enable the microphone, say "hey jarvis".
.venv-v2\Scripts\python.exe -m server.ws_server --config configs\desktop.yaml
