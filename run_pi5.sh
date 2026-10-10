#!/usr/bin/env bash
# ==============================================================================
# Run SPAN Display + Tobi Voice AI Assistant on Raspberry Pi 5
#
# Usage:
#   ./run_pi5.sh          # Starts both UI and voice pipeline
#   ./run_pi5.sh --kiosk  # Starts UI in fullscreen kiosk mode on Pi display + pipeline
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-5000}"
UI_DIR="ai-voice-assistant-ui"
V2_DIR="stt-vad-tts/personal_offline_assistant/v2"

# Trap Ctrl+C and kill background processes cleanly
cleanup() {
  echo ""
  echo "Stopping services..."
  kill $(jobs -p) 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "=========================================================="
echo " Starting Voice AI + Display on Raspberry Pi 5"
echo "=========================================================="

# 1. Start SPAN Screen
cd "$UI_DIR"
if [ "${1:-}" = "--kiosk" ]; then
  echo "[1/2] Starting SPAN Screen with Chromium Kiosk..."
  ./run.sh --kiosk &
else
  echo "[1/2] Starting SPAN Display Server on port $PORT..."
  ./run.sh &
fi
cd ..

# Wait for display endpoint to be healthy
echo "Waiting for SPAN screen to initialize..."
until curl -s "http://127.0.0.1:$PORT/api/state" >/dev/null 2>&1; do
  sleep 0.5
done
echo "SPAN display is ready at http://127.0.0.1:$PORT"

# 2. Start Voice Pipeline
echo "[2/2] Starting Voice Pipeline with Pi 5 profile..."
cd "$V2_DIR"
VENV=".venv-v2"
PYTHON_BIN="python3"
if [ -d "$VENV" ]; then
  PYTHON_BIN="$VENV/bin/python"
fi

"$PYTHON_BIN" pipeline_runner.py \
  --config configs/pi5-4gb.yaml \
  --display "http://127.0.0.1:$PORT/api/event"
