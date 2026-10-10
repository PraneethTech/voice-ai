#!/usr/bin/env bash
# Start SPAN on the Raspberry Pi.
#   ./run.sh           start the server (open http://<PI-IP>:5000 from any browser)
#   ./run.sh --kiosk   start the server and open the bare screen fullscreen on the Pi's display
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
PORT="${PORT:-5000}"

# --system-site-packages lets the venv use Raspberry Pi OS's lgpio, which gpiozero needs on a Pi 5.
if [ ! -d "$VENV" ]; then
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install -r requirements.txt
  # Pi-only hardware (GPIO button, WS2812). Skipped on laptops.
  if [ -f requirements-pi.txt ] && uname -m | grep -qE '^(aarch64|armv7l|armv6l)$'; then
    "$VENV/bin/pip" install -r requirements-pi.txt
  fi
fi

if [ "${1:-}" = "--kiosk" ]; then
  PORT="$PORT" "$VENV/bin/python" app.py &
  SERVER=$!
  trap 'kill $SERVER' EXIT
  until curl -s "http://127.0.0.1:$PORT/api/state" >/dev/null; do sleep 0.5; done
  BROWSER=$(command -v chromium-browser || command -v chromium)
  "$BROWSER" --kiosk --noerrdialogs --disable-infobars --incognito "http://127.0.0.1:$PORT/?kiosk=1"
else
  PORT="$PORT" exec "$VENV/bin/python" app.py
fi
