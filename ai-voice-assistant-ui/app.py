"""SPAN — browser replica of the 240x320 device screen.

One background monitor thread reads hardware state (Wi-Fi, internet, mic mute,
battery) and pushes changes to browsers over Server-Sent Events. The voice
pipeline (wake word, VAD, STT, SLM, TTS) reports progress with POST /api/event
or, in-process, with publish(event).
"""
from collections import deque
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import glob
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import queue
import shutil
import socket
import subprocess
import threading
import time

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # LAN kiosk: always revalidate tiny static files
LOCK = threading.RLock()


def configure_logger():
    """Create a small local log that can be shared for hardware diagnosis."""
    logger = logging.getLogger("span")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    log_path = os.environ.get("SPAN_LOG_FILE", os.path.join(app.root_path, "logs", "span.log"))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        handler = RotatingFileHandler(log_path, maxBytes=256 * 1024, backupCount=3, encoding="utf-8")
    except OSError:
        # The screen must still work if a read-only filesystem prevents logging.
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOG = configure_logger()

# Momentary push button on GPIO 5 toggles microphone mute on each press.
MIC = "@DEFAULT_AUDIO_SOURCE@"
# Listening automatically returns to the ready screen after this quiet period.
# Thinking / tool / speaking from a stalled pipeline also fall back so the
# chat never sticks. Recording stays active until the pipeline explicitly stops it.
FOLLOW_UP_TIMEOUT_SECONDS = max(1, int(os.environ.get("FOLLOW_UP_TIMEOUT_SECONDS", "10")))

# Single WS2812 pixel. Exactly three live cases: muted red, any activity orange,
# idle green. Connectivity is shown only by the Wi-Fi icon, never the light.
LED_COLORS = {
    "red": (255, 0, 0),        # mic muted
    "orange": (255, 140, 0),    # listening / recording / thinking / tool / speaking
    "green": (34, 197, 94),    # idle with mic on
    "off": (0, 0, 0),          # explicit light-off test or hardware setting
}
MODES = ("idle", "listening", "recording", "thinking", "reasoning", "tool", "speaking")

STATE = {
    "wifi": False,
    "internet": False,
    "mic_muted": False,
    "light_enabled": True,
    "battery": {"percent": None, "charging": None, "status": "unavailable"},
    "mode": "idle",
    "last_activity_at": time.monotonic(),
    "recording_started_at": None,
    "led": {"name": "green", "hex": "#22c55e"},
    "hardware": {"gpio_button": False, "ws2812": False,
                 "audio_control": bool(shutil.which("wpctl")), "battery": False, "errors": []},
}
button = None
pixels = None
_last_rgb = None


def add_error(msg):
    LOG.warning("%s", msg)
    with LOCK:
        STATE["hardware"]["errors"] = (STATE["hardware"]["errors"] + [msg])[-5:]


# ---------- SSE fan-out ----------

CLIENTS = []
CLIENTS_LOCK = threading.Lock()
# Recent chat events, replayed to a browser that connects or reloads mid-conversation.
HISTORY = deque(maxlen=600)


def _state_copy():
    with LOCK:
        return json.loads(json.dumps(STATE))


def broadcast(event):
    data = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
    with CLIENTS_LOCK:
        if event.get("type") != "state":
            HISTORY.append(data)
        clients = list(CLIENTS)
    for q in clients:
        try:
            q.put_nowait(data)
        except queue.Full:
            # Slow client: drop its backlog and resync it with the full state.
            with q.mutex:
                q.queue.clear()
            q.put_nowait("data: " + json.dumps({"type": "state", "state": _state_copy()}) + "\n\n")


def push_state():
    broadcast({"type": "state", "state": _state_copy()})


# ---------- hardware ----------

def run_cmd(args, timeout=2):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)


def read_mic_muted():
    if not shutil.which("wpctl"):
        return None
    try:
        return "[MUTED]" in run_cmd(["wpctl", "get-volume", MIC]).stdout
    except Exception:
        return None


def set_mic_muted(mute):
    if not shutil.which("wpctl"):
        raise RuntimeError("wpctl is not installed or not on PATH")
    run_cmd(["wpctl", "set-mute", MIC, "1" if mute else "0"])


def _wifi_iface_up(operstate, carrier):
    """Kernel interface state counts as Wi-Fi up. operstate alone is unreliable
    for wireless (hotspot links often report dormant/unknown), so a live
    carrier confirms the association."""
    operstate = (operstate or "").strip()
    carrier = (carrier or "").strip()
    if operstate == "up":
        return True
    return operstate in ("dormant", "unknown", "testing") and carrier == "1"


_last_wifi_result = None


def read_wifi():
    # Prefer NetworkManager when present; fall back to kernel interface state
    # so Raspberry Pi OS installs without NetworkManager still report Wi-Fi.
    # Probe detail is logged on every transition, so /api/log shows exactly
    # why the icon reads on or off (nmcli rows, operstate/carrier, iw link).
    global _last_wifi_result
    notes = []
    result = False
    if shutil.which("nmcli"):
        try:
            out = run_cmd(["nmcli", "-t", "-f", "TYPE,STATE", "device", "status"]).stdout
            notes.append("nmcli=" + "|".join(out.splitlines()))
            if any(line == "wifi:connected" for line in out.splitlines()):
                result = True
        except Exception as exc:
            notes.append(f"nmcli=error({exc})")
    for path in sorted(glob.glob("/sys/class/net/*")):
        name = os.path.basename(path)
        if not os.path.exists(os.path.join(path, "wireless")):
            continue
        try:
            with open(os.path.join(path, "operstate")) as f:
                operstate = f.read().strip()
            carrier = ""
            try:
                with open(os.path.join(path, "carrier")) as f:
                    carrier = f.read().strip()
            except OSError:
                pass
            notes.append(f"{name}:{operstate}/carrier={carrier}")
            if _wifi_iface_up(operstate, carrier):
                result = True
        except OSError:
            continue
    if not result and shutil.which("iw"):
        for path in sorted(glob.glob("/sys/class/net/*")):
            name = os.path.basename(path)
            if not os.path.exists(os.path.join(path, "wireless")):
                continue
            try:
                linked = "Connected to " in run_cmd(["iw", "dev", name, "link"]).stdout
                notes.append(f"iw-{name}={'linked' if linked else 'down'}")
                if linked:
                    result = True
                    break
            except Exception:
                continue
    if not result and shutil.which("iwgetid"):
        try:
            has = bool(run_cmd(["iwgetid", "-r"]).stdout.strip())
            notes.append(f"iwgetid={'up' if has else 'down'}")
            result = has
        except Exception:
            pass
    if result != _last_wifi_result:
        LOG.info("wifi_probe result=%s %s", result, " ".join(notes) or "no-data")
        _last_wifi_result = result
    return result


def read_internet():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1).close()
        return True
    except OSError:
        return False


def read_battery():
    """UPS HATs that expose a kernel power_supply driver show up here. A bare Pi 5 has none."""
    observed = []
    for path in glob.glob("/sys/class/power_supply/*"):
        name = os.path.basename(path)
        try:
            with open(os.path.join(path, "type")) as f:
                supply_type = f.read().strip()
                if supply_type != "Battery":
                    observed.append(f"{name}: type={supply_type}")
                    continue
            with open(os.path.join(path, "capacity")) as f:
                percent = max(0, min(100, int(f.read().strip())))
            charging = None
            status = "unknown"
            status_file = os.path.join(path, "status")
            if os.path.exists(status_file):
                with open(status_file) as f:
                    status = f.read().strip()
                    charging = status == "Charging"
            result = {"percent": percent, "charging": charging, "status": status}
            observed.append(f"{name}: type=Battery capacity={percent} status={status}")
            LOG.info("battery_probe result=%s supplies=%s", result, "; ".join(observed))
            return result
        except (OSError, ValueError) as exc:
            observed.append(f"{name}: unreadable ({exc})")
            continue
    result = {"percent": None, "charging": None, "status": "unavailable"}
    if observed:
        LOG.warning("battery_probe result=%s supplies=%s", result, "; ".join(observed))
    else:
        LOG.debug("battery_probe result=%s supplies=none (mains-powered)", result)
    return result


def choose_led(s):
    # Three cases only, top to bottom: light off, muted red,
    # any non-idle mode orange, idle green. No cyan/green flicker,
    # no connectivity colour (Wi-Fi icon covers that).
    if not s["light_enabled"]:
        name = "off"
    elif s["mic_muted"]:
        name = "red"
    elif s["mode"] != "idle":
        name = "orange"
    else:
        name = "green"
    return name


def update_led():
    """Recompute the LED; write the pixel only when the colour changes."""
    global _last_rgb
    with LOCK:
        name = choose_led(STATE)
        rgb = LED_COLORS[name]
        STATE["led"] = {"name": name, "hex": "#%02x%02x%02x" % rgb}
        mode = STATE["mode"]
        mic_muted = STATE["mic_muted"]
        wifi = STATE["wifi"]
        internet = STATE["internet"]
        light_enabled = STATE["light_enabled"]
    changed = rgb != _last_rgb
    if pixels is not None and changed:
        try:
            pixels[0] = rgb
            _last_rgb = rgb
        except Exception as exc:
            add_error(f"WS2812 write: {exc}")
    elif changed:
        _last_rgb = rgb
    if changed:
        LOG.info("led_state name=%s rgb=%s mode=%s mic_muted=%s wifi=%s internet=%s enabled=%s",
                 name, rgb, mode, mic_muted, wifi, internet, light_enabled)


def apply_switch():
    if button is None:
        add_error("GPIO 5 switch pressed but button is not initialised")
        return
    with LOCK:
        mute = not STATE["mic_muted"]
    try:
        set_mic_muted(mute)
        verify = read_mic_muted()
        if verify is not None and verify != mute:
            add_error(f"Microphone mute not applied: wanted {mute}, hardware reports {verify}")
    except Exception as exc:
        add_error(f"Microphone mute: {exc}")
    with LOCK:
        STATE["mic_muted"] = mute
    LOG.info("mic_mute_toggle muted=%s", mute)
    update_led()
    push_state()


def sync_mic_state():
    muted = read_mic_muted()
    if muted is not None:
        with LOCK:
            STATE["mic_muted"] = muted


def init_hardware():
    global button, pixels
    try:
        from gpiozero import Button
        button = Button(5, pull_up=False, bounce_time=0.2)
        button.when_pressed = apply_switch
        STATE["hardware"]["gpio_button"] = True
        LOG.info("gpio_button pin=5 pull_up=False bounce_time=0.2s")
    except Exception as exc:
        button = None
        add_error(f"GPIO 5 switch init: {exc}")
    try:
        import board
        import neopixel
        pixels = neopixel.NeoPixel(board.D13, 1, brightness=0.3, auto_write=True)
        STATE["hardware"]["ws2812"] = True
    except Exception as exc:
        pixels = None
        add_error(f"WS2812 board.D13 init: {exc}")
    sync_mic_state()
    update_led()
    LOG.info("hardware_initialized gpio_button=%s ws2812=%s audio_control=%s",
             STATE["hardware"]["gpio_button"], STATE["hardware"]["ws2812"],
             STATE["hardware"]["audio_control"])


def monitor():
    """Single poller for all slow system state. Pushes only on change."""
    tick = 0
    while True:
        changes = {}
        if tick % 3 == 0:
            changes["wifi"] = read_wifi()
        if tick % 5 == 0:
            changes["internet"] = read_internet()
        if tick % 2 == 0:
            muted = read_mic_muted()
            if muted is not None:
                changes["mic_muted"] = muted
        if tick % 15 == 0:
            changes["battery"] = read_battery()
        with LOCK:
            if "battery" in changes:
                STATE["hardware"]["battery"] = changes["battery"]["percent"] is not None
            if "mic_muted" in changes and STATE["mic_muted"] != changes["mic_muted"]:
                LOG.info("mic_muted_poll muted=%s (hardware overrode UI state)",
                         changes["mic_muted"])
            changed = any(STATE[k] != v for k, v in changes.items())
            STATE.update(changes)
            follow_up_expired = (
                STATE["mode"] in ("listening", "thinking", "tool", "speaking")
                and time.monotonic() - STATE["last_activity_at"] >= FOLLOW_UP_TIMEOUT_SECONDS
            )
            if follow_up_expired:
                _set_mode("idle")
        if changed:
            update_led()
            push_state()
        elif follow_up_expired:
            update_led()
            push_state()
        tick += 1
        time.sleep(1)


# ---------- events from the voice pipeline ----------

EVENT_TYPES = {"wake", "user_partial", "user_final", "assistant_start", "assistant_token",
               "assistant_done", "assistant_interrupted", "tool", "error", "mode", "recording", "light", "reasoning"}


def _set_mode(mode):
    STATE["mode"] = mode
    STATE["last_activity_at"] = time.monotonic()
    if mode == "recording":
        STATE["recording_started_at"] = STATE["recording_started_at"] or time.time()
    else:
        STATE["recording_started_at"] = None


def publish(event):
    """Apply an event to device state and fan it out to every browser."""
    etype = event.get("type")
    if etype not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {etype!r}")
    with LOCK:
        before = STATE["mode"]
        before_light_enabled = STATE["light_enabled"]
        recording = before == "recording"
        if etype == "light":
            if not isinstance(event.get("on"), bool):
                raise ValueError("light event requires boolean on")
            STATE["light_enabled"] = event["on"]
        elif etype == "mode":
            if event.get("mode") not in MODES:
                raise ValueError("mode must be one of " + ", ".join(MODES))
            _set_mode(event["mode"])
        elif etype == "recording":
            _set_mode("recording" if event.get("on") else "idle")
        elif etype == "reasoning":
            _set_mode("reasoning")
        elif recording:
            pass  # transcripts stream into the recording; mode stays until stopped
        elif etype in ("wake", "user_partial", "assistant_interrupted"):
            _set_mode("listening")
        elif etype in ("user_final", "assistant_start"):
            _set_mode("thinking")
        elif etype == "tool":
            _set_mode("tool" if event.get("status") == "start" else "thinking")
        elif etype == "assistant_token":
            _set_mode("speaking")
        elif etype == "assistant_done":
            # Keep the wake-word conversation screen active for follow-ups.
            nxt = event.get("next_mode", "listening")
            _set_mode(nxt if nxt in MODES else "idle")
        mode_changed = STATE["mode"] != before
        light_changed = STATE["light_enabled"] != before_light_enabled
        if mode_changed:
            LOG.info("mode_change %s -> %s (%s)", before, STATE["mode"], etype)
    if etype not in ("mode", "recording", "light"):
        if recording and etype.startswith("user_"):
            event = dict(event, recording=True)
        broadcast(event)
    if mode_changed or light_changed or etype in ("mode", "recording", "light"):
        update_led()
        push_state()


# ---------- routes ----------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/state")
def get_state():
    return jsonify(_state_copy())


@app.get("/api/log")
def get_log():
    """Tail the hardware log for remote diagnosis (no SSH needed)."""
    try:
        count = max(1, min(500, int(request.args.get("lines", "100"))))
    except ValueError:
        count = 100
    path = os.environ.get("SPAN_LOG_FILE", os.path.join(app.root_path, "logs", "span.log"))
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-count:]
    except OSError:
        lines = []
    return jsonify({"lines": [line.rstrip("\n") for line in lines]})


@app.get("/api/events")
def events():
    q = queue.Queue(maxsize=256)
    with CLIENTS_LOCK:
        backlog = list(HISTORY)
        CLIENTS.append(q)

    @stream_with_context
    def gen():
        try:
            yield "retry: 2000\n" + "data: " + json.dumps({"type": "state", "state": _state_copy()}) + "\n\n"
            if backlog:
                yield "".join(backlog)
            while True:
                try:
                    yield q.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with CLIENTS_LOCK:
                if q in CLIENTS:
                    CLIENTS.remove(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/event")
def post_event():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "JSON object required"}), 400
    try:
        publish(data)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True})


init_hardware()
threading.Thread(target=monitor, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
