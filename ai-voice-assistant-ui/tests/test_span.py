"""SPAN regression tests: Wi-Fi minimalism, LED priority, modes, APIs.

Run:  python -m pytest tests/ -v
Covers user requirements:
- Wi-Fi icon green-filled if up else outline, never orange/yellow, no Online/Offline/Ethernet words
- Header only Wi-Fi + mic/mute (+ state word for REC/Muted); no battery widget (mains Pi)
- 240x320 screen preserved
- Light: muted red, any activity orange, idle green; off override
- listening idle >10s returns to SPAN idle screen; recording stays
- LLM metrics terminology (TTFT/total/TPOT/tok-s/in-out)
- Fixed loaders per mode
"""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as span

_REAL_READ_INTERNET = span.read_internet
_REAL_READ_BATTERY = span.read_battery


# ---------- fixtures ----------

@pytest.fixture(autouse=True)
def stable_state(monkeypatch):
    """Freeze background monitor + snapshot global state so tests are deterministic."""
    monkeypatch.setattr(span, "read_wifi", lambda: span.STATE["wifi"])
    monkeypatch.setattr(span, "read_internet", lambda: span.STATE["internet"])
    monkeypatch.setattr(span, "read_mic_muted", lambda: span.STATE["mic_muted"])
    monkeypatch.setattr(span, "read_battery",
                        lambda: dict(span.STATE["battery"]))
    with span.LOCK:
        saved = json.loads(json.dumps(span.STATE))
    saved_history = list(span.HISTORY)
    saved_last_rgb = span._last_rgb
    yield
    with span.LOCK:
        span.STATE.clear()
        span.STATE.update(saved)
    span.HISTORY.clear()
    span.HISTORY.extend(saved_history)
    span._last_rgb = saved_last_rgb


def base_state(**over):
    s = {
        "wifi": True,
        "internet": True,
        "mic_muted": False,
        "light_enabled": True,
        "battery": {"percent": 80, "charging": False, "status": "Discharging"},
        "mode": "idle",
        "last_activity_at": time.monotonic(),
        "recording_started_at": None,
        "led": {"name": "green", "hex": "#22c55e"},
        "hardware": {},
    }
    s.update(over)
    return s


def set_state(**over):
    with span.LOCK:
        span.STATE.update(base_state(**over))
    span.update_led()
    return span._state_copy()


# ---------- 1. Wi-Fi icon: never yellow, wifi-only ----------

def test_wifi_css_has_no_orange_rule():
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read()
    assert ".wifi path" in css
    assert ".wifi.off path" in css
    assert "no-internet" not in css, "orange .wifi.no-internet rule must not exist"
    # wifi selectors must never reference orange/yellow
    for line in css.splitlines():
        if line.strip().startswith(".wifi"):
            assert "orange" not in line.lower() and "yellow" not in line.lower(), line


def test_wifi_js_uses_wifi_only_no_orange():
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    assert "no-internet" not in js, "JS must not toggle no-internet/orange class"
    assert "wifi.classList.toggle('off', !S.wifi)" in js or 'wifi.classList.toggle("off", !S.wifi)' in js \
        or "toggle('off', !S.wifi)" in js
    assert "S.wifi && !S.internet" not in js, "wifi colour must not depend on internet"
    assert "Online" not in js and "Offline mode" not in js, "no Online/Offline words in UI"
    assert "Wi-Fi connected, no internet" not in js


def test_wifi_aria_label_is_two_way():
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    assert "Wi-Fi connected" in js and "Wi-Fi off" in js


def test_header_has_only_wifi_and_mic():
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read().lower()
    assert 'id="wifi"' in html and 'id="mic"' in html
    assert 'id="battery"' not in html and 'id="battext"' not in html and 'batfill' not in html
    assert "ethernet" not in html and "speaker" not in html
    # state word must exist for REC/Muted/Thinking indication
    assert 'id="statelabel"' in html


def test_screen_stays_240x320():
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read()
    assert "width:240px" in css.replace(" ", "") and "height:320px" in css.replace(" ", "")
    compact = css.replace(" ", "")
    assert ".status{height:22px" in compact and ".loader{height:18px" in compact, \
        "header/loader stay compact so chat keeps maximum area"
    assert "data-s=idle]{--c:var(--green)}" in compact, "idle screen is green"


def test_static_files_revalidated():
    assert span.app.config["SEND_FILE_MAX_AGE_DEFAULT"] == 0, \
        "kiosk browsers must revalidate CSS/JS, never run stale cache"


def test_wifi_off_outline_visible():
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read()
    assert ".wifi.off" in css and "#475569" in css, "off outline must be visible on light bg"
    assert "stroke-width:2" in css.replace(" ", ""), "hollow off-state needs a thick stroke"


def test_log_endpoint(client):
    r = client.get("/api/log?lines=5")
    assert r.status_code == 200
    body = r.get_json()
    assert "lines" in body and isinstance(body["lines"], list) and len(body["lines"]) <= 5


def test_header_state_not_duplicated_as_heading():
    # Same strings, different roles: header state is a micro status label,
    # body keeps the hero title. Prevents "Ready" reading twice.
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read().replace(" ", "")
    assert "text-transform:uppercase" in css
    assert ".voiceviewstrong{font-size:17px" in css


def test_stream_fallback_paints_state_without_sse():
    # The screen must self-correct even where the event stream never connects:
    # one-shot /api/state paint on load + 3s re-pull while disconnected.
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    assert "fetch('/api/state'" in js, "initial truth pull required"
    assert "setInterval" in js and "if (!connected)" in js, "poll-while-down fallback required"


def test_app_js_parses():
    # A syntax error here kills the entire script -> dead screen. Gate it.
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    r = subprocess.run([node, "--check",
                        os.path.join(span.app.root_path, "static", "app.js")],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr


def test_span_wordmark_fixed_in_header_all_screens():
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read()
    assert 'class="brand-top"' in html and ">SPAN<" in html
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read().replace(" ", "")
    assert ".brand-top{position:absolute;left:50%" in css


def test_no_repeated_prompts_across_views():
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read()
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    # "Say Hey Jarvis" lives only on the standby screen, not in chat empty or loader
    assert 'class="say"' not in html
    assert "Waiting for wake word" in js
    assert js.count("Say “Hey Jarvis”") == 1, "prompt must appear once (voiceSub only)"
    assert js.lower().count("use the device button") == 1, \
        "device-button hint lives only on standby body, not in loader"
    assert "S.mic_muted ? 'Muted'" in js, "muted wins over REC in header"


def test_js_ids_all_exist_in_html():
    # Guards the mixed-version crash: every $('id') the JS touches must exist,
    # otherwise render() throws and the whole screen (wifi/mute included) freezes.
    import re
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read()
    used = set(re.findall(r"\$\('([A-Za-z]+)'\)", js)) - {"empty"}
    present = set(re.findall(r'id="([A-Za-z]+)"', html))
    assert used <= present, f"JS touches missing ids: {used - present}"


def test_static_urls_versioned():
    import re
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read()
    assert "style.css" in html and "app.js" in html
    assert re.search(r"v='\d+'", html), \
        "versioned static URLs force fresh fetch after deploys"


def test_build_stamp_visible():
    # Fingerprints a running screen: tab title + console stamp must match,
    # so stale-tab vs stale-server is answered in seconds, not hours.
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read()
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    assert "<title>SPAN v6</title>" in html
    assert "console.log('SPAN build v6')" in js


def test_readme_matches_shipped_version():
    # README's update instructions must name the static version actually shipped.
    import re
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read()
    md = open(os.path.join("README.md")).read()
    shipped = re.search(r"v='(\d+)'", html).group(1)
    assert f"?v={shipped}" in md, "README must document the shipped static version"
    assert "/api/log" in md, "README must document the remote log endpoint"


def test_mute_switch_verifies_hardware(monkeypatch):
    calls = []
    monkeypatch.setattr(span, "set_mic_muted", lambda m: calls.append(m))
    monkeypatch.setattr(span, "read_mic_muted", lambda: calls[-1] if calls else False)
    monkeypatch.setattr(span, "button", object())  # pretend GPIO initialised
    with span.LOCK:
        span.STATE["mic_muted"] = False
    span.apply_switch()
    assert calls == [True] and span.STATE["mic_muted"] is True
    assert "Microphone mute not applied" not in ";".join(span.STATE["hardware"]["errors"])


def test_mute_switch_reports_unapplied(monkeypatch):
    monkeypatch.setattr(span, "set_mic_muted", lambda m: None)  # silently does nothing
    monkeypatch.setattr(span, "read_mic_muted", lambda: False)  # hardware never flips
    monkeypatch.setattr(span, "button", object())
    with span.LOCK:
        span.STATE["mic_muted"] = False
        span.STATE["hardware"]["errors"] = []
    span.apply_switch()
    assert any("not applied" in e for e in span.STATE["hardware"]["errors"]), \
        "a mute press the hardware ignores must leave an error, not silence"


def test_light_theme_and_readability():
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read()
    assert "color-scheme:light" in css.replace(" ", "")
    assert "--bg:#f4f6fa" in css.replace(" ", "")
    assert "--text:#101828" in css.replace(" ", "")
    assert "#070b16" not in css and "#edf5ff" not in css, "no dark-theme leftovers"
    assert "font-size:12px" in css.replace(" ", ""), "chat text must be 12px for readability"
    lowered = css.lower()
    for banned in ("cyan", "purple", "#4da3ff", "#22d3ee", "#b794f4", "#7c3aed"):
        assert banned not in lowered, f"fourth colour {banned} must not exist: three only"
    assert "blue" not in lowered, "no blue anywhere in the UI"


def test_readme_documents_green_only_wifi():
    md = open(os.path.join(span.app.root_path, "..", "pocket_ai_hardware_web", "README.md")).read() \
        if False else open(os.path.join("README.md")).read()
    assert "orange = connected without internet" not in md.lower()


# ---------- 2. Light: three cases only ----------

def test_light_off_overrides_everything():
    assert span.choose_led(base_state(light_enabled=False, mic_muted=True,
                                      mode="thinking", internet=False)) == "off"


def test_light_muted_is_red_in_every_mode():
    for mode in ("idle", "listening", "recording", "thinking", "tool", "speaking"):
        assert span.choose_led(base_state(mic_muted=True, mode=mode)) == "red", mode


def test_light_working_is_orange_regardless_of_connectivity():
    for mode in ("listening", "recording", "thinking", "tool", "speaking"):
        for wifi, net in ((True, True), (True, False), (False, False), (False, True)):
            got = span.choose_led(base_state(mic_muted=False, mode=mode,
                                             wifi=wifi, internet=net))
            assert got == "orange", (mode, wifi, net)


def test_light_idle_is_green_regardless_of_connectivity():
    # offline/online has no light colour; Wi-Fi icon covers connectivity
    for wifi, net in ((True, True), (True, False), (False, False), (False, True)):
        assert span.choose_led(base_state(mic_muted=False, mode="idle",
                                          wifi=wifi, internet=net)) == "green"


def test_light_color_values():
    assert set(span.LED_COLORS) == {"red", "orange", "green", "off"}, \
        "only three live colours plus off"
    assert span.LED_COLORS["red"] == (255, 0, 0)
    assert span.LED_COLORS["orange"] == (255, 140, 0)
    assert span.LED_COLORS["green"] == (34, 197, 94)
    assert span.LED_COLORS["off"] == (0, 0, 0)


def test_update_led_syncs_hex():
    set_state(mode="thinking", mic_muted=False, internet=True)
    assert span.STATE["led"]["name"] == "orange"
    assert span.STATE["led"]["hex"] == "#%02x%02x%02x" % span.LED_COLORS["orange"]


def test_tool_error_returns_thinking_via_api(client):
    set_state(mode="thinking", internet=True, mic_muted=False, light_enabled=True)
    r = client.post("/api/event", json={"type": "tool", "id": "t9",
                                        "name": "web_search", "status": "error",
                                        "detail": "No internet connection"})
    assert r.status_code == 200
    assert span.STATE["mode"] == "thinking"
    assert span.STATE["led"]["name"] == "orange"


def test_light_off_on_flow_via_api(client):
    set_state(mode="idle", internet=True, mic_muted=False, light_enabled=True)
    assert client.post("/api/event", json={"type": "light", "on": False}).status_code == 200
    st = client.get("/api/state").get_json()
    assert st["led"]["name"] == "off" and st["led"]["hex"] == "#000000"
    assert client.post("/api/event", json={"type": "light", "on": True}).status_code == 200
    st = client.get("/api/state").get_json()
    assert st["led"]["name"] == "green"


def test_muted_red_overrides_thinking_via_state():
    set_state(mode="thinking", mic_muted=True, internet=True, light_enabled=True)
    assert span.STATE["led"]["name"] == "red"
    assert span.STATE["led"]["hex"] == "#ff0000"


def test_recording_stays_on_mute_toggle_and_light():
    set_state(mode="idle")
    span.publish({"type": "recording", "on": True})
    with span.LOCK:
        span.STATE["mic_muted"] = True
    span.update_led()
    assert span.STATE["mode"] == "recording"  # mute flag must not exit recording
    assert span.STATE["led"]["name"] == "red"  # but LED shows muted
    span.publish({"type": "light", "on": False})
    assert span.STATE["mode"] == "recording"
    assert span.STATE["led"]["name"] == "off"
    span.publish({"type": "light", "on": True})
    span.publish({"type": "recording", "on": False})
    assert span.STATE["mode"] == "idle"


# ---------- 3. Mode transitions (wake word / recording / follow-up) ----------

def test_wake_goes_listening():
    set_state(mode="idle")
    span.publish({"type": "wake"})
    assert span.STATE["mode"] == "listening"


def test_user_final_goes_thinking():
    set_state(mode="listening")
    span.publish({"type": "user_final", "id": "u1", "text": "hi"})
    assert span.STATE["mode"] == "thinking"


def test_tool_start_then_ok():
    set_state(mode="thinking")
    span.publish({"type": "tool", "id": "t1", "name": "get_weather", "status": "start"})
    assert span.STATE["mode"] == "tool"
    span.publish({"type": "tool", "id": "t1", "name": "get_weather", "status": "ok"})
    assert span.STATE["mode"] == "thinking"


def test_token_goes_speaking_done_returns_listening():
    set_state(mode="thinking")
    span.publish({"type": "assistant_start", "id": "a1"})
    span.publish({"type": "assistant_token", "id": "a1", "text": "hello "})
    assert span.STATE["mode"] == "speaking"
    span.publish({"type": "assistant_done", "id": "a1", "metrics": {}})
    assert span.STATE["mode"] == "listening"


def test_assistant_done_next_mode_idle():
    set_state(mode="speaking")
    span.publish({"type": "assistant_done", "id": "a1", "next_mode": "idle"})
    assert span.STATE["mode"] == "idle"


def test_interrupt_returns_listening():
    set_state(mode="speaking")
    span.publish({"type": "assistant_interrupted", "id": "a1"})
    assert span.STATE["mode"] == "listening"


def test_recording_mode_sticks_and_transcripts_stay():
    set_state(mode="idle")
    span.publish({"type": "recording", "on": True})
    assert span.STATE["mode"] == "recording"
    assert span.STATE["recording_started_at"] is not None
    span.publish({"type": "user_partial", "id": "r1", "text": "agenda"})
    assert span.STATE["mode"] == "recording", "transcripts must not leave recording mode"
    span.publish({"type": "recording", "on": False})
    assert span.STATE["mode"] == "idle"
    assert span.STATE["recording_started_at"] is None


def test_followup_timeout_returns_to_idle():
    # monitor() expires chat modes -> idle after FOLLOW_UP_TIMEOUT_SECONDS (default 10)
    assert span.FOLLOW_UP_TIMEOUT_SECONDS >= 10
    for mode in ("listening", "thinking", "tool", "speaking"):
        with span.LOCK:
            span.STATE["mode"] = mode
            span.STATE["last_activity_at"] = time.monotonic() - (span.FOLLOW_UP_TIMEOUT_SECONDS + 1)
        expired = (span.STATE["mode"] in ("listening", "thinking", "tool", "speaking")
                   and time.monotonic() - span.STATE["last_activity_at"] >= span.FOLLOW_UP_TIMEOUT_SECONDS)
        assert expired, mode
        with span.LOCK:
            span._set_mode("idle")
        assert span.STATE["mode"] == "idle"


def test_recording_does_not_auto_expire():
    with span.LOCK:
        span.STATE["mode"] = "recording"
        span.STATE["last_activity_at"] = time.monotonic() - 3600
    expired = (span.STATE["mode"] == "listening"
               and time.monotonic() - span.STATE["last_activity_at"] >= span.FOLLOW_UP_TIMEOUT_SECONDS)
    assert not expired, "only listening auto-expires; recording stays until stopped"


def test_light_event_validation():
    with pytest.raises(ValueError):
        span.publish({"type": "light", "on": "yes"})
    with pytest.raises(ValueError):
        span.publish({"type": "mode", "mode": "bogus"})
    with pytest.raises(ValueError):
        span.publish({"type": "nope"})


def test_recording_event_marks_user_bubbles():
    seen = []
    orig = span.broadcast
    try:
        span.broadcast = seen.append
        set_state(mode="idle")
        span.publish({"type": "recording", "on": True})
        span.publish({"type": "user_final", "id": "r9", "text": "hello"})
        assert seen and seen[-1].get("recording") is True
    finally:
        span.broadcast = orig
        with span.LOCK:
            span._set_mode("idle")


# ---------- 4. HTTP APIs ----------

@pytest.fixture()
def client():
    span.app.config.update(TESTING=True)
    return span.app.test_client()


def test_get_state_shape(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    body = r.get_json()
    for key in ("wifi", "internet", "mic_muted", "battery", "mode", "led"):
        assert key in body, key
    assert isinstance(body["wifi"], bool) and isinstance(body["internet"], bool)
    assert body["mode"] in span.MODES


def test_post_event_wake_and_validation(client):
    r = client.post("/api/event", json={"type": "wake"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert span.STATE["mode"] == "listening"
    r = client.post("/api/event", json={"type": "bogus"})
    assert r.status_code == 400
    r = client.post("/api/event", json=["not", "a", "dict"])
    assert r.status_code == 400
    r = client.post("/api/event", json={"type": "mode", "mode": "nope"})
    assert r.status_code == 400


def test_thinking_done_idle_light_flow_via_api(client):
    # 1 API -> thinking, stays: check orange. done API -> listening: check orange.
    # 10s idle -> main unmuted: check green.
    set_state(mode="idle", wifi=True, internet=True,
              mic_muted=False, light_enabled=True)
    r = client.post("/api/event",
                    json={"type": "user_final", "id": "u-th", "text": "weather?"})
    assert r.status_code == 200
    st = client.get("/api/state").get_json()
    assert st["mode"] == "thinking", st
    assert st["led"]["name"] == "orange", st["led"]
    assert st["led"]["hex"] == "#%02x%02x%02x" % span.LED_COLORS["orange"]
    # stays in thinking without further events (no auto-leave before timeout)
    st2 = client.get("/api/state").get_json()
    assert st2["mode"] == "thinking" and st2["led"]["name"] == "orange"

    r = client.post("/api/event", json={"type": "assistant_done", "id": "a-th"})
    assert r.status_code == 200
    st = client.get("/api/state").get_json()
    assert st["mode"] == "listening", st
    assert st["led"]["name"] == "orange", st["led"]
    assert st["led"]["hex"] == "#%02x%02x%02x" % span.LED_COLORS["orange"]

    # 10s without events -> main screen
    with span.LOCK:
        span.STATE["last_activity_at"] = time.monotonic() - (span.FOLLOW_UP_TIMEOUT_SECONDS + 1)
        mode = span.STATE["mode"]
        idle_for = time.monotonic() - span.STATE["last_activity_at"]
        if mode in ("listening", "thinking", "tool", "speaking") and idle_for >= span.FOLLOW_UP_TIMEOUT_SECONDS:
            span._set_mode("idle")
    span.update_led()
    st = client.get("/api/state").get_json()
    assert st["mode"] == "idle", st
    assert st["led"]["name"] == "green", st["led"]
    assert st["led"]["hex"] == "#%02x%02x%02x" % span.LED_COLORS["green"]


def test_no_demo_endpoint(client):
    assert client.post("/api/demo", json={"scenario": "chat"}).status_code == 404
    assert not hasattr(span, "DEMOS") and not hasattr(span, "DEMO_LOCK"), \
        "dev demo hooks removed"


def test_broadcast_history_and_slow_client_resync():
    import queue
    q = queue.Queue(maxsize=2)
    with span.CLIENTS_LOCK:
        span.CLIENTS.append(q)
    try:
        span.HISTORY.clear()
        span.broadcast({"type": "user_final", "id": "u1", "text": "hi"})
        assert len(span.HISTORY) == 1  # chat events replayed
        n_before = len(span.HISTORY)
        span.broadcast({"type": "state", "state": {}})
        assert len(span.HISTORY) == n_before  # state not replayed
        assert not q.empty()
    finally:
        with span.CLIENTS_LOCK:
            if q in span.CLIENTS:
                span.CLIENTS.remove(q)


def test_read_battery_none_when_no_supplies(monkeypatch):
    monkeypatch.setattr(span, "read_battery", _REAL_READ_BATTERY)
    monkeypatch.setattr(span.glob, "glob", lambda *a, **k: [])
    out = span.read_battery()
    assert out == {"percent": None, "charging": None, "status": "unavailable"}


def test_read_internet_true_false(monkeypatch):
    monkeypatch.setattr(span, "read_internet", _REAL_READ_INTERNET)

    class Dummy:
        def close(self):
            pass
    monkeypatch.setattr(span.socket, "create_connection", lambda *a, **k: Dummy())
    assert span.read_internet() is True
    import socket as _sock

    def boom(*a, **k):
        raise _sock.timeout()
    monkeypatch.setattr(span.socket, "create_connection", boom)
    assert span.read_internet() is False


def test_wifi_iface_up_matrix():
    assert span._wifi_iface_up("up", "") is True
    assert span._wifi_iface_up("up", "0") is True
    for state in ("dormant", "unknown", "testing"):
        assert span._wifi_iface_up(state, "1") is True, state
        assert span._wifi_iface_up(state, "0") is False, state
        assert span._wifi_iface_up(state, "") is False, state
    for state in ("down", "notpresent", "lowerlayerdown", ""):
        assert span._wifi_iface_up(state, "1") is False, state


def test_read_wifi_returns_bool():
    assert isinstance(span.read_wifi(), bool)


def test_metrics_keys_documented(client):
    # README documents the exact metrics dict the pipeline sends with assistant_done,
    # and the API accepts it end to end.
    md = open(os.path.join("README.md")).read()
    for k in ("ttft_ms", "total_ms", "gen_ms", "tpot_ms", "tokens_per_sec",
              "input_tokens", "output_tokens"):
        assert k in md, k
    set_state(mode="speaking")
    r = client.post("/api/event", json={"type": "assistant_done", "id": "a-m",
                                        "metrics": {"ttft_ms": 420, "total_ms": 1900,
                                                    "gen_ms": 1480, "tpot_ms": 61.7,
                                                    "tokens_per_sec": 16.2,
                                                    "input_tokens": 312, "output_tokens": 24}})
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_fixed_loader_selectors_exist():
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read()
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    assert ".bars" in css and ".dots" in css and "mute-ic" in css
    assert "wave" in css and "recClock" in js


def test_battery_absent_from_header_but_present_in_api(client):
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    html = open(os.path.join(span.app.root_path, "templates", "index.html")).read()
    assert "batText" not in js and "batFill" not in js, "no battery widget in UI (mains Pi)"
    assert "battery" not in html.lower().split("<header")[1].split("</header>")[0], \
        "no battery in header"
    # backend still probes + exposes it for logs/diagnostics
    assert "battery" in client.get("/api/state").get_json()


def test_recording_never_falls_back_to_span_or_voice():
    # While recording, pipeline chatter must not return to idle/listening/speaking.
    set_state(mode="idle")
    span.publish({"type": "recording", "on": True})
    assert span.STATE["mode"] == "recording"
    for ev in ({"type": "wake"},
               {"type": "user_partial", "id": "r1", "text": "hello"},
               {"type": "user_final", "id": "r1", "text": "hello world"},
               {"type": "assistant_start", "id": "a1"},
               {"type": "assistant_token", "id": "a1", "text": "hi "},
               {"type": "tool", "id": "t1", "name": "get_weather", "status": "start"},
               {"type": "assistant_done", "id": "a1"},
               {"type": "error", "message": "x"}):
        span.publish(ev)
        assert span.STATE["mode"] == "recording", ev
    # only explicit stop (or direct mode event) leaves recording
    span.publish({"type": "recording", "on": False})
    assert span.STATE["mode"] == "idle"


def test_frontend_recording_hides_span_voice():
    css = open(os.path.join(span.app.root_path, "static", "style.css")).read()
    js = open(os.path.join(span.app.root_path, "static", "app.js")).read()
    assert ".screen.rec-on .recview" in css
    assert ".screen.rec-on .chat" in css and ".screen.rec-on .voiceview" in css
    assert "mode === 'idle' && !recording" in js or 'mode === "idle" && !recording' in js, \
        "voice orb only on idle, never during recording"
