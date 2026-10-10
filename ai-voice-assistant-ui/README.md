# SPAN — browser replica of the 240x320 device screen

A Flask app that serves three static files (HTML, CSS, vanilla JS). No Node, no build step, no
frameworks, no external fonts or CDNs, so it works offline. Chromium on the Pi (or any phone/laptop on
the same network) renders the 240x320 screen scaled to fit, inside a device frame.
Clean light theme, 12px chat text, three light colours only (red / orange / green).

## Run

```bash
source /path/to/aivenv/bin/activate
python -m pip install -r requirements.txt          # core (Flask) + tests, all machines
python -m pip install -r requirements-pi.txt       # Pi 5 only: GPIO button + WS2812 (optional)
python app.py                # PORT=5055 python app.py to change the port
```

- Device frame view: `http://<PI-IP>:5000/`
- Bare screen for the Pi's own display (Chromium kiosk): `http://127.0.0.1:5000/?kiosk=1`
- `?scale=1` renders at true 1x size.

Stop any old TFT script first so it doesn't compete for GPIO 5 or the WS2812 (`pgrep -af final.py`).
Run as your desktop user, not with sudo, so `wpctl` reaches your PipeWire session.

After every update on the Pi: `git pull`, restart `app.py`, close all Pi tabs and open one
fresh tab — its title must read `SPAN v6`, or it runs stale files. Static files are versioned
(`?v=6`) plus `no-cache`, so a fresh tab always loads the matching HTML/JS/CSS set — never
debug visuals on a stale copy. If mute presses do nothing, check `GET /api/log`: `mic_mute_toggle`
(press seen), `Microphone mute not applied` (hardware ignored it), `mic_muted_poll` (2s hardware
poll reverted the UI), or a `GPIO 5 switch init` error (button not initialised).

## Screen

- Header: state micro-label on the left (Ready / Listening / ● REC 00:42 / Thinking / Using tool / Responding / Muted),
  SPAN wordmark centred (same spot on every screen), Wi-Fi + mic icons right.
  The right side has Wi-Fi (filled green = connected, empty outline = off) and
  a read-only mic status icon (green, or red with a slash when muted). No battery widget:
  the Pi is mains-powered. The physical GPIO button is the only mute control.
- Body, idle: voice standby screen. A wake-word or local pipeline event opens the full-height chat. Under each reply:
  `TTFT · total · tok/s` and `in · out · avg ms/tok` (hover for the full names).
- Body, recording mode: big waveform, timer and live transcript card. When recording stops, the
  transcript is kept in the chat as a REC bubble.
- Fixed loader row at the bottom: waveform (listening / responding), dots (thinking / tool), mic-off (muted).
  Idle shows `Waiting for wake word`; the `Say “Hey Jarvis”` prompt lives only on the standby screen.

## Hardware

| Part | How it's used |
|---|---|
| Momentary push button, GPIO 5 | Each press toggles mic mute. |
| USB mic (default source) | `wpctl set-mute @DEFAULT_AUDIO_SOURCE@` |
| MAX98357A speaker | Not shown on screen. |
| WS2812, board.D13 | One pixel, brightness 0.3, written only when the colour changes. |
| Battery | Probed from `/sys/class/power_supply/*` for logs/diagnostics only; not shown on screen (mains-powered Pi). |

One background thread polls Wi-Fi (3s), internet (5s), mic mute (2s) and battery (15s), and pushes
only changes to the browser over Server-Sent Events. The browser is push-first: it paints
`/api/state` once on load and re-pulls it every 3s only while the stream is down.

## Hardware log

The app writes a rotating diagnostic log to `logs/span.log` (three 256 KB backups). It records every
battery probe with the detected `/sys/class/power_supply` devices, their type, capacity and charge
status, as well as GPIO, audio and LED initialization errors. After running the app on the Pi for a
minute, share this output when diagnosing battery detection:

```bash
tail -n 200 logs/span.log
```

Set `SPAN_LOG_FILE=/path/to/span.log` before starting the app to store it elsewhere. Log files are
excluded from Git. From any browser on the network: `GET /api/log?lines=100` tails the same log
without SSH (mute toggles, mode changes, Wi-Fi probe detail, init errors).

Light cases, exactly three (highest first):

| Condition | Colour |
|---|---|
| Mic muted (any mode) | red |
| Listening / recording / thinking / using a tool / responding | orange |
| Idle with mic on (unmuted) | green |
| Explicit light-off event | off |

Offline/online has no light colour — the Wi-Fi icon covers connectivity.
After the 10s follow-up the screen returns to idle unmuted, so the light
settles back to green.

## Pipeline integration

Your wake word / VAD / STT / SLM / TTS code reports progress with `POST /api/event` (JSON), or, in the
same process, `from app import publish; publish({...})`. The screen mode follows the events.

How the light works: curl never sets the light. Each event goes through `publish()`,
which sets the mode (`_set_mode`), then `update_led()` calls the `choose_led()` function
(`app.py`): light off? -> off; mic muted? -> red; mode not idle? -> orange; else green.
It writes the pixel (`board.D13`, only when the colour changes) and stores
`STATE.led`. The `S=` check below only *reads* that result via `GET /api/state`;
the browser LED bar reads the same field over SSE.

```bash
E='curl -s -XPOST -H Content-Type:application/json localhost:5000/api/event -d'
$E '{"type":"wake"}'                                              # -> listening
$E '{"type":"user_partial","id":"u1","text":"what is the"}'
$E '{"type":"user_final","id":"u1","text":"What is the weather?"}' # -> thinking
$E '{"type":"tool","id":"t1","name":"get_weather","status":"start"}'                # -> tool
$E '{"type":"tool","id":"t1","name":"get_weather","status":"ok","detail":"31°C"}'   # -> thinking
$E '{"type":"assistant_start","id":"a1"}'
$E '{"type":"assistant_token","id":"a1","text":"It is "}'          # -> responding
$E '{"type":"assistant_done","id":"a1","metrics":{"ttft_ms":420,"total_ms":1900,"gen_ms":1480,"tpot_ms":61.7,"tokens_per_sec":16.2,"input_tokens":312,"output_tokens":24}}'
$E '{"type":"assistant_interrupted","id":"a1"}'                    # user barged in -> listening
$E '{"type":"recording","on":true}'                                # push switch -> recording view
$E '{"type":"recording","on":false}'
$E '{"type":"light","on":false}'                               # -> WS2812 off for a hardware test
$E '{"type":"light","on":true}'                                # -> restore the state colour
$E '{"type":"mode","mode":"idle"}'                                 # set mode directly
$E '{"type":"error","message":"STT failed"}'
```

`assistant_done` keeps the conversation ready for a follow-up (default `listening`). After ten seconds
without another pipeline event it returns to the ready screen; configure this with
`FOLLOW_UP_TIMEOUT_SECONDS`. Pass `"next_mode":"idle"` to return immediately.
While recording, `user_partial` / `user_final` go to the recording card instead of the chat.
Ids only need to be unique among bubbles still in progress, so reusing `u1`/`a1` every turn is fine.

Check with `S='curl -s localhost:5000/api/state | python3 -c "import json,sys; s=json.load(sys.stdin); print(s[\"mode\"], s[\"led\"])"'`
after each step. Every non-idle mode shares the one orange working colour.

## Check sequences (copy-paste)

1. Thinking stays orange (one API), done -> listening orange, 10s -> main green:
```bash
E='curl -s -XPOST -H Content-Type:application/json localhost:5000/api/event -d'
S='curl -s localhost:5000/api/state | python3 -c "import json,sys; s=json.load(sys.stdin); print(s[\"mode\"], s[\"led\"])"'
$E '{"type":"wake"}' && $S                                                          # listening orange, chat
$E '{"type":"user_final","id":"u1","text":"What is the weather?"}' && $S            # thinking orange, stays
$E '{"type":"tool","id":"t1","name":"get_weather","status":"start"}' && $S          # tool orange (same)
$E '{"type":"tool","id":"t1","name":"get_weather","status":"ok","detail":"31C"}' && $S  # thinking orange
$E '{"type":"assistant_done","id":"a1"}' && $S                                       # listening orange
sleep 12 && $S                                                                      # idle green, main SPAN orb
```

2. Recording (button only, online or offline), mute never exits it:
```bash
$E '{"type":"recording","on":true}' && $S                                            # recording orange, recview
$E '{"type":"user_final","id":"r1","text":"Agenda for today"}' && $S                # stays recording
$E '{"type":"recording","on":false}' && $S                                          # idle green, REC archived
```

3. Offline tool error (Wi-Fi off or ISP down, light unchanged):
```bash
$E '{"type":"wake"}'
$E '{"type":"user_final","id":"u2","text":"Latest news in USA?"}'
$E '{"type":"tool","id":"t3","name":"web_search","status":"error","detail":"No internet connection"}' && $S  # thinking orange
$E '{"type":"assistant_done","id":"a2"}' && $S                                       # listening orange
sleep 12 && $S                                                                      # idle green
```

4. Interrupt / follow-up:
```bash
$E '{"type":"wake"}'
$E '{"type":"user_final","id":"u3","text":"Tell me about the Pi 5"}'
$E '{"type":"assistant_start","id":"a3"}'
$E '{"type":"assistant_token","id":"a3","text":"It has "}' && $S                     # speaking orange
$E '{"type":"assistant_interrupted","id":"a3"}' && $S                               # listening orange
sleep 12 && $S                                                                      # idle green
```

5. Light + stuck recovery (thinking with no done falls back after 10s):
```bash
$E '{"type":"light","on":false}' && $S                                               # led off
$E '{"type":"light","on":true}' && $S                                                # restored colour
$E '{"type":"user_final","id":"u9","text":"stuck?"}' && $S                           # thinking orange
sleep 12 && $S                                                                      # idle green, never stuck in chat
```

Metric names: TTFT = time to first token; total = end-to-end latency (request → last token);
gen = decode time (first → last token); TPOT ("avg") = average time per output token;
tok/s = output tokens per second; input/output = prompt and completion token counts.

Recent events are replayed to a browser that connects or reloads mid-conversation.
Keep this server on a trusted network; it has no authentication.
