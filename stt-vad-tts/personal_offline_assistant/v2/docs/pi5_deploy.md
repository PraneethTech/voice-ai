# Raspberry Pi 5 deploy — tobi v2

Target: Pi 5 4GB, Raspberry Pi OS 64-bit (Bookworm), ReSpeaker Lite as USB sound card.

## 1. Base system

1. Flash Raspberry Pi OS 64-bit (Raspberry Pi Imager; set hostname, ssh, wifi via the gear).
2. Boot with the 27W PSU + active cooler. `sudo apt update && sudo apt full-upgrade -y`.
3. Audio stack: PipeWire is default. Force 16 kHz (ReSpeaker Lite + our pipeline):
   ```
   pw-metadata -n settings 0 clock.force-rate 16000
   # persist: default.clock.rate = 16000 in /etc/pipewire/pipewire.conf
   ```
4. Verify the ReSpeaker Lite shows up (USB firmware, no driver needed):
   `arecord -l` (see "ReSpeaker Lite" or XMOS). If USB errors in dmesg: add
   `usbcore.autosuspend=-1` to /boot/firmware/cmdline.txt (Seeed forum fix).
5. Copy the v2 folder to the Pi (rsync the repo; models/ + .venv live on the PC — on the Pi put
   models/ on the SD card). Create a fresh venv: `python3 -m venv .venv && pip install -r requirements-pi.txt`
   (needs `sudo apt install portaudio19-dev python3-dev libatlas3-base`).
6. `python tools/download_models.py` (once, with network) then `python tools/check_offline.py`.
7. `python tools/e2e_offline.py` — must pass before touching the mic.
8. Run with the ReSpeaker (server-side capture):
   - install `sounddevice` (already in requirements-pi) and add an ALSA backend to audio_bus
     (P10/P11 TODO — for the demo, run the browser UI from your phone: Pi serves on 0.0.0.0;
     phone browser mic + speaker keeps AEC for barge-in. This is the fastest demo path.)

## 2. Autostart (systemd)

```
# /etc/systemd/system/tobi.service
[Unit]
Description=tobi v2 offline s2s
After=network-online.target sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/personal_assistant/v2
ExecStart=/home/pi/personal_assistant/v2/.venv/bin/python -m server.ws_server --config configs/pi5-4gb.yaml
Restart=always

[Install]
WantedBy=multi-user.target
```
`sudo systemctl enable --now tobi`

## 3. Numbers to record on the Pi (fill in when the unit arrives)

- model load time: ____ s
- EOT → first audio (stub brain): ____ ms
- partial transcription cadence: ____ ms
- RAM: `free -m` after a session: ____
- meeting diarization RTF for a 3-min recording: ____

## 4. Known Pi caveats

- Browser barge-in = phone/laptop browser AEC. A raw server-side mic hears the agent's own
  speaker — add the echo guard (don't barge while speaker output energy is high; only accept
  wake-word re-trigger) before enabling barge-in on that path.
- tflite-runtime has no Trixie/Py3.13 wheels — openWakeWord falls back to onnxruntime (fine).
- If audio crackles: keep the 16 kHz clock forced; the 48k firmware mismatch is a known Seeed issue.
