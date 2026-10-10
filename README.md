# Voice AI Assistant & Real-time Display Screen

An offline, full-duplex conversational voice assistant with an integrated real-time 240x320 display screen, built for **Raspberry Pi 5** and **Desktop (Windows / Linux)**.

---

## 🌟 Key Features

1. **Full-Duplex Speech Pipeline (`stt-vad-tts`)**:
   - **Wake Word Detection**: OpenWakeWord (*"Hey Jarvis"*).
   - **Voice Activity Detection (VAD)**: Silero VAD v5 (low-latency audio segmentation).
   - **Speech-to-Text (STT)**: Faster-Whisper (streaming partials & final transcripts).
   - **Agent Core & Reasoning**: StubBrain (offline) or GroqBrain (cloud LLM) with tool integration (*weather, web search, telegram*).
   - **Text-to-Speech (TTS)**: Piper TTS (streaming neural synthesis, resampled to 16 kHz).
   - **Voice Barge-in & Interruption**: Interrupt the assistant mid-sentence anytime.

2. **Real-time 240x320 Display Screen (`ai-voice-assistant-ui`)**:
   - **Voice Standby View**: Pulsing voice orb with *"Say 'Hey Jarvis'"*.
   - **Chat Box View**: Displays user speech bubbles and streaming assistant responses.
   - **Active Visual Loaders**:
     - *Tool calling loader*: Rotating gear spinner (`Calling <tool>…`) resolving to success/error pills.
     - *Thinking & reasoning loader*: Inline animated card with pulsing orb and bounce dots.
     - *Listening loader*: Animated soundwave equalizer bars.
     - *Speaking indicator*: Live typing cursor `▍` with latency benchmarks (`TTFT · total · tok/s · in · out`).
   - **Record / Meeting Mode**: Fullscreen animated waveform, stopwatch timer (`● REC MM:SS`), and live transcription card.
   - **Physical Hardware Support**: WS2812 status LED (Green/Orange/Red), GPIO 5 physical mic mute switch.

---

## 📁 Repository Structure

```
voice-ai/
├── ai-voice-assistant-ui/            # SPAN 240x320 Flask Web Display Screen
│   ├── static/                       # UI styling & vanilla JS
│   ├── templates/                    # index.html device screen replica
│   ├── app.py                        # Flask server with SSE stream (/api/events)
│   └── run.sh                        # Raspberry Pi launcher (with --kiosk option)
│
├── stt-vad-tts/                      # Full-Duplex Speech-to-Speech Pipeline
│   └── personal_offline_assistant/v2/
│       ├── agent_core/               # VAD, STT, Brain, Tools, TTS, DisplayBridge, AudioEngine
│       ├── configs/                  # desktop.yaml and pi5-4gb.yaml
│       ├── server/                   # WebSocket transport server (ws_server.py)
│       ├── pipeline_runner.py        # Direct hardware mic/speaker pipeline runner
│       └── tools/                    # Model download and offline verification tools
│
├── run_unified.py                    # Master launcher for both UI and Voice Pipeline
├── run_pi5.sh                        # One-command Raspberry Pi 5 startup script
└── INTEGRATION_GUIDE.md              # Complete integration and deployment guide
```

---

## 🚀 Quickstart

### On PC (Windows / Linux)

```bash
# 1. Install dependencies
pip install -r ai-voice-assistant-ui/requirements.txt
pip install -r stt-vad-tts/personal_offline_assistant/v2/requirements.txt

# 2. Run the interactive live demo (shows all loaders and screen transitions)
python run_unified.py --demo

# 3. Run both the Display Screen and Voice Pipeline together
python run_unified.py
```
Open **http://127.0.0.1:5000** in your browser to view the screen!

---

### On Raspberry Pi 5

```bash
# 1. One-time offline neural model download
cd stt-vad-tts/personal_offline_assistant/v2
python tools/download_models.py
cd ../../..

# 2. Run display in fullscreen kiosk mode on Pi screen + voice pipeline:
./run_pi5.sh --kiosk

# Or run headless (viewable from any phone/laptop browser on the network):
./run_pi5.sh
```

For full details, wiring instructions, and service autostart, see [INTEGRATION_GUIDE.md](file:///c:/Users/HP/Desktop/voice-ai/INTEGRATION_GUIDE.md).
