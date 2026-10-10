# Voice AI Assistant & Display Screen Integration Guide

This guide explains how **`stt-vad-tts`** (Tobi full-duplex voice pipeline) and **`ai-voice-assistant-ui`** (SPAN 240x320 display screen) are connected and how to run them together.

---

## 1. Architectural Overview

```
                      ┌──────────────────────────────────────────────┐
                      │            SPAN Display UI (Flask)           │
                      │         http://127.0.0.1:5000 (?kiosk=1)     │
                      └──────────────────────▲───────────────────────┘
                                             │  HTTP POST /api/event
                                             │  (non-blocking queue)
┌────────────────────────────────────────────┴───────────────────────────────────────┐
│ Voice Pipeline (Full-Duplex)                                                      │
│                                                                                   │
│  [Microphone] ──► Silero VAD ──► Wake Word ("Hey Jarvis")                         │
│                                        │                                          │
│                                        ▼                                          │
│                            Faster-Whisper STT                                     │
│                         (user_partial & user_final)                               │
│                                        │                                          │
│                                        ▼                                          │
│                         Agent Core (Brain & Tools)                                │
│                         - Tool calling: weather, search, telegram                 │
│                         - Reasoning / Thinking                                    │
│                         - Streaming text tokens                                   │
│                                        │                                          │
│                                        ▼                                          │
│                            Piper TTS (Resampled 16kHz)                            │
│                                        │                                          │
│                                        ▼                                          │
│                           [Speaker / Audio Output]                                │
│                           (Barge-in / Echo cancellation)                          │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Display Loaders & UI States

The SPAN display screen has dedicated animations and loaders for each pipeline phase:

| Phase | Screen Mode | Chat View Visual | Footer Loader Visual |
|---|---|---|---|
| **Voice Standby** | `idle` | Voice orb with breathing pulse, *"Say 'Hey Jarvis'"* | *"Waiting for wake word"* |
| **Listening** | `listening` | Live streaming text bubble (`partial`) | Animated soundwave equalizer bars (`.bars`) |
| **Tool Calling** | `tool` | Tool pill with animated spinning cog loader: `Calling <tool_name>…` | Rotating tool spinner (`.tool-spin`) |
| **Tool Completed** | `thinking` | Success pill: `✓ <tool_name> · <result>` | Returns to thinking state |
| **Thinking / Reasoning**| `thinking` / `reasoning` | Inline animated thinking card: pulsing orb + bounce dots (`Thinking…` or `Reasoning…`) | Bouncing dots (`.dots`) |
| **Responding** | `speaking` | Bot bubble streaming words with live cursor (`▍`) | Animated soundwave equalizer bars |
| **Turn Finished** | `listening` | Performance metrics under bubble (`TTFT · total · tok/s · in · out`) | Listening for follow-up questions |
| **Barge-in (Interrupt)**| `listening` | Bot bubble marked with `— interrupted` | Floor yielded back to user |
| **Record / Meeting Mode**| `recording` | Full record view with big animated waveform, live clock (`● REC MM:SS`), and live transcript card | Hidden (record view active) |

---

## 3. How to Run

### Option A: One-Command Unified Launcher (Recommended)

From the project root:

```bash
# Run both the SPAN UI Display and the Voice Pipeline together:
python run_unified.py

# Run the live animated demo showing all loaders and state transitions on the screen:
python run_unified.py --demo

# Run only the SPAN Display UI server:
python run_unified.py --ui-only
```

Open your browser at **http://127.0.0.1:5000** to view the live 240x320 screen!

---

### Option B: Running on Raspberry Pi (Dual Process / Kiosk)

#### Step 1: Start the SPAN Screen (Chromium Kiosk on Raspberry Pi Display)
```bash
cd ai-voice-assistant-ui
./run.sh --kiosk
```
*Starts Flask on port 5000 and launches fullscreen Chromium kiosk at `http://127.0.0.1:5000/?kiosk=1`.*

#### Step 2: Start the Voice Pipeline with Hardware Audio
```bash
cd stt-vad-tts/personal_offline_assistant/v2
python pipeline_runner.py --config configs/pi5-4gb.yaml --display http://127.0.0.1:5000/api/event
```
*Captures from ReSpeaker Lite USB microphone, outputs to speaker, and streams all UI events in real-time to the screen.*

---

## 4. Key Components Added & Updated

1. **`ai-voice-assistant-ui`**:
   - [`templates/index.html`](file:///c:/Users/HP/Desktop/voice-ai/ai-voice-assistant-ui/templates/index.html): Added dedicated `.tool-spin` footer loader element.
   - [`static/style.css`](file:///c:/Users/HP/Desktop/voice-ai/ai-voice-assistant-ui/static/style.css): Added animated tool spinner (`.tool-spinner`), `.chip.start` pulse, `.thinking-card` with bouncing dots, `.voice-orb` pulse, and `.tool-spin` SVG rotation.
   - [`static/app.js`](file:///c:/Users/HP/Desktop/voice-ai/ai-voice-assistant-ui/static/app.js): Added `showThinking()` / `removeThinking()`, tool calling spinner rendering, and reasoning mode handling.
   - [`app.py`](file:///c:/Users/HP/Desktop/voice-ai/ai-voice-assistant-ui/app.py): Added `reasoning` event type and mode.

2. **`stt-vad-tts/personal_offline_assistant/v2`**:
   - [`agent_core/display.py`](file:///c:/Users/HP/Desktop/voice-ai/stt-vad-tts/personal_offline_assistant/v2/agent_core/display.py): New `DisplayBridge` class for non-blocking asynchronous event streaming to SPAN UI.
   - [`agent_core/pipeline.py`](file:///c:/Users/HP/Desktop/voice-ai/stt-vad-tts/personal_offline_assistant/v2/agent_core/pipeline.py): Integrated `DisplayBridge` into wake detection, partials, turn completions, assistant token streams, and barge-in.
   - [`agent_core/tools/router.py`](file:///c:/Users/HP/Desktop/voice-ai/stt-vad-tts/personal_offline_assistant/v2/agent_core/tools/router.py): Emits tool start (`status: "start"`) and completion (`status: "ok"`) events.
   - [`agent_core/audio.py`](file:///c:/Users/HP/Desktop/voice-ai/stt-vad-tts/personal_offline_assistant/v2/agent_core/audio.py): New `AudioDeviceEngine` using `sounddevice` for 16 kHz mic input and speaker playback.
   - [`pipeline_runner.py`](file:///c:/Users/HP/Desktop/voice-ai/stt-vad-tts/personal_offline_assistant/v2/pipeline_runner.py): Standalone live pipeline runner with keyboard and hardware triggers.

3. **Workspace Root**:
   - [`run_unified.py`](file:///c:/Users/HP/Desktop/voice-ai/run_unified.py): Unified entrypoint supporting `--demo`, `--ui-only`, and full live pipeline execution.
