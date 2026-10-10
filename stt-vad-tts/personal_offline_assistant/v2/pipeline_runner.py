"""Direct Hardware Voice Pipeline Runner with SPAN UI Display Integration.

Runs:
Microphone -> Silero VAD -> Wake Word ("Hey Jarvis") -> Faster-Whisper STT
-> Brain (Agent Core with Tools: Weather, Search, Telegram) -> Piper TTS -> Speaker.

In parallel, all events are streamed via DisplayBridge to the SPAN 240x320 UI:
- Standby / Ready screen -> Wake detection -> Chat screen
- Live user partial transcriptions
- Animated tool calling loaders (cog spinner, executing status)
- Animated thinking & reasoning loaders
- Assistant live streaming response with metrics
- Record / Meeting mode screen with live waveform and transcript card.

Usage:
    python pipeline_runner.py --config configs/desktop.yaml --display http://127.0.0.1:5000/api/event
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent_core.audio import AudioDeviceEngine
from agent_core.brain.groq import GroqBrain
from agent_core.brain.stub import StubBrain
from agent_core.config import Config, load
from agent_core.debuglog import SessionLog
from agent_core.display import DisplayBridge
from agent_core.embedder import Embedder
from agent_core.pipeline import VoicePipeline
from agent_core.store import Store
from agent_core.stt import WhisperSTT
from agent_core.tools.router import ToolRouter
from agent_core.tts import PiperTTS
from agent_core.turn import SmartTurn


def make_brain(cfg: Config, router: ToolRouter, store: Store, embedder: Embedder):
    if cfg.brain.type == "groq":
        brain = GroqBrain(
            api_key_env=cfg.brain.api_key_env,
            model=cfg.brain.model,
            store=store,
            embedder=embedder,
            router=router,
        )
        if brain.has_key():
            print("[brain] Using GroqBrain")
            return brain
        print(f"[brain] Groq key {cfg.brain.api_key_env} missing, falling back to StubBrain")
    return StubBrain(router=router, store=store, embedder=embedder)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Tobi Voice Pipeline + SPAN Screen")
    parser.add_argument("--config", default=str(ROOT / "configs" / "desktop.yaml"))
    parser.add_argument("--display", default=os.environ.get("SPAN_EVENT_URL", "http://127.0.0.1:5000/api/event"))
    parser.add_argument("--meeting", action="store_true", help="Start directly in Record/Meeting mode")
    parser.add_argument("--mic-device", default=None, help="Input audio device index or name")
    parser.add_argument("--out-device", default=None, help="Output audio device index or name")
    args = parser.parse_args()

    cfg = load(args.config)
    print(f"[pipeline] Config loaded from {args.config}")

    # 1. Display Bridge
    display = DisplayBridge(endpoint_url=args.display)
    display.start()
    print(f"[display] DisplayBridge target: {args.display}")

    # 2. Tool router & brain
    router = ToolRouter(cfg, on_event=display.emit)
    store = Store(cfg.store.path)
    embedder = Embedder(str(ROOT / "models" / "embeddings"))
    brain = make_brain(cfg, router, store, embedder)

    # 3. Neural models
    print("[models] Loading STT, Turn detector, and Piper TTS...")
    try:
        stt = WhisperSTT(cfg.stt.model_dir, cfg.stt.compute_type, cfg.stt.language)
        turn = SmartTurn(cfg.turn.model)
        tts = PiperTTS(cfg.tts.voice, cfg.tts.gain)
    except Exception as exc:
        print(f"[warning] Offline neural models could not be loaded ({exc}).")
        print("          If you haven't run 'python tools/download_models.py', please run it first.")
        return

    log = SessionLog(cfg.session.log_dir, cfg.session.save_utterance_audio)
    loop = asyncio.get_running_loop()

    # 4. Audio Engine
    audio_engine: AudioDeviceEngine | None = None

    def on_mic_frame(frame: np.ndarray) -> None:
        if pipeline:
            pipeline.feed(frame)

    def on_playback_state(playing: bool) -> None:
        if pipeline:
            pipeline.set_client_playback(playing)

    audio_engine = AudioDeviceEngine(
        on_mic_frame=on_mic_frame,
        on_playback_state=on_playback_state,
        input_device=args.mic_device,
        output_device=args.out_device,
    )

    # 5. Voice Pipeline
    def on_pipeline_event(d: dict) -> None:
        pass

    def on_pipeline_audio(pcm: bytes) -> None:
        if audio_engine:
            audio_engine.play_chunk(pcm)

    pipeline = VoicePipeline(
        cfg,
        log,
        emit=on_pipeline_event,
        emit_audio=on_pipeline_audio,
        brain=brain,
        stt=stt,
        turn=turn,
        tts=tts,
        meeting_mode=args.meeting,
        store=store,
        display=display,
    )

    audio_engine.start()
    print("\n=======================================================")
    print("Voice Assistant Pipeline is LIVE!")
    print(f"Wake word:       '{cfg.wake.names}' (Say to open Chat Box screen)")
    print(f"Display screen:  {args.display}")
    print("Controls:")
    print("  'r' + Enter    Toggle Record / Meeting Mode screen")
    print("  'w' + Enter    Simulate Wake Word")
    print("  'm' + Enter    Toggle Microphone Mute")
    print("  'q' + Enter    Quit")
    print("=======================================================\n")

    meeting_state = args.meeting

    # Input command thread
    def terminal_listener():
        nonlocal meeting_state
        while True:
            try:
                line = sys.stdin.readline().strip().lower()
                if line == "q":
                    loop.call_soon_threadsafe(stop_event.set)
                    break
                elif line == "r":
                    meeting_state = not meeting_state
                    display.set_recording(meeting_state)
                    print(f"-> Record/Meeting Mode: {'ON' if meeting_state else 'OFF'}")
                elif line == "w":
                    display.wake()
                    print("-> Wake word triggered manually!")
                elif line == "m":
                    is_muted = not pipeline.muted
                    pipeline.set_muted(is_muted)
                    display.emit({"type": "muted", "muted": is_muted})
                    print(f"-> Mute: {is_muted}")
            except Exception:
                break

    stop_event = asyncio.Event()
    t = threading.Thread(target=terminal_listener, daemon=True)
    t.start()

    # Background dormancy watcher
    async def dormancy_loop():
        while not stop_event.is_set():
            await asyncio.sleep(2)
            pipeline.check_dormancy()

    dormancy_task = asyncio.create_task(dormancy_loop())

    await stop_event.wait()
    print("\nStopping Voice Pipeline...")
    dormancy_task.cancel()
    if audio_engine:
        audio_engine.stop()
    pipeline.close()
    await display.stop()
    print("Clean shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
