"""Unified Voice AI Assistant & Display System Launcher.

Combines:
1. SPAN UI Display Server (Flask, 240x320 screen replica at http://127.0.0.1:5000)
2. STT-VAD-TTS + Agent (Silero VAD, WakeWord, Faster-Whisper, Brain/Tools, Piper TTS)
3. Real-time Display Bridge with animated loaders for:
   - Tool calling (animated cog/spinner and executing status)
   - Thinking & reasoning (inline bubble + bouncing dots)
   - Listening (soundwave equalizer bars)
   - Responding (speech stream + latency metrics)
   - Record / Meeting mode (waveform + stopwatch + transcript card)

Usage:
    python run_unified.py            # Run both UI and Voice Pipeline together
    python run_unified.py --demo     # Run animated live pipeline demo on the screen
    python run_unified.py --ui-only  # Run only the SPAN Display UI server
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ai-voice-assistant-ui"
V2_DIR = ROOT / "stt-vad-tts" / "personal_offline_assistant" / "v2"

sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(V2_DIR))


def start_ui_server(port: int = 5000):
    """Start the SPAN display Flask application in a background thread."""
    from werkzeug.serving import make_server
    import app as span_app

    server = make_server("0.0.0.0", port, span_app.app, threaded=True)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[UI] SPAN Display Server running on http://127.0.0.1:{port}", flush=True)
    time.sleep(0.5)
    return span_app


def run_demo(span_publish):
    """Runs an interactive demonstration showing all UI states and loaders."""
    print("\n--- Starting Live Pipeline & Loader Demo on Display ---", flush=True)
    time.sleep(1)

    print("1. Standby: Ready screen ('Say Hey Jarvis')", flush=True)
    span_publish({"type": "mode", "mode": "idle"})
    time.sleep(2.0)

    print("2. Wake Word Triggered: Screen transitions to Chat Box", flush=True)
    span_publish({"type": "wake"})
    time.sleep(1.5)

    print("3. Listening: User speaks partial words...")
    span_publish({"type": "user_partial", "id": "u1", "text": "What is the"})
    time.sleep(1.0)
    span_publish({"type": "user_partial", "id": "u1", "text": "What is the weather in"})
    time.sleep(1.0)
    span_publish({"type": "user_final", "id": "u1", "text": "What is the weather in Hyderabad today?"})
    print("   -> User turn finished. Entering Thinking / Tool Calling state...")
    time.sleep(1.2)

    print("4. Tool Calling Loader: Calling weather tool with animated spinner...")
    span_publish({"type": "tool", "id": "t1", "name": "get_weather", "status": "start"})
    time.sleep(2.5)

    print("5. Tool Result: Success checkmark + details")
    span_publish({"type": "tool", "id": "t1", "name": "get_weather", "status": "ok", "detail": "31°C, Clear Sky"})
    time.sleep(1.2)

    print("6. Reasoning / Thinking Loader before LLM tokens...")
    span_publish({"type": "reasoning"})
    time.sleep(2.0)

    print("7. Assistant Responding: Streaming tokens with live cursor...")
    span_publish({"type": "assistant_start", "id": "a1"})
    tokens = ["The ", "weather ", "in ", "Hyderabad ", "is ", "currently ", "31°C ", "with ", "clear ", "skies."]
    for tok in tokens:
        span_publish({"type": "assistant_token", "id": "a1", "text": tok})
        time.sleep(0.2)

    print("8. Assistant Done: Finished speaking, display latency metrics")
    span_publish({
        "type": "assistant_done",
        "id": "a1",
        "metrics": {
            "ttft_ms": 380,
            "total_ms": 1650,
            "gen_ms": 1270,
            "tpot_ms": 55.2,
            "tokens_per_sec": 18.1,
            "input_tokens": 128,
            "output_tokens": 23,
        },
    })
    time.sleep(4.0)

    print("\n9. Record / Meeting Mode Screen: Big waveform + live transcript card...")
    span_publish({"type": "recording", "on": True})
    time.sleep(1.5)
    span_publish({"type": "user_partial", "id": "r1", "text": "Discussion on Raspberry Pi", "recording": True})
    time.sleep(1.5)
    span_publish({"type": "user_final", "id": "r1", "text": "Discussion on Raspberry Pi 5 audio pipelines and display integration.", "recording": True})
    time.sleep(2.5)
    print("   -> Stopping Record Mode (archives transcript into chat)...")
    span_publish({"type": "recording", "on": False})
    time.sleep(3.0)

    print("10. Follow-up timeout returns to Ready screen...")
    span_publish({"type": "mode", "mode": "idle"})
    print("\n--- Demo Complete! All display modes and loaders verified. ---\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Voice AI + SPAN Display")
    parser.add_argument("--port", type=int, default=5000, help="Display port")
    parser.add_argument("--demo", action="store_true", help="Run interactive loader and pipeline demo")
    parser.add_argument("--ui-only", action="store_true", help="Start only the SPAN display server")
    parser.add_argument("--config", default=str(V2_DIR / "configs" / "desktop.yaml"))
    args = parser.parse_args()

    # 1. Start SPAN UI
    span_app = start_ui_server(port=args.port)

    if args.ui_only:
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    if args.demo:
        run_demo(span_app.publish)
        print("Demo finished. Server remains open on http://127.0.0.1:5000 (Ctrl+C to quit).")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    # 2. Start Full Pipeline Runner
    print(f"[pipeline] Launching Voice Pipeline with display http://127.0.0.1:{args.port}/api/event...")
    runner_script = V2_DIR / "pipeline_runner.py"
    cmd = [sys.executable, str(runner_script), "--config", args.config, "--display", f"http://127.0.0.1:{args.port}/api/event"]
    try:
        proc = subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
