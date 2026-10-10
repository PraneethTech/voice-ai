"""Tests for the integrated Voice Pipeline Display Bridge and SPAN UI screen.

Verifies:
1. DisplayBridge correctly queues and posts all assistant events:
   - wake
   - user_partial & user_final
   - tool calling (start with animated spinner, and ok with detail)
   - reasoning & thinking
   - assistant_start, assistant_token, assistant_done (with metrics)
   - assistant_interrupted (barge-in)
   - recording on/off (record/meeting mode)
2. SPAN UI state and event handlers process each event without crashing.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
UI_DIR = ROOT / "ai-voice-assistant-ui"
V2_DIR = ROOT / "stt-vad-tts" / "personal_offline_assistant" / "v2"

sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(V2_DIR))

import app as span
from agent_core.display import DisplayBridge


def test_display_bridge_in_process_events():
    events_received = []

    def capture(evt):
        events_received.append(evt)
        span.publish(evt)

    bridge = DisplayBridge(in_process_publish=capture)

    # 1. Wake
    bridge.wake()
    assert events_received[-1]["type"] == "wake"
    assert span.STATE["mode"] == "listening"

    # 2. User partial & final
    bridge.user_partial("u1", "What is the")
    assert events_received[-1]["type"] == "user_partial"
    assert span.STATE["mode"] == "listening"

    bridge.user_final("u1", "What is the weather?")
    assert events_received[-1]["type"] == "user_final"
    assert span.STATE["mode"] == "thinking"

    # 3. Tool call start & done
    bridge.tool_start("t1", "get_weather")
    assert events_received[-1]["type"] == "tool"
    assert events_received[-1]["status"] == "start"
    assert span.STATE["mode"] == "tool"

    bridge.tool_done("t1", "get_weather", "31C, sunny")
    assert events_received[-1]["type"] == "tool"
    assert events_received[-1]["status"] == "ok"
    assert span.STATE["mode"] == "thinking"

    # 4. Reasoning & Assistant tokens
    bridge.reasoning("analyzing forecast")
    assert events_received[-1]["type"] == "reasoning"
    assert span.STATE["mode"] == "reasoning"

    bridge.assistant_start("a1")
    assert events_received[-1]["type"] == "assistant_start"

    bridge.assistant_token("a1", "The weather is sunny.")
    assert events_received[-1]["type"] == "assistant_token"
    assert span.STATE["mode"] == "speaking"

    bridge.assistant_done("a1", {"ttft_ms": 350, "total_ms": 1200})
    assert events_received[-1]["type"] == "assistant_done"
    assert span.STATE["mode"] == "listening"

    # 5. Barge-in
    bridge.assistant_interrupted("a1")
    assert events_received[-1]["type"] == "assistant_interrupted"
    assert span.STATE["mode"] == "listening"

    # 6. Record / Meeting mode screen
    bridge.set_recording(True)
    assert events_received[-1]["type"] == "recording"
    assert span.STATE["mode"] == "recording"

    bridge.user_final("r1", "Discussion notes...", recording=True)
    assert events_received[-1]["type"] == "user_final"
    assert events_received[-1].get("recording") is True
    assert span.STATE["mode"] == "recording"

    bridge.set_recording(False)
    assert events_received[-1]["type"] == "recording"
    assert span.STATE["mode"] == "idle"
