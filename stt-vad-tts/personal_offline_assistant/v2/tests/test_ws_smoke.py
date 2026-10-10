"""WebSocket smoke test: full DORMANT -> wake -> greeting -> turn -> reply cycle,
plus meeting mode, through the real FastAPI app (no browser, no mic, no network).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from agent_core.config import load
from server.ws_server import build_app
from tests.test_vad_turn import ROOT, load_wav

CFG = load(ROOT / "configs" / "desktop.yaml")


def feed(ws, audio: np.ndarray, tail_s: float = 0.7) -> None:
    pcm = np.concatenate([audio, np.zeros(int(16000 * tail_s), np.float32)])
    i16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
    for i in range(0, len(i16), 1024):
        ws.send_bytes(i16[i : i + 1024])


def recv_until(ws, want: set[str], min_audio: int = 0, timeout: float = 60.0) -> list[dict]:
    """Collect JSON events (counting audio frames) until all wanted types arrive."""
    events, audio_frames, end = [], 0, time.time() + timeout
    while time.time() < end:
        msg = ws.receive()
        if msg.get("bytes"):
            audio_frames += 1
            continue
        if not msg.get("text"):
            continue
        ev = json.loads(msg["text"])
        events.append(ev)
        needed_events = {e["type"] for e in events}
        if want <= needed_events and audio_frames >= min_audio:
            return events
    raise TimeoutError(f"did not see {want} (got {sorted({e['type'] for e in events})}, audio={audio_frames})")


def _wake_and_reply_body() -> None:
    from fastapi.testclient import TestClient

    with TestClient(build_app(CFG)) as client, client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["wake_words"], "wake words must be advertised in assistant mode"

        # 1) wake word -> greeting audio -> back to listening
        feed(ws, load_wav("wake_jarvis.wav"))
        evs = recv_until(ws, {"wake"}, min_audio=1, timeout=90)
        assert any(e["type"] == "wake" for e in evs)

        # 2) spoken turn -> final transcript -> streamed reply audio -> latency
        feed(ws, load_wav("complete_1.wav"), tail_s=1.0)
        evs = recv_until(ws, {"final", "latency"}, min_audio=1, timeout=90)
        final = next(e for e in evs if e["type"] == "final")
        assert "meeting" in final["text"].lower(), final
        lat = next(e for e in evs if e["type"] == "latency")
        assert 0 < lat["eot_to_first_audio_ms"] < 5000, lat


def _meeting_body() -> None:
    from fastapi.testclient import TestClient

    with TestClient(build_app(CFG)) as client, client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        ws.send_json({"type": "meeting", "value": True})
        evs = recv_until(ws, {"hello"}, timeout=30)
        assert any(e.get("meeting") for e in evs if e["type"] == "hello")
        feed(ws, load_wav("complete_1.wav"), tail_s=1.0)
        evs = recv_until(ws, {"meeting_segment"}, timeout=90)
        seg = next(e for e in evs if e["type"] == "meeting_segment")
        assert "meeting" in seg["text"].lower(), seg
        ws.send_json({"type": "mute", "value": True})
        evs = recv_until(ws, {"muted"}, timeout=30)
        assert any(e.get("muted") for e in evs if e["type"] == "muted")


def test_wake_and_reply_cycle() -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_wake_and_reply_body).result(timeout=280)


def test_meeting_mode_transcribes_without_wake() -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_meeting_body).result(timeout=280)
