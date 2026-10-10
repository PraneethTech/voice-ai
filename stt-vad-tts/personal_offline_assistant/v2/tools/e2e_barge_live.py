"""Live E2E: barge-in cancels the reply mid-playback; a pause does not split.

Runs against a REAL server instance (Groq brain + base.en STT + medium TTS),
feeding real user utterance wavs frame-by-frame in realtime over the WebSocket.
Usage: .venv-v2/Scripts/python.exe tools/e2e_barge_live.py <port>
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
FRAME = 512


def wav(path: str) -> np.ndarray:
    with wave.open(str(ROOT / path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def stream(ws, audio: np.ndarray, realtime: bool = True,
                  trail_s: float = 0.0) -> None:
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        await ws.send(pcm_bytes(audio[i:i + FRAME]))
        if realtime:
            await asyncio.sleep(FRAME / 16000)
    # The server only sees frames we send: give the tracker its closing silence.
    for _ in range(int(trail_s * 16000 / FRAME)):
        await ws.send(pcm_bytes(np.zeros(FRAME, np.float32)))
        if realtime:
            await asyncio.sleep(FRAME / 16000)


class EventLog:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.audio_bytes = 0
        self.audio_after: dict[float, int] = {}   # mark -> bytes received after

    def mark(self, name: str) -> None:
        self.audio_after[name] = self.audio_bytes


async def reader(ws, log: EventLog, stop: asyncio.Event) -> None:
    try:
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                log.audio_bytes += len(msg)
            else:
                e = json.loads(msg)
                log.events.append({**e, "_t": time.monotonic()})
    except websockets.ConnectionClosed:
        pass
    finally:
        stop.set()


async def wait_for(log: EventLog, pred, timeout: float = 30.0) -> dict | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for e in log.events:
            if pred(e):
                return e
        await asyncio.sleep(0.05)
    return None


async def main() -> None:
    q1 = wav("logs/session_20261007-131446-173/001_utterance.wav")   # "What time is the meeting tomorrow?"
    q2 = wav("logs/session_20261007-225828-080/001_utterance.wav")   # second real question
    log = EventLog()
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=2**22) as ws:
        stop = asyncio.Event()
        asyncio.create_task(reader(ws, log, stop))
        hello = await wait_for(log, lambda e: e["type"] == "hello")
        assert hello and hello["brain"] == "groq", f"expected groq brain, got {hello}"
        print(f"[e2e] connected, brain={hello['brain']}, log={hello['log_dir']}")

        # --- wake (any audio passes threshold 0.01) ---
        await stream(ws, wav("tests/fixtures/wake_jarvis.wav"))
        wake = await wait_for(log, lambda e: e["type"] == "wake")
        assert wake, "no wake event"
        print("[e2e] wake OK")

        # --- turn 1: ask a question, let the reply START playing ---
        await stream(ws, q1, trail_s=1.0)
        final1 = await wait_for(log, lambda e: e["type"] == "final" and e.get("text"))
        assert final1, "no final transcript for turn 1"
        print(f"[e2e] turn 1 STT: {final1['text']!r}")
        log.events.clear()   # drop greeting/turn-1 history: only NEW events from here
        chunk = await wait_for(log, lambda e: e["type"] == "reply_chunk", timeout=20)
        assert chunk, "no reply chunk for turn 1"
        print(f"[e2e] LLM streaming: {chunk['text'][:60]!r}")
        await ws.send(json.dumps({"type": "playback", "value": True}))

        async def audio_flowing() -> bool:
            return log.audio_bytes > 0
        t0 = time.monotonic()
        while log.audio_bytes == 0 and time.monotonic() - t0 < 15:
            await asyncio.sleep(0.1)
        assert log.audio_bytes > 0, "reply never produced audio"
        await asyncio.sleep(0.6)   # let playback get going
        mark = log.audio_bytes
        print(f"[e2e] reply playing ({mark} bytes so far)")

        # --- barge: stream a REAL second question over the playback ---
        await stream(ws, q2, trail_s=1.0)
        barge = await wait_for(log, lambda e: e["type"] == "bargein")
        assert barge, "NO barge-in while speaking over the reply"
        print(f"[e2e] BARGE-IN fired")
        # A real leak = old-reply audio still arriving AFTER the bargein event.
        # The fresh turn's reply only becomes possible after STT+LLM+TTS
        # (~1.5s), so any audio in the first 0.4s window is a leak (allow one
        # in-flight sentence chunk the server had already queued).
        log.mark("barge")
        await asyncio.sleep(0.4)
        leaked = log.audio_bytes - log.audio_after["barge"]
        await asyncio.sleep(1.1)   # fresh turn proceeds; its reply audio is correct
        await ws.send(json.dumps({"type": "playback", "value": False}))

        # the interrupting utterance becomes a fresh turn
        final2 = await wait_for(log, lambda e: e["type"] == "final" and e.get("text"), timeout=40)
        assert final2, "interrupting speech never became a turn"
        print(f"[e2e] fresh turn STT: {final2['text']!r} (stt {final2.get('stt_ms')}ms)")
        chunk2 = await wait_for(log, lambda e: e["type"] == "reply_chunk"
                          and e["_t"] > (final2 or {"_t": 0})["_t"], timeout=40)
        assert chunk2, "no fresh reply"
        print(f"[e2e] fresh reply: {chunk2['text'][:60]!r}")
        lat = await wait_for(log, lambda e: e["type"] == "latency")
        print(f"[e2e] latency: {lat}")

        # --- pause-continuation: one question split by a 250ms mid-sentence dip ---
        n = len(q1) // 2
        log.events.clear()
        await stream(ws, q1[:n])
        await asyncio.sleep(0.25)          # the dip (< min_silence 500ms)
        await stream(ws, q1[n:], trail_s=1.0)
        await asyncio.sleep(2.0)           # let it finish + close
        finals = [e for e in log.events if e["type"] == "final" and e.get("text")]
        print(f"[e2e] pause scenario: {len(finals)} final(s): {[f['text'][:30] for f in finals]}")
        assert len(finals) == 1, f"expected ONE merged turn, got {len(finals)}"

        print(f"\n[e2e] RESULT: {'PASS' if leaked == 0 else f'AUDIO LEAK AFTER BARGE: {leaked} bytes'}")
        print(f"[e2e] audio leaked after barge: {leaked} bytes (0 = TTS stopped dead)")
    stop.set()


if __name__ == "__main__":
    asyncio.run(main())
