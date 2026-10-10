"""Barge-in regression tests.

Root causes pinned here (each test fails on the pre-fix code):
1. A short speech blip during playback must NOT barge (sustain guard).
2. Sustained speech must barge AND cancel the running turn task (the LLM
   stream used to keep going); the brain generator must be observed closed.
3. A barged turn must not emit latency or leave partial reply text in
   history/store; the interrupting utterance must then run as a FRESH turn.

No models are loaded: VAD/wake are patched out and pipeline.tracker is
replaced with a fake whose in_utterance flag the test controls directly.
"""
from __future__ import annotations

import asyncio
import gc
import json
import time
import types

import numpy as np

import agent_core.pipeline as pl
from agent_core.config import load
from agent_core.debuglog import SessionLog
from agent_core.pipeline import VoicePipeline
from agent_core.stt import Transcript
from agent_core.turn import TurnResult
from tests.test_vad_turn import ROOT

CFG = load(ROOT / "configs" / "desktop.yaml")
FRAME = np.zeros(512, np.float32)
SUSTAIN_FRAMES = CFG.vad.barge_sustain_ms // 32 + 2  # comfortably past the gate


# --------------------------------------------------------------- minimal fakes

class FakeTracker:
    """Stands in for UtteranceTracker: in_utterance is test-controlled."""
    frame_ms = 32  # 512 samples @ 16 kHz, same as the real tracker
    last_prob = 0.95  # confident close speech by default

    def __init__(self) -> None:
        self.speaking = False
        self.utt: np.ndarray | None = None

    @property
    def in_utterance(self) -> bool:
        return self.speaking

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        utt, self.utt = self.utt, None
        return utt


class FakeBrain:
    """Streams sentences slowly; records calls and generator closure."""

    def __init__(self, sentences: int = 40, pace_s: float = 0.02) -> None:
        self.sentences, self.pace_s = sentences, pace_s
        self.calls: list[str] = []
        self.closed: list[str] = []  # appended from the generator's finally

    async def respond(self, text: str, history: list):
        self.calls.append(text)
        try:
            for i in range(self.sentences):
                yield f"Sentence {i}."
                await asyncio.sleep(self.pace_s)
        finally:
            self.closed.append(text)


class FakeSTT:
    def __init__(self, text: str = "stop the music") -> None:
        self.text = text
        self.calls: list[int] = []

    def transcribe(self, audio: np.ndarray) -> Transcript:
        self.calls.append(len(audio))
        return Transcript(text=self.text, language="en", inference_ms=1.0)


class FakeTTS:
    def synth_all(self, text: str) -> np.ndarray:
        return np.zeros(1600, np.float32)  # 0.1 s of silence, instant


class FakeTurnModel:
    def predict(self, audio: np.ndarray) -> TurnResult:
        return TurnResult(complete=True, probability=0.99, inference_ms=1.0)


class FakeStore:
    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def start_session(self, mode: str) -> int:
        return 1

    def add_utterance(self, *args, **kwargs) -> None:
        self.rows.append(args)  # (session_id, t_ms, role, text, ...)

    def end_session(self, *args, **kwargs) -> None:
        pass


# --------------------------------------------------------------------- helpers

async def make_pipeline(monkeypatch, tmp_path, *, brain: FakeBrain | None = None,
                        store: FakeStore | None = None):
    monkeypatch.setattr(pl, "SileroVAD", lambda *a, **k: types.SimpleNamespace(reset=lambda: None))
    monkeypatch.setattr(pl, "WakeWord", lambda *a, **k: None)
    brain = brain or FakeBrain()
    stt = FakeSTT()
    emitted: list[dict] = []
    audio: list[bytes] = []
    log = SessionLog(tmp_path, save_utterance_audio=False)
    p = VoicePipeline(CFG, log, emitted.append, audio.append, brain,
                      stt=stt, turn=FakeTurnModel(), tts=FakeTTS(), store=store)
    p.tracker = FakeTracker()
    p._set_state("listening")
    return types.SimpleNamespace(p=p, emitted=emitted, audio=audio, brain=brain, stt=stt, log=log)


async def wait_until(pred, what: str, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


async def settle() -> None:
    """Pump the loop so cancelled tasks unwind and async generators finalize."""
    for _ in range(20):
        await asyncio.sleep(0)
    gc.collect()
    for _ in range(20):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)


def log_events(p) -> list[dict]:
    lines = p.log.events_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ----------------------------------------------------------------------- tests

def test_short_blip_does_not_barge(tmp_path, monkeypatch):
    """(a) Speech < barge_sustain_ms during SPEAKING: no barge, reply continues."""

    async def body():
        h = await make_pipeline(monkeypatch, tmp_path)
        try:
            h.p._on_utterance_end(np.zeros(16000, np.float32))
            await wait_until(lambda: h.p.state == "speaking", "first reply playing")

            h.p.tracker.speaking = True
            for _ in range(3):  # 3 x 32ms = 96ms blip < 300ms sustain
                h.p.feed(FRAME)
            h.p.tracker.speaking = False
            for _ in range(3):
                h.p.feed(FRAME)

            assert h.p.state == "speaking"
            assert not any(e["type"] == "bargein" for e in h.emitted)
            n_audio = len(h.audio)
            await asyncio.sleep(0.3)
            assert len(h.audio) > n_audio, "reply must keep playing through a blip"
            assert h.p.state == "speaking"

            # exactly one barge_blocked per playback segment, with observed ms
            blocked = [e for e in log_events(h.p) if e["event"] == "barge_blocked"]
            assert len(blocked) == 1, blocked
            assert blocked[0]["speech_ms"] == 96

            await wait_until(lambda: h.p.state == "listening", "reply completes")
            assert any(e["type"] == "latency" for e in h.emitted)
            assert h.brain.closed == h.brain.calls  # generator ran to completion
        finally:
            h.p.close()
            await settle()

    asyncio.run(body())


def test_sustained_speech_barges_and_cancels_turn(tmp_path, monkeypatch):
    """(b) >= barge_sustain_ms of continuous speech: barge + turn task cancelled."""

    async def body():
        h = await make_pipeline(monkeypatch, tmp_path)
        try:
            h.p._on_utterance_end(np.zeros(16000, np.float32))
            await wait_until(lambda: len(h.audio) >= 1, "first sentence audible")

            h.p.tracker.speaking = True
            for _ in range(SUSTAIN_FRAMES):  # >= barge_sustain_ms of confident speech
                h.p.feed(FRAME)
            h.p.tracker.speaking = False
            for _ in range(2):
                h.p.feed(FRAME)

            assert h.p.state == "listening"
            assert any(e["type"] == "bargein" for e in h.emitted)
            assert h.p._turn_task is None, "turn task must be dropped"
            assert h.p._pending == []
            barge = [e for e in log_events(h.p) if e["event"] == "barge_in"]
            assert barge and barge[-1]["speech_ms"] >= 300, barge

            await settle()
            assert h.brain.closed == h.brain.calls, \
                "cancelled turn's brain generator must be closed (finally ran)"
        finally:
            h.p.close()
            await settle()

    asyncio.run(body())


def test_barge_mid_reply_stops_output_and_history(tmp_path, monkeypatch):
    """(c) Barge after >=1 sentence: no further output, no latency event,
    cancelled turn leaves no partial reply in history or store."""

    async def body():
        store = FakeStore()
        h = await make_pipeline(monkeypatch, tmp_path, store=store)
        try:
            h.p._on_utterance_end(np.zeros(16000, np.float32))
            await wait_until(lambda: len(h.audio) >= 2, "two sentences audible")

            n_audio, n_chunks = len(h.audio), sum(e["type"] == "reply_chunk" for e in h.emitted)
            h.p.tracker.speaking = True
            for _ in range(SUSTAIN_FRAMES):
                h.p.feed(FRAME)
            h.p.tracker.speaking = False
            for _ in range(2):
                h.p.feed(FRAME)

            await settle()
            assert h.p.state == "listening"
            assert len(h.audio) == n_audio, "no further TTS audio after barge"
            assert sum(e["type"] == "reply_chunk" for e in h.emitted) == n_chunks, \
                "no further reply text after barge"
            assert not any(e["type"] == "latency" for e in h.emitted)
            assert [t.role for t in h.p.history] == ["user"], \
                "cancelled turn must not append partial reply text to history"
            assert not any(r[2] == "assistant" for r in store.rows), \
                "cancelled turn must not store partial reply text"
            # (the barged turn's user row is also not stored: _respond_to persists
            # it only after playback completes; history keeps the text instead)
        finally:
            h.p.close()
            await settle()

    asyncio.run(body())


def test_interrupting_utterance_runs_as_fresh_turn(tmp_path, monkeypatch):
    """(d) After the barge, the interrupting utterance ends -> fresh _process_turn."""

    async def body():
        h = await make_pipeline(monkeypatch, tmp_path)
        try:
            h.p._on_utterance_end(np.zeros(16000, np.float32))
            await wait_until(lambda: len(h.audio) >= 1, "first sentence audible")

            h.p.tracker.speaking = True
            for _ in range(SUSTAIN_FRAMES):  # sustained speech -> barge
                h.p.feed(FRAME)
            h.p.tracker.speaking = False
            for _ in range(2):
                h.p.feed(FRAME)
            first_calls = list(h.brain.calls)

            # the utterance that triggered the barge now ends (tracker emits it),
            # fed synchronously: on the old code the still-running turn task would
            # queue it into _pending as a "continuation" instead of a fresh turn.
            h.p.tracker.utt = np.zeros(16000, np.float32)
            h.p.feed(FRAME)
            assert h.p._pending == [], "interrupting utterance must not be queued as continuation"
            await wait_until(lambda: len(h.brain.calls) == len(first_calls) + 1,
                             "fresh turn calls the brain")
            assert h.brain.calls[-1] == h.stt.text
            assert h.p._pending == []
            await settle()
            assert h.p.state == "speaking", "fresh turn should be answering"
        finally:
            h.p.close()
            await settle()

    asyncio.run(body())


def test_user_turn_during_playback_stops_playback(tmp_path, monkeypatch):
    """(e) A completed user turn while the client still plays (a quick "stop"
    shorter than the sustain window) must stop the playback AND run as a
    fresh turn. Live-E2E gap: the server finishes emitting long before the
    browser finishes playing, so state is LISTENING while audio is audible."""

    async def body():
        h = await make_pipeline(monkeypatch, tmp_path)
        try:
            h.p._on_utterance_end(np.zeros(16000, np.float32))
            await wait_until(lambda: h.p.state == "listening", "reply fully emitted")
            h.p.set_client_playback(True)          # browser still playing

            h.p.tracker.utt = np.zeros(8000, np.float32)
            h.p.feed(FRAME)                        # quick utterance ends mid-playback

            assert any(e["type"] == "bargein" for e in h.emitted), \
                "user turn during playback must stop the playback"
            await wait_until(lambda: len(h.brain.calls) == 2, "fresh turn runs")
            await wait_until(lambda: h.p.state == "speaking", "fresh reply plays")
        finally:
            h.p.close()
            await settle()

    asyncio.run(body())


def test_low_confidence_speech_does_not_barge(tmp_path, monkeypatch):
    """(f) Sustained but low-confidence speech (background chatter, distant
    voices, AEC leakage) must NOT interrupt the reply — only confident
    close speech counts toward the sustain window."""

    async def body():
        h = await make_pipeline(monkeypatch, tmp_path)
        try:
            h.p._on_utterance_end(np.zeros(16000, np.float32))
            await wait_until(lambda: h.p.state == "speaking", "reply playing")

            h.p.tracker.speaking = True
            h.p.tracker.last_prob = 0.45  # above VAD trigger, below barge_prob
            for _ in range(SUSTAIN_FRAMES * 2):
                h.p.feed(FRAME)
            h.p.tracker.speaking = False
            for _ in range(3):
                h.p.feed(FRAME)

            assert h.p.state == "speaking", "noise/distant speech must not barge"
            assert not any(e["type"] == "bargein" for e in h.emitted)
            blocked = [e for e in log_events(h.p) if e["event"] == "barge_blocked"]
            assert blocked and blocked[0]["speech_ms"] == 0 or not blocked
        finally:
            h.p.close()
            await settle()

    asyncio.run(body())
