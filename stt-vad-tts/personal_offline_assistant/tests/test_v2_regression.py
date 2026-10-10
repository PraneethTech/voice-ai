"""v2 regression tests. Deterministic: synthetic PCM + state machine only.
No mic, no network. Model files load from local models/ (offline-safe)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from v2.echo import EchoGate
from v2.engine import DORMANT, LISTENING, SPEAKING, Engine
from v2.log import SessionLog
from v2.reply import FIXED, reply_for
from v2.stt import transcribe
from v2.turn import FLOOR_DBFS, TurnSegmenter
from v2.vad_eou import SileroVAD
from v2.smart_turn import SmartTurnJudge


def _silence(seconds: float) -> bytes:
    return (np.zeros(int(16000 * seconds), dtype=np.int16)).tobytes()


def _hiss(seconds: float, amp: float = 0.004) -> bytes:
    rng = np.random.default_rng(7)
    return (
        (rng.normal(0, amp, int(16000 * seconds)).astype(np.float32) * 32768)
        .astype(np.int16)
        .tobytes()
    )


def _frames(pcm: bytes, n: int = 1024):
    return [pcm[i : i + n] for i in range(0, len(pcm), n)]


def test_silence_never_starts_turn():
    seg = TurnSegmenter(SileroVAD(), SmartTurnJudge())
    for f in _frames(_silence(3.0)):
        assert seg.feed(f, chunk_ms=32.0) is None


def test_hiss_never_starts_turn():
    seg = TurnSegmenter(SileroVAD(), SmartTurnJudge())
    for f in _frames(_hiss(3.0)):
        assert seg.feed(f, chunk_ms=32.0) is None


def test_frame_dbfs_scale():
    from v2.turn import frame_dbfs

    assert frame_dbfs(_silence(0.1)) == -100.0
    assert frame_dbfs(b"") == -100.0
    half = (np.ones(1600, dtype=np.int16) * 16384).tobytes()
    assert frame_dbfs(half) < -5.5  # half scale ~ -6 dBFS
    full = (np.ones(1600, dtype=np.int16) * 32767).tobytes()
    assert -1.0 < frame_dbfs(full) < 1.0  # full scale ~ 0 dBFS


def test_live_extend_never_walks_backwards():
    from v2.pipeline import Pipeline

    p = Pipeline()
    p._live_extend("he hoped there")
    assert p._live_words == ["he", "hoped", "there"]
    p._live_extend("he hoped there would be")  # agrees: extends
    assert p._live_words == ["he", "hoped", "there", "would", "be"]
    p._live_extend("he hoped their would be stew")  # revises committed: held
    assert p._live_words == ["he", "hoped", "there", "would", "be"]
    p._live_extend("")  # empty pass: held
    assert p._live_words == ["he", "hoped", "there", "would", "be"]


def test_turn_tail_bounded():
    seg = TurnSegmenter(SileroVAD(), SmartTurnJudge())
    seg._buf = bytearray(b"\x01\x02" * 16000 * 6)  # 6s of bytes
    assert len(seg.tail(5.0)) == int(16000 * 2 * 5.0)
    assert seg.tail(5.0) == bytes(seg._buf[-int(16000 * 2 * 5.0) :])


def test_stt_gate_silence_no_model():
    text, lang, lp = transcribe(_silence(1.0), language="en")
    assert (text, lang, lp) == ("", "", 0.0)


def test_reply_is_single_string():
    from v2.reply import reply_stream

    assert reply_for("hello world", "en") == FIXED
    assert reply_for("hola mundo", "es") == FIXED
    assert reply_for("", "en") == FIXED
    assert list(reply_stream("anything", "es")) == [FIXED]


def test_session_logs_every_step_in_order():
    import json

    s = SessionLog()
    s.config(lang="en", reply="fixed")
    s.attempt_start("001")
    s.turn_start("001", -20.0, 0.9)
    s.candidate("001", 0.5, 300.0, 260.0, 0.75)
    s.candidate("001", 0.9, 300.0, 500.0, 0.75)
    s.stt("001", "hello", "en", 1.0, 120.0)
    s.smart_turn("001", 0.9, 0.75)
    s.reply("001", "Got it.", 400.0)
    s.tts("001", "Got it.", "en", 150.0, 800.0)
    s.playback_done("001", 800.0)
    s.attempt_end("001", "turn_done")
    s.session_end("stop")
    events = [json.loads(line)["event"] for line in open(s.path)]
    assert events == [
        "session_start",
        "config",
        "attempt_start",
        "turn_start",
        "candidate",
        "candidate",
        "stt",
        "smart_turn",
        "reply",
        "tts",
        "playback_done",
        "attempt_end",
        "session_end",
    ]
    aids = {
        json.loads(line).get("attempt")
        for line in open(s.path)
        if "attempt" in json.loads(line)
    }
    assert aids == {"001"}


def test_echo_gate_catches_own_reply():
    g = EchoGate()
    g.note_spoken("Got it.")
    assert g.is_echo("got it") is True
    assert g.is_echo("Got it, thanks") is True
    assert g.is_echo("hello world") is False
    assert g.is_echo("ok") is False  # too short to judge


def test_engine_wake_idle_dormant():
    eng = Engine(inactivity_ms=200)
    assert eng.state == DORMANT
    assert eng.attempt is None  # pipeline owns session files, not the engine
    eng.on_wake()
    assert eng.state == LISTENING
    time.sleep(0.3)
    eng.tick_idle()
    assert eng.state == DORMANT


def test_engine_speaking_interrupt_epoch():
    eng = Engine()
    eng.on_wake()
    e0 = eng.epoch()
    eng.force_state(SPEAKING)
    eng.bump()
    assert eng.epoch() == e0 + 1
    eng.force_state(LISTENING)
    assert eng.state == LISTENING


def test_silence_records_no_candidates():
    seg = TurnSegmenter(SileroVAD(), SmartTurnJudge())
    for f in _frames(_silence(3.0)):
        assert seg.feed(f, chunk_ms=32.0) is None
    assert seg.candidates == []
    assert seg.last_candidates == []


def test_wakeword_offline_silence_no_fire():
    from v2.wakeword import WakeWord

    w = WakeWord()
    assert w.available is True
    sil = (np.zeros(16000 * 2, dtype=np.int16)).tobytes()
    assert not any(w.feed(sil[i : i + 1024]) for i in range(0, len(sil), 1024))


def test_wakeword_refractory_and_missing_model():
    import pytest

    from v2.wakeword import WakeWord

    w = WakeWord(threshold=0.0, refractory_s=60.0)  # fires on anything
    frame = (np.zeros(1280, dtype=np.int16)).tobytes()
    assert w.feed(frame) is True
    assert w.feed(frame) is False  # cooldown suppresses repeats
    with pytest.raises(FileNotFoundError):
        WakeWord(phrase="no_such_model_xyz").feed(frame)


def test_pipeline_turn_start_logs_attempt_id():
    """The exact loop path: VAD edge -> attempt file carries the attempt id.
    Guards the turn_start(aid, db, prob) signature against drift."""
    import json

    from v2.pipeline import Pipeline

    p = Pipeline()
    path = p.start_session()
    p.seg.turn_started = True
    p.seg.start_db = -20.0
    p.seg.start_prob = 0.9
    p._on_turn_started()
    p.stop_session()
    by_event = {}
    for line in open(path):
        e = json.loads(line)
        by_event.setdefault(e["event"], []).append(e)
    assert [e["attempt"] for e in by_event["attempt_start"]] == ["001"]
    ts = by_event["turn_start"]
    assert len(ts) == 1 and ts[0]["attempt"] == "001"
    assert ts[0]["db"] == -20.0


def test_smartturn_checks_throttled_during_silence():
    """67 per-chunk predicts stalled the consumer and overflowed the mic
    queue (the CFFI dialog). Now: one check per 500ms of silence."""
    seg = TurnSegmenter(SileroVAD(), SmartTurnJudge())
    calls = []
    seg.judge.predict = lambda seg_bytes: calls.append(len(seg_bytes)) or 0.1
    seg._speaking = True
    seg._buf = bytearray(b"\x01\x02" * 8000)
    seg._speech_ms = 500.0  # past min_speech: eligible for checks
    seg._voice_db = -20.0  # loud enough to commit as speech, not noise
    for f in _frames(_silence(2.6)):  # 2600ms silence -> max-silence commit
        out = seg.feed(f, chunk_ms=32.0)
        if out is not None:
            break
    assert out[1] == "speech"
    assert 2 <= len(calls) <= 6, f"{len(calls)} neural checks in one silence"


def test_mic_push_never_raises_on_full_queue():
    from v2.audio_io import AudioIO

    io = AudioIO()
    for _ in range(3):
        for _ in range(io._mic_q.maxsize + 50):
            io._push_mic(b"\x00\x01" * 256)  # must not raise, stays bounded
        assert io._mic_q.qsize() <= io._mic_q.maxsize
