"""Offline end-to-end: fixture WAV -> VAD -> Smart-Turn -> STT -> StubBrain -> TTS WAV.

No mic, no network. This is the regression gate for the whole chain.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from agent_core.brain.stub import StubBrain
from agent_core.stt import WhisperSTT
from agent_core.tts import PiperTTS
from tests.test_vad_turn import ROOT, load_wav, run_tracker

MODEL = str(ROOT / "models")


@pytest.fixture(scope="module")
def stt() -> WhisperSTT:
    return WhisperSTT(f"{MODEL}/stt-tiny", compute_type="int8", language="en")


@pytest.fixture(scope="module")
def tts() -> PiperTTS:
    return PiperTTS(f"{MODEL}/tts/en_US-lessac-low.onnx")


@pytest.fixture(scope="module")
def vad():
    from agent_core.vad import SileroVAD

    return SileroVAD(f"{MODEL}/silero_vad_v5.onnx")


def test_full_chain_wav_to_reply(vad, stt: WhisperSTT, tts: PiperTTS) -> None:
    audio = load_wav("complete_1.wav")
    utts = run_tracker(audio, vad)
    assert len(utts) == 1

    transcript = stt.transcribe(utts[0])
    assert "meeting" in transcript.text.lower(), f"unexpected transcript: {transcript.text!r}"

    async def stream() -> list[str]:
        return [chunk async for chunk in StubBrain(pace_ms=0).respond(transcript.text, [])]

    reply_words = asyncio.run(stream())
    reply_text = "".join(reply_words).strip()
    assert "stub" in reply_text.lower()

    reply_audio = tts.synth_all(reply_text)
    assert len(reply_audio) > 0.5 * tts.sample_rate, "TTS reply too short"


def test_tts_stream_is_incremental(tts: PiperTTS) -> None:
    chunks = list(tts.stream("This is a streaming test with several words and a second sentence. More words here."))
    assert len(chunks) >= 2, "Piper should stream more than one chunk for a two-sentence input"


def test_partial_transcription_updates(stt: WhisperSTT, vad) -> None:
    """Simulate growing utterance: later partial should contain at least as many words."""
    audio = load_wav("complete_2.wav")
    utts = run_tracker(audio, vad)
    full = stt.transcribe(utts[0]).text.lower()
    half = stt.transcribe(utts[0][: len(utts[0]) // 2]).text.lower()
    assert len(full.split()) >= len(half.split()) > 0, (full, half)
