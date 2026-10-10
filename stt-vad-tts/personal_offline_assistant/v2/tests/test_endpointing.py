"""Endpointing regression tests at the UtteranceTracker seam (pause handling).

Audio is real speech (local Piper TTS, tests/fixtures/vad_{a,b}.wav): Silero v5
does not sustain its speech probability on stationary synthetic carriers, so
sine-burst fixtures under-test the tracker. Knob values mirror
configs/desktop.yaml; the code beneath the knobs is what is under test.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from agent_core.vad import SileroVAD, UtteranceTracker

ROOT = Path(__file__).resolve().parent.parent
FRAME = 512  # 32ms at 16kHz

# mirrors configs/desktop.yaml vad section
KW = dict(threshold=0.5, min_silence_ms=500, speech_pad_ms=30, min_speech_ms=250)


def speech(name: str) -> np.ndarray:
    with wave.open(str(ROOT / "tests" / "fixtures" / f"{name}.wav"), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def vad() -> SileroVAD:
    return SileroVAD(str(ROOT / "models" / "silero_vad_v5.onnx"))


def feed(audio: np.ndarray, vad: SileroVAD) -> list[np.ndarray]:
    """Feed 512-sample frames through a fresh tracker; return emitted utterances."""
    tr = UtteranceTracker(vad, **KW)
    utts = []
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        u = tr.process(audio[i:i + FRAME])
        if u is not None:
            utts.append(u)
    return utts


def test_250ms_pause_does_not_split_utterance(vad: SileroVAD) -> None:
    """A 250ms dip inside speech is held (< min_silence_ms=500): one utterance."""
    a = speech("vad_a")
    dip = np.zeros(4000, np.float32)        # 250ms
    audio = np.concatenate([a, dip, a, np.zeros(16000, np.float32)])
    utts = feed(audio, vad)
    assert len(utts) == 1
    # spans both clips and the dip between them (minus at most the closing
    # silence window and frame quantization)
    assert len(utts[0]) >= len(a) + 4000 + len(a) - 17 * FRAME


def test_700ms_pause_splits_into_two_utterances(vad: SileroVAD) -> None:
    """Silence >= min_silence_ms ends the utterance; next clip starts a new one."""
    audio = np.concatenate([
        speech("vad_a"),
        np.zeros(11200, np.float32),        # 700ms silence (> min_silence_ms=500)
        speech("vad_b"),
        np.zeros(16000, np.float32),
    ])
    utts = feed(audio, vad)
    assert len(utts) == 2


def test_burst_shorter_than_min_speech_ms_is_dropped(vad: SileroVAD) -> None:
    """~100ms of speech < min_speech_ms=250: process() never returns an utterance."""
    audio = np.concatenate([
        np.zeros(8000, np.float32),
        speech("vad_a")[:1600],             # 100ms: attack of "Tell me"
        np.zeros(16000, np.float32),
    ])
    assert feed(audio, vad) == []


def test_utterance_onset_includes_pre_speech_pad(vad: SileroVAD) -> None:
    """speech_pad_ms pre-roll must survive frame quantization.

    Speech starts exactly at frame 15 (sample 7680). Real-speech onsets make
    Silero cross threshold within ~2 frames, and the (now non-zero) pad covers
    one more, so the emitted utterance must begin at most one frame after the
    true speech start — a vanished pad delays it further.
    """
    speech_start = 7680  # sample offset, frame-aligned (15 frames of silence)
    clip = speech("vad_a")
    audio = np.concatenate([
        np.zeros(speech_start, np.float32),
        clip,
        np.zeros(19200, np.float32),        # close the utterance
    ])
    energetic = np.abs(clip) > 0.02
    speech_end = speech_start + len(clip) - np.argmax(energetic[::-1])  # last energetic sample
    tr = UtteranceTracker(vad, **KW)
    onset = None
    fed = 0
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        fed += 1
        u = tr.process(audio[i:i + FRAME])
        if u is not None:
            onset = (fed - len(u) // FRAME) * FRAME  # utterance start in the global timeline
            assert onset + len(u) >= speech_end  # every energetic sample made it in
            break
    assert onset is not None, "no utterance emitted"
    assert onset <= speech_start + FRAME, f"onset {onset} is >1 frame after speech start {speech_start}"


def test_dip_recovery_merges_both_bursts(vad: SileroVAD) -> None:
    """temp_end recovery: clip -> 250ms dip -> clip -> 600ms+ silence merges."""
    a = speech("vad_b")
    audio = np.concatenate([
        np.zeros(4800, np.float32),         # 0.3s run-in
        a,
        np.zeros(4000, np.float32),         # 250ms dip
        a,
        np.zeros(16000, np.float32),        # 600ms+ silence closes the merged utterance
    ])
    utts = feed(audio, vad)
    assert len(utts) == 1
    u = utts[0]
    half = len(a) + 2000                    # midpoint: end of clip 1 + half the dip
    assert np.abs(u[:half]).max() > 0.05    # first clip present
    assert np.abs(u[half:]).max() > 0.05    # second clip present
