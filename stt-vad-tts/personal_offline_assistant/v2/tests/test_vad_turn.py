"""VAD + Smart-Turn regression tests, pinned to verified behavior on synthetic fixtures.

Known fixture limitation (documented, do not 'fix' blindly): Piper gives every
utterance final declarative prosody, so a truncated sentence still scores
"complete". Incomplete-utterance detection must be verified with a real mic
(manual test). The filler fixture DOES hold the turn — that is the signal that
matters for regressions here.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from agent_core.turn import SmartTurn
from agent_core.vad import SileroVAD, UtteranceTracker

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
MODEL = str(ROOT / "models")


def load_wav(name: str) -> np.ndarray:
    with wave.open(str(FIX / name)) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def run_tracker(audio: np.ndarray, vad: SileroVAD, **kw) -> list[np.ndarray]:
    tr = UtteranceTracker(vad, **kw)
    utts = []
    pad = np.zeros(512 * 15, np.float32)  # ~0.5s trailing silence to close any open utterance
    for chunk in (audio, pad):
        for i in range(0, max(len(chunk) - 511, 0), 512):
            u = tr.process(chunk[i : i + 512])
            if u is not None:
                utts.append(u)
    return utts


@pytest.fixture(scope="module")
def vad() -> SileroVAD:
    return SileroVAD(f"{MODEL}/silero_vad_v5.onnx")


@pytest.fixture(scope="module")
def turn() -> SmartTurn:
    return SmartTurn(f"{MODEL}/turn/smart-turn-v3.2-cpu.onnx")


@pytest.mark.parametrize("name", ["complete_1.wav", "complete_2.wav", "incomplete_1.wav", "filler_1.wav"])
def test_speech_produces_exactly_one_utterance(vad: SileroVAD, name: str) -> None:
    utts = run_tracker(load_wav(name), vad, threshold=0.5, min_silence_ms=300, min_speech_ms=250)
    assert len(utts) == 1, f"{name}: expected 1 utterance, got {len(utts)}"
    assert len(utts[0]) > 0.3 * 16000


@pytest.mark.parametrize("name", ["noise.wav", "silence.wav"])
def test_non_speech_produces_no_utterance(vad: SileroVAD, name: str) -> None:
    assert run_tracker(load_wav(name), vad) == []


def test_short_click_is_dropped(vad: SileroVAD) -> None:
    rng = np.random.default_rng(0)
    click = np.zeros(16000, np.float32)
    click[2000:2230] = rng.standard_normal(230) * 0.9  # 14ms burst < min_speech_ms
    assert run_tracker(click, vad) == []


@pytest.mark.parametrize("name", ["complete_1.wav", "complete_2.wav"])
def test_complete_utterances_score_complete(vad: SileroVAD, turn: SmartTurn, name: str) -> None:
    utt = run_tracker(load_wav(name), vad)[0]
    result = turn.predict(utt)
    assert result.complete, f"{name} should be a complete turn (prob={result.probability:.3f})"
    assert result.inference_ms < 500  # budget guard: smart-turn must stay cheap


def test_filler_holds_the_turn(vad: SileroVAD, turn: SmartTurn) -> None:
    utt = run_tracker(load_wav("filler_1.wav"), vad)[0]
    result = turn.predict(utt)
    assert not result.complete, f"'Umm' must not end the turn (prob={result.probability:.3f})"


def test_truncated_scores_complete_on_tts_prosody(vad: SileroVAD, turn: SmartTurn) -> None:
    """Pins the documented fixture limitation: synthetic prosody reads as complete.

    With a real mic this behavior differs — verify manually (see docs/TESTING.md).
    """
    utt = run_tracker(load_wav("incomplete_1.wav"), vad)[0]
    assert turn.predict(utt).complete
