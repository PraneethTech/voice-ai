"""Offline E2E from the command line: fixture WAV in -> transcript + reply WAV out.

No mic, no browser, no network. Also the first probe to run on a fresh Pi.
Usage: python tools/e2e_offline.py [tests/fixtures/complete_1.wav] [out.wav]
"""
from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_core  # noqa: F401,E402
from agent_core.brain.stub import StubBrain  # noqa: E402
from agent_core.config import load  # noqa: E402
from agent_core.stt import WhisperSTT  # noqa: E402
from agent_core.tts import PiperTTS  # noqa: E402
from agent_core.turn import SmartTurn  # noqa: E402
from agent_core.vad import SileroVAD, UtteranceTracker  # noqa: E402


def load_wav(path: Path) -> np.ndarray:  # noqa: F821
    import numpy as np

    with wave.open(str(path)) as wf:
        assert wf.getframerate() == 16000, "input must be 16 kHz mono"
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main() -> int:
    import numpy as np

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "fixtures" / "complete_1.wav"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "e2e_reply.wav"
    cfg = load(ROOT / "configs" / "desktop.yaml")

    t0 = time.perf_counter()
    vad = SileroVAD(str(ROOT / "models" / "silero_vad_v5.onnx"))
    tracker = UtteranceTracker(vad, threshold=cfg.vad.threshold, min_silence_ms=cfg.vad.min_silence_ms,
                               speech_pad_ms=cfg.vad.speech_pad_ms, min_speech_ms=cfg.vad.min_speech_ms)
    turn = SmartTurn(cfg.turn.model)
    stt = WhisperSTT(cfg.stt.model_dir, cfg.stt.compute_type, cfg.stt.language)
    tts = PiperTTS(cfg.tts.voice, cfg.tts.gain)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")

    audio = load_wav(src)
    feed = np.concatenate([audio, np.zeros(16000, np.float32)])  # 1s tail closes the utterance
    utterances = []
    for i in range(0, feed.size - 511, 512):
        u = tracker.process(feed[i : i + 512])
        if u is not None:
            utterances.append(u)
    if not utterances:
        print("NO UTTERANCE DETECTED")
        return 1
    print(f"utterance: {len(utterances[0]) / 16000:.2f}s")

    result = turn.predict(utterances[0])
    print(f"smart-turn: complete={result.complete} prob={result.probability:.3f} ({result.inference_ms:.0f} ms)")

    transcript = stt.transcribe(utterances[0])
    print(f"transcript: {transcript.text!r} ({transcript.inference_ms:.0f} ms)")

    async def brain_reply() -> str:
        return "".join([c async for c in StubBrain(pace_ms=0).respond(transcript.text, [])])

    t_eot = time.perf_counter()
    reply = asyncio.run(brain_reply())
    print(f"reply: {reply!r}")

    pcm = np.concatenate(list(tts.stream(reply)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.sample_rate)
        wf.writeframes(data.tobytes())
    print(f"reply audio: {dst} ({len(pcm) / tts.sample_rate:.2f}s, "
          f"first-audio latency from EOT: {(time.perf_counter() - t_eot) * 1000:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
