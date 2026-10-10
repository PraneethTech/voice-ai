"""Generate test fixture WAVs with the local Piper voice (offline, reproducible).

Usage: python tools/gen_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"

SPEECH = {
    "wake_jarvis.wav": "Hey Jarvis.",
    "complete_1.wav": "What time is the meeting tomorrow.",
    "complete_2.wav": "Tell me a fun fact about space.",
    "incomplete_1.wav": "Can you remind me to",
    "filler_1.wav": "Umm",
}


def main() -> None:
    from agent_core.tts import PiperTTS

    FIXTURES.mkdir(parents=True, exist_ok=True)
    import wave

    tts = PiperTTS(str(ROOT / "models" / "tts" / "en_US-lessac-low.onnx"))
    sr = tts.sample_rate

    def write(name: str, audio: np.ndarray) -> None:
        data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(FIXTURES / name), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data.tobytes())
        print(f"[wav ] {name} ({len(audio) / sr:.2f}s)")

    for name, text in SPEECH.items():
        write(name, tts.synth_all(text))

    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(sr * 2) * 0.05).astype(np.float32)  # room hiss level
    write("noise.wav", noise)
    write("silence.wav", np.zeros(sr * 2, dtype=np.float32))
    print("done")


if __name__ == "__main__":
    main()
