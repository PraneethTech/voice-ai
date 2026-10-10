"""STT: faster-whisper (CTranslate2) — torch-free, fully local model dir."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    inference_ms: float


class WhisperSTT:
    def __init__(self, model_dir: str, compute_type: str = "int8", language: str | None = "en") -> None:
        from faster_whisper import WhisperModel

        self.language = language
        self.model = WhisperModel(
            model_dir, device="cpu", compute_type=compute_type, local_files_only=True
        )

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> Transcript:
        """Transcribe one utterance (float32 mono 16 kHz). Blocking; call from a worker."""
        started = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=language or self.language,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text for s in segments).strip()
        return Transcript(text=text, language=info.language, inference_ms=(time.perf_counter() - started) * 1000)
