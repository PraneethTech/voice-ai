"""Per-session debug logging.

Every mic session (web client, Pi capture, or offline test) creates
logs/session_<timestamp>/events.jsonl automatically. No manual logging needed
when something misbehaves — replay the jsonl.
"""
from __future__ import annotations

import json
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


class SessionLog:
    def __init__(self, log_dir: str | Path, save_utterance_audio: bool = True) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        self.dir = Path(log_dir) / f"session_{ts}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_utterance_audio = save_utterance_audio
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._n_audio = 0

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    def log(self, event: str, **fields: object) -> None:
        rec = {"t_ms": round((time.perf_counter() - self._t0) * 1000), "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def save_wav(self, name: str, audio: np.ndarray, sample_rate: int = 16000) -> Path:
        """Save float32 mono audio (or int16) as a 16-bit wav for later replay."""
        self._n_audio += 1
        path = self.dir / f"{self._n_audio:03d}_{name}.wav"
        data = np.asarray(audio)
        if data.dtype != np.int16:
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if peak > 1.0:
                data = data / peak
            data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())
        return path
