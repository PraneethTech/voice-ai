"""Wake word: openWakeWord ONNX classifiers, torch-free.

openWakeWord consumes 80 ms (1280-sample) int16 chunks. The pipeline feeds
32 ms VAD frames; this module re-frames them. Model files must exist locally —
this package never downloads (offline guarantee).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CHUNK = 1280  # 80 ms at 16 kHz


class WakeWord:
    def __init__(self, model_dir: str, names: list[str], threshold: float = 0.5) -> None:
        # Names are logical ("hey_jarvis"); files are versioned ("hey_jarvis_v0.1.onnx").
        resolved: list[tuple[str, Path]] = []
        missing: list[str] = []
        for name in names:
            matches = sorted(Path(model_dir).glob(f"{name}*.onnx"))
            if matches:
                resolved.append((name, matches[0]))
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError(
                f"wake word model(s) missing for {missing} in {model_dir} — run tools/download_models.py"
            )
        from openwakeword.model import Model

        self.names = [name for name, _ in resolved]
        self.threshold = threshold
        self.model = Model(wakeword_models=[str(p) for _, p in resolved])
        self._buf = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> dict[str, float]:
        """Feed one 512-sample float32 frame; return latest scores per wake word."""
        self._buf = np.concatenate([self._buf, np.asarray(frame, dtype=np.float32).reshape(-1)])
        if self._buf.size < CHUNK:
            return {}
        chunks = []
        while self._buf.size >= CHUNK:
            chunks.append(self._buf[:CHUNK])
            self._buf = self._buf[CHUNK:]
        scores: dict[str, float] = {}
        for c in chunks:
            pred = self.model.predict((np.clip(c, -1, 1) * 32767).astype(np.int16))
            for name in self.names:
                hits = [v for k, v in pred.items() if k.startswith(name)]
                if hits:
                    scores[name] = max(scores.get(name, 0.0), float(hits[0]))
        return scores

    def triggered(self, scores: dict[str, float]) -> bool:
        return any(v >= self.threshold for v in scores.values())
