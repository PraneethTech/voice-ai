"""Smart Turn v3.2 end-of-turn inference (ONNX, torch-free).

Ported from vendors/huggingface-speech-to-speech (VAD/smart_turn.py), with the
hub download replaced by a local model path. The pipeline always feeds 16 kHz
audio, so the vendor's resample branch is intentionally dropped.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 8


@dataclass(frozen=True)
class TurnResult:
    complete: bool
    probability: float
    inference_ms: float


class SmartTurn:
    """Run the Smart Turn v3.2 ONNX model on up to eight seconds of audio."""

    def __init__(self, model_path: str, *, threshold: float = 0.5, cpu_count: int = 1, warmup: bool = True) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be 0..1, got {threshold}")
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor  # numpy-only path, no torch

        self.threshold = threshold
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Smart Turn model not found: {path} (run tools/download_models.py)")

        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = cpu_count
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # Constructed directly (not from_pretrained): no hub access, no files needed.
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=MAX_AUDIO_SECONDS)
        logger.info("Smart Turn v3.2 loaded from %s (threshold=%.2f)", path, threshold)
        if warmup:
            self.predict(np.zeros(MODEL_SAMPLE_RATE, dtype=np.float32))

    def _prepare(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        max_samples = MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE
        if x.size > max_samples:
            x = x[-max_samples:]
        elif x.size < max_samples:
            x = np.pad(x, (max_samples - x.size, 0))
        return x

    def predict(self, audio: np.ndarray, *, sample_rate: int = MODEL_SAMPLE_RATE) -> TurnResult:
        if sample_rate != MODEL_SAMPLE_RATE:
            raise ValueError(f"pipeline feeds {MODEL_SAMPLE_RATE} Hz only, got {sample_rate}")
        started = time.perf_counter()
        features = self.feature_extractor(
            self._prepare(audio),
            sampling_rate=MODEL_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        input_features = np.asarray(features.input_features, dtype=np.float32)
        out = self.session.run(None, {self.input_name: input_features})
        probability = float(np.asarray(out[0]).reshape(-1)[0])
        if not np.isfinite(probability):
            raise RuntimeError(f"Smart Turn returned non-finite probability: {probability}")
        return TurnResult(
            complete=probability > self.threshold,
            probability=probability,
            inference_ms=(time.perf_counter() - started) * 1000,
        )

    def is_complete(self, audio: np.ndarray) -> bool:
        return self.predict(audio).complete
