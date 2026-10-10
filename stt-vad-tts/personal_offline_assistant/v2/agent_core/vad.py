"""Silero VAD v5 (ONNX, stateful) and an utterance tracker.

The tracker logic is a faithful port of the vendor implementation in
vendors/huggingface-speech-to-speech (VAD/vad_iterator.py), rewritten from
torch to numpy + the official Silero v5 ONNX graph.

Call convention (mirrors the official OnnxWrapper): each frame is 512 new
samples at 16 kHz, prepended with a 64-sample context tail from the previous
frame — the graph consumes 576 samples. The audio bus guarantees 512-sample
frames before they reach this module.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort


class SileroVAD:
    """Stateful Silero v5 ONNX session. Feed one 512-sample frame at a time."""

    FRAME_SAMPLES = 512
    CONTEXT_SAMPLES = 64
    STATE_DIM = 128

    def __init__(self, model_path: str, threads: int = 1) -> None:
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._state = np.zeros((2, 1, self.STATE_DIM), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT_SAMPLES, dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def reset(self) -> None:
        self._state = np.zeros_like(self._state)
        self._context = np.zeros_like(self._context)

    def prob(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 frame."""
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size != self.FRAME_SAMPLES:
            raise ValueError(f"VAD needs exactly {self.FRAME_SAMPLES} samples, got {x.size}")
        model_in = np.concatenate([self._context, x])[None, :]  # rank 2, 576 samples
        out, state = self.session.run(
            None, {self._input_name: model_in, "state": self._state, "sr": self._sr}
        )
        self._state = state
        self._context = model_in[0, -self.CONTEXT_SAMPLES:]
        return float(np.asarray(out).reshape(-1)[0])


class UtteranceTracker:
    """Groups 512-sample frames into utterances with hysteresis and padding.

    process() returns the utterance audio (float32, 16 kHz mono) exactly once,
    when speech has ended and stayed silent for min_silence_ms. Utterances
    shorter than min_speech_ms are dropped (clicks, breaths).
    """

    def __init__(
        self,
        vad: SileroVAD,
        *,
        threshold: float = 0.5,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 30,
        min_speech_ms: int = 250,
        sample_rate: int = 16000,
    ) -> None:
        self.vad = vad
        self.threshold = threshold
        self.frame = SileroVAD.FRAME_SAMPLES
        self.frame_ms = self.frame * 1000 // sample_rate
        # Ceil, never floor: flooring lost the 30ms pad entirely (30//32 == 0
        # frames) and shaved up to frame_ms off the silence window.
        self.min_silence_frames = max(1, -(-min_silence_ms // self.frame_ms))
        self.speech_pad_frames = max(1, -(-speech_pad_ms // self.frame_ms))
        self.min_speech_samples = min_speech_ms * sample_rate // 1000
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.triggered = False
        self.temp_end = 0            # frame index where the current silence dip started
        self.current = 0             # frame index counter
        self.last_prob = 0.0         # speech probability of the most recent frame
        self.buffer: list[np.ndarray] = []
        self.prefix: list[np.ndarray] = []
        self.pre_speech: list[np.ndarray] = []
        self.active_speech_samples = 0

    @property
    def in_utterance(self) -> bool:
        return self.triggered

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size != self.frame:
            raise ValueError(f"VAD needs exactly {self.frame} samples, got {x.size}")
        self.current += 1
        p = self.vad.prob(x)
        self.last_prob = p

        if not self.triggered:
            if p >= self.threshold:
                self.triggered = True
                self.prefix = self.pre_speech
                self.pre_speech = []
                self.buffer = [x]
                self.active_speech_samples = x.size
            else:
                self.pre_speech.append(x)
                excess = len(self.pre_speech) - self.speech_pad_frames
                if excess > 0:
                    del self.pre_speech[:excess]
            return None

        # Triggered: every frame goes into the utterance, including trailing dips.
        self.buffer.append(x)
        if p >= self.threshold - 0.15:
            self.active_speech_samples += x.size
            if self.temp_end:
                self.temp_end = 0  # recovered from a short dip

        if p < self.threshold - 0.15:
            if not self.temp_end:
                self.temp_end = self.current
            if self.current - self.temp_end < self.min_silence_frames:
                return None

            utterance = np.concatenate(self.prefix + self.buffer) if (self.prefix or self.buffer) else None
            spoken = self.active_speech_samples
            self.reset()
            if utterance is None or spoken < self.min_speech_samples:
                return None
            return utterance
        return None
