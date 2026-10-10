"""Meeting mode backend: speaker labels + summary, computed AFTER the meeting.

Audio is held in RAM only while the meeting runs; diarization uses per-segment
speaker embeddings (sherpa-onnx, torch-free) with greedy cosine clustering.
The summary is extractive until the SLM plugs into the Brain seam.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np


def diarize_segments(segments: list[tuple[int, np.ndarray]], embedding_model: str,
                     threshold: float = 0.5) -> list[str]:
    """Assign a speaker label to each (t_ms, audio) segment.

    Greedy clustering: a segment joins the nearest cluster centre if cosine
    similarity >= threshold, else it opens a new speaker. Keeps meeting audio
    order, so labels are stable (SPEAKER_1, SPEAKER_2, ...).
    """
    if not segments:
        return []
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding_model))
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    if ext.dim <= 0:
        raise RuntimeError("invalid speaker embedding model")

    centres: list[np.ndarray] = []
    labels: list[str] = []
    for _, audio in segments:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size < 16000 * 0.3:  # too short to embed reliably
            labels.append("SPEAKER_?")
            continue
        stream = ext.create_stream()
        stream.accept_waveform(16000, x)
        stream.input_finished()
        vec = np.asarray(ext.compute(stream), dtype=np.float32)
        vec /= max(float(np.linalg.norm(vec)), 1e-9)
        if centres:
            sims = [float(c @ vec) for c in centres]
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                centres[best] = (centres[best] + vec) / max(float(np.linalg.norm(centres[best] + vec)), 1e-9)
                labels.append(f"SPEAKER_{best + 1}")
                continue
        centres.append(vec)
        labels.append(f"SPEAKER_{len(centres)}")
    return labels


_STOP = set("the a an and or but is are was were be been i you he she it we they my your our "
            "to of in on at for with about that this these those what when where how why "
            "can could would should will shall do does did have has had not no yes okay um "
            "so just really very much more most some any".split())


def summarize_stub(entries: list[dict], duration_ms: int) -> str:
    """Extractive placeholder summary. The SLM (LFM2.5 / Qwen3) replaces this later."""
    if not entries:
        return "The meeting had no speech."
    n_speakers = len({e.get("speaker") for e in entries if e.get("speaker")} - {"SPEAKER_?"})
    words = Counter(w.lower() for e in entries for w in re.findall(r"[a-z']+", e["text"])
                    if w.lower() not in _STOP and len(w) > 3)
    topics = ", ".join(w for w, _ in words.most_common(6)) or "no strong topics"
    mins = max(1, duration_ms // 60000)
    first = entries[0]["text"][:120]
    return (f"{len(entries)} spoken segments over ~{mins} min from {n_speakers} speaker(s). "
            f"Opening: \"{first}\". Frequent terms: {topics}.")
