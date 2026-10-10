"""One-time model download. After this runs, the whole runtime is offline.

Usage: python tools/download_models.py
Safe to re-run — existing files are skipped.
"""
from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

# (url, dest relative to v2 root, mode) — mode "file", or "tar-member:<path inside archive>"
_GH_OW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
MANIFEST: list[tuple[str, str, str]] = [
    (
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
        "models/turn/smart-turn-v3.2-cpu.onnx",
        "file",
    ),
    # openWakeWord ships models on GitHub releases (the HF mirror is gated).
    (f"{_GH_OW}/hey_jarvis_v0.1.onnx", "models/wakeword/hey_jarvis_v0.1.onnx", "file"),
    (f"{_GH_OW}/alexa_v0.1.onnx", "models/wakeword/alexa_v0.1.onnx", "file"),
    # Feature frontends every openWakeWord classifier needs.
    (f"{_GH_OW}/melspectrogram.onnx", "models/wakeword/melspectrogram.onnx", "file"),
    (f"{_GH_OW}/embedding_model.onnx", "models/wakeword/embedding_model.onnx", "file"),
    # STT upgrade: base.en (better accuracy than tiny.en, still CPU-friendly).
    (
        "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/model.bin",
        "models/stt-base/model.bin",
        "file",
    ),
    (
        "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/config.json",
        "models/stt-base/config.json",
        "file",
    ),
    (
        "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/tokenizer.json",
        "models/stt-base/tokenizer.json",
        "file",
    ),
    (
        "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/vocabulary.txt",
        "models/stt-base/vocabulary.txt",
        "file",
    ),
    # TTS upgrade: lessac-medium (much better than low, still faster than realtime).
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "models/tts/en_US-lessac-medium.onnx",
        "file",
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "models/tts/en_US-lessac-medium.onnx.json",
        "file",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "models/diarize/segmentation.onnx",
        "tar-member:sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "models/diarize/embedding.onnx",
        "file",
    ),
    # MiniLM-L6-v2 int8 ONNX + tokenizer — RAG embeddings (torch-free).
    (
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx",
        "models/embeddings/model.onnx",
        "file",
    ),
    (
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
        "models/embeddings/tokenizer.json",
        "file",
    ),
    (
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/config.json",
        "models/embeddings/config.json",
        "file",
    ),
    (
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json",
        "models/embeddings/tokenizer_config.json",
        "file",
    ),
    (
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json",
        "models/embeddings/special_tokens_map.json",
        "file",
    ),
]

# Not downloadable here (already on disk / separate channels) — verified at the end.
MUST_EXIST = [
    "models/stt-base/model.bin",            # faster-whisper base.en (CTranslate2)
    "models/stt-base/tokenizer.json",
    "models/stt-tiny/model.bin",            # fallback STT (Pi profile)
    "models/tts/en_US-lessac-medium.onnx",  # Piper voice (desktop profile)
    "models/tts/en_US-lessac-low.onnx",     # fallback voice (Pi profile)
    "models/silero_vad_v5.onnx",            # Silero VAD v5
]


def _fetch(url: str) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            chunks = []
            for chunk in resp.iter_bytes():
                chunks.append(chunk)
                print(".", end="", flush=True)
            print()
            return b"".join(chunks)


def _extract_tar_member(blob: bytes, member: str, dest: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        with tarfile.open(tmp_path, "r:bz2") as tar:
            for entry in tar.getmembers():
                if entry.name == member:
                    src = tar.extractfile(entry)
                    if src is None:
                        raise FileNotFoundError(f"{member} inside archive is not a file")
                    dest.write_bytes(src.read())
                    return
        raise FileNotFoundError(f"member {member!r} not found in archive")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _install_openwakeword_frontends() -> None:
    """Copy the feature frontend models into the package's resources dir.

    openwakeword looks for melspectrogram/embedding models in its own resources
    folder; without them it would try to download at first run — which would
    break the offline guarantee. Copies are cheap and kept in models/wakeword/ too.
    """
    import shutil

    import openwakeword  # noqa: PLC0415 — only needed at setup time

    res = Path(openwakeword.__file__).parent / "resources" / "models"
    res.mkdir(parents=True, exist_ok=True)
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        src = ROOT / "models" / "wakeword" / name
        if src.is_file():
            shutil.copy2(src, res / name)
            print(f"[pkg ] openwakeword resources <- {name}")


def main() -> int:
    ok = True
    for url, rel, mode in MANIFEST:
        dest = ROOT / rel
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[skip] {rel} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[get ] {rel}")
        try:
            blob = _fetch(url)
            if mode.startswith("tar-member:"):
                _extract_tar_member(blob, mode.split(":", 1)[1], dest)
            else:
                dest.write_bytes(blob)
            print(f"[ ok ] {rel} ({len(blob) / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 — report every failure, keep downloading the rest
            print(f"[FAIL] {rel}: {exc}")
            ok = False
    _install_openwakeword_frontends()
    for rel in MUST_EXIST:
        p = ROOT / rel
        if p.is_file() and p.stat().st_size > 0:
            print(f"[have] {rel} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"[MISS] {rel} — obtain it manually (see README)")
            ok = False
    print("\nALL MODELS READY — runtime is offline from here on." if ok else "\nINCOMPLETE — fix the lines marked above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
