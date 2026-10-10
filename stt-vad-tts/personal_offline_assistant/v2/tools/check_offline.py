"""Offline audit — the sovereignty gate.

Loads EVERY model with networking blocked at the socket layer. If anything
tries to phone home (hub download, telemetry, update check), this fails.

Usage: python tools/check_offline.py
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_core  # noqa: F401,E402 — sets offline env guards
from agent_core.config import load  # noqa: E402


def _block(*args: object, **kwargs: object) -> None:
    raise OSError("NETWORK ACCESS ATTEMPTED — runtime must be fully offline")


def main() -> int:
    # Block at the connection layer, not socket creation: imports themselves
    # stay healthy, any real attempt to reach the network raises.
    socket.socket.connect = _block  # type: ignore[method-assign]
    socket.socket.connect_ex = _block  # type: ignore[method-assign]
    socket.create_connection = _block  # type: ignore[assignment]
    socket.getaddrinfo = _block  # type: ignore[assignment]

    cfg = load(ROOT / "configs" / "desktop.yaml")
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            results.append((name, True, detail or "ok"))
        except Exception as exc:  # noqa: BLE001 — report every failure
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    def vad() -> str:
        import time

        import numpy as np

        from agent_core.vad import SileroVAD

        v = SileroVAD(str(ROOT / "models" / "silero_vad_v5.onnx"))
        t0 = time.perf_counter()
        p = v.prob(np.zeros(512, dtype=np.float32))
        return f"prob={p:.2f} ({(time.perf_counter() - t0) * 1000:.1f} ms)"

    def turn() -> str:
        import numpy as np

        from agent_core.turn import SmartTurn

        t = SmartTurn(cfg.turn.model)
        r = t.predict(np.zeros(16000, dtype=np.float32))
        return f"prob={r.probability:.2f} in {r.inference_ms:.0f} ms"

    def stt() -> str:
        import numpy as np

        from agent_core.stt import WhisperSTT

        s = WhisperSTT(cfg.stt.model_dir, cfg.stt.compute_type, cfg.stt.language)
        t = s.transcribe(np.zeros(16000, dtype=np.float32))
        return f"lang={t.language} text={t.text!r}"

    def tts() -> str:
        from agent_core.tts import PiperTTS

        p = PiperTTS(cfg.tts.voice)
        n = sum(len(a) for a in p.stream("Offline check."))
        return f"sr={p.sample_rate} samples={n}"

    def wake() -> str:
        import numpy as np

        from agent_core.wakeword import WakeWord

        w = WakeWord(cfg.wake.dir, cfg.wake.names)
        for _ in range(5):
            scores = w.process_frame(np.zeros(512, dtype=np.float32))
        return f"scores={ {k: round(v, 2) for k, v in scores.items()} }"

    def embed() -> str:
        import numpy as np

        from agent_core.embedder import Embedder

        e = Embedder(str(ROOT / "models" / "embeddings"))
        v = e.embed("sovereign offline assistant")
        return f"dim={len(v)} norm={np.linalg.norm(v):.2f}"

    check("config", lambda: f"{cfg.server.host}:{cfg.server.port}")
    check("vad/silero", vad)
    check("turn/smart-turn", turn)
    check("stt/faster-whisper", stt)
    check("tts/piper", tts)
    check("wake/openwakeword", wake)
    check("embeddings/minilm", embed)

    ok = all(passed for _, passed, _ in results)
    for name, passed, detail in results:
        print(f"{'[PASS]' if passed else '[FAIL]'} {name:22s} {detail}")
    print("\nSOVEREIGN: all models load and run with networking blocked." if ok else "\nNOT OFFLINE-SAFE — see failures above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
