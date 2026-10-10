"""WebSocket transport: browser mic in, events + TTS audio out.

Usage: python -m server.ws_server --config configs/desktop.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_core  # noqa: F401,E402 — offline env guards first
from agent_core.brain.base import Brain  # noqa: E402
from agent_core.brain.groq import GroqBrain  # noqa: E402
from agent_core.brain.stub import StubBrain  # noqa: E402
from agent_core.config import Config, load  # noqa: E402
from agent_core.debuglog import SessionLog  # noqa: E402
from agent_core.display import DisplayBridge  # noqa: E402
from agent_core.embedder import Embedder  # noqa: E402
from agent_core.meeting import diarize_segments, summarize_stub  # noqa: E402
from agent_core.pipeline import VoicePipeline  # noqa: E402
from agent_core.stt import WhisperSTT  # noqa: E402
from agent_core.store import Store  # noqa: E402
from agent_core.tools.router import ToolRouter  # noqa: E402
from agent_core.tts import PiperTTS  # noqa: E402
from agent_core.turn import SmartTurn  # noqa: E402

# WebSocket must be resolvable at MODULE level: `from __future__ import annotations`
# makes endpoint annotations strings, and FastAPI resolves them against globals().
from fastapi import FastAPI, WebSocket  # noqa: E402

UI_DIR = Path(__file__).resolve().parent / "ui"

_cached: dict[str, object] = {}
_cache_lock = threading.Lock()
_env_loaded = False


def _load_env() -> None:
    """Load v2/.env (GROQ_API_KEY, TOBI_TG_TOKEN, TOBI_TG_CHAT) if present."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def make_brain(cfg: Config, shared: dict) -> Brain:
    """Brain factory: 'groq' dev bridge (keyed) or the offline stub."""
    if cfg.brain.type == "groq":
        _load_env()
        brain: Brain = GroqBrain(api_key_env=cfg.brain.api_key_env, model=cfg.brain.model,
                                 store=shared["store"], embedder=shared["embedder"],
                                 router=shared["router"])
        if brain.has_key():
            return brain
        print(f"[brain] GROQ key {cfg.brain.api_key_env} missing — falling back to StubBrain",
              file=sys.stderr, flush=True)
    return StubBrain(router=shared["router"], store=shared["store"], embedder=shared["embedder"])


def _get_shared(cfg: Config) -> dict[str, object]:
    """Process-wide heavy objects (stateless): loaded once, shared across sessions."""
    with _cache_lock:
        if "stt" not in _cached:
            _cached["stt"] = WhisperSTT(cfg.stt.model_dir, cfg.stt.compute_type, cfg.stt.language)
            _cached["turn"] = SmartTurn(cfg.turn.model)
            _cached["tts"] = PiperTTS(cfg.tts.voice, cfg.tts.gain)
            _cached["store"] = Store(cfg.store.path)
            _cached["embedder"] = Embedder(str(ROOT / "models" / "embeddings"))
            _cached["router"] = ToolRouter(cfg)
        return _cached


def create_app(config_path: str) -> FastAPI:
    return build_app(load(config_path))


def build_app(cfg: Config) -> FastAPI:
    from fastapi.responses import FileResponse

    app = FastAPI(title="tobi v2 — offline s2s")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/app.js")
    async def app_js() -> FileResponse:
        return FileResponse(UI_DIR / "app.js")

    @app.get("/mic-worklet.js")
    async def worklet_js() -> FileResponse:
        return FileResponse(UI_DIR / "mic-worklet.js")

    @app.get("/style.css")
    async def style_css() -> FileResponse:
        return FileResponse(UI_DIR / "style.css")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        def trace(step: str) -> None:
            print(f"[ws] {step}", file=sys.stderr, flush=True)

        try:
            trace("accept")
            await ws.accept()
            trace("models")
            shared = _get_shared(cfg)
            trace("log")
            log = SessionLog(cfg.session.log_dir, cfg.session.save_utterance_audio)
            log.log("session_start", meeting=False)
            await _ws_session(ws, cfg, shared, log, trace)
        except Exception:  # noqa: BLE001 — print the real reason the socket died
            import traceback

            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

    return app


async def _ws_session(ws, cfg: Config, shared: dict, log: SessionLog, trace) -> None:  # noqa: ANN001
    from fastapi import WebSocketDisconnect

    loop = asyncio.get_running_loop()

    async def send_json(data: dict) -> None:
        try:
            await ws.send_json(data)
        except Exception:  # noqa: BLE001 — client gone mid-emit is normal
            pass

    async def send_audio(pcm: bytes) -> None:
        try:
            await ws.send_bytes(pcm)
        except Exception:  # noqa: BLE001
            pass

    state = {"meeting": False}
    display = DisplayBridge(endpoint_url="http://127.0.0.1:5000/api/event")
    display.start()
    if "router" in shared and shared["router"] is not None:
        shared["router"].on_event = display.emit

    def build_pipeline(meeting: bool) -> VoicePipeline:
        return VoicePipeline(
            cfg, log,
            emit=lambda d: asyncio.create_task(send_json(d)),
            emit_audio=lambda b: asyncio.create_task(send_audio(b)),
            brain=make_brain(cfg, shared),
            stt=shared["stt"], turn=shared["turn"], tts=shared["tts"],
            meeting_mode=meeting,
            store=shared["store"],
            display=display,
        )

    pipeline = build_pipeline(False)
    await send_json({"type": "hello", "log_dir": str(log.dir), "log_events": str(log.events_path),
                     "brain": cfg.brain.type,
                     "wake_words": cfg.wake.names if not state["meeting"] else []})

    async def dormancy_watch() -> None:
        while True:
            await asyncio.sleep(2)
            pipeline.check_dormancy()

    watcher = asyncio.create_task(dormancy_watch())

    async def finish_meeting(old: VoicePipeline) -> None:
        """Diarize + summarize the meeting that just ended; update the store."""
        sid = old._session_id
        try:
            buf = old.take_meeting_audio()
            if not buf:
                await send_json({"type": "meeting_report", "segments": [], "summary": "No speech captured."})
                return
            labels = await loop.run_in_executor(
                None, diarize_segments, buf, cfg.meeting.diarize.embedding, cfg.meeting.diarize.threshold)
            store = shared["store"]
            rows = [u for u in store.recent(limit=500) if u["session_id"] == sid and u["role"] == "speaker"]
            pairs = list(zip(rows, labels))
            store.set_speakers({r["id"]: lab for r, lab in pairs})
            entries = [{**r, "speaker": lab} for r, lab in pairs]
            summary = summarize_stub(entries, duration_ms=buf[-1][0])
            if sid is not None:
                store.set_summary(sid, summary)
            log.log("meeting_reported", segments=len(pairs), speakers=sorted(set(labels)))
            await send_json({"type": "meeting_report",
                             "segments": [{"t_ms": r["t_ms"], "speaker": lab, "text": r["text"]}
                                          for r, lab in pairs],
                             "summary": summary})
        except Exception as exc:  # noqa: BLE001 — report, never crash the session
            log.log("error", where="diarize", error=repr(exc))
            await send_json({"type": "error", "message": f"diarization failed: {exc}"})

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                # Chunk into 512-sample frames whatever the browser sent.
                for i in range(0, pcm.size - pcm.size % 512, 512):
                    pipeline.feed(pcm[i : i + 512])
            elif (text := msg.get("text")):
                import json

                ctl = json.loads(text)
                t = ctl.get("type")
                if t == "mute":
                    pipeline.set_muted(bool(ctl.get("value")))
                elif t == "playback":
                    pipeline.set_client_playback(bool(ctl.get("value")))
                elif t == "meeting":
                    on = bool(ctl.get("value"))
                    if on != state["meeting"]:
                        state["meeting"] = on
                        display.set_recording(on)
                        old = pipeline
                        old.close()
                        pipeline = build_pipeline(on)
                        log.log("session_start", meeting=on)
                        await send_json({"type": "hello", "log_dir": str(log.dir),
                                         "meeting": on,
                                         "wake_words": [] if on else cfg.wake.names})
                        if not on:
                            asyncio.create_task(finish_meeting(old))
                elif t == "ping":
                    await send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        watcher.cancel()
        pipeline.close()
        await display.stop()
        log.log("session_end")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "desktop.yaml"))
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    cfg = load(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    import uvicorn

    print(f"tobi v2 on http://{host}:{port}  (config: {args.config})")
    uvicorn.run(build_app(cfg), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
