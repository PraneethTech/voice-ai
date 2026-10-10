"""Display bridge for SPAN 240x320 screen UI.

Bridges the VoicePipeline (WakeWord -> VAD -> STT -> Brain/Agent -> TTS) with
the SPAN UI display server over HTTP POST /api/event or in-process publish().

Events are delivered asynchronously via an internal non-blocking queue so
display networking never blocks or delays the real-time audio frame budget.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

LOG = logging.getLogger("display_bridge")


class DisplayBridge:
    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:5000/api/event",
        *,
        in_process_publish: Callable[[dict[str, Any]], None] | None = None,
        enabled: bool = True,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.in_process_publish = in_process_publish
        self.enabled = enabled
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._worker_task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None
        self._is_running = False

    def start(self) -> None:
        """Start background sender loop."""
        if self._is_running:
            return
        self._is_running = True
        self._queue = asyncio.Queue(maxsize=500)
        self._client = httpx.AsyncClient(timeout=1.5)
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Flush and stop background sender."""
        self._is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None

    def emit(self, event: dict[str, Any]) -> None:
        """Queue event for async delivery to the display screen."""
        if not self.enabled:
            return
        if self.in_process_publish:
            try:
                self.in_process_publish(event)
                return
            except Exception as exc:
                LOG.warning("in_process_publish error: %s", exc)

        if not self._is_running:
            self.start()

        if self._queue is not None:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                LOG.warning("Display queue full, dropping event: %s", event.get("type"))

    async def _worker(self) -> None:
        while self._is_running:
            if self._queue is None:
                await asyncio.sleep(0.05)
                continue
            try:
                event = await self._queue.get()
                if self._client is not None:
                    try:
                        await self._client.post(self.endpoint_url, json=event)
                    except Exception:
                        # Display offline or unreachable; silently drop without crashing
                        pass
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                LOG.debug("Display sender error: %s", exc)

    # ------------------------------------------------------------- helper API
    def wake(self) -> None:
        self.emit({"type": "wake"})

    def user_partial(self, uid: str, text: str, recording: bool = False) -> None:
        evt: dict[str, Any] = {"type": "user_partial", "id": uid, "text": text}
        if recording:
            evt["recording"] = True
        self.emit(evt)

    def user_final(self, uid: str, text: str, recording: bool = False) -> None:
        evt: dict[str, Any] = {"type": "user_final", "id": uid, "text": text}
        if recording:
            evt["recording"] = True
        self.emit(evt)

    def tool_start(self, tid: str, name: str) -> None:
        self.emit({"type": "tool", "id": tid, "name": name, "status": "start"})

    def tool_done(self, tid: str, name: str, detail: str = "") -> None:
        self.emit({"type": "tool", "id": tid, "name": name, "status": "ok", "detail": detail})

    def tool_error(self, tid: str, name: str, detail: str = "") -> None:
        self.emit({"type": "tool", "id": tid, "name": name, "status": "error", "detail": detail})

    def reasoning(self, text: str = "") -> None:
        self.emit({"type": "reasoning", "text": text})

    def assistant_start(self, aid: str) -> None:
        self.emit({"type": "assistant_start", "id": aid})

    def assistant_token(self, aid: str, text: str) -> None:
        self.emit({"type": "assistant_token", "id": aid, "text": text})

    def assistant_done(self, aid: str, metrics: dict[str, Any] | None = None) -> None:
        evt: dict[str, Any] = {"type": "assistant_done", "id": aid}
        if metrics:
            evt["metrics"] = metrics
        self.emit(evt)

    def assistant_interrupted(self, aid: str) -> None:
        self.emit({"type": "assistant_interrupted", "id": aid})

    def set_recording(self, on: bool) -> None:
        self.emit({"type": "recording", "on": on})

    def set_mode(self, mode: str) -> None:
        self.emit({"type": "mode", "mode": mode})

    def error(self, message: str) -> None:
        self.emit({"type": "error", "message": message})
