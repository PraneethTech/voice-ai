"""Server-side hardware audio I/O using sounddevice (Microphone & Speaker).

Supports:
- Raspberry Pi (ReSpeaker Lite USB sound card or standard USB mic + MAX98357A / speaker)
- PC / Laptop (Built-in or USB microphone and default audio output device)

Rates:
- Mic input: 16,000 Hz, 1 channel, float32, chunk size 512 samples (~32 ms frame budget)
- Speaker out: 16,000 Hz, 1 channel, int16 PCM (synthesized by Piper TTS)
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

LOG = logging.getLogger("audio_io")
FRAME_SIZE = 512
SAMPLE_RATE = 16000


class AudioDeviceEngine:
    def __init__(
        self,
        on_mic_frame: Callable[[np.ndarray], None],
        on_playback_state: Callable[[bool], None] | None = None,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self.on_mic_frame = on_mic_frame
        self.on_playback_state = on_playback_state
        self.input_device = input_device
        self.output_device = output_device

        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._playback_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=100)
        self._playback_thread: threading.Thread | None = None
        self._running = False
        self._is_playing = False

    def start(self) -> None:
        if sd is None:
            LOG.error("sounddevice is not installed. Hardware audio unavailable.")
            return
        if self._running:
            return

        self._running = True

        # Input stream (Mic)
        def _in_callback(indata, frames, time_info, status):
            if status:
                LOG.debug("Mic status: %s", status)
            if self._running:
                # 512 mono float32 frame
                frame = indata[:, 0].astype(np.float32)
                try:
                    self.on_mic_frame(frame)
                except Exception as exc:
                    LOG.error("on_mic_frame error: %s", exc)

        try:
            self._in_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_SIZE,
                channels=1,
                dtype="float32",
                device=self.input_device,
                callback=_in_callback,
            )
            self._in_stream.start()
            LOG.info("Mic input stream started at %d Hz", SAMPLE_RATE)
        except Exception as exc:
            LOG.error("Failed to start mic input stream: %s", exc)

        # Output thread (Speaker)
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    def play_chunk(self, pcm_bytes: bytes) -> None:
        """Enqueue int16 PCM audio chunk for speaker playback."""
        if not self._running:
            return
        try:
            self._playback_queue.put_nowait(pcm_bytes)
        except queue.Full:
            LOG.warning("Audio playback buffer full, dropping chunk")

    def _playback_worker(self) -> None:
        if sd is None:
            return

        try:
            out_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=self.output_device,
            )
            out_stream.start()
        except Exception as exc:
            LOG.error("Failed to start speaker output stream: %s", exc)
            return

        while self._running:
            try:
                chunk = self._playback_queue.get(timeout=0.1)
                if chunk is None:
                    break

                if not self._is_playing:
                    self._is_playing = True
                    if self.on_playback_state:
                        self.on_playback_state(True)

                out_stream.write(np.frombuffer(chunk, dtype=np.int16))

                # Check if queue has drained
                if self._playback_queue.empty():
                    self._is_playing = False
                    if self.on_playback_state:
                        self.on_playback_state(False)

            except queue.Empty:
                if self._is_playing:
                    self._is_playing = False
                    if self.on_playback_state:
                        self.on_playback_state(False)
            except Exception as exc:
                LOG.error("Playback error: %s", exc)

        try:
            out_stream.stop()
            out_stream.close()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

        self._playback_queue.put_nowait(None)
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
