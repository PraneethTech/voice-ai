"""Full-duplex voice pipeline — the deep module behind the WebSocket transport.

States: DORMANT (wake word only) -> LISTENING (open mic, partials) -> THINKING
-> SPEAKING (TTS out, barge-in armed) -> LISTENING. Long silence returns to
DORMANT. Meeting mode skips wake word and brain: everything is transcribed.

feed() only runs VAD + wake word (a few ms) so the 32 ms frame budget holds;
every heavy step (smart-turn, STT, TTS, brain) runs via the executor.

Echo/barge-in honesty: the browser sends AEC-clean mic audio, so speaking over
the agent works on the web path. A raw server-side mic will hear the agent's
own voice — the Pi hardware path needs the echo guard (see docs/pi5_deploy.md).
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np

from agent_core.brain.base import Brain, Turn
from agent_core.config import Config
from agent_core.debuglog import SessionLog
from agent_core.display import DisplayBridge
from agent_core.stt import WhisperSTT
from agent_core.tts import PiperTTS
from agent_core.turn import SmartTurn
from agent_core.vad import SileroVAD, UtteranceTracker
from agent_core.wakeword import WakeWord

DORMANT = "dormant"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"

_SENTENCE_END = re.compile(r"[.!?]\s*$")

Emitter = Callable[[dict], None]      # JSON events
AudioOut = Callable[[bytes], None]    # int16 PCM chunks


class VoicePipeline:
    def __init__(
        self,
        cfg: Config,
        log: SessionLog,
        emit: Emitter,
        emit_audio: AudioOut,
        brain: Brain,
        *,
        stt: WhisperSTT,
        turn: SmartTurn,
        tts: PiperTTS,
        meeting_mode: bool = False,
        store=None,
        display: DisplayBridge | None = None,
    ) -> None:
        self.cfg = cfg
        self.log = log
        self.emit = emit
        self.emit_audio = emit_audio
        self.brain = brain
        self.stt = stt
        self.turn = turn
        self.tts = tts
        self.meeting_mode = meeting_mode
        self.store = store
        self.display = display
        self._turn_count = 0
        self._session_id = store.start_session("meeting" if meeting_mode else "assistant") if store else None
        self._meeting_audio: list[tuple[int, np.ndarray]] = []  # (t_ms, audio) — RAM only, discarded after diarization

        self.vad = SileroVAD(str(Path(cfg.root) / "models" / "silero_vad_v5.onnx"))
        self.tracker = UtteranceTracker(
            self.vad,
            threshold=cfg.vad.threshold,
            min_silence_ms=cfg.vad.min_silence_ms,
            speech_pad_ms=cfg.vad.speech_pad_ms,
            min_speech_ms=cfg.vad.min_speech_ms,
        )
        self.wake: WakeWord | None = None if meeting_mode else WakeWord(
            cfg.wake.dir, cfg.wake.names, cfg.wake.threshold
        )

        self.state = LISTENING if meeting_mode else DORMANT
        self.muted = False
        self.history: deque[Turn] = deque(maxlen=20)

        self._loop = asyncio.get_running_loop()
        self._t0 = time.monotonic()
        self._frames: list[np.ndarray] = []   # current utterance
        self._pending: list[np.ndarray] = []  # audio from held (incomplete) turns
        self._partial_at = 0                  # utterance sample count for the next partial
        self._partial_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._force_task: asyncio.Task | None = None
        self._barge = asyncio.Event()
        self._barge_armed = False          # arms once the mic is quiet during playback
        self._barge_speech_ms = 0.0        # continuous speech seen during playback
        self._barge_blocked_logged = False # one barge_blocked debug event per playback segment
        self._client_playing = False       # client is playing TTS (echo gate)
        self._client_playback_end = 0.0    # playback end + tail margin (mono time)
        self._utt_start_mono: float | None = None
        self._reply_end = 0.0
        self._last_voice = time.monotonic()
        self._wake_ok_at = 0.0          # wake refractory deadline
        self._wake_grace_until = 0.0    # drop wake-phrase tails right after a wake
        self._wake_probe_max = 0.0      # near-miss wake score logging (debugging aid)
        self._wake_probe_at = 0.0
        self._last_turn_ms: float | None = None
        self.emit({"type": "state", "state": self.state, "meeting": meeting_mode})
        if self.display:
            self.display.set_recording(meeting_mode)
            if not meeting_mode:
                self.display.set_mode("idle")

    # ------------------------------------------------------------------ feed

    def feed(self, frame: np.ndarray) -> None:
        """One 512-sample float32 frame from the transport (sync, ~ms budget)."""
        if self.muted:
            return
        if self.state == DORMANT:
            assert self.wake is not None
            scores = self.wake.process_frame(frame)
            if scores and self.wake.triggered(scores) and time.monotonic() >= self._wake_ok_at:
                self._wake_ok_at = time.monotonic() + self.cfg.wake.refractory_ms / 1000
                self._wake_grace_until = time.monotonic() + 1.5
                self.log.log("wake", scores=scores)
                self.emit({"type": "wake", "scores": scores})
                if self.display:
                    self.display.wake()
                self._set_state(LISTENING)
                self._last_voice = time.monotonic()
                self._speak_text(self.cfg.greeting)
            elif scores:
                # Near-miss visibility: below-threshold scores must be debuggable.
                now = time.monotonic()
                m = max(scores.values())
                if m > self._wake_probe_max:
                    self._wake_probe_max = m
                if now - self._wake_probe_at > 2.0 and self._wake_probe_max > 0.2:
                    self.log.log("wake_probe", score=round(self._wake_probe_max, 3),
                                 threshold=self.cfg.wake.threshold)
                    self._wake_probe_at = now
                    self._wake_probe_max = 0.0
            return

        utterance = self.tracker.process(frame)
        if self.tracker.in_utterance:
            self._last_voice = time.monotonic()

        if self.state == SPEAKING or self._client_playing:
            # Barge window covers the whole time the user hears our voice:
            # emission (SPEAKING) AND client playback afterwards — a short
            # reply finishes emitting long before the browser finishes playing
            # it. An interrupt needs barge_sustain_ms of CONTINUOUS
            # high-confidence speech (barge_prob): close clear speech scores
            # 0.9+, while distant chatter, background noise and AEC leakage
            # score lower and must not cut the reply. Also disarmed during the
            # wake grace window AND until the mic has been quiet once: the
            # echo tail of the previous reply must not kill the fresh reply.
            if self.tracker.in_utterance:
                if self.tracker.last_prob >= self.cfg.vad.barge_prob:
                    self._barge_speech_ms += self.tracker.frame_ms
                    if (self._barge_armed and time.monotonic() >= self._wake_grace_until
                            and not self._barge.is_set()
                            and self._barge_speech_ms >= self.cfg.vad.barge_sustain_ms):
                        self._barge_in()
                else:
                    # low-confidence frame: the confident run is broken
                    self._flush_barge_run()
            else:
                self._flush_barge_run()
                self._barge_armed = True
            if self.state == SPEAKING:
                return

        if self.tracker.in_utterance:
            if self._utt_start_mono is None:
                self._utt_start_mono = time.monotonic()
            self._frames.append(frame)
            self._maybe_partial()
        if utterance is not None:
            started = self._utt_start_mono
            self._utt_start_mono = None
            if started is not None and -0.7 < started - self._client_playback_end < 1.2 and len(utterance) < 1.5 * 16000:
                # Started inside the playback tail window: room echo of our own voice.
                self.log.log("echo_tail_dropped", dur_s=round(len(utterance) / 16000, 2))
            elif time.monotonic() < self._wake_grace_until and len(utterance) < 0.7 * 16000:
                # Tail of the wake phrase itself — not a user request.
                self.log.log("wake_tail_dropped", dur_s=round(len(utterance) / 16000, 2))
            else:
                self._on_utterance_end(utterance)

    # ---------------------------------------------------------- turn handling

    def _on_utterance_end(self, utterance: np.ndarray) -> None:
        if self.meeting_mode:
            self._turn_task = self._loop.create_task(self._meeting_segment(utterance))
            return
        if self._client_playing and not self._barge.is_set():
            # A user turn landing while our reply still plays (e.g. speech
            # shorter than the sustain window, like a quick "stop") stops the
            # playback too — Gemini-style: user speech always wins the floor.
            self._barge_in()
        if self._turn_task and not self._turn_task.done():
            # Previous turn is still deciding/playing; queue as a continuation.
            self._pending.append(utterance)
            self._schedule_force()
            return
        self._frames = []
        self._set_state(THINKING)
        self._turn_task = self._loop.create_task(self._process_turn(utterance))

    async def _process_turn(self, utterance: np.ndarray) -> None:
        try:
            result = await self._loop.run_in_executor(None, self.turn.predict, utterance)
            self._last_turn_ms = result.inference_ms
            self.log.log("turn_check", probability=round(result.probability, 3), complete=result.complete,
                         inference_ms=round(result.inference_ms))
            if not result.complete:
                # Hold the turn. If the user resumes, audio merges; if not, force after the delay.
                self._pending.append(utterance)
                self.log.log("turn_hold", pending_segments=len(self._pending))
                self._set_state(LISTENING)
                self._schedule_force()
                return
            if self._pending:
                utterance = np.concatenate([*self._pending, utterance])
                self._pending = []
            await self._respond_to(utterance)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a dead turn must never kill the session
            self.log.log("error", where="process_turn", error=repr(exc))
            self.emit({"type": "error", "message": f"turn failed: {exc}"})
            self._set_state(LISTENING)

    def _schedule_force(self) -> None:
        if self._force_task and not self._force_task.done():
            return
        delay = self.cfg.turn.force_after_ms / 1000

        async def force() -> None:
            await asyncio.sleep(delay)
            if self._pending and self.state in (LISTENING, THINKING) and not self.tracker.in_utterance:
                audio = np.concatenate(self._pending)
                self._pending = []
                self.log.log("turn_force")
                self._set_state(THINKING)
                self._turn_task = self._loop.create_task(self._respond_to(audio))

        self._force_task = self._loop.create_task(force())

    async def _respond_to(self, utterance: np.ndarray) -> None:
        self._turn_count += 1
        utt_id = f"u{self._turn_count}"
        resp_id = f"a{self._turn_count}"
        t_eot = time.monotonic()
        transcript = await self._loop.run_in_executor(None, self.stt.transcribe, utterance)
        self.log.log("final", text=transcript.text, lang=transcript.language,
                     stt_ms=round(transcript.inference_ms), dur_s=round(len(utterance) / 16000, 2))
        if self.log.save_utterance_audio:
            self.log.save_wav("utterance", utterance)
        self.emit({"type": "final", "text": transcript.text, "lang": transcript.language})
        if self.display:
            self.display.user_final(utt_id, transcript.text, recording=self.meeting_mode)
        if not transcript.text:
            self._set_state(LISTENING)
            return

        self.history.append(Turn("user", transcript.text))
        t_llm0 = time.perf_counter()
        first_text_ms: list[float | None] = [None]
        word_count = 0

        if self.display:
            self.display.assistant_start(resp_id)

        async def timed_words():
            nonlocal word_count
            async for word in self.brain.respond(transcript.text, list(self.history)):
                word_count += 1
                if first_text_ms[0] is None:
                    first_text_ms[0] = (time.perf_counter() - t_llm0) * 1000
                if self.display:
                    self.display.assistant_token(resp_id, word)
                yield word

        first_audio_ms, reply_text = await self._play_sentences(
            self._sentences(timed_words()), t_eot
        )
        if self.store is not None and transcript.text:
            self.store.add_utterance(self._session_id, round((time.monotonic() - self._t0) * 1000),
                                     "user", transcript.text, lang=transcript.language)
        if reply_text:
            self.history.append(Turn("assistant", reply_text))
            if self.store is not None:
                self.store.add_utterance(self._session_id, round((time.monotonic() - self._t0) * 1000),
                                         "assistant", reply_text)
        if first_audio_ms is not None:
            self.emit({"type": "latency", "eot_to_first_audio_ms": round(first_audio_ms),
                       "stt_ms": round(transcript.inference_ms),
                       "llm_ttft_ms": round(first_text_ms[0]) if first_text_ms[0] is not None else None,
                       "turn_ms": round(self._last_turn_ms) if self._last_turn_ms is not None else None})
        if self.display and not self._barge.is_set():
            tot = (time.monotonic() - t_eot) * 1000
            gen = tot - (first_text_ms[0] or 0)
            metrics = {
                "ttft_ms": round(first_text_ms[0] or 0),
                "total_ms": round(tot),
                "gen_ms": round(gen),
                "tpot_ms": round(gen / max(1, word_count), 1),
                "tokens_per_sec": round(word_count / max(0.1, gen / 1000), 1),
                "input_tokens": len(transcript.text.split()),
                "output_tokens": word_count,
            }
            self.display.assistant_done(resp_id, metrics)
        self._set_state(LISTENING)

    async def _meeting_segment(self, utterance: np.ndarray) -> None:
        t_ms = round((time.monotonic() - self._t0) * 1000)
        transcript = await self._loop.run_in_executor(None, self.stt.transcribe, utterance)
        self.log.log("meeting_segment", text=transcript.text, t_ms=t_ms)
        if self.log.save_utterance_audio:
            self.log.save_wav("meeting_segment", utterance)
        if transcript.text:
            self._meeting_audio.append((t_ms, utterance))
            if self.store is not None:
                self.store.add_utterance(self._session_id, t_ms, "speaker", transcript.text,
                                         speaker="SPEAKER_?", lang=transcript.language)
            self.emit({"type": "meeting_segment", "text": transcript.text, "t_ms": t_ms})
            if self.display:
                self.display.user_final(f"m{t_ms}", transcript.text, recording=True)

    def take_meeting_audio(self) -> list[tuple[int, np.ndarray]]:
        """Hand the meeting buffer to the diarizer and empty it."""
        buf, self._meeting_audio = self._meeting_audio, []
        return buf

    # ------------------------------------------------------------- speaking

    async def _play_sentences(self, sentences: AsyncIterator[str], t_eot: float) -> tuple[float | None, str]:
        """Synthesize + emit sentence by sentence. Returns (first_audio_ms, reply_text).

        first_audio_ms is None when barged in (reply aborted).
        """
        self._barge.clear()
        self._barge_armed = not self.tracker.in_utterance
        self._barge_speech_ms = 0.0        # fresh playback segment: reset sustain + blocked-log
        self._barge_blocked_logged = False
        first_audio_ms: float | None = None
        reply_text = ""
        self._set_state(SPEAKING)
        try:
            async for sentence in sentences:
                if self._barge.is_set():
                    break
                if not sentence.strip():
                    continue
                reply_text += sentence
                self.emit({"type": "reply_chunk", "text": sentence})  # live streaming text
                audio = await self._loop.run_in_executor(None, self.tts.synth_all, sentence)
                if self._barge.is_set():
                    break
                if first_audio_ms is None:
                    first_audio_ms = (time.monotonic() - t_eot) * 1000
                self.emit_audio((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
                await asyncio.sleep(0)  # let the transport flush between sentences
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.log("error", where="play_sentences", error=repr(exc))
            self.emit({"type": "error", "message": f"tts failed: {exc}"})
        if self._barge.is_set():
            return None, reply_text
        self._set_state(LISTENING)  # greeting/turn finished speaking: open the mic again
        self._reply_end = time.monotonic()
        return first_audio_ms, reply_text

    def _speak_text(self, text: str) -> None:
        """Short agent line (greeting). Barge-in can cut it."""

        async def go() -> None:
            await self._play_sentences(_one(text), time.monotonic())

        self._turn_task = self._loop.create_task(go())

    def _sentences(self, word_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Merge a streamed word async-iterator into sentence-sized chunks."""

        async def gen() -> AsyncIterator[str]:
            buf = ""
            async for word in word_stream:
                buf += word
                if _SENTENCE_END.search(buf) or len(buf) >= 140:
                    yield buf
                    buf = ""
            if buf.strip():
                yield buf

        return gen()

    # -------------------------------------------------------------- barge-in

    def _flush_barge_run(self) -> None:
        """A confident-speech run ended (silence or low-confidence frame):
        log it once per playback segment if it never reached the sustain."""
        if self._barge_speech_ms and not self._barge_blocked_logged:
            self.log.log("barge_blocked", speech_ms=round(self._barge_speech_ms))
            self._barge_blocked_logged = True
        self._barge_speech_ms = 0.0

    def _barge_in(self) -> None:
        self.log.log("barge_in", speech_ms=round(self._barge_speech_ms))
        self.emit({"type": "bargein"})
        if self.display:
            self.display.assistant_interrupted(f"a{self._turn_count}")
        self._barge.set()
        if self._force_task and not self._force_task.done():
            self._force_task.cancel()
        if self._turn_task and not self._turn_task.done():
            # Cancel the whole turn: CancelledError unwinds _play_sentences and
            # the brain/TTS generator chain, aborting the in-flight LLM stream.
            self._turn_task.cancel()
        self._turn_task = None
        self._pending = []  # a held/continuation turn must not resurrect after the barge
        self._set_state(LISTENING)

    # -------------------------------------------------------------- partials

    def _maybe_partial(self) -> None:
        samples = sum(len(f) for f in self._frames)
        if samples < self._partial_at:
            return
        self._partial_at = samples + self.cfg.stt.partial_every_ms * 16
        if self._partial_task and not self._partial_task.done():
            return
        audio = np.concatenate(self._frames)

        async def run_partial() -> None:
            transcript = await self._loop.run_in_executor(None, self.stt.transcribe, audio)
            if transcript.text:
                self.emit({"type": "partial", "text": transcript.text})
                self.log.log("partial", text=transcript.text)
                if self.display:
                    self.display.user_partial(f"u{self._turn_count + 1}", transcript.text, recording=self.meeting_mode)

        self._partial_task = self._loop.create_task(run_partial())

    # ------------------------------------------------------------- lifecycle

    def check_dormancy(self) -> None:
        if self.state == LISTENING and not self.meeting_mode and not self.tracker.in_utterance:
            if time.monotonic() - self._last_voice > self.cfg.dormancy.timeout_s:
                self.log.log("dormant")
                self._set_state(DORMANT)
                self._pending = []
                if self.display:
                    self.display.set_mode("idle")

    def set_client_playback(self, playing: bool) -> None:
        """Client reports TTS playback start/stop (echo gate bookkeeping)."""
        self._client_playing = playing
        if not playing:
            self._client_playback_end = time.monotonic() + 0.6  # room tail margin
        self.log.log("playback", on=playing)

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self.log.log("mute", muted=muted)
        self.emit({"type": "muted", "muted": muted})

    def close(self) -> None:
        for t in (self._turn_task, self._partial_task, self._force_task):
            if t and not t.done():
                t.cancel()
        if self.store is not None and self._session_id is not None:
            try:
                self.store.end_session(self._session_id)
            except Exception:  # noqa: BLE001 — session close must never crash shutdown
                pass
            self._session_id = None

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.log.log("state", state=state)
            self.emit({"type": "state", "state": state})
            if self.display and not self.meeting_mode:
                if state == DORMANT:
                    self.display.set_mode("idle")
                elif state == LISTENING:
                    self.display.set_mode("listening")
                elif state == THINKING:
                    self.display.set_mode("thinking")
                elif state == SPEAKING:
                    self.display.set_mode("speaking")


async def _one(text: str) -> AsyncIterator[str]:
    yield text
