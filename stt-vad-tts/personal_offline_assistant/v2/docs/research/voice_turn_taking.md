# Voice-mode turn-taking: how the big players do it, and what tobi v2 ships

Researched 2026-10-08/09 while fixing "it answers after a short pause" and
"interrupting doesn't stop it". Sources are official docs where reachable;
claims we could not open directly are marked [summarized].

## OpenAI — Realtime API / ChatGPT voice

- Two turn-detection modes (VAD docs, developers.openai.com):
  - **Server VAD** (default): commits the user turn after `silence_duration_ms`
    of silence — **default 500 ms**; `prefix_padding_ms` 300 (audio kept before
    speech start); `threshold` 0.5.
  - **Semantic VAD** (preview): a model scores whether the user *finished
    speaking based on the words*, exposed as `eagerness` (`low`/`auto`/
    `medium`/`high`). `low` waits longer — community reports even `low` can
    cut users off, so eagerness low + patience is the safe direction.
- Interruption: VAD emits `input_audio_buffer.speech_started` while a response
  is streaming; the in-flight response is truncated/cancelled server-side
  (`response.cancel`, output audio truncated). Only assistant audio already
  *played* before the interruption stays in conversation context.
- Practical tuning guide (Koenig Solutions): if the agent answers during the
  user's pause, start at `silence_duration_ms: 700` and lower after real-user
  testing.

## Google — Gemini Live API

- Server-side VAD runs continuously; when the user speaks over the model,
  **ongoing generation is cancelled and discarded** — an `interrupted`-type
  signal tells the client to stop playing immediately (Cloud Live API ref,
  docs.cloud.google.com; dev forum discuss.ai.google.dev).
- The discarded (never-played) part of the generation does **not** persist into
  context — only what the user actually heard.
- Barge-in robustness depends on the client's echo control; complaints on the
  forum are all echo/AEC-related, not model-related.

## pipecat / LiveKit smart-turn

- pipecat docs (docs.pipecat.ai): VAD detects speech vs silence but "a pause
  doesn't mean the user is done" — semantic turn models (their `smart-turn`)
  run *on top of* VAD-end signals and decide hold vs commit. Typical configs
  keep the VAD silence window short-ish and let the semantic model add the
  patience.
- Gotcha from a Feb 2026 GitHub issue: smart-turn silently breaks at 8 kHz —
  feed it 16 kHz (we do).

## What tobi v2 shipped (2026-10-09)

| concern | big-player reference | tobi v2 |
|---|---|---|
| silence before end-of-turn | OpenAI server VAD 500 ms | `vad.min_silence_ms: 500` (was 300) |
| semantic "done?" decision | OpenAI semantic VAD / pipecat smart-turn | Smart-Turn v3.2, threshold 0.5 → **0.65** (hold on doubt) |
| force-commit a stalled hold | — | `turn.force_after_ms: 1200` (was 2000) |
| interruption cancels generation | Gemini Live cancels + discards | `_barge_in()` now cancels the turn task → Groq stream aborts; partial reply never reaches history/store |
| interruption sensitivity | Gemini VAD sensitivity | `vad.barge_sustain_ms: 300` desktop / 500 Pi of *continuous* speech before barge |
| echo control | client AEC (ChatGPT web, Gemini) | mic stays open during playback; browser `echoCancellation: true` + sustain guard; `suppressMic` kept as debug fallback |
| latency budget | — | measured: STT base.en 554 ms median, Smart-Turn ~50 ms, TTS medium 251 ms, LLM TTFT ~545-860 ms |

## Model benchmarks (real user audio, this desktop, int8, median of 3)

- STT tiny.en vs base.en on 6 logged utterances: tiny 254 ms vs base 554 ms
  (2 s clips); base clearly more accurate on real speech and does not
  hallucinate "Thank you." into near-silence (tiny does). Pi 5 estimate:
  tiny ≈ 0.8–1.3 s — stays on tiny for RAM reasons.
- TTS lessac-low 181 ms vs lessac-medium 251 ms for a 4 s sentence (RTF 0.05
  vs 0.06) — medium is effectively free on desktop; both fine on Pi.
- Smart-Turn on VAD-finalized real utterances: 0.83–0.99, all above the new
  0.65 threshold (expected — these are finished turns, not mid-thought holds).

## Sources

- OpenAI Realtime VAD + Semantic VAD: https://developers.openai.com (Realtime API → voice activity detection)
- Tuning advice: https://www.koenig-solutions.com (Realtime VAD tuning blog)
- Gemini Live interruption semantics: https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/multimodal-live and https://discuss.ai.google.dev
- pipecat turn detection: https://docs.pipecat.ai ; smart-turn models: https://huggingface.co/pipecat-ai
