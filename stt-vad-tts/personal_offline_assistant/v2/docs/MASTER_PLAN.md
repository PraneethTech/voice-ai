# tobi v2 — Master Plan (end to end)

Sovereign, offline, hands-free, full-duplex speech-to-speech companion for Raspberry Pi 5 (4GB).
Core is BUILT and VERIFIED on desktop (this document tells you what exists, what to do next, and how).

## 1. Product definition

One sentence: a device that listens hands-free, wakes to its name, transcribes live, ends turns
with a neural model (no fixed timer), streams replies out loud, gets interrupted by voice, goes
dormant in silence, records meetings with speaker labels, remembers past talks, and only ever
touches the network for lookups you asked for (weather, search, telegram send).

Boundary (matches the problem statement): all reasoning runs on-device. Raw audio never leaves
the device and is not stored — only transcripts (SQLite). Online calls: weather + search + the
user-requested telegram send. Everything else is local.

## 2. Hardware bill of materials + availability (India / Hyderabad)

| # | Part | Role | Where to get (India) |
|---|------|------|----------------------|
| 1 | Raspberry Pi 5 (4GB) | compute | robu.in, SilverEle (Hyderabad, Gachibowli area), amazon.in |
| 2 | Seeed ReSpeaker Lite 2-mic (XU316) | mic + 5W speaker amp, USB sound card — no driver | robu.in, thingbits.in, Seeed Bazaar (ship to India) |
| 2b | Fallback: ReSpeaker 2-Mics Pi HAT | if Lite unavailable; needs driver script, has 3 LEDs | robu.in (widely stocked) |
| 2c | Fallback: any USB conference speakerphone | zero-config; weakest far-field | amazon.in |
| 3 | Passive speaker 4-8Ω 3-5W, JST-PH2.0 or bare wire | plugs into ReSpeaker Lite amp port | robu.in |
| 4 | Official Pi 27W USB-C PSU | power (required for Pi 5) | robu.in / amazon.in |
| 5 | Active cooler for Pi 5 | thermals | with the board |
| 6 | SanDisk Extreme/A2 microSD 128GB | OS + models + SQLite transcripts | local |
| 7 | 3.4" SPI TFT (treat as 3.5" ILI9486 class) | display (optional for demo) | robu.in — see wiring.md note |
| 8 | MTS-202 DPDT toggle | physical mute (GPIO pole; 2nd pole spare) | robu.in |
| 9 | KY-040 rotary encoder | volume knob | robu.in |
| 10 | Resistor kit + breadboard + jumpers + 5mm LEDs (green/red) | indicators | local |
| 11 | (optional) WS2812 / RGB LED | "whisper-hat" style mode light: green=live, red=muted, blue=thinking, purple=meeting | robu.in |

The Jdaie Lin video (kFmhSTh167U) uses a mic board you can't buy — our ReSpeaker Lite is the
equivalent path and is a plain USB sound card on the Pi (Seeed wiki: respeaker_lite_pi5).

## 3. Architecture (as built)

```
browser mic (AEC) ──ws──► FastAPI ──► VoicePipeline (asyncio state machine)
                                        ├─ WakeWord openWakeWord (DORMANT)
                                        ├─ Silero VAD v5 ONNX (always)
                                        ├─ UtteranceTracker (hysteresis + padding)
                                        ├─ Smart-Turn v3.2 ONNX (end of turn)
                                        ├─ faster-whisper tiny.en (live partials + final)
                                        ├─ Brain seam ── StubBrain(+ToolRouter, +memory)
                                        ├─ Piper TTS (sentence streaming)
                                        ├─ barge-in: VAD over playback stops TTS
                                        └─ dormancy: silence → DORMANT
meeting mode ──► same VAD/STT, no wake word ──► diarize (sherpa-onnx) ──► summary
memory ──► SQLite: transcripts + FTS5/BM25 + MiniLM embeddings + RRF hybrid retrieval
```

Torch-free by design (onnxruntime + ctranslate2 + piper). RAM on Pi: ~1.0–1.4GB core;
diarization loads on demand; leaves room for the SLM.

Folder map: agent_core/ (vad, turn, wakeword, stt, tts, brain/, tools/, meeting, store, pipeline,
config, debuglog) · server/ (ws_server.py, ui/) · tools/ (download_models, check_offline,
e2e_offline, gen_fixtures, bench) · tests/ (26 tests + fixtures) · docs/ · vendors/ (3 reference
clones, kept intact).

## 4. Phases

- [x] P0 git hygiene (legacy preserved on `v2-legacy` branch)
- [x] P1 stack + models (one-time download; `tools/check_offline.py` proves runtime is offline)
- [x] P2 VAD + Smart-Turn, pinned by fixtures (see docs/TESTING.md for the synthetic-prosody caveat)
- [x] P3 STT + StubBrain + Piper + offline E2E (wav → transcript → reply wav)
- [x] P4 wake word + dormancy + full-duplex state machine (barge-in, wake-tail drop, refractory)
- [x] P5 WebSocket transport + clean UI (live partials, streaming reply, latency chip, meeting panel)
- [x] P6 desktop E2E verified live: wake 0.71-0.83 → greeting intact → turn → 562ms EOT→voice → barge-in
- [x] P7 store + hybrid RAG
- [x] P8 tools (weather / search / telegram / memory) behind the router seam
- [x] P9 meeting mode + diarization + stub summary
- [x] P10 load to Pi: see docs/pi5_deploy.md; hardware per docs/wiring.md
- [x] P10.5 echo/barge hardening from live logs: client mic gate during playback, echo-tail
  discard, barge armed-after-quiet; GroqBrain dev bridge (.env GROQ_API_KEY) for E2E testing
  with real reasoning + RAG + tool phrasing (desktop config only; Pi stays sovereign)
- [ ] P11 brain swap: StubBrain → LlmBrain (llama.cpp OpenAI-compatible on localhost)
  - LFM2.5-1.2B-Instruct: LFM Open License v1.0 — free below $10M revenue (hackathon fine), 32K ctx
    (not 2K), llama.cpp supported. https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
  - Qwen3-0.6B / 1.7B: Apache-2.0 (no clause), 32K ctx, ~20-40 tok/s est. on Pi 5 Q4 — safest sovereign pick
  - Long transcripts: chunked map-reduce summarization (2-3k-token speaker-labeled chunks)
- [ ] P12 multilingual stretch: whisper `small` multilingual + per-segment language detect → Piper
  te_IN (maya/venkatesh) + hi_IN (pratham) voice switch; Smart-Turn already scores 90.11% Hindi
- [ ] P13 wake word: train `hey_tobi` offline (openWakeWord synthetic pipeline; ~1hr; tune on your voice);
  drop the .onnx into models/wakeword/ — config slot already points there
- [ ] P14 Pi hardware: MTS-202 mute → GPIO + LEDs, meeting button + purple LED, KY-040, TFT (wiring.md)

## 5. Demo script (hackathon)

1. Mute switch off/on → red/green LED (physical privacy).
2. "Hey Jarvis" → greeting → ask anything → streamed voice reply (all on-device; show UI).
3. "What's the weather in Hyderabad" → clear ONLINE moment (open-meteo), reasoning still local.
4. "Search for AI models released this month" → DuckDuckGo → spoken + on screen.
5. "Send me the summary of our conversation" → telegram arrives on your phone.
6. Meeting mode button (purple): 3-4 people speak → end → diarized transcript + summary.
7. "What did we discuss earlier?" → hybrid memory retrieval answers from local transcripts.
8. Privacy story: no raw audio stored, no cloud reasoning, offline audit passes with sockets blocked.

## 6. Sources (verified October 2026)

- huggingface/speech-to-speech (Apache-2.0) — architecture reference; Smart-Turn port source. 13.4k stars.
  https://github.com/huggingface/speech-to-speech (their `main` is 16GB-class; we ported the algorithms)
- pipecat-ai/smart-turn v3.2 (BSD-2) — 8MB ONNX, 92.6% EN / 90.1% HI. https://huggingface.co/pipecat-ai/smart-turn-v3
- Silero VAD v5 (MIT) — 64-sample-context call convention. https://github.com/snakers4/silero-vad
- openWakeWord (Apache-2.0 code; models CC-BY-NC-SA). https://github.com/dscripka/openWakeWord
- faster-whisper (MIT, CTranslate2). https://github.com/SYSTRAN/faster-whisper — tiny.en int8 75.5MB
- Piper (piper1-gpl, GPL-3.0) — voices incl. hi_IN, te_IN. https://github.com/OHF-Voice/piper1-gpl
- sherpa-onnx (Apache-2.0) — speaker diarization/embeddings, aarch64 wheels. https://github.com/k2-fsa/sherpa-onnx
- Seeed wiki — ReSpeaker Lite + Pi 5 USB audio: https://wiki.seeedstudio.com/respeaker_lite_pi5/
- wyoming-satellite — mic/LED/mute plumbing reference. https://github.com/rhasspy/wyoming-satellite
- LFM Open License v1.0: https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/LICENSE
- Qwen3-0.6B (Apache-2.0): https://huggingface.co/Qwen/Qwen3-0.6B
- open-meteo (no key): https://open-meteo.com · Telegram Bot API (free): https://core.telegram.org/bots
- Video walkthroughs: offline Pi5 assistant kFmhSTh167U; NetworkChuck XvbVePuP7NY; ReSpeaker kit k1eo25SAq9M;
  KY-040 4kypUKRMGYk; SPI LCD vCAGzLGTUk4; switch+LED yL5BNA_Ex6s
