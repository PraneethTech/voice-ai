# Testing guide — tobi v2

## Automated (no mic needed)

```
python tools/check_offline.py     # sovereignty gate: all models load with sockets BLOCKED
python -m pytest tests/ -q        # 26 tests: config drift, VAD, turn, E2E, WS, store, tools
python tools/e2e_offline.py       # fixture wav -> transcript + reply wav (data/e2e_reply.wav)
```

Fixtures are Piper-synthesized (tools/gen_fixtures.py regenerates them).

Known fixture limit (pinned by a test, do not "fix" blindly): Piper gives every clip final
declarative prosody, so a truncated sentence still scores "complete" on Smart-Turn. Real humans
trail off — verify that case by ear below. The important automated signals: "Umm" HOLDS the turn,
completes complete, noise/silence produce nothing.

## Manual (your real mic)

1. `python -m server.ws_server --config configs/desktop.yaml` → open http://127.0.0.1:8000
2. Click "enable microphone" (one click; browser AEC makes barge-in safe).
3. Acceptance list:
   - [ ] page shows "dormant" + wake hint
   - [ ] say "Hey Jarvis" → greeting plays, pill turns listening
   - [ ] speak → live partial transcript appears while talking
   - [ ] finish a sentence → reply streams as text AND audio (~1s or less)
   - [ ] mid-sentence pause → it WAITS (smart-turn), then continues when you continue
   - [ ] talk over the reply → audio stops within ~300ms, back to listening
   - [ ] stay silent 90s → dormant; wake word re-activates
   - [ ] music/typing noise → no false wake, no phantom utterances
   - [ ] "what's the weather in Hyderabad" → online weather, spoken + shown
   - [ ] "send me the summary on telegram" → message on your phone (set TOBI_TG_TOKEN/TOBI_TG_CHAT first)
   - [ ] meeting mode → purple state, segments listed → end meeting → speaker labels + summary
   - [ ] logs/session_*/events.jsonl exists for EVERY attempt (wake scores, turns, latencies)

## Debugging a bad mic session

Every session auto-writes logs/session_<ts>/events.jsonl (+ utterance wavs): wake scores,
state phases, turn probabilities, partials/finals, barge-ins, echo drops, latencies. One file
per session — grep it, don't guess. Key events: `echo_tail_dropped` / `wake_tail_dropped`
(suppressed ghosts), `turn_hold` / `turn_force` (end-of-turn decisions), `barge_in`.

## Echo and barge-in (read this before testing)

Open speakers + mic in the same room = the agent hears itself. Three layers handle it:
1. Client echo gate (default ON): mic frames are NOT sent while TTS plays + 300ms tail.
2. Server drops any utterance that started inside the playback tail window (`echo_tail_dropped`).
3. Barge-in arms only after the mic is quiet — a reply's echo tail can't kill a fresh reply.

Trade-off: with the gate ON, you cannot interrupt by voice on open speakers (mic is closed while
it talks; the reply finishes, then ~0.3s later you talk). For TRUE voice barge-in, use a headset
or real Chrome/Edge with working AEC: set `suppressMic = false` in server/ui/app.js — the
server-side barge-in path is fully armed. On the Pi, the ReSpeaker's DSP/software echo guard
takes this role (see pi5_deploy.md).

## Groq dev bridge (temporary, for E2E testing)

`configs/desktop.yaml` has `brain.type: groq`. Copy `.env.example` to `.env`, set `GROQ_API_KEY`,
restart the server — replies, memory answers and summaries then come from Groq with your local
RAG context injected. Without a key it logs a warning and uses the offline stub. The Pi config
stays `stub` (sovereign); `llm` (local llama.cpp) replaces both later.

## Benchmarks

Desktop (this laptop, measured): EOT → first audio = 562ms with the stub brain; Smart-Turn
~50-60ms per check; Silero ~1ms/frame. Pi targets: EOT → voice < 2.5s; record numbers in
docs/pi5_deploy.md when the unit arrives.
