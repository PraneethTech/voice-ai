# v2 vendor pins (read-only reference, never edit, never delete)

Cloned with `git clone --depth 1`. Pinned 2026-10-07.

| vendor | url | sha | date | note |
|---|---|---|---|---|
| huggingface-speech-to-speech | https://github.com/huggingface/speech-to-speech.git | `20d41cc2bc309ecaf10d1f24043d9d20160d1b94` | 2026-10-06 | Canonical VAD->STT->LLM->TTS, OpenAI Realtime WS, offline via HF_HUB_OFFLINE + llama.cpp. Proves streaming pipeline + turn detection. |
| on-device-s2s | https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI.git | `fb93dde47704c828b2f615f28e4c34d235d9e5cc` | 2025-04-17 | On-CPU recipe (VAD + Kokoro/Piper + Ollama). Closest Pi 5 4GB reference. |
| xtalk | https://github.com/xcc-zach/xtalk.git | `5f0d9959edf1026588246efbed827b078cbb114c` | 2026-10-03 | Pure-Python full-duplex cascaded, interruptible. Barge-in reference. |
| mattpocock-skills | https://github.com/mattpocock/skills.git | `b0618bc436ad893b3c5e84e55fba86586d34a404` | 2026-10-08 | Agent working skills (tdd, diagnosing-bugs, codebase-design, code-review) vendored into docs/skills/. |

To refresh: `git -C vendors/<name> fetch --depth 1 origin main && git -C vendors/<name> rev-parse HEAD` then update this table.
