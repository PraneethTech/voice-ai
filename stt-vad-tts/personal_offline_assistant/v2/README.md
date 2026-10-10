# tobi v2 — offline full-duplex speech-to-speech (Pi 5 4GB)

Wake word → open mic → live transcription → neural end-of-turn → streaming reply → voice barge-in
→ dormancy → meeting mode with speaker labels → memory that answers from past transcripts.
100% offline after a one-time model download; torch-free (onnxruntime + ctranslate2 + piper).
Only weather/search/telegram touch the network — that boundary is the product.

## Setup (Windows, fresh clone)

Prerequisite: Python 3.11+ from https://www.python.org/downloads/ (tick "Add python.exe to PATH").

```
git clone https://github.com/kaidoreddy-hash/personal_offline_assistant.git
cd personal_offline_assistant\v2
setup.bat
```

`setup.bat` does everything, in order: creates `.venv-v2`, installs `requirements.txt`,
copies `.env.example` to `.env` (if missing — put your `GROQ_API_KEY` there), downloads
models once (`tools/download_models.py`), runs the test suite, then starts the server.
Open http://127.0.0.1:8000 → "enable microphone" → say "hey jarvis".

Manual equivalent (if a `setup.bat` step fails and you want to run it by hand):

```
cd v2
python -m venv .venv-v2
.venv-v2\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   & :: then edit .env
python tools/download_models.py   &:: once, then runtime is offline forever
python tools/check_offline.py     &:: proves it: all models load with sockets blocked
python -m pytest tests/ -q        &:: 38 tests green
python -m server.ws_server --config configs/desktop.yaml
```

Never commit `.env` (gitignored) — keys stay on your machine only.

## Brains: stub vs Groq vs local LFM 2.5

The LLM is a seam, not the pipeline: any class implementing `Brain.respond()`
(an async generator yielding text chunks, see `agent_core/brain/base.py`) can think.
Switching brains = 2 lines in the yaml + restart. Nothing else changes.

| `brain.type` | class | needs | when to use |
|---|---|---|---|
| `stub` | `StubBrain` | nothing | default on Pi; fully offline hardcoded replies; proves the loop |
| `groq` | `GroqBrain` | `GROQ_API_KEY` in `v2/.env` | dev testing with a real LLM (falls back to stub without a key) |
| `llm` (not built yet) | `brain/llm_openai.py` → localhost llama.cpp | llama.cpp server + LFM-2.5 weights | the Pi product path |

### Switching to LFM 2.5 running locally

1. Get LFM-2.5 weights (GGUF) and a `llama-server` binary (llama.cpp releases).
2. Serve it OpenAI-compatibly: `llama-server -m lfm2.5.gguf --port 8080`
   (chat endpoint lands at `http://127.0.0.1:8080/v1/chat/completions`).
3. Add `agent_core/brain/llm_openai.py`: a `Brain` subclass shaped like `GroqBrain`
   (`respond()` POSTs `{model, messages, stream: True}` and yields `delta.content`
   chunks) but pointed at the llama.cpp base URL, no API key needed.
4. Register it: add an `llm` branch in `make_brain()` (`server/ws_server.py`,
   ~5 lines, same shape as the `groq` branch) and set in your yaml:
   `brain: {type: llm, model: lfm-2.5, api_key_env: ...}`.
5. Restart the server. Pipeline, tools, memory, and UI are untouched;
   the keyword tool router gets replaced by function calling later.

## What lives where

```
agent_core/  pipeline.py (state machine) · vad.py · turn.py · wakeword.py · stt.py · tts.py
             brain/ (base + stub + groq — the LLM seam) · tools/ (router, weather, search, telegram)
             meeting.py (diarization) · store.py (SQLite + FTS5 + embeddings) · config.py · debuglog.py
server/      ws_server.py (FastAPI transport) · ui/ (single-page client)
tools/       download_models · check_offline · e2e_offline · gen_fixtures
tests/       fixtures + 38 tests (VAD/turn/barge/E2E/WS/store/tools)
docs/        MASTER_PLAN.md (read this) · TESTING.md · pi5_deploy.md · wiring.md
vendors/     reference clones — never delete (huggingface-speech-to-speech, on-device-s2s, xtalk)
models/      local weights (gitignored) · logs/ auto debug sessions · data/ SQLite
```

Note: models/, vendors/, data/, .env, and the venv are all gitignored and stay on your
machine. (This checkout uses `v2/.venv-v2`; the venv carries its own `.gitignore`.)

## Next steps

See docs/MASTER_PLAN.md phases P10-P14 (Pi deploy, LLM brain swap, multilingual, hey_tobi wake
model, hardware GPIO). Testing checklist: docs/TESTING.md.
