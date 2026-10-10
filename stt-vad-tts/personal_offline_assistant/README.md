# Tobi — Personal Offline Voice Assistant (v2)

An offline, full-duplex speech-to-speech voice assistant designed for **Raspberry Pi 5 (4GB)** and **Desktop (Windows / Linux)**.

**Pipeline Flow:**
`Wake word ("hey jarvis")` ➔ `Microphone streaming` ➔ `Silero VAD` ➔ `Faster-Whisper STT` ➔ `SmartTurn end-of-turn detection` ➔ `Brain reasoning` ➔ `Piper TTS` ➔ `Voice Barge-in & Interruption`

- **100% Offline Core**: Runs locally with ONNX Runtime & CTranslate2 (no PyTorch runtime needed).
- **Full-Duplex & Barge-In**: Interrupt the assistant mid-sentence by speaking over it.
- **Persistent Memory & RAG**: SQLite + FTS5 full-text search + all-MiniLM vector embeddings.
- **Optional Tools**: Weather, web search, and Telegram bot summaries.

---

## Quickstart (Windows)

### Option 1: One-Click Setup (`setup.bat`)

Double-click `setup.bat` (or run it from PowerShell/CMD):

```powershell
.\setup.bat
```

What `setup.bat` does automatically:
1. Creates a Python virtual environment (`v2\.venv-v2`)
2. Installs required dependencies (`requirements.txt`)
3. Creates `v2\.env` from `.env.example` if it doesn't exist
4. Downloads all offline neural models (`tools/download_models.py`)
5. Runs the test suite to verify everything passes
6. Starts the server on `http://127.0.0.1:8000`

---

### Option 2: Manual Setup

If you prefer running step-by-step in your terminal:

```powershell
# 1. Navigate to the v2 codebase
cd v2

# 2. Create and activate virtual environment
python -m venv .venv-v2
.\.venv-v2\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your local environment file (optional API keys)
copy .env.example .env

# 5. One-time offline model download (Whisper, Piper TTS, Silero, SmartTurn)
python tools\download_models.py

# 6. Verify all models load offline with network blocked
python tools\check_offline.py

# 7. Run the tests (38 tests)
python -m pytest tests/ -q

# 8. Start the voice server
python -m server.ws_server --config configs/desktop.yaml --host 127.0.0.1 --port 8000
```

Once started:
1. Open your browser at **http://127.0.0.1:8000**
2. Click **Enable Microphone**
3. Say **"Hey Jarvis"** to wake the assistant and start speaking!

*(Note: Normal startup warnings about PyTorch not being found or onnxruntime fallback are expected and harmless — the project is deliberately torch-free for performance).*

---

## How to Switch Brain / LLM Models

The reasoning brain is decoupled from the audio pipeline via the `Brain` interface (`v2/agent_core/brain/base.py`). You can switch between brains simply by editing the config file (`v2/configs/desktop.yaml` or `v2/configs/pi5-4gb.yaml`).

### 1. Stub Brain (100% Offline, Default)
Returns fast canned replies with local memory lookup. Ideal for testing hardware and audio latency without needing any LLM keys or extra RAM.

In `configs/desktop.yaml`:
```yaml
brain:
  type: stub
  model: ""
```

---

### 2. Groq Cloud Bridge (Fast Cloud LLM)
Uses Groq's high-speed inference for development and testing.

1. Add your Groq API key to `v2/.env`:
   ```bash
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```
2. In `configs/desktop.yaml`:
   ```yaml
   brain:
     type: groq
     model: openai/gpt-oss-20b       # or llama-3.3-70b-versatile
     api_key_env: GROQ_API_KEY
   ```

---

### 3. Local LLM — Running LFM 2.5 / Qwen / Llama (100% Sovereign Offline)

To run a true local SLM (such as **Liquid LFM 2.5**, **Qwen 2.5 3B/7B**, or **Llama 3.2 3B**) completely on your own hardware using `llama.cpp`:

#### Step A: Download and run `llama-server`
Download the pre-built `llama.cpp` binary and your model's GGUF file:
```powershell
# Run llama-server with your local model
llama-server -m lfm-2.5.gguf --port 8080 --host 127.0.0.1 -c 4096
```
This serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1/chat/completions`.

#### Step B: Point Tobi to your local server
The bridge client (`v2/agent_core/brain/groq.py`) connects to any OpenAI-compatible Chat Completions endpoint.
In `v2/configs/desktop.yaml`:
```yaml
brain:
  type: groq                        # Uses OpenAI-compatible streaming
  model: lfm-2.5                    # Your local model name
  api_key_env: LOCAL_LLM_KEY        # Any dummy string in .env
```
And set the server URL in `v2/agent_core/brain/groq.py` to `http://127.0.0.1:8080/v1/chat/completions`.

Restart the server and your voice assistant is now powered 100% locally by your own private model!

---

## Hardware Profiles

- **`v2/configs/desktop.yaml`**: Configured for PC/Desktop (Whisper `base.en`, medium TTS voice, higher audio buffers).
- **`v2/configs/pi5-4gb.yaml`**: Tuned for Raspberry Pi 5 4GB (Whisper `tiny.en`, low TTS voice, aggressive compute savings).

---

## Directory Overview

```text
├── setup.bat              # 1-Click launcher & installer for Windows
├── README.md              # Project documentation
├── v2/
│   ├── setup.bat          # Core setup script
│   ├── requirements.txt   # Python dependencies
│   ├── configs/           # Hardware profiles (desktop.yaml, pi5-4gb.yaml)
│   ├── agent_core/        # Audio pipeline, VAD, STT, TTS, memory, and brains
│   │   ├── brain/         # Brain implementations (stub, groq, base)
│   │   ├── tools/         # Weather, Search, Telegram integration
│   │   ├── pipeline.py    # Main full-duplex state machine & barge-in logic
│   │   ├── vad.py         # Silero VAD voice activity tracking
│   │   ├── stt.py         # Faster-Whisper live speech-to-text
│   │   └── tts.py         # Piper neural text-to-speech
│   ├── server/            # WebSocket server & browser mic client
│   ├── tools/             # download_models.py, check_offline.py
│   ├── tests/             # Pytest suite (unit and end-to-end regression)
│   └── docs/              # Master plan, Raspberry Pi wiring, deployment guides
```

---

## Running the Tests

To verify that the offline pipeline, speech barge-in, and turn-taking logic work properly:

```powershell
cd v2
.\.venv-v2\Scripts\python.exe -m pytest tests/ -q
```
All 38 tests should pass.
