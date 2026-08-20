# Local Voice Assistant

A deliberately small, local-first foundation for a real-time macOS voice assistant.

The first slice is complete end to end:

```text
hold Space → record microphone → local Whisper STT → local OpenAI-compatible LLM
→ sentence chunks → local Piper TTS → speaker
```

The macOS app is intentionally thin; the Python service owns model and tool orchestration. STT, LLM, and TTS are each behind a provider interface so their implementation can change without changing the client protocol. Each client session has short in-memory conversation context; “New chat” clears it. Nothing is persisted.

## What you need

- macOS 14+ and Xcode command-line tools
- Python 3.11+
- A local OpenAI-compatible chat endpoint. [Ollama](https://ollama.com/) works with `http://127.0.0.1:11434/v1`.
- `whisper-cli` from [whisper.cpp](https://github.com/ggerganov/whisper.cpp), plus a GGML Whisper model
- `piper` and a local Piper voice `.onnx` file

Nothing in this starter sends audio or text to a cloud service. Web access is deliberately deferred to a later, explicit tool phase.

## Run it

```bash
cd service
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

export WHISPER_CPP_BIN=/absolute/path/to/whisper-cli
export WHISPER_MODEL=/absolute/path/to/ggml-base.en.bin
export PIPER_BIN=/absolute/path/to/piper
export PIPER_MODEL=/absolute/path/to/en_US-lessac-medium.onnx
export LLM_MODEL=qwen3:14b
uvicorn app.main:app --host 127.0.0.1 --port 8787
```

`service/.env.example` contains the same configuration as a copyable reference. The commands above intentionally keep the service on `127.0.0.1`.

In a second terminal:

```bash
cd clients/macos
swift run
```

Grant microphone permission when macOS asks. Hold the Space bar while the app window is focused, speak, and release to submit. Press Escape to cancel speech or an in-flight turn. The app starts playing each synthesized sentence as it arrives; cancelling stops playback and closes the streaming request (barge-in).

For a quick service smoke test, use `curl` with a WAV file:

```bash
curl -N -F 'audio=@sample.wav;type=audio/wav' http://127.0.0.1:8787/v1/turn
```

## Repository layout

```text
service/                 Python agent service
  app/providers.py       Swappable STT, chat, and TTS providers
  app/main.py            Streaming turn protocol and sentence buffering
clients/macos/           Small SwiftUI push-to-talk client
docs/roadmap.md          Phased plan after the vertical slice
```

## Model choices for the M4 Pro / 64 GiB

Start with a 9–14B instruct/coding model at 4-bit quantization for the voice path. Keep a larger coding model registered separately for deliberate, long-context work. The endpoint contract is OpenAI-compatible, so MLX, Ollama, llama.cpp server, or another local server can replace each other through environment configuration.

## Important limitations of this starter

- It is push-to-talk, not wake-word/VAD. That keeps the first latency and cancellation path easy to debug.
- TTS is synthesized one sentence at a time. Piper starts promptly for short sentences; a future backend can use a genuinely streaming model without changing the wire protocol.
- There is no authentication because the service binds to loopback only. Do not expose it on your LAN unchanged.
