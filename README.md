# Local Voice Assistant

Local Voice Assistant is a private, real-time voice assistant for Apple
Silicon Macs. It pairs a native SwiftUI app with fully local, on-device speech
models from Kyutai and an OpenAI-compatible cloud reasoning model.

The app opens as a compact, floating voice control inspired by ChatGPT Voice
for macOS. Right-click the window and choose **Open Debug Mode** to open the
full dashboard, inspect metrics and logs, or configure each stage of the
`VAD → STT → LLM → TTS` pipeline.

```text
Swift macOS app
  ├─ local microphone + barge-in
  ├─ Kyutai STT 1B (MLX) + semantic VAD
  ├─ Kyutai TTS 1.6B (MLX)
  └─ local agent service
       └─ Ollama Cloud / OpenAI-compatible LLM stream
```

## Setup (once)

Install the Python engine, dependencies, and model weights (~6.4 GB download)
and run a smoke test that synthesizes and transcribes a phrase:

```bash
scripts/setup-kyutai.sh
```

Put cloud credentials in `config/local.env` (see `config/local.env.example`)
to use an OpenAI-compatible cloud model, or configure the endpoint in the app.
Credentials entered in the app are stored in the macOS Keychain rather than
UserDefaults or the repository.

## Run

Build and launch the app:

```bash
scripts/build-local-moshi-app.sh
open "Local Voice Assistant.app"
```

The Swift app starts and owns the local agent process. Closing the app stops
the agent. First launch downloads models on demand and takes a minute to load
them.

## Layout

```text
apps/LocalMoshi/       Native macOS SwiftUI app
engine/kyutai/         Local Python agent: Kyutai STT + TTS (MLX), LLM streaming
scripts/               Setup and build scripts
docs/                  Product notes
config/                Secrets and model configuration
```

## How it works

The Swift app captures the microphone and streams 24 kHz Float32 PCM over a
WebSocket to `engine/kyutai/agent.py`. Kyutai STT 1B transcribes on-device and
its built-in semantic VAD works with configurable energy and timing safeguards
to detect natural end-of-turn. The transcript goes to an OpenAI-compatible LLM,
and the reply is spoken by Kyutai TTS 1.6B running on MLX.

The agent speaks a stable WebSocket protocol (JSON text events + binary PCM), so
the STT/TTS/LLM providers can be swapped without touching the Swift UI.

## Configuring the pipeline

Right-click the compact assistant and open **Debug Mode**, then select any
pipeline card. VAD, STT, and TTS include a
Hugging Face model browser; LLM accepts an OpenAI-compatible endpoint, model
name, and optional API key. The built-in Apple Silicon adapters currently load
Kyutai delayed-stream STT/TTS checkpoints. Other Hugging Face repositories can
be selected and persisted for adapter development, and the UI identifies when
an additional runtime adapter is required.

The assistant name and behavior are included in the system message sent to the
LLM. Changes take effect after **Save & Restart**.

## Local data

Model weights are stored outside this repository in
`~/.cache/huggingface/hub`. The engine's Python venv and any generated files are
local-only and ignored by Git. Microphone audio is streamed only to the local
agent. Text leaves the Mac only when the configured LLM endpoint is remote.
