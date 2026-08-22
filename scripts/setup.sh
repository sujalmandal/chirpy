#!/bin/zsh
# One-time setup for the Chirpy voice engine: create the Python venv, install
# dependencies, install the self-hosted LiveKit server, pre-download the
# lightest on-device models (tiny STT, the default Kokoro voice, TEN-VAD, and
# the bundled Qwen LLM), and run a smoke test so STT + TTS are proven to work
# on this Mac before you launch the app.
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
engine_dir="$project_dir/engine/chirpy"

# Locate a Python 3.12 interpreter (Homebrew's is the documented path, but accept
# one already on PATH so the script "just works" on a fresh machine).
PYTHON=""
for cand in python3.12 /opt/homebrew/bin/python3.12; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.12 not found." >&2
    echo "  Install it with:  brew install python@3.12" >&2
    echo "  (or ensure 'python3.12' is on your PATH) and re-run scripts/setup.sh." >&2
    exit 1
fi
venv_python="$engine_dir/.venv/bin/python"

echo "==> Creating Python 3.12 virtual environment"
[ -d "$engine_dir/.venv" ] || "$PYTHON" -m venv "$engine_dir/.venv"

echo "==> Installing dependencies"
"$venv_python" -m pip install --upgrade pip
"$venv_python" -m pip install -r "$engine_dir/requirements.txt"

echo "==> Installing self-hosted LiveKit server"
if ! command -v livekit-server >/dev/null 2>&1; then
    brew install livekit
else
    echo "livekit-server already installed"
fi

# The Japanese tokenizer dictionary (UniDic, ~526 MB) is deliberately NOT
# pre-downloaded here: it's only needed for Japanese TTS and would stall every
# fresh install. Kokoro fetches it on demand if you switch to a Japanese voice.

echo "==> Pre-downloading lightest speech models (tiny STT, default TTS voice, TEN-VAD)"
"$venv_python" "$engine_dir/download.py" stt tiny
"$venv_python" "$engine_dir/download.py" tts af_heart
"$venv_python" "$engine_dir/download.py" vad

echo "==> Pre-downloading bundled local LLM (Qwen2.5-0.5B, ~1GB)"
"$venv_python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

echo "==> Running smoke test (proves STT + TTS work on this Mac)"
"$venv_python" "$engine_dir/validate.py"

echo ""
echo "Setup complete. Build the app with scripts/build-chirpy-app.sh,"
echo "then open 'Chirpy.app'."
echo ""
echo "It works out of the box: a bundled local LLM, tiny STT, and Kokoro TTS"
echo "are all pre-downloaded. To use a stronger LLM later, open the debug"
echo "window (orb -> </> icon), click Settings, set your endpoint + model,"
echo "then Save & Restart. For LM Studio use http://localhost:1234/v1."

