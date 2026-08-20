#!/bin/zsh
# One-time setup for the Kyutai voice engine: create the Python venv, install
# dependencies, and run a smoke test that downloads model weights (~6.4 GB) and
# proves STT + TTS work on this Mac before you launch the app.
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
engine_dir="$project_dir/engine/kyutai"
venv_python="$engine_dir/.venv/bin/python"

echo "==> Creating Python 3.12 virtual environment"
[ -d "$engine_dir/.venv" ] || /opt/homebrew/bin/python3.12 -m venv "$engine_dir/.venv"

echo "==> Installing dependencies"
"$venv_python" -m pip install --upgrade pip
"$venv_python" -m pip install -r "$engine_dir/requirements.txt"

echo "==> Running smoke test (downloads model weights + synthesizes a phrase)"
"$venv_python" "$engine_dir/validate.py"

echo ""
echo "Setup complete. Build the app with scripts/build-local-moshi-app.sh,"
echo "then open 'Local Voice Assistant.app'."
