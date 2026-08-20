#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
service_dir="$project_dir/service"
client_dir="$project_dir/clients/macos"
settings_file="$service_dir/.env"
service_url="http://127.0.0.1:8787/health"

if [[ ! -x "$service_dir/.venv/bin/python" ]]; then
  print "Missing Python environment. Run the setup steps in README.md first."
  exit 1
fi

if [[ ! -f "$settings_file" ]]; then
  print "Missing $settings_file"
  print "Create it from service/.env.example and set the paths to your local models."
  exit 1
fi

# The settings file contains only local model paths and endpoint configuration.
set -a
source "$settings_file"
set +a

for variable in WHISPER_CPP_BIN WHISPER_MODEL PIPER_BIN PIPER_MODEL; do
  if [[ -z "${(P)variable:-}" ]]; then
    print "Missing $variable in $settings_file"
    exit 1
  fi
done

cleanup() {
  [[ -n "${service_pid:-}" ]] && kill "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$service_dir"
"$service_dir/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8787 &
service_pid=$!

for attempt in {1..20}; do
  if curl --silent --fail "$service_url" >/dev/null; then
    break
  fi
  sleep 0.25
done

if ! curl --silent --fail "$service_url" >/dev/null; then
  print "The local agent service did not start. Check the messages above."
  exit 1
fi

print "Local agent is ready. Starting the macOS client…"
cd "$client_dir"
swift run
