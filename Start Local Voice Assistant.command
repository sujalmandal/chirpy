#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
exec "$project_dir/start.sh"
