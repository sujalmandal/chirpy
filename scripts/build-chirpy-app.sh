#!/bin/zsh
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
app_dir="$project_dir/Chirpy.app"

cd "$project_dir/apps/chirpy"
npm install
npm run build
npm run tauri build -- --bundles app

# The Tauri bundle is written to target/release/bundle/macos/Chirpy.app.
bundle_app="$project_dir/apps/chirpy/src-tauri/target/release/bundle/macos/Chirpy.app"
if [[ -d "$bundle_app" ]]; then
    rm -rf "$app_dir"
    cp -R "$bundle_app" "$app_dir"
fi

# Ad-hoc sign so the microphone TCC permission is stable, and clear any
# quarantine flag so a locally-built app launches without a Gatekeeper block.
codesign --force --deep --sign - "$app_dir"
xattr -dr com.apple.quarantine "$app_dir" 2>/dev/null || true

touch "$app_dir"
echo "Built $app_dir"
