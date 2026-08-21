#!/bin/zsh
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
package_dir="$project_dir/apps/LocalVoiceAssistant"
app_dir="$project_dir/Local Voice Assistant.app"

cd "$package_dir"
swift build -c release

# Recreate the generated bundle so renamed executables and other stale build
# artifacts cannot survive an incremental rebuild.
if [[ "$app_dir" == "$project_dir/Local Voice Assistant.app" ]]; then
    rm -rf "$app_dir"
fi
mkdir -p "$app_dir/Contents/MacOS"
cp ".build/arm64-apple-macosx/release/LocalVoiceAssistant" "$app_dir/Contents/MacOS/LocalVoiceAssistant"
cp "$package_dir/Resources/Info.plist" "$app_dir/Contents/Info.plist"

# Ad-hoc sign so the microphone TCC permission is stable, and clear any
# quarantine flag so a locally-built app launches without a Gatekeeper block.
codesign --force --deep --sign - "$app_dir"
xattr -dr com.apple.quarantine "$app_dir" 2>/dev/null || true

touch "$app_dir"
echo "Built $app_dir"
