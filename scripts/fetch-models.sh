#!/usr/bin/env bash
# Fetches the model files presence-voice v2 needs into ./models/ (gitignored),
# so a fresh clone can run `voice serve` without any manual model hunting.
#
# - Kokoro: the latest fp16 ONNX export + voices pack from
#   thewh1teagle/kokoro-onnx's "model-files-v1.0" release (GitHub's current
#   "Latest" tag for the English models - v1.1 is a separate Chinese-only
#   release, not a newer English version).
# - Piper: one .onnx + .onnx.json per preset named in config.yaml's
#   `preload:` list (parsed directly out of config.yaml - no YAML parser
#   dependency, just grep/awk, matching the same set the daemon warms up
#   at startup). To fetch additional Piper voices, add them to `preload:`
#   first, or download them by hand into models/piper/ using the same
#   rhasspy/piper-voices URL pattern shown below.
#
# Usage: ./scripts/fetch-models.sh [config.yaml]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config.yaml}"
KOKORO_DIR="models/kokoro"
PIPER_DIR="models/piper"
KOKORO_RELEASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

mkdir -p "$KOKORO_DIR" "$PIPER_DIR"

fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "[fetch-models] already have $dest, skipping"
    return
  fi
  echo "[fetch-models] downloading $dest"
  curl -fL --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

echo "[fetch-models] Kokoro (model-files-v1.0)..."
fetch "$KOKORO_RELEASE/kokoro-v1.0.fp16.onnx" "$KOKORO_DIR/kokoro-v1.0.fp16.onnx"
fetch "$KOKORO_RELEASE/voices-v1.0.bin" "$KOKORO_DIR/voices-v1.0.bin"

echo "[fetch-models] Piper (presets in ${CONFIG}'s preload: list)..."

# Names under the top-level `preload:` block (indented "- name" lines,
# stops at the next top-level "key:" line) - same shape src/config.zig
# parses.
preload_names=$(awk '
  /^preload:/ { in_block=1; next }
  /^[a-zA-Z_-]+:/ { in_block=0 }
  in_block && /^[[:space:]]*-/ {
    sub(/^[[:space:]]*-[[:space:]]*/, "");
    sub(/[[:space:]]*#.*/, "");
    print
  }
' "$CONFIG")

for name in $preload_names; do
  # This preset's block: from "  name:" to the next same-or-lower-indent key.
  block=$(awk -v name="$name" '
    $0 ~ "^  "name":" { in_block=1; next }
    in_block && /^  [a-zA-Z_-]+:/ { in_block=0 }
    in_block { print }
  ' "$CONFIG" | grep -v '^[[:space:]]*#')

  engine=$(echo "$block" | grep -oP '(?<=engine:\s)\S+' | head -1 || true)
  voice=$(echo "$block" | grep -oP '(?<=voice:\s")[^"]+' | head -1 || true)

  if [[ "$engine" != "piper" || -z "$voice" ]]; then
    continue # kokoro presets need no per-voice file - the shared model above covers all of them
  fi

  # en_US-lessac-high -> locale=en_US, name=lessac, quality=high
  locale="${voice%%-*}"
  rest="${voice#*-}"
  quality="${rest##*-}"
  voice_name="${rest%-*}"
  lang="${locale%%_*}"

  url_base="$PIPER_BASE/$lang/$locale/$voice_name/$quality/$voice"
  fetch "$url_base.onnx" "$PIPER_DIR/$voice.onnx"
  fetch "$url_base.onnx.json" "$PIPER_DIR/$voice.onnx.json"
done

echo "[fetch-models] done. See ./models/ ."
