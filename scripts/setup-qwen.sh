#!/usr/bin/env bash
# Sets up the Qwen3-TTS engine (the "qwen" engine) for the `voice` CLI.
# Idempotent: anything already in place is skipped, so this is safe to re-run.
#
# What it does:
#   1. Downloads the Q4_K_M backbone + Q8_0 mmproj -> /workspace/models/qwen3-tts/
#      (via `hf download`; skipped if both files already exist)
#
# What it does NOT do:
#   - it does NOT fetch the voice-clone source WAV. The speaker file you
#     clone from (e.g. ~/Downloads/ruth2.wav) is your own voice recording -
#     keep it out of git and point your config.yaml preset's `speaker:` field
#     at its location (default: /workspace/models/qwen3-tts/ruth2.wav).
#   - it does NOT install llama.cpp. The engine shells out to the
#     system-installed `llama-tts` binary (Debian/Arch: llama.cpp /
#     llama.cpp-vulkan etc.); install that first.
#
# Model weights (ggml-org mirror of Alibaba's Qwen3-TTS-12Hz-1.7B-Base,
# Apache-2.0-class licensing - see the model README).
#
# Usage: ./scripts/setup-qwen.sh [--no-download]

set -euo pipefail
cd "$(dirname "$0")/.."

NO_DOWNLOAD=0
for arg in "$@"; do
  [[ "$arg" == "--no-download" ]] && NO_DOWNLOAD=1
done

MODEL_REPO="ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
MODEL_DIR="/workspace/models/qwen3-tts"
BACKBONE="Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
MMPROJ="mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"

echo "[setup-qwen] model dir: $MODEL_DIR"
echo "[setup-qwen] binary:   /usr/bin/llama-tts (system-installed llama.cpp)"

if ! command -v /usr/bin/llama-tts >/dev/null 2>&1; then
  echo "[setup-qwen] warning: /usr/bin/llama-tts not found - install llama.cpp first" >&2
  echo "             the engine shells out to it; nothing will work without it" >&2
fi

missing=0
for f in "$BACKBONE" "$MMPROJ"; do
  [[ -f "$MODEL_DIR/$f" ]] && echo "[setup-qwen] have $f, skipping" || missing=1
done

if [[ "$missing" == "0" ]]; then
  echo "[setup-qwen] both files present, nothing to do"
  exit 0
fi

if [[ "$NO_DOWNLOAD" == "1" ]]; then
  echo "[setup-qwen] --no-download set; weights are missing" >&2
  echo "             disable the flag and re-run to fetch them, or place the files manually" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "[setup-qwen] error: hf CLI not found (pip install huggingface_hub[cli])" >&2
  exit 1
fi

echo "[setup-qwen] downloading $MODEL_REPO -> $MODEL_DIR"
mkdir -p "$MODEL_DIR"
hf download "$MODEL_REPO" "$BACKBONE" "$MMPROJ" --local-dir "$MODEL_DIR"

echo "[setup-qwen] done. Next: set your preset's speaker: field in config.yaml to your clone source."
