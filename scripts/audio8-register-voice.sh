#!/usr/bin/env bash
# Registers a new Audio8 voice from a reference recording + its exact
# transcript (the voice-cloning step). Writes
# vendor/audio8-runtime/voices/<name>/{codes.npy,meta.json} - the same pair
# BOTH backends (CPU ONNX + GPU PyTorch) synthesize from, so one
# registration serves `voice local <preset>` with or without --gpu.
#
# Requirements: the CPU runtime venv (./scripts/setup-audio8.sh), a
# reference recording 0.5-30s long (<50MB, wav/flac/ogg), and its
# word-for-word transcript.
#
# Usage:
#   ./scripts/audio8-register-voice.sh --wav ref.wav --name ruth \
#       --transcript transcript.txt
#   ./scripts/audio8-register-voice.sh --wav ref.wav --name ruth \
#       --text "the exact words spoken ..." [--overwrite]
#
# Then expose it as a preset in config.yaml:
#   ruth:
#     engine: audio8
#     voice: "Ruth"

set -euo pipefail
cd "$(dirname "$0")/.."

RUNTIME_DIR="vendor/audio8-runtime"
VENV_PYTHON="$RUNTIME_DIR/.venv/bin/python"
MODEL_DIR="/workspace/models/audio8-tts"
VOICES_DIR="$RUNTIME_DIR/voices"

WAV=""; NAME=""; TRANSCRIPT=""; TEXT=""; OVERWRITE="--no-overwrite"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wav) WAV="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --transcript) TRANSCRIPT="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --overwrite) OVERWRITE="--overwrite"; shift;;
    *) echo "unknown arg: $1 (see header)" >&2; exit 2;;
  esac
done

[[ -n "$WAV" && -n "$NAME" ]] || { echo "error: --wav and --name are required" >&2; exit 2; }
if [[ -n "$TRANSCRIPT" && -n "$TEXT" ]]; then { echo "error: --transcript and --text are mutually exclusive" >&2; exit 2; }; fi
if [[ -z "$TRANSCRIPT" && -z "$TEXT" ]]; then { echo "error: one of --transcript or --text is required" >&2; exit 2; }; fi
[[ -x "$VENV_PYTHON" ]] || { echo "error: CPU runtime venv missing - run ./scripts/setup-audio8.sh first" >&2; exit 1; }
[[ -f "$MODEL_DIR/runtime_manifest.json" ]] || { echo "error: model missing at $MODEL_DIR - run ./scripts/setup-audio8.sh first" >&2; exit 1; }

if [[ -n "$TRANSCRIPT" ]]; then
  TEXT="$(cat "$TRANSCRIPT")"
fi

"$VENV_PYTHON" - "$WAV" "$NAME" "$TEXT" "$OVERWRITE" <<'EOF'
import json
import sys
from pathlib import Path
from arktts_runtime.registration import VoiceRegistration

wav_path, name, text, overwrite = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "--overwrite"
model_dir = Path("/workspace/models/audio8-tts")
manifest = json.loads((model_dir / "runtime_manifest.json").read_text())
reg = VoiceRegistration(model_dir / "registration", Path("vendor/audio8-runtime/voices"),
                        manifest["model_fingerprint"])
meta = reg.register(Path(wav_path).read_bytes(), Path(wav_path).name, text, name, overwrite=overwrite)
print(f"registered {meta['name']}: shape={tuple(meta['shape'])}")
EOF
