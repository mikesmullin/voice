#!/usr/bin/env bash
# Sets up Audio8 TTS (the "audio8" engine) for the `voice` CLI. Idempotent:
# anything already in place is skipped, so this is safe to re-run.
#
# Steps:
#   1. Download model weights -> /workspace/models/audio8-tts/
#      (via `hf download`; skipped if the manifest is already present)
#   2. Vendor the official inference runtime -> vendor/audio8-runtime/
#      (pinned upstream commit; skipped if already vendored)
#   3. Create a venv in vendor/audio8-runtime/.venv/ and install the deps
#   4. Register the packaged reference voice ("default") -> vendor/audio8-runtime/voices/
#
# Model weights use the Audio8 Community License v1.0 (see the model README /
# LICENSE); the runtime code is Apache-2.0 (see vendor/audio8-runtime/README.md).
# Note: Audio8 inference is CPU-only (ONNX INT8 per-token graphs + FP16 codec).
#
# Usage: ./scripts/setup-audio8.sh [--no-download]
#
# `./scripts/fetch-models.sh` is the kokoro/piper counterpart (kept separate
# on purpose: different tooling, different license, different output dir).

set -euo pipefail
cd "$(dirname "$0")/.."

NO_DOWNLOAD=0
for arg in "$@"; do
  [[ "$arg" == "--no-download" ]] && NO_DOWNLOAD=1
done

MODEL_REPO="Audio8/audio8-TTS-0.1B-ONNX-INT8"
MODEL_DIR="/workspace/models/audio8-tts"
RUNTIME_DIR="vendor/audio8-runtime"
RUNTIME_COMMIT="aef426db7229b93c8cb9a00c7c054b2b234e2979" # upstream master as of 2026-11-26
VENV_PYTHON="$RUNTIME_DIR/.venv/bin/python"

echo "[setup-audio8] model dir: $MODEL_DIR"
echo "[setup-audio8] runtime dir: $RUNTIME_DIR"

# --- step 1: model weights ---------------------------------------------------
if [[ -f "$MODEL_DIR/runtime_manifest.json" ]]; then
  echo "[setup-audio8] model already present at $MODEL_DIR, skipping download"
elif [[ "$NO_DOWNLOAD" == "1" ]]; then
  echo "[setup-audio8] --no-download set; model MISSING at $MODEL_DIR" >&2
  echo "              disable the flag and re-run to fetch it, or place the files manually" >&2
  exit 1
else
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  else
    echo "[setup-audio8] error: hf CLI not found (pip install huggingface_hub[cli])" >&2
    exit 1
  fi
fi

# --- step 2: vendor the inference runtime ------------------------------------
if [[ -f "$RUNTIME_DIR/arktts_runtime/runtime.py" ]]; then
  echo "[setup-audio8] runtime already vendored, skipping clone"
else
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  echo "[setup-audio8] cloning upstream Audio8_TTS @$RUNTIME_COMMIT"
  git clone --quiet https://github.com/Audio8-AI/Audio8_TTS.git "$tmp/repo"
  git -C "$tmp/repo" checkout --quiet "$RUNTIME_COMMIT"
  mkdir -p "$RUNTIME_DIR"
  cp -r "$tmp/repo/onnx_runtime_0_1b_int8/." "$RUNTIME_DIR/"
  rm -rf "$RUNTIME_DIR/.venv" "$RUNTIME_DIR/voices" "$RUNTIME_DIR/outputs" "$RUNTIME_DIR/model"
fi

# --- step 3: venv + deps ------------------------------------------------------
if [[ -x "$VENV_PYTHON" ]]; then
  echo "[setup-audio8] venv already exists, skipping pip install"
else
  # Upstream setup.sh uses whatever `python3` resolves to; on this box the
  # system Python 3.14 may not have a compatible onnxruntime wheel, and one
  # of the uv-managed 3.12 installs is broken in this environment - so try a
  # known-good list instead of trusting PATH blindly.
  picked=""
  for py in python3.10 python3.11 python3.12 python3.13 python3; do
    if ! command -v "$py" >/dev/null 2>&1; then continue; fi
    if "$py" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then continue; fi
    echo "[setup-audio8] trying venv interpreter: $($py --version 2>&1) ($(command -v "$py"))"
    if "$py" -m venv "$RUNTIME_DIR/.venv" 2>/tmp/setup-audio8-venv.log \
       && "$RUNTIME_DIR/.venv/bin/python" -m pip install --quiet --upgrade pip 2>/tmp/setup-audio8-pip.log \
       && "$RUNTIME_DIR/.venv/bin/python" -m pip install --quiet \
            -r "$RUNTIME_DIR/requirements.txt" 2>/tmp/setup-audio8-pip.log; then
      picked="$py"
      break
    fi
    echo "[setup-audio8]   $py failed, trying next..." 
    rm -rf "$RUNTIME_DIR/.venv"
  done
  if [[ -z "$picked" ]]; then
    echo "[setup-audio8] error: could not create a working venv (see /tmp/setup-audio8-venv.log)" >&2
    exit 1
  fi
fi

# --- step 4: register the packaged "default" voice -----------------------------
if [[ -f "$RUNTIME_DIR/voices/default/meta.json" ]]; then
  echo "[setup-audio8] 'default' voice already registered, skipping"
else
  "$VENV_PYTHON" "$RUNTIME_DIR/scripts/register_default_voice.py" \
    --model-dir "$MODEL_DIR" \
    --voices-dir "$RUNTIME_DIR/voices"
fi

echo "[setup-audio8] done."
echo "  voice audio8 \"Hello from Audio8.\"   # (client - needs the daemon running)"
echo "  voice local audio8 \"Hello from Audio8.\"   # standalone, no daemon"
