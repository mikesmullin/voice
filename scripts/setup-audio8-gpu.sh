#!/usr/bin/env bash
# Sets up the Audio8 GPU backend (the "audio8 ... --gpu" path): the
# full-precision PyTorch checkpoint + a CUDA venv that runs
# scripts/audio8_gpu_synth.py. Idempotent - anything already in place is
# skipped, safe to re-run.
#
# Layout (deliberately OUTSIDE this repo - ~4.5GB total):
#   /workspace/models/audio8-tts-pt/   model.safetensors + codec.pth (via `hf download`)
#   /workspace/models/audio8-gpu/.venv torch (CUDA) + transformers + soundfile
# This repo only commits the driver script (scripts/audio8_gpu_synth.py);
# the weights use the Audio8 Community License v1.0, same as the CPU export.
#
# Needs: `hf` on PATH, a Python 3.10+ interpreter for the venv, an NVIDIA
# GPU with a recent driver (torch cu130 wheels; tested on RTX 5090).
# CPU-only machines should skip this - the default CPU backend needs none of it.
#
# Usage: ./scripts/setup-audio8-gpu.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_REPO="Audio8/Audio8-TTS-Preview-0.1b"
MODEL_DIR="/workspace/models/audio8-tts-pt"
GPU_DIR="/workspace/models/audio8-gpu"
VENV_PYTHON="$GPU_DIR/.venv/bin/python"

echo "[setup-audio8-gpu] model dir: $MODEL_DIR"
echo "[setup-audio8-gpu] venv dir:  $GPU_DIR/.venv"

# --- step 1: PyTorch checkpoint -------------------------------------------
if [[ -f "$MODEL_DIR/model.safetensors" && -f "$MODEL_DIR/codec.pth" ]]; then
  echo "[setup-audio8-gpu] checkpoint already present, skipping download"
else
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  else
    echo "[setup-audio8-gpu] error: hf CLI not found (pip install huggingface_hub[cli])" >&2
    exit 1
  fi
fi

# --- step 2: CUDA venv ------------------------------------------------------
if [[ -x "$VENV_PYTHON" ]]; then
  echo "[setup-audio8-gpu] venv already exists, refreshing key packages"
else
  picked=""
  for py in python3.10 python3.11 python3.12 python3.13 python3; do
    if ! command -v "$py" >/dev/null 2>&1; then continue; fi
    if "$py" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then continue; fi
    echo "[setup-audio8-gpu] trying venv interpreter: $($py --version 2>&1) ($(command -v "$py"))"
    if "$py" -m venv "$GPU_DIR/.venv" 2>/tmp/setup-audio8-gpu-venv.log \
       && "$GPU_DIR/.venv/bin/python" -m pip install --quiet --upgrade pip 2>/tmp/setup-audio8-gpu-pip.log; then
      picked="$py"
      break
    fi
    echo "[setup-audio8-gpu]   $py failed, trying next..."
    rm -rf "$GPU_DIR/.venv"
  done
  if [[ -z "$picked" ]]; then
    echo "[setup-audio8-gpu] error: could not create a working venv (see /tmp/setup-audio8-gpu-venv.log)" >&2
    exit 1
  fi
fi

"$VENV_PYTHON" -m pip install --quiet \
  "torch" "torchaudio" \
  "transformers>=4.57.5,<5" \
  "soundfile>=0.12,<1" "safetensors>=0.4" "numpy>=1.24,<3" \
  2>/tmp/setup-audio8-gpu-pip.log || {
    echo "[setup-audio8-gpu] error: pip install failed (see /tmp/setup-audio8-gpu-pip.log)" >&2
    exit 1
  }

"$VENV_PYTHON" -c "import torch; assert torch.cuda.is_available(), 'torch has no CUDA'" 2>/dev/null || {
  echo "[setup-audio8-gpu] WARNING: torch installed but CUDA is not available - --gpu will fail at runtime" >&2
}

echo "[setup-audio8-gpu] done."
echo "  voice local ruth --gpu \"Hello from the GPU backend.\""
