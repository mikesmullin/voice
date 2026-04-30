# 🗣️ Voice

Fast local text-to-speech (TTS) from the command line, with both Piper and Kokoro voice presets.

**UPDATE:** Piper is now the recommended default for this project. In regular CLI use it is noticeably faster than Kokoro, to the point that it is usually the better choice unless you specifically want a Kokoro voice.

Kokoro is still available and still useful, but it benefits much more from `voice serve` and `voice hot` because direct invocation has more noticeable startup delay. Kokoro voices also download automatically on first use.

## Why Piper First

- Piper is the practical default: fast, local, offline, and does not require a GPU or VRAM reservation.
- Kokoro still offers strong voice quality, but repeated or latency-sensitive Kokoro usage is better through `serve` and `hot`.
- The configured fallback voice is `lessac`, which is also one of the preferred Piper presets.

## Features

- Two local TTS engines: Piper and Kokoro
- Fast CPU-first workflow with Piper
- Optional GPU acceleration with Kokoro on NVIDIA CUDA systems
- Automatic first-use Kokoro voice downloads
- Play audio immediately or save to WAV
- Optional stinger sounds before playback
- Simple voice presets in `src/config.yaml`

## Installation

### Prerequisites

- Python 3.10-3.12
- [uv](https://docs.astral.sh/uv/) package manager
- PortAudio for local playback on Linux
- Optional: NVIDIA GPU with CUDA 12.x if you want Kokoro GPU acceleration

### Quick Install (Piper only — recommended for macOS)

Install without Kokoro — gets you Piper TTS with no PyTorch dependency:

```bash
uv tool install --editable . --with pip
```

### Install with Kokoro

To also enable Kokoro voices, install with the `kokoro` extra:

```bash
uv tool install --editable ".[kokoro]" --with pip
```

On **macOS** (Apple Silicon or Intel), this pulls `torch` from PyPI (CPU/MPS). No CUDA index, no NVIDIA driver required.

On **Linux / Windows**, `torch` is automatically pulled from the PyTorch CUDA 12.9 index for GPU acceleration.

The `--with pip` flag is required for the transformer stack.

### GPU Support

If you installed with Kokoro on an NVIDIA GPU, verify CUDA after install:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Operator Guide

Operational details live in `SKILLS.md`.

Use that file for:

- command examples
- preferred Piper and Kokoro presets
- stdin, file output, and config-path usage
- `serve` and `hot` workflow guidance
- practical operator rules for choosing Piper versus Kokoro

## Performance Notes

- Piper is the fastest general-purpose choice in this repo.
- Kokoro direct calls are slower to warm up.
- Kokoro voices download automatically on first use.
- If you need repeated low-latency Kokoro generation, use `voice serve` and then `voice hot`.
- If you mostly want reliable local TTS from the shell, use Piper presets first.

## Platform Notes

### Windows

- Kokoro GPU usage requires NVIDIA CUDA 12.x drivers.

### macOS

- Piper works well for local CPU usage and requires no extra dependencies.
- To use Kokoro on macOS, install with `.[kokoro]` — PyTorch will be pulled from PyPI (CPU/MPS, no CUDA needed). The CUDA index is skipped automatically.

### Linux

- Install PortAudio plus your normal ALSA or PulseAudio stack for playback.

## Development

```bash
git clone https://github.com/yourusername/voice.git
cd voice
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Technical Details

- Kokoro engine: Kokoro-82M through the `kokoro` package
- Piper engine: Piper ONNX voices through `piper-tts`
- Output format: WAV
- Sample rates: 24 kHz for Kokoro, 22.05 kHz for Piper

## Acknowledgments

- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)
- [Piper](https://github.com/rhasspy/piper)
- [PyTorch](https://pytorch.org/)
- [Hugging Face](https://huggingface.co/)