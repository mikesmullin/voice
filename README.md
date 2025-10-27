# 🗣️Voice

A simple text-to-speech (TTS) system using Kokoro with 28 English voice presets. Cross-platform with local processing for privacy.

## Features

- 🎭 **28 Voice Presets**: American and British English voices from Kokoro
- 🖥️ **Cross-Platform**: Windows, macOS, and Linux
- 🔒 **Privacy-First**: All processing happens locally
- ⚡ **GPU Accelerated**: Leverages PyTorch with GPU support
- 🎵 **WAV Output**: Play audio or save to WAV format

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU support (NVIDIA CUDA or Apple Silicon) optional but recommended

### Global Installation (Recommended)

Install `voice` globally as a command-line tool:

```bash
uv tool install --from . --with pip voice
```

Or install from the repository directory:

```bash
cd /path/to/voice
uv tool install --editable . --with pip
```

**Note:** The `--with pip` flag is required for the transformers library used by Kokoro.

### Local Development Installation

For development, you can install in a virtual environment:

1. **Create virtual environment:**
   ```bash
   uv venv --python 3.12.8
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Install the package:**
   ```bash
   uv pip install -e .
   ```

**Note:** Models will be automatically downloaded from Hugging Face on first use (~500KB per voice).

## Quick Start

### List available voices:
```bash
voice --list
```

### Synthesize speech:
```bash
voice heart "Hello from the heart voice."
voice bella "It's a beautiful morning. Time for a cup of coffee."
voice adam "Another day another dollar."
```

### Save to file:
```bash
voice -o output.wav heart "Save this to a file"
```

### Get voice information:
```bash
voice --info heart
```

### Use custom config:
```bash
voice --config config.local.yaml myvoice "Custom text"
```

## Usage

```
voice <preset> <text> [options]

Positional Arguments:
  preset                Voice preset name (heart, bella, adam, etc.)
  text                  Text to synthesize

Options:
  -o FILE, --output FILE    Save audio to file (WAV format)
  -c FILE, --config FILE    Use custom config file
  -l, --list                List available voice presets
  -i PRESET, --info PRESET  Show information about a voice preset
  -v, --version             Show version
  -h, --help                Show help message

Examples:
  voice heart "Hello from the heart voice."
  voice bella "You've got mail."
  voice -o output.wav heart "Save to file"
  voice --list
  voice --info heart
```

## Configuration

Voice presets are configured in `config.yaml`. Each preset specifies:

- **voice**: Kokoro voice ID (e.g., `af_heart`, `af_bella`, `am_adam`)
- **speed**: Speech rate multiplier (default: 1.0)

### Example Voice Preset

```yaml
voices:
  heart:
    voice: "af_heart"
    speed: 1.0
  
  bella:
    voice: "af_bella"
    speed: 1.0
  
  adam:
    voice: "am_adam"
    speed: 1.2
```

### Adding Custom Voices

1. Edit `config.yaml` (or create `config.local.yaml`)
2. Add a new voice preset under `voices:`
3. Specify the Kokoro voice ID and speed

```yaml
voices:
  my_custom_voice:
    voice: "af_sarah"
    speed: 0.9
```

Available Kokoro voice IDs include:
- **American Female**: `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `af_alloy`, `af_echo`, `af_fable`, `af_onyx`, `af_nova`, `af_shimmer`, `af_heart`, `af_aoede`, `af_kore`, `af_jessica`, `af_emma`, `af_alice`, `af_lily`, `af_isabella`, `af_river`
- **American Male**: `am_adam`, `am_eric`, `am_michael`, `am_daniel`, `am_liam`, `am_lewis`, `am_santa`, `am_fenrir`, `am_puck`
- **British Female**: `bf_emma`, `bf_isabella`
- **British Male**: `bm_george`, `bm_lewis`

## Platform-Specific Notes

### Windows
- GPU acceleration via CUDA (if NVIDIA GPU available)
- Models automatically cached in `%USERPROFILE%\.cache\huggingface\`

### macOS
- GPU acceleration via Metal (if Apple Silicon)
- Models automatically cached in `~/.cache/huggingface/`

### Linux
- GPU acceleration via CUDA (if NVIDIA GPU available)
- Models automatically cached in `~/.cache/huggingface/`

## Troubleshooting

### First run is slow
The first time you use a voice, Kokoro downloads the model from Hugging Face (~500KB per voice). Subsequent runs use the cached model and are much faster.

### Audio playback issues
- **Windows**: Ensure audio drivers are installed
- **macOS**: Grant audio permissions if prompted
- **Linux**: Ensure ALSA/PulseAudio is configured
- Use `-o output.wav` flag to save to file instead of playing

### GPU not detected
GPU acceleration is automatic if available. CPU fallback works on all platforms.

## Acknowledgments

- Uses [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) for high-quality voice synthesis
- Built with PyTorch and Hugging Face Hub
