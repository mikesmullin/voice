# 🗣️ Voice

A simple, fast text-to-speech (TTS) CLI using Kokoro with 28 English voice presets. Features GPU acceleration and a low-latency server mode for instant synthesis.

## Features

- 🎭 **28 Voice Presets**: 20 American + 8 British English voices from Kokoro
- � **Server Mode**: Pre-load model for near-instant synthesis (<0.2s)
- �🖥️ **Cross-Platform**: Windows, macOS, and Linux
- ⚡ **GPU Accelerated**: CUDA support for NVIDIA GPUs (10-20x realtime)
- 🔒 **Privacy-First**: All processing happens locally, no cloud required
- 🎵 **WAV Output**: Play audio or save to file

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- **Optional but recommended**: NVIDIA GPU with CUDA 12.x for GPU acceleration

### Quick Install

Install `voice` globally as a CLI tool:

```bash
git clone https://github.com/yourusername/voice.git
cd voice
uv tool install --editable . --with pip
```

**Note:** The `--with pip` flag is required for transformers dependencies.

### GPU Support (NVIDIA)

For NVIDIA GPU acceleration, install PyTorch with CUDA support:

1. **Check your CUDA version:**
   ```bash
   nvidia-smi  # Look for "CUDA Version: X.X"
   ```

2. **Install PyTorch with matching CUDA version:**
   ```bash
   # For CUDA 12.9 (adjust cu129 to match your version: cu121, cu124, etc.)
   uv pip install --python %APPDATA%\Roaming\uv\tools\voice torch --index-url https://download.pytorch.org/whl/cu129 --reinstall
   ```

   On Linux/macOS:
   ```bash
   uv pip install --python ~/.local/share/uv/tools/voice torch --index-url https://download.pytorch.org/whl/cu129 --reinstall
   ```

**First run** will download models (~500KB per voice) from Hugging Face automatically.

## Usage

### Three Modes

#### 1. Direct Synthesis (Default)
Synthesize and play immediately:

```bash
voice heart "Hello from the heart voice."
voice bella "It's a beautiful morning."
voice adam "Another day, another dollar."
```

#### 2. Server Mode (Low-Latency)
Start a server that pre-loads the model for instant synthesis:

```bash
# Terminal 1: Start server
voice serve

# Terminal 2: Synthesize instantly
voice hot heart "This generates in under 0.2 seconds!"
voice hot bella "Multiple requests stay fast."
```

Server mode is ideal for:
- Interactive applications
- Real-time voice generation
- Multiple rapid synthesis requests

#### 3. Save to File

```bash
voice heart "Save this" -o output.wav
voice hot bella "Server mode too" -o bella.wav
```

### Command Reference

**Direct synthesis:**
```bash
voice <preset> <text> [options]
```

**Server mode:**
```bash
voice serve [options]              # Start server
voice hot <preset> <text> [options]  # Send to server
```

**Options:**
```
  -o FILE, --output FILE    Save audio to WAV file
  -c FILE, --config FILE    Use custom config file
  --cpu                     Force CPU (disable GPU)
  -l, --list                List available voice presets
  -i PRESET, --info PRESET  Show preset information
  -v, --version             Show version
  -h, --help                Show help
```

**Server options:**
```
  --host HOST               Bind to host (default: 127.0.0.1)
  --port PORT               Bind to port (default: 3124)
  --cpu                     Force CPU usage
```

### Examples

**List voices:**
```bash
voice --list
```

**Get voice info:**
```bash
voice --info heart
```

**Custom config:**
```bash
voice --config my-config.yaml myvoice "Custom voice"
```

**Force CPU usage:**
```bash
voice heart "Use CPU" --cpu
voice serve --cpu
```

## Available Voices

### American Female (20 voices)
Best quality: `heart`, `bella`, `sarah`, `sky`
- `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `af_alloy`, `af_echo`
- `af_fable`, `af_onyx`, `af_nova`, `af_shimmer`, `af_heart`, `af_aoede`
- `af_kore`, `af_jessica`, `af_emma`, `af_alice`, `af_lily`, `af_isabella`
- `af_river`, `ada` (custom: aoede at 0.5x speed)

### American Male (9 voices)
Best quality: `adam`, `eric`, `michael`
- `am_adam`, `am_eric`, `am_michael`, `am_daniel`, `am_liam`, `am_lewis`
- `am_santa`, `am_fenrir`, `am_puck`

### British Female (2 voices)
- `bf_emma`, `bf_isabella`

### British Male (6 voices)
- `bm_george`, `bm_lewis`, `bm_puck`, `bm_fenrir`, `bm_santa`, `bm_daniel`

## Configuration

Voice presets are defined in `src/config.yaml`:

```yaml
voices:
  heart:
    voice: "af_heart"
    speed: 1.0
  
  bella:
    voice: "af_bella"
    speed: 1.0
  
  ada:
    voice: "af_aoede"
    speed: 0.5  # Custom slower speed
```

### Custom Configuration

Create a custom config file:

```yaml
voices:
  my_voice:
    voice: "af_sarah"
    speed: 1.2  # 20% faster
```

Use it with:
```bash
voice --config my-config.yaml my_voice "Hello"
```

## Performance

### GPU vs CPU

With NVIDIA GPU (CUDA):
- **First synthesis**: ~0.5-2.0s (voice loading)
- **Subsequent**: ~0.14-0.20s (cached voice)
- **Speed**: ~15-20x realtime generation

With CPU only:
- **First synthesis**: ~0.6-2.0s (voice loading)
- **Subsequent**: ~0.17-0.25s (cached voice)
- **Speed**: ~10-15x realtime generation

### Server Mode Benefits

Server mode pre-loads the Kokoro model at startup:
- ✅ Model stays in GPU/CPU memory
- ✅ Voice cache persists between requests
- ✅ Consistent ~0.14-0.20s generation time
- ✅ No model reload overhead

## Platform-Specific Notes

### Windows
- **GPU**: Requires NVIDIA GPU with CUDA 12.x drivers
- **Cache**: `%USERPROFILE%\.cache\huggingface\`
- **Audio**: DirectSound for playback

### macOS
- **GPU**: Apple Silicon uses Metal (not yet supported by PyTorch 2.x)
- **Cache**: `~/.cache/huggingface/`
- **Audio**: CoreAudio for playback

### Linux
- **GPU**: NVIDIA GPU with CUDA 12.x
- **Cache**: `~/.cache/huggingface/`
- **Audio**: ALSA/PulseAudio

## Troubleshooting

### GPU not detected

Check CUDA availability:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If False:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Install CUDA PyTorch (see Installation section)
3. Or use `--cpu` flag to force CPU mode

### First run downloads models

The first synthesis with each voice downloads its model (~500KB) from Hugging Face. This is a one-time download per voice.

### Server connection refused

Ensure the server is running:
```bash
# Terminal 1
voice serve

# Terminal 2  
voice hot heart "test"
```

### Audio playback issues

Save to file instead of playing:
```bash
voice heart "test" -o test.wav
```

## Development

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/voice.git
cd voice

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install in editable mode
uv pip install -e .
```

### Project Structure

```
voice/
├── src/
│   ├── __init__.py
│   ├── cli.py           # Command-line interface
│   ├── voice_engine.py  # Core TTS engine
│   ├── audio_utils.py   # Audio playback/saving
│   ├── server.py        # TCP server for hot mode
│   ├── client.py        # TCP client for hot mode
│   ├── timing.py        # Timestamp logging
│   └── config.yaml      # Voice presets
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Technical Details

- **Model**: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (82 million parameters)
- **Framework**: PyTorch 2.9+ with transformers
- **Sample Rate**: 24kHz
- **Format**: WAV (16-bit PCM)
- **Protocol**: TCP JSON for server mode (port 3124)

## Acknowledgments

- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad - High-quality open-source TTS
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers library
