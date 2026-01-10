# 🗣️ Voice

A simple, fast text-to-speech (TTS) CLI with voice presets. Features GPU acceleration with Kokoro and CPU fallback with Piper.

## Features

- 🎭 **Multiple TTS Engines**: 
  - **Kokoro**: 28 high-quality English voices (GPU-accelerated)
  - **Piper**: 25+ fast CPU-optimized voices (no GPU required)
- 🔥 **Server Mode**: Pre-load model for near-instant synthesis (<0.2s)
- 🖥️ **Cross-Platform**: Windows, macOS, and Linux
- ⚡ **GPU Accelerated**: CUDA support for NVIDIA GPUs (10-20x realtime)
- 💻 **CPU Fallback**: Piper engine runs efficiently without GPU
- 🔒 **Privacy-First**: All processing happens locally, no cloud required
- 🎵 **WAV Output**: Play audio or save to file
- 🔔 **Stinger Support**: Optional sound effects before speech (alerts, notifications, etc.)


## Installation

### Prerequisites

- Python 3.10-3.12
- [uv](https://docs.astral.sh/uv/) package manager
- **PortAudio** (system dependency for audio playback on linux)
- **Optional but recommended**: NVIDIA GPU with CUDA 12.x for GPU acceleration


### Quick Install

#### 1. Install Voice CLI

Install `voice` globally as a CLI tool:

```bash
uv tool install --editable . --with pip
```

**Note:** The `--with pip` flag is required for transformers dependencies.

### GPU Support (NVIDIA)

GPU support is automatically configured in `pyproject.toml`. The installation will use CUDA-enabled PyTorch if available.

To verify CUDA is working:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**First run** will download Kokoro models (~500KB per voice) from Hugging Face automatically.

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
  --stinger NAME            Play stinger sound before speech
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

**Stinger sound effects:**
```bash
# Use default stinger (if configured for preset)
voice ada "Important message"

# Override with specific stinger
voice ada "Error occurred" --stinger error
voice ada "Alert notification" --stinger alert

# Works in server mode too
voice hot ada "Server notification" --stinger alert
```

## Available Voices

### Kokoro Voices (GPU-accelerated, highest quality)

#### American Female
Best quality: `heart`, `bella`, `sarah`, `sky`
- `bella`, `nicole`, `alloy`, `aoede`, `ada`, `kore`, `sarah`, `nova`
- `jessica`, `river`, `sky`, `heart`

#### American Male
Best quality: `adam`, `eric`, `michael`
- `fenrir`, `michael`, `puck`, `echo`, `eric`, `liam`, `onyx`, `santa`, `adam`

#### British Female
- `emma`, `isabella`, `alice`, `lily`

#### British Male
- `fable`, `george`, `lewis`, `daniel`

### Piper Voices (CPU-optimized, fast)

Piper voices run efficiently on CPU without requiring a GPU. Great for systems without NVIDIA graphics.

#### American Female (Piper)
Best quality: `lessac`, `ljspeech`
- `amy`, `hfc_female`, `kathleen`, `kristin`, `lessac`, `libritts`, `ljspeech`

#### American Male (Piper)
Best quality: `ryan`, `norman`
- `arctic`, `bryce`, `danny`, `hfc_male`, `joe`, `john`, `kusal`, `norman`, `reza`, `ryan`, `sam`

#### British Female (Piper)
Best quality: `cori`
- `alba`, `cori`, `jenny`, `southern_female`

#### British Male (Piper)
Best quality: `alan`, `aru`, `vctk`
- `alan`, `aru`, `northern_male`, `semaine`, `vctk`

## Configuration

Voice presets are defined in `src/config.yaml`:

```yaml
voices:
  # Kokoro voice (GPU-accelerated)
  heart:
    engine: kokoro
    voice: "af_heart"
    speed: 1.0
  
  bella:
    engine: kokoro
    voice: "af_bella"
    speed: 1.0
  
  # Piper voice (CPU-optimized)
  lessac:
    engine: piper
    voice: "en_US-lessac-high"
  
  norman:
    engine: piper
    voice: "en_US-norman-medium"
    speed: 1.7
  
  ada:
    engine: kokoro
    voice: "af_aoede"
    speed: 1.5
    # Optional stinger sound effects
    stingers:
      alert: tmp/alert.wav
      error: tmp/error.wav
    default_stinger: alert  # Play automatically unless overridden
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

### Stinger Configuration

Stingers are short sound effects played before speech synthesis. They're useful for:
- **Alerts and notifications**: Get attention before speaking
- **Error messages**: Distinct sound for error notifications
- **Status indicators**: Different sounds for different message types

**Configuration:**

1. **Define stingers per-preset** in `config.yaml`:
   ```yaml
   voices:
     ada:
       voice: "af_aoede"
       speed: 1.5
       stingers:
         alert: tmp/alert.wav      # Path relative to project root
         error: tmp/error.wav
         success: tmp/success.wav
       default_stinger: alert      # Optional: auto-play this stinger
   ```

2. **Use via CLI**:
   ```bash
   # Use default stinger (if configured)
   voice ada "Message"
   
   # Override with specific stinger
   voice ada "Error message" --stinger error
   
   # No stinger (even if default configured)
   voice ada "Message" --stinger none
   ```

**Notes:**
- Stingers are only played during audio playback (not when saving to file with `-o`)
- The `--stinger` parameter is only available for direct synthesis and `hot` mode (not `serve`)
- Stinger files must be WAV format
- If a stinger name doesn't exist in the config, it's silently ignored (no-op)
- Stinger audio is loaded early but played right before synthesized speech for optimal timing

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
│   ├── kokoro_engine.py # Kokoro TTS (GPU-accelerated)
│   ├── piper_engine.py  # Piper TTS (CPU-optimized)
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

- **Kokoro Engine**: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (82M parameters, GPU-accelerated)
- **Piper Engine**: [Piper](https://github.com/rhasspy/piper) (ONNX models, CPU-optimized)
- **Framework**: PyTorch 2.9+ with transformers
- **Sample Rate**: 24kHz (Kokoro), 22kHz (Piper)
- **Format**: WAV (16-bit PCM)
- **Protocol**: TCP JSON for server mode (port 3124)

## Acknowledgments

- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad - High-quality GPU-accelerated TTS
- [Piper](https://github.com/rhasspy/piper) by rhasspy - Fast CPU-optimized TTS
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers library
