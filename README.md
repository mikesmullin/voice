# 🗣️ Voice

A simple, fast text-to-speech (TTS) CLI using Kokoro and StyleTTS2 with voice presets. Features GPU acceleration, low-latency server mode, and zero-shot voice cloning with StyleTTS2.

## Features

- 🎭 **Multiple TTS Models**: 
  - **Kokoro**: 28 English voice presets (20 American + 8 British)
  - **StyleTTS2**: Zero-shot voice cloning with reference audio
- 🔥 **Server Mode**: Pre-load model for near-instant synthesis (<0.2s)
- 🖥️ **Cross-Platform**: Windows, macOS, and Linux
- ⚡ **GPU Accelerated**: CUDA support for NVIDIA GPUs (10-20x realtime)
- 🔒 **Privacy-First**: All processing happens locally, no cloud required
- 🎵 **WAV Output**: Play audio or save to file
- 🔔 **Stinger Support**: Optional sound effects before speech (alerts, notifications, etc.)
- 🎤 **Zero-Shot Cloning**: Clone any voice with a single reference audio file (StyleTTS2)

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- **Optional but recommended**: NVIDIA GPU with CUDA 12.x for GPU acceleration
- **For StyleTTS2**: espeak-ng (system dependency)

### Quick Install

#### 1. Install Voice CLI

Install `voice` globally as a CLI tool:

```bash
git clone https://github.com/yourusername/voice.git
cd voice

# Basic install (Kokoro TTS only)
uv tool install --editable . --with pip

# Or with StyleTTS2 support for zero-shot voice cloning
uv tool install --editable .[styletts2] --with pip
```

**Note:** The `--with pip` flag is required for transformers dependencies.

#### 2. Install System Dependencies (StyleTTS2 only)

If you installed with `[styletts2]`, you need espeak-ng for phonemization:

**Automated (recommended):**
```bash
python setup_deps.py
```

**Manual installation:**

- **Windows**: 
  - With Chocolatey: `choco install espeak-ng`
  - With Scoop: `scoop install espeak-ng`
  - Or download from https://github.com/espeak-ng/espeak-ng/releases

- **macOS**: 
  ```bash
  brew install espeak-ng
  ```

- **Linux**: 
  ```bash
  # Debian/Ubuntu
  sudo apt-get install espeak-ng
  
  # Fedora
  sudo dnf install espeak-ng
  
  # Arch
  sudo pacman -S espeak-ng
  ```

Verify installation:
```bash
espeak-ng --version
```

#### 3. Download StyleTTS2 Models (Optional)

If using StyleTTS2, download the pre-trained models:

```bash
# Clone StyleTTS2 repository
git clone https://github.com/yl4579/StyleTTS2.git tmp/StyleTTS2

# Download models from Hugging Face
# Visit: https://huggingface.co/yl4579/StyleTTS2-LibriTTS
# Download and place in tmp/StyleTTS2/Models/LibriTTS/:
#   - config.yml
#   - epochs_2nd_00020.pth
```

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

### Zero-Shot Voice Cloning with StyleTTS2

In addition to Kokoro's preset voices, you can use **StyleTTS2** for zero-shot voice cloning - clone any voice using just a single reference audio file!

**Setup:**

1. **Clone StyleTTS2 repository**:
   ```bash
   git clone https://github.com/yl4579/StyleTTS2.git tmp/StyleTTS2
   ```

2. **Download pre-trained models**:
   - Download from [https://huggingface.co/yl4579/StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS)
   - Extract to `tmp/StyleTTS2/Models/LibriTTS/`
   - You need: `config.yml` and `epochs_2nd_00020.pth`

3. **Install additional dependencies**:
   ```bash
   uv pip install phonemizer librosa nltk munch
   
   # Install espeak (required for phonemizer)
   # Windows: Download from https://github.com/espeak-ng/espeak-ng/releases
   # macOS: brew install espeak-ng
   # Linux: sudo apt-get install espeak-ng
   ```

**Configuration:**

Add a voice preset using StyleTTS2 in `config.yaml`:

```yaml
voices:
  ada:
    model: "styletts2"                    # Use StyleTTS2 instead of Kokoro
    reference_audio: tmp/sc2-adjutant.wav # Reference audio for voice cloning
    speed: 1.5
    # Optional: StyleTTS2 parameters (defaults shown)
    alpha: 0.3              # Timbre blending (0=reference, 1=predicted)
    beta: 0.7               # Prosody blending (higher=more emotional)
    diffusion_steps: 5      # Quality vs speed (5-10 recommended)
    embedding_scale: 1.0    # Classifier-free guidance scale
```

**Usage:**

```bash
# Use StyleTTS2 voice (same as Kokoro)
voice ada "Affirmative, Commander. All systems operational."

# The voice will be cloned from tmp/sc2-adjutant.wav
```

**StyleTTS2 Parameters:**

- **`alpha` (0.0-1.0)**: Controls timbre similarity to reference
  - `0.0` = Maximum similarity to reference voice (good for maintaining acoustic environment)
  - `1.0` = More generic, model-predicted timbre
  - Default: `0.3` (good balance)

- **`beta` (0.0-1.0)**: Controls prosody and emotion
  - `0.0` = Maintains reference emotion/prosody
  - `1.0` = More emotional and text-driven prosody
  - Default: `0.7` (expressive but controlled)

- **`diffusion_steps` (1-15)**: Quality vs speed tradeoff
  - `5` = Fast, good quality (default)
  - `10` = Higher quality, more diverse samples
  - `15+` = Best quality, slowest

- **`embedding_scale` (0.8-2.0)**: Style conditioning strength
  - `1.0` = Standard (default)
  - `1.5-2.0` = More expressive, emotional speech
  - Higher values may introduce artifacts

**Tips:**

- Use high-quality reference audio (clear speech, minimal background noise)
- Reference audio length: 5-15 seconds works best
- For consistent timbre, use `alpha=0.0, beta=0.5`
- For emotional variation, use `alpha=0.3, beta=0.9, embedding_scale=2.0`
- StyleTTS2 is slower than Kokoro (~2-5s vs ~0.2s) but offers unlimited voice options

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
