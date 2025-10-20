# 🗣️Voice

A cross-platform text-to-speech (TTS) system with voice presets and optional LLM text transformation. Works on Windows 11 and macOS with local processing for privacy.

## Features

- 🎭 **Multiple Voice Presets**: GLaDOS, Kokoro (multiple voices), ElevenLabs, and more
- 🤖 **Optional LLM Integration**: Transform text with personality using local Ollama
- 🖥️ **Cross-Platform**: Windows 11 (NVIDIA GPU) and macOS (M1/M2)
- 🔒 **Privacy-First**: All processing happens locally (except ElevenLabs)
- ⚡ **GPU Accelerated**: Leverages ONNX Runtime with GPU support
- 🎵 **Flexible Output**: Play audio or save to OGG format

## Installation

### Prerequisites

**Windows 11:**
- Python 3.9+
- NVIDIA GPU with CUDA support (optional but recommended)

**macOS:**
- Python 3.9+
- Apple Silicon (M1/M2) recommended

### Setup

1. **Create virtual environment:**
  ```bash
  uv venv --python 3.12.8
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```

2. **Install dependencies:**
   
   **For Windows with NVIDIA GPU:**
   ```cmd
   uv pip install -r requirements.txt
   uv pip install onnxruntime-gpu
   ```
   
   **For macOS:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Install the package:**
   ```bash
   uv pip install -e .
   ```

4. **Set up Ollama (optional, for text-to-text LLM features; ie. if you want to let the voice change the text to match a personality):**
   - Download and install [Ollama](https://ollama.ai)
   - Pull a model: `ollama pull llama3.2`
   - Ensure Ollama is running: `ollama serve`

5. **Download TTS models:**
   - The default config expects models in `../models/TTS/`
   - Ensure you have:
     - `glados.onnx` and `glados.json` for GLaDOS voice
     - `kokoro-v1.0.fp16.onnx` for Kokoro voices
     - `phomenizer_en.onnx` for phonemization

## Quick Start

### List available voices:
```bash
voice --list
```

### Synthesize speech:
```bash
voice glados "Hello, test subject."
voice bella "It's a beautiful morning. Time for a cup of coffee."
voice adam "Another day another dollar."
```

### Save to file:
```bash
voice -o output.ogg glados "Save this to a file"
```

### Get voice information:
```bash
voice --info glados
```

### Use custom config:
```bash
voice --config custom_config.yaml myvoice "Custom text"
```

## Usage

```
voice <preset> <text> [options]

Positional Arguments:
  preset                Voice preset name (glados, pirate, neutral, etc.)
  text                  Text to synthesize

Options:
  -o FILE, --output FILE    Save audio to file (OGG format)
  -c FILE, --config FILE    Use custom config file
  -l, --list                List available voice presets
  -i PRESET, --info PRESET  Show information about a voice preset
  -v, --version             Show version
  -h, --help                Show help message

Examples:
  voice glados "Hello, test subject."
  voice bella "You've got mail."
  voice -o output.ogg glados "Save to file"
  voice --list
  voice --info glados
```

## Configuration

Voice presets are configured in `config.yaml`. Each preset can specify:

- **TTS Engine**: `glados`, or `kokoro`
- **Model Path**: Path to ONNX model files
- **Voice Parameters**: Speed, voice variant, etc.
- **LLM Integration**: Enable text transformation with system prompts
- **System Prompt**: Personality/style for text transformation

### Example Voice Preset

```yaml
voices:
  glados:
    tts_engine: "glados"
    model_path: "../models/TTS/glados.onnx"
    phonemizer_path: "../models/TTS/phomenizer_en.onnx"
    speed: 1.0
    enable_llm: true
    system_prompt: |
      You are GLaDOS, a sarcastic AI.
      Transform text to match your signature dry, 
      emotionless tone from Portal.
```

### Adding Custom Voices

1. Edit `config.yaml`
2. Add a new voice preset under `voices:`
3. Specify the TTS engine and parameters
4. Optionally add a system prompt for LLM transformation

```yaml
voices:
  my_custom_voice:
    tts_engine: "kokoro"
    model_path: "../models/TTS/kokoro-v1.0.fp16.onnx"
    phonemizer_path: "../models/TTS/phomenizer_en.onnx"
    voice: "af_bella"
    speed: 1.0
    enable_llm: true
    system_prompt: |
      Your custom personality instructions here...
```

## Supported TTS Engines

### 1. GLaDOS Voice
- Custom-trained GLaDOS voice from Portal
- ONNX-based, GPU-accelerated
- Requires `glados.onnx` model

### 2. Kokoro Multi-Voice
- Multiple voice variants (male/female)
- Supported voices: `af_bella`, `am_adam`, `af_sarah`, `am_michael`, etc.
- ONNX-based, GPU-accelerated
- Requires `kokoro-v1.0.fp16.onnx` model

## LLM Integration

Voice can optionally transform text using a local LLM before synthesis:

1. **Install Ollama**: https://ollama.ai
2. **Pull a model**: `ollama pull llama3.2`
3. **Enable in config**: Set `enable_llm: true` for a voice preset
4. **Add system prompt**: Define the transformation style

Example personalities:
- **GLaDOS**: Sarcastic, scientific AI from Portal
- **Pirate**: Arr, matey! Talk like a pirate!
- **Custom**: Define your own personality

## Platform-Specific Notes

### Windows 11 (NVIDIA RTX 5070 Ti)
- Install `onnxruntime-gpu` for GPU acceleration
- CUDA support enables fast inference
- Ensure NVIDIA drivers are up to date

### macOS (M1/M2)
- Uses ONNX Runtime with CoreML backend
- CPU inference is efficient on Apple Silicon
- No GPU package needed (automatically optimized)

## Troubleshooting

### Models not found
Ensure model files are in the correct location specified in `config.yaml`. Update paths as needed:
```yaml
model_path: "/absolute/path/to/models/TTS/glados.onnx"
```

### Ollama connection failed
- Ensure Ollama is running: `ollama serve`
- Check the API URL in config: `http://localhost:11434/api/chat`
- LLM features will be skipped if Ollama is unavailable

### Audio playback issues
- **Windows**: Ensure audio drivers are installed
- **macOS**: Grant microphone/audio permissions if prompted
- Use `-o` flag to save to file instead of playing

### GPU not detected
- **Windows**: Install CUDA and `onnxruntime-gpu`
- **macOS**: ONNX Runtime automatically uses optimized backends
- CPU fallback is automatic if GPU is unavailable

## Acknowledgments

- Based on the [GLaDOS project](https://github.com/dnhkng/GLaDOS)
- Uses ONNX Runtime for cross-platform inference
- Integrates with Ollama for local LLM processing
