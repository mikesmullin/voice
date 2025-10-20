# TTS Models Directory

This directory contains the required model files for the Voice TTS system.

## Downloading Models

Run the download script to fetch the main ONNX model files:

```bash
python download.py
```

This will download:
- `TTS/glados.onnx` - GLaDOS voice model
- `TTS/kokoro-v1.0.fp16.onnx` - Kokoro multi-voice model
- `TTS/kokoro-voices-v1.0.bin` - Kokoro voice embeddings
- `TTS/phomenizer_en.onnx` - English phonemizer model

## Required Additional Files

The following files are required but not available for automatic download. They must be restored from a backup or obtained from the original sources:

### GLaDOS Voice Files
- `TTS/glados.json` - GLaDOS model configuration
- `TTS/phoneme_to_id.pkl` - Phoneme-to-ID mapping for GLaDOS

### Phonemizer Support Files
- `TTS/lang_phoneme_dict.pkl` - Language phoneme dictionary
- `TTS/token_to_idx.pkl` - Token-to-index mapping
- `TTS/idx_to_token.pkl` - Index-to-token mapping

## Sources

- Main ONNX models: [GLaDOS GitHub Releases](https://github.com/dnhkng/GLaDOS/releases)
- PKL/JSON files: Originally from [Piper TTS](https://github.com/rhasspy/piper) via GLaDOS project

## Directory Structure

```
models/
├── download.py          # Download script for ONNX models
├── README.md           # This file
└── TTS/
    ├── glados.onnx
    ├── glados.json
    ├── phoneme_to_id.pkl
    ├── kokoro-v1.0.fp16.onnx
    ├── kokoro-voices-v1.0.bin
    ├── phomenizer_en.onnx
    ├── lang_phoneme_dict.pkl
    ├── token_to_idx.pkl
    └── idx_to_token.pkl
```

## Verifying Installation

After placing all required files, test the installation:

```bash
voice glados "Hello, test subject."
voice bella "This is a test."
```
