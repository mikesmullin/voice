//! Fixed absolute paths to this project's own checkout on this machine.
//!
//! Needed because `voice` is meant to be installed globally on $PATH and
//! invoked from any directory (e.g. `cd ~ && voice hello world`), but its
//! config/model/vendor assets aren't installed anywhere standard - they
//! just live in this git checkout. Rather than silently depending on the
//! caller's cwd (which broke the first time this was tried - see the
//! session notes), every default path is anchored to this repo's known
//! location.
//!
//! Known limitation: single-machine, single-checkout-location assumption
//! (matches ESPEAK_LIB_PATH/ESPEAK_DATA_PATH's existing style). If this
//! repo is ever cloned somewhere else, update ROOT below (or better: add
//! a proper install step / environment variable override - not done yet).

pub const ROOT = "/workspace/voice";

pub const CONFIG = ROOT ++ "/config.yaml";
pub const ZIG_PHONEMES_DATA = ROOT ++ "/vendor/zig-phonemes/data";
pub const KOKORO_VOCAB = ZIG_PHONEMES_DATA ++ "/kokoro_vocab.json";
pub const KOKORO_MODEL = ROOT ++ "/models/kokoro/kokoro-v1.0.fp16.onnx";
pub const KOKORO_VOICES_BIN = ROOT ++ "/models/kokoro/voices-v1.0.bin";
pub const PIPER_MODELS_DIR = ROOT ++ "/models/piper";
pub const WEB_INDEX = ROOT ++ "/web/index.html";

// Audio8 TTS (see src/engines/audio8.zig). The model weights are downloaded
// once via `hf download` into /workspace/models/audio8-tts/ (see
// scripts/setup-audio8.sh) - deliberately OUTSIDE this repo since the full
// model is ~820MB. The inference code itself (official `arktts_runtime`,
// Apache 2.0) is vendored at `vendor/audio8-runtime/` (gitignored) alongside
// onnxruntime and zig-phonemes - see scripts/setup-audio8.sh for how to recreate it.
pub const AUDIO8_MODEL_DIR = "/workspace/models/audio8-tts";
pub const AUDIO8_RUNTIME_DIR = ROOT ++ "/vendor/audio8-runtime";
pub const AUDIO8_VENV_PYTHON = AUDIO8_RUNTIME_DIR ++ "/.venv/bin/python";
/// Registered voices ("default" = the packaged reference voice in the model
/// itself). Lives next to the runtime code, not inside the download dir.
pub const AUDIO8_VOICES_DIR = AUDIO8_RUNTIME_DIR ++ "/voices";
/// Temporary WAV written (and unlinked immediately) around each synthesis.
// (Absolute path - the Python subprocess writes it, Zig reads it back via
// wav.readMono, which supports absolute paths through .cwd().)
pub const AUDIO8_TMP_WAV = "/tmp/presence-voice-audio8-";

// Audio8 GPU backend (PyTorch checkpoint, bf16/CUDA - see
// scripts/audio8_gpu_synth.py and scripts/setup-audio8-gpu.sh). Same
// voices dir as the CPU backend: synthesis reads the registered voice's
// codes.npy + reference_text straight out of voices/<name>/, so one
// registration serves both backends.
pub const AUDIO8_PT_MODEL_DIR = "/workspace/models/audio8-tts-pt";
pub const AUDIO8_GPU_VENV_PYTHON = "/workspace/models/audio8-gpu/.venv/bin/python";
pub const AUDIO8_GPU_SYNTH_SCRIPT = ROOT ++ "/scripts/audio8_gpu_synth.py";

// Qwen3-TTS engine (see src/engines/qwen.zig). Weights live outside this
// repo at /workspace/models/qwen3-tts/ (fetched via `hf download` - see
// scripts/setup-qwen.sh). Two modes, one preset, selected per request:
//   - CPU (default): llama-tts with `-ngl 0` (Q4_K_M backbone + Q8_0 mmproj)
//   - GPU:           `-ngl 99` (full CUDA offload; needs an NVIDIA GPU)
// The binary is the system-packaged llama.cpp `llama-tts`; it handles the
// whole text->audio pipeline (tokenization, prompt construction, sampling,
// vocoder), so Zig never reimplements the Qwen3-TTS stack.
pub const QWEN_MODEL_DIR = "/workspace/models/qwen3-tts";
pub const QWEN_BACKBONE = QWEN_MODEL_DIR ++ "/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf";
pub const QWEN_MMPROJ = QWEN_MODEL_DIR ++ "/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf";
/// System-installed llama.cpp binary (pacman/apt: llama.cpp). Not vendored.
pub const QWEN_LLAMATTS = "/usr/bin/llama-tts";
/// Temporary WAV written (and unlinked immediately) around each synthesis.
pub const QWEN_TMP_WAV = "/tmp/presence-voice-qwen-";
