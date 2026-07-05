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
