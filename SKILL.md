# voice v2 - Operator Guide

Fast operator manual for using the `voice` CLI (v2, Zig + ONNX Runtime).
The always-on daemon runs as the `presence-voice` systemd service (see
`presence-voice.service`); `voice` is the CLI binary and the name of the package
as a whole. Successor to v1's `SKILLS.md` (kept on the `v1` branch). See
`README.md` for the v1/v2 comparison and `tmp/PHASE3_PLAN.md` for the
full design rationale.

## Status

v2 is a working prototype, not yet the default branch. All commands below
are real and tested, but read the "Known gaps" section - some things
(gain/output over the daemon, stingers, model paths in config.yaml) are
intentionally not finished yet.

## Default Operating Guidance

- The daemon (`voice serve`) is meant to be always-on (e.g. via
  `presence-voice.service`) - prefer the bare shorthand or `client` over `local`
  once it's running, since `local` pays full model-load cost every call.
- Kokoro runs on GPU (CUDA execution provider) when available, falling
  back to CPU automatically - you'll see which one was used logged at
  load time (`[onnxruntime] CUDA execution provider ready` or
  `using CPU execution provider`).
- Piper always runs on CPU (tiny model, already realtime - not worth GPU).

## Fetching models

Model files aren't committed (`./models/` is gitignored). Fetch them once:

```bash
git submodule update --init   # vendor/zig-phonemes (build-time G2P dependency)
./scripts/fetch-models.sh     # ./models/kokoro/ + ./models/piper/
```

By default this fetches Kokoro's latest fp16 ONNX export + voices pack
(from `thewh1teagle/kokoro-onnx`'s releases - one shared file covers
every Kokoro preset), plus **every** Piper preset in `config.yaml`'s
`voices:` map (~28 voices, ~1GB total) - so everything configured is
ready to go, not just the preloaded ones. On a slow connection, pass
`--preload-only` to fetch only the Piper presets in `preload:` instead:

```bash
./scripts/fetch-models.sh --preload-only
```

The **Audio8 TTS** engine (compact multilingual zero-shot TTS) has TWO
backends sharing one engine name and one voice registration:
- **CPU** (default): official INT8 ONNX runtime, ~820MB of weights at
  /workspace/models/audio8-tts/ - set up with `./scripts/setup-audio8.sh`
  (below). No GPU needed.
- **GPU** (`--gpu`): full-precision PyTorch checkpoint on CUDA, ~1.6GB at
  /workspace/models/audio8-tts-pt/ + a torch venv at
  /workspace/models/audio8-gpu/.venv - set up with
  `./scripts/setup-audio8-gpu.sh`. Needs an NVIDIA GPU; CPU-only machines
  skip this. Both backends read the SAME registered voice
  (codes.npy + reference_text out of vendor/audio8-runtime/voices/<name>/),
  so one clone serves `voice local ruth ...` with or without `--gpu`.

```bash
./scripts/setup-audio8.sh       # CPU backend (required for any audio8 use)
./scripts/setup-audio8-gpu.sh   # GPU backend (only if you want --gpu)
```

`./scripts/audio8-register-voice.sh --wav ref.wav --name ruth
--transcript t.txt` clones a new voice from a 0.5-30s reference recording
+ its exact transcript (writes vendor/audio8-runtime/voices/<name>/, then
add a matching `engine: audio8` preset to config.yaml). "Ruth" was
registered this way from ~/Downloads/ruth2.wav (29.6s) + the narration
transcript kept in the commit message that introduced it.

This downloads the model to /workspace/models/audio8-tts/ (via `hf download`),
vendors the official `arktts_runtime` Python package into
vendor/audio8-runtime/ (gitignored - cloned from Audio8-AI/Audio8_TTS at a
pinned commit), creates a venv there, and registers the packaged reference
voice ("default"). Requires `hf` on PATH and a Python 3.10+ interpreter
for the venv (the script tries python3.10 -> python3.13 -> python3; it
skips broken ones and fails with a clear message if nothing works).

## Core Commands

### Direct synthesis (daemon-backed - the fast path)

```bash
voice [preset] <text>              # bare shorthand for "client"
voice client [preset] <text>       # explicit, identical result
```

Preset is optional - omit it to use `config.yaml`'s `default_preset`/
`fallback_voice`. Fails fast with a clear error (exit 1) if the daemon
isn't reachable - it will NOT silently fall back to standalone synthesis.

```bash
voice lessac "Piper is fast and CPU-only."
voice bella "Kokoro, GPU-accelerated when available."
voice "No preset given, uses the configured default."

# Audio8 (two backends, one preset - --cpu default, --gpu for CUDA):
voice audio8 "Compact multilingual zero-shot TTS, cloned from the packaged reference voice."
voice local ruth "Ruth on CPU (default, no flag needed)."
voice local ruth --gpu "Ruth on CUDA (audio8 presets only)."
voice ruth --gpu "Same, but via the daemon (needs the daemon running)."
```

`--cpu`/`-C` and `--gpu`/`-gpu` are mutually exclusive (combining them is
a hard error); `--gpu` with a non-audio8 preset is a hard error too. `-C`
keeps its old meaning for Kokoro (force CPU); for audio8 it just selects
the default CPU backend explicitly.

### Standalone synthesis (no daemon required)

```bash
voice local [options] [preset] <text>
```

Pays full model-load cost every invocation (that's the tradeoff for not
needing a daemon). Supports `-o` (save to WAV), `-g` (gain), `-C` (force
CPU for Kokoro) - these are **not yet available** for `client`/bare
requests, only `local` (see "Known gaps").

```bash
voice local -o out.wav lessac "Write this to disk."
voice local -g 1.4 -C bella "Force CPU, boost volume."
```

### List / inspect presets

```bash
voice list                # table of every configured preset
voice -i alan              # details for one preset
```

### Start the daemon

```bash
voice serve                # unix socket only
voice serve --http         # ...also start the HTTP API (127.0.0.1:3124)
```

Preloads every preset in `config.yaml`'s `preload:` list at startup (same
mechanism validated in Phase 2 - one Kokoro preload warms *all* Kokoro
presets; each Piper preset needs its own preload entry). See
`presence-voice.service` for the systemd --user unit.

### HTTP API (with `serve --http`)

```
GET  /health           -> {"status":"ok"}
GET  /voices            -> [{"name","engine","voice"}, ...] for every preset
POST /speak              -> body {"text", "voice"?, "mode"?, "gain"?}
                             mode="play" (default): plays through the
                               daemon's own speakers
                             mode="download": returns WAV bytes (what the
                               browser SPA at GET / uses)
```

```bash
curl -X POST http://127.0.0.1:3124/speak \
  -d '{"text":"Hello","voice":"bella","mode":"download"}' -o out.wav
```

The browser SPA (`web/index.html`, served at `GET /`) is a working
Alpine.js + Tailwind demo page - open `http://127.0.0.1:3124/` once
`serve --http` is running.

## Known gaps (intentional, not oversights)

- **Audio8 is subprocess-backed, not in-process** - each `audio8` request
  spawns a Python runtime and reads the resulting WAV back into the normal
  output path. CPU path: official ONNX runtime
  (`vendor/audio8-runtime/.venv/bin/python -m arktts_runtime.cli ...`),
  ~12-15s per call including session load. GPU path (`--gpu`):
  `scripts/audio8_gpu_synth.py` under the torch venv (~7-20s cold per call -
  no persistent process holds the CUDA kernels warm yet; a warm in-process
  run does the same work in ~4s, so a persistent GPU worker is the obvious
  next latency win). Neither backend is preloadable (the preload list only
  covers Kokoro/Piper presets with persistent sessions) and the `speed`
  config field is ignored for audio8 (no speed knob in either upstream CLI).
  `-C/--cpu` is meaningless for audio8 beyond selecting the default.
- **Unix-socket protocol is v2**: `preset\tspeaker\teffects\tschedule\t[backend\t]text`
  (`backend` = `cpu`/`gpu`, audio8 only). The daemon accepts old 5-field
  lines (no backend = CPU); new CLIs always send 6 fields. Same for HTTP
  `POST /speak`, which takes an optional `"backend"` field.
  The engine also ignores `-C/--cpu` (the export is CPU-only by design) -
  that flag is meaningless for it, it just works.

- **Model paths are constants, not a `config.yaml` field**
  (`src/daemon.zig`): they now point at `./models/kokoro/` and
  `./models/piper/` (self-contained, fetched via
  `./scripts/fetch-models.sh` - see "Fetching models" above) rather than
  a dev-machine-specific path, but there's still no approved v2 schema
  for per-preset model paths yet (see `tmp/PHASE3_PLAN.md`'s "Still
  open" section). Only presets added to `config.yaml` *after* running
  `fetch-models.sh` (or ones deliberately skipped via `--preload-only`)
  need their `.onnx`/`.onnx.json` fetched into `models/piper/` by hand.
- **`-o`/`-g` only work with `local`**, not `client`/bare requests - the
  unix socket protocol is a `preset\tspeaker\teffects\ttext\n` line (empty
  `speaker`/`effects` fields for "default sink"/"no effects"), with no
  room for `-o`/`-g` yet. HTTP's `POST /speak` DOES support `gain` and
  `mode=download` (a different, JSON-based protocol), plus `speaker`/
  `effects` fields matching the unix-socket protocol's semantics.
- **`-s/--stinger`** (the standalone flag) is still parsed but unwired -
  stinger playback is only reachable today via an effect preset's
  `chain:` (a `stinger:` step with a `file:` param), applied with
  `-e/--effect <preset_name>` - see `tmp/FUN_PLAN.md` section 2 and
  `config.yaml`'s `effects:` block for an example (`radio_comms`).
- **`-d/--speaker`/`-e/--effect`** work for `local`, `client`, and HTTP.
  Speaker selection is Linux-only (`src/audio/linux_sink.zig`, a direct
  `pa_simple` connection - sokol_audio has no device-selection API on any
  platform); combining an explicit speaker with an effect chain that has
  a `stinger` step plays the voice audio through that speaker but skips
  the stinger pre-roll (stingers need the World's channel queue, which
  the speaker-specific path bypasses) - a known, documented gap, not a
  silent one.
- **Single-threaded unix socket loop** - one request at a time, no
  concurrent connections.
- **No NFD Unicode normalization** in Piper's phoneme splitting (Zig's
  std has none built in) - a few precomposed diacritics may be silently
  dropped as "missing phoneme". Hasn't been audibly wrong so far.
- **G2P for Kokoro** uses Fable's `zig-phonemes` (evaluated as a misaki
  alternative, see `tmp/PHENOMES.md`) - a git submodule at
  `vendor/zig-phonemes` (run `git submodule update --init` after
  cloning). **espeak-ng is attached as OOV fallback** so proper names
  are not silent; without it, missing lexicon entries produce empty
  phoneme spans (gaps in speech). Custom overrides live in
  `~/.config/voice/names.yaml` (created with header docs on first run;
  restart `presence-voice` after edits). Debug: `voice --phonemize "…"`.

## Toolchain notes

- Zig version is pinned in `build.zig.zon`'s `minimum_zig_version` -
  currently a Zig master snapshot, required because this machine's
  bleeding-edge glibc broke Zig 0.16.0's linker (fixed in master; see
  `/memories/repo/voice-v2-zig-notes.md` for the full story if you hit
  linker errors on a different machine).
- ONNX Runtime is vendored under `vendor/onnxruntime` (gitignored, fetch
  via the `gpu_cuda13` release tarball - see that same memory file).
  Needs system `cudnn` (`pacman -S cudnn` or equivalent) for the CUDA
  execution provider; falls back to CPU automatically if unavailable.
- `espeak-ng` must be installed system-wide (`pacman -S espeak-ng` or
  equivalent) - both engines dlopen `libespeak-ng.so` at runtime.
