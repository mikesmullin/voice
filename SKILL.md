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
```

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
  cloning). Revisit once the comparison against plain espeak-ng is
  finalized.

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
