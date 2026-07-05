# 🗣️ voice v2

Always-on local text-to-speech CLI + daemon (Piper + Kokoro), written in
Zig + ONNX Runtime. CLI, unix socket, HTTP API, and a browser SPA all
talk to the same running process (the `presence-voice` systemd service,
see `voice.service`).

| Version | Branch | Notes |
|---|---|---|
| **v2** (this branch, in progress) | `v2` | Zig daemon, ONNX Runtime (CPU/CUDA), HTTP + CLI + SPA, always-on |
| [v1](https://github.com/mikesmullin/voice/tree/v1) | `v1` | Original Python CLI/server (Piper + Kokoro), still fully working |

v2 is a rewrite (see `tmp/PHASE3_PLAN.md` for the full design/decisions
log), not yet the default branch. See `SKILL.md` for the operator guide
and known gaps.

## Why a rewrite

v1 (Python) worked, but every CLI invocation paid Python interpreter
startup + PyTorch import cost, and there was no single persistent process
keeping models warm *and* an audio device open at once. v2 is a single
Zig daemon: ONNX Runtime sessions loaded once, one PulseAudio connection
kept open and reused, unix socket for the CLI, optional HTTP API for
everything else (curl, the browser SPA).

## Quickstart

```bash
git clone --recurse-submodules git@github.com:mikesmullin/voice.git
cd voice
git checkout v2   # not yet the default branch

# system dependencies (Arch shown; adjust for your distro)
sudo pacman -S espeak-ng cudnn   # cudnn optional, only needed for Kokoro's CUDA EP

# Zig toolchain - see build.zig.zon's minimum_zig_version (currently a
# Zig master snapshot; a stable release didn't work on this dev machine's
# bleeding-edge glibc - see SKILL.md's "Toolchain notes")
curl https://www.zvm.app/install.sh | bash
zvm install master && zvm use master

# ONNX Runtime (vendored, not committed - vendor/onnxruntime is gitignored)
mkdir -p vendor && cd vendor
curl -fLO https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-gpu_cuda13-1.27.0.tgz
tar xzf onnxruntime-linux-x64-gpu_cuda13-1.27.0.tgz
mv onnxruntime-linux-x64-gpu_cuda13-1.27.0 onnxruntime
cd ..

# Model files (not committed - ./models/ is gitignored). Also initializes
# vendor/zig-phonemes (a git submodule - build-time G2P dependency) if you
# didn't clone with --recurse-submodules above.
git submodule update --init
./scripts/fetch-models.sh

zig build
./zig-out/bin/voice list
./zig-out/bin/voice local lessac "Hello from Piper."
```

To run the daemon (recommended - see `voice.service` for systemd
--user):

```bash
./zig-out/bin/voice serve --http
# in another shell:
./zig-out/bin/voice bella "Hello from Kokoro, via the daemon."
curl http://127.0.0.1:3124/health
```

## Status

Working end-to-end: both engines synthesize real audio via ONNX Runtime
(Kokoro on GPU when available, falling back to CPU; Piper always CPU),
`local`/`client`/`list`/`serve --http` all function, the HTTP API and
browser SPA work, systemd unit drafted. See `SKILL.md`'s "Known gaps" for
what's intentionally unfinished (per-request gain/output over the daemon
protocol, stingers, hardcoded model paths).

## Documentation

- `SKILL.md` - operator guide (commands, HTTP API, known gaps, toolchain notes)
- `tmp/PHASE3_PLAN.md` - full design rationale and decisions log
- `voice.service` - systemd --user unit

- [Hugging Face](https://huggingface.co/)