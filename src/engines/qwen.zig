//! Qwen engine: Alibaba's Qwen3-TTS-12Hz-1.7B-Base, zero-shot voice cloning.
//!
//! Two backends share this engine's contract (f32 samples + rate back into
//! the normal output stack):
//!   - CPU (`synthesize`): resident llama-server sidecar with `-ngl 0`
//!     (Q4_K_M backbone + Q8_0 mmproj)
//!   - GPU (`synthesizeGpu`): same sidecar with `-ngl 99` (full CUDA
//!     offload), with a loud-logged fallback to CPU when CUDA init fails
//!     (e.g. VRAM pressure) - degraded-but-audible beats a failed request,
//!     matching the LLM-rewrite fallback philosophy (see tmp/FUN_PLAN.md
//!     section 4).
//! Both read the same speaker WAV and model files, so one speaker reference
//! serves both modes.
//!
//! Strategy is sidecar-first: a persistent `llama-server` (see
//! qwen_server.zig - started lazily on the first qwen request, never at
//! `voice serve` startup) holds the ~1.5 GB weights resident, so repeat
//! calls pay inference only instead of a full reload (~1.7s saved per
//! call). When the sidecar can't start or a request fails, the engine
//! falls back to the one-shot `llama-tts` subprocess (same flags as
//! before - correct audio, just slower).
//!
//! Why not a native Zig engine: Qwen3-TTS is a small but intricate model
//! (AR talker + 4-layer codec, 12Hz token rate) and the upstream llama.cpp
//! implementation already owns the whole text->audio pipeline
//! (tokenization, prompt construction, sampling, vocoder). Reimplementing
//! that in Zig would duplicate the runtime for no gain. We just bounce the
//! resulting WAV through our own wav/output stack (gain, effects, speaker
//! routing, the daemon's audio pipeline all stay working unchanged).
//!
//! Weights (~1.5GB) live outside this repo at /workspace/models/qwen3-tts/
//! (fetch via scripts/setup-qwen.sh). Like audio8, this engine is NOT
//! preloadable by the daemon: the first qwen request pays the sidecar
//! model load instead.
//!
//! Reuses `wav.readMono` for the WAV parse - both paths produce PCM 16-bit
//! little-endian mono at 24kHz.

const std = @import("std");
const wav = @import("../audio/wav.zig");
const paths = @import("../paths.zig");
const qwen_server = @import("qwen_server.zig");

pub const SynthResult = struct { samples: []f32, sample_rate: u32 };

/// Default generation settings from the reference benchmark flags: 8 CPU
/// threads, 4096-token context window. (`-ngl 0` selects pure CPU.)
pub const THREADS: u32 = 8;
pub const CTX_SIZE: u32 = 4096;

pub const QwenError = error{
    /// The llama-tts binary is missing (not installed / not on the expected path).
    RuntimeNotInstalled,
    /// The preset has no speaker file configured, or it doesn't exist.
    SpeakerNotConfigured,
    /// The subprocess exited non-zero (stderr included in the log line by the caller).
    SubprocessFailed,
    /// The produced WAV was missing/unreadable.
    WAVReadFailed,
} || wav.WavError || std.process.RunError || std.mem.Allocator.Error;

/// True when the subprocess stderr looks like a CUDA/VRAM failure (as
/// opposed to a bad flag, missing file, etc.) - the cases where retrying
/// the exact same spawn on CPU is worth one shot.
fn isCudaFailure(stderr: []const u8) bool {
    const markers = [_][]const u8{
        "CUDA error",
        "cudaMalloc failed",
        "out of memory",
        "failed to fit",
        "cublas",
    };
    for (markers) |m| {
        if (std.ascii.findIgnoreCase(stderr, m) != null) return true;
    }
    return false;
}

fn spawn(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    speaker_path: []const u8,
    gpu_layers: u32,
    allow_cpu_fallback: bool,
    log: ?*std.Io.Writer,
) !SynthResult {
    // Fail fast with a clear error rather than letting the subprocess spin.
    const speaker_file = std.Io.Dir.cwd().openFile(io, speaker_path, .{}) catch {
        return error.SpeakerNotConfigured;
    };
    speaker_file.close(io);

    var out_path_buf: [256]u8 = undefined;
    const out_path = try std.fmt.bufPrint(
        &out_path_buf,
        "{s}{d}.wav",
        .{ paths.QWEN_TMP_WAV, std.c.getpid() },
    );

    // Ensure a clean slate: a crashed prior run may have left a stale WAV
    // behind; the subprocess rewrites it, but never read a stale file.
    std.Io.Dir.deleteFileAbsolute(io, out_path) catch {};

    var gpu_buf: [16]u8 = undefined;
    var threads_buf: [16]u8 = undefined;
    var ctx_buf: [16]u8 = undefined;
    const gpu = try std.fmt.bufPrint(&gpu_buf, "{d}", .{gpu_layers});
    const threads = try std.fmt.bufPrint(&threads_buf, "{d}", .{THREADS});
    const ctx = try std.fmt.bufPrint(&ctx_buf, "{d}", .{CTX_SIZE});

    // CPU spawns hide the GPU entirely (CUDA_VISIBLE_DEVICES= empty): the
    // llama-tts binary is a CUDA build and initializes cuBLAS even for
    // -ngl 0, which fails when VRAM is exhausted. `env(1)` keeps the rest
    // of the parent environment inherited (arg0 resolves via parent PATH
    // either way). GPU spawns inherit the parent environment untouched.
    var argv_buf: [21][]const u8 = undefined;
    var argv_len: usize = 0;
    if (gpu_layers == 0) {
        argv_buf[0] = "env";
        argv_buf[1] = "CUDA_VISIBLE_DEVICES=";
        argv_len = 2;
    }
    const rest = [_][]const u8{
        paths.QWEN_LLAMATTS,
        "-m",
        paths.QWEN_BACKBONE,
        "-mm",
        paths.QWEN_MMPROJ,
        "-p",
        text,
        "--tts-lang",
        "en",
        "--tts-speaker-file",
        speaker_path,
        "-o",
        out_path,
        "-ngl",
        gpu,
        "-t",
        threads,
        "-c",
        ctx,
    };
    @memcpy(argv_buf[argv_len..][0..rest.len], &rest);
    const argv = argv_buf[0 .. argv_len + rest.len];

    const result = std.process.run(alloc, io, .{ .argv = argv }) catch |err| {
        switch (err) {
            error.FileNotFound, error.AccessDenied => return error.RuntimeNotInstalled,
            else => return err,
        }
    };

    if (!result.term.success()) {
        // GPU was asked for but CUDA is unusable right now (usually VRAM
        // pressure): one loud-logged retry on CPU. Recursion is depth-1
        // (the retry passes allow_cpu_fallback=false).
        if (gpu_layers != 0 and allow_cpu_fallback and isCudaFailure(result.stderr)) {
            if (log) |l| {
                l.print("[Qwen] CUDA backend failed, falling back to CPU (-ngl 0)\n", .{}) catch {};
            }
            return spawn(alloc, io, text, speaker_path, 0, false, log);
        }
        if (log) |l| {
            l.print("[Qwen] subprocess failed (exited nonzero): {s}\n", .{result.stderr}) catch {};
        }
        return error.SubprocessFailed;
    }
    if (log) |l| {
        if (result.stderr.len > 0) l.print("[Qwen] subprocess stderr: {s}\n", .{result.stderr}) catch {};
    }

    const data = wav.readMono(alloc, io, out_path) catch {
        return error.WAVReadFailed;
    };

    // Best-effort cleanup of the temp file.
    std.Io.Dir.deleteFileAbsolute(io, out_path) catch {};

    return .{ .samples = data.samples, .sample_rate = data.sample_rate };
}

/// Synthesizes `text` with the given speaker reference WAV (CPU backend)
/// and optional sibling transcript file (may be null - the server still
/// clones without it). Sidecar first, one-shot subprocess on failure.
/// The result is allocated from `alloc` and lives as long as that
/// allocator does - matching the other engines' contract.
pub fn synthesize(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    voice: []const u8,
    speaker_path: []const u8,
    speaker_text_path: ?[]const u8,
    log: ?*std.Io.Writer,
) !SynthResult {
    if (qwen_server.synthesizeViaServer(alloc, io, text, voice, speaker_path, speaker_text_path, false, log)) |data| {
        return .{ .samples = data.samples, .sample_rate = data.sample_rate };
    } else |err| {
        if (log) |l| l.print("[Qwen] sidecar unavailable ({t}), using one-shot subprocess\n", .{err}) catch {};
        return spawn(alloc, io, text, speaker_path, 0, false, log);
    }
}

/// GPU twin: same contract (text in, samples + rate out), via full CUDA
/// offload (`-ngl 99`). Same speaker WAV and model files as the CPU path.
/// Sidecar first; one-shot subprocess on failure (which itself falls back
/// to CPU, loudly logged, when CUDA init fails - see `spawn`).
pub fn synthesizeGpu(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    voice: []const u8,
    speaker_path: []const u8,
    speaker_text_path: ?[]const u8,
    log: ?*std.Io.Writer,
) !SynthResult {
    if (qwen_server.synthesizeViaServer(alloc, io, text, voice, speaker_path, speaker_text_path, true, log)) |data| {
        return .{ .samples = data.samples, .sample_rate = data.sample_rate };
    } else |err| {
        if (log) |l| l.print("[Qwen] sidecar unavailable ({t}), using one-shot subprocess\n", .{err}) catch {};
        return spawn(alloc, io, text, speaker_path, 99, true, log);
    }
}
