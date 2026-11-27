//! Qwen engine: drives the system-installed llama.cpp `llama-tts` binary
//! (Alibaba's Qwen3-TTS-12Hz-1.7B-Base, zero-shot voice cloning) via a
//! subprocess, then reads the produced WAV back into f32 samples.
//!
//! Two backends share this engine's contract (f32 samples + rate back into
//! the normal output stack):
//!   - CPU (`synthesize`): `llama-tts -ngl 0` (Q4_K_M backbone + Q8_0 mmproj)
//!   - GPU (`synthesizeGpu`): `llama-tts -ngl 99` (full CUDA offload)
//! Both read the same speaker WAV and model files, so one speaker reference
//! serves both modes.
//!
//! Why a subprocess instead of a native Zig engine: Qwen3-TTS is a small
//! but intricate model (AR talker + 4-layer codec, 12Hz token rate) and the
//! upstream llama.cpp implementation already owns the whole
//! text->audio pipeline (tokenization, prompt construction, sampling,
//! vocoder). Reimplementing that in Zig would duplicate the runtime for no
//! gain. We just bounce the resulting WAV through our own wav/output stack
//! (gain, effects, speaker routing, the daemon's audio pipeline all stay
//! working unchanged).
//!
//! Weights (~1.5GB) live outside this repo at /workspace/models/qwen3-tts/
//! (fetch via scripts/setup-qwen.sh). Like audio8, this engine is NOT
//! preloadable: every request pays a fresh subprocess spawn + model load.
//!
//! Reuses `wav.readMono` for the WAV parse - llama-tts writes PCM 16-bit
//! little-endian mono at 24kHz.

const std = @import("std");
const wav = @import("../audio/wav.zig");
const paths = @import("../paths.zig");

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

fn spawn(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    speaker_path: []const u8,
    gpu_layers: u32,
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

    const argv = [_][]const u8{
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

    const result = std.process.run(alloc, io, .{ .argv = &argv }) catch |err| {
        switch (err) {
            error.FileNotFound, error.AccessDenied => return error.RuntimeNotInstalled,
            else => return err,
        }
    };

    if (!result.term.success()) {
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

/// Synthesizes `text` with the given speaker reference WAV (CPU backend).
/// The result is allocated from `alloc` and lives as long as that allocator
/// does - matching the other engines' contract.
pub fn synthesize(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    speaker_path: []const u8,
    log: ?*std.Io.Writer,
) !SynthResult {
    return spawn(alloc, io, text, speaker_path, 0, log);
}

/// GPU twin: same contract (text in, samples + rate out), via full CUDA
/// offload (`-ngl 99`). Same speaker WAV and model files as the CPU path.
pub fn synthesizeGpu(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    speaker_path: []const u8,
    log: ?*std.Io.Writer,
) !SynthResult {
    return spawn(alloc, io, text, speaker_path, 99, log);
}
