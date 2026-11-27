//! Audio8 engine: drives the official `arktts_runtime` Python package
//! (vendored under vendor/audio8-runtime/, see scripts/setup-audio8.sh)
//! via a subprocess, then reads the produced WAV back into f32 samples.
//!
//! Two backends share this engine's contract (f32 samples + rate back into
//! the normal output stack):
//!   - CPU (`synthesize`): official INT8 ONNX runtime, CPU-only by design.
//!   - GPU (`synthesizeGpu`): full-precision PyTorch checkpoint on CUDA
//!     (scripts/audio8_gpu_synth.py). Reads the SAME registered voice
//!     (codes.npy + reference_text from voices/<name>/) as the CPU path,
//!     so one registration serves both.
//!
//! Why a subprocess instead of a native Zig engine: the 0.1B export is a
//! Falcon-H1 hybrid (per-token slow AR with 24 layers of recurrent state,
//! a 4-layer fast AR, an FP16 codec decoder) - reimplementing that in Zig
//! would duplicate the whole runtime with high risk of subtle numeric
//! drift. The upstream CLI owns the whole text->audio pipeline (tokenization,
//! prompt construction, sampling, codec decode), then we just bounce the
//! resulting WAV through our own wav/output stack (gain, effects, speaker
//! routing, the daemon's audio pipeline all stay working unchanged).
//!
//! Weights (~820MB) live outside this repo at /workspace/models/audio8-tts/
//! (fetch via scripts/setup-audio8.sh). Inference is CPU-only by design
//! (the export is ONNX INT8 per-token graphs with an FP16 codec; there is
//! no GPU path in this export).
//!
//! Reuses `wav.readMono` for the WAV parse (PCM 8/16-bit, mono) - the
//! Python CLI writes PCM_16 mono at 44.1kHz.
//!
//! NOTE: there is intentionally no persistent in-process engine state -
//! every request pays a fresh subprocess spawn + session load (same cost
//! class as `voice local` kokoro/piper). A daemon-cached Audio8 session
//! would mean hosting the runtime in-process or via its own HTTP server -
//! not done yet (see tmp/PHASE3_PLAN.md's "Still open" section for that
//! pattern's siblings).

const std = @import("std");
const wav = @import("../audio/wav.zig");
const paths = @import("../paths.zig");

pub const SynthResult = struct { samples: []f32, sample_rate: u32 };

/// Default generation settings, matching the upstream CLI (which is what
/// the docs/demos in the runtime were built against): 256 new frames is
/// ~12s of audio at 44.1kHz, plenty for a short utterance; seed=42 keeps
/// runs reproducible (the runtime seeds numpy with it).
pub const MAX_NEW_TOKENS: u32 = 256;
pub const SEED: u32 = 42;

pub const Audio8Error = error{
    /// The Python venv interpreter is missing (not set up yet).
    RuntimeNotInstalled,
    /// A registered voice dir is missing codes.npy/meta.json, or its
    /// meta.json has no reference_text.
    VoiceNotRegistered,
    /// The subprocess exited non-zero (stderr included in the log line by the caller).
    SubprocessFailed,
    /// The produced WAV was missing/unreadable (e.g. the runtime crashed after writing it).
    WAVReadFailed,
} || wav.WavError || std.process.RunError || std.mem.Allocator.Error;

/// Synthesizes `text` (already LLM-rewritten etc. - same contract as the
/// other engines) with the named registered voice (e.g. "default"). The
/// result is allocated from `alloc` and lives as long as that allocator
/// does - matching the other engines' contract.
pub fn synthesize(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    voice_name: []const u8,
    log: ?*std.Io.Writer,
) !SynthResult {
    var out_path_buf: [256]u8 = undefined;
    const out_path = try std.fmt.bufPrint(
        &out_path_buf,
        "{s}{d}.wav",
        .{ paths.AUDIO8_TMP_WAV, std.c.getpid() },
    );
    var tokens_buf: [32]u8 = undefined;
    var seed_buf: [32]u8 = undefined;
    const tokens = try std.fmt.bufPrint(&tokens_buf, "{d}", .{MAX_NEW_TOKENS});
    const seed = try std.fmt.bufPrint(&seed_buf, "{d}", .{SEED});

    // Ensure a clean slate: a crashed prior run may have left a stale WAV
    // behind; the subprocess rewrites it, but never read a stale file.
    std.Io.Dir.deleteFileAbsolute(io, out_path) catch {};

    const argv = [_][]const u8{
        paths.AUDIO8_VENV_PYTHON,
        "-m",
        "arktts_runtime.cli",
        "--model-dir",
        paths.AUDIO8_MODEL_DIR,
        "--voices-dir",
        paths.AUDIO8_VOICES_DIR,
        "--text",
        text,
        "--voice",
        voice_name,
        "--output",
        out_path,
        "--max-new-tokens",
        tokens,
        "--seed",
        seed,
    };

    const result = std.process.run(alloc, io, .{
        .argv = &argv,
        // Run from the runtime dir (not the caller's cwd) so the `arktts_runtime`
        // package imports when we pass `-m arktts_runtime.cli`.
        .cwd = .{ .path = paths.AUDIO8_RUNTIME_DIR },
    }) catch |err| {
        switch (err) {
            error.FileNotFound, error.AccessDenied => return error.RuntimeNotInstalled,
            else => return err,
        }
    };

    if (!result.term.success()) {
        // Surface the runtime's own error (it's usually a Python traceback or
        // a clear "voice not found" style message) so the caller can diagnose.
        if (log) |l| {
            l.print("[Audio8] subprocess failed (exited nonzero): {s}\n", .{result.stderr}) catch {};
        }
        return error.SubprocessFailed;
    }
    if (log) |l| {
        if (result.stderr.len > 0) l.print("[Audio8] subprocess stderr: {s}\n", .{result.stderr}) catch {};
    }

    const data = wav.readMono(alloc, io, out_path) catch {
        return error.WAVReadFailed;
    };

    // Best-effort cleanup of the temp file.
    std.Io.Dir.deleteFileAbsolute(io, out_path) catch {};

    return .{ .samples = data.samples, .sample_rate = data.sample_rate };
}

/// Reads a registered voice's reference material (codes path + transcript)
/// out of the shared voices dir. Both backends consume this same pair, so
/// "keep the first clone" means exactly this: one registration, two engines.
pub const VoiceRef = struct { codes_path: []u8, reference_text: []u8 };

pub fn readVoiceRef(alloc: std.mem.Allocator, io: std.Io, voice_name: []const u8) !VoiceRef {
    if (voice_name.len == 0 or std.mem.indexOfScalar(u8, voice_name, '/') != null or std.mem.eql(u8, voice_name, ".") or std.mem.eql(u8, voice_name, "..")) {
        return error.VoiceNotRegistered;
    }
    const dir_path = try std.fmt.allocPrint(alloc, "{s}/{s}", .{ paths.AUDIO8_VOICES_DIR, voice_name });
    const codes_path = try std.fmt.allocPrint(alloc, "{s}/codes.npy", .{dir_path});
    const meta_path = try std.fmt.allocPrint(alloc, "{s}/meta.json", .{dir_path});

    const meta_bytes = std.Io.Dir.readFileAlloc(.cwd(), io, meta_path, alloc, .limited(1 << 20)) catch {
        return error.VoiceNotRegistered;
    };
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, alloc, meta_bytes, .{}) catch {
        return error.VoiceNotRegistered;
    };
    const ref_text = parsed.object.get("reference_text") orelse return error.VoiceNotRegistered;
    if (ref_text != .string or ref_text.string.len == 0) return error.VoiceNotRegistered;

    // codes.npy must exist too (the GPU script would fail on it otherwise).
    const codes_file = std.Io.Dir.cwd().openFile(io, codes_path, .{}) catch {
        return error.VoiceNotRegistered;
    };
    codes_file.close(io);
    return .{
        .codes_path = codes_path,
        .reference_text = try alloc.dupe(u8, ref_text.string),
    };
}

/// GPU twin of `synthesize`: same contract (text in, samples + rate out),
/// via the full-precision PyTorch checkpoint on CUDA. `voice_name` is a
/// registered voice (e.g. "default", "Ruth") - its codes + transcript come
/// from the shared voices dir, identical to what the CPU backend uses.
pub fn synthesizeGpu(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    voice_name: []const u8,
    log: ?*std.Io.Writer,
) !SynthResult {
    const ref = try readVoiceRef(alloc, io, voice_name);

    var out_path_buf: [256]u8 = undefined;
    const out_path = try std.fmt.bufPrint(
        &out_path_buf,
        "{s}gpu-{d}.wav",
        .{ paths.AUDIO8_TMP_WAV, std.c.getpid() },
    );
    var tokens_buf: [32]u8 = undefined;
    var seed_buf: [32]u8 = undefined;
    const tokens = try std.fmt.bufPrint(&tokens_buf, "{d}", .{MAX_NEW_TOKENS});
    const seed = try std.fmt.bufPrint(&seed_buf, "{d}", .{SEED});

    // See `synthesize`'s comment: never read a stale WAV from a crashed run.
    std.Io.Dir.deleteFileAbsolute(io, out_path) catch {};

    const argv = [_][]const u8{
        paths.AUDIO8_GPU_VENV_PYTHON,
        paths.AUDIO8_GPU_SYNTH_SCRIPT,
        "--model-dir",
        paths.AUDIO8_PT_MODEL_DIR,
        "--codes",
        ref.codes_path,
        "--ref-text",
        ref.reference_text,
        "--text",
        text,
        "--output",
        out_path,
        "--max-new-tokens",
        tokens,
        "--seed",
        seed,
    };

    // No cwd override needed here (unlike the CPU path): the script takes
    // absolute paths and is not part of an importable package.
    const result = std.process.run(alloc, io, .{ .argv = &argv }) catch |err| {
        switch (err) {
            error.FileNotFound, error.AccessDenied => return error.RuntimeNotInstalled,
            else => return err,
        }
    };

    if (!result.term.success()) {
        if (log) |l| {
            l.print("[Audio8GPU] subprocess failed (exited nonzero): {s}\n", .{result.stderr}) catch {};
        }
        return error.SubprocessFailed;
    }
    if (log) |l| {
        if (result.stderr.len > 0) l.print("[Audio8GPU] subprocess stderr: {s}\n", .{result.stderr}) catch {};
    }

    const data = wav.readMono(alloc, io, out_path) catch {
        return error.WAVReadFailed;
    };

    std.Io.Dir.deleteFileAbsolute(io, out_path) catch {};

    return .{ .samples = data.samples, .sample_rate = data.sample_rate };
}
