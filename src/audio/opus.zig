//! Ogg/Opus reader via libopusfile (system library) - lets the voice server
//! play Ogg/Opus music tracks (the background-music path in
//! src/audio/output.zig's `playChain`) directly during runtime playback,
//! with no WAV conversion step. All output is mono f32 at 48 kHz (libopusfile's
//! fixed decode rate); playback resamples from there (see
//! src/audio/resample.zig).
//!
//! Hand-declared externs (no translate-c) - only the handful of libopusfile
//! entry points used here, so there's no translated C module to maintain.
//! libopusfile is always the system library (libopusfile.so.0); build.zig
//! links it.

const std = @import("std");
const wav = @import("wav.zig");

const c = struct {
    const OggOpusFile = opaque {};

    extern "c" fn op_open_file(path: [*:0]const u8, error_out: ?*c_int) ?*OggOpusFile;
    extern "c" fn op_free(of: ?*OggOpusFile) void;
    extern "c" fn op_pcm_total(of: ?*const OggOpusFile, li: c_int) c_longlong;
    extern "c" fn op_channel_count(of: ?*const OggOpusFile, li: c_int) c_int;
    extern "c" fn op_pcm_seek(of: ?*OggOpusFile, pcm_offset: c_longlong) c_int;
    extern "c" fn op_read_float(of: ?*OggOpusFile, pcm: [*]f32, buf_size: c_int, li: ?*c_int) c_int;
};

pub const Header = struct {
    /// libopusfile always decodes at 48 kHz.
    sample_rate: u32 = 48000,
    channels: u16 = 1,
    /// Total frames at 48 kHz (0 when the total cannot be determined).
    total_frames: u64 = 0,
};

pub const Decoded = wav.WavData;

pub const OpusError = error{
    OpenFailed,
    NoAudio,
    SeekFailed,
    ReadFailed,
};

/// Opens the file. `path_z` is freed as soon as the library has the handle;
/// libopusfile keeps its own I/O descriptor.
fn open(alloc: std.mem.Allocator, path: []const u8) !?*c.OggOpusFile {
    const path_z = try alloc.dupeSentinel(u8, path, 0);
    defer alloc.free(path_z);
    var err: c_int = 0;
    const of = c.op_open_file(path_z.ptr, &err) orelse return OpusError.OpenFailed;
    return of;
}

/// Header-only view of an Ogg/Opus file: channels and total frame count
/// (48 kHz). Never loads the payload - a multi-hour track stays cheap to
/// inspect.
pub fn readHeader(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Header {
    _ = io;
    const of = try open(alloc, path);
    defer c.op_free(of);
    const total = c.op_pcm_total(of, -1);
    const channels = c.op_channel_count(of, -1);
    if (channels <= 0) return OpusError.NoAudio;
    return .{
        .sample_rate = 48000,
        .channels = @intCast(channels),
        .total_frames = if (total > 0) @intCast(total) else 0,
    };
}

/// Reads `frame_count` frames starting at `start_frame`, downmixed to mono.
/// Wraps around the end of the file if the window runs past EOF (same
/// behavior as the WAV slice reader) - the window always fills, so the
/// music is tiled exactly to the voice's length downstream. Reads only
/// the window needed - never the whole track.
pub fn readSlice(alloc: std.mem.Allocator, io: std.Io, path: []const u8, start_frame: usize, frame_count: usize) !Decoded {
    _ = io;
    const of = try open(alloc, path);
    defer c.op_free(of);
    const channels = c.op_channel_count(of, -1);
    if (channels <= 0) return OpusError.NoAudio;
    const ch: usize = @intCast(channels);

    const total = c.op_pcm_total(of, -1);

    // Start decoding at the requested offset, wrapped into the file's
    // range. When the total is unknown, offsetting would be unsafe - play
    // from the beginning instead (callers only ever pass an offset when
    // the total is known).
    if (total > 0) {
        const offset = start_frame % @as(usize, @intCast(total));
        if (offset > 0) {
            if (c.op_pcm_seek(of, @intCast(offset)) != 0) return OpusError.SeekFailed;
        }
    }

    const out = try alloc.alloc(f32, frame_count);
    errdefer alloc.free(out);

    // One scratch buffer sized for the whole window (frames * channels).
    // Bounded - the window is at most a few seconds long.
    const scratch = try alloc.alloc(f32, frame_count * ch);
    defer alloc.free(scratch);

    var filled: usize = 0;
    while (filled < frame_count) {
        const want = frame_count - filled;
        const got = c.op_read_float(of, scratch.ptr, @intCast(want * ch), null);
        if (got <= 0) {
            // EOF - wrap to the start and keep filling (a short track may
            // be shorter than the voice, and must not break playback).
            if (c.op_pcm_seek(of, 0) != 0) return OpusError.ReadFailed;
            continue;
        }
        const g: usize = @intCast(got);
        var j: usize = 0;
        while (j < g) : (j += 1) {
            var sum: f32 = 0;
            for (0..ch) |k| sum += scratch[j * ch + k];
            out[filled + j] = sum / @as(f32, @floatFromInt(ch));
        }
        filled += g;
    }

    return .{ .samples = out, .sample_rate = 48000 };
}

/// Reads a whole Ogg/Opus file into mono f32. Only practical for short
/// files (stinger-sized ambience); background music should go through
/// `readSlice` (or `readHeader` + `readSlice`) so a multi-hour track is
/// never fully loaded.
pub fn readMono(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Decoded {
    _ = io;
    const of = try open(alloc, path);
    defer c.op_free(of);
    const channels = c.op_channel_count(of, -1);
    if (channels <= 0) return OpusError.NoAudio;
    const ch: usize = @intCast(channels);

    var out = std.ArrayList(f32).empty;
    errdefer out.deinit(alloc);

    // 100 ms chunks - bounded scratch, never the whole track up front.
    const chunk_samples = 4800;
    const scratch = try alloc.alloc(f32, chunk_samples * ch);
    defer alloc.free(scratch);

    while (true) {
        const got = c.op_read_float(of, scratch.ptr, @intCast(scratch.len), null);
        if (got <= 0) break;
        const g: usize = @intCast(got);
        for (0..g) |j| {
            var sum: f32 = 0;
            for (0..ch) |k| sum += scratch[j * ch + k];
            try out.append(alloc, sum / @as(f32, @floatFromInt(ch)));
        }
    }
    if (out.items.len == 0) return OpusError.NoAudio;
    return .{ .samples = try out.toOwnedSlice(alloc), .sample_rate = 48000 };
}
