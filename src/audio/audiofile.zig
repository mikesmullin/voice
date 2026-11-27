//! Format-agnostic reader for voice/background audio files: PCM WAV
//! (src/audio/wav.zig, .wav) and Ogg/Opus (src/audio/opus.zig, .ogg/.opus/.oga).
//! Routes by file extension so callers can say "load this music track"
//! without caring which container it came in.
//!
//! Background-music playback (src/audio/output.zig's `playChain`) always
//! goes through here; WAV-only engine/encode paths (src/main.zig's `-o`,
//! src/http.zig's WAV download) deliberately keep using wav.zig directly.

const std = @import("std");
const wav = @import("wav.zig");
const opus = @import("opus.zig");

pub const Decoded = wav.WavData;

pub const Header = struct {
    sample_rate: u32 = 48000,
    total_frames: u64 = 0,
    channels: u16 = 1,
};

fn isOgg(path: []const u8) bool {
    const exts = [_][]const u8{ ".ogg", ".opus", ".oga" };
    for (exts) |e| {
        if (path.len >= e.len and std.ascii.eqlIgnoreCase(path[path.len - e.len ..], e)) return true;
    }
    return false;
}

/// Header-only view: sample rate, total frames (0 = unknown), channels.
pub fn readHeader(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Header {
    if (isOgg(path)) {
        const h = try opus.readHeader(alloc, io, path);
        return .{ .sample_rate = h.sample_rate, .total_frames = h.total_frames, .channels = h.channels };
    }
    const h = try wav.readHeader(alloc, io, path);
    return .{ .sample_rate = h.sample_rate, .total_frames = h.total_frames, .channels = h.channels };
}

/// Reads exactly frame_count frames starting at start_frame (wrapping past
/// EOF), downmixed to mono. Never loads the whole file (multi-hour tracks
/// stay cheap).
pub fn readSlice(alloc: std.mem.Allocator, io: std.Io, path: []const u8, start_frame: usize, frame_count: usize) !Decoded {
    if (isOgg(path)) return opus.readSlice(alloc, io, path, start_frame, frame_count);
    return wav.readSlice(alloc, io, path, start_frame, frame_count);
}

/// Reads the whole file into mono. Practical for stinger/ambience-sized
/// files; do not use for multi-hour music (use readHeader + readSlice).
pub fn readMono(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Decoded {
    if (isOgg(path)) return opus.readMono(alloc, io, path);
    return wav.readMono(alloc, io, path);
}
