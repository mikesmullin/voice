//! Minimal 16-bit PCM WAV encoder/writer, plus a small reader (8/16-bit
//! PCM, mono or stereo - downmixed to mono) used for stinger/background
//! audio files (tmp/FUN_PLAN.md section 2).

const std = @import("std");

pub const WavData = struct {
    samples: []f32,
    sample_rate: u32,
};

pub const WavError = error{ NotRiffWave, NotPcm, UnsupportedBitDepth, MissingDataChunk };

/// Reads a WAV file into mono `f32` samples in [-1, 1] (stereo files are
/// downmixed by averaging channels). Only PCM 8-bit (unsigned) or 16-bit
/// (signed) is supported - plenty for the sfx/ambience files this project
/// actually uses.
pub fn readMono(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !WavData {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(64 << 20));
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) {
        return WavError.NotRiffWave;
    }

    var pos: usize = 12;
    var sample_rate: u32 = 0;
    var channels: u16 = 1;
    var bits_per_sample: u16 = 16;
    var audio_format: u16 = 1;
    var data: []const u8 = &.{};

    while (pos + 8 <= bytes.len) {
        const chunk_id = bytes[pos .. pos + 4];
        const chunk_size = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        const chunk_start = pos + 8;
        if (chunk_start + chunk_size > bytes.len) break;

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            audio_format = std.mem.readInt(u16, bytes[chunk_start..][0..2], .little);
            channels = std.mem.readInt(u16, bytes[chunk_start + 2 ..][0..2], .little);
            sample_rate = std.mem.readInt(u32, bytes[chunk_start + 4 ..][0..4], .little);
            bits_per_sample = std.mem.readInt(u16, bytes[chunk_start + 14 ..][0..2], .little);
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data = bytes[chunk_start .. chunk_start + chunk_size];
        }

        // Chunks are word-aligned (padded to an even size).
        pos = chunk_start + chunk_size + (chunk_size & 1);
    }

    if (audio_format != 1) return WavError.NotPcm;
    if (data.len == 0) return WavError.MissingDataChunk;
    if (bits_per_sample != 8 and bits_per_sample != 16) return WavError.UnsupportedBitDepth;
    if (channels == 0) channels = 1;

    const bytes_per_sample: usize = bits_per_sample / 8;
    const frame_size = bytes_per_sample * channels;
    const num_frames = data.len / frame_size;

    const out = try alloc.alloc(f32, num_frames);
    for (0..num_frames) |i| {
        var sum: f32 = 0;
        for (0..channels) |c| {
            const off = i * frame_size + c * bytes_per_sample;
            const s: f32 = if (bits_per_sample == 8)
                (@as(f32, @floatFromInt(data[off])) - 128.0) / 128.0
            else
                @as(f32, @floatFromInt(std.mem.readInt(i16, data[off..][0..2], .little))) / 32768.0;
            sum += s;
        }
        out[i] = sum / @as(f32, @floatFromInt(channels));
    }

    return .{ .samples = out, .sample_rate = sample_rate };
}

pub fn writeMono16(io: std.Io, path: []const u8, sample_rate: u32, samples: []const f32) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buf: [1 << 16]u8 = undefined;
    var writer = file.writer(io, &buf);
    const w = &writer.interface;

    try writeHeaderAndSamples(w, sample_rate, samples);
    try w.flush();
}

/// Encodes to an in-memory buffer (e.g. for an HTTP response body), rather
/// than a file - same format as `writeMono16`.
pub fn encodeMono16(alloc: std.mem.Allocator, sample_rate: u32, samples: []const f32) ![]u8 {
    const total_size = 44 + samples.len * 2;
    const buf = try alloc.alloc(u8, total_size);
    var w: std.Io.Writer = .fixed(buf);
    try writeHeaderAndSamples(&w, sample_rate, samples);
    return buf;
}

fn writeHeaderAndSamples(w: *std.Io.Writer, sample_rate: u32, samples: []const f32) !void {
    const bytes_per_sample: u32 = 2;
    const data_size: u32 = @intCast(samples.len * bytes_per_sample);
    const byte_rate: u32 = sample_rate * bytes_per_sample;

    try w.writeAll("RIFF");
    try w.writeInt(u32, 36 + data_size, .little);
    try w.writeAll("WAVE");
    try w.writeAll("fmt ");
    try w.writeInt(u32, 16, .little); // fmt chunk size
    try w.writeInt(u16, 1, .little); // PCM
    try w.writeInt(u16, 1, .little); // mono
    try w.writeInt(u32, sample_rate, .little);
    try w.writeInt(u32, byte_rate, .little);
    try w.writeInt(u16, @intCast(bytes_per_sample), .little); // block align
    try w.writeInt(u16, 16, .little); // bits per sample
    try w.writeAll("data");
    try w.writeInt(u32, data_size, .little);

    for (samples) |s| {
        const clamped = std.math.clamp(s, -1.0, 1.0);
        const i: i16 = @intFromFloat(clamped * 32767.0);
        try w.writeInt(i16, i, .little);
    }
}
