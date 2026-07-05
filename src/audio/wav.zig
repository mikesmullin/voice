//! Minimal 16-bit PCM WAV encoder/writer.

const std = @import("std");

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
