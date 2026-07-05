//! Minimal 16-bit PCM WAV file writer.

const std = @import("std");

pub fn writeMono16(io: std.Io, path: []const u8, sample_rate: u32, samples: []const f32) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buf: [1 << 16]u8 = undefined;
    var writer = file.writer(io, &buf);
    const w = &writer.interface;

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
    try w.flush();
}
