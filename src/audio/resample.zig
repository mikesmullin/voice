//! Simple linear-interpolation sample-rate converter. Used by
//! src/audio/world.zig to normalize every Entity's buffer to the World's
//! one fixed mix_rate before it's enqueued (per tmp/FUN_PLAN.md section 2)
//! - not audiophile-grade, but plenty for speech, and Kokoro (the common
//! case, 24000Hz) already matches the default mix_rate so this is skipped
//! entirely for it; only non-24000Hz Piper voices actually get resampled.

const std = @import("std");

/// Returns a newly-allocated buffer at `to_rate`, independently owned/freed
/// from `samples` (a plain copy when `from_rate == to_rate`).
pub fn resample(alloc: std.mem.Allocator, samples: []const f32, from_rate: u32, to_rate: u32) ![]f32 {
    if (from_rate == to_rate or samples.len == 0) {
        return alloc.dupe(f32, samples);
    }
    const ratio = @as(f64, @floatFromInt(to_rate)) / @as(f64, @floatFromInt(from_rate));
    const out_len: usize = @intFromFloat(@as(f64, @floatFromInt(samples.len)) * ratio);
    const out = try alloc.alloc(f32, out_len);
    for (out, 0..) |*s, i| {
        const src_pos = @as(f64, @floatFromInt(i)) / ratio;
        const idx0: usize = @intFromFloat(@floor(src_pos));
        const frac: f32 = @floatCast(src_pos - @floor(src_pos));
        const idx0_clamped = @min(idx0, samples.len - 1);
        const idx1 = @min(idx0 + 1, samples.len - 1);
        const s0 = samples[idx0_clamped];
        const s1 = samples[idx1];
        s.* = s0 + (s1 - s0) * frac;
    }
    return out;
}
