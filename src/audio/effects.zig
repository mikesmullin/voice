//! DSP effects (tmp/FUN_PLAN.md section 2) - simple, stateless-across-
//! calls buffer transforms applied once to a synthesized voice buffer
//! before it becomes a World `Entity` (see src/audio/world.zig and
//! src/audio/output.zig's `playChain`). None of these run live per audio
//! callback; `World.mix()` only ever slices/sums already-processed
//! buffers.
//!
//! `stinger` is deliberately NOT handled here - it isn't a buffer
//! transform, it's "play this other file first, on the same channel"
//! (see `Output.playChain`), so callers should extract/skip it before
//! calling `applyChain`.

const std = @import("std");
const config_mod = @import("../config.zig");
const paths_mod = @import("../paths.zig");

/// Result of resolving one or more named effect presets (`-e`/`--effect`,
/// or the HTTP `"effects"` array) into a concrete plan: the combined,
/// concatenated DSP chain (stinger steps excluded) plus any stinger file
/// paths (resolved to absolute paths, in the order they should play),
/// plus at most one background choice (tmp/FUN_PLAN.md section 2: "when
/// you chain presets, the last preset's background wins").
pub const ResolvedChain = struct {
    chain: std.ArrayList(config_mod.EffectStep) = .empty,
    stinger_files: std.ArrayList([]const u8) = .empty,
    background: ?BackgroundChoice = null,
};

pub const BackgroundChoice = struct {
    path: []const u8,
    volume: f32,
};

pub const ResolveError = error{UnknownEffect};

/// Concatenates the chains of every named preset, in the order given
/// (per tmp/FUN_PLAN.md section 2: "chain steps run top-to-bottom within
/// each named preset" and presets themselves apply in the order listed
/// on the command line/request).
pub fn resolveEffects(alloc: std.mem.Allocator, cfg: *const config_mod.Config, effect_names: []const []const u8) !ResolvedChain {
    var result: ResolvedChain = .{};
    for (effect_names) |name| {
        const preset = cfg.getEffect(name) orelse return ResolveError.UnknownEffect;
        for (preset.chain.items) |step| {
            if (std.mem.eql(u8, step.kind, "stinger")) {
                if (step.params.get("file")) |file| {
                    const abs = if (file.len > 0 and file[0] == '/')
                        file
                    else
                        try std.fmt.allocPrint(alloc, "{s}/{s}", .{ paths_mod.ROOT, file });
                    try result.stinger_files.append(alloc, abs);
                }
            } else {
                try result.chain.append(alloc, step);
            }
        }
        if (preset.background.sources.items.len > 0) {
            // A fresh seed per process (ASLR-derived stack address) is
            // plenty for "pick a random ambience clip" - not meant to be
            // cryptographically random, and each CLI invocation is its
            // own process anyway.
            var seed: u64 = undefined;
            seed = @intCast(@intFromPtr(&seed));
            var prng = std.Random.DefaultPrng.init(seed);
            const idx = prng.random().intRangeAtMost(usize, 0, preset.background.sources.items.len - 1);
            const file = preset.background.sources.items[idx];
            const abs = if (file.len > 0 and file[0] == '/')
                file
            else
                try std.fmt.allocPrint(alloc, "{s}/{s}", .{ paths_mod.ROOT, file });
            result.background = .{ .path = abs, .volume = preset.background.volume };
        }
    }
    return result;
}

fn getFloat(params: std.StringHashMap([]const u8), key: []const u8, default: f32) f32 {
    if (params.get(key)) |v| return std.fmt.parseFloat(f32, v) catch default;
    return default;
}

/// Applies every non-`stinger` step in `chain`, in order, to `samples`.
/// May return a longer buffer than it was given (`delay`/`reverb` extend
/// the tail) - always returns a buffer allocated from `alloc` (a fresh
/// copy even if the chain is empty or every step is in-place), so
/// callers can free/ignore the input independently.
pub fn applyChain(alloc: std.mem.Allocator, samples_in: []const f32, sample_rate: u32, chain: []const config_mod.EffectStep) ![]f32 {
    var samples = try alloc.dupe(f32, samples_in);
    for (chain) |step| {
        if (std.mem.eql(u8, step.kind, "stinger")) continue;
        samples = try applyStep(alloc, samples, sample_rate, step);
    }
    return samples;
}

fn applyStep(alloc: std.mem.Allocator, samples: []f32, sample_rate: u32, step: config_mod.EffectStep) ![]f32 {
    const p = step.params;
    if (std.mem.eql(u8, step.kind, "gain")) {
        applyGain(samples, getFloat(p, "amount", 1.0));
    } else if (std.mem.eql(u8, step.kind, "distortion")) {
        applyDistortion(samples, getFloat(p, "drive", 2.0));
    } else if (std.mem.eql(u8, step.kind, "lowpass")) {
        applyBiquad(samples, sample_rate, .lowpass, getFloat(p, "cutoff", 4000), getFloat(p, "q", 0.707));
    } else if (std.mem.eql(u8, step.kind, "highpass")) {
        applyBiquad(samples, sample_rate, .highpass, getFloat(p, "cutoff", 300), getFloat(p, "q", 0.707));
    } else if (std.mem.eql(u8, step.kind, "bandpass")) {
        const low = getFloat(p, "low", 300);
        const high = getFloat(p, "high", 3000);
        const center = @sqrt(@max(low * high, 1.0));
        const q = center / @max(high - low, 1.0);
        applyBiquad(samples, sample_rate, .bandpass, center, q);
    } else if (std.mem.eql(u8, step.kind, "delay") or std.mem.eql(u8, step.kind, "echo")) {
        return applyDelay(alloc, samples, sample_rate, getFloat(p, "time_ms", 300), getFloat(p, "feedback", 0.35), getFloat(p, "wet", 0.3));
    } else if (std.mem.eql(u8, step.kind, "reverb")) {
        return applyReverb(alloc, samples, sample_rate, getFloat(p, "decay", 1.0), getFloat(p, "wet", 0.3));
    } else if (std.mem.eql(u8, step.kind, "chorus")) {
        try applyModDelay(alloc, samples, sample_rate, getFloat(p, "depth_ms", 8.0), getFloat(p, "rate_hz", 0.8), getFloat(p, "feedback", 0.1), getFloat(p, "wet", 0.5));
    } else if (std.mem.eql(u8, step.kind, "flanger")) {
        try applyModDelay(alloc, samples, sample_rate, getFloat(p, "depth_ms", 3.0), getFloat(p, "rate_hz", 0.25), getFloat(p, "feedback", 0.5), getFloat(p, "wet", 0.5));
    } else if (std.mem.eql(u8, step.kind, "phaser")) {
        applyPhaser(samples, sample_rate, getFloat(p, "rate_hz", 0.5), getFloat(p, "depth", 0.7), @intFromFloat(getFloat(p, "stages", 4)), getFloat(p, "feedback", 0.3));
    } else if (std.mem.eql(u8, step.kind, "compressor")) {
        applyCompressor(samples, sample_rate, getFloat(p, "threshold", -18.0), getFloat(p, "ratio", 3.0), getFloat(p, "attack_ms", 10.0), getFloat(p, "release_ms", 100.0));
    }
    // Unknown step kinds are ignored, not fatal - a typo'd effect name
    // shouldn't break synthesis, just silently not apply.
    return samples;
}

pub fn applyGain(samples: []f32, amount: f32) void {
    if (amount == 1.0) return;
    for (samples) |*s| s.* = std.math.clamp(s.* * amount, -1.0, 1.0);
}

pub fn applyDistortion(samples: []f32, drive: f32) void {
    const d = @max(drive, 0.0001);
    const norm = std.math.tanh(d);
    for (samples) |*s| s.* = std.math.tanh(s.* * d) / norm;
}

const BiquadKind = enum { lowpass, highpass, bandpass };

/// RBJ ("cookbook") biquad filter, applied in one pass.
pub fn applyBiquad(samples: []f32, sample_rate: u32, kind: BiquadKind, freq: f32, q: f32) void {
    const clamped_freq = std.math.clamp(freq, 20.0, @as(f32, @floatFromInt(sample_rate)) / 2.0 - 20.0);
    const w0 = 2.0 * std.math.pi * clamped_freq / @as(f32, @floatFromInt(sample_rate));
    const cos_w0 = @cos(w0);
    const sin_w0 = @sin(w0);
    const alpha = sin_w0 / (2.0 * @max(q, 0.05));

    var b0: f32 = 0;
    var b1: f32 = 0;
    var b2: f32 = 0;
    const a0 = 1 + alpha;
    const a1 = -2 * cos_w0;
    const a2 = 1 - alpha;

    switch (kind) {
        .lowpass => {
            b0 = (1 - cos_w0) / 2;
            b1 = 1 - cos_w0;
            b2 = (1 - cos_w0) / 2;
        },
        .highpass => {
            b0 = (1 + cos_w0) / 2;
            b1 = -(1 + cos_w0);
            b2 = (1 + cos_w0) / 2;
        },
        .bandpass => {
            b0 = alpha;
            b1 = 0;
            b2 = -alpha;
        },
    }
    const nb0 = b0 / a0;
    const nb1 = b1 / a0;
    const nb2 = b2 / a0;
    const na1 = a1 / a0;
    const na2 = a2 / a0;

    var x1: f32 = 0;
    var x2: f32 = 0;
    var y1: f32 = 0;
    var y2: f32 = 0;
    for (samples) |*s| {
        const x0 = s.*;
        const y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        s.* = y0;
    }
}

/// Extends the buffer with `repeats` decaying, delayed copies of the dry
/// signal - an approximation of a recursive feedback delay, bounded so
/// it can't grow unboundedly even at high `feedback`.
pub fn applyDelay(alloc: std.mem.Allocator, samples: []const f32, sample_rate: u32, time_ms: f32, feedback: f32, wet: f32) ![]f32 {
    const delay_samples: usize = @intFromFloat(@max(time_ms, 1.0) / 1000.0 * @as(f32, @floatFromInt(sample_rate)));
    if (delay_samples == 0) return alloc.dupe(f32, samples);
    const fb = std.math.clamp(feedback, 0.0, 0.95);
    const repeats: usize = 4;
    const out_len = samples.len + delay_samples * repeats;
    const out = try alloc.alloc(f32, out_len);
    @memset(out, 0);
    @memcpy(out[0..samples.len], samples);

    var gain = std.math.clamp(wet, 0.0, 1.0);
    var offset = delay_samples;
    var rep: usize = 0;
    while (rep < repeats) : (rep += 1) {
        for (samples, 0..) |s, i| {
            if (offset + i >= out_len) break;
            out[offset + i] += s * gain;
        }
        gain *= fb;
        offset += delay_samples;
    }
    for (out) |*s| s.* = std.math.clamp(s.*, -1.0, 1.0);
    return out;
}

/// Simplified Schroeder-style reverb (a handful of parallel comb filters,
/// classic tunings scaled to `sample_rate`) - not the full multi-comb +
/// allpass Freeverb algorithm, but a real, audible algorithmic reverb
/// tail rather than a stub.
pub fn applyReverb(alloc: std.mem.Allocator, samples: []const f32, sample_rate: u32, decay: f32, wet: f32) ![]f32 {
    const clamped_decay = std.math.clamp(decay, 0.05, 4.0);
    const tail_ms: f32 = 350.0 * clamped_decay;
    const tail_samples: usize = @intFromFloat(tail_ms / 1000.0 * @as(f32, @floatFromInt(sample_rate)));
    const out_len = samples.len + tail_samples;

    const out = try alloc.alloc(f32, out_len);
    const wet_buf = try alloc.alloc(f32, out_len);
    defer alloc.free(wet_buf);
    @memset(wet_buf, 0);

    const comb_ms = [_]f32{ 29.7, 37.1, 41.1, 43.7 };
    const fb = std.math.clamp(0.6 + 0.35 * clamped_decay, 0.0, 0.98);
    for (comb_ms) |ms| {
        const d: usize = @intFromFloat(ms / 1000.0 * @as(f32, @floatFromInt(sample_rate)));
        if (d == 0) continue;
        const buf = try alloc.alloc(f32, d);
        defer alloc.free(buf);
        @memset(buf, 0);
        var idx: usize = 0;
        for (0..out_len) |i| {
            const dry: f32 = if (i < samples.len) samples[i] else 0;
            const delayed = buf[idx];
            wet_buf[i] += delayed;
            buf[idx] = dry + delayed * fb;
            idx = (idx + 1) % d;
        }
    }

    const wet_amount = std.math.clamp(wet, 0.0, 1.0);
    const wet_scale = 1.0 / @as(f32, @floatFromInt(comb_ms.len));
    for (out, 0..) |*o, i| {
        const dry: f32 = if (i < samples.len) samples[i] else 0;
        const wet_s = wet_buf[i] * wet_scale;
        o.* = std.math.clamp(dry * (1 - wet_amount) + wet_s * wet_amount, -1.0, 1.0);
    }
    return out;
}

/// Shared LFO-modulated delay line - chorus (long-ish delay, low/no
/// feedback) and flanger (short delay, higher feedback) are the same
/// mechanism with different default depth/rate/feedback. In-place
/// (doesn't change buffer length).
fn applyModDelay(alloc: std.mem.Allocator, samples: []f32, sample_rate: u32, depth_ms: f32, rate_hz: f32, feedback: f32, wet: f32) !void {
    const sr_f: f32 = @floatFromInt(sample_rate);
    const max_delay_samples: usize = @intFromFloat(@ceil((depth_ms + 1.0) / 1000.0 * sr_f));
    if (max_delay_samples < 2) return;
    const buf = try alloc.alloc(f32, max_delay_samples + 2);
    defer alloc.free(buf);
    @memset(buf, 0);

    const fb = std.math.clamp(feedback, -0.95, 0.95);
    const wet_amount = std.math.clamp(wet, 0.0, 1.0);
    var widx: usize = 0;
    for (samples, 0..) |*s, i| {
        const lfo = (@sin(2.0 * std.math.pi * rate_hz * @as(f32, @floatFromInt(i)) / sr_f) + 1.0) / 2.0;
        const delay_f = lfo * depth_ms / 1000.0 * sr_f;
        const delay_floor: usize = @intFromFloat(@floor(delay_f));
        const frac = delay_f - @floor(delay_f);
        const read_idx0 = (widx + buf.len - @min(delay_floor, buf.len - 1)) % buf.len;
        const read_idx1 = (read_idx0 + buf.len - 1) % buf.len;
        const delayed = buf[read_idx0] * (1 - frac) + buf[read_idx1] * frac;
        const dry = s.*;
        buf[widx] = dry + delayed * fb;
        s.* = std.math.clamp(dry * (1 - wet_amount) + delayed * wet_amount, -1.0, 1.0);
        widx = (widx + 1) % buf.len;
    }
}

/// N-stage first-order allpass phaser with an LFO-modulated coefficient.
/// In-place (doesn't change buffer length).
pub fn applyPhaser(samples: []f32, sample_rate: u32, rate_hz: f32, depth: f32, stages: usize, feedback: f32) void {
    var ap_state: [8]f32 = @splat(0);
    const n = @min(stages, ap_state.len);
    if (n == 0) return;
    const sr_f: f32 = @floatFromInt(sample_rate);
    const fb = std.math.clamp(feedback, -0.95, 0.95);
    const clamped_depth = std.math.clamp(depth, 0.0, 0.9);
    var fb_sample: f32 = 0;
    for (samples, 0..) |*s, i| {
        const lfo = (@sin(2.0 * std.math.pi * rate_hz * @as(f32, @floatFromInt(i)) / sr_f) + 1.0) / 2.0;
        const a = std.math.clamp(0.05 + clamped_depth * lfo, -0.99, 0.99);
        var x = s.* + fb_sample * fb;
        for (0..n) |j| {
            const y = -a * x + ap_state[j];
            ap_state[j] = x + a * y;
            x = y;
        }
        fb_sample = x;
        s.* = std.math.clamp(s.* * 0.5 + x * 0.5, -1.0, 1.0);
    }
}

/// Single-band compressor (envelope follower + gain computer). In-place.
pub fn applyCompressor(samples: []f32, sample_rate: u32, threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) void {
    const sr_f: f32 = @floatFromInt(sample_rate);
    const attack_coef = @exp(-1.0 / (@max(attack_ms, 0.1) / 1000.0 * sr_f));
    const release_coef = @exp(-1.0 / (@max(release_ms, 0.1) / 1000.0 * sr_f));
    const threshold = std.math.pow(f32, 10.0, threshold_db / 20.0);
    const clamped_ratio = @max(ratio, 1.0);
    var envelope: f32 = 0;
    for (samples) |*s| {
        const abs_s = @abs(s.*);
        const coef = if (abs_s > envelope) attack_coef else release_coef;
        envelope = coef * envelope + (1 - coef) * abs_s;
        var gain: f32 = 1.0;
        if (envelope > threshold and envelope > 0) {
            const excess_db = 20.0 * std.math.log10(envelope / threshold);
            const reduced_db = excess_db * (1.0 - 1.0 / clamped_ratio);
            gain = std.math.pow(f32, 10.0, -reduced_db / 20.0);
        }
        s.* = std.math.clamp(s.* * gain, -1.0, 1.0);
    }
}
