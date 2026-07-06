//! Persistent audio output via sokol_audio's callback model (portable:
//! Windows/WASAPI, macOS/CoreAudio, Linux/ALSA - see tmp/FUN_PLAN.md
//! section 2's "Prerequisite" subsection for why this replaced the
//! original PulseAudio pa_simple binding, which was Linux-only). Actual
//! mixing across concurrent channels happens in src/audio/world.zig's
//! World.mix(), called directly by sokol_audio's stream callback -
//! sokol_audio explicitly does not mix multiple streams for you.
//!
//! `saudio_setup()` is a process-wide C singleton (can only be called
//! once), so the `World` it calls back into must be one too - `Output`
//! is a thin handle onto a single module-level `World`, not an owner of
//! its own. (An earlier version of this file gave each `Output.init()`
//! its own `World`; since only the first call's `World` ever actually got
//! wired into sokol_audio, every subsequent `Output` - e.g. the one
//! `Daemon.init()` makes internally plus the separate one `main.zig`'s
//! "local" path makes for real playback - silently played into a `World`
//! nothing ever mixed, hanging `drain()` forever. Worth remembering if
//! this file needs touching again.)
//!
//! Keeps the same public API shape (`init`/`play`/`drain`/`deinit`) the
//! previous PulseAudio-backed version had, so daemon.zig/main.zig call
//! sites barely change - `play()` now resamples to the World's fixed
//! mix_rate and enqueues an Entity instead of blocking on a
//! `pa_simple_write()`; playback happens asynchronously from then on
//! (see `drain()` for how callers wait for it to finish - it now needs
//! an `Io` to sleep-poll with, since there's no more per-call blocking
//! syscall to just wait on).

const std = @import("std");
const sokol = @import("sokol");
const saudio = sokol.audio;
const world_mod = @import("world.zig");
const feature_tap = @import("feature_tap.zig");
const resample = @import("resample.zig").resample;
const wav_mod = @import("wav.zig");
const effects_mod = @import("effects.zig");

/// Matches Kokoro's native rate (tmp/FUN_PLAN.md section 2) - only Piper
/// voices at other rates ever actually get resampled by `play()`.
const MIX_RATE: u32 = 24000;

var g_world: world_mod.World = undefined;
var g_setup_done: bool = false;

fn streamCb(buffer: [*c]f32, num_frames: c_int, num_channels: c_int, user_data: ?*anyopaque) callconv(.c) void {
    const world: *world_mod.World = @ptrCast(@alignCast(user_data.?));
    const n: usize = @intCast(num_frames);
    const ch: usize = @intCast(num_channels);
    world.mix(buffer[0 .. n * ch], @intCast(num_frames), @intCast(num_channels));
    // Ada feature-frame tap: this callback IS the playback clock, so frames
    // computed here are aligned to what the ears hear (mono mix, ch == 1).
    feature_tap.processMixed(buffer[0..n], !world.idle());
}

pub const Output = struct {
    pub fn init(alloc: std.mem.Allocator) Output {
        // Entity buffers are allocated from std.heap.c_allocator (see
        // world.zig's doc comment on Entity), independent of `alloc` -
        // kept as a parameter for call-site compatibility with the
        // previous PulseAudio-backed Output.
        _ = alloc;
        if (!g_setup_done) {
            g_world = world_mod.World.init(std.heap.c_allocator, MIX_RATE);
            saudio.setup(.{
                .sample_rate = @intCast(MIX_RATE),
                .num_channels = 1,
                .stream_userdata_cb = streamCb,
                .user_data = &g_world,
            });
            g_setup_done = true;
        }
        return .{};
    }

    /// Resamples `samples` (at `rate`) to the World's fixed mix rate and
    /// enqueues it as an Entity on the first idle channel. Returns once
    /// enqueued - playback itself happens asynchronously (see `drain`).
    pub fn play(self: *Output, samples: []const f32, rate: u32) !void {
        _ = self;
        const resampled = try resample(std.heap.c_allocator, samples, rate, g_world.mix_rate);
        _ = try g_world.enqueue(null, .{ .samples = resampled });
    }

    /// Like `play`, but first enqueues each file in `stinger_paths` (in
    /// order, resolved absolute paths) onto the *same* channel the voice
    /// buffer lands on - the channel's strict FIFO queue (world.zig) is
    /// what makes "stinger blocks the voice" fall out for free, with no
    /// special-cased blocking logic needed here. A stinger file that
    /// fails to load (missing/corrupt) is skipped, not fatal - a typo'd
    /// path shouldn't break the actual speech.
    ///
    /// `background`, if given, is enqueued on a *different* channel (so
    /// `World.mix()`'s cross-channel summing plays it concurrently with
    /// the stinger+voice, not sequentially after) - tiled/truncated to
    /// exactly the stinger+voice's combined length rather than looping
    /// forever, so a background ambience can't outlive its request (this
    /// matters for the long-running daemon: a real infinite loop would
    /// never stop once started).
    pub fn playChain(self: *Output, alloc: std.mem.Allocator, io: std.Io, samples: []const f32, rate: u32, stinger_paths: []const []const u8, background: ?effects_mod.BackgroundChoice) !void {
        _ = self;
        var channel: ?usize = null;
        var total_frames: usize = 0;
        for (stinger_paths) |path| {
            const wav_data = wav_mod.readMono(alloc, io, path) catch continue;
            const resampled = resample(std.heap.c_allocator, wav_data.samples, wav_data.sample_rate, g_world.mix_rate) catch continue;
            total_frames += resampled.len;
            channel = try g_world.enqueue(channel, .{ .samples = resampled });
        }
        const voice_resampled = try resample(std.heap.c_allocator, samples, rate, g_world.mix_rate);
        total_frames += voice_resampled.len;
        _ = try g_world.enqueue(channel, .{ .samples = voice_resampled });

        if (background) |bg| {
            const wav_data = wav_mod.readMono(alloc, io, bg.path) catch return;
            const bg_resampled = resample(alloc, wav_data.samples, wav_data.sample_rate, g_world.mix_rate) catch return;
            const tiled = try std.heap.c_allocator.alloc(f32, total_frames);
            if (bg_resampled.len == 0) {
                @memset(tiled, 0);
            } else {
                for (tiled, 0..) |*s, i| s.* = std.math.clamp(bg_resampled[i % bg_resampled.len] * bg.volume, -1.0, 1.0);
            }
            _ = try g_world.enqueue(null, .{ .samples = tiled });
        }
    }

    /// Waits until everything enqueued so far has finished playing.
    pub fn drain(self: *Output, io: std.Io) void {
        _ = self;
        while (!g_world.idle()) {
            std.Io.sleep(io, .fromMilliseconds(20), .awake) catch return;
        }
    }

    /// No-op: `saudio`/`World` are process-wide singletons (see file doc
    /// comment) shared by every `Output` handle, so an individual handle
    /// going out of scope can't safely tear either down - nothing in this
    /// codebase calls this today, but it's kept for API completeness.
    pub fn deinit(self: *Output) void {
        _ = self;
    }
};
