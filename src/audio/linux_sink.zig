//! Linux-only direct-to-sink playback, used when a request explicitly
//! names a speaker (tmp/FUN_PLAN.md section 1) - bypasses the portable
//! sokol_audio/World mixer (src/audio/output.zig, src/audio/world.zig)
//! entirely for that one request, since sokol_audio has no device
//! selection API on any platform (confirmed by reading its backend
//! source - see tmp/FUN_PLAN.md section 2's "Prerequisite" subsection).
//! A one-off blocking PulseAudio "simple" connection, closed after the
//! one request completes - simplicity over connection reuse, since
//! explicit speaker selection is expected to be occasional/deliberate,
//! not the hot path (that's still the default sokol_audio path).

const builtin = @import("builtin");
const std = @import("std");

pub const SinkPlayError = error{ UnsupportedPlatform, ConnectFailed };

/// Blocks until `samples` (at `rate`) have finished playing on the sink
/// named `sink_name` (a raw PulseAudio/PipeWire sink name, e.g. from
/// `voice speakers` - NOT a config alias, callers resolve that first).
pub fn playToSink(alloc: std.mem.Allocator, samples: []const f32, rate: u32, sink_name: []const u8) SinkPlayError!void {
    if (comptime builtin.os.tag != .linux) return SinkPlayError.UnsupportedPlatform;

    const c = @import("pulse_c");
    const sink_name_z = alloc.dupeSentinel(u8, sink_name, 0) catch return SinkPlayError.ConnectFailed;
    defer alloc.free(sink_name_z);

    const spec = c.pa_sample_spec{
        .format = c.PA_SAMPLE_FLOAT32NE,
        .rate = rate,
        .channels = 1,
    };
    var err: c_int = 0;
    const handle = c.pa_simple_new(
        null, // default server
        "presence-voice",
        c.PA_STREAM_PLAYBACK,
        sink_name_z.ptr,
        "synthesized speech (explicit speaker)",
        &spec,
        null,
        null,
        &err,
    ) orelse return SinkPlayError.ConnectFailed;
    defer c.pa_simple_free(handle);

    const bytes = std.mem.sliceAsBytes(samples);
    if (c.pa_simple_write(handle, bytes.ptr, bytes.len, &err) < 0) {
        return SinkPlayError.ConnectFailed;
    }
    _ = c.pa_simple_drain(handle, &err);
}
