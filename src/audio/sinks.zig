//! Lists PulseAudio/PipeWire sinks by shelling out to `pactl -f json list
//! sinks` (tmp/FUN_PLAN.md section 1), rather than implementing
//! PulseAudio's full async `pa_context_*` introspection API in Zig - this
//! only needs to run once, occasionally, at the terminal (the `speakers`
//! subcommand). Linux-only, matching the rest of the speaker-selection
//! feature (see src/audio/linux_sink.zig).

const std = @import("std");

pub const SinkInfo = struct {
    name: []const u8,
    description: []const u8,
};

pub const SinkError = error{PactlFailed};

/// Returns a caller-owned slice of sinks (parsed from `pactl`'s JSON
/// output). `alloc` should be an arena or similar - the returned strings
/// point into memory owned by the same allocator.
pub fn listSinks(alloc: std.mem.Allocator, io: std.Io) ![]SinkInfo {
    const result = try std.process.run(alloc, io, .{
        .argv = &.{ "pactl", "-f", "json", "list", "sinks" },
    });
    if (!result.term.success()) return SinkError.PactlFailed;

    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, result.stdout, .{});
    const items = parsed.array.items;
    const out = try alloc.alloc(SinkInfo, items.len);
    for (items, 0..) |item, i| {
        const obj = item.object;
        out[i] = .{
            .name = if (obj.get("name")) |v| v.string else "",
            .description = if (obj.get("description")) |v| v.string else "",
        };
    }
    return out;
}
