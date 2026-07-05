//! Elapsed-time logging, restoring v1's `timing.py` behavior: every log
//! line is prefixed with seconds-since-process-start (`{d:>6.2}`), so
//! successive lines' deltas show how long each step took - useful for
//! spotting further optimization opportunities without a profiler.

const std = @import("std");

var start: ?std.Io.Clock.Timestamp = null;

/// Call once, as early as possible in `main()`.
pub fn startTimer(io: std.Io) void {
    start = std.Io.Clock.Timestamp.now(io, .awake);
}

pub fn elapsedSeconds(io: std.Io) f64 {
    const s = start orelse return 0.0;
    const now = std.Io.Clock.Timestamp.now(io, .awake);
    const dur = s.durationTo(now);
    return @as(f64, @floatFromInt(dur.raw.nanoseconds)) / 1e9;
}

/// Prints `fmt`/`args` prefixed with `{elapsed:>6.2} ` (matching v1's
/// `timing.py` log format), then flushes. Swallows write errors (logging
/// shouldn't crash the program) - `fmt` must still end with its own `\n`.
pub fn logf(w: *std.Io.Writer, io: std.Io, comptime fmt: []const u8, args: anytype) void {
    w.print("{d:>6.2} " ++ fmt, .{elapsedSeconds(io)} ++ args) catch {};
    w.flush() catch {};
}
