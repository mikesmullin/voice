//! Persistent audio output stream via PulseAudio's "simple" blocking API
//! (system libpulse-simple; PipeWire provides a compatible socket on this
//! machine). Keeps one connection open per distinct sample rate seen so
//! far, reused across requests - this is the actual latency win over v1
//! (which opened a fresh `sd.play()` stream per request): device/stream
//! negotiation happens once per rate, not once per synthesis.
//!
//! Piper and Kokoro voices can use different sample rates (16k/22050/
//! 24000 Hz depending on voice), so this can't be a single fixed stream -
//! a small cache keyed by rate is the simplest fix that still gets the
//! reuse benefit for the common case (repeated requests to the same
//! preloaded voice).

const std = @import("std");
const c = @import("pulse_c");

pub const AudioError = error{ConnectFailed};

const Stream = struct {
    handle: *c.pa_simple,
    rate: u32,
};

pub const Output = struct {
    allocator: std.mem.Allocator,
    streams: std.ArrayList(Stream) = .empty,

    pub fn init(alloc: std.mem.Allocator) Output {
        return .{ .allocator = alloc };
    }

    fn getStream(self: *Output, rate: u32) !*c.pa_simple {
        for (self.streams.items) |s| {
            if (s.rate == rate) return s.handle;
        }

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
            null, // default device
            "synthesized speech",
            &spec,
            null, // default channel map
            null, // default buffering
            &err,
        ) orelse return AudioError.ConnectFailed;

        try self.streams.append(self.allocator, .{ .handle = handle, .rate = rate });
        return handle;
    }

    /// Blocks until the samples have been written to the server's buffer
    /// (not until they've finished playing - see `drain` for that).
    pub fn play(self: *Output, samples: []const f32, rate: u32) !void {
        const handle = try self.getStream(rate);
        var err: c_int = 0;
        const bytes = std.mem.sliceAsBytes(samples);
        if (c.pa_simple_write(handle, bytes.ptr, bytes.len, &err) < 0) {
            return AudioError.ConnectFailed;
        }
    }

    /// Waits for all data written so far to actually finish playing.
    pub fn drain(self: *Output, rate: u32) void {
        const handle = self.getStream(rate) catch return;
        var err: c_int = 0;
        _ = c.pa_simple_drain(handle, &err);
    }

    pub fn deinit(self: *Output) void {
        for (self.streams.items) |s| c.pa_simple_free(s.handle);
        self.streams.deinit(self.allocator);
    }
};
