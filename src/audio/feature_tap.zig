//! Ada feature-frame tap (spec: /workspace/ada/docs/PROTOCOL.md, request:
//! tmp/ADA_FEATURE_FRAMES_REQUEST.md): publishes ~60 Hz audio feature
//! frames of whatever the World is playing, plus speak-start/speak-end
//! JSON events, to unix-socket subscribers (`subscribe\tlevels` /
//! `subscribe\tevents` — see daemon.zig).
//!
//! The tap hooks output.zig's sokol_audio stream callback, i.e. the
//! actual playback clock — frames are aligned to what the ears hear by
//! construction. The audio thread only accumulates window sums, steps
//! four biquad bandpasses, and pushes packed frames into a lock-free SPSC
//! ring (same "no Io on the audio thread" constraint world.zig documents);
//! a fan-out thread drains the ring to subscribers so a slow client can
//! never stall mixing.

const std = @import("std");

/// 'AVF1' little-endian (bytes "AVF1"); layout per PROTOCOL.md §1.
pub const FRAME_MAGIC: u32 = 0x31465641;
pub const STREAM_TTS: u32 = 1;
pub const FLAG_VAD: u32 = 1 << 0;
pub const FLAG_START: u32 = 1 << 1;
pub const FLAG_END: u32 = 1 << 2;

pub const Frame = extern struct {
    magic: u32 = FRAME_MAGIC,
    stream_id: u32 = STREAM_TTS,
    rms: f32 = 0,
    band: [4]f32 = .{ 0, 0, 0, 0 },
    pitch_hint: f32 = 0,
    flags: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Frame) == 36);
}

const FRAMES_PER_SEC = 60;

/// RBJ bandpass biquad (constant-skirt). Runs per-sample on the audio
/// thread — a handful of flops, negligible next to the mixing itself.
const Biquad = struct {
    b0: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32 = 0,
    x2: f32 = 0,
    y1: f32 = 0,
    y2: f32 = 0,

    fn bandpass(fc: f32, q: f32, fs: f32) Biquad {
        const w = 2.0 * std.math.pi * fc / fs;
        const alpha = @sin(w) / (2.0 * q);
        const a0 = 1.0 + alpha;
        return .{
            .b0 = alpha / a0,
            .b2 = -alpha / a0,
            .a1 = -2.0 * @cos(w) / a0,
            .a2 = (1.0 - alpha) / a0,
        };
    }

    fn step(self: *Biquad, x: f32) f32 {
        const y = self.b0 * x + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        return y;
    }
};

/// Same perceptual compression perception-voice's FeatureExtractor uses,
/// so both streams drive the shader on a comparable 0..1 scale.
fn compress(x: f32) f32 {
    return @min(1.0, std.math.pow(f32, x * 8.0, 0.7));
}

const RING_SIZE = 256; // power of two; ~4s of frames

const G = struct {
    // -- audio-thread state (single producer) --
    // band centers ~geometric mids of perception-voice's 0-300/300-1k/
    // 1k-3k/3k-8k Hz edges; Q chosen to roughly span each band
    var biquads: [4]Biquad = undefined;
    var win_sum2: f64 = 0;
    var band_sum2: [4]f64 = .{ 0, 0, 0, 0 };
    var win_count: usize = 0;
    var window_len: usize = 400; // mix_rate / 60, set in start()
    var was_active: bool = false;
    var pending_start: bool = false;

    // SPSC ring: audio thread writes, fan-out thread reads
    var ring: [RING_SIZE]Frame = undefined;
    var ring_head: std.atomic.Value(usize) = .init(0); // write index
    var ring_tail: std.atomic.Value(usize) = .init(0); // read index

    // -- subscriber registry + event queue (spin-locked, world.zig style:
    //    touched from daemon/HTTP threads and the fan-out thread) --
    var mutex: std.atomic.Mutex = .unlocked;
    var levels_subs: std.ArrayList(std.Io.net.Stream) = .empty;
    var events_subs: std.ArrayList(std.Io.net.Stream) = .empty;
    var event_queue: std.ArrayList([]u8) = .empty;

    var alloc: std.mem.Allocator = undefined;
    var io: std.Io = undefined;
    var started: bool = false;
};

fn lock() void {
    while (!G.mutex.tryLock()) std.atomic.spinLoopHint();
}

/// Spawns the fan-out thread. Call once from the daemon before serving.
pub fn start(alloc: std.mem.Allocator, io: std.Io, mix_rate: u32) !void {
    if (G.started) return;
    G.alloc = alloc;
    G.io = io;
    G.window_len = mix_rate / FRAMES_PER_SEC;
    G.biquads = .{
        Biquad.bandpass(150.0, 0.6, @floatFromInt(mix_rate)),
        Biquad.bandpass(600.0, 0.8, @floatFromInt(mix_rate)),
        Biquad.bandpass(1800.0, 0.8, @floatFromInt(mix_rate)),
        Biquad.bandpass(5000.0, 0.7, @floatFromInt(mix_rate)),
    };
    G.started = true;
    const t = try std.Thread.spawn(.{}, fanoutThread, .{});
    t.detach();
}

// ---------------------------------------------------------------------------
// Audio thread side

/// Called from the sokol_audio stream callback with the freshly mixed
/// mono block. `active` = whether any channel is playing (World not idle).
pub fn processMixed(samples: []const f32, active: bool) void {
    if (!G.started) return;

    if (!active) {
        if (G.was_active) {
            // flush whatever partial window remains as the closing frame
            emitWindow(FLAG_END);
            G.was_active = false;
            resetWindow();
            for (&G.biquads) |*bq| {
                bq.x1 = 0;
                bq.x2 = 0;
                bq.y1 = 0;
                bq.y2 = 0;
            }
        }
        return;
    }

    if (!G.was_active) {
        G.was_active = true;
        G.pending_start = true;
    }

    for (samples) |s| {
        G.win_sum2 += @as(f64, s) * @as(f64, s);
        for (&G.biquads, 0..) |*bq, i| {
            const y = bq.step(s);
            G.band_sum2[i] += @as(f64, y) * @as(f64, y);
        }
        G.win_count += 1;
        if (G.win_count >= G.window_len) {
            const start_flag: u32 = if (G.pending_start) FLAG_START else 0;
            G.pending_start = false;
            emitWindow(start_flag);
        }
    }
}

fn resetWindow() void {
    G.win_sum2 = 0;
    G.band_sum2 = .{ 0, 0, 0, 0 };
    G.win_count = 0;
}

fn emitWindow(extra_flags: u32) void {
    const n: f64 = @floatFromInt(@max(G.win_count, 1));
    var frame = Frame{
        .rms = compress(@floatCast(@sqrt(G.win_sum2 / n))),
        .flags = extra_flags | FLAG_VAD,
    };
    for (&frame.band, G.band_sum2) |*b, sum2| {
        b.* = compress(@floatCast(@sqrt(sum2 / n)));
    }
    resetWindow();

    // SPSC push; drop the frame if the consumer is behind (disposable)
    const head = G.ring_head.load(.monotonic);
    const tail = G.ring_tail.load(.acquire);
    if (head -% tail >= RING_SIZE) return;
    G.ring[head % RING_SIZE] = frame;
    G.ring_head.store(head +% 1, .release);
}

// ---------------------------------------------------------------------------
// Daemon side

/// Takes ownership of `conn`; it will be written to by the fan-out thread
/// and closed on the first failed write.
pub fn addLevelsSubscriber(conn: std.Io.net.Stream) !void {
    lock();
    defer G.mutex.unlock();
    try G.levels_subs.append(G.alloc, conn);
}

pub fn addEventsSubscriber(conn: std.Io.net.Stream) !void {
    lock();
    defer G.mutex.unlock();
    try G.events_subs.append(G.alloc, conn);
}

/// Queues a speak-start/speak-end JSON line for events subscribers.
/// Called from the daemon/HTTP request threads (never the audio thread).
pub fn publishEvent(comptime fmt: []const u8, args: anytype) void {
    if (!G.started) return;
    lock();
    defer G.mutex.unlock();
    if (G.events_subs.items.len == 0) return;
    const line = std.fmt.allocPrint(G.alloc, fmt ++ "\n", args) catch return;
    G.event_queue.append(G.alloc, line) catch {
        G.alloc.free(line);
    };
}

pub fn speakStart(text: []const u8) void {
    // JSON-escape the text minimally (quotes/backslashes/newlines)
    if (!G.started) return;
    var buf: [512]u8 = undefined;
    var n: usize = 0;
    for (text) |c| {
        if (n + 2 >= buf.len) break;
        switch (c) {
            '"', '\\' => {
                buf[n] = '\\';
                buf[n + 1] = c;
                n += 2;
            },
            '\n', '\r', '\t' => {
                buf[n] = ' ';
                n += 1;
            },
            else => {
                buf[n] = c;
                n += 1;
            },
        }
    }
    publishEvent("{{\"ev\":\"speak-start\",\"text\":\"{s}\"}}", .{buf[0..n]});
}

pub fn speakEnd() void {
    publishEvent("{{\"ev\":\"speak-end\"}}", .{});
}

// ---------------------------------------------------------------------------
// Fan-out thread

fn fanoutThread() void {
    while (true) {
        // drain frames to levels subscribers
        var batch: [32]Frame = undefined;
        var count: usize = 0;
        const head = G.ring_head.load(.acquire);
        var tail = G.ring_tail.load(.monotonic);
        while (tail != head and count < batch.len) : (tail +%= 1) {
            batch[count] = G.ring[tail % RING_SIZE];
            count += 1;
        }
        G.ring_tail.store(tail, .release);

        if (count > 0) {
            const bytes = std.mem.sliceAsBytes(batch[0..count]);
            broadcast(&G.levels_subs, bytes);
            // playback finished (World went idle) -> tell events subscribers
            for (batch[0..count]) |f| {
                if (f.flags & FLAG_END != 0) speakEnd();
            }
        }

        // drain queued JSON events to events subscribers
        while (true) {
            lock();
            const line: ?[]u8 = if (G.event_queue.items.len > 0) G.event_queue.orderedRemove(0) else null;
            G.mutex.unlock();
            if (line == null) break;
            broadcast(&G.events_subs, line.?);
            G.alloc.free(line.?);
        }

        std.Io.sleep(G.io, .fromMilliseconds(8), .awake) catch return;
    }
}

fn broadcast(subs: *std.ArrayList(std.Io.net.Stream), bytes: []const u8) void {
    lock();
    const conns = G.alloc.dupe(std.Io.net.Stream, subs.items) catch {
        G.mutex.unlock();
        return;
    };
    G.mutex.unlock();
    defer G.alloc.free(conns);

    for (conns) |conn| {
        var c = conn;
        var wbuf: [2048]u8 = undefined;
        var w = c.writer(G.io, &wbuf);
        const ok = blk: {
            w.interface.writeAll(bytes) catch break :blk false;
            w.interface.flush() catch break :blk false;
            break :blk true;
        };
        if (!ok) dropSubscriber(subs, conn);
    }
}

fn dropSubscriber(subs: *std.ArrayList(std.Io.net.Stream), conn: std.Io.net.Stream) void {
    lock();
    for (subs.items, 0..) |c, i| {
        if (c.socket.handle == conn.socket.handle) {
            _ = subs.swapRemove(i);
            break;
        }
    }
    G.mutex.unlock();
    var dead = conn;
    dead.close(G.io);
}
