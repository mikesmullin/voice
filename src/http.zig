//! HTTP server (milestone 4, section 5 of tmp/PHASE3_PLAN.md): `GET /health`,
//! `GET /voices`, `POST /speak`. Opt-in via `serve --http`, running
//! alongside the unix socket listener in the same daemon process (per
//! section 6's "serve/http merge" decision) - on its own OS thread since
//! std.Io's default threaded backend is safe to share across threads.
//!
//! `mode=download` returns WAV bytes (for the browser SPA, milestone 6);
//! `mode=play` (the default) plays through the daemon's own persistent
//! audio output, same as the unix socket protocol.

const std = @import("std");
const daemon_mod = @import("daemon.zig");
const wav = @import("audio/wav.zig");
const effects = @import("audio/effects.zig");
const cli = @import("cli.zig");
const paths = @import("paths.zig");
const timing = @import("timing.zig");

pub fn serve(daemon: *daemon_mod.Daemon, io: std.Io, port: u16, log: *std.Io.Writer) !void {
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", port);
    var server = try addr.listen(io, .{});
    defer server.socket.close(io);

    timing.logf(log, io, "[Daemon] HTTP API listening on 127.0.0.1:{d}\n", .{port});

    while (true) {
        var conn = server.accept(io) catch |err| {
            timing.logf(log, io, "[HTTP] accept error: {t}\n", .{err});
            continue;
        };
        defer conn.close(io);
        handleConnection(daemon, &conn, io, log) catch |err| {
            timing.logf(log, io, "[HTTP] request error: {t}\n", .{err});
        };
    }
}

fn handleConnection(daemon: *daemon_mod.Daemon, conn: *std.Io.net.Stream, io: std.Io, log: *std.Io.Writer) !void {
    var in_buf: [1 << 16]u8 = undefined;
    var out_buf: [1 << 16]u8 = undefined;
    var reader = conn.reader(io, &in_buf);
    var writer = conn.writer(io, &out_buf);

    while (true) {
        var http_server = std.http.Server.init(&reader.interface, &writer.interface);
        var request = http_server.receiveHead() catch break;

        const target = request.head.target;
        const method = request.head.method;

        if (method == .GET and std.mem.eql(u8, target, "/health")) {
            try request.respond("{\"status\":\"ok\"}\n", .{
                .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
            });
        } else if (method == .GET and std.mem.eql(u8, target, "/voices")) {
            try respondVoices(daemon, &request);
        } else if (method == .POST and std.mem.eql(u8, target, "/speak")) {
            try handleSpeak(daemon, &request, io, log);
        } else if (method == .GET and std.mem.eql(u8, target, "/")) {
            try respondIndex(daemon, &request, io);
        } else {
            try request.respond("not found\n", .{ .status = .not_found });
        }

        if (!request.head.keep_alive) break;
    }
}

fn respondIndex(daemon: *daemon_mod.Daemon, request: *std.http.Server.Request, io: std.Io) !void {
    var frame_arena = std.heap.ArenaAllocator.init(daemon.allocator);
    defer frame_arena.deinit();
    const alloc = frame_arena.allocator();

    const html = std.Io.Dir.cwd().readFileAlloc(io, paths.WEB_INDEX, alloc, .limited(1 << 20)) catch {
        try request.respond("presence-voice v2 - see /health, /voices, /speak\n(web/index.html not found)\n", .{});
        return;
    };
    try request.respond(html, .{
        .extra_headers = &.{.{ .name = "content-type", .value = "text/html; charset=utf-8" }},
    });
}

fn respondVoices(daemon: *daemon_mod.Daemon, request: *std.http.Server.Request) !void {
    var frame_arena = std.heap.ArenaAllocator.init(daemon.allocator);
    defer frame_arena.deinit();
    const alloc = frame_arena.allocator();

    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(alloc, "[");
    var it = daemon.config.voices.iterator();
    var first = true;
    while (it.next()) |kv| {
        if (!first) try out.appendSlice(alloc, ",");
        first = false;
        const entry = try std.fmt.allocPrint(alloc, "{{\"name\":\"{s}\",\"engine\":\"{s}\",\"voice\":\"{s}\"}}", .{
            kv.key_ptr.*, kv.value_ptr.engine, kv.value_ptr.voice,
        });
        try out.appendSlice(alloc, entry);
    }
    try out.appendSlice(alloc, "]\n");

    try request.respond(out.items, .{
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

fn handleSpeak(daemon: *daemon_mod.Daemon, request: *std.http.Server.Request, io: std.Io, log: *std.Io.Writer) !void {
    var frame_arena = std.heap.ArenaAllocator.init(daemon.allocator);
    defer frame_arena.deinit();
    const alloc = frame_arena.allocator();

    var body_buf: [1 << 16]u8 = undefined;
    const body_reader = request.readerExpectNone(&body_buf);
    const body = try body_reader.allocRemaining(alloc, .limited(1 << 20));

    const parsed = std.json.parseFromSliceLeaky(std.json.Value, alloc, body, .{}) catch {
        try request.respond("{\"error\":\"invalid JSON body\"}\n", .{ .status = .bad_request });
        return;
    };
    const obj = parsed.object;
    const text = if (obj.get("text")) |v| v.string else {
        try request.respond("{\"error\":\"missing 'text'\"}\n", .{ .status = .bad_request });
        return;
    };
    const voice_name = if (obj.get("voice")) |v| v.string else daemon.config.default_preset orelse daemon.config.fallback_voice orelse "";

    const preset = daemon.config.getPreset(voice_name) orelse {
        try request.respond("{\"error\":\"unknown voice\"}\n", .{ .status = .bad_request });
        return;
    };

    const mode = if (obj.get("mode")) |v| v.string else "play";
    const gain: f32 = if (obj.get("gain")) |v| switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 1.0,
    } else 1.0;
    const speaker: ?[]const u8 = if (obj.get("speaker")) |v| switch (v) {
        .string => |s| if (s.len == 0) null else s,
        else => null,
    } else null;
    var effect_names: std.ArrayList([]const u8) = .empty;
    if (obj.get("effects")) |v| switch (v) {
        .array => |arr| for (arr.items) |item| if (item == .string) try effect_names.append(alloc, item.string),
        else => {},
    };

    if (std.mem.eql(u8, mode, "download")) {
        const t0 = timing.elapsedSeconds(io);
        var result = daemon.synthesize(alloc, preset, text, log) catch |err| {
            timing.logf(log, io, "[HTTP] synth error: {t}\n", .{err});
            try request.respond("{\"error\":\"synthesis failed\"}\n", .{ .status = .internal_server_error });
            return;
        };
        const resolved = effects.resolveEffects(alloc, &daemon.config, effect_names.items) catch {
            try request.respond("{\"error\":\"unknown effect (see config.yaml's effects: block)\"}\n", .{ .status = .bad_request });
            return;
        };
        if (resolved.chain.items.len > 0) {
            result.samples = try effects.applyChain(alloc, result.samples, result.sample_rate, resolved.chain.items);
        }
        cli.applyGain(result.samples, std.math.clamp(gain, 0.0, 2.0));
        const wav_bytes = try wav.encodeMono16(alloc, result.sample_rate, result.samples);
        timing.logf(log, io, "[HTTP] /speak {s} (download): {d} samples ({d:.2}s)\n", .{ voice_name, result.samples.len, timing.elapsedSeconds(io) - t0 });
        try request.respond(wav_bytes, .{
            .extra_headers = &.{.{ .name = "content-type", .value = "audio/wav" }},
        });
        return;
    }

    // mode=play schedules an utterance on the daemon's output, so the
    // client must say what happens if speech is already playing —
    // "schedule": "enqueue" (queue behind it) or "interrupt" (silence it
    // first). Required, mirroring the unix-socket protocol (daemon.zig).
    const schedule: []const u8 = if (obj.get("schedule")) |v| switch (v) {
        .string => |s| s,
        else => "",
    } else "";
    const interrupt = std.mem.eql(u8, schedule, "interrupt");
    if (!interrupt and !std.mem.eql(u8, schedule, "enqueue")) {
        try request.respond("{\"error\":\"schedule is required for mode=play: \\\"enqueue\\\" or \\\"interrupt\\\"\"}\n", .{ .status = .bad_request });
        return;
    }
    if (interrupt) daemon.output.stopSpeech();

    const t0 = timing.elapsedSeconds(io);
    const n = daemon.synthesizeAndPlay(preset, text, true, log, speaker, effect_names.items) catch |err| {
        if (err == error.UnknownSpeaker) {
            try request.respond("{\"error\":\"unknown speaker (see 'voice speakers')\"}\n", .{ .status = .bad_request });
            return;
        }
        if (err == error.UnknownEffect) {
            try request.respond("{\"error\":\"unknown effect (see config.yaml's effects: block)\"}\n", .{ .status = .bad_request });
            return;
        }
        timing.logf(log, io, "[HTTP] synth error: {t}\n", .{err});
        try request.respond("{\"error\":\"synthesis failed\"}\n", .{ .status = .internal_server_error });
        return;
    };
    timing.logf(log, io, "[HTTP] /speak {s}: {d} samples ({d:.2}s)\n", .{ voice_name, n, timing.elapsedSeconds(io) - t0 });

    try request.respond("{\"status\":\"ok\"}\n", .{
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}
