//! HTTP server (milestone 4, section 5 of tmp/PHASE3_PLAN.md): `GET /health`,
//! `GET /voices`, `POST /speak`. Opt-in via `serve --http`, running
//! alongside the unix socket listener in the same daemon process (per
//! section 6's "serve/http merge" decision) - on its own OS thread since
//! std.Io's default threaded backend is safe to share across threads.
//!
//! `mode=download` (returning WAV bytes instead of playing) is not yet
//! implemented - only `mode=play` (the default) works so far.

const std = @import("std");
const daemon_mod = @import("daemon.zig");

pub fn serve(daemon: *daemon_mod.Daemon, io: std.Io, port: u16, log: *std.Io.Writer) !void {
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", port);
    var server = try addr.listen(io, .{});
    defer server.socket.close(io);

    try log.print("[Daemon] HTTP API listening on 127.0.0.1:{d}\n", .{port});
    try log.flush();

    while (true) {
        var conn = server.accept(io) catch |err| {
            try log.print("[HTTP] accept error: {t}\n", .{err});
            try log.flush();
            continue;
        };
        defer conn.close(io);
        handleConnection(daemon, &conn, io, log) catch |err| {
            try log.print("[HTTP] request error: {t}\n", .{err});
            try log.flush();
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
            try request.respond("presence-voice v2 - see /health, /voices, /speak\n", .{});
        } else {
            try request.respond("not found\n", .{ .status = .not_found });
        }

        if (!request.head.keep_alive) break;
    }
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
    _ = io;
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

    const n = daemon.synthesizeAndPlay(preset, text, true) catch |err| {
        try log.print("[HTTP] synth error: {t}\n", .{err});
        try log.flush();
        try request.respond("{\"error\":\"synthesis failed\"}\n", .{ .status = .internal_server_error });
        return;
    };
    try log.print("[HTTP] /speak {s}: {d} samples\n", .{ voice_name, n });
    try log.flush();

    try request.respond("{\"status\":\"ok\"}\n", .{
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}
