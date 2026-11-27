//! Persistent llama-server sidecar for Qwen3-TTS (see src/engines/qwen.zig).
//!
//! The one-shot `llama-tts` binary reloads ~1.5 GB of weights per request
//! (~1.7s tax every call). `llama-server` (v1.0.0+) loads the same GGUFs
//! once and serves `POST /v1/audio/speech` against the resident model, so
//! repeat calls pay inference only. The sidecar starts LAZILY on the first
//! qwen request (never at `voice serve` startup - same lazy-load contract
//! as every other engine) and lives until the machine reboots or someone
//! kills it (`pkill -f "llama-server.*:8081"`), outliving any single
//! `voice` process - `voice local` calls adopt the daemon's sidecar
//! through the same port health check.
//!
//! One sidecar per process backend: the first qwen request fixes the
//! backend (CPU `-ngl 0` / GPU `-ngl 99`); a later request needing the
//! other backend gets `error.BackendMismatch` and the caller falls back to
//! the one-shot subprocess (correct audio, just slower - never a failure).
//! Module-level state mirrors src/audio/output.zig's `g_world` singleton.

const std = @import("std");
const wav = @import("../audio/wav.zig");
const paths = @import("../paths.zig");

pub const ServerError = error{
    StartFailed,
    NotReady,
    BackendMismatch,
    RequestFailed,
    BadResponse,
} || std.mem.Allocator.Error;

const QWEN_SERVER_MODEL = "qwen3-tts";

const State = struct {
    started: bool = false,
    /// Another thread is mid-start (spawn + health poll); wait for it
    /// instead of spawning a second sidecar.
    starting: bool = false,
    gpu: bool = false,
};

var g_state: State = .{};
var g_lock: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!g_lock.tryLock()) std.atomic.spinLoopHint();
}

fn serverBase(alloc: std.mem.Allocator) ![]const u8 {
    return std.fmt.allocPrint(alloc, "http://127.0.0.1:{d}", .{paths.QWEN_SERVER_PORT});
}

/// Starts the sidecar on first use (per backend) and waits until its
/// /health reports ok. Concurrent first-requests serialize on `starting`:
/// exactly one spawns, the rest wait for it.
pub fn ensureServer(alloc: std.mem.Allocator, io: std.Io, gpu: bool, log: ?*std.Io.Writer) ServerError!void {
    lock();
    if (g_state.started and g_state.gpu == gpu) {
        g_lock.unlock();
        return;
    }
    if (g_state.started) {
        g_lock.unlock();
        return error.BackendMismatch;
    }
    if (g_state.starting) {
        g_lock.unlock();
        return waitForStarted(io);
    }
    g_state.starting = true;
    g_lock.unlock();

    startSidecar(alloc, io, gpu, log) catch {
        lock();
        g_state.starting = false;
        g_lock.unlock();
        return error.StartFailed;
    };
    // The starter polls the endpoint itself (waitForStarted is for
    // follower threads - it waits on the flag set below).
    pollHealth(alloc, io) catch {
        lock();
        g_state.starting = false;
        g_lock.unlock();
        return error.NotReady;
    };
    lock();
    g_state.started = true;
    g_state.starting = false;
    g_state.gpu = gpu;
    g_lock.unlock();
    if (log) |l| l.print("[Qwen] sidecar ready (backend {s})\n", .{if (gpu) "gpu" else "cpu"}) catch {};
}

/// Another thread is starting the sidecar; poll until it reports ready.
fn waitForStarted(io: std.Io) ServerError!void {
    var tries: usize = 0;
    while (tries < 150) : (tries += 1) {
        lock();
        const done = g_state.started;
        const failed = !g_state.starting and !done;
        g_lock.unlock();
        if (done) return;
        if (failed) return error.NotReady;
        std.Io.sleep(io, .fromSeconds(2), .awake) catch return error.NotReady;
    }
    return error.NotReady;
}

/// Poll GET /health until it reports ok (cold model load can take
/// minutes on first boot; warm page cache is seconds).
fn pollHealth(alloc: std.mem.Allocator, io: std.Io) ServerError!void {
    const base = try serverBase(alloc);
    defer alloc.free(base);
    var tries: usize = 0;
    while (tries < 150) : (tries += 1) {
        if (healthOk(alloc, io, base)) return;
        std.Io.sleep(io, .fromSeconds(2), .awake) catch return error.NotReady;
    }
    return error.NotReady;
}

fn startSidecar(_: std.mem.Allocator, io: std.Io, gpu: bool, log: ?*std.Io.Writer) !void {
    var port_buf: [8]u8 = undefined;
    const port = std.fmt.bufPrint(&port_buf, "{d}", .{paths.QWEN_SERVER_PORT}) catch unreachable;
    // Short utterances only: small context + 2 slots (covers a socket and
    // an HTTP request in flight at once). Keeps the CPU sidecar's RAM
    // footprint near the weights instead of the 32k-ctx x slots default.
    // CPU sidecars also hide the GPU entirely (same CUDA-init problem the
    // one-shot path works around - see qwen.zig's spawn).
    var argv_buf: [20][]const u8 = undefined;
    var argv_len: usize = 0;
    if (!gpu) {
        argv_buf[0] = "env";
        argv_buf[1] = "CUDA_VISIBLE_DEVICES=";
        argv_len = 2;
    }
    const rest = [_][]const u8{
        paths.QWEN_SERVER_BIN,
        "-m",
        paths.QWEN_BACKBONE,
        "--mmproj",
        paths.QWEN_MMPROJ,
        "--tts-lang",
        "en",
        "--port",
        port,
        "--n-gpu-layers",
        if (gpu) "99" else "0",
        "-c",
        "4096",
        "-np",
        "2",
    };
    @memcpy(argv_buf[argv_len..][0..rest.len], &rest);
    const argv = argv_buf[0 .. argv_len + rest.len];
    if (log) |l| l.print("[Qwen] starting sidecar (port {s}, backend {s})...\n", .{ port, if (gpu) "gpu" else "cpu" }) catch {};
    // Intentionally unreaped: the sidecar outlives any single `voice`
    // process by design (persistent model). The pid survives as a zombie
    // entry only until the spawning process exits; the server itself is
    // unaffected. Stop it with: pkill -f "llama-server.*:8081".
    const child = std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    }) catch return error.StartFailed;
    _ = child;
}

/// Multipart upload (not JSON): the server only engages voice cloning
/// when the reference audio arrives as file bytes. A JSON `speaker_ref`
/// path is silently ignored (server logs "no speaker reference
/// provided" and synthesizes the default voice instead - a wrong-voice
/// success, worse than a loud failure).

/// Health probe: GET /health must report {"status":"ok"}.
fn healthOk(alloc: std.mem.Allocator, io: std.Io, base: []const u8) bool {
    const url = std.fmt.allocPrint(alloc, "{s}/health", .{base}) catch return false;
    defer alloc.free(url);
    var client: std.http.Client = .{ .allocator = alloc, .io = io };
    defer client.deinit();
    var buf: std.Io.Writer.Allocating = .init(alloc);
    defer buf.deinit();
    const res = client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .extra_headers = &.{},
        .response_writer = &buf.writer,
        .keep_alive = false,
    }) catch return false;
    if (res.status != .ok) return false;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, alloc, buf.written(), .{}) catch return false;
    const status = parsed.object.get("status") orelse return false;
    return status == .string and std.mem.eql(u8, status.string, "ok");
}

/// One synthesis via the resident sidecar. `speaker_ref` is the clone
/// source WAV (server-local path); `speaker_text` is an optional sibling
/// transcript file (omitted when missing/unreadable - the server still
/// clones, per-probe, just with less to anchor on). Returns f32 mono
/// samples + rate, same contract as the subprocess path.
pub fn synthesizeViaServer(
    alloc: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    voice: []const u8,
    speaker_ref: []const u8,
    speaker_text_path: ?[]const u8,
    gpu: bool,
    log: ?*std.Io.Writer,
) ServerError!wav.WavData {
    try ensureServer(alloc, io, gpu, log);

    var transcript: ?[]const u8 = null;
    if (speaker_text_path) |p| {
        transcript = std.Io.Dir.cwd().readFileAlloc(io, p, alloc, .limited(1 << 20)) catch null;
    }
    defer if (transcript) |t| alloc.free(t);

    const base = try serverBase(alloc);
    defer alloc.free(base);
    // The sidecar may have died between health checks (OOM-killed, user
    // `pkill`, etc.) - one resurrection attempt before giving up to the
    // subprocess fallback upstream.
    const data = postSpeech(alloc, io, base, text, voice, speaker_ref, transcript, log) catch {
        lock();
        g_state.started = false;
        g_state.starting = false;
        g_lock.unlock();
        try ensureServer(alloc, io, gpu, log);
        return postSpeech(alloc, io, base, text, voice, speaker_ref, transcript, log);
    };
    return data;
}

fn postSpeech(
    alloc: std.mem.Allocator,
    io: std.Io,
    base: []const u8,
    text: []const u8,
    voice: []const u8,
    speaker_ref: []const u8,
    transcript: ?[]const u8,
    log: ?*std.Io.Writer,
) ServerError!wav.WavData {
    // Multipart upload: text fields + the speaker WAV bytes + optional
    // transcript. Hand-rolled (no multipart helper in std.http.Client).
    const audio_bytes = std.Io.Dir.cwd().readFileAlloc(io, speaker_ref, alloc, .limited(64 << 20)) catch {
        if (log) |l| l.print("[Qwen] sidecar: cannot read speaker file {s}\n", .{speaker_ref}) catch {};
        return error.RequestFailed;
    };
    defer alloc.free(audio_bytes);
    const boundary = "presence-voice-qwen-7f3a";
    var body = std.Io.Writer.Allocating.init(alloc);
    defer body.deinit();
    const w = &body.writer;
    writeTextPart(w, boundary, "model", QWEN_SERVER_MODEL) catch return error.RequestFailed;
    writeTextPart(w, boundary, "input", text) catch return error.RequestFailed;
    writeTextPart(w, boundary, "voice", voice) catch return error.RequestFailed;
    w.print("--{s}\r\nContent-Disposition: form-data; name=\"speaker_ref\"; filename=\"speaker.wav\"\r\nContent-Type: audio/wav\r\n\r\n", .{boundary}) catch return error.RequestFailed;
    w.writeAll(audio_bytes) catch return error.RequestFailed;
    w.writeAll("\r\n") catch return error.RequestFailed;
    if (transcript) |t| writeTextPart(w, boundary, "speaker_ref_text", t) catch return error.RequestFailed;
    w.print("--{s}--\r\n", .{boundary}) catch return error.RequestFailed;
    const content_type = try std.fmt.allocPrint(alloc, "multipart/form-data; boundary={s}", .{boundary});
    defer alloc.free(content_type);
    const url = try std.fmt.allocPrint(alloc, "{s}/v1/audio/speech", .{base});
    defer alloc.free(url);

    var client: std.http.Client = .{ .allocator = alloc, .io = io };
    defer client.deinit();
    var buf: std.Io.Writer.Allocating = .init(alloc);
    defer buf.deinit();
    const res = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body.written(),
        .extra_headers = &.{
            .{ .name = "Content-Type", .value = content_type },
        },
        .response_writer = &buf.writer,
        .keep_alive = false,
    }) catch {
        if (log) |l| l.print("[Qwen] sidecar request failed (connection)\n", .{}) catch {};
        return error.RequestFailed;
    };
    if (res.status != .ok) {
        if (log) |l| l.print("[Qwen] sidecar request failed (http {d})\n", .{@intFromEnum(res.status)}) catch {};
        return error.RequestFailed;
    }
    // The body IS a WAV file - round-trip through a tmp file to reuse the
    // existing wav.readMono parser (same pattern as the subprocess path).
    var out_path_buf: [256]u8 = undefined;
    const out_path = std.fmt.bufPrint(&out_path_buf, "{s}{d}.wav", .{ paths.QWEN_TMP_WAV, std.c.getpid() }) catch return error.RequestFailed;
    const f = std.Io.Dir.cwd().createFile(io, out_path, .{}) catch return error.RequestFailed;
    var wbuf: [32768]u8 = undefined;
    var fw = f.writer(io, &wbuf);
    fw.interface.writeAll(buf.written()) catch {
        f.close(io);
        return error.RequestFailed;
    };
    fw.interface.flush() catch {
        f.close(io);
        return error.RequestFailed;
    };
    f.close(io);
    const data = wav.readMono(alloc, io, out_path) catch return error.BadResponse;
    std.Io.Dir.deleteFileAbsolute(io, out_path) catch {};
    return data;
}

fn writeTextPart(w: *std.Io.Writer, boundary: []const u8, name: []const u8, value: []const u8) !void {
    try w.print("--{s}\r\nContent-Disposition: form-data; name=\"{s}\"\r\n\r\n", .{ boundary, name });
    try w.writeAll(value);
    try w.writeAll("\r\n");
}
