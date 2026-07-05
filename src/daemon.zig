//! Daemon skeleton (milestone 3): unix socket + persistent audio output +
//! preload, per tmp/PHASE3_PLAN.md sections 4/8. Single-threaded, blocking
//! accept loop - concurrency/HTTP are later milestones (4/5).
//!
//! Model files live under ./models/ (gitignored - fetch with
//! ./scripts/fetch-models.sh). Paths are fixed absolute paths (src/
//! paths.zig) rather than cwd-relative, since `voice` is meant to be
//! runnable from anywhere once installed on $PATH - see that file's doc
//! comment for why.
//!
//! Known gap (not yet covered by an approved config schema - see
//! tmp/PHASE3_PLAN.md's "Still open" items): these paths are constants
//! here rather than config.yaml fields. Fine for a skeleton; needs a real
//! config field before voices outside config.yaml's preload list can use
//! anything other than models/piper/<voice-id>.onnx by convention.

const std = @import("std");
const config_mod = @import("config.zig");
const kokoro = @import("engines/kokoro.zig");
const piper = @import("engines/piper.zig");
const ort = @import("engines/onnxruntime.zig");
const audio_output = @import("audio/output.zig");
const paths = @import("paths.zig");
const timing = @import("timing.zig");

// TODO: move to config.yaml once the v2 schema for model locations is
// decided (see tmp/PHASE3_PLAN.md "Still open").
const KOKORO_MODEL_PATH = paths.KOKORO_MODEL;
const KOKORO_VOICES_BIN_PATH = paths.KOKORO_VOICES_BIN;
const KOKORO_VOCAB_PATH = paths.KOKORO_VOCAB;
const PIPER_MODELS_DIR = paths.PIPER_MODELS_DIR;
const ESPEAK_LIB_PATH = "/usr/lib/libespeak-ng.so";
const ESPEAK_DATA_PATH = "/usr/share/espeak-ng-data";

pub const Daemon = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: config_mod.Config,
    rt: ort.Runtime,
    output: audio_output.Output,
    kokoro_phonemizer: kokoro.Phonemizer,
    kokoro_voice: ?kokoro.Voice = null,
    piper_phonemizer: piper.Phonemizer,
    piper_voices: std.StringHashMap(piper.Voice),
    force_cpu: bool = false,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, config: config_mod.Config) !Daemon {
        return .{
            .allocator = alloc,
            .io = io,
            .config = config,
            .rt = try ort.Runtime.init(),
            .output = audio_output.Output.init(alloc),
            .kokoro_phonemizer = try kokoro.Phonemizer.init(alloc, io, paths.ZIG_PHONEMES_DATA, false),
            .piper_phonemizer = try piper.Phonemizer.init(alloc, ESPEAK_LIB_PATH, ESPEAK_DATA_PATH, false),
            .piper_voices = std.StringHashMap(piper.Voice).init(alloc),
        };
    }

    fn getKokoroVoice(self: *Daemon) !*kokoro.Voice {
        if (self.kokoro_voice == null) {
            self.kokoro_voice = try kokoro.Voice.loadWithProvider(&self.rt, self.allocator, self.io, KOKORO_MODEL_PATH, KOKORO_VOCAB_PATH, KOKORO_VOICES_BIN_PATH, !self.force_cpu);
        }
        return &self.kokoro_voice.?;
    }

    fn getPiperVoice(self: *Daemon, voice_id: []const u8) !*piper.Voice {
        if (self.piper_voices.getPtr(voice_id)) |v| return v;

        const model_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}.onnx", .{ PIPER_MODELS_DIR, voice_id });
        const config_path = try std.fmt.allocPrint(self.allocator, "{s}.json", .{model_path});
        const voice = try piper.Voice.load(&self.rt, self.allocator, self.io, model_path, config_path);
        const key = try self.allocator.dupe(u8, voice_id);
        try self.piper_voices.put(key, voice);
        return self.piper_voices.getPtr(key).?;
    }

    /// Loads + warms every preset named in config.preload (one throwaway
    /// synthesis each), same mechanism validated in Phase 2.
    pub fn preload(self: *Daemon, log: *std.Io.Writer) !void {
        for (self.config.preload.items) |name| {
            const preset = self.config.getPreset(name) orelse {
                timing.logf(log, self.io, "[Daemon] preload skipped: preset '{s}' not found in config\n", .{name});
                continue;
            };
            timing.logf(log, self.io, "[Daemon] preloading '{s}' ({s})...\n", .{ name, preset.engine });
            const t0 = timing.elapsedSeconds(self.io);
            _ = try self.synthesizeAndPlay(preset, "Ready.", false, log);
            timing.logf(log, self.io, "[Daemon] '{s}' ready ({d:.2}s)\n", .{ name, timing.elapsedSeconds(self.io) - t0 });
        }
    }

    pub const SynthResult = struct { samples: []f32, sample_rate: u32 };

    /// Synthesizes `text` with `preset`, returning samples + rate allocated
    /// from `alloc` (caller-owned, e.g. for writing to a WAV file - unlike
    /// `synthesizeAndPlay`'s internal frame arena).
    ///
    /// If `log` is given, emits two engine-performance timestamps (server
    /// side, since the client never sees a stream of samples to measure
    /// this against - see the session notes on why client-side TTFB/TTLB
    /// didn't make sense for a non-streaming protocol):
    ///   - "phonemize done": G2P complete, about to run the ONNX model -
    ///     the closest analog to "time to first byte" this pipeline has.
    ///   - "synthesize done": the ONNX Run() call returned every sample at
    ///     once (Piper/Kokoro aren't streaming inference) - "time to last
    ///     byte", i.e. the actual engine/neural-net compute time.
    pub fn synthesize(self: *Daemon, alloc: std.mem.Allocator, preset: config_mod.VoicePreset, text: []const u8, log: ?*std.Io.Writer) !SynthResult {
        const t0 = timing.elapsedSeconds(self.io);
        if (std.mem.eql(u8, preset.engine, "kokoro")) {
            const voice = try self.getKokoroVoice();
            const phonemes = try self.kokoro_phonemizer.phonemize(alloc, text);
            if (log) |l| timing.logf(l, self.io, "[Engine] phonemize done ({d:.3}s)\n", .{timing.elapsedSeconds(self.io) - t0});
            const samples = try voice.synthesize(alloc, phonemes, preset.voice, preset.speed);
            if (log) |l| timing.logf(l, self.io, "[Engine] synthesize done ({d:.3}s)\n", .{timing.elapsedSeconds(self.io) - t0});
            return .{ .samples = samples, .sample_rate = 24000 };
        } else {
            const voice = try self.getPiperVoice(preset.voice);
            const ipa = try self.piper_phonemizer.plainIpa(alloc, text);
            if (log) |l| timing.logf(l, self.io, "[Engine] phonemize done ({d:.3}s)\n", .{timing.elapsedSeconds(self.io) - t0});
            const samples = try voice.synthesize(alloc, ipa, 1.0 / preset.speed);
            if (log) |l| timing.logf(l, self.io, "[Engine] synthesize done ({d:.3}s)\n", .{timing.elapsedSeconds(self.io) - t0});
            return .{ .samples = samples, .sample_rate = voice.config.sample_rate };
        }
    }

    /// Synthesizes `text` with `preset`, optionally playing it. Returns the
    /// sample count (for logging).
    pub fn synthesizeAndPlay(self: *Daemon, preset: config_mod.VoicePreset, text: []const u8, play: bool, log: ?*std.Io.Writer) !usize {
        var frame_arena = std.heap.ArenaAllocator.init(self.allocator);
        defer frame_arena.deinit();
        const alloc = frame_arena.allocator();

        const result = try self.synthesize(alloc, preset, text, log);
        if (play) try self.output.play(result.samples, result.sample_rate);
        return result.samples.len;
    }

    /// Single-threaded blocking accept loop on a unix socket. Protocol: one
    /// line per request, `preset<TAB>text\n`; replies `OK\n` or `ERR <msg>\n`.
    pub fn serve(self: *Daemon, socket_path: []const u8, log: *std.Io.Writer) !void {
        std.Io.Dir.cwd().deleteFile(self.io, socket_path) catch {};

        const addr = try std.Io.net.UnixAddress.init(socket_path);
        var server = try addr.listen(self.io, .{});
        defer server.socket.close(self.io);

        timing.logf(log, self.io, "[Daemon] Listening on unix://{s}\n", .{socket_path});
        timing.logf(log, self.io, "presence-voice ready. Use 'voice <preset> <text>' to synthesize.\n", .{});

        while (true) {
            var conn = server.accept(self.io) catch |err| {
                timing.logf(log, self.io, "[Daemon] accept error: {t}\n", .{err});
                continue;
            };
            defer conn.close(self.io);
            self.handleConnection(&conn, log) catch |err| {
                timing.logf(log, self.io, "[Daemon] request error: {t}\n", .{err});
            };
        }
    }

    fn handleConnection(self: *Daemon, conn: *std.Io.net.Stream, log: *std.Io.Writer) !void {
        var read_buf: [1 << 16]u8 = undefined;
        var reader = conn.reader(self.io, &read_buf);
        const line = try reader.interface.takeDelimiterExclusive('\n');

        const tab = std.mem.indexOfScalar(u8, line, '\t') orelse return error.BadRequest;
        const preset_name = line[0..tab];
        const text = line[tab + 1 ..];

        const preset = self.config.getPreset(preset_name) orelse {
            try writeResponse(conn, self.io, "ERR unknown preset\n");
            return;
        };

        const t0 = timing.elapsedSeconds(self.io);
        const n = try self.synthesizeAndPlay(preset, text, true, log);
        timing.logf(log, self.io, "[Daemon] {s}: {d} samples ({d:.2}s)\n", .{ preset_name, n, timing.elapsedSeconds(self.io) - t0 });
        try writeResponse(conn, self.io, "OK\n");
    }

    fn writeResponse(conn: *std.Io.net.Stream, io: std.Io, msg: []const u8) !void {
        var write_buf: [256]u8 = undefined;
        var writer = conn.writer(io, &write_buf);
        try writer.interface.writeAll(msg);
        try writer.interface.flush();
    }
};
