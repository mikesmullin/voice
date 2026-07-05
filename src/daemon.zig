//! Daemon skeleton (milestone 3): unix socket + persistent audio output +
//! preload, per tmp/PHASE3_PLAN.md sections 4/8. Single-threaded, blocking
//! accept loop - concurrency/HTTP are later milestones (4/5).
//!
//! Known gaps (not yet covered by an approved config schema - see
//! tmp/PHASE3_PLAN.md's "Still open" items): Kokoro model/voices-bin paths
//! and the Piper models directory are hardcoded to this dev machine's
//! known file locations rather than read from config.yaml. Fine for a
//! skeleton; needs a real config field before this leaves this machine.

const std = @import("std");
const config_mod = @import("config.zig");
const kokoro = @import("engines/kokoro.zig");
const piper = @import("engines/piper.zig");
const ort = @import("engines/onnxruntime.zig");
const audio_output = @import("audio/output.zig");

// TODO: move to config.yaml once the v2 schema for model locations is
// decided (see tmp/PHASE3_PLAN.md "Still open").
const KOKORO_MODEL_PATH = "/workspace/Making_Games/GLaDOS/models/TTS/kokoro-v1.0.fp16.onnx";
const KOKORO_VOICES_BIN_PATH = "/workspace/Making_Games/GLaDOS/models/TTS/kokoro-voices-v1.0.bin";
const KOKORO_VOCAB_PATH = "tmp/zig-phenomes/data/kokoro_vocab.json";
const PIPER_MODELS_DIR = "/home/user/.cache/voice/piper-models";
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

    pub fn init(alloc: std.mem.Allocator, io: std.Io, config: config_mod.Config) !Daemon {
        return .{
            .allocator = alloc,
            .io = io,
            .config = config,
            .rt = try ort.Runtime.init(),
            .output = audio_output.Output.init(alloc),
            .kokoro_phonemizer = try kokoro.Phonemizer.init(alloc, io, "tmp/zig-phenomes/data", false),
            .piper_phonemizer = try piper.Phonemizer.init(alloc, ESPEAK_LIB_PATH, ESPEAK_DATA_PATH, false),
            .piper_voices = std.StringHashMap(piper.Voice).init(alloc),
        };
    }

    fn getKokoroVoice(self: *Daemon) !*kokoro.Voice {
        if (self.kokoro_voice == null) {
            self.kokoro_voice = try kokoro.Voice.load(&self.rt, self.allocator, self.io, KOKORO_MODEL_PATH, KOKORO_VOCAB_PATH, KOKORO_VOICES_BIN_PATH);
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
                try log.print("[Daemon] preload skipped: preset '{s}' not found in config\n", .{name});
                try log.flush();
                continue;
            };
            try log.print("[Daemon] preloading '{s}' ({s})...\n", .{ name, preset.engine });
            try log.flush();
            _ = try self.synthesizeAndPlay(preset, "Ready.", false);
        }
    }

    /// Synthesizes `text` with `preset`, optionally playing it. Returns the
    /// sample count (for logging).
    fn synthesizeAndPlay(self: *Daemon, preset: config_mod.VoicePreset, text: []const u8, play: bool) !usize {
        var frame_arena = std.heap.ArenaAllocator.init(self.allocator);
        defer frame_arena.deinit();
        const alloc = frame_arena.allocator();

        if (std.mem.eql(u8, preset.engine, "kokoro")) {
            const voice = try self.getKokoroVoice();
            const phonemes = try self.kokoro_phonemizer.phonemize(alloc, text);
            const samples = try voice.synthesize(alloc, phonemes, preset.voice, preset.speed);
            if (play) try self.output.play(samples, 24000);
            return samples.len;
        } else {
            const voice = try self.getPiperVoice(preset.voice);
            const ipa = try self.piper_phonemizer.plainIpa(alloc, text);
            const samples = try voice.synthesize(alloc, ipa, 1.0 / preset.speed);
            if (play) try self.output.play(samples, voice.config.sample_rate);
            return samples.len;
        }
    }

    /// Single-threaded blocking accept loop on a unix socket. Protocol: one
    /// line per request, `preset<TAB>text\n`; replies `OK\n` or `ERR <msg>\n`.
    pub fn serve(self: *Daemon, socket_path: []const u8, log: *std.Io.Writer) !void {
        std.Io.Dir.cwd().deleteFile(self.io, socket_path) catch {};

        const addr = try std.Io.net.UnixAddress.init(socket_path);
        var server = try addr.listen(self.io, .{});
        defer server.socket.close(self.io);

        try log.print("[Daemon] Listening on unix://{s}\n", .{socket_path});
        try log.print("presence-voice ready. Use 'voice <preset> <text>' to synthesize.\n", .{});
        try log.flush();

        while (true) {
            var conn = server.accept(self.io) catch |err| {
                try log.print("[Daemon] accept error: {t}\n", .{err});
                try log.flush();
                continue;
            };
            defer conn.close(self.io);
            self.handleConnection(&conn, log) catch |err| {
                try log.print("[Daemon] request error: {t}\n", .{err});
                try log.flush();
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

        const n = try self.synthesizeAndPlay(preset, text, true);
        try log.print("[Daemon] {s}: {d} samples\n", .{ preset_name, n });
        try log.flush();
        try writeResponse(conn, self.io, "OK\n");
    }

    fn writeResponse(conn: *std.Io.net.Stream, io: std.Io, msg: []const u8) !void {
        var write_buf: [256]u8 = undefined;
        var writer = conn.writer(io, &write_buf);
        try writer.interface.writeAll(msg);
        try writer.interface.flush();
    }
};
