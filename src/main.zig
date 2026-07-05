//! presence-voice v2 - CLI entry point, per tmp/PHASE3_PLAN.md section 6.
//! `local`/`client`/`list`/`serve`/bare-shorthand + -o/-c/-i/-C/-g/-h are
//! implemented (milestone 5); `-s/--stinger` is parsed but not yet wired
//! up to anything. The various `--xyz-*` debug commands below predate the
//! real CLI and are kept for ad-hoc engine-level testing.

const std = @import("std");
const kokoro = @import("engines/kokoro.zig");
const piper = @import("engines/piper.zig");
const ort = @import("engines/onnxruntime.zig");
const wav = @import("audio/wav.zig");
const config_mod = @import("config.zig");
const audio_output = @import("audio/output.zig");
const daemon_mod = @import("daemon.zig");
const http_mod = @import("http.zig");
const cli = @import("cli.zig");

fn httpThread(d: *daemon_mod.Daemon, io: std.Io) void {
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    http_mod.serve(d, io, 3124, &w.interface) catch |err| {
        std.debug.print("[HTTP] fatal: {t}\n", .{err});
    };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    const stdout = &stdout_writer.interface;

    if (args.len > 1 and std.mem.eql(u8, args[1], "--phonemize")) {
        const text = if (args.len > 2) args[2] else "Hello world, this is a test.";
        const phonemizer = try kokoro.Phonemizer.init(arena, init.io, "tmp/zig-phenomes/data", false);
        try stdout.print("[Kokoro] G2P (zig-phenomes) loaded\n", .{});
        const phonemes = try phonemizer.phonemize(arena, text);
        try stdout.print("text:  {s}\nphon:  {s}\n", .{ text, phonemes });
        try stdout.flush();
        return;
    }

    if (args.len > 1 and std.mem.eql(u8, args[1], "--espeak-ipa")) {
        const text = if (args.len > 2) args[2] else "Hello world, this is a test.";
        const phonemizer = try piper.Phonemizer.init(arena, "/usr/lib/libespeak-ng.so", "/usr/share/espeak-ng-data", false);
        try stdout.print("[Piper] espeak-ng (dlopen) ready\n", .{});
        const ipa = try phonemizer.rawIpa(arena, text);
        try stdout.print("text:  {s}\nipa:   {s}\n", .{ text, ipa });
        try stdout.flush();
        return;
    }

    if (args.len > 4 and std.mem.eql(u8, args[1], "--piper-synth")) {
        const model_path = args[2];
        const text = args[3];
        const out_path = args[4];

        const phonemizer = try piper.Phonemizer.init(arena, "/usr/lib/libespeak-ng.so", "/usr/share/espeak-ng-data", false);
        const ipa = try phonemizer.plainIpa(arena, text);
        try stdout.print("[Piper] ipa: {s}\n", .{ipa});

        const config_path = try std.mem.concat(arena, u8, &.{ model_path, ".json" });
        const rt = try ort.Runtime.init();
        var voice = try piper.Voice.load(&rt, arena, init.io, model_path, config_path);
        try stdout.print("[Piper] model + config loaded ({d}Hz)\n", .{voice.config.sample_rate});

        const samples = try voice.synthesize(arena, ipa, null);
        try stdout.print("[Piper] synthesized {d} samples ({d:.2}s)\n", .{
            samples.len,
            @as(f64, @floatFromInt(samples.len)) / @as(f64, @floatFromInt(voice.config.sample_rate)),
        });

        try wav.writeMono16(init.io, out_path, voice.config.sample_rate, samples);
        try stdout.print("[Piper] wrote {s}\n", .{out_path});
        try stdout.flush();
        return;
    }

    if (args.len > 1 and std.mem.eql(u8, args[1], "--config-test")) {
        const config_path = if (args.len > 2) args[2] else "config.yaml";
        const cfg = try config_mod.Config.load(arena, init.io, config_path);
        try stdout.print("fallback_voice: {s}\n", .{cfg.fallback_voice orelse "(none)"});
        try stdout.print("default_preset: {s}\n", .{cfg.default_preset orelse "(none)"});
        try stdout.print("preload ({d}):\n", .{cfg.preload.items.len});
        for (cfg.preload.items) |name| try stdout.print("  - {s}\n", .{name});
        try stdout.print("voices ({d}):\n", .{cfg.voices.count()});
        var it = cfg.voices.iterator();
        var shown: usize = 0;
        while (it.next()) |kv| : (shown += 1) {
            if (shown >= 10) {
                try stdout.print("  ... ({d} more)\n", .{cfg.voices.count() - 10});
                break;
            }
            try stdout.print("  {s}: engine={s} voice={s} speed={d}\n", .{ kv.key_ptr.*, kv.value_ptr.engine, kv.value_ptr.voice, kv.value_ptr.speed });
        }
        try stdout.flush();
        return;
    }

    if (args.len > 6 and std.mem.eql(u8, args[1], "--kokoro-synth")) {
        const model_path = args[2];
        const voices_bin_path = args[3];
        const voice_name = args[4];
        const text = args[5];
        const out_path = args[6];

        const phonemizer = try kokoro.Phonemizer.init(arena, init.io, "tmp/zig-phenomes/data", false);
        const phonemes = try phonemizer.phonemize(arena, text);
        try stdout.print("[Kokoro] phonemes: {s}\n", .{phonemes});

        const rt = try ort.Runtime.init();
        var voice = try kokoro.Voice.load(&rt, arena, init.io, model_path, "tmp/zig-phenomes/data/kokoro_vocab.json", voices_bin_path);
        try stdout.print("[Kokoro] model + voices pack loaded\n", .{});

        const samples = try voice.synthesize(arena, phonemes, voice_name, 1.0);
        const sample_rate: u32 = 24000;
        try stdout.print("[Kokoro] synthesized {d} samples ({d:.2}s)\n", .{
            samples.len,
            @as(f64, @floatFromInt(samples.len)) / @as(f64, @floatFromInt(sample_rate)),
        });

        try wav.writeMono16(init.io, out_path, sample_rate, samples);
        try stdout.print("[Kokoro] wrote {s}\n", .{out_path});
        try stdout.flush();
        return;
    }

    if (args.len > 4 and std.mem.eql(u8, args[1], "--kokoro-play")) {
        const model_path = args[2];
        const voices_bin_path = args[3];
        const voice_name = args[4];
        const text = if (args.len > 5) args[5] else "Hello world, this is a test.";

        const phonemizer = try kokoro.Phonemizer.init(arena, init.io, "tmp/zig-phenomes/data", false);
        const phonemes = try phonemizer.phonemize(arena, text);

        var out = audio_output.Output.init(arena);
        const rt = try ort.Runtime.init();
        var voice = try kokoro.Voice.load(&rt, arena, init.io, model_path, "tmp/zig-phenomes/data/kokoro_vocab.json", voices_bin_path);

        const samples = try voice.synthesize(arena, phonemes, voice_name, 1.0);
        try stdout.print("[Kokoro] synthesized {d} samples\n", .{samples.len});

        try out.play(samples, 24000);
        out.drain(24000);
        try stdout.print("[Audio] played through persistent PulseAudio stream\n", .{});
        try stdout.flush();
        return;
    }

    if (args.len > 1 and std.mem.eql(u8, args[1], "serve")) {
        var http_enabled = false;
        var config_path: []const u8 = "config.yaml";
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "--http")) {
                http_enabled = true;
            } else {
                config_path = a;
            }
        }

        const cfg = try config_mod.Config.load(arena, init.io, config_path);
        var d = try daemon_mod.Daemon.init(arena, init.io, cfg);
        try d.preload(stdout);
        try stdout.print("[Daemon] Preload complete, engines ready\n", .{});
        try stdout.flush();

        if (http_enabled) {
            _ = try std.Thread.spawn(.{}, httpThread, .{ &d, init.io });
        }
        try d.serve("/tmp/presence-voice.sock", stdout);
        return;
    }

    // "list" is its own subcommand (not a flag) - section 6 decision.
    if (args.len > 1 and std.mem.eql(u8, args[1], "list")) {
        const opts = try cli.parseOptions(arena, args[2..]);
        const cfg = try config_mod.Config.load(arena, init.io, opts.config_path);
        var it = cfg.voices.iterator();
        while (it.next()) |kv| {
            try stdout.print("{s: <12} {s: <8} {s}\n", .{ kv.key_ptr.*, kv.value_ptr.engine, kv.value_ptr.voice });
        }
        try stdout.flush();
        return;
    }

    // "local" / "client" / bare shorthand (for "client") - section 6.
    const is_local = args.len > 1 and std.mem.eql(u8, args[1], "local");
    const is_explicit_client = args.len > 1 and std.mem.eql(u8, args[1], "client");
    const rest = if (is_local or is_explicit_client) args[2..] else args[1..];

    const opts = try cli.parseOptions(arena, rest);

    if (opts.help) {
        try stdout.print(cli.HELP_TEXT, .{});
        try stdout.flush();
        return;
    }

    const cfg = try config_mod.Config.load(arena, init.io, opts.config_path);

    if (opts.info) |preset_name| {
        const preset = cfg.getPreset(preset_name) orelse {
            try stdout.print("error: no such preset '{s}'\n", .{preset_name});
            try stdout.flush();
            std.process.exit(1);
        };
        try stdout.print("Preset:  {s}\nEngine:  {s}\nVoice:   {s}\nSpeed:   {d}\n", .{ preset_name, preset.engine, preset.voice, preset.speed });
        try stdout.flush();
        return;
    }

    const resolved = try cli.resolvePresetAndText(arena, &cfg, opts.positionals.items) orelse {
        try stdout.print(cli.HELP_TEXT, .{});
        try stdout.flush();
        std.process.exit(2);
    };
    const preset = cfg.getPreset(resolved.preset_name) orelse {
        try stdout.print("error: no such preset '{s}'\n", .{resolved.preset_name});
        try stdout.flush();
        std.process.exit(1);
    };

    if (is_local) {
        var d = try daemon_mod.Daemon.init(arena, init.io, cfg);
        d.force_cpu = opts.cpu;
        const result = try d.synthesize(arena, preset, resolved.text);
        cli.applyGain(result.samples, opts.gain);

        if (opts.output) |out_path| {
            try wav.writeMono16(init.io, out_path, result.sample_rate, result.samples);
            try stdout.print("[Voice] Saved to: {s}\n", .{out_path});
        } else {
            var out = audio_output.Output.init(arena);
            try out.play(result.samples, result.sample_rate);
            out.drain(result.sample_rate);
        }
        try stdout.flush();
        return;
    }

    // "client" (explicit or bare shorthand): fails fast if the daemon isn't
    // reachable, per the "no more implicit fallback" decision - never
    // silently falls back to "local".
    if (opts.output != null or opts.gain != 1.0) {
        try stdout.print("error: -o/--output and -g/--gain are not yet supported for client/bare requests (local only so far)\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }

    const socket_path = "/tmp/presence-voice.sock";
    const addr = std.Io.net.UnixAddress.init(socket_path) catch unreachable;
    var conn = addr.connect(init.io) catch {
        try stdout.print("error: presence-voice daemon is not reachable\n       (unix://{s})\n       Start it with: voice serve\n", .{socket_path});
        try stdout.flush();
        std.process.exit(1);
    };
    defer conn.close(init.io);

    var write_buf: [4096]u8 = undefined;
    var writer = conn.writer(init.io, &write_buf);
    try writer.interface.print("{s}\t{s}\n", .{ resolved.preset_name, resolved.text });
    try writer.interface.flush();

    var read_buf: [4096]u8 = undefined;
    var reader = conn.reader(init.io, &read_buf);
    const line = try reader.interface.takeDelimiterExclusive('\n');
    if (std.mem.startsWith(u8, line, "ERR")) {
        try stdout.print("error: {s}\n", .{line});
        try stdout.flush();
        std.process.exit(1);
    }
}
