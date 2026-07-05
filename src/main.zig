//! presence-voice v2 - CLI entry point (scaffold, per tmp/PHASE3_PLAN.md section 6).
//! Not yet implemented: local/client/list/serve subcommands, config.yaml parsing,
//! ONNX Runtime inference. This is milestone 1's starting skeleton, now with a
//! `--phonemize` debug path wired to zig-phenomes' G2P (milestone 2, partial).

const std = @import("std");
const kokoro = @import("engines/kokoro.zig");
const piper = @import("engines/piper.zig");
const ort = @import("engines/onnxruntime.zig");
const wav = @import("audio/wav.zig");

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

    try stdout.print("presence-voice v2 (scaffold) - not yet implemented\n", .{});
    if (args.len > 1) {
        try stdout.print("args:", .{});
        for (args[1..]) |arg| try stdout.print(" {s}", .{arg});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
