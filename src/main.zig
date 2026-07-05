//! presence-voice v2 - CLI entry point (scaffold, per tmp/PHASE3_PLAN.md section 6).
//! Not yet implemented: local/client/list/serve subcommands, config.yaml parsing,
//! ONNX Runtime inference. This is milestone 1's starting skeleton, now with a
//! `--phonemize` debug path wired to zig-phenomes' G2P (milestone 2, partial).

const std = @import("std");
const kokoro = @import("engines/kokoro.zig");
const piper = @import("engines/piper.zig");

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

    try stdout.print("presence-voice v2 (scaffold) - not yet implemented\n", .{});
    if (args.len > 1) {
        try stdout.print("args:", .{});
        for (args[1..]) |arg| try stdout.print(" {s}", .{arg});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
