//! presence-voice v2 - CLI entry point (scaffold, per tmp/PHASE3_PLAN.md section 6).
//! Not yet implemented: local/client/list/serve subcommands, config.yaml parsing,
//! espeak-ng/ONNX Runtime engines. This is milestone 1's starting skeleton.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    const stdout = &stdout_writer.interface;

    try stdout.print("presence-voice v2 (scaffold) - not yet implemented\n", .{});
    if (args.len > 1) {
        try stdout.print("args:", .{});
        for (args[1..]) |arg| try stdout.print(" {s}", .{arg});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
