//! Kokoro engine: G2P front-end wired up (per tmp/PHASE3_PLAN.md section 2).
//! ONNX Runtime inference is not yet implemented (milestone 2) - this only
//! covers text -> phonemes -> Kokoro vocab token ids so far.
//!
//! G2P is provided by Fable's zig-phenomes (tmp/zig-phenomes), evaluated as
//! an alternative to plain espeak-ng. Referenced as a build.zig module
//! import ("zig_phenomes") pointed directly at that directory - not
//! vendored/copied into src/ yet, since it's still under active
//! development there. Do not edit tmp/zig-phenomes/ from this side; treat
//! it as a read-only dependency until the comparison in PHENOMES.md is
//! settled.

const std = @import("std");
const zig_phenomes = @import("zig_phenomes");

pub const G2P = zig_phenomes.G2P;

pub const Phonemizer = struct {
    g2p: G2P,

    pub fn init(arena: std.mem.Allocator, io: std.Io, data_dir: []const u8, british: bool) !Phonemizer {
        return .{ .g2p = try G2P.init(arena, io, data_dir, british) };
    }

    pub fn phonemize(self: *const Phonemizer, arena: std.mem.Allocator, text: []const u8) ![]const u8 {
        return self.g2p.convert(arena, text);
    }
};
