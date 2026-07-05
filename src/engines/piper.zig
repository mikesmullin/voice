//! Piper engine: G2P front-end wired up (per tmp/PHASE3_PLAN.md section 2 -
//! "Piper is low-risk: text -> espeak-ng phonemes -> phoneme-id lookup ->
//! ONNX model -> audio"). ONNX Runtime inference is not yet implemented
//! (milestone 2) - this only covers text -> raw espeak IPA so far.
//!
//! Reuses zig-phenomes' dlopen-based Espeak wrapper (tmp/zig-phenomes/src/
//! espeak.zig, re-exported via g2p.zig as `Espeak`) rather than adding a
//! separate link-time espeak-ng dependency: espeak-ng is loaded at runtime
//! via dlopen (see that file), so no `-lespeak-ng`/pkg-config wiring is
//! needed in build.zig - only the shared library + espeak-ng-data need to
//! exist on disk at runtime (already true here via the system `espeak-ng`
//! package).

const std = @import("std");
const zig_phenomes = @import("zig_phenomes");

pub const Espeak = zig_phenomes.Espeak;

pub const Phonemizer = struct {
    espeak: Espeak,

    pub fn init(alloc: std.mem.Allocator, lib_path: []const u8, data_path: []const u8, british: bool) !Phonemizer {
        return .{ .espeak = try Espeak.init(alloc, lib_path, data_path, british) };
    }

    /// Raw espeak IPA (tie-mode), the same representation Piper's own
    /// phonemize_espeak backend produces - not yet mapped through a voice's
    /// phoneme_id_map (needs ONNX Runtime + a loaded voice, milestone 2).
    pub fn rawIpa(self: *const Phonemizer, alloc: std.mem.Allocator, text: []const u8) ![]const u8 {
        return self.espeak.rawIpa(alloc, text);
    }
};
