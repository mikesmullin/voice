//! Piper engine: G2P front-end wired up (per tmp/PHASE3_PLAN.md section 2 -
//! "Piper is low-risk: text -> espeak-ng phonemes -> phoneme-id lookup ->
//! ONNX model -> audio"). Milestone 2: ONNX Runtime inference now wired up
//! (CPU execution provider only so far).
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
const ort = @import("onnxruntime.zig");

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

    /// Plain (non-tie) espeak IPA - the mode Piper's own phoneme_id_map
    /// actually expects (confirmed no tie-joined entries in the voice
    /// configs on this machine). NOT NFD-normalized (unlike Piper's Python
    /// phonemize_espeak, which does `unicodedata.normalize("NFD", ...)`
    /// before splitting into codepoints) - Zig's std has no built-in
    /// Unicode decomposition, so a few precomposed diacritic combinations
    /// may come through as "missing phoneme" and get skipped. Good enough
    /// for a first working synth; revisit if it audibly matters.
    pub fn plainIpa(self: *const Phonemizer, alloc: std.mem.Allocator, text: []const u8) ![]const u8 {
        const PLAIN_MODE: c_int = 0x02; // espeakPHONEMES_IPA, no tie
        const text_z = try alloc.dupeSentinel(u8, text, 0);
        var out: std.ArrayList(u8) = .empty;
        var ptr: ?*const anyopaque = @ptrCast(text_z.ptr);
        while (ptr != null) {
            const part = self.espeak.text_to_phonemes(&ptr, 1, PLAIN_MODE) orelse break;
            const slice = std.mem.span(part);
            if (slice.len > 0) {
                if (out.items.len > 0) try out.append(alloc, ' ');
                try out.appendSlice(alloc, slice);
            }
        }
        return std.mem.trim(u8, out.items, " \t\r\n");
    }
};

pub const VoiceConfig = struct {
    sample_rate: u32,
    noise_scale: f32,
    length_scale: f32,
    noise_w_scale: f32,
    phoneme_id_map: std.StringHashMap(i64),

    pub fn load(alloc: std.mem.Allocator, io: std.Io, config_path: []const u8) !VoiceConfig {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path, alloc, .limited(1 << 20));
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, bytes, .{});
        const root = parsed.object;

        const audio = root.get("audio").?.object;
        const inference = root.get("inference").?.object;

        var map = std.StringHashMap(i64).init(alloc);
        var it = root.get("phoneme_id_map").?.object.iterator();
        while (it.next()) |kv| {
            const ids = kv.value_ptr.*.array;
            try map.put(kv.key_ptr.*, ids.items[0].integer);
        }

        return .{
            .sample_rate = @intCast(audio.get("sample_rate").?.integer),
            .noise_scale = numAsF32(inference.get("noise_scale").?),
            .length_scale = numAsF32(inference.get("length_scale").?),
            .noise_w_scale = numAsF32(inference.get("noise_w").?),
            .phoneme_id_map = map,
        };
    }

    fn numAsF32(v: std.json.Value) f32 {
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| @floatCast(f),
            else => 0,
        };
    }

    /// BOS/PAD-interleaved/EOS phoneme id sequence, per Piper's own
    /// phoneme_ids.py: [BOS, PAD] + for each phoneme: [id, PAD] + [EOS].
    pub fn phonemesToIds(self: *const VoiceConfig, alloc: std.mem.Allocator, ipa: []const u8) ![]i64 {
        var ids: std.ArrayList(i64) = .empty;
        const bos = self.phoneme_id_map.get("^").?;
        const pad = self.phoneme_id_map.get("_").?;
        const eos = self.phoneme_id_map.get("$").?;

        try ids.append(alloc, bos);
        try ids.append(alloc, pad);

        var iter = std.unicode.Utf8View.initUnchecked(ipa).iterator();
        while (iter.nextCodepointSlice()) |cp_slice| {
            if (self.phoneme_id_map.get(cp_slice)) |id| {
                try ids.append(alloc, id);
                try ids.append(alloc, pad);
            }
        }
        try ids.append(alloc, eos);
        return ids.toOwnedSlice(alloc);
    }
};

pub const Voice = struct {
    session: ort.Session,
    config: VoiceConfig,

    pub fn load(rt: *const ort.Runtime, alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, config_path: []const u8) !Voice {
        return .{
            // Piper's model is tiny and already realtime on CPU (per Fable's
            // tmp/onnx-cuda-lab/REPORT.md, GPU is where Kokoro's ~16x win
            // is - not worth it here), so skip the CUDA EP attempt.
            .session = try ort.Session.load2(rt, alloc, model_path, false),
            .config = try VoiceConfig.load(alloc, io, config_path),
        };
    }

    /// Synthesize `ipa` (already-phonemized text) to f32 PCM samples at
    /// self.config.sample_rate. Single-speaker only so far.
    pub fn synthesize(self: *Voice, alloc: std.mem.Allocator, ipa: []const u8, length_scale_override: ?f32) ![]f32 {
        const ids = try self.config.phonemesToIds(alloc, ipa);

        const input_shape = [_]i64{ 1, @intCast(ids.len) };
        const input_value = try self.session.createInt64Tensor(alloc, ids, &input_shape);
        defer self.session.api.ReleaseValue.?(input_value);

        const lengths = [_]i64{@intCast(ids.len)};
        const lengths_shape = [_]i64{1};
        const lengths_value = try self.session.createInt64Tensor(alloc, &lengths, &lengths_shape);
        defer self.session.api.ReleaseValue.?(lengths_value);

        const scales = [_]f32{
            self.config.noise_scale,
            length_scale_override orelse self.config.length_scale,
            self.config.noise_w_scale,
        };
        const scales_shape = [_]i64{3};
        const scales_value = try self.session.createF32Tensor(alloc, &scales, &scales_shape);
        defer self.session.api.ReleaseValue.?(scales_value);

        const input_names = [_][:0]const u8{ "input", "input_lengths", "scales" };
        const inputs = [_]*ort.c.OrtValue{ input_value, lengths_value, scales_value };

        return self.session.runF32(alloc, &input_names, &inputs, "output");
    }
};
