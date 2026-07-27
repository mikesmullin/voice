//! Kokoro engine: G2P front-end wired up (per tmp/PHASE3_PLAN.md section 2),
//! plus ONNX Runtime inference (milestone 2). Model I/O and the phoneme ->
//! token id / style-vector-bucket scheme were confirmed against the real
//! `kokoro` Python package's model.py/pipeline.py (read directly from its
//! installed site-packages, not re-derived from scratch):
//!   - tokens: int64 [1, N] = [0, *vocab[phoneme] for phoneme in ps, 0]
//!   - style: float32 [1, 256] = voices[name][len(ps)-1] (clamped to 509)
//!     from the (510, 1, 256) per-voice pack in kokoro-voices-*.bin (a
//!     plain uncompressed-zip .npz - see npz.zig)
//!   - speed: float32 [1]
//!   - output "audio": float32 1D waveform
//!
//! G2P is Fable's zig-phonemes (vendor/zig-phonemes): gold/silver lexicon,
//! optional espeak-ng OOV fallback (required so proper names are not silent),
//! plus optional ~/.config/voice/names.yaml overrides (src/names.zig).

const std = @import("std");
const zig_phonemes = @import("zig_phonemes");
const ort = @import("onnxruntime.zig");
const npz = @import("npz.zig");
const names = @import("../names.zig");

pub const G2P = zig_phonemes.G2P;
pub const Espeak = zig_phonemes.Espeak;

const ESPEAK_LIB_PATH = "/usr/lib/libespeak-ng.so";
const ESPEAK_DATA_PATH = "/usr/share/espeak-ng-data";

pub const Phonemizer = struct {
    g2p: G2P,
    /// Heap-stable so `g2p.espeak` can point at it for the process lifetime.
    espeak: ?*Espeak = null,
    /// How many names.yaml IPA entries were loaded (for logging).
    names_loaded: usize = 0,
    names_path: []const u8 = "",

    pub fn init(arena: std.mem.Allocator, io: std.Io, data_dir: []const u8, british: bool) !Phonemizer {
        var g2p = try G2P.init(arena, io, data_dir, british);

        // Custom IPA pronunciations win over dictionary + espeak.
        const names_path = try names.defaultPath(arena);
        names.ensureTemplate(io, names_path);
        const n_names = names.applyToLexicon(arena, io, names_path, &g2p.lexicon) catch 0;

        // espeak-ng OOV fallback — without this, unknown names phonemize to ""
        // and Kokoro speaks a gap ("…") instead of the word.
        var espeak_ptr: ?*Espeak = null;
        if (Espeak.init(arena, ESPEAK_LIB_PATH, ESPEAK_DATA_PATH, british)) |es| {
            const p = try arena.create(Espeak);
            p.* = es;
            g2p.espeak = p;
            espeak_ptr = p;
        } else |_| {
            espeak_ptr = null;
        }

        return .{
            .g2p = g2p,
            .espeak = espeak_ptr,
            .names_loaded = n_names,
            .names_path = names_path,
        };
    }

    pub fn phonemize(self: *const Phonemizer, arena: std.mem.Allocator, text: []const u8) ![]const u8 {
        return self.g2p.convert(arena, text);
    }

    pub fn hasEspeak(self: *const Phonemizer) bool {
        return self.espeak != null;
    }
};

pub const Vocab = struct {
    map: std.StringHashMap(i64),

    pub fn load(alloc: std.mem.Allocator, io: std.Io, vocab_path: []const u8) !Vocab {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, vocab_path, alloc, .limited(1 << 20));
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, bytes, .{});
        var map = std.StringHashMap(i64).init(alloc);
        var it = parsed.object.iterator();
        while (it.next()) |kv| {
            try map.put(kv.key_ptr.*, kv.value_ptr.*.integer);
        }
        return .{ .map = map };
    }

    /// [0, *vocab[phoneme] for phoneme in ps if present, 0] (BOS/EOS = the
    /// silence token 0, per kokoro/model.py's KModel.forward).
    pub fn tokensToIds(self: *const Vocab, alloc: std.mem.Allocator, phonemes: []const u8) ![]i64 {
        var ids: std.ArrayList(i64) = .empty;
        try ids.append(alloc, 0);
        var iter = std.unicode.Utf8View.initUnchecked(phonemes).iterator();
        while (iter.nextCodepointSlice()) |cp| {
            if (self.map.get(cp)) |id| try ids.append(alloc, id);
        }
        try ids.append(alloc, 0);
        return ids.toOwnedSlice(alloc);
    }
};

pub const Voice = struct {
    session: ort.Session,
    vocab: Vocab,
    voices_bin: []const u8,

    pub fn load(rt: *const ort.Runtime, alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, vocab_path: []const u8, voices_bin_path: []const u8) !Voice {
        return loadWithProvider(rt, alloc, io, model_path, vocab_path, voices_bin_path, true);
    }

    /// `try_cuda = false` forces CPU (the CLI's `-C/--cpu` flag, "local" only).
    pub fn loadWithProvider(rt: *const ort.Runtime, alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, vocab_path: []const u8, voices_bin_path: []const u8, try_cuda: bool) !Voice {
        const voices_bin = try std.Io.Dir.cwd().readFileAlloc(io, voices_bin_path, alloc, .limited(64 << 20));
        return .{
            .session = try ort.Session.load2(rt, alloc, model_path, try_cuda),
            .vocab = try Vocab.load(alloc, io, vocab_path),
            .voices_bin = voices_bin,
        };
    }

    /// Synthesize `phonemes` (already-phonemized text) using `voice_name`'s
    /// style pack, to f32 PCM samples at Kokoro's native 24000Hz.
    pub fn synthesize(self: *Voice, alloc: std.mem.Allocator, phonemes: []const u8, voice_name: []const u8, speed: f32) ![]f32 {
        const ids = try self.vocab.tokensToIds(alloc, phonemes);
        const num_phonemes = ids.len - 2; // excluding the two 0 (silence) tokens

        const pack = try npz.loadVoice(alloc, self.voices_bin, voice_name);
        const bucket = @min(if (num_phonemes == 0) 0 else num_phonemes - 1, pack.shape[0] - 1);
        const style = pack.data[bucket * 256 ..][0..256];

        const tokens_shape = [_]i64{ 1, @intCast(ids.len) };
        const tokens_value = try self.session.createInt64Tensor(alloc, ids, &tokens_shape);
        defer self.session.api.ReleaseValue.?(tokens_value);

        const style_shape = [_]i64{ 1, 256 };
        const style_value = try self.session.createF32Tensor(alloc, style, &style_shape);
        defer self.session.api.ReleaseValue.?(style_value);

        const speed_arr = [_]f32{speed};
        const speed_shape = [_]i64{1};
        const speed_value = try self.session.createF32Tensor(alloc, &speed_arr, &speed_shape);
        defer self.session.api.ReleaseValue.?(speed_value);

        const input_names = [_][:0]const u8{ "tokens", "style", "speed" };
        const inputs = [_]*ort.c.OrtValue{ tokens_value, style_value, speed_value };

        return self.session.runF32(alloc, &input_names, &inputs, "audio");
    }
};
