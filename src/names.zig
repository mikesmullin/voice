//! Custom word pronunciations for Kokoro G2P (`~/.config/voice/names.yaml`).
//!
//! Two value forms:
//!   1. IPA string — injected into the gold lexicon (wins over espeak OOV).
//!   2. `say:E Lisa` — orthographic rewrite *before* G2P (word → phrase).
//!      Use this when a name should be spoken as separate known words
//!      (e.g. Elisa → "E Lisa" so letter-E + name Lisa stay distinct).
//!
//! File format (purpose-built mini-YAML — same spirit as config.zig):
//!   Word: "ipa"
//!   Word: "say:E Lisa"
//! Blank lines and `#` comments (full-line or trailing) are ignored.

const std = @import("std");
const zig_phonemes = @import("zig_phonemes");

const Lexicon = zig_phonemes.Lexicon;

/// Default path: `$XDG_CONFIG_HOME/voice/names.yaml` or `~/.config/voice/names.yaml`.
/// Override with env `VOICE_NAMES_YAML`.
fn envOrNull(key: [*:0]const u8) ?[]const u8 {
    const p = std.c.getenv(key) orelse return null;
    return std.mem.span(p);
}

pub fn defaultPath(alloc: std.mem.Allocator) ![]const u8 {
    if (envOrNull("VOICE_NAMES_YAML")) |p| {
        return try alloc.dupe(u8, p);
    }
    const home = envOrNull("HOME") orelse "/tmp";
    if (envOrNull("XDG_CONFIG_HOME")) |base| {
        return try std.fmt.allocPrint(alloc, "{s}/voice/names.yaml", .{base});
    }
    return try std.fmt.allocPrint(alloc, "{s}/.config/voice/names.yaml", .{home});
}

pub const NAMES_YAML_TEMPLATE =
    \\# voice names.yaml — custom Kokoro pronunciations
    \\#
    \\# PURPOSE
    \\#   Teach Ada how to say names/brands that the default G2P mangles.
    \\#
    \\# TWO VALUE FORMS
    \\#   1) IPA (single token stays one word):
    \\#        Smullin: "smˈʌlᵻn"
    \\#   2) Orthographic rewrite (before G2P) — use when you want *two*
    \\#      known words spoken clearly, e.g. letter-E + the name Lisa:
    \\#        Elisa: "say:E Lisa"
    \\#      That expands "Elisa" → "E Lisa" so you get ˈi + lˈisə, not a
    \\#      collapsed "uh-lissa" monoword.
    \\#
    \\# HOW TO EDIT (for agentic coding assistants)
    \\#   1. Inspect:
    \\#        voice --phonemize "Elisa"
    \\#   2. Prefer `say:…` when the name is clearly compound (E + Lisa).
    \\#      Prefer IPA when it's one orthographic unit to fine-tune.
    \\#   3. Edit this file; case-sensitive keys (store Elisa/ELISA/elisa).
    \\#   4. Restart: systemctl --user restart presence-voice
    \\#   5. Play for Mike and iterate until approved.
    \\#
    \\# SYNTAX
    \\#   Key: "value"     # value = IPA  OR  say:<words>
    \\#   # comments and blank lines ok
    \\
;

/// Whole-word rewrites (key → replacement phrase), applied before G2P.
pub const RewriteMap = std.StringHashMap([]const u8);

pub const LoadResult = struct {
    ipa_count: usize = 0,
    say_count: usize = 0,
    rewrites: RewriteMap,
};

/// Ensure the config file exists with the documented template (no overwrite).
pub fn ensureTemplate(io: std.Io, path: []const u8) void {
    if (std.Io.Dir.accessAbsolute(io, path, .{})) |_| {
        return;
    } else |_| {}

    if (std.fs.path.dirname(path)) |dir| {
        mkdirP(dir);
    }
    const file = std.Io.Dir.createFileAbsolute(io, path, .{}) catch return;
    defer file.close(io);
    file.writeStreamingAll(io, NAMES_YAML_TEMPLATE) catch {};
}

fn mkdirP(dir: []const u8) void {
    var buf: [512]u8 = undefined;
    if (dir.len == 0 or dir.len >= buf.len) return;
    var i: usize = if (dir[0] == '/') 1 else 0;
    while (i <= dir.len) : (i += 1) {
        if (i != dir.len and dir[i] != '/') continue;
        if (i == 0) continue;
        @memcpy(buf[0..i], dir[0..i]);
        buf[i] = 0;
        _ = std.c.mkdir(@ptrCast(&buf), 0o755);
        if (i == dir.len) break;
    }
}

/// Load names.yaml: IPA → gold lexicon; `say:…` → rewrite map.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8, lexicon: *Lexicon) !LoadResult {
    var rewrites = RewriteMap.init(alloc);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20)) catch {
        return .{ .rewrites = rewrites };
    };
    var ipa_count: usize = 0;
    var say_count: usize = 0;
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw| {
        const line = trimCommentAndSpace(raw);
        if (line.len == 0) continue;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        if (colon == 0) continue;
        const key_raw = std.mem.trim(u8, line[0..colon], " \t");
        var val_raw = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (key_raw.len == 0 or val_raw.len == 0) continue;
        if (val_raw.len >= 2 and ((val_raw[0] == '"' and val_raw[val_raw.len - 1] == '"') or
            (val_raw[0] == '\'' and val_raw[val_raw.len - 1] == '\'')))
        {
            val_raw = val_raw[1 .. val_raw.len - 1];
        }
        if (val_raw.len == 0) continue;

        const key = try alloc.dupe(u8, key_raw);
        // say:… → orthographic rewrite (before G2P)
        if (std.ascii.startsWithIgnoreCase(val_raw, "say:")) {
            const phrase = std.mem.trim(u8, val_raw["say:".len..], " \t");
            if (phrase.len == 0) continue;
            try rewrites.put(key, try alloc.dupe(u8, phrase));
            say_count += 1;
            continue;
        }
        // else IPA → gold lexicon
        const ipa = try alloc.dupe(u8, val_raw);
        try lexicon.golds.put(key, .{ .str = ipa });
        ipa_count += 1;
    }
    return .{ .ipa_count = ipa_count, .say_count = say_count, .rewrites = rewrites };
}

/// Replace whole-word keys (letter/digit/' runs) with their `say:` phrases.
pub fn applyRewrites(alloc: std.mem.Allocator, text: []const u8, rewrites: *const RewriteMap) ![]const u8 {
    if (rewrites.count() == 0) return text;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);

    var i: usize = 0;
    while (i < text.len) {
        // non-word run
        const start = i;
        while (i < text.len and !isWordByte(text[i])) : (i += 1) {}
        if (i > start) try out.appendSlice(alloc, text[start..i]);
        if (i >= text.len) break;

        // word run
        const w0 = i;
        while (i < text.len and isWordByte(text[i])) : (i += 1) {}
        const word = text[w0..i];
        if (rewrites.get(word)) |phrase| {
            try out.appendSlice(alloc, phrase);
        } else {
            try out.appendSlice(alloc, word);
        }
    }
    return try out.toOwnedSlice(alloc);
}

fn isWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '\'' or c == '’';
}

fn trimCommentAndSpace(raw: []const u8) []const u8 {
    var s = std.mem.trim(u8, raw, " \t\r");
    if (s.len == 0 or s[0] == '#') return "";
    var in_dquote = false;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (c == '"' and (i == 0 or s[i - 1] != '\\')) in_dquote = !in_dquote;
        if (c == '#' and !in_dquote) {
            s = std.mem.trimEnd(u8, s[0..i], " \t");
            break;
        }
    }
    return s;
}
