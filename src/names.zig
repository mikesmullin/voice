//! Custom word pronunciations for Kokoro G2P (`~/.config/voice/names.yaml`).
//!
//! Entries are injected into the gold lexicon *before* espeak OOV fallback,
//! so a listed IPA always wins over dictionary and espeak guesses.
//!
//! File format (purpose-built mini-YAML — same spirit as config.zig):
//!   Word: "ipa"
//!   word: ipa
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
    \\#   Map written words (usually proper names) to Kokoro IPA phoneme strings.
    \\#   These override the built-in gold/silver dictionaries AND espeak-ng OOV
    \\#   fallback, so you can teach Ada how to say family names, brands, etc.
    \\#
    \\# HOW TO EDIT (for agentic coding assistants)
    \\#   1. Hear the current pronunciation (use Ada's preset — positional, not -p):
    \\#        voice --phonemize "Elise Smullin"
    \\#        voice nova "Her name is Elise Smullin."
    \\#      Ada uses preset `nova` (Kokoro af_nova). There is no -p flag;
    \\#      `voice -p nova "…"` falls back to default_preset/fallback (often lessac).
    \\#   2. If wrong, capture a better IPA (espeak OOV is on by default):
    \\#        voice --phonemize "Elise"
    \\#   3. Add/update a line below:
    \\#        Smullin: "smˈʌlᵻn"
    \\#   4. Restart the daemon so the file is re-read:
    \\#        systemctl --user restart presence-voice
    \\#   5. Play for Mike and ask for approval:
    \\#        voice nova "Smullin."
    \\#   6. Iterate on the IPA until approved; leave a trailing comment with date/note.
    \\#
    \\# SYNTAX
    \\#   Word: "ipa-string"
    \\#   - Keys are matched case-sensitively against G2P tokens (also store
    \\#     Capitalized and ALLCAPS variants if needed).
    \\#   - Values are Kokoro IPA (misaki/zig-phonemes alphabet), usually quoted.
    \\#   - One orthographic token per line (not multi-word keys).
    \\#   - Lines starting with # are comments; blank lines are fine.
    \\#
    \\# EXAMPLES
    \\# Smullin: "smˈʌlᵻn"
    \\
;

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

/// Load names.yaml and inject into the gold lexicon (highest priority).
/// Returns count of entries applied.
pub fn applyToLexicon(alloc: std.mem.Allocator, io: std.Io, path: []const u8, lexicon: *Lexicon) !usize {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20)) catch return 0;
    var count: usize = 0;
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
        // Ignore legacy say: lines if any remain in an old file
        if (std.ascii.startsWithIgnoreCase(val_raw, "say:")) continue;

        const key = try alloc.dupe(u8, key_raw);
        const ipa = try alloc.dupe(u8, val_raw);
        try lexicon.golds.put(key, .{ .str = ipa });
        count += 1;
    }
    return count;
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
