//! Minimal, purpose-built parser for this project's config.yaml shape -
//! NOT a general YAML parser (per Phase 3 plan decisions log #3: "keep
//! config.yaml - will need a YAML parser in Zig"). Handles exactly the
//! subset actually used here:
//!   - top-level `key: value` scalars (e.g. `fallback_voice: lessac`)
//!   - a top-level `key:` list block (e.g. `preload:` + `  - item`)
//!   - a top-level `key:` dict-of-dicts block (`voices:` + per-voice
//!     `  name:` + `    field: value` entries)
//!   - `#` line and inline comments (outside double quotes)
//!   - double-quoted string values (quotes stripped)
//! Blank lines and extra indentation quirks beyond this shape are not
//! handled - if config.yaml's structure changes materially (e.g. the
//! weight/notes fields from Phase 3 plan section 7), this will need
//! updating alongside it.

const std = @import("std");

pub const VoicePreset = struct {
    engine: []const u8 = "kokoro",
    voice: []const u8 = "",
    speed: f32 = 1.0,
};

pub const Config = struct {
    fallback_voice: ?[]const u8 = null,
    default_preset: ?[]const u8 = null,
    preload: std.ArrayList([]const u8) = .empty,
    voices: std.StringHashMap(VoicePreset),

    pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Config {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20));

        var config: Config = .{ .voices = std.StringHashMap(VoicePreset).init(alloc) };

        const Block = enum { none, preload, voices };
        var block: Block = .none;
        var current_voice_name: ?[]const u8 = null;
        var current_voice: VoicePreset = .{};

        var line_iter = std.mem.splitScalar(u8, bytes, '\n');
        while (line_iter.next()) |raw_line| {
            const line = stripComment(raw_line);
            if (std.mem.trim(u8, line, " \t\r").len == 0) continue;

            const indent = indentOf(line);
            const content = std.mem.trim(u8, line, " \t\r");

            if (indent == 0) {
                // Flush any in-progress voice entry before switching blocks.
                if (current_voice_name) |name| {
                    try config.voices.put(name, current_voice);
                    current_voice_name = null;
                    current_voice = .{};
                }

                if (parseKeyValue(content)) |kv| {
                    if (std.mem.eql(u8, kv.key, "fallback_voice")) {
                        config.fallback_voice = try alloc.dupe(u8, unquote(kv.value));
                    } else if (std.mem.eql(u8, kv.key, "default_preset")) {
                        config.default_preset = try alloc.dupe(u8, unquote(kv.value));
                    }
                    block = .none;
                } else if (parseBlockKey(content)) |key| {
                    if (std.mem.eql(u8, key, "preload")) {
                        block = .preload;
                    } else if (std.mem.eql(u8, key, "voices")) {
                        block = .voices;
                    } else {
                        block = .none;
                    }
                }
                continue;
            }

            switch (block) {
                .preload => {
                    if (std.mem.startsWith(u8, content, "- ") or std.mem.eql(u8, content, "-")) {
                        const item = std.mem.trim(u8, content[1..], " \t");
                        try config.preload.append(alloc, try alloc.dupe(u8, unquote(item)));
                    }
                },
                .voices => {
                    if (indent <= 2) {
                        // New voice name (a bare "name:" block key at 1-level nesting).
                        if (current_voice_name) |name| {
                            try config.voices.put(name, current_voice);
                            current_voice = .{};
                        }
                        if (parseBlockKey(content)) |name| {
                            current_voice_name = try alloc.dupe(u8, name);
                        }
                    } else if (current_voice_name != null) {
                        if (parseKeyValue(content)) |kv| {
                            if (std.mem.eql(u8, kv.key, "engine")) {
                                current_voice.engine = try alloc.dupe(u8, unquote(kv.value));
                            } else if (std.mem.eql(u8, kv.key, "voice")) {
                                current_voice.voice = try alloc.dupe(u8, unquote(kv.value));
                            } else if (std.mem.eql(u8, kv.key, "speed")) {
                                current_voice.speed = std.fmt.parseFloat(f32, unquote(kv.value)) catch 1.0;
                            }
                        }
                    }
                },
                .none => {},
            }
        }
        if (current_voice_name) |name| {
            try config.voices.put(name, current_voice);
        }

        return config;
    }

    pub fn getPreset(self: *const Config, name: []const u8) ?VoicePreset {
        return self.voices.get(name);
    }
};

fn indentOf(line: []const u8) usize {
    var n: usize = 0;
    for (line) |ch| {
        if (ch == ' ') n += 1 else break;
    }
    return n;
}

/// Strips a trailing `# comment`, but only outside double-quoted spans.
fn stripComment(line: []const u8) []const u8 {
    var in_quotes = false;
    for (line, 0..) |ch, i| {
        if (ch == '"') in_quotes = !in_quotes;
        if (ch == '#' and !in_quotes) return line[0..i];
    }
    return line;
}

fn unquote(s: []const u8) []const u8 {
    const t = std.mem.trim(u8, s, " \t");
    if (t.len >= 2 and t[0] == '"' and t[t.len - 1] == '"') return t[1 .. t.len - 1];
    return t;
}

const KV = struct { key: []const u8, value: []const u8 };

/// `key: value` (value non-empty after trimming).
fn parseKeyValue(content: []const u8) ?KV {
    const colon = std.mem.indexOfScalar(u8, content, ':') orelse return null;
    const key = std.mem.trim(u8, content[0..colon], " \t");
    const value = std.mem.trim(u8, content[colon + 1 ..], " \t");
    if (value.len == 0) return null;
    if (!isIdentifier(key)) return null;
    return .{ .key = key, .value = value };
}

/// `key:` with nothing (a block-opening key).
fn parseBlockKey(content: []const u8) ?[]const u8 {
    if (content.len == 0 or content[content.len - 1] != ':') return null;
    const key = std.mem.trim(u8, content[0 .. content.len - 1], " \t");
    if (!isIdentifier(key)) return null;
    return key;
}

fn isIdentifier(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |ch| {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '-')) return false;
    }
    return true;
}
