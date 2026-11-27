//! Minimal, purpose-built parser for this project's config.yaml shape -
//! NOT a general YAML parser (per Phase 3 plan decisions log #3: "keep
//! config.yaml - will need a YAML parser in Zig"). Handles exactly the
//! subset actually used here:
//!   - top-level `key: value` scalars (e.g. `fallback_voice: lessac`)
//!   - a top-level `key:` list block (e.g. `preload:` + `  - item`)
//!   - a top-level `key:` dict-of-dicts block (`voices:` + per-voice
//!     `  name:` + `    field: value` entries)
//!   - a top-level `key:` flat dict-of-scalars block (`speakers:` +
//!     `  alias: raw_name` entries - one level of nesting, unlike
//!     `voices:`'s two)
//!   - a top-level `key:` dict-of-(list-of-one-key-maps) block
//!     (`effects:` + per-preset `  name:` + `    chain:` +
//!     `      - step_kind:` + `          field: value` entries, PLUS an
//!     optional sibling `    background:` + `      volume: value` +
//!     `      sources:` + `        - file: "path"` entries - see
//!     tmp/FUN_PLAN.md section 2's "Parser impact" subsection). Deepest/
//!     newest addition here; deliberately hardcoded to these two known
//!     shapes rather than a general nested-value parser.
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
    /// System prompt for the personality-prompt feature (tmp/FUN_PLAN.md
    /// section 4) - when set, `Daemon.synthesize()` rewrites the input
    /// text through the LLM at `Config.llm_completion_url` with this as
    /// the system message, and speaks the *response* instead. `null`
    /// (the default) means no rewrite, no added latency - opt-in only.
    personality_prompt: ?[]const u8 = null,
    /// Optional sibling transcript of the `speaker` WAV (qwen: sent as
    /// `speaker_ref_text` so the clone has text to anchor on; generated
    /// once with whisper, e.g. alongside the WAV). Missing/unreadable =
    /// omitted, the clone still works.
    speaker_text: ?[]const u8 = null,
    /// Voice-clone source for engines that need one (qwen: the speaker
    /// WAV that gets cloned, a path). `null` (the default) means the
    /// engine doesn't use it (kokoro/piper/audio8 use `voice` as their
    /// voice id instead).
    speaker: ?[]const u8 = null,
    /// Preferred compute backend for engines that have two (audio8, qwen):
    /// "cpu" or "gpu". `null` (the default) means CPU. This is the single
    /// source of truth - the removed --gpu/--cpu CLI flags error out with
    /// a pointer here. Ignored by kokoro/piper (no CUDA backend switch).
    backend: ?[]const u8 = null,
    /// Default effect presets applied to every utterance from this preset
    /// (config.yaml `effects: [name1, name2]` - inline list). Request-
    /// provided `-e`/HTTP effects are appended after these (applied in
    /// order; later effects win backgrounds/chains).
    effects: std.ArrayList([]const u8) = .empty,
};

/// One step in an effect preset's `chain:` (tmp/FUN_PLAN.md section 2),
/// e.g. `distortion: { drive: 2.8 }` becomes `.{ .kind = "distortion",
/// .params = {"drive": "2.8"} }` - params are kept as raw strings; the
/// DSP code that consumes a step parses whichever fields it expects.
pub const EffectStep = struct {
    kind: []const u8,
    params: std.StringHashMap([]const u8),
};

pub const EffectPreset = struct {
    chain: std.ArrayList(EffectStep) = .empty,
    background: Background = .{},
};

/// A preset's `background:` (tmp/FUN_PLAN.md section 2) - a looping
/// ambience mixed under the voice. `sources` are relative file paths as
/// written in config.yaml (resolved to absolute paths by the caller that
/// reads them, same convention as a `stinger` step's `file:`). v1 is
/// deliberately simple: exactly one source is picked at random each time
/// the preset is used (no `pick: all`/atlas region-picking yet).
pub const Background = struct {
    volume: f32 = 0.25,
    sources: std.ArrayList([]const u8) = .empty,
    /// When true, playback starts at a random point into the file instead
    /// of at the beginning (off on by default - the same music start every
    /// time is usually not what you want).
    random_start: bool = false,
};

pub const Config = struct {
    fallback_voice: ?[]const u8 = null,
    default_preset: ?[]const u8 = null,
    preload: std.ArrayList([]const u8) = .empty,
    voices: std.StringHashMap(VoicePreset),
    /// alias -> raw PulseAudio/PipeWire sink name (tmp/FUN_PLAN.md
    /// section 1) - Linux-only feature, but the config block itself is
    /// harmless to parse on any platform.
    speakers: std.StringHashMap([]const u8),
    /// name -> effect preset (tmp/FUN_PLAN.md section 2).
    effects: std.StringHashMap(EffectPreset),
    /// LM Studio's local server (tmp/FUN_PLAN.md section 4) - defaults
    /// applied by the caller (`daemon.zig`), not here, so a `config.yaml`
    /// with no `llm:` block at all still behaves sensibly.
    llm_completion_url: ?[]const u8 = null,
    llm_model: ?[]const u8 = null,
    /// Reasoning level for hybrid-reasoning local models (wired to the
    /// request's `reasoning_effort`; "low"/"medium"/"high"). `null` omits
    /// the field entirely.
    llm_reasoning_effort: ?[]const u8 = null,

    pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Config {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20));

        var config: Config = .{
            .voices = std.StringHashMap(VoicePreset).init(alloc),
            .speakers = std.StringHashMap([]const u8).init(alloc),
            .effects = std.StringHashMap(EffectPreset).init(alloc),
        };

        const Block = enum { none, preload, voices, speakers, effects, llm };
        var block: Block = .none;
        var current_voice_name: ?[]const u8 = null;
        var current_voice: VoicePreset = .{};

        // `effects:` sub-parser state (see the "dict-of-(list-of-one-key-
        // maps)" doc comment above) - reset whenever a new preset starts.
        var current_effect_name: ?[]const u8 = null;
        var current_effect: EffectPreset = .{};
        var effect_preset_indent: ?usize = null;
        // Which sibling section (`chain:` or `background:`) we're
        // currently inside, and the indent those keyword lines sit at.
        const EffectSection = enum { none, chain, background };
        var effect_section: EffectSection = .none;
        var effect_section_indent: ?usize = null;
        // `chain:` state.
        var chain_item_indent: ?usize = null;
        var current_step_kind: ?[]const u8 = null;
        var current_step_params: std.StringHashMap([]const u8) = undefined;
        // `background:` state.
        var bg_field_indent: ?usize = null;
        var bg_in_sources: bool = false;
        var bg_source_indent: ?usize = null;

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
                // Flush any in-progress effect preset (+ its in-progress step).
                if (current_step_kind) |kind| {
                    try current_effect.chain.append(alloc, .{ .kind = kind, .params = current_step_params });
                    current_step_kind = null;
                }
                if (current_effect_name) |name| {
                    try config.effects.put(name, current_effect);
                    current_effect_name = null;
                    current_effect = .{};
                }
                effect_preset_indent = null;
                chain_item_indent = null;
                effect_section = .none;
                effect_section_indent = null;
                bg_field_indent = null;
                bg_in_sources = false;
                bg_source_indent = null;

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
                    } else if (std.mem.eql(u8, key, "speakers")) {
                        block = .speakers;
                    } else if (std.mem.eql(u8, key, "effects")) {
                        block = .effects;
                    } else if (std.mem.eql(u8, key, "llm")) {
                        block = .llm;
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
                            } else if (std.mem.eql(u8, kv.key, "personality_prompt")) {
                                current_voice.personality_prompt = try alloc.dupe(u8, unquote(kv.value));
                            } else if (std.mem.eql(u8, kv.key, "speaker")) {
                                current_voice.speaker = try alloc.dupe(u8, unquote(kv.value));
                            } else if (std.mem.eql(u8, kv.key, "speaker_text")) {
                                current_voice.speaker_text = try alloc.dupe(u8, unquote(kv.value));
                            } else if (std.mem.eql(u8, kv.key, "backend")) {
                                current_voice.backend = try alloc.dupe(u8, unquote(kv.value));
                            } else if (std.mem.eql(u8, kv.key, "effects")) {
                                try parseInlineList(alloc, kv.value, &current_voice.effects);
                            }
                        }
                    }
                },
                .speakers => {
                    if (parseKeyValue(content)) |kv| {
                        try config.speakers.put(try alloc.dupe(u8, kv.key), try alloc.dupe(u8, unquote(kv.value)));
                    }
                },
                .llm => {
                    if (parseKeyValue(content)) |kv| {
                        if (std.mem.eql(u8, kv.key, "completion_url")) {
                            config.llm_completion_url = try alloc.dupe(u8, unquote(kv.value));
                        } else if (std.mem.eql(u8, kv.key, "model")) {
                            config.llm_model = try alloc.dupe(u8, unquote(kv.value));
                        } else if (std.mem.eql(u8, kv.key, "reasoning_effort")) {
                            config.llm_reasoning_effort = try alloc.dupe(u8, unquote(kv.value));
                        }
                    }
                },
                .effects => {
                    if (effect_preset_indent == null) effect_preset_indent = indent;

                    if (indent == effect_preset_indent.?) {
                        // New preset name - flush the previous one (+ its step).
                        if (current_step_kind) |kind| {
                            try current_effect.chain.append(alloc, .{ .kind = kind, .params = current_step_params });
                            current_step_kind = null;
                        }
                        if (current_effect_name) |name| {
                            try config.effects.put(name, current_effect);
                        }
                        current_effect = .{};
                        chain_item_indent = null;
                        effect_section = .none;
                        effect_section_indent = null;
                        bg_field_indent = null;
                        bg_in_sources = false;
                        bg_source_indent = null;
                        if (parseBlockKey(content)) |name| {
                            current_effect_name = try alloc.dupe(u8, name);
                        } else {
                            current_effect_name = null;
                        }
                    } else if (current_effect_name != null) {
                        if (effect_section_indent == null) effect_section_indent = indent;

                        if (indent == effect_section_indent.?) {
                            // New sibling section under this preset - flush
                            // whichever section we were just in.
                            if (current_step_kind) |kind| {
                                try current_effect.chain.append(alloc, .{ .kind = kind, .params = current_step_params });
                                current_step_kind = null;
                            }
                            chain_item_indent = null;
                            bg_field_indent = null;
                            bg_in_sources = false;
                            bg_source_indent = null;

                            if (parseBlockKey(content)) |key| {
                                if (std.mem.eql(u8, key, "chain")) {
                                    effect_section = .chain;
                                } else if (std.mem.eql(u8, key, "background")) {
                                    effect_section = .background;
                                } else {
                                    effect_section = .none;
                                }
                            } else {
                                effect_section = .none;
                            }
                        } else switch (effect_section) {
                            .chain => {
                                if (chain_item_indent == null) {
                                    // Still looking for `chain:`'s list items.
                                    if (std.mem.startsWith(u8, content, "- ")) {
                                        chain_item_indent = indent;
                                        if (parseBlockKey(content[2..])) |kind| {
                                            current_step_kind = try alloc.dupe(u8, kind);
                                            current_step_params = std.StringHashMap([]const u8).init(alloc);
                                        }
                                    }
                                } else if (indent == chain_item_indent.?) {
                                    // New chain item - flush the previous step.
                                    if (current_step_kind) |kind| {
                                        try current_effect.chain.append(alloc, .{ .kind = kind, .params = current_step_params });
                                        current_step_kind = null;
                                    }
                                    if (std.mem.startsWith(u8, content, "- ")) {
                                        if (parseBlockKey(content[2..])) |kind| {
                                            current_step_kind = try alloc.dupe(u8, kind);
                                            current_step_params = std.StringHashMap([]const u8).init(alloc);
                                        }
                                    }
                                } else if (current_step_kind != null) {
                                    // A field within the current step's params.
                                    if (parseKeyValue(content)) |kv| {
                                        try current_step_params.put(try alloc.dupe(u8, kv.key), try alloc.dupe(u8, unquote(kv.value)));
                                    }
                                }
                            },
                            .background => {
                                if (bg_field_indent == null) bg_field_indent = indent;

                                if (indent == bg_field_indent.?) {
                                    bg_in_sources = false;
                                    bg_source_indent = null;
                                    if (parseKeyValue(content)) |kv| {
                                        if (std.mem.eql(u8, kv.key, "volume")) {
                                            current_effect.background.volume = std.fmt.parseFloat(f32, unquote(kv.value)) catch 0.25;
                                        } else if (std.mem.eql(u8, kv.key, "random_start")) {
                                            current_effect.background.random_start = std.mem.eql(u8, unquote(kv.value), "true") or std.mem.eql(u8, unquote(kv.value), "yes");
                                        }
                                    } else if (parseBlockKey(content)) |key| {
                                        bg_in_sources = std.mem.eql(u8, key, "sources");
                                    }
                                } else if (bg_in_sources) {
                                    if (bg_source_indent == null) bg_source_indent = indent;
                                    if (indent == bg_source_indent.? and std.mem.startsWith(u8, content, "- ")) {
                                        if (parseKeyValue(content[2..])) |kv| {
                                            if (std.mem.eql(u8, kv.key, "file")) {
                                                try current_effect.background.sources.append(alloc, try alloc.dupe(u8, unquote(kv.value)));
                                            }
                                        }
                                    }
                                }
                            },
                            .none => {},
                        }
                    }
                },
                .none => {},
            }
        }
        if (current_voice_name) |name| {
            try config.voices.put(name, current_voice);
        }
        if (current_step_kind) |kind| {
            try current_effect.chain.append(alloc, .{ .kind = kind, .params = current_step_params });
        }
        if (current_effect_name) |name| {
            try config.effects.put(name, current_effect);
        }

        return config;
    }

    pub fn getPreset(self: *const Config, name: []const u8) ?VoicePreset {
        return self.voices.get(name);
    }

    /// Resolves a configured speaker alias to its raw sink name; `null`
    /// if `alias` isn't in the `speakers:` block (callers should fail
    /// fast on `null`, not fall back to treating `alias` as a raw name -
    /// see tmp/FUN_PLAN.md section 1's "aliases are the only accepted
    /// values" decision).
    pub fn getSpeakerSink(self: *const Config, alias: []const u8) ?[]const u8 {
        return self.speakers.get(alias);
    }

    pub fn getEffect(self: *const Config, name: []const u8) ?EffectPreset {
        return self.effects.get(name);
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

/// Parses an inline comma-separated list from a `key: [a, b]` value
/// (the only list syntax the config parser handles).
fn parseInlineList(alloc: std.mem.Allocator, raw: []const u8, out: *std.ArrayList([]const u8)) !void {
    var s = std.mem.trim(u8, raw, " \t\"");
    if (s.len >= 2 and s[0] == '[' and s[s.len - 1] == ']') s = s[1 .. s.len - 1];
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |item| {
        const i = unquote(std.mem.trim(u8, item, " \t"));
        if (i.len == 0) continue;
        try out.append(alloc, try alloc.dupe(u8, i));
    }
}

fn isIdentifier(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |ch| {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '-')) return false;
    }
    return true;
}
