//! CLI argument parsing + dispatch helpers for `local`/`client`/`list`/`-i`/
//! `-h`, per tmp/PHASE3_PLAN.md section 6. Options can appear anywhere in
//! the argument list (the "flexible option positioning" decision) - this
//! module classifies every arg as an option or a positional regardless of
//! order, then resolves preset/text from whatever positionals remain.

const std = @import("std");
const config_mod = @import("config.zig");
const paths = @import("paths.zig");

pub const Options = struct {
    output: ?[]const u8 = null,
    config_path: []const u8 = paths.CONFIG,
    info: ?[]const u8 = null,
    cpu: bool = false,
    stinger: ?[]const u8 = null,
    speaker: ?[]const u8 = null,
    effects: std.ArrayList([]const u8) = .empty,
    /// -I/--interrupt: stop any playing/queued speech before this one.
    /// The wire protocol's `schedule` field is always explicit — this
    /// flag selects `interrupt`; without it the client sends `enqueue`.
    interrupt: bool = false,
    gain: f32 = 1.0,
    help: bool = false,
    positionals: std.ArrayList([]const u8) = .empty,
};

fn eqlAny(a: []const u8, opts: []const []const u8) bool {
    for (opts) |o| if (std.mem.eql(u8, a, o)) return true;
    return false;
}

pub fn parseOptions(alloc: std.mem.Allocator, args: []const []const u8) !Options {
    var opts: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (eqlAny(a, &.{ "-o", "--output" }) and i + 1 < args.len) {
            i += 1;
            opts.output = args[i];
        } else if (std.mem.startsWith(u8, a, "--output=")) {
            opts.output = a[9..];
        } else if (eqlAny(a, &.{ "-c", "--config" }) and i + 1 < args.len) {
            i += 1;
            opts.config_path = args[i];
        } else if (std.mem.startsWith(u8, a, "--config=")) {
            opts.config_path = a[9..];
        } else if (eqlAny(a, &.{ "-i", "--info" }) and i + 1 < args.len) {
            i += 1;
            opts.info = args[i];
        } else if (std.mem.startsWith(u8, a, "--info=")) {
            opts.info = a[7..];
        } else if (eqlAny(a, &.{ "-C", "--cpu" })) {
            opts.cpu = true;
        } else if (eqlAny(a, &.{ "-s", "--stinger" }) and i + 1 < args.len) {
            i += 1;
            opts.stinger = args[i];
        } else if (std.mem.startsWith(u8, a, "--stinger=")) {
            opts.stinger = a[10..];
        } else if (eqlAny(a, &.{ "-I", "--interrupt" })) {
            opts.interrupt = true;
        } else if (eqlAny(a, &.{ "-d", "--speaker" }) and i + 1 < args.len) {
            i += 1;
            opts.speaker = args[i];
        } else if (std.mem.startsWith(u8, a, "--speaker=")) {
            opts.speaker = a[10..];
        } else if (eqlAny(a, &.{ "-e", "--effect" }) and i + 1 < args.len) {
            i += 1;
            try opts.effects.append(alloc, args[i]);
        } else if (std.mem.startsWith(u8, a, "--effect=")) {
            try opts.effects.append(alloc, a[9..]);
        } else if (eqlAny(a, &.{ "-g", "--gain" }) and i + 1 < args.len) {
            i += 1;
            opts.gain = std.fmt.parseFloat(f32, args[i]) catch 1.0;
        } else if (std.mem.startsWith(u8, a, "--gain=")) {
            opts.gain = std.fmt.parseFloat(f32, a[7..]) catch 1.0;
        } else if (eqlAny(a, &.{ "-h", "--help" })) {
            opts.help = true;
        } else if (std.mem.eql(u8, a, "--")) {
            i += 1;
            while (i < args.len) : (i += 1) try opts.positionals.append(alloc, args[i]);
            break;
        } else {
            try opts.positionals.append(alloc, a);
        }
    }
    opts.gain = std.math.clamp(opts.gain, 0.0, 2.0);
    return opts;
}

pub const Resolved = struct { preset_name: []const u8, text: []const u8 };

/// Resolution rule (section 6): if the first positional matches a
/// configured preset name, it's the preset and the rest is text;
/// otherwise every positional is text and default_preset/fallback_voice
/// (in that order) is used instead.
pub fn resolvePresetAndText(alloc: std.mem.Allocator, cfg: *const config_mod.Config, positionals: []const []const u8) !?Resolved {
    if (positionals.len == 0) return null;
    if (cfg.getPreset(positionals[0]) != null) {
        const text = try std.mem.join(alloc, " ", positionals[1..]);
        return .{ .preset_name = positionals[0], .text = text };
    }
    const preset_name = cfg.default_preset orelse cfg.fallback_voice orelse return null;
    const text = try std.mem.join(alloc, " ", positionals);
    return .{ .preset_name = preset_name, .text = text };
}

/// Multiplies in place, clamped to [-1, 1] (matches the WAV writer's own
/// clamp, so gain > 1.0 clips the same way it would on real hardware).
pub fn applyGain(samples: []f32, gain: f32) void {
    if (gain == 1.0) return;
    for (samples) |*s| s.* = std.math.clamp(s.* * gain, -1.0, 1.0);
}

pub const HELP_TEXT =
    \\voice — text-to-speech CLI + always-on daemon (Piper + Kokoro)
    \\
    \\Usage:
    \\    voice [options] [preset] <text>          (shorthand for "client")
    \\    voice <command> [options] [preset] <text>
    \\    voice <command> [options]
    \\
    \\COMMANDS:
    \\    local                  Synthesize standalone, in-process (never uses the daemon)
    \\    client                 Synthesize via the running daemon (fails if it isn't running)
    \\    list                   List available voice presets
    \\    speakers               List audio output sinks + configured aliases (Linux only)
    \\    serve                  Start the always-on daemon (unix socket)
    \\    serve --http             ...also start the HTTP API on the same process
    \\
    \\DIRECT synthesis ("local"/"client", or the bare shorthand for "client"):
    \\    [preset] <text>        Speak the provided text immediately
    \\                           Preset is optional - omit it to use config.yaml's default_preset.
    \\
    \\Options:
    \\    -o, --output <file>   Save audio to WAV instead of playing it (local only, so far)
    \\    -c, --config <file>   Use a custom config.yaml
    \\    -i, --info <preset>   Show preset details
    \\    -C, --cpu             Force CPU usage instead of GPU (Kokoro only; "local" only)
    \\    -s, --stinger <name>  Play a configured stinger before speech (not yet implemented)
    \\    -d, --speaker <alias> Route audio to a configured speaker alias (see `voice speakers`; Linux only so far)
    \\    -e, --effect <name>   Apply a configured effect preset (repeatable, applied in order given)
    \\    -I, --interrupt       Stop any playing/queued speech before this one ("client" only;
    \\                          default is to enqueue behind whatever is already speaking)
    \\    -g, --gain <value>    Apply linear gain to synthesized voice audio (range: 0.0-2.0, default 1.0; local only, so far)
    \\    -h, --help            Show this help
    \\
    \\Examples:
    \\    voice lessac "Hello from Piper."           # shorthand for "client" - daemon must be running
    \\    voice "Hello with no preset given"         # shorthand for "client" + default_preset
    \\    voice local lessac "Hello from Piper."     # standalone, no daemon required
    \\    voice local -g 1.4 alan "Louder playback"
    \\    voice list
    \\    voice speakers
    \\    voice -d headphones lessac "Routed to my headphones alias"
    \\    voice -e radio_comms lessac "This should sound like a field radio"
    \\    voice -i alan
    \\    voice serve
    \\    voice serve --http
    \\
;
