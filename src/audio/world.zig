//! Audio World: a small simulation-loop-driven channel mixer, ported from
//! the Voice/AudioSourceInstance/MixingBus pattern in the user's Game9
//! prototype (src/game/systems/Audio.c, under /workspace/Making_Games/
//! Game9/Code), adapted per tmp/FUN_PLAN.md section 2. TTS output,
//! stingers, and (later) background loops are all just Entities enqueued
//! onto one of a fixed pool of Channels; `mix()` is called directly by
//! sokol_audio's stream callback (on its own audio thread, per
//! sokol_audio.h) and does the actual sample-domain summing across
//! channels - unlike the previous PulseAudio-based design, sokol_audio
//! explicitly does not mix multiple streams for you, so this is now our
//! job (see the "Prerequisite" subsection of tmp/FUN_PLAN.md section 2
//! for why).
//!
//! Channels are strict FIFOs on purpose: this is how "stinger blocks the
//! voice" will fall out for free once stingers are wired up (section 2's
//! `Entities, in practice`) - push the stinger Entity, then the voice
//! Entity, onto the same channel, and they play back-to-back with no gap
//! (mix() moves on to the next queued Entity within the same audio block
//! if the current one finishes mid-block).

const std = @import("std");

pub const CHANNEL_COUNT = 16; // matches Game9's AUDIO_VOICES

pub const Entity = struct {
    /// Already resampled to `World.mix_rate` and gain/effects-applied
    /// (see tmp/FUN_PLAN.md's "DSP effects" subsection - the World never
    /// runs a live per-callback DSP graph, only slices/sums/advances).
    /// Ownership transfers to the `World` on `enqueue()` - allocated from
    /// the same allocator passed to `World.init()`, freed automatically
    /// once fully played (or on `World.deinit()`).
    samples: []f32,
    read_pos: usize = 0,
    /// Background audio wraps back to 0 at EOF instead of finishing.
    loop: bool = false,
};

const Channel = struct {
    queue: std.ArrayList(Entity) = .empty,
};

pub const World = struct {
    /// Used to free `Entity.samples` once fully played. Entities are
    /// freed from `mix()` (the audio thread) on completion - acceptable
    /// for speech-rate audio (not hardcore low-latency realtime), and
    /// simpler than deferring frees to the main thread.
    allocator: std.mem.Allocator,
    mix_rate: u32,
    mutex: std.atomic.Mutex = .unlocked,
    channels: [CHANNEL_COUNT]Channel = blk: {
        var chs: [CHANNEL_COUNT]Channel = undefined;
        for (&chs) |*ch| ch.* = .{};
        break :blk chs;
    },

    pub fn init(alloc: std.mem.Allocator, mix_rate: u32) World {
        return .{ .allocator = alloc, .mix_rate = mix_rate };
    }

    /// `World.mix()` runs on a raw C callback thread owned by sokol_audio
    /// (no `Io` instance available there), so this is a plain spin-lock
    /// (`std.Thread.Mutex`/`std.Io.Mutex` both need more than that) - fine
    /// for the short, allocation-light critical sections here.
    fn lock(self: *World) void {
        while (!self.mutex.tryLock()) std.atomic.spinLoopHint();
    }

    /// Takes ownership of `entity.samples`. `channel_idx = null` picks the
    /// first idle channel (round-robin-ish; if all 16 are busy, piles onto
    /// channel 0 rather than silently dropping the request). Returns the
    /// channel index the entity landed on, so a caller (e.g. a stinger)
    /// can enqueue a follow-up Entity onto the exact same channel.
    pub fn enqueue(self: *World, channel_idx: ?usize, entity: Entity) !usize {
        self.lock();
        defer self.mutex.unlock();
        const idx = channel_idx orelse self.firstIdleLocked();
        try self.channels[idx].queue.append(self.allocator, entity);
        return idx;
    }

    fn firstIdleLocked(self: *World) usize {
        for (0..CHANNEL_COUNT) |i| {
            if (self.channels[i].queue.items.len == 0) return i;
        }
        return 0;
    }

    /// True once every channel's queue is empty (nothing left to play).
    pub fn idle(self: *World) bool {
        self.lock();
        defer self.mutex.unlock();
        for (&self.channels) |*ch| {
            if (ch.queue.items.len != 0) return false;
        }
        return true;
    }

    /// Called directly by sokol_audio's stream callback. `buffer` holds
    /// `num_frames * num_channels` interleaved f32 samples to fill.
    pub fn mix(self: *World, buffer: []f32, num_frames: u32, num_channels: u32) void {
        @memset(buffer, 0);
        self.lock();
        defer self.mutex.unlock();

        for (&self.channels) |*ch| {
            var frame: u32 = 0;
            while (frame < num_frames) {
                if (ch.queue.items.len == 0) break;
                var entity = &ch.queue.items[0];
                if (entity.read_pos >= entity.samples.len) {
                    if (entity.loop and entity.samples.len > 0) {
                        entity.read_pos = 0;
                    } else {
                        self.allocator.free(entity.samples);
                        _ = ch.queue.orderedRemove(0);
                        continue;
                    }
                }
                const s = entity.samples[entity.read_pos];
                entity.read_pos += 1;
                var c: u32 = 0;
                while (c < num_channels) : (c += 1) {
                    buffer[frame * num_channels + c] += s;
                }
                frame += 1;
            }
        }
    }

    pub fn deinit(self: *World) void {
        self.lock();
        defer self.mutex.unlock();
        for (&self.channels) |*ch| {
            for (ch.queue.items) |e| self.allocator.free(e.samples);
            ch.queue.deinit(self.allocator);
        }
    }
};
