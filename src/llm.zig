//! Personality prompt: rewrites input text through a local LLM before
//! synthesis (tmp/FUN_PLAN.md section 4). Uses LM Studio's local server,
//! which speaks the OpenAI chat-completions wire format (NOT Ollama's
//! own `/api/chat` shape - see the plan doc for why LM Studio was chosen
//! over Ollama). A minimal use of `std.http.Client`'s one-shot `fetch()`
//! rather than a hand-rolled socket client - the request/response shapes
//! (JSON in, JSON out, one call, no streaming) don't need anything more.
//!
//! This is opt-in per preset (`personality_prompt` in config.yaml) for a
//! reason worth restating: an LLM round trip is *seconds*, not
//! milliseconds - directly against this whole project's original
//! lowest-TTFB goal. Presets without `personality_prompt` never pay this
//! cost.

const std = @import("std");

pub const LlmError = error{ BadResponse, RequestFailed };

const ChatMessage = struct { role: []const u8, content: []const u8 };
const ChatRequest = struct { model: []const u8, messages: []const ChatMessage, stream: bool };

/// Sends `user_text` to the LLM at `completion_url` with `system_prompt`
/// as the system message, and returns the model's reply (allocated from
/// `alloc`). Callers should treat any error as "LLM unreachable/bad
/// response" and fall back to speaking `user_text` literally rather than
/// failing the whole request - see tmp/FUN_PLAN.md section 4's
/// "operational dependency" note.
pub fn rewrite(alloc: std.mem.Allocator, io: std.Io, completion_url: []const u8, model: []const u8, system_prompt: []const u8, user_text: []const u8) ![]const u8 {
    const messages = [_]ChatMessage{
        .{ .role = "system", .content = system_prompt },
        .{ .role = "user", .content = user_text },
    };
    const req: ChatRequest = .{ .model = model, .messages = &messages, .stream = false };
    const req_body = try std.json.Stringify.valueAlloc(alloc, req, .{});

    var client: std.http.Client = .{ .allocator = alloc, .io = io };
    defer client.deinit();

    var response_buf: std.Io.Writer.Allocating = .init(alloc);
    defer response_buf.deinit();

    const result = try client.fetch(.{
        .location = .{ .url = completion_url },
        .method = .POST,
        .payload = req_body,
        .extra_headers = &.{.{ .name = "Content-Type", .value = "application/json" }},
        .response_writer = &response_buf.writer,
        .keep_alive = false,
    });
    if (result.status != .ok) return LlmError.RequestFailed;

    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, response_buf.written(), .{});
    const choices = parsed.object.get("choices") orelse return LlmError.BadResponse;
    if (choices.array.items.len == 0) return LlmError.BadResponse;
    const message = choices.array.items[0].object.get("message") orelse return LlmError.BadResponse;
    const content = message.object.get("content") orelse return LlmError.BadResponse;
    if (content != .string) return LlmError.BadResponse;
    return try alloc.dupe(u8, content.string);
}
