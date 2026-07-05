//! Thin Zig bindings over the ONNX Runtime C API (vendor/onnxruntime,
//! v1.27.0 - CPU execution provider only for now). Enough surface to load
//! a Piper/Kokoro voice model, run inference with named tensors, and read
//! back the output float buffer.

const std = @import("std");
pub const c = @import("onnxruntime_c");

pub const OrtError = error{
    GetApiBaseFailed,
    GetApiFailed,
    CreateEnvFailed,
    CreateSessionOptionsFailed,
    CreateSessionFailed,
    CreateMemoryInfoFailed,
    CreateTensorFailed,
    RunFailed,
    GetTensorDataFailed,
    GetTensorShapeFailed,
    SessionInputOutputCountFailed,
    SessionInputOutputNameFailed,
};

fn check(api: *const c.OrtApi, status: ?*c.OrtStatus, err: OrtError) OrtError!void {
    if (status) |s| {
        const msg = api.GetErrorMessage.?(s);
        std.debug.print("[onnxruntime] {s}\n", .{msg});
        api.ReleaseStatus.?(s);
        return err;
    }
}

pub const Runtime = struct {
    api: *const c.OrtApi,
    env: *c.OrtEnv,
    memory_info: *c.OrtMemoryInfo,

    pub fn init() !Runtime {
        const base = c.OrtGetApiBase() orelse return OrtError.GetApiBaseFailed;
        const raw_api = base.*.GetApi.?(c.ORT_API_VERSION) orelse return OrtError.GetApiFailed;
        const api: *const c.OrtApi = @ptrCast(raw_api);

        var env: ?*c.OrtEnv = null;
        try check(api, api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "presence-voice", &env), OrtError.CreateEnvFailed);

        var memory_info: ?*c.OrtMemoryInfo = null;
        try check(api, api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info), OrtError.CreateMemoryInfoFailed);

        return .{ .api = api, .env = env.?, .memory_info = memory_info.? };
    }
};

pub const Session = struct {
    api: *const c.OrtApi,
    session: *c.OrtSession,
    memory_info: *c.OrtMemoryInfo,
    allocator: std.mem.Allocator,

    pub fn load(rt: *const Runtime, alloc: std.mem.Allocator, model_path: []const u8) !Session {
        const path_z = try alloc.dupeSentinel(u8, model_path, 0);
        defer alloc.free(path_z);

        var opts: ?*c.OrtSessionOptions = null;
        try check(rt.api, rt.api.CreateSessionOptions.?(&opts), OrtError.CreateSessionOptionsFailed);
        defer rt.api.ReleaseSessionOptions.?(opts);
        _ = rt.api.SetIntraOpNumThreads.?(opts, 1);

        var session: ?*c.OrtSession = null;
        try check(rt.api, rt.api.CreateSession.?(rt.env, path_z.ptr, opts, &session), OrtError.CreateSessionFailed);

        return .{ .api = rt.api, .session = session.?, .memory_info = rt.memory_info, .allocator = alloc };
    }

    pub fn deinit(self: *Session) void {
        self.api.ReleaseSession.?(self.session);
    }

    /// Runs the session with int64 and float32 named input tensors, returns
    /// the first output as an owned f32 slice (caller frees).
    pub fn runF32(
        self: *const Session,
        alloc: std.mem.Allocator,
        input_names: []const [:0]const u8,
        inputs: []const *c.OrtValue,
        output_name: [:0]const u8,
    ) ![]f32 {
        var input_name_ptrs = try alloc.alloc([*c]const u8, input_names.len);
        defer alloc.free(input_name_ptrs);
        for (input_names, 0..) |n, i| input_name_ptrs[i] = n.ptr;

        const output_name_ptrs = [_][*c]const u8{output_name.ptr};
        var output_values = [_]?*c.OrtValue{null};

        try check(self.api, self.api.Run.?(
            self.session,
            null,
            input_name_ptrs.ptr,
            @ptrCast(inputs.ptr),
            inputs.len,
            &output_name_ptrs,
            1,
            &output_values,
        ), OrtError.RunFailed);
        defer self.api.ReleaseValue.?(output_values[0]);

        var data_ptr: ?*anyopaque = null;
        try check(self.api, self.api.GetTensorMutableData.?(output_values[0], &data_ptr), OrtError.GetTensorDataFailed);

        var type_info: ?*c.OrtTensorTypeAndShapeInfo = null;
        try check(self.api, self.api.GetTensorTypeAndShape.?(output_values[0], &type_info), OrtError.GetTensorShapeFailed);
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(type_info);

        var elem_count: usize = 0;
        try check(self.api, self.api.GetTensorShapeElementCount.?(type_info, &elem_count), OrtError.GetTensorShapeFailed);

        const out = try alloc.alloc(f32, elem_count);
        const src: [*]f32 = @ptrCast(@alignCast(data_ptr.?));
        @memcpy(out, src[0..elem_count]);
        return out;
    }

    pub fn createInt64Tensor(self: *const Session, alloc: std.mem.Allocator, data: []const i64, shape: []const i64) !*c.OrtValue {
        _ = alloc;
        var value: ?*c.OrtValue = null;
        try check(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(@constCast(data.ptr)),
            data.len * @sizeOf(i64),
            shape.ptr,
            shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &value,
        ), OrtError.CreateTensorFailed);
        return value.?;
    }

    pub fn createF32Tensor(self: *const Session, alloc: std.mem.Allocator, data: []const f32, shape: []const i64) !*c.OrtValue {
        _ = alloc;
        var value: ?*c.OrtValue = null;
        try check(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(@constCast(data.ptr)),
            data.len * @sizeOf(f32),
            shape.ptr,
            shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &value,
        ), OrtError.CreateTensorFailed);
        return value.?;
    }
};
