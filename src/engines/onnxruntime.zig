//! Thin Zig bindings over the ONNX Runtime C API (vendor/onnxruntime,
//! v1.27.0 gpu_cuda13 release - includes both CPU and CUDA execution
//! providers in the same libonnxruntime.so). Enough surface to load a
//! Piper/Kokoro voice model, run inference with named tensors, and read
//! back the output float buffer.
//!
//! CUDA EP: per Fable's tmp/onnx-cuda-lab/REPORT.md investigation, ORT
//! 1.27.0's official cu13 build works on this machine's CUDA 13.3 +
//! Blackwell (sm_120) card, needing only system cuDNN 9 (`pacman -S
//! cudnn`) beyond what /opt/cuda already provides - ~16x faster than CPU
//! for Kokoro (80ms steady-state vs. ~1.3-1.5s). `Session.load` tries CUDA
//! first and falls back to CPU on failure (e.g. no GPU/cuDNN present),
//! logging which one was used - never a hard failure just because CUDA
//! isn't available.

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
        return load2(rt, alloc, model_path, true);
    }

    /// `try_cuda`: attempt the CUDA execution provider first, falling back
    /// to CPU (with a log line either way) if it can't be appended (no
    /// GPU, missing cuDNN, etc.) - see the module doc comment.
    pub fn load2(rt: *const Runtime, alloc: std.mem.Allocator, model_path: []const u8, try_cuda: bool) !Session {
        const path_z = try alloc.dupeSentinel(u8, model_path, 0);
        defer alloc.free(path_z);

        var opts: ?*c.OrtSessionOptions = null;
        try check(rt.api, rt.api.CreateSessionOptions.?(&opts), OrtError.CreateSessionOptionsFailed);
        defer rt.api.ReleaseSessionOptions.?(opts);
        _ = rt.api.SetIntraOpNumThreads.?(opts, 1);

        var used_cuda = false;
        if (try_cuda) {
            var cuda_opts: ?*c.OrtCUDAProviderOptionsV2 = null;
            const create_status = rt.api.CreateCUDAProviderOptions.?(&cuda_opts);
            if (create_status == null) {
                var keys = [_][*c]const u8{"device_id"};
                var vals = [_][*c]const u8{"0"};
                const update_status = rt.api.UpdateCUDAProviderOptions.?(cuda_opts, &keys, &vals, 1);
                const append_status = if (update_status == null)
                    rt.api.SessionOptionsAppendExecutionProvider_CUDA_V2.?(opts, cuda_opts)
                else
                    update_status;
                rt.api.ReleaseCUDAProviderOptions.?(cuda_opts);
                if (append_status == null) {
                    used_cuda = true;
                    std.debug.print("[onnxruntime] CUDA execution provider ready\n", .{});
                } else {
                    const msg = rt.api.GetErrorMessage.?(append_status);
                    std.debug.print("[onnxruntime] CUDA EP unavailable ({s}), falling back to CPU\n", .{msg});
                    rt.api.ReleaseStatus.?(append_status);
                }
            } else {
                const msg = rt.api.GetErrorMessage.?(create_status);
                std.debug.print("[onnxruntime] CUDA EP unavailable ({s}), falling back to CPU\n", .{msg});
                rt.api.ReleaseStatus.?(create_status);
            }
        }
        if (!used_cuda) std.debug.print("[onnxruntime] using CPU execution provider\n", .{});

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
