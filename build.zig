const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // G2P: Fable's zig-phenomes (tmp/zig-phenomes, evaluated per
    // tmp/PHENOMES.md/PHASE3_PLAN.md section 2 as an alternative to plain
    // espeak-ng). Referenced in place, read-only - not vendored/copied yet
    // since it's still under active development in its own directory.
    const g2p_module = b.createModule(.{
        .root_source_file = b.path("tmp/zig-phenomes/src/g2p.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe_mod.addImport("zig_phenomes", g2p_module);

    // ONNX Runtime (vendor/onnxruntime, v1.27.0 gpu_cuda13 release - CPU
    // and CUDA execution providers both in one libonnxruntime.so - see
    // src/engines/onnxruntime.zig and Fable's tmp/onnx-cuda-lab/REPORT.md).
    // Not vendored into git (vendor/ is gitignored); fetched once via a
    // prebuilt release tarball. Also needs system cuDNN 9 (`pacman -S
    // cudnn`) for the CUDA EP; CPU EP works regardless.
    // @cImport was removed from this Zig snapshot - translate-c now runs as
    // its own build step producing a real module.
    const onnx_translate = b.addTranslateC(.{
        .root_source_file = b.path("vendor/onnxruntime/include/onnxruntime_c_api.h"),
        .target = target,
        .optimize = optimize,
    });
    onnx_translate.addIncludePath(b.path("vendor/onnxruntime/include"));
    const onnx_c_module = onnx_translate.createModule();
    exe_mod.addImport("onnxruntime_c", onnx_c_module);

    exe_mod.addIncludePath(b.path("vendor/onnxruntime/include"));
    exe_mod.addLibraryPath(b.path("vendor/onnxruntime/lib"));
    exe_mod.linkSystemLibrary("onnxruntime", .{});
    exe_mod.addRPath(b.path("vendor/onnxruntime/lib"));

    const exe = b.addExecutable(.{
        .name = "voice",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the voice CLI");
    run_step.dependOn(&run_cmd.step);
}
