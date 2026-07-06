const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // G2P: Fable's zig-phonemes (vendor/zig-phonemes, evaluated per
    // tmp/PHENOMES.md/PHASE3_PLAN.md section 2 as an alternative to plain
    // espeak-ng). A git submodule (see .gitmodules) - run
    // `git submodule update --init` after cloning this repo.
    const g2p_module = b.createModule(.{
        .root_source_file = b.path("vendor/zig-phonemes/src/g2p.zig"),
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
    exe_mod.addImport("zig_phonemes", g2p_module);

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

    // Persistent audio output (section 4/8 of the original plan, migrated
    // to sokol_audio per tmp/FUN_PLAN.md section 2's "Prerequisite" -
    // restores the cross-platform portability v1 (Python/PortAudio) had,
    // which the original PulseAudio pa_simple binding regressed. Official
    // Zig bindings (github.com/floooh/sokol-zig), same dependency as
    // Game9 (this project's inspiration for the channel mixer in
    // src/audio/world.zig). `.dont_link_system_libs = true` skips
    // sokol-zig's default GL/X11/asound linking (we don't use
    // sokol_app/sokol_gfx, only sokol_audio) - we link just `asound`
    // ourselves below, so this binary has no graphics/windowing runtime
    // dependency at all. See src/audio/{output,world,resample}.zig.
    const sokol_dep = b.dependency("sokol", .{
        .target = target,
        .optimize = optimize,
        .dont_link_system_libs = true,
    });
    exe_mod.addImport("sokol", sokol_dep.module("sokol"));
    exe_mod.linkSystemLibrary("asound", .{});

    // Linux-only direct-to-sink playback (tmp/FUN_PLAN.md section 1's
    // speaker/sink selection - a parallel opt-in path alongside the
    // portable sokol_audio default, since sokol_audio has no device
    // selection API on any platform). See src/audio/linux_sink.zig.
    if (target.result.os.tag == .linux) {
        const pulse_translate = b.addTranslateC(.{
            .root_source_file = b.graph.cwdRelativePath("/usr/include/pulse/simple.h"),
            .target = target,
            .optimize = optimize,
        });
        const pulse_c_module = pulse_translate.createModule();
        exe_mod.addImport("pulse_c", pulse_c_module);
        exe_mod.linkSystemLibrary("pulse-simple", .{});
        exe_mod.linkSystemLibrary("pulse", .{});
    }

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
