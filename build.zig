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
