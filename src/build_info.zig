//! Build metadata exported by `build.zig` for CLI version reporting.
//!
//! The values in this module are generated at compile time from build options
//! such as `-Dversion` and `-Dcommit`, then printed by `zinc --version`.
//! @section CLI & Entrypoints
const std = @import("std");
const build_options = @import("build_options");

pub const version = build_options.version;
pub const commit = build_options.commit;
pub const target = build_options.target;
pub const optimize = build_options.optimize;
pub const backend = build_options.backend;

pub fn writeVersion(writer: anytype) !void {
    try writer.print(
        \\zinc {s}
        \\commit: {s}
        \\target: {s}
        \\optimize: {s}
        \\backends: {s}
        \\
    , .{
        version,
        commit,
        target,
        optimize,
        backend,
    });
}

test "version metadata is present" {
    try std.testing.expect(version.len > 0);
    try std.testing.expect(commit.len > 0);
    try std.testing.expect(target.len > 0);
    try std.testing.expect(optimize.len > 0);
    try std.testing.expect(backend.len > 0);
}
