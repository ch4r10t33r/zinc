//! Standalone Vulkan validation-tool imports.
//!
//! This module is intentionally small: tools under `tools/` use it as an
//! umbrella module when they need to exercise shader pipelines outside the main
//! ZINC binary.

pub const vk = @import("vk.zig");
pub const instance = @import("instance.zig");
pub const Instance = instance.Instance;
pub const Buffer = @import("buffer.zig").Buffer;
pub const command = @import("command.zig");
pub const pipeline = @import("pipeline.zig");
