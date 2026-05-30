//! Per-kernel Metal dispatch timing probe — default-off, env-flag-gated.
//!
//! When `ZINC_METAL_KERNEL_TIMING=1` is set at engine init, every compute
//! dispatch is wrapped in commit+wait+restart inside `MetalCommand.dispatch*`
//! so we can measure CPU-side end-to-end ns per dispatch. The probe is
//! intentionally destructive to throughput (each dispatch becomes a GPU sync
//! point) and is intended ONLY for `--profile` runs where evidence about
//! which kernels dominate dispatch cost matters more than absolute tok/s.
//!
//! Aggregation is keyed by pipeline pointer; the human-readable label comes
//! from `MetalPipeline.name` set at shader load time in forward_metal.zig.
//! @section Metal Runtime
const std = @import("std");

const MAX_SLOTS: usize = 512;

const Slot = struct {
    pipe_handle: ?*const anyopaque = null,
    name: []const u8 = "",
    calls: u64 = 0,
    total_ns: u64 = 0,
    max_ns: u64 = 0,
};

/// Toggled true at engine init when `ZINC_METAL_KERNEL_TIMING=1`.
pub var enabled: bool = false;

var slots: [MAX_SLOTS]Slot = [_]Slot{.{}} ** MAX_SLOTS;
var slot_count: usize = 0;
var mutex: std.Thread.Mutex = .{};

/// Snapshot view of one pipeline's aggregated dispatch cost.
pub const Entry = struct {
    name: []const u8,
    calls: u64,
    total_ns: u64,
    max_ns: u64,
    avg_ns: u64,
};

/// Enable the probe for the rest of the process. Idempotent.
pub fn enable() void {
    enabled = true;
}

/// Clear accumulated stats. Typically called at the start of a profile request.
pub fn reset() void {
    mutex.lock();
    defer mutex.unlock();
    slots = [_]Slot{.{}} ** MAX_SLOTS;
    slot_count = 0;
}

/// Record one dispatch worth of elapsed ns against a pipeline. Cheap when
/// `enabled` is false (skips early at the call site).
pub fn record(pipe_handle: ?*const anyopaque, name: ?[]const u8, elapsed_ns: u64) void {
    if (pipe_handle == null) return;

    mutex.lock();
    defer mutex.unlock();

    var i: usize = 0;
    while (i < slot_count) : (i += 1) {
        if (slots[i].pipe_handle == pipe_handle) {
            slots[i].calls += 1;
            slots[i].total_ns += elapsed_ns;
            if (elapsed_ns > slots[i].max_ns) slots[i].max_ns = elapsed_ns;
            return;
        }
    }

    if (slot_count >= MAX_SLOTS) return;

    slots[slot_count] = .{
        .pipe_handle = pipe_handle,
        .name = name orelse "<unnamed>",
        .calls = 1,
        .total_ns = elapsed_ns,
        .max_ns = elapsed_ns,
    };
    slot_count += 1;
}

/// Fill `buf` with up to `buf.len` entries ranked by descending total_ns.
/// Returns the populated prefix slice.
pub fn topByTotalNs(buf: []Entry) []Entry {
    mutex.lock();
    defer mutex.unlock();

    var scratch: [MAX_SLOTS]Entry = undefined;
    var n: usize = 0;
    var i: usize = 0;
    while (i < slot_count) : (i += 1) {
        const slot = slots[i];
        if (slot.calls == 0) continue;
        scratch[n] = .{
            .name = slot.name,
            .calls = slot.calls,
            .total_ns = slot.total_ns,
            .max_ns = slot.max_ns,
            .avg_ns = slot.total_ns / @max(slot.calls, 1),
        };
        n += 1;
    }

    const out_n = @min(n, buf.len);
    var k: usize = 0;
    while (k < out_n) : (k += 1) {
        var best = k;
        var j = k + 1;
        while (j < n) : (j += 1) {
            if (scratch[j].total_ns > scratch[best].total_ns) best = j;
        }
        if (best != k) std.mem.swap(Entry, &scratch[k], &scratch[best]);
        buf[k] = scratch[k];
    }
    return buf[0..out_n];
}

test "record and topByTotalNs ranks by total_ns" {
    reset();
    const pa = @as(*const anyopaque, @ptrFromInt(0x1000));
    const pb = @as(*const anyopaque, @ptrFromInt(0x2000));
    record(pa, "small", 100);
    record(pa, "small", 200);
    record(pb, "big", 5000);

    var buf: [4]Entry = undefined;
    const top = topByTotalNs(&buf);
    try std.testing.expectEqual(@as(usize, 2), top.len);
    try std.testing.expectEqualStrings("big", top[0].name);
    try std.testing.expectEqual(@as(u64, 5000), top[0].total_ns);
    try std.testing.expectEqualStrings("small", top[1].name);
    try std.testing.expectEqual(@as(u64, 300), top[1].total_ns);
    try std.testing.expectEqual(@as(u64, 150), top[1].avg_ns);
    reset();
}

test "record drops past MAX_SLOTS without crashing" {
    reset();
    var i: usize = 0;
    while (i < MAX_SLOTS + 10) : (i += 1) {
        const fake = @as(*const anyopaque, @ptrFromInt(0x10000 + i * 0x100));
        record(fake, "x", 1);
    }
    try std.testing.expectEqual(@as(usize, MAX_SLOTS), slot_count);
    reset();
}
