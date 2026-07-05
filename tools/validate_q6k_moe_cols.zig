const std = @import("std");

const vulkan = @import("vulkan");
const vk = vulkan.vk;
const instance_mod = vulkan.instance;
const Instance = vulkan.Instance;
const Buffer = vulkan.Buffer;
const command = vulkan.command;
const pipeline_mod = vulkan.pipeline;

const MoeColsDmmvPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
    ids_stride: u32,
    x_route_divisor: u32,
    accumulate: u32 = 0,
};

fn s8ToF32(x: u8) f32 {
    return @floatFromInt(if (x < 128) @as(i32, x) else @as(i32, x) - 256);
}

fn dequantQ6KRow(raw: []const u8, row: usize, K: usize, out: []f32) void {
    const blocks_per_row = K / 256;
    const row_bytes = blocks_per_row * 210;
    const row_base = row * row_bytes;

    var out_i: usize = 0;
    for (0..blocks_per_row) |b| {
        const bb = row_base + b * 210;
        const d_bits = std.mem.readInt(u16, raw[bb + 208 ..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

        for (0..2) |half| {
            const ql_base = bb + half * 64;
            const qh_base = bb + 128 + half * 32;
            const sc_base = bb + 192 + half * 8;

            for (0..32) |l| {
                const is = l / 16;
                const ql0 = raw[ql_base + l];
                const ql1 = raw[ql_base + 32 + l];
                const qh = raw[qh_base + l];

                const q0: f32 = @floatFromInt(@as(u32, ql0 & 0x0f) | (@as(u32, qh & 0x03) << 4));
                const q1: f32 = @floatFromInt(@as(u32, ql1 & 0x0f) | (@as(u32, (qh >> 2) & 0x03) << 4));
                const q2: f32 = @floatFromInt(@as(u32, ql0 >> 4) | (@as(u32, (qh >> 4) & 0x03) << 4));
                const q3: f32 = @floatFromInt(@as(u32, ql1 >> 4) | (@as(u32, (qh >> 6) & 0x03) << 4));

                out[out_i + l] = d * s8ToF32(raw[sc_base + is]) * (q0 - 32.0);
                out[out_i + l + 32] = d * s8ToF32(raw[sc_base + 2 + is]) * (q1 - 32.0);
                out[out_i + l + 64] = d * s8ToF32(raw[sc_base + 4 + is]) * (q2 - 32.0);
                out[out_i + l + 96] = d * s8ToF32(raw[sc_base + 6 + is]) * (q3 - 32.0);
            }
            out_i += 128;
        }
    }
}

fn uploadPattern(
    weight: []u8,
    input: []f32,
    counts: []u32,
    ids: []u32,
    active_blocks: []u32,
    comptime M: usize,
    comptime K: usize,
    comptime n_tokens: usize,
    comptime k_used: usize,
    comptime n_experts: usize,
) void {
    const blocks_per_row = K / 256;
    const row_bytes = blocks_per_row * 210;
    const expert_stride = M * row_bytes;

    @memset(weight, 0);
    @memset(counts, 0);
    @memset(ids, std.math.maxInt(u32));
    @memset(active_blocks, std.math.maxInt(u32));

    for (0..n_experts) |expert| {
        for (0..M) |row| {
            for (0..blocks_per_row) |blk| {
                const base = expert * expert_stride + row * row_bytes + blk * 210;
                const d: f16 = @floatCast(0.03125 * @as(f32, @floatFromInt(1 + expert + (row % 5) + blk)));
                const d_bits: u16 = @bitCast(d);
                weight[base + 208] = @truncate(d_bits);
                weight[base + 209] = @truncate(d_bits >> 8);
                for (0..192) |i| {
                    weight[base + i] = @intCast((expert * 23 + row * 11 + blk * 17 + i * 7) & 0xff);
                }
                for (0..16) |i| {
                    weight[base + 192 + i] = @intCast((expert * 29 + row * 13 + blk * 19 + i * 5) & 0xff);
                }
            }
        }
    }

    for (0..n_tokens * k_used) |route| {
        for (0..K) |i| {
            const raw: i32 = @intCast((route * 31 + i * 17 + 9) % 25);
            input[route * K + i] = 0.125 * @as(f32, @floatFromInt(raw - 12));
        }
    }

    counts[1] = n_tokens;
    counts[2] = n_tokens;
    for (0..n_tokens) |token| {
        ids[1 * n_tokens + token] = @intCast(token * k_used);
        ids[2 * n_tokens + token] = @intCast(token * k_used + 1);
    }
    active_blocks[0] = 1;
    active_blocks[1] = 2;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const shader_path = "zig-out/share/zinc/shaders/dmmv_q6k_moe_cols.spv";

    const device_index = if (std.posix.getenv("ZINC_GPU")) |raw|
        std.fmt.parseInt(u32, raw, 10) catch instance_mod.auto_select_device_index
    else
        instance_mod.auto_select_device_index;

    var instance = try Instance.init(allocator, device_index);
    defer instance.deinit();
    if (instance.push_descriptor_fn == null) return error.PushDescriptorsUnavailable;

    var pool = try command.CommandPool.init(&instance);
    defer pool.deinit();
    var cmd = try command.CommandBuffer.init(&instance, &pool);
    defer cmd.deinit(&pool);

    var pipe = try pipeline_mod.createFromSpirvWithOptions(
        &instance,
        shader_path,
        6,
        @sizeOf(MoeColsDmmvPush),
        &.{},
        .{ .required_subgroup_size = 64, .require_full_subgroups = true, .push_descriptors = true },
        allocator,
    );
    defer pipe.deinit();

    const M: usize = 2048;
    const K: usize = 768;
    const n_tokens: usize = 5;
    const k_used: usize = 2;
    const n_experts: usize = 3;
    const route_slots = n_tokens * k_used;
    const blocks_per_row = K / 256;
    const row_bytes = blocks_per_row * 210;
    const expert_stride = M * row_bytes;

    var weight_buf = try Buffer.initHostVisibleStorage(&instance, n_experts * expert_stride);
    defer weight_buf.deinit();
    var input_buf = try Buffer.initHostVisibleStorage(&instance, route_slots * K * @sizeOf(f32));
    defer input_buf.deinit();
    var output_buf = try Buffer.initHostVisibleStorage(&instance, route_slots * M * @sizeOf(f32));
    defer output_buf.deinit();
    var counts_buf = try Buffer.initHostVisibleStorage(&instance, n_experts * @sizeOf(u32));
    defer counts_buf.deinit();
    var ids_buf = try Buffer.initHostVisibleStorage(&instance, n_experts * n_tokens * @sizeOf(u32));
    defer ids_buf.deinit();
    var active_buf = try Buffer.initHostVisibleStorage(&instance, 2 * @sizeOf(u32));
    defer active_buf.deinit();

    const weight = weight_buf.mapped.?[0..weight_buf.size];
    const input: [*]f32 = @ptrCast(@alignCast(input_buf.mapped.?));
    const output: [*]f32 = @ptrCast(@alignCast(output_buf.mapped.?));
    const counts: [*]u32 = @ptrCast(@alignCast(counts_buf.mapped.?));
    const ids: [*]u32 = @ptrCast(@alignCast(ids_buf.mapped.?));
    const active_blocks: [*]u32 = @ptrCast(@alignCast(active_buf.mapped.?));

    uploadPattern(
        weight,
        input[0 .. route_slots * K],
        counts[0..n_experts],
        ids[0 .. n_experts * n_tokens],
        active_blocks[0..2],
        M,
        K,
        n_tokens,
        k_used,
        n_experts,
    );
    var push = MoeColsDmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .expert_stride = expert_stride,
        .x_offset = 0,
        .y_offset = 0,
        .ids_stride = n_tokens,
        .x_route_divisor = 1,
        .accumulate = 0,
    };
    const infos = [_]vk.c.VkDescriptorBufferInfo{
        .{ .buffer = weight_buf.handle, .offset = 0, .range = weight_buf.size },
        .{ .buffer = input_buf.handle, .offset = 0, .range = input_buf.size },
        .{ .buffer = output_buf.handle, .offset = 0, .range = output_buf.size },
        .{ .buffer = counts_buf.handle, .offset = 0, .range = counts_buf.size },
        .{ .buffer = ids_buf.handle, .offset = 0, .range = ids_buf.size },
        .{ .buffer = active_buf.handle, .offset = 0, .range = active_buf.size },
    };

    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    for (0..2) |mode| {
        const accumulate = mode == 1;
        push.accumulate = if (accumulate) 1 else 0;
        for (0..route_slots * M) |i| {
            output[i] = if (accumulate)
                (@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.03125)
            else
                0.0;
        }

        if (mode > 0) try cmd.reset();
        try cmd.beginOneTime();
        cmd.pushDescAndDispatch(
            &pipe,
            instance.push_descriptor_fn,
            infos[0..],
            std.mem.asBytes(&push),
            @intCast((M + 1) / 2),
            2,
            1,
        );
        try cmd.end();
        try cmd.submitAndWait(instance.compute_queue);

        var max_diff: f32 = 0.0;
        var max_route: usize = 0;
        var max_row: usize = 0;
        for (0..route_slots) |route| {
            const expert_id: usize = if (route % 2 == 0) 1 else 2;
            const matrix_raw = weight[expert_id * expert_stride ..][0..expert_stride];
            const input_slice = input[route * K .. (route + 1) * K];
            for (0..M) |row| {
                dequantQ6KRow(matrix_raw, row, K, ref_row);
                var expected: f32 = if (accumulate)
                    (@as(f32, @floatFromInt(@as(i32, @intCast((route * M + row) % 17)) - 8)) * 0.03125)
                else
                    0.0;
                for (0..K) |i| expected += ref_row[i] * input_slice[i];
                const actual = output[route * M + row];
                const diff = @abs(expected - actual);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_route = route;
                    max_row = row;
                }
            }
        }

        std.debug.print("q6k_moe_cols {s} max_diff={d:.6} route={d} row={d}\n", .{
            if (accumulate) "accumulate" else "overwrite",
            max_diff,
            max_route,
            max_row,
        });
        if (max_diff > 0.05) return error.ValidationFailed;
    }
}
