//! CUDA per-layer debug dump for the qwen35 forward pass.
//!
//! Drives `src/compute/forward_cuda.zig` layer-by-layer via its public hooks
//! (`decodeStep(run_layers=false)` to embed, then `attentionLayerPub`/
//! `ssmLayerPub` + `ffnBlockPub` + `readHidden`) and prints the residual-stream
//! norm + first values after the embedding and after each mixer/FFN. Diff this
//! against a llama.cpp per-layer reference to pinpoint where the CUDA forward
//! diverges. Read-only w.r.t. the engine — uses only public methods.
//!
//!   zig build cuda-dbg -Dbackend=cuda -- [token] [model.gguf]
//! @section CUDA Runtime
const std = @import("std");
const device = @import("cuda/device.zig");
const loader = @import("model/loader_cuda.zig");
const forward = @import("compute/forward_cuda.zig");

const DEFAULT_MODEL = "/home/agent-zinc/workspace/Qwen3.5-9B-Q4_K_M.gguf";

fn stats(label: []const u8, v: []const f32) void {
    var ss: f64 = 0;
    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    var bad: usize = 0;
    for (v) |x| {
        if (!std.math.isFinite(x)) bad += 1;
        ss += @as(f64, x) * @as(f64, x);
        mn = @min(mn, x);
        mx = @max(mx, x);
    }
    std.debug.print("{s:<9} norm={d:>10.3} min={d:>9.3} max={d:>9.3} nan={d} [0..3]={d:.3} {d:.3} {d:.3}\n", .{ label, std.math.sqrt(ss), mn, mx, bad, v[0], v[1], v[2] });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next();
    const token: u32 = blk: {
        const a = args.next() orelse break :blk 100;
        break :blk std.fmt.parseInt(u32, a, 10) catch 100;
    };
    const model_path = args.next() orelse DEFAULT_MODEL;

    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try forward.ForwardCuda.init(allocator, &model, 512);
    defer fwd.deinit();

    const n = fwd.d.n_embd;
    const interval = fwd.d.full_attn_interval;
    const buf = try allocator.alloc(f32, @max(n, fwd.d.vocab));
    defer allocator.free(buf);

    std.debug.print("=== CUDA per-layer dump, token {d}, pos 0, {d} layers (interval {d}) ===\n", .{ token, fwd.d.n_layers, interval });

    // Embed only: run_layers=false embeds then runs the tail (on norm_buf), so
    // self.hidden is left holding the embedding.
    _ = try fwd.decodeStep(token, 0, false);
    fwd.readHidden(buf[0..n]);
    stats("embed", buf[0..n]);

    var L: u32 = 0;
    while (L < fwd.d.n_layers) : (L += 1) {
        const is_attn = ((L + 1) % interval) == 0;
        if (is_attn) try fwd.attentionLayerPub(L, 0) else try fwd.ssmLayerPub(L);
        fwd.readHidden(buf[0..n]);
        var lbl: [24]u8 = undefined;
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-{s}", .{ L, if (is_attn) "att" else "ssm" }), buf[0..n]);
        try fwd.ffnBlockPub(L);
        fwd.readHidden(buf[0..n]);
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-ffn", .{L}), buf[0..n]);
    }
}
