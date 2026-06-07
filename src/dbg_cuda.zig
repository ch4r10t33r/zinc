//! CUDA forward-pass debug harness for the qwen35 hybrid-SSM model. Two modes:
//!
//!   zig build cuda-dbg -- <token> [model.gguf]
//!       Per-layer residual-norm dump at pos 0 (used to pinpoint the attention
//!       gate bug: diff vs a llama.cpp eval-callback `l_out-N` reference).
//!
//!   zig build cuda-dbg -- gen <id,id,...> <ngen> [model.gguf]
//!       Autoregressive greedy generation from a prompt token-id list. Prefills
//!       the ids (exercising pos>0 RoPE + multi-entry attention + SSM state
//!       carry), then greedily emits ngen tokens. Diff GEN_IDS vs `/tmp/gen`
//!       (llama.cpp greedy) to validate the full decode path beyond pos 0.
//!
//! Read-only w.r.t. the engine — uses only public ForwardCuda methods.
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
    const first = args.next() orelse "100";

    if (std.mem.eql(u8, first, "gen")) {
        const ids_arg = args.next() orelse "760,6511,314,9338,369";
        const ngen: u32 = std.fmt.parseInt(u32, args.next() orelse "16", 10) catch 16;
        const model_path = args.next() orelse DEFAULT_MODEL;
        try genMode(allocator, ids_arg, ngen, model_path);
    } else if (std.mem.eql(u8, first, "prof")) {
        const model_path = args.next() orelse DEFAULT_MODEL;
        try profMode(allocator, model_path);
    } else if (std.mem.eql(u8, first, "logits")) {
        const token: u32 = std.fmt.parseInt(u32, args.next() orelse "100", 10) catch 100;
        const out_path = args.next() orelse "/tmp/zinc_logits.bin";
        const model_path = args.next() orelse DEFAULT_MODEL;
        try logitsMode(allocator, token, out_path, model_path);
    } else {
        const token: u32 = std.fmt.parseInt(u32, first, 10) catch 100;
        const model_path = args.next() orelse DEFAULT_MODEL;
        try dumpMode(allocator, token, model_path);
    }
}

/// Autoregressive greedy generation from a comma-separated prompt token list.
fn genMode(allocator: std.mem.Allocator, ids_arg: []const u8, ngen: u32, model_path: []const u8) !void {
    var prompt_buf: [256]u32 = undefined;
    var np: usize = 0;
    var it = std.mem.splitScalar(u8, ids_arg, ',');
    while (it.next()) |s| {
        const trimmed = std.mem.trim(u8, s, " ");
        if (trimmed.len == 0 or np >= prompt_buf.len) continue;
        prompt_buf[np] = try std.fmt.parseInt(u32, trimmed, 10);
        np += 1;
    }
    const prompt = prompt_buf[0..np];

    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try forward.ForwardCuda.init(allocator, &model, 512);
    defer fwd.deinit();

    std.debug.print("PROMPT_IDS:", .{});
    for (prompt, 0..) |t, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", t });
    std.debug.print("\n", .{});

    // Prefill: process each prompt token at its position; the argmax after the
    // last prompt token is the first generated token.
    var pos: u32 = 0;
    var tok: u32 = 0;
    for (prompt) |t| {
        tok = try fwd.decodeStep(t, pos, true);
        pos += 1;
    }

    std.debug.print("GEN_IDS:{d}", .{tok});
    var timer = try std.time.Timer.start(); // steady-state: exclude prefill + first token
    var g: u32 = 1;
    while (g < ngen) : (g += 1) {
        const next = try fwd.decodeStep(tok, pos, true);
        pos += 1;
        std.debug.print(",{d}", .{next});
        tok = next;
    }
    const ns = timer.read();
    std.debug.print("\n", .{});
    if (ngen > 1) {
        const secs = @as(f64, @floatFromInt(ns)) / 1e9;
        const steps: f64 = @floatFromInt(ngen - 1);
        std.debug.print("DECODE: {d} tokens in {d:.3}s = {d:.2} tok/s (correctness-first, sync-per-layer)\n", .{ ngen - 1, secs, steps / secs });
    }
}

/// Decode-bottleneck profile: splits per-token time into embed+tail vs the
/// 32-layer stack (via the run_layers flag) to size the sync-per-layer overhead
/// that the async CUstream/CUevent ring would remove. Read-only.
fn profMode(allocator: std.mem.Allocator, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try forward.ForwardCuda.init(allocator, &model, 512);
    defer fwd.deinit();

    const K: u32 = 40;
    // warmup
    var i: u32 = 0;
    while (i < 5) : (i += 1) _ = try fwd.decodeStep(100, i, true);

    var t = try std.time.Timer.start();
    i = 0;
    while (i < K) : (i += 1) _ = try fwd.decodeStep(100, i, false); // embed + tail only
    const et_ns = t.read();

    t.reset();
    i = 0;
    while (i < K) : (i += 1) _ = try fwd.decodeStep(100, i, true); // full forward
    const full_ns = t.read();

    const et_ms = @as(f64, @floatFromInt(et_ns)) / 1e6 / @as(f64, @floatFromInt(K));
    const full_ms = @as(f64, @floatFromInt(full_ns)) / 1e6 / @as(f64, @floatFromInt(K));
    const layers_ms = full_ms - et_ms;
    // ~65 commitAndWait/token: 32 layers x (mixer + ffn) + 1 tail.
    const commits: f64 = 65;
    std.debug.print("=== decode profile (4090, {d} iters) ===\n", .{K});
    std.debug.print("embed+tail : {d:.3} ms/token\n", .{et_ms});
    std.debug.print("32 layers  : {d:.3} ms/token\n", .{layers_ms});
    std.debug.print("full       : {d:.3} ms/token  ({d:.2} tok/s)\n", .{ full_ms, 1000.0 / full_ms });
    std.debug.print("~per-commit: {d:.3} ms  ({d:.0} sync round-trips/token)\n", .{ full_ms / commits, commits });
    std.debug.print("(async ring batches these into ~1 submit/token — the headroom to the 97 t/s bar)\n", .{});
}

/// Dump the full vocab logit vector for `token` at pos 0 to a raw-f32 file,
/// for numerical-fidelity comparison vs a llama.cpp logit dump.
fn logitsMode(allocator: std.mem.Allocator, token: u32, out_path: []const u8, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try forward.ForwardCuda.init(allocator, &model, 512);
    defer fwd.deinit();

    const vocab = fwd.d.vocab;
    const buf = try allocator.alloc(f32, vocab);
    defer allocator.free(buf);
    _ = try fwd.decodeStep(token, 0, true);
    fwd.readLogits(buf);

    const f = try std.fs.cwd().createFile(out_path, .{});
    defer f.close();
    try f.writeAll(std.mem.sliceAsBytes(buf));

    var bi: usize = 0;
    var bm = buf[0];
    for (buf, 0..) |v, i| if (v > bm) {
        bm = v;
        bi = i;
    };
    std.debug.print("wrote {d} logits to {s}; argmax={d} ({d:.4})\n", .{ vocab, out_path, bi, bm });
}

/// Per-layer residual-norm dump at pos 0 (single token).
fn dumpMode(allocator: std.mem.Allocator, token: u32, model_path: []const u8) !void {
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
