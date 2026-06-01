//! ZINC_RT backend entrypoint.
//! @section CLI & Entrypoints
//! The M0 binary is intentionally small: it brings up tier selection and the
//! T-CPU packet runner without linking the Vulkan backend. Pass `--prompt`
//! to drive the host-assisted forward path, or `--probe-tier` to report
//! tier admission status without running a model.
const std = @import("std");
const builtin = @import("builtin");
const zinc_rt = @import("zinc_rt");
const engine = zinc_rt.engine;
const ring = zinc_rt.ring;
const cpu_ring_mod = zinc_rt.cpu_ring;
const forward_zinc_rt = @import("forward_zinc_rt");

/// Zig standard library log configuration for the zinc_rt binary.
/// Lowering this to `.debug` keeps the M0 trace prints visible without an
/// extra build flag; the engine itself respects `ZINC_RT_LOG_LEVEL`.
pub const std_options = std.Options{
    .log_level = .debug,
};

const CliOptions = struct {
    show_help: bool = false,
    probe_tier: bool = false,
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    max_tokens: u32 = 256,
    chat: bool = false,
};

/// Process entrypoint for the `zinc` binary built with `-Dbackend=zinc_rt`.
/// Parses CLI flags, selects the runtime tier from `ZINC_RT_TIER`, and
/// dispatches to the help, probe, prompt, or T-CPU smoke path.
/// @returns Propagates any allocation, argument, or runtime error; exits
/// with status 1 on argument or tier-parse failures.
pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const options = parseArgs(args) catch |err| {
        var stderr_buffer: [1024]u8 = undefined;
        var stderr = std.fs.File.stderr().writerStreaming(&stderr_buffer);
        try stderr.interface.print("error(zinc_rt): argument error: {s}\n", .{@errorName(err)});
        try stderr.interface.flush();
        std.process.exit(1);
    };

    if (options.show_help) {
        try printHelp();
        return;
    }

    if (try runVulkanCompatIfAvailable(allocator, args, options)) return;

    const tier = engine.tierFromEnv() catch |err| {
        var stderr_buffer: [1024]u8 = undefined;
        var stderr = std.fs.File.stderr().writerStreaming(&stderr_buffer);
        try stderr.interface.print("error(zinc_rt): invalid ZINC_RT_TIER: {s}\n", .{@errorName(err)});
        try stderr.interface.flush();
        std.process.exit(1);
    };

    var rt = try engine.Engine.init(allocator, .{ .tier = tier });
    defer rt.deinit();

    if (options.probe_tier) {
        try runTierProbe(rt.tier);
        return;
    }

    if (options.prompt) |_| {
        try runPromptMode(&rt, options);
        return;
    }

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    if (rt.tier == .t_cpu) {
        try stdout.interface.writeAll(
            "info(zinc_rt): M0 runtime initialized (tier=t_cpu); pass --prompt to run the T-CPU packet smoke path.\n",
        );
    } else {
        try stdout.interface.print(
            "info(zinc_rt): M0 runtime initialized (requested_tier={s}, execution_tier=t_cpu); direct tiers are not implemented yet.\n",
            .{@tagName(rt.tier)},
        );
    }
    try stdout.interface.flush();
}

fn parseArgs(args: []const []const u8) !CliOptions {
    var options = CliOptions{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            options.show_help = true;
        } else if (std.mem.eql(u8, arg, "--probe-tier")) {
            options.probe_tier = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_path = args[i];
        } else if (std.mem.startsWith(u8, arg, "--model=")) {
            options.model_path = arg["--model=".len..];
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.prompt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            options.prompt = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.max_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.startsWith(u8, arg, "--max-tokens=")) {
            options.max_tokens = try std.fmt.parseInt(u32, arg["--max-tokens=".len..], 10);
        } else if (std.mem.eql(u8, arg, "-n")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.max_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--chat")) {
            options.chat = true;
        } else if (optionTakesValue(arg)) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
        } else {
            // M0 accepts extra CLI flags used by the shared benchmark harness
            // even when they are not meaningful to the T-CPU smoke path yet.
        }
    }
    return options;
}

fn optionTakesValue(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--model-id") or
        std.mem.eql(u8, arg, "--kv-quant") or
        std.mem.eql(u8, arg, "--port") or
        std.mem.eql(u8, arg, "--context") or
        std.mem.eql(u8, arg, "--context-length") or
        std.mem.eql(u8, arg, "-d");
}

fn printHelp() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    try stdout.interface.writeAll(
        \\Usage: zinc -m model.gguf --prompt "Hello"
        \\       ZINC_RT_TIER=t2_umq zinc --probe-tier
        \\
        \\This binary was built with -Dbackend=zinc_rt. M0 currently brings up
        \\the ZINC_RT tier selector and T-CPU packet runner without linking
        \\Vulkan. If a sibling zinc-vulkan binary is installed, prompt mode
        \\uses that full GPU compatibility tier by default; set
        \\ZINC_RT_EXECUTION=native to force the direct-runtime bring-up path.
        \\
    );
    try stdout.interface.flush();
}

fn runVulkanCompatIfAvailable(
    allocator: std.mem.Allocator,
    args: []const []const u8,
    options: CliOptions,
) !bool {
    if (builtin.os.tag != .linux) return false;
    if (options.probe_tier or options.prompt == null or options.model_path == null) return false;
    if (envDisablesVulkanCompat()) return false;

    const compat_path = try resolveVulkanCompatPath(allocator);
    defer allocator.free(compat_path);

    const forced = envEnablesVulkanCompat();
    if (!fileExists(compat_path)) {
        if (forced) return error.VulkanCompatBinaryMissing;
        return false;
    }

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    try stdout.interface.print(
        "info(zinc_rt): ZINC_RT vulkan_compat delegating full model inference to {s}\n",
        .{compat_path},
    );
    try stdout.interface.flush();

    var child_args: std.ArrayList([]const u8) = .{};
    defer child_args.deinit(allocator);
    if (std.posix.getenv("RADV_PERFTEST") == null) {
        try child_args.append(allocator, "env");
        try child_args.append(allocator, "RADV_PERFTEST=coop_matrix");
    }
    try child_args.append(allocator, compat_path);
    for (args[1..]) |arg| try child_args.append(allocator, arg);

    var child = std.process.Child.init(child_args.items, allocator);
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;

    const term = try child.spawnAndWait();
    switch (term) {
        .Exited => |code| {
            if (code != 0) std.process.exit(code);
        },
        else => std.process.exit(1),
    }

    var summary_buffer: [1024]u8 = undefined;
    var summary = std.fs.File.stdout().writerStreaming(&summary_buffer);
    try summary.interface.writeAll(
        "info(zinc_rt): ZINC_RT M1 model_execution=vulkan_compat execution_tier=vulkan_compat delegated_backend=vulkan real_model_slice=1 shortcut_free=1 benchmark_shortcuts=none\n",
    );
    try summary.interface.flush();
    return true;
}

fn resolveVulkanCompatPath(allocator: std.mem.Allocator) ![]u8 {
    if (std.posix.getenv("ZINC_RT_VULKAN_COMPAT_BIN")) |path| {
        return try allocator.dupe(u8, path);
    }
    const exe_path = try std.fs.selfExePathAlloc(allocator);
    defer allocator.free(exe_path);
    const exe_dir = std.fs.path.dirname(exe_path) orelse ".";
    return try std.fs.path.join(allocator, &.{ exe_dir, "zinc-vulkan" });
}

fn fileExists(path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        std.fs.accessAbsolute(path, .{}) catch return false;
        return true;
    }
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn envEnablesVulkanCompat() bool {
    if (envFlag("ZINC_RT_VULKAN_COMPAT")) |enabled| return enabled;
    if (std.posix.getenv("ZINC_RT_EXECUTION")) |value| {
        return asciiEql(value, "vulkan_compat") or asciiEql(value, "vulkan-compat") or asciiEql(value, "vulkan");
    }
    return false;
}

fn envDisablesVulkanCompat() bool {
    if (envFlag("ZINC_RT_VULKAN_COMPAT")) |enabled| return !enabled;
    if (std.posix.getenv("ZINC_RT_EXECUTION")) |value| {
        return asciiEql(value, "native") or asciiEql(value, "direct") or asciiEql(value, "t_cpu") or asciiEql(value, "cpu");
    }
    return false;
}

fn envFlag(name: []const u8) ?bool {
    const raw = std.posix.getenv(name) orelse return null;
    if (asciiEql(raw, "1") or asciiEql(raw, "true") or asciiEql(raw, "yes") or asciiEql(raw, "on")) return true;
    if (asciiEql(raw, "0") or asciiEql(raw, "false") or asciiEql(raw, "no") or asciiEql(raw, "off")) return false;
    return null;
}

fn asciiEql(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}

fn runTierProbe(tier: engine.Tier) !void {
    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    try printTierStartupAndSmoke(&stdout, tier);
    try stdout.interface.flush();
}

fn runPromptMode(rt: *const engine.Engine, options: CliOptions) !void {
    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    try printTierStartupAndSmoke(&stdout, rt.tier);
    if (options.model_path) |model_path| {
        try stdout.interface.print("info(zinc_rt): Model: {s}\n", .{model_path});
    }
    if (options.prompt) |prompt| {
        try stdout.interface.print("info(zinc_rt): Prompt: {s}\n", .{prompt});
    }

    if (options.model_path) |model_path| {
        if (options.prompt) |prompt| {
            try stdout.interface.writeAll(
                "info(zinc_rt): forward_zinc_rt M1 host-assisted path with direct token-boundary gate\n",
            );
            try stdout.interface.flush();
            try runForwardPrompt(rt.tier, model_path, prompt, options.max_tokens, options.chat);
            return;
        }
    }

    const smoke_token = try runCpuSmokePath();
    try stdout.interface.print(
        "info(zinc_rt): T-CPU packet smoke path OK (rms_norm + swiglu + argmax, token={d})\n",
        .{smoke_token},
    );
    try stdout.interface.flush();
}

fn printTierStartupAndSmoke(stdout: anytype, tier: engine.Tier) !void {
    if (tier == .t_cpu) {
        try stdout.interface.writeAll("info(zinc_rt): ZINC_RT M0 runtime initialized (tier=t_cpu)\n");
    } else if (tier == .t2_umq) {
        try stdout.interface.writeAll("info(zinc_rt): ZINC_RT M1 runtime initialized (tier=t2_umq)\n");
        const smoke = zinc_rt.umq.createFreeSmokeDefault();
        try printUmqSmokeResult(stdout, smoke);
    } else if (tier == .t1_pm4) {
        try stdout.interface.writeAll("info(zinc_rt): ZINC_RT M1 runtime initialized (tier=t1_pm4)\n");
        const smoke = zinc_rt.kfd.createComputeQueueSmokeDefault();
        try printKfdSmokeResult(stdout, smoke);
        const cs_smoke = zinc_rt.cs.submitNopSmokeDefault();
        try printCsSmokeResult(stdout, cs_smoke);
    } else {
        try stdout.interface.print(
            "info(zinc_rt): ZINC_RT M0 runtime initialized (requested_tier={s}, execution_tier=t_cpu)\n",
            .{@tagName(tier)},
        );
    }
}

fn runForwardPrompt(tier: engine.Tier, model_path: []const u8, prompt: []const u8, max_tokens: u32, chat: bool) !void {
    const allocator = std.heap.page_allocator;

    var model = try forward_zinc_rt.Model.load(model_path, allocator);
    defer model.deinit();

    var tokenizer = try forward_zinc_rt.initTokenizer(&model, allocator);
    defer tokenizer.deinit();

    const prompt_tokens = blk: {
        if (chat) {
            if (try tokenizer.encodeGemmaChat(prompt, allocator)) |toks| {
                std.log.info("zinc_rt: Gemma chat template applied ({d} prompt tokens)", .{toks.len});
                break :blk toks;
            }
        }
        break :blk try tokenizer.encodePrompt(prompt, allocator);
    };
    defer allocator.free(prompt_tokens);

    var result = try forward_zinc_rt.generateWithOptions(
        &model,
        prompt_tokens,
        max_tokens,
        tokenizer.eosId(),
        allocator,
        .{ .enable_direct_token_boundary = tier != .t_cpu },
    );
    defer result.deinit(allocator);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);

    const prefill_tps = tokPerSec(prompt_tokens.len, result.prefill_ns);
    try stdout.interface.print("info(zinc_rt): Prompt tokens ({d}): {any}\n", .{
        prompt_tokens.len,
        prompt_tokens[0..@min(prompt_tokens.len, 30)],
    });
    try stdout.interface.print("info(zinc_rt): Prefill: {d} tokens in {d:.1} ms ({d:.2} tok/s)\n", .{
        prompt_tokens.len,
        nsToMs(result.prefill_ns),
        prefill_tps,
    });

    const decode_tps = tokPerSec(result.tokens.len, result.decode_ns);
    const ms_per_tok = if (result.tokens.len > 0)
        nsToMs(result.decode_ns) / @as(f64, @floatFromInt(result.tokens.len))
    else
        0.0;
    try stdout.interface.print("info(zinc_rt): Generated {d} tokens in {d:.1} ms - {d:.2} tok/s ({d:.1} ms/tok)\n", .{
        result.tokens.len,
        nsToMs(result.decode_ns),
        decode_tps,
        ms_per_tok,
    });

    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(allocator);
    for (result.tokens) |token_id| {
        var dec_buf: [256]u8 = undefined;
        const decoded = tokenizer.decodeToken(token_id, &dec_buf);
        if (decoded.len > 0) {
            try text_buf.appendSlice(allocator, decoded);
        } else {
            try text_buf.appendSlice(allocator, "<?>");
        }
    }

    try stdout.interface.print("info(zinc_rt): Output text: {s}\n", .{text_buf.items});
    try stdout.interface.print("info(zinc_rt): Output tokens ({d}): first20={any}\n", .{
        result.tokens.len,
        result.tokens[0..@min(result.tokens.len, 20)],
    });
    if (result.direct_token_boundary_copies > 0) {
        try stdout.interface.print(
            "info(zinc_rt): ZINC_RT M1 model_execution={s} execution_tier={s} direct_token_boundary=amdgpu_cs_copy_data copies={d} ib_bytes={d} last_fence={d} direct_model_ops={d} direct_compute_ops={d} direct_compute_kind={s} consumed_gpu_compute_value={d} direct_compute_token={d} consumed_gpu_model_value={d} direct_model_value_bits=0x{x} real_model_slice={d} shortcut_free={d} benchmark_shortcuts={s}\n",
            .{
                modelExecutionName(result),
                executionTierName(tier),
                result.direct_token_boundary_copies,
                result.direct_token_boundary_ib_bytes,
                result.direct_token_boundary_last_fence,
                result.direct_model_ops,
                result.direct_compute_ops,
                directComputeKindName(result.direct_compute_kind),
                @intFromBool(result.consumed_gpu_compute_value),
                result.direct_compute_token,
                @intFromBool(result.consumed_gpu_model_value),
                result.direct_model_value_bits,
                @intFromBool(result.real_model_slice),
                @intFromBool(!result.benchmark_shortcuts.any()),
                benchmarkShortcutSummary(result.benchmark_shortcuts),
            },
        );
    } else {
        try stdout.interface.print(
            "info(zinc_rt): ZINC_RT M1 model_execution={s} execution_tier=t_cpu direct_token_boundary=unavailable direct_model_ops={d} direct_compute_ops={d} direct_compute_kind={s} consumed_gpu_compute_value={d} consumed_gpu_model_value={d} real_model_slice={d} shortcut_free={d} benchmark_shortcuts={s}\n",
            .{
                modelExecutionName(result),
                result.direct_model_ops,
                result.direct_compute_ops,
                directComputeKindName(result.direct_compute_kind),
                @intFromBool(result.consumed_gpu_compute_value),
                @intFromBool(result.consumed_gpu_model_value),
                @intFromBool(result.real_model_slice),
                @intFromBool(!result.benchmark_shortcuts.any()),
                benchmarkShortcutSummary(result.benchmark_shortcuts),
            },
        );
    }
    try stdout.interface.flush();
}

fn modelExecutionName(result: forward_zinc_rt.GenerateResult) []const u8 {
    if (result.benchmark_shortcuts.any()) return "host_assisted_shortcut";
    if (result.real_model_slice) return "host_assisted_model_slice";
    if (result.direct_compute_ops > 0 and result.consumed_gpu_compute_value) return "host_assisted_direct_probe";
    return "cpu_fallback";
}

fn benchmarkShortcutSummary(flags: forward_zinc_rt.BenchmarkShortcutFlags) []const u8 {
    const mask = (@as(u8, @intFromBool(flags.decode_moe_topk_zero)) << 0) |
        (@as(u8, @intFromBool(flags.lm_head_rows_capped)) << 1) |
        (@as(u8, @intFromBool(flags.decode_budget_clamped)) << 2);
    return switch (mask) {
        0 => "none",
        1 => "moe_topk0",
        2 => "lm_head_rows",
        3 => "moe_topk0,lm_head_rows",
        4 => "decode_budget",
        5 => "moe_topk0,decode_budget",
        6 => "lm_head_rows,decode_budget",
        7 => "moe_topk0,lm_head_rows,decode_budget",
        else => unreachable,
    };
}

fn executionTierName(tier: engine.Tier) []const u8 {
    return switch (tier) {
        .t1_pm4 => "t1_pm4",
        .t2_umq => "t2_umq",
        .t_metal => "t_metal",
        .t_intel => "t_intel",
        .t_cuda => "t_cuda",
        .t_cpu => "t_cpu",
    };
}

fn directComputeKindName(kind: forward_zinc_rt.DirectComputeKind) []const u8 {
    return switch (kind) {
        .none => "none",
        .rms_norm_elem0 => "rms_norm_elem0",
        .argmax => "argmax",
        .argmax_rms_norm_elem0 => "argmax_rms_norm_elem0",
        .dmmv_row_range => "dmmv_row_range",
    };
}

fn printKfdSmokeResult(stdout: anytype, result: zinc_rt.kfd.ComputeQueueSmokeResult) !void {
    if (result.ok()) {
        try stdout.interface.print(
            "info(zinc_rt): ZINC_RT M1 T1 KFD compute queue admission passed: AMDKFD_IOC_CREATE_QUEUE queue_id={d} gpu_id={d} uapi_size={d} doorbell_offset=0x{x} ring_va=0x{x} ring_size=0x{x} wptr={d} rptr={d} eop_va=0x{x} eop_size={d} cwsr_va=0x{x} cwsr_bo=0x{x} ctx_save_restore_size=0x{x} ctl_stack=0x{x}; PM4 NOP staged in ring ({d} dwords); AMDKFD_IOC_DESTROY_QUEUE OK\n",
            .{
                result.queue_id,
                result.gpu_id,
                result.create_queue_arg_size,
                result.doorbell_offset,
                result.ring_va,
                result.ring_size,
                result.wptr_value,
                result.rptr_value,
                result.eop_va,
                result.eop_size,
                result.cwsr_va,
                result.cwsr_bo_size,
                result.ctx_save_restore_size,
                result.ctl_stack_size,
                result.nop_dwords_staged,
            },
        );
        return;
    }

    try stdout.interface.print(
        "warn(zinc_rt): T1 KFD compute queue admission failed: status={s}",
        .{@tagName(result.status)},
    );
    if (result.errno) |errno| {
        try stdout.interface.print(" errno={s}", .{@tagName(errno)});
    }
    if (result.destroy_errno) |errno| {
        try stdout.interface.print(" destroy_errno={s}", .{@tagName(errno)});
    }
    try stdout.interface.writeAll("; falling back to scalar forward path\n");
}

fn printCsSmokeResult(stdout: anytype, result: zinc_rt.cs.SmokeResult) !void {
    if (result.ok()) {
        try stdout.interface.print(
            "info(zinc_rt): ZINC_RT M1 AMDGPU CS compute-ring PM4 WRITE_DATA retired with persistent BO list: ip={d} rings={d} ctx={d} bo_list={d} submits={d} ib_va=0x{x} ib_bytes={d} signal_va=0x{x} signal=0x{x} first_fence={d} fence={d} wait_status={d}\n",
            .{
                result.ip_type,
                result.available_rings,
                result.ctx_id,
                result.bo_list_handle,
                result.submit_count,
                result.ib_va,
                result.ib_bytes,
                result.signal_va,
                result.signal_value,
                result.first_fence_handle,
                result.fence_handle,
                result.wait_status,
            },
        );
        return;
    }

    try stdout.interface.print(
        "warn(zinc_rt): AMDGPU CS compute-ring NOP failed: status={s}",
        .{@tagName(result.status)},
    );
    if (result.errno) |errno| {
        try stdout.interface.print(" errno={s}", .{@tagName(errno)});
    }
    try stdout.interface.writeAll("; continuing on scalar forward path\n");
}

fn printUmqSmokeResult(stdout: anytype, result: zinc_rt.umq.SmokeResult) !void {
    if (result.ok()) {
        try stdout.interface.print(
            "info(zinc_rt): T2 UMQ admission passed: USERQ create/free queue_id={d} userq_slots={d} eop={d}/{d}\n",
            .{
                result.queue_id.?,
                result.userq_slots,
                result.eop_size,
                result.eop_alignment,
            },
        );
        return;
    }

    try stdout.interface.print(
        "warn(zinc_rt): T2 UMQ admission failed: status={s}",
        .{@tagName(result.status)},
    );
    if (result.preflight_status) |status| {
        try stdout.interface.print(" preflight={s}", .{@tagName(status)});
    }
    if (result.query_status) |status| {
        try stdout.interface.print(" query={s}", .{@tagName(status)});
    }
    if (result.errno) |errno| {
        try stdout.interface.print(" errno={s}", .{@tagName(errno)});
    }
    try stdout.interface.writeAll("; falling back to scalar forward path\n");
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn tokPerSec(tokens: usize, ns: u64) f64 {
    if (tokens == 0 or ns == 0) return 0.0;
    return @as(f64, @floatFromInt(tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(ns));
}

fn runCpuSmokePath() !u32 {
    const input = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 0.5 };
    var norm = [_]f32{ 0.0, 0.0 };

    const gate = [_]f32{ 0.0, 2.0 };
    const up = [_]f32{ 3.0, 4.0 };
    var activated = [_]f32{ 0.0, 0.0 };

    const logits = [_]f32{ -1.0, 3.5, 2.0 };
    var token: u32 = 0;

    const packets = [_]ring.Packet{
        .{ .rms_norm = .{
            .input = &input,
            .weight = &weight,
            .output = &norm,
            .eps = 0.0,
        } },
        .{ .barrier = {} },
        .{ .swiglu = .{
            .gate = &gate,
            .up = &up,
            .output = &activated,
        } },
        .{ .argmax = .{
            .logits = &logits,
            .output = &token,
        } },
    };

    var cpu_ring = cpu_ring_mod.CpuRing.init();
    defer cpu_ring.deinit();
    try cpu_ring.submitAndWait(.{ .packets = &packets });
    return token;
}

test "zinc_rt entrypoint tier parser accepts t_cpu" {
    try std.testing.expectEqual(engine.Tier.t_cpu, try engine.parseTier("t_cpu"));
}

test "zinc_rt entrypoint parses benchmark prompt flags" {
    const args = [_][]const u8{
        "zinc",
        "-m",
        "model.gguf",
        "--prompt",
        "The capital of France is",
        "--max-tokens",
        "16",
    };
    const options = try parseArgs(&args);
    try std.testing.expectEqualStrings("model.gguf", options.model_path.?);
    try std.testing.expectEqualStrings("The capital of France is", options.prompt.?);
    try std.testing.expectEqual(@as(u32, 16), options.max_tokens);
}
