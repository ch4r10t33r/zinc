import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import {
  decideMigrateKeep,
  detectZincRtExecutionMode,
  hasDirectDecodeModelSliceEvidence,
  isShortcutFreeZincRtOutput,
  parseArgs,
  parseLlamaCliTimings,
  type ABBenchmark,
  type BenchmarkResult,
} from "./zinc_rt_autopilot";

function benchmarkResult(overrides: Partial<BenchmarkResult> = {}): BenchmarkResult {
  return {
    decodeTps: 3.8,
    prefillTps: 3.4,
    decodeSamples: [3.8],
    prefillSamples: [3.4],
    buildExitCode: 0,
    buildOutput: "",
    runOutput: "",
    runExitCode: 0,
    coherentText: false,
    garbageOutput: false,
    tokensGenerated: 8,
    bandwidthUtil: null,
    effectiveBW: null,
    error: null,
    backendFlagRecognized: true,
    ...overrides,
  };
}

function abBenchmark(zincRt: BenchmarkResult, baseline: BenchmarkResult = benchmarkResult({ decodeTps: 77, coherentText: true })): ABBenchmark {
  return {
    comparisonTarget: "llama",
    vulkan: baseline,
    zinc_rt: zincRt,
    ratio: zincRt.decodeTps != null && baseline.decodeTps != null ? zincRt.decodeTps / baseline.decodeTps : null,
    prefillRatio: zincRt.prefillTps != null && baseline.prefillTps != null ? zincRt.prefillTps / baseline.prefillTps : null,
  };
}

describe("detectZincRtExecutionMode", () => {
  test("treats direct admission plus T-CPU forward as CPU fallback after admission", () => {
    const output = [
      "info(zinc_rt): ZINC_RT M1 runtime initialized (tier=t1_pm4, execution_tier=t_cpu_after_admission)",
      "info(zinc_rt): ZINC_RT M1 T1 KFD compute queue admission passed",
      "info(zinc_rt): forward_zinc_rt M0 T-CPU scalar forward path",
    ].join("\n");

    expect(detectZincRtExecutionMode(output)).toBe("cpu_after_admission");
  });

  test("detects plain T-CPU execution from forward_zinc_rt logs", () => {
    expect(detectZincRtExecutionMode("info(zinc_rt): forward_zinc_rt M0 T-CPU scalar forward path")).toBe("cpu");
  });

  test("does not treat direct tier admission as direct compute by itself", () => {
    expect(detectZincRtExecutionMode("info(zinc_rt): runtime initialized execution_tier=t1_pm4")).toBe("unknown");
  });

  test("treats scalar path with direct copy gates as CPU fallback after admission", () => {
    const output = [
      "info(zinc_rt): forward_zinc_rt M1 scalar path with direct token-boundary gate",
      "info(zinc_rt): ZINC_RT M1 model_execution=cpu_fallback execution_tier=t1_pm4 direct_token_boundary=amdgpu_cs_copy_data direct_model_ops=1 direct_compute_ops=0 consumed_gpu_model_value=1",
    ].join("\n");

    expect(detectZincRtExecutionMode(output)).toBe("cpu_after_admission");
  });

  test("detects direct execution only from explicit direct compute evidence", () => {
    const output = [
      "info(zinc_rt): runtime initialized execution_tier=t1_pm4",
      "info(zinc_rt): ZINC_RT M1 model_execution=direct execution_tier=t1_pm4 direct_compute_ops=1 direct_compute_kind=lm_head_row_range consumed_gpu_model_value=1",
    ].join("\n");

    expect(detectZincRtExecutionMode(output)).toBe("direct");
  });
});

describe("M1 migration keep signals", () => {
  test("recognizes decode-phase direct model slice evidence", () => {
    expect(hasDirectDecodeModelSliceEvidence([
      "info(zinc_rt_forward): M1 AMDGPU CS direct compute consumed: direct_compute_ops=2 direct_compute_kind=dmmv_row_range op=lm_head_q4_0_best_row phase=decode consumed_gpu_model_value=1",
    ].join("\n"))).toBe(true);

    expect(hasDirectDecodeModelSliceEvidence([
      "info(zinc_rt_forward): M1 AMDGPU CS direct compute consumed: direct_compute_ops=2 direct_compute_kind=dmmv_row_range op=lm_head_q4_0_best_row phase=prefill consumed_gpu_model_value=1",
    ].join("\n"))).toBe(false);

    expect(hasDirectDecodeModelSliceEvidence("direct_decode_model_slices=1")).toBe(true);
  });

  test("keeps new decode-phase model-slice evidence with bounded slowdown", () => {
    const before = abBenchmark(benchmarkResult({
      decodeTps: 3.82,
      runOutput: "model_execution=host_assisted benchmark_shortcuts=decode_budget shortcut_free=0",
    }));
    const after = abBenchmark(benchmarkResult({
      decodeTps: 3.75,
      runOutput: "model_execution=host_assisted direct_compute_kind=dmmv_row_range op=lm_head_q4_0_best_row phase=decode consumed_gpu_model_value=1 direct_decode_model_slices=1 benchmark_shortcuts=decode_budget shortcut_free=0",
    }));

    const decision = decideMigrateKeep(before, after, null);
    expect(decision.keep).toBe(true);
    expect(decision.reason).toContain("decode-phase");
  });

  test("keeps incremental decode-phase model-slice evidence with bounded slowdown", () => {
    const before = abBenchmark(benchmarkResult({
      decodeTps: 3.37,
      runOutput: [
        "info(zinc_rt_forward): M1 AMDGPU CS direct model slice consumed: direct_compute_ops=5 direct_compute_kind=dmmv_row_range op=lm_head_q4_0_best_row phase=decode row=13 cols=4096 cpu=20.351450 gpu=20.351440 abs_delta=0.000010",
        "info(zinc_rt): ZINC_RT M1 model_execution=host_assisted_model_slice direct_decode_model_slices=1 benchmark_shortcuts=none shortcut_free=1",
      ].join("\n"),
    }));
    const after = abBenchmark(benchmarkResult({
      decodeTps: 3.37,
      runOutput: [
        "info(zinc_rt_forward): M1 AMDGPU CS direct model slice consumed: direct_compute_ops=4 direct_compute_kind=dmmv_row_range op=ssm_alpha_q4_0_row0 phase=decode layer=0 row=0 cols=4096 cpu=3.656229 gpu=3.656232 abs_delta=0.000003",
        "info(zinc_rt_forward): M1 AMDGPU CS direct model slice consumed: direct_compute_ops=6 direct_compute_kind=dmmv_row_range op=lm_head_q4_0_best_row phase=decode row=13 cols=4096 cpu=20.351450 gpu=20.351423 abs_delta=0.000027",
        "info(zinc_rt): ZINC_RT M1 model_execution=host_assisted_model_slice direct_decode_model_slices=2 benchmark_shortcuts=none shortcut_free=1",
      ].join("\n"),
    }));

    const decision = decideMigrateKeep(before, after, null);
    expect(decision.keep).toBe(true);
    expect(decision.reason).toContain("decode-phase");
  });

  test("rejects incremental slice widening that drifts far below the best checkpoint", () => {
    const before = abBenchmark(benchmarkResult({
      decodeTps: 32.42,
      coherentText: true,
      runOutput: [
        "info(zinc_rt_forward): M1 AMDGPU CS direct model slice consumed: direct_compute_ops=7084 direct_compute_kind=dmmv_row_range op=router_q8_0_row_range_parallel64_trusted phase=decode consumed_gpu_model_value=1",
        "info(zinc_rt): ZINC_RT M1 model_execution=host_assisted_model_slice direct_decode_model_slices=7084 benchmark_shortcuts=none shortcut_free=1",
      ].join("\n"),
    }));
    const after = abBenchmark(benchmarkResult({
      decodeTps: 32.18,
      coherentText: true,
      runOutput: [
        "info(zinc_rt_forward): M1 AMDGPU CS direct model slice consumed: direct_compute_ops=7804 direct_compute_kind=dmmv_row_range op=router_q8_0_row_range_parallel64_trusted phase=decode consumed_gpu_model_value=1",
        "info(zinc_rt): ZINC_RT M1 model_execution=host_assisted_model_slice direct_decode_model_slices=7804 benchmark_shortcuts=none shortcut_free=1",
      ].join("\n"),
    }));

    const decision = decideMigrateKeep(before, after, 34.95);
    expect(decision.keep).toBe(false);
  });

  test("keeps performance recovery that preserves consumed decode-slice evidence", () => {
    const before = abBenchmark(benchmarkResult({
      decodeTps: 32.2,
      coherentText: true,
      runOutput: [
        "info(zinc_rt_forward): M1 AMDGPU CS direct model slice consumed: direct_compute_ops=8188 direct_compute_kind=dmmv_row_range op=router_q8_0_row_range_parallel64_trusted phase=decode consumed_gpu_model_value=1",
        "info(zinc_rt): ZINC_RT M1 model_execution=host_assisted_model_slice direct_decode_model_slices=8188 benchmark_shortcuts=none shortcut_free=1",
      ].join("\n"),
    }));
    const after = abBenchmark(benchmarkResult({
      decodeTps: 32.65,
      coherentText: true,
      runOutput: [
        "info(zinc_rt_forward): M1 AMDGPU CS direct model slice consumed: direct_compute_ops=4096 direct_compute_kind=dmmv_row_range op=router_q8_0_row_range_parallel64_trusted phase=decode consumed_gpu_model_value=1",
        "info(zinc_rt): ZINC_RT M1 model_execution=host_assisted_model_slice direct_decode_model_slices=4096 benchmark_shortcuts=none shortcut_free=1",
      ].join("\n"),
    }));

    const decision = decideMigrateKeep(before, after, 34.95);
    expect(decision.keep).toBe(true);
    expect(decision.reason).toContain("performance recovery");
  });

  test("keeps shortcut-free M1 measurement cleanup despite scalar slowdown", () => {
    const before = abBenchmark(benchmarkResult({
      decodeTps: 3.81,
      tokensGenerated: 8,
      runOutput: "model_execution=host_assisted direct_compute_kind=dmmv_row_range consumed_gpu_model_value=1 benchmark_shortcuts=decode_budget shortcut_free=0 path clamped decode budget",
    }));
    const after = abBenchmark(benchmarkResult({
      decodeTps: 3.36,
      tokensGenerated: 96,
      runOutput: "model_execution=host_assisted_model_slice real_model_slice=1 direct_compute_kind=dmmv_row_range consumed_gpu_model_value=1 benchmark_shortcuts=none shortcut_free=1",
    }));

    expect(isShortcutFreeZincRtOutput(after.zinc_rt.runOutput)).toBe(true);
    const decision = decideMigrateKeep(before, after, null);
    expect(decision.keep).toBe(true);
    expect(decision.reason).toContain("shortcut-free");
  });
});

describe("zinc_rt autopilot CLI and baselines", () => {
  test("parseArgs supports llama baseline and a shared max token cap", () => {
    const args = parseArgs([
      "--baseline",
      "llama",
      "--max-tokens",
      "32",
      "--runs-per-binary",
      "1",
      "--target-ratio",
      "1.03",
    ]);

    expect(args.comparisonTarget).toBe("llama");
    expect(args.maxTokens).toBe(32);
    expect(args.runsPerBinary).toBe(1);
    expect(args.targetRatio).toBe(1.03);
  });

  test("parseLlamaCliTimings extracts prompt and decode throughput", () => {
    const parsed = parseLlamaCliTimings(`
llama_print_timings: prompt eval time =   152.58 ms /    16 tokens (    9.54 ms per token,   104.87 tokens per second)
llama_print_timings:        eval time =  1276.72 ms /    96 runs   (   13.30 ms per token,    75.20 tokens per second)
`);

    expect(parsed.prefillTps).toBe(104.87);
    expect(parsed.decodeTps).toBe(75.2);
    expect(parsed.generatedTokens).toBe(96);
  });

  test("parseLlamaCliTimings accepts bracket-style perf summaries", () => {
    const parsed = parseLlamaCliTimings("[ Prompt: 171.2 t/s | Generation: 78.2 t/s ]");
    expect(parsed.prefillTps).toBe(171.2);
    expect(parsed.decodeTps).toBe(78.2);
    expect(parsed.generatedTokens).toBe(0);
  });

  test("remote sync commands exclude local secrets and run state", () => {
    const autopilot = readFileSync(new URL("./zinc_rt_autopilot.ts", import.meta.url), "utf8");
    expect(autopilot).toContain('"--exclude", ".env"');
    expect(autopilot).toContain('"--exclude", ".env.*"');
    expect(autopilot).toContain('"--exclude", ".zinc_rt_autopilot"');
    expect(autopilot).toContain("--exclude '.env'");
    expect(autopilot).toContain("--exclude '.env.*'");
  });
});
