import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { detectZincRtExecutionMode, parseArgs, parseLlamaCliTimings } from "./zinc_rt_autopilot";

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
