import { describe, expect, test } from "bun:test";

import { detectZincRtExecutionMode } from "./zinc_rt_autopilot";

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

  test("detects the Vulkan compatibility tier explicitly", () => {
    const output = [
      "info(zinc_rt): ZINC_RT vulkan_compat delegating full model inference to ./zig-out/bin/zinc-vulkan",
      "info(forward): Generated 96 tokens in 1070.8 ms — 89.65 tok/s",
      "info(zinc_rt): ZINC_RT M1 model_execution=vulkan_compat execution_tier=vulkan_compat delegated_backend=vulkan real_model_slice=1",
    ].join("\n");

    expect(detectZincRtExecutionMode(output)).toBe("vulkan_compat");
  });
});
