import { expect, test } from "bun:test";

import {
  buildArtifact,
  buildComparison,
  buildMeasurementPhases,
  benchmarkFailureReason,
  canonicalModelIdFromPath,
  compareModelsByName,
  detectRdnaServerStartupFailure,
  DEFAULT_LOCAL_MODEL_ROOT,
  defaultIntelCases,
  defaultMetalCases,
  defaultMaxTokensForModelId,
  defaultPromptForModelId,
  defaultRdnaCases,
  defaultScenarioDefsForModel,
  guessFamily,
  localZincCommand,
  mergeArtifacts,
  parseArgs,
  parseDotEnv,
  parseLlamaCliOutput,
  parseLlamaCppVersionOutput,
  parseOpenAiCompletionOutput,
  parseZincCliOutput,
  prefersChatPrompt,
  intelZincCommand,
  rdnaZincCommand,
  resolveLocalLlamaServer,
  summarizeValues,
} from "./performance_suite.mjs";

test("parseArgs reads suite options", () => {
  const args = parseArgs([
    "--target",
    "metal",
    "--runs",
    "5",
    "--warmup",
    "2",
    "--models",
    "gemma4-26b-a4b-q4k-m,qwen35-9b-q4k-m",
    "--llama-cli",
    "/tmp/llama-cli",
    "--llama-server",
    "/tmp/llama-server",
    "--phase",
    "zinc",
    "--no-site-write",
  ]);

  expect(args.target).toBe("metal");
  expect(args.runs).toBe(5);
  expect(args.warmupRuns).toBe(2);
  expect(args.llamaCli).toBe("/tmp/llama-cli");
  expect(args.llamaServer).toBe("/tmp/llama-server");
  expect(args.phase).toBe("zinc");
  expect(args.writeSiteData).toBe(false);
  expect(args.models && [...args.models]).toEqual(["gemma4-26b-a4b-q4k-m", "qwen35-9b-q4k-m"]);
});

test("parseArgs reads Intel suite options", () => {
  const args = parseArgs([
    "--target",
    "intel",
    "--intel-sync",
    "--intel-build",
    "--intel-start-llama",
    "--intel-model-root",
    "/home/tempuser/.cache/zinc/models/models",
    "--intel-workdir",
    "/home/tempuser/zinc-intel-loop",
    "--intel-xdg-cache-home",
    "/home/tempuser/.cache",
    "--intel-remote-libc-conf",
    "/workspace/zinc/.build-support/libc.conf",
  ]);

  expect(args.target).toBe("intel");
  expect(args.intelSync).toBe(true);
  expect(args.intelBuild).toBe(true);
  expect(args.intelStartLlama).toBe(true);
  expect(args.intelModelRoot).toBe("/home/tempuser/.cache/zinc/models/models");
  expect(args.intelWorkdir).toBe("/home/tempuser/zinc-intel-loop");
  expect(args.intelXdgCacheHome).toBe("/home/tempuser/.cache");
  expect(args.intelRemoteLibcConf).toBe("/workspace/zinc/.build-support/libc.conf");
});

test("parseArgs enables discovery mode", () => {
  const args = parseArgs(["--target", "metal", "--discover-models"]);
  expect(args.discoverModels).toBe(true);
});

test("parseArgs enables managed Metal pulls", () => {
  const args = parseArgs(["--target", "metal", "--metal-pull-missing"]);
  expect(args.metalPullMissing).toBe(true);
});

test("resolveLocalLlamaServer prefers explicit path, then PATH, then docker fallback", () => {
  expect(resolveLocalLlamaServer({ llamaServer: "/tmp/explicit" }, "/tmp/path", "/tmp/docker")).toBe("/tmp/explicit");
  expect(resolveLocalLlamaServer({ llamaServer: null }, "/tmp/path", "/tmp/docker")).toBe("/tmp/path");
  expect(resolveLocalLlamaServer({ llamaServer: null }, null, "/tmp/docker")).toBe("/tmp/docker");
});

test("Gemma uses the chat prompt path in the performance suite", () => {
  expect(prefersChatPrompt("gemma4-26b-a4b-q4k-m")).toBe(true);
  expect(defaultPromptForModelId("gemma4-26b-a4b-q4k-m")).toContain("benchmark screenshots");
  expect(defaultMaxTokensForModelId("gemma4-26b-a4b-q4k-m")).toBe(96);
  expect(prefersChatPrompt("qwen35-9b-q4k-m")).toBe(false);
  expect(defaultPromptForModelId("qwen35-9b-q4k-m")).toContain("Developer question");
  expect(defaultMaxTokensForModelId("qwen35-9b-q4k-m")).toBe(96);
});

test("default Metal cases use managed cache ids and include Qwen 3.6", () => {
  const cases = defaultMetalCases("/tmp/models");

  const qwen36 = cases.find((entry) => entry.id === "qwen36-35b-a3b-q4k-xl");
  expect(qwen36?.model_id).toBe("qwen36-35b-a3b-q4k-xl");
  expect(qwen36?.model_path).toBe("/tmp/models/qwen36-35b-a3b-q4k-xl/model.gguf");

  const qwen36Dense = cases.find((entry) => entry.id === "qwen36-27b-q4k-m");
  expect(qwen36Dense?.model_id).toBe("qwen36-27b-q4k-m");
  expect(qwen36Dense?.model_path).toBe("/tmp/models/qwen36-27b-q4k-m/model.gguf");
});

test("default RDNA cases include Qwen 3.5 9B and Qwen 3.6 27B dense", () => {
  const cases = defaultRdnaCases("/root/models");
  const qwen36Dense = cases.find((entry) => entry.id === "qwen36-27b-q4k-m");
  const qwen35 = cases.find((entry) => entry.id === "qwen35-9b-q4k-m");

  expect(qwen36Dense?.model_path).toBe("/root/models/Qwen3.6-27B-Q4_K_M.gguf");
  expect(qwen36Dense?.prompt_mode).toBe("raw");
  expect(qwen36Dense?.prompt).toContain("Developer question");
  expect(qwen36Dense?.max_tokens).toBe(96);

  expect(qwen35?.model_path).toBe("/root/models/Qwen3.5-9B-Q4_K_M.gguf");
  expect(qwen35?.prompt_mode).toBe("raw");
  expect(qwen35?.prompt).toContain("Developer question");
  expect(qwen35?.max_tokens).toBe(96);
});

test("default Intel cases use the remote managed cache layout", () => {
  const cases = defaultIntelCases("/remote/cache");
  const qwen = cases.find((entry) => entry.id === "qwen35-9b-q4k-m");

  expect(qwen?.model_path).toBe("/remote/cache/qwen35-9b-q4k-m/model.gguf");
  expect(qwen?.prompt_mode).toBe("raw");
});

test("performance suite canonicalizes and labels Qwen 3.6 GGUFs", () => {
  expect(canonicalModelIdFromPath("/tmp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")).toBe("qwen36-35b-a3b-q4k-xl");
  expect(canonicalModelIdFromPath("/tmp/Qwen3.6-27B-Q4_K_M.gguf")).toBe("qwen36-27b-q4k-m");
  expect(canonicalModelIdFromPath("/tmp/Qwen_Qwen3.6-27B-Q4_K_M.gguf")).toBe("qwen36-27b-q4k-m");
  expect(canonicalModelIdFromPath("/tmp/models/qwen36-35b-a3b-q4k-xl/model.gguf")).toBe("qwen36-35b-a3b-q4k-xl");
  expect(guessFamily("qwen36-35b-a3b-q4k-xl")).toBe("Qwen 3.6");
  expect(guessFamily("qwen36-27b-q4k-m")).toBe("Qwen 3.6");
});

test("performance suite canonicalizes RDNA Gemma GGUF filenames to published ids", () => {
  expect(canonicalModelIdFromPath("/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")).toBe("gemma4-26b-a4b-q4k-m");
  expect(canonicalModelIdFromPath("/root/models/gemma-4-31B-it-Q4_K_M.gguf")).toBe("gemma4-31b-q4k-m");
  expect(guessFamily("gemma4-26b-a4b-q4k-m")).toBe("Gemma 4");
});

test("local ZINC command prefers managed model ids when using the default cache", () => {
  const cmd = localZincCommand({
    model_id: "qwen35-9b-q4k-m",
    model_path: `${DEFAULT_LOCAL_MODEL_ROOT}/qwen35-9b-q4k-m/model.gguf`,
    prompt_mode: "raw",
    prompt: "The capital of France is",
    max_tokens: 8,
  });
  expect(cmd).toContain("--model-id qwen35-9b-q4k-m");
  expect(cmd).not.toContain(" -m ");
});

test("RDNA ZINC command preserves chat prompt mode", () => {
  const cmd = rdnaZincCommand({
    model_path: "/root/models/gpt.gguf",
    prompt_mode: "chat",
    prompt: "What is the capital of France?",
    max_tokens: 48,
  }, {
    host: "bench.local",
    user: "root",
    port: "2222",
    workdir: "/root/zinc",
  });

  expect(cmd).toContain("./zig-out/bin/zinc");
  expect(cmd).toContain("--chat");
  expect(cmd).toContain("--prompt");
});

test("Intel ZINC command does not inject RDNA-specific environment", () => {
  const cmd = intelZincCommand({
    model_path: "/home/tempuser/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf",
    prompt_mode: "raw",
    prompt: "The capital of France is",
    max_tokens: 8,
  }, {
    host: "intel.local",
    user: "tempuser",
    port: "8888",
    workdir: "/home/tempuser/zinc",
    env: {},
  });

  expect(cmd).toContain("tempuser@intel.local");
  expect(cmd).toContain("./zig-out/bin/zinc");
  expect(cmd).not.toContain("RADV_PERFTEST");
  expect(cmd).not.toContain("--chat");
});

test("RDNA startup failure detection spots unsupported model architecture logs", () => {
  const failure = detectRdnaServerStartupFailure(`
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'gemma4'
srv load_model: failed to load model
main: exiting due to model loading error
`);

  expect(failure).toBe("unknown model architecture: 'gemma4'");
  expect(detectRdnaServerStartupFailure("server ready")).toBeNull();
});

test("benchmark failure reasons do not publish shell commands", () => {
  const error = new Error("Command failed (1): remote benchmark command with private args\nprivate details");
  expect(benchmarkFailureReason("ZINC run failed", error)).toBe("ZINC run failed: command exited unsuccessfully (1).");
  const diagnostic = new Error("Command failed (1): remote benchmark command with private args\nerr(zinc): Failed to init inference engine: QueueSubmitFailed");
  expect(benchmarkFailureReason("ZINC run failed", diagnostic)).toBe("ZINC run failed: err(zinc): Failed to init inference engine: QueueSubmitFailed");
  expect(benchmarkFailureReason("Intel baseline failed", new Error("Remote server failed to start"))).toBe("Intel baseline failed: Remote server failed to start");
});

test("benchmark suite uses a multi-scenario matrix instead of a single prompt", () => {
  const qwen = defaultScenarioDefsForModel("qwen35-9b-q4k-m", "raw", defaultPromptForModelId("qwen35-9b-q4k-m"));
  expect(qwen.map((scenario) => scenario.id)).toEqual(["core", "context-medium", "context-long", "decode-extended"]);
  expect(qwen.map((scenario) => scenario.label)).toEqual(["Quick Chat", "Coding Review", "Incident Context", "Long Coding Draft"]);
  expect(qwen[1]?.prompt).not.toBe(qwen[0]?.prompt);
  expect(qwen[1]?.prompt).toContain("src/cache.ts");
  expect(qwen[1]?.max_tokens).toBe(160);
  expect(qwen[2]?.prompt).toContain("Incident notes");
  expect(qwen[2]?.max_tokens).toBe(128);
  expect(qwen[3]?.prompt).toContain("stable benchmark preset");
  expect(qwen[3]?.max_tokens).toBe(256);
});

test("benchmark suite measures all ZINC scenarios before starting baselines", () => {
  const phases = buildMeasurementPhases("qwen36-35b-a3b-q4k-xl", "raw", "The capital of France is");
  expect(phases.map((phase) => phase.phase)).toEqual([
    "zinc",
    "zinc",
    "zinc",
    "zinc",
    "baseline",
    "baseline",
    "baseline",
    "baseline",
  ]);
  expect(phases.slice(0, 4).map((phase) => phase.scenarioDef.id)).toEqual(["core", "context-medium", "context-long", "decode-extended"]);
  expect(phases.slice(4).map((phase) => phase.scenarioDef.id)).toEqual(["core", "context-medium", "context-long", "decode-extended"]);
});

test("benchmark suite can split ZINC and baseline phases for clean reboot runs", () => {
  const zincOnly = buildMeasurementPhases("qwen36-35b-a3b-q4k-xl", "raw", "The capital of France is", "zinc");
  expect(zincOnly.map((phase) => phase.phase)).toEqual(["zinc", "zinc", "zinc", "zinc"]);

  const baselineOnly = buildMeasurementPhases("qwen36-35b-a3b-q4k-xl", "raw", "The capital of France is", "baseline");
  expect(baselineOnly.map((phase) => phase.phase)).toEqual(["baseline", "baseline", "baseline", "baseline"]);
});

test("parseDotEnv handles export lines and quotes", () => {
  const env = parseDotEnv(`
    export ZINC_HOST=bench.local
    ZINC_USER="root"
    ZINC_PORT='2222'
  `);
  expect(env.ZINC_HOST).toBe("bench.local");
  expect(env.ZINC_USER).toBe("root");
  expect(env.ZINC_PORT).toBe("2222");
});

test("parseZincCliOutput extracts prompt, prefill, decode, and output preview", () => {
  const parsed = parseZincCliOutput(`
info(zinc): Prompt tokens (25): { 1, 2, 3 }
info(forward): Prefill: 25 tokens in 40716.9 ms (0.6 tok/s)
info(forward): Generated 8 tokens in 1269.5 ms — 0.79 tok/s (1269.5 ms/tok)
info(zinc): Output (1 tokens): Paris
`);

  expect(parsed.promptTokens).toBe(25);
  expect(parsed.prefillMs).toBe(40716.9);
  expect(parsed.prefillTps).toBe(0.6);
  expect(parsed.decodeTps).toBe(0.79);
  expect(parsed.msPerToken).toBe(1269.5);
  expect(parsed.outputPreview).toBe("Paris");
});

test("parseZincCliOutput accepts Vulkan-style prefill complete lines", () => {
  const parsed = parseZincCliOutput(`
info(zinc): Prompt tokens (10): { 1, 2, 3 }
info(forward): Prefill complete: 10 tokens in 1.3 ms (7459.27 tok/s)
info(forward): Generated 8 tokens in 61.4 ms — 130.32 tok/s (7.7 ms/tok)
info(zinc): Output text: Paris.
`);

  expect(parsed.promptTokens).toBe(10);
  expect(parsed.prefillMs).toBe(1.3);
  expect(parsed.prefillTps).toBeCloseTo(7459.27, 2);
  expect(parsed.decodeTps).toBe(130.32);
  expect(parsed.outputPreview).toContain("Paris.");
});

test("parseZincCliOutput tolerates missing prefill lines and parses output text", () => {
  const parsed = parseZincCliOutput(`
info(zinc): Prompt tokens (5): { 1, 2, 3 }
info(forward): Generated 8 tokens in 61.4 ms — 130.32 tok/s (7.7 ms/tok)
info(zinc): Output text:  Paris.
A. True
info(zinc): Output tokens (8): first20={ 1, 2, 3 }
`);

  expect(parsed.promptTokens).toBe(5);
  expect(parsed.prefillTokens).toBe(5);
  expect(parsed.prefillMs).toBeNull();
  expect(parsed.prefillTps).toBeNull();
  expect(parsed.decodeTps).toBe(130.32);
  expect(parsed.outputPreview).toContain("Paris.");
});

test("parseLlamaCliOutput extracts prompt and decode timings", () => {
  const parsed = parseLlamaCliOutput(`
llama_print_timings: prompt eval time =   152.58 ms /    16 tokens (    9.54 ms per token,   104.87 tokens per second)
llama_print_timings:        eval time =   474.72 ms /    15 runs   (   31.65 ms per token,    31.60 tokens per second)
`);

  expect(parsed.promptTokens).toBe(16);
  expect(parsed.prefillTps).toBe(104.87);
  expect(parsed.generatedTokens).toBe(15);
  expect(parsed.decodeTps).toBe(31.6);
  expect(parsed.msPerToken).toBeCloseTo(31.648, 3);
});

test("parseLlamaCppVersionOutput extracts version and commit from llama.cpp --version output", () => {
  const parsed = parseLlamaCppVersionOutput(`
load_backend: loaded BLAS backend from /opt/homebrew/Cellar/ggml/0.9.11/libexec/libggml-blas.so
version: 8610 (2b86e5cae)
built with AppleClang 17.0.0.17000604 for Darwin arm64
`);

  expect(parsed?.version).toBe("8610");
  expect(parsed?.commit).toBe("2b86e5cae");
});

test("parseOpenAiCompletionOutput extracts throughput from server JSON", () => {
  const parsed = parseOpenAiCompletionOutput(JSON.stringify({
    usage: {
      prompt_tokens: 24,
      completion_tokens: 128,
    },
    timings: {
      prompt_per_second: 220.5,
      predicted_per_second: 107.2,
    },
    choices: [{ text: " Paris" }],
  }));

  expect(parsed.promptTokens).toBe(24);
  expect(parsed.prefillMs).toBeCloseTo((24 / 220.5) * 1000, 6);
  expect(parsed.prefillTps).toBe(220.5);
  expect(parsed.decodeMs).toBeCloseTo((128 / 107.2) * 1000, 6);
  expect(parsed.decodeTps).toBe(107.2);
  expect(parsed.msPerToken).toBeCloseTo(1000 / 107.2, 6);
  expect(parsed.outputPreview).toBe("Paris");
});

test("summarizeValues includes median, p95, and stddev", () => {
  const summary = summarizeValues([10, 20, 30, 40]);
  expect(summary?.avg).toBe(25);
  expect(summary?.median).toBe(25);
  expect(summary?.p95).toBe(38.5);
  expect(summary?.stddev).toBeGreaterThan(11);
});

test("buildComparison adds prompt and latency deltas", () => {
  const comparison = buildComparison(
    {
      name: "ZINC",
      prompt_tokens: 10,
      generated_tokens: 20,
      prefill_tps: { median: 50, avg: 50 },
      decode_tps: { median: 40, avg: 40 },
      total_latency_ms: { median: 2500, avg: 2500 },
      end_to_end_tps: { median: 30, avg: 30 },
    },
    {
      name: "llama.cpp",
      prompt_tokens: 10,
      generated_tokens: 20,
      prefill_tps: { median: 100, avg: 100 },
      decode_tps: { median: 80, avg: 80 },
      total_latency_ms: { median: 2000, avg: 2000 },
      end_to_end_tps: { median: 60, avg: 60 },
    },
  );

  expect(comparison?.pct_of_baseline).toBe(50);
  expect(comparison?.prompt_pct_of_baseline).toBe(50);
  expect(comparison?.latency_pct_of_baseline).toBe(125);
  expect(comparison?.latency_delta_ms).toBe(500);
  expect(comparison?.end_to_end_pct_of_baseline).toBe(50);
  expect(comparison?.end_to_end_delta_tps).toBe(-30);
  expect(comparison?.overall_pct_of_baseline).toBe(50);
  expect(comparison?.zinc_overall_tps).toBeCloseTo(42.857, 3);
  expect(comparison?.baseline_overall_tps).toBeCloseTo(85.714, 3);
  expect(comparison?.overall_delta_tps).toBeCloseTo(-42.857, 3);
});

test("mergeArtifacts replaces matching targets and preserves others", () => {
  const merged = mergeArtifacts(
    {
      schema_version: 1,
      generated_at: "old",
      targets: [
        {
          id: "rdna",
          label: "RDNA",
          provenance: {
            zinc: { version: "old-rdna", commit: "old-rdna-commit" },
            llama_cpp: { binary: "llama-server", version: "10", commit: "abc" },
          },
        },
        {
          id: "metal",
          label: "Metal old",
          provenance: {
            zinc: { version: "old-metal", commit: "old-metal-commit" },
            llama_cpp: { binary: "llama-cli", version: "20", commit: "def" },
          },
        },
      ],
    },
    [{
      id: "metal",
      label: "Metal new",
      provenance: {
        zinc: { version: "new-metal", commit: "new-metal-commit" },
        llama_cpp: { binary: "llama-server", version: "30", commit: "ghi" },
      },
    }],
  );

  expect(merged.targets[0]?.id).toBe("rdna");
  expect(merged.targets[0]?.label).toBe("RDNA");
  expect(merged.targets[0]?.provenance).toEqual({
    zinc: { version: "old-rdna", commit: "old-rdna-commit" },
    llama_cpp: { binary: "llama-server", version: "10", commit: "abc" },
  });
  expect(merged.targets[0]?.summary.fastest_model_id).toBeNull();
  expect(merged.targets[1].id).toBe("metal");
  expect(merged.targets[1].label).toBe("Metal new");
  expect(merged.targets[1].provenance).toEqual({
    zinc: { version: "new-metal", commit: "new-metal-commit" },
    llama_cpp: { binary: "llama-server", version: "30", commit: "ghi" },
  });
  expect(merged.targets[1].models).toEqual([]);
  expect(merged.targets[1].summary.fastest_model_id).toBeNull();
});

test("compareModelsByName normalizes published model label variants", () => {
  const models = [
    { id: "qwen36-35b-a3b-q4k-xl", label: "Qwen36 35B A3B Q4K XL" },
    { id: "qwen36-27b-q4k-m", label: "Qwen 3.6 27B Dense Q4_K_M" },
    { id: "qwen35-9b-q4k-m", label: "Qwen3.5 9B Q4K M" },
    { id: "gemma4-31b-q4k-m", label: "Gemma 4 31B Q4_K_M" },
    { id: "gemma4-26b-a4b-q4k-m", label: "Gemma 4 26B-A4B MoE Q4_K_M" },
  ];

  expect(models.sort(compareModelsByName).map((model) => model.id)).toEqual([
    "gemma4-26b-a4b-q4k-m",
    "gemma4-31b-q4k-m",
    "qwen35-9b-q4k-m",
    "qwen36-27b-q4k-m",
    "qwen36-35b-a3b-q4k-xl",
  ]);
});

test("mergeArtifacts replaces an existing target to avoid stale model rows", () => {
  const merged = mergeArtifacts(
    {
      schema_version: 1,
      generated_at: "old",
      targets: [
        {
          id: "metal",
          label: "Metal",
          models: [
            { id: "a", label: "A", zinc: { decode_tps: { median: 10, avg: 10 } } },
            { id: "b", label: "B", zinc: { decode_tps: { median: 20, avg: 20 } } },
          ],
          summary: {},
        },
      ],
    },
    [
      {
        id: "metal",
        label: "Metal",
        provenance: {
          zinc: { version: "zinc", commit: "zinc-commit" },
          llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
        },
        models: [
          { id: "b", label: "B", zinc: { decode_tps: { median: 30, avg: 30 } } },
          { id: "c", label: "C", zinc: { decode_tps: { median: 40, avg: 40 } } },
        ],
        summary: {},
      },
    ],
  );

  const metal = merged.targets.find((target) => target.id === "metal");
  expect(metal?.models.map((model) => model.id)).toEqual(["b", "c"]);
  expect(metal?.summary.fastest_model_id).toBe("c");
  expect(metal?.provenance).toEqual({
    zinc: { version: "zinc", commit: "zinc-commit" },
    llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
  });
});

test("mergeArtifacts can preserve missing phase data for split benchmark runs", () => {
  const merged = mergeArtifacts(
    {
      schema_version: 1,
      generated_at: "old",
      targets: [
        {
          id: "rdna",
          label: "RDNA",
          models: [
            {
              id: "qwen",
              label: "Qwen",
              zinc: { name: "ZINC", decode_tps: { median: 120, avg: 120 } },
              baseline: null,
              scenarios: [
                {
                  id: "core",
                  label: "Core Prompt",
                  zinc: { name: "ZINC", decode_tps: { median: 120, avg: 120 } },
                },
              ],
            },
          ],
        },
      ],
    },
    [
      {
        id: "rdna",
        label: "RDNA",
        models: [
          {
            id: "qwen",
            label: "Qwen",
            baseline: { name: "llama.cpp", decode_tps: { median: 100, avg: 100 } },
            scenarios: [
              {
                id: "core",
                label: "Core Prompt",
                baseline: { name: "llama.cpp", decode_tps: { median: 100, avg: 100 } },
              },
            ],
          },
        ],
      },
    ],
    { preserveMissingPhases: true },
  );

  const model = merged.targets[0]?.models[0];
  expect(model?.zinc?.decode_tps.median).toBe(120);
  expect(model?.baseline?.decode_tps.median).toBe(100);
  expect(model?.comparison?.pct_of_baseline).toBe(120);
  expect(model?.scenarios[0]?.comparison?.pct_of_baseline).toBe(120);
});

test("buildArtifact writes only the incoming targets", () => {
  const artifact = buildArtifact([
    {
      id: "metal",
      label: "Metal",
      provenance: {
        zinc: { version: "zinc", commit: "zinc-commit" },
        llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
      },
      models: [{ id: "m", label: "Model M", zinc: { decode_tps: { median: 12, avg: 12 } } }],
    },
  ]);

  expect(artifact.targets.map((target) => target.id)).toEqual(["metal"]);
  expect(artifact.targets[0]?.summary.fastest_model_id).toBe("m");
  expect(artifact.targets[0]?.provenance).toEqual({
    zinc: { version: "zinc", commit: "zinc-commit" },
    llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
  });
});
