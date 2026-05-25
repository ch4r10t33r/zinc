import { describe, expect, test } from "bun:test";
import {
  bestKeptCorrectTokPerSec,
  buildPrompt,
  buildQwen36PrefillPlateauAnalysis,
  buildQwen36PrefillPostBreakthroughAnalysis,
  buildReflectionSummary,
  buildSelfReview,
  currentAcceptedTokPerSec,
  decideKeep,
  detectPhase,
  evaluateOutputText,
  keepBaselinesForCycle,
  mergeUniqueEntries,
  parseTokPerSec,
  recentAcceptedProgress,
  shouldRejectQwen36PlateauNeutralKeep,
  snapshotFromResult,
} from "./implement_metal";
import type { BuildRunResult, ControllerState, CycleResult, RunState } from "./implement_metal";

function makeResult(overrides: Partial<BuildRunResult> = {}): BuildRunResult {
  return {
    buildExitCode: 0,
    buildOutput: "",
    testExitCode: 0,
    testOutput: "",
    runExitCode: 0,
    runOutput: "",
    phase: "implement",
    tokPerSec: null,
    tokPerSecSamples: [],
    tokensGenerated: 5,
    outputText: "",
    containsReference: false,
    strongAnswer: false,
    outputQualityScore: 0,
    offTopic: false,
    evaluationNotes: [],
    error: null,
    ...overrides,
  };
}

function makeCycle(overrides: Partial<CycleResult> = {}): CycleResult {
  return {
    cycle: 1,
    timestamp: new Date().toISOString(),
    phase: "optimize",
    description: "Test change",
    kept: true,
    tokPerSec: 36,
    tokensGenerated: 64,
    containsReference: true,
    buildExitCode: 0,
    testExitCode: 0,
    runExitCode: 0,
    outputText: "ĠParis.ĠTheĠcapitalĠof",
    stepKind: "optimization",
    selfAnalysis: "",
    nextIdeas: [],
    ...overrides,
  };
}

function makeState(overrides: Partial<RunState> = {}): RunState {
  return {
    runId: "test-run",
    cycles: [],
    failedApproaches: [],
    ideas: [],
    phase: "optimize",
    currentBest: null,
    stalledCycles: 0,
    bestTokPerSec: 36,
    lastProfileOutput: null,
    lastProfileCycle: null,
    reviewSummaries: [],
    ...overrides,
  };
}

// ── evaluateOutputText ──────────────────────────────────────────────

describe("evaluateOutputText", () => {
  test("accepts BPE-marked Paris prefix as a strong answer", () => {
    const result = evaluateOutputText("ĠParis.ĠTheĠcapitalĠof");
    expect(result.normalizedText).toBe("Paris. The capital of");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(true);
    expect(result.offTopic).toBe(false);
  });

  test("penalizes contradictory continuations", () => {
    const result = evaluateOutputText("ĠParis.ĠTheĠcapitalĠofĠGermanyĠisĠBerlin");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(false);
    expect(result.offTopic).toBe(true);
  });

  test("handles empty string", () => {
    const result = evaluateOutputText("");
    expect(result.normalizedText).toBe("");
    expect(result.containsReference).toBe(false);
    expect(result.strongAnswer).toBe(false);
  });

  test("detects Paris without BPE markers", () => {
    const result = evaluateOutputText("Paris is the capital");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(true);
  });
});

// ── parseTokPerSec ──────────────────────────────────────────────────

describe("parseTokPerSec", () => {
  test("prefers generated throughput in decode mode", () => {
    const output = [
      "info(forward_metal): Prefill: 122 tokens in 610.0 ms (200.0 tok/s)",
      "info(forward_metal): Generated 64 tokens in 1280.0 ms - 50.0 tok/s",
    ].join("\n");
    expect(parseTokPerSec(output, "decode")).toBe(50);
  });

  test("parses prefill throughput in prefill mode", () => {
    const output = [
      "info(forward_metal): Prefill: 122 tokens in 610.0 ms (200.0 tok/s)",
      "info(forward_metal): Generated 64 tokens in 1280.0 ms - 50.0 tok/s",
    ].join("\n");
    expect(parseTokPerSec(output, "prefill")).toBe(200);
  });

  test("computes prefill throughput when rate is not printed", () => {
    const output = "info(forward): Prefill complete: 50 tokens in 250 ms";
    expect(parseTokPerSec(output, "prefill")).toBe(200);
  });
});

// ── detectPhase ─────────────────────────────────────────────────────

describe("detectPhase", () => {
  test("stays in optimize when answer is correct but tok/s is missing", () => {
    expect(
      detectPhase(
        makeResult({
          outputText: "ĠParis.",
          containsReference: true,
          strongAnswer: true,
          outputQualityScore: 4,
        }),
      ),
    ).toBe("optimize");
  });

  test("stays in implement for partial Paris mentions", () => {
    expect(
      detectPhase(
        makeResult({
          outputText: "Somewhere near Paris",
          containsReference: true,
          strongAnswer: false,
          outputQualityScore: 1,
        }),
      ),
    ).toBe("implement");
  });

  test("returns fix on build failure", () => {
    expect(detectPhase(makeResult({ buildExitCode: 1 }))).toBe("fix");
  });

  test("returns fix on test failure", () => {
    expect(detectPhase(makeResult({ testExitCode: 1 }))).toBe("fix");
  });

  test("returns fix on runtime crash", () => {
    expect(detectPhase(makeResult({ runExitCode: 139 }))).toBe("fix");
  });

  test("returns fix on error string", () => {
    expect(detectPhase(makeResult({ error: "segfault" }))).toBe("fix");
  });
});

// ── decideKeep ──────────────────────────────────────────────────────

describe("decideKeep", () => {
  test("keeps the first strong correct output", () => {
    const baseline = snapshotFromResult(
      1,
      makeResult({ tokensGenerated: 5, outputQualityScore: 0 }),
    );
    const verify = makeResult({
      tokPerSec: 28,
      tokPerSecSamples: [28, 28.5, 27.8],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const state: ControllerState = { lastAccepted: null, bestSoFar: null, bestCorrect: null };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(true);
    expect(decision.improvedBestCorrect).toBe(true);
  });

  test("rejects slower correct output that does not beat the best", () => {
    const baselineResult = makeResult({
      tokPerSec: 30,
      tokPerSecSamples: [29.5, 30, 30.5],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const baseline = snapshotFromResult(2, baselineResult);
    const verify = makeResult({
      tokPerSec: 29.4,
      tokPerSecSamples: [29.2, 29.4, 29.5],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: baseline };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(false);
  });

  test("keeps significant correct-throughput gains", () => {
    const baselineResult = makeResult({
      tokPerSec: 30,
      tokPerSecSamples: [30, 30.2, 29.8],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const baseline = snapshotFromResult(3, baselineResult);
    const verify = makeResult({
      tokPerSec: 31.5,
      tokPerSecSamples: [31.4, 31.5, 31.6],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: baseline };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(true);
    expect(decision.improvedBestCorrect).toBe(true);
  });

  test("rejects loss of correctness after a correct baseline exists", () => {
    const baselineResult = makeResult({
      tokPerSec: 30,
      tokPerSecSamples: [30, 30.1, 29.9],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const baseline = snapshotFromResult(4, baselineResult);
    const verify = makeResult({
      tokPerSec: 40,
      tokPerSecSamples: [40, 40, 40],
      outputText: "ĠBerlin",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: baseline };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(false);
  });

  test("keeps pre-correctness progress when tokens increase materially", () => {
    const baselineResult = makeResult({
      tokensGenerated: 5,
      outputText: "ĠThe",
      outputQualityScore: 0,
    });
    const baseline = snapshotFromResult(5, baselineResult);
    const verify = makeResult({
      tokensGenerated: 8,
      outputText: "ĠTheĠcapital",
      outputQualityScore: 1,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: null };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(true);
    expect(decision.improvedBestCorrect).toBe(false);
  });
});

// ── mergeUniqueEntries ──────────────────────────────────────────────

describe("mergeUniqueEntries", () => {
  test("dedupes near-duplicate ideas and caps memory", () => {
    const merged = mergeUniqueEntries(
      ["add real Metal per-dispatch timing behind --profile"],
      [
        "add real per-dispatch Metal timing behind `--profile`",
        "benchmark flash_attn vs MoE down-projection on-device",
      ],
      5,
    );
    expect(merged.length).toBe(2);
    expect(merged[1]).toContain("flash_attn");
  });

  test("respects maxEntries cap", () => {
    const merged = mergeUniqueEntries(
      ["a", "b", "c"],
      ["d", "e"],
      3,
    );
    expect(merged.length).toBe(3);
  });

  test("skips empty strings", () => {
    const merged = mergeUniqueEntries(["", "  ", "real idea"], [], 10);
    expect(merged.length).toBe(1);
    expect(merged[0]).toBe("real idea");
  });
});

// ── keep baselines ──────────────────────────────────────────────────

describe("keep baseline helpers", () => {
  test("recovers stale bestTokPerSec from kept correct cycle history", () => {
    const state = makeState({
      bestTokPerSec: 43.4,
      currentBest: { tokPerSec: 43.2, containsReference: true },
      cycles: [
        makeCycle({ cycle: 100, kept: true, containsReference: true, tokPerSec: 43.8 }),
        makeCycle({ cycle: 104, kept: true, containsReference: true, tokPerSec: 44.0 }),
        makeCycle({ cycle: 106, kept: true, containsReference: true, tokPerSec: 43.2 }),
      ],
    });
    expect(bestKeptCorrectTokPerSec(state)).toBe(44.0);
  });

  test("anchors neutral keeps to this cycle's measured accepted baseline", () => {
    const state = makeState({
      bestTokPerSec: 43.4,
      currentBest: { tokPerSec: 43.2, containsReference: true },
      cycles: [
        makeCycle({ cycle: 104, kept: true, containsReference: true, tokPerSec: 44.0 }),
        makeCycle({ cycle: 107, kept: true, containsReference: true, tokPerSec: 43.2 }),
      ],
    });
    const baselines = keepBaselinesForCycle(
      state,
      makeResult({ containsReference: true, strongAnswer: true, tokPerSec: 44.0 }),
    );
    expect(baselines.bestTokPerSec).toBe(44.0);
    expect(baselines.acceptedTokPerSec).toBe(44.0);
  });

  test("falls back to latest kept correct cycle for current accepted baseline", () => {
    const state = makeState({
      bestTokPerSec: 43.8,
      currentBest: null,
      cycles: [
        makeCycle({ cycle: 124, kept: true, containsReference: true, tokPerSec: 44.7 }),
        makeCycle({ cycle: 125, kept: true, containsReference: true, tokPerSec: 45.1 }),
      ],
    });
    expect(currentAcceptedTokPerSec(state)).toBe(45.1);
  });

  test("detects small recent accepted progress separately from a stall", () => {
    const state = makeState({
      bestTokPerSec: 44.8,
      currentBest: { tokPerSec: 45.1, containsReference: true },
      cycles: [
        makeCycle({ cycle: 115, kept: true, containsReference: true, tokPerSec: 44.8 }),
        ...Array.from({ length: 8 }, (_, idx) =>
          makeCycle({ cycle: 116 + idx, kept: true, containsReference: true, tokPerSec: 44.7 }),
        ),
        makeCycle({ cycle: 124, kept: true, containsReference: true, tokPerSec: 44.7 }),
        makeCycle({ cycle: 125, kept: true, containsReference: true, tokPerSec: 45.1 }),
      ],
    });
    const progress = recentAcceptedProgress(state);
    expect(progress.hasProgress).toBe(true);
    expect(progress.start).toBe(44.8);
    expect(progress.end).toBe(45.1);
  });
});

// ── buildReflectionSummary ──────────────────────────────────────────

describe("buildReflectionSummary", () => {
  test("summarizes the last 20 cycles and highlights repeated failure basins", () => {
    const cycles = Array.from({ length: 20 }, (_, idx) => ({
      cycle: idx + 1,
      phase: "implement",
      shortTokPerSec: null,
      shortTokPerSecSamples: [],
      shortTokensGenerated: 32,
      shortContainsReference: true,
      shortStrongAnswer: false,
      shortOutputQualityScore: 2,
      shortOutputText: "ĠParis.ĠTheĠcapitalĠofĠGermanyĠisĠBerlin",
      longTokPerSec: 22,
      longTokPerSecSamples: [22],
      longTokensGenerated: 32,
      longContainsReference: true,
      longStrongAnswer: false,
      longOutputQualityScore: 2,
      longOutputText: "ĠParis.ĠTheĠcapitalĠofĠGermanyĠisĠBerlin",
      timestamp: new Date().toISOString(),
      description: "Attempted a speculative attention fix",
      kept: false,
      buildExitCode: 0,
      testExitCode: 0,
      runExitCode: 0,
      offTopic: true,
      evaluationNotes: ["contains contradictory capital/country terms"],
      decisionReason: "lost short-benchmark correctness relative to accepted baseline",
      selfAnalysis: "",
      nextIdeas: [],
    }));
    const summary = buildReflectionSummary({
      runId: "r",
      cycles,
      failedApproaches: [],
      ideas: [],
      phase: "implement",
      lastAccepted: null,
      bestSoFar: null,
      bestCorrect: null,
      currentBest: null,
      stalledCycles: 20,
      reviewSummaries: [],
      lastProfileExcerpt: null,
      lastProfileCycle: null,
      acceptedCommit: null,
    } as any);
    expect(summary).toContain("Last 20 cycles");
    expect(summary).toContain("Paris->Germany list drift");
    expect(summary).toContain("Prioritize parity tests");
  });
});

// ── buildSelfReview ─────────────────────────────────────────────────

describe("buildSelfReview", () => {
  test("returns empty string with no cycles", () => {
    const state = makeState({ cycles: [] });
    expect(buildSelfReview(state)).toBe("");
  });

  test("categorizes shader changes and reports success rate", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: i < 6 ? "Tune shader threadgroup size" : "Rearrange buffer alloc",
      kept: i < 4, // 4 kept, 6 reverted
      tokPerSec: 36 + (i < 4 ? i * 0.5 : 0),
    }));
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Self-Review");
    expect(review).toContain("4/10 changes");
    expect(review).toContain("shader");
    expect(review).toContain("memory");
  });

  test("reports positive tok/s progress", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: "Optimize dispatch batching",
      kept: true,
      tokPerSec: 36 + i * 0.8,
    }));
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Progress is positive");
    expect(review).toContain("dispatch");
  });

  test("warns on low progress", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: "Try random tweak",
      kept: i % 3 === 0,
      tokPerSec: 36 + (i % 3 === 0 ? 0.02 : -0.2),
    }));
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Low progress");
    expect(review).toContain("strategic pivot");
  });

  test("labels sub-1 tok/s accepted movement as small progress", () => {
    const state = makeState({
      cycles: [
        makeCycle({ cycle: 1, kept: true, containsReference: true, tokPerSec: 44.8 }),
        ...Array.from({ length: 8 }, (_, idx) =>
          makeCycle({ cycle: 2 + idx, kept: true, containsReference: true, tokPerSec: 44.7 }),
        ),
        makeCycle({ cycle: 10, kept: true, containsReference: true, tokPerSec: 45.1 }),
      ],
    });
    const review = buildSelfReview(state);
    expect(review).toContain("Small accepted progress");
    expect(review).not.toContain("Low progress");
  });

  test("does not count reverted-cycle movement as accepted progress", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: "Retune Q8 threadgroup",
      kept: false,
      tokPerSec: 34 + i * 0.4,
    }));
    const state = makeState({
      cycles,
      bestTokPerSec: 37.8,
      currentBest: { tokPerSec: 37.8, containsReference: true },
    });
    const review = buildSelfReview(state);
    expect(review).toContain("No accepted progress");
    expect(review).toContain("Do NOT treat faster reverted candidates as progress");
    expect(review).not.toContain("Progress is positive");
  });

  test("shows top performing changes", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Fuse RMS+DMMV kernels", kept: true, tokPerSec: 38.5 }),
      makeCycle({ cycle: 2, description: "Batch MoE expert dispatch", kept: true, tokPerSec: 40.2 }),
      makeCycle({ cycle: 3, description: "Reduce encoder switches", kept: false, tokPerSec: 35 }),
      makeCycle({ cycle: 4, description: "Use bfloat16 intermediates", kept: true, tokPerSec: 41.0 }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Top performing");
    expect(review).toContain("bfloat16");
    expect(review).toContain("41.0");
  });

  test("categorizes MoE and attention changes", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Batch MoE expert routing with topk", kept: true }),
      makeCycle({ cycle: 2, description: "Optimize flash attention KV cache", kept: false }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("moe");
    expect(review).toContain("attention");
  });

  test("categorizes fusion changes", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Fused SwiGLU kernel to avoid write-back", kept: true }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("fusion");
  });

  test("falls back to 'other' category for unrecognized descriptions", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Reorder loop iterations", kept: true }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("other");
  });
});

// ── buildPrompt ─────────────────────────────────────────────────────

describe("buildPrompt", () => {
  test("optimize phase includes bandwidth analysis and optimization targets", () => {
    const state = makeState({
      phase: "optimize",
      currentBest: { tokPerSec: 36, containsReference: true },
    });
    const result = makeResult({
      tokPerSec: 36,
      tokPerSecSamples: [35.8, 36, 36.2],
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("OPTIMIZE");
    expect(prompt).toContain("Bandwidth Analysis");
    expect(prompt).toContain("Reduce command buffer");
    expect(prompt).toContain("Threadgroup size");
    expect(prompt).toContain("MoE expert dispatch");
    expect(prompt).toContain("Fused kernels");
    expect(prompt).not.toContain("What Needs Implementation");
  });

  test("fix phase includes project structure but not bandwidth analysis", () => {
    const state = makeState({ phase: "fix" });
    const result = makeResult({ buildExitCode: 1, buildOutput: "error: undefined symbol", phase: "fix" });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("FIX");
    expect(prompt).toContain("BUILD FAILURE");
    expect(prompt).toContain("Project Structure");
    expect(prompt).not.toContain("Bandwidth Analysis");
  });

  test("includes stall warning after threshold", () => {
    const state = makeState({ stalledCycles: 5 });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("STALL");
    expect(prompt).toContain("llama.cpp");
    expect(prompt).toContain("vllm");
    expect(prompt).toContain("ggml-metal");
  });

  test("suppresses hard stall warning when accepted baseline moved recently", () => {
    const state = makeState({
      stalledCycles: 12,
      currentBest: { tokPerSec: 45.1, containsReference: true },
      cycles: [
        makeCycle({ cycle: 115, kept: true, containsReference: true, tokPerSec: 44.8 }),
        ...Array.from({ length: 9 }, (_, idx) =>
          makeCycle({ cycle: 116 + idx, kept: true, containsReference: true, tokPerSec: 44.7 }),
        ),
        makeCycle({ cycle: 125, kept: true, containsReference: true, tokPerSec: 45.1 }),
      ],
    });
    const result = makeResult({
      tokPerSec: 45.0,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "Paris",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Limited Progress");
    expect(prompt).toContain("44.80 → 45.10");
    expect(prompt).not.toContain("STUDY THE REFERENCES");
  });

  test("Qwen effort prompt tells the next cycle to use route-pack density evidence", () => {
    const state = makeState({
      effortId: 16,
      effortFile: "MULTI_HOUR_EFFORT_16_METAL_QWEN36_35B_PREFILL_M4.md",
      effortPlan: "# Effort 16\nQwen 3.6 35B-A3B prefill",
      currentBest: { tokPerSec: 45.1, containsReference: true },
      lastProfileOutput: "Metal profile: Qwen route-pack candidate blocks prompt_tokens=134 route_slots=1072 active_block_upper=358 dense_dispatch_blocks=8704 upper/dense=4.1%",
      cycles: [
        makeCycle({
          cycle: 125,
          kept: true,
          containsReference: true,
          tokPerSec: 45.1,
          description: "Added Qwen route-pack active-block-density profile counters and candidate blocker logging.",
        }),
      ],
    });
    const result = makeResult({
      tokPerSec: 45.0,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "Paris",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("route-pack candidate blocks");
    expect(prompt).toContain("do not add another passive counter");
    expect(prompt).toContain("consume that evidence");
  });

  test("Qwen effort prompt puts repeated route-pack reverts on cooldown", () => {
    const state = makeState({
      effortId: 16,
      effortFile: "MULTI_HOUR_EFFORT_16_METAL_QWEN36_35B_PREFILL_M4.md",
      effortPlan: "# Effort 16\nQwen 3.6 35B-A3B prefill",
      currentBest: { tokPerSec: 45.8, containsReference: true },
      cycles: [
        makeCycle({
          cycle: 127,
          kept: false,
          containsReference: false,
          tokPerSec: 45.6,
          description: "Default layer-0 Qwen route-packed prefill materialized the F32 shared-gate scalar.",
          outputText: "!!!!!!!!!!!!!!!!",
        }),
        makeCycle({
          cycle: 128,
          kept: false,
          containsReference: true,
          tokPerSec: 44.7,
          description: "Added active-block materialized F32 shared-gate route-pack validation.",
        }),
      ],
    });
    const result = makeResult({
      tokPerSec: 44.8,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "Paris",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("ROUTE-PACK COOLDOWN");
    expect(prompt).toContain("do not edit route-pack/shared-gate validators");
    expect(prompt).toContain("must not run local Metal model commands");
  });

  test("Qwen effort prompt adds plateau analysis after many neutral keeps", () => {
    const cycles = Array.from({ length: 32 }, (_, idx) => makeCycle({
      cycle: 194 + idx,
      kept: true,
      containsReference: true,
      tokPerSec: idx === 5 ? 51.6 : 51.1,
      description: idx % 2 === 0
        ? "Retuned fixed-K TG128 repacked Q8 SSM projection kernel"
        : "Adjusted Qwen SSM delta threadgroup arithmetic",
    }));
    const state = makeState({
      effortId: 16,
      effortFile: "MULTI_HOUR_EFFORT_16_METAL_QWEN36_35B_PREFILL_M4.md",
      effortPlan: "# Effort 16\nQwen 3.6 35B-A3B prefill",
      bestTokPerSec: 51.6,
      currentBest: { tokPerSec: 51.1, containsReference: true },
      stalledCycles: 32,
      cycles,
    });
    const result = makeResult({
      tokPerSec: 51.1,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "Paris",
    });

    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Qwen3.6 35B Prefill Plateau Analysis");
    expect(prompt).toContain("PLATEAU MODE");
    expect(prompt).toContain("Neutral keeps are dominating");
    expect(prompt).toContain("@@@STEP_KIND:");
  });

  test("plateau analysis can be generated directly from cycle history", () => {
    const state = makeState({
      effortId: 16,
      effortFile: "MULTI_HOUR_EFFORT_16_METAL_QWEN36_35B_PREFILL_M4.md",
      effortPlan: "# Effort 16\nQwen 3.6 35B-A3B prefill",
      bestTokPerSec: 51.6,
      currentBest: { tokPerSec: 51.1, containsReference: true },
      stalledCycles: 32,
      cycles: Array.from({ length: 32 }, (_, idx) => makeCycle({
        cycle: idx + 1,
        kept: true,
        containsReference: true,
        tokPerSec: 51.1,
        description: "Fixed-K TG128 repacked Q8 SSM cleanup",
      })),
    });
    const analysis = buildQwen36PrefillPlateauAnalysis(state).join("\n");
    expect(analysis).toContain("PLATEAU MODE");
    expect(analysis).toContain("Cooldown");
    expect(analysis).toContain("Required pivot");
  });

  test("Qwen post-breakthrough focus pivots from router to profile-dominant SSM", () => {
    const state = makeState({
      effortId: 16,
      effortFile: "MULTI_HOUR_EFFORT_16_METAL_QWEN36_35B_PREFILL_M4.md",
      effortPlan: "# Effort 16\nQwen 3.6 35B-A3B prefill",
      bestTokPerSec: 69.9,
      currentBest: { tokPerSec: 69.9, containsReference: true },
      lastProfileCycle: 231,
      lastProfileOutput: [
        "info(forward):   barriers/step: embed 1.0 attn 75.1 ssm 116.1 router 77.1 gpu-moe 118.7 fallback-moe 0.0 dense 0.0 final 0.0",
        "info(forward):   path bytes: ssm 132.98 GiB attn 33.38 GiB dense 0.00 GiB moe-expert 75.84 GiB shared 16.52 GiB lm-head 1.51 GiB router 10.37 GiB",
        "info(forward):   prefill buckets: ssm proj 98.69 GiB recurrent conv/delta/gated 3887/3887/3887 out 32.27 GiB | router 10.21 GiB topk 5227 cpu 0.00 ms",
      ].join("\n"),
      cycles: [
        makeCycle({
          cycle: 231,
          kept: true,
          containsReference: true,
          tokPerSec: 69.9,
          description: "Routed Qwen3.6 prompt F32 routers through fused router_f32_topk_batched with input-offset support.",
        }),
      ],
    });

    const analysis = buildQwen36PrefillPostBreakthroughAnalysis(state).join("\n");
    expect(analysis).toContain("Post-60 Prefill Jump Focus");
    expect(analysis).toContain("cycle 231");
    expect(analysis).toContain("ssm=132.98 GiB");
    expect(analysis).toContain("After the fused F32 router/top-k win, SSM is larger than router");
    expect(analysis).toContain("gpu-moe=118.70/step");
  });

  test("Qwen plateau mode rejects neutral optimization churn but allows analysis", () => {
    const state = makeState({
      effortId: 16,
      effortFile: "MULTI_HOUR_EFFORT_16_METAL_QWEN36_35B_PREFILL_M4.md",
      effortPlan: "# Effort 16\nQwen 3.6 35B-A3B prefill",
      currentBest: { tokPerSec: 51.1, containsReference: true },
      stalledCycles: 32,
    });

    expect(shouldRejectQwen36PlateauNeutralKeep({
      state,
      stepKind: "optimization",
      description: "Retune fixed-K Q8 math",
      selfAnalysis: "Expected tiny speedup.",
      verifyTokPerSec: 51.1,
      acceptedTokPerSec: 51.1,
      currentProgressBand: 0.15,
    })).toBe(true);

    expect(shouldRejectQwen36PlateauNeutralKeep({
      state,
      stepKind: "analysis",
      description: "Add profile counter for route-pack validator",
      selfAnalysis: "Unlocks layer-specific correctness diff.",
      verifyTokPerSec: 51.1,
      acceptedTokPerSec: 51.1,
      currentProgressBand: 0.15,
    })).toBe(false);
  });

  test("Gemma effort prompt uses Gemma model facts instead of Qwen facts", () => {
    const state = makeState({ effortId: 11 });
    const result = makeResult({
      tokPerSec: 37.8,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "The capital of France is Paris.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Model (Gemma 4 26B-A4B MoE Q4_K_M)");
    expect(prompt).toContain("hidden_dim=2816");
  });

  test("includes pre-stall warning at 3 cycles", () => {
    const state = makeState({ stalledCycles: 3 });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("3/5 cycles without improvement");
    expect(prompt).toContain("reference study");
  });

  test("no stall warning when stalledCycles is low", () => {
    const state = makeState({ stalledCycles: 1 });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).not.toContain("STALL");
    expect(prompt).not.toContain("reference study");
  });

  test("includes profile output when available", () => {
    const state = makeState({
      lastProfileOutput: "Phase: decode_step total=12.5ms\n  rms_norm: 0.8ms\n  dmmv_q4k: 5.2ms",
      lastProfileCycle: 5,
    });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Profile Output (cycle 5)");
    expect(prompt).toContain("dmmv_q4k: 5.2ms");
  });

  test("includes latest review summary", () => {
    const state = makeState({
      reviewSummaries: ["## Self-Review (last 10 cycles)\n\nshader: 3/5 kept"],
    });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Self-Review (last 10 cycles)");
    expect(prompt).toContain("shader: 3/5 kept");
  });

  test("Effort 16 prompt includes Qwen prefill target focus and fast-wrong traps", () => {
    const state = makeState({
      effortId: 16,
      effortFile: "MULTI_HOUR_EFFORT_16_METAL_QWEN36_35B_PREFILL_M4.md",
      effortPlan: "# Effort 16\nQwen 3.6 35B-A3B prefill",
      bestTokPerSec: 43.8,
      stalledCycles: 11,
      cycles: [
        makeCycle({
          cycle: 89,
          kept: true,
          tokPerSec: 43.4,
          description: "Adapted vLLM topk_weight_and_reduce with a token-major Qwen F32 shared-gate MoE combine kernel.",
        }),
        makeCycle({
          cycle: 95,
          kept: false,
          tokPerSec: 43.5,
          containsReference: false,
          description: "Adapted a dual Q8 SSM attn_qkv+attn_gate path.",
          outputText: "!!!!!!!!!!!!!!!!",
        }),
        makeCycle({
          cycle: 96,
          kept: false,
          tokPerSec: 44.5,
          containsReference: false,
          description: "Enabled Qwen layer-0 F32 shared-gate route-packed prefill.",
          outputText: "!!!!!!!!!!!!!!!!",
        }),
        makeCycle({
          cycle: 100,
          kept: true,
          tokPerSec: 43.8,
          description: "Adapted early graph submission by committing a 16-token leading prompt chunk.",
        }),
      ],
    });
    const result = makeResult({
      tokPerSec: 43.8,
      tokPerSecSamples: [43.7, 43.8, 43.9],
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "Paris",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Qwen3.6 35B Prefill Target Focus");
    expect(prompt).toContain("50.0 prefill tok/s");
    expect(prompt).toContain("token-major Qwen F32 shared-gate");
    expect(prompt).toContain("ZINC_QWEN36_LAYER0_ROUTE_PACK_PREFILL=1");
    expect(prompt).toContain("full active-prompt validation");
    expect(prompt).toContain("dual-Q8 SSM");
  });

  test("correctness regression prompt tells agent to restore output", () => {
    const state = makeState({
      currentBest: { tokPerSec: 36, containsReference: true },
    });
    const result = makeResult({
      tokPerSec: 42,
      containsReference: false,
      strongAnswer: false,
      outputText: "ĠBerlin",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("CORRECTNESS REGRESSION");
    expect(prompt).toContain("restore correct output");
  });

  test("target reached status", () => {
    const state = makeState();
    const result = makeResult({
      tokPerSec: 52,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("TARGET REACHED");
    expect(prompt).toContain("52.0");
  });

  test("includes benchmark samples in diagnosis", () => {
    const state = makeState();
    const result = makeResult({
      tokPerSec: 36,
      tokPerSecSamples: [35.5, 36.0, 36.5],
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("35.5");
    expect(prompt).toContain("36.5");
  });

  test("warns when benchmark samples are too noisy for direction", () => {
    const state = makeState();
    const result = makeResult({
      tokPerSec: 30.2,
      tokPerSecSamples: [37.6, 25.4, 30.2],
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "The capital of France is Paris.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Benchmark variance warning");
    expect(prompt).toContain("too wide for reliable direction");
  });
});
