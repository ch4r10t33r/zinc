import { describe, expect, test } from "bun:test";
import {
  aggregateTimedSamples,
  applyBestTreeRestoreCommit,
  backfillNearMiss,
  bestKeptCorrectTokPerSec,
  bestTreeCandidateCycle,
  buildCrossEffortStatus,
  buildFamilyCooldownDirective,
  buildNearMissDirective,
  buildStepKindDiversityNudge,
  buildStructuralPivotDirective,
  classifyAttemptFamilies,
  cooledFamilyRejectionReason,
  countNearMissFamilyReverts,
  detectAutoStopForPlateau,
  noiseAwareImproveBand,
  parseDiagnosticEnv,
  parsePromptFingerprint,
  buildPrompt,
  buildGemmaPrefillPostBreakthroughAnalysis,
  buildQwen36PrefillPlateauAnalysis,
  buildQwen36PrefillPostBreakthroughAnalysis,
  buildReflectionSummary,
  buildSelfReview,
  currentAcceptedTokPerSec,
  decideKeep,
  detectPhase,
  evaluateOutputText,
  gemmaPrefillActualPathIsStructuralRoutePack,
  gemmaPrefillActualPathIsQueuedOffPath,
  inferGemmaRouteTokensForMetalShapes,
  keepBaselinesForCycle,
  mergeUniqueEntries,
  parseRestorePathList,
  parseTokPerSec,
  recentAcceptedProgress,
  recordNearMiss,
  shouldRejectPlateauNeutralKeep,
  shouldConfirmCandidate,
  shouldFinalizeBestTree,
  shouldRunCandidateEvidence,
  shouldRunMetalShapesEvidence,
  shouldRestorePromotedBestDuringPlateau,
  shouldRejectQwen36PlateauNeutralKeep,
  snapshotFromResult,
  workloadMismatchReason,
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
    metricMode: "decode",
    currentBest: null,
    stalledCycles: 0,
    bestTokPerSec: 36,
    lastProfileOutput: null,
    lastProfileCycle: null,
    lastMetalShapesOutput: null,
    lastMetalShapesCycle: null,
    reviewSummaries: [],
    ...overrides,
  };
}

// ── evaluateOutputText ──────────────────────────────────────────────

describe("evaluateOutputText", () => {
  test("accepts BPE-marked Paris prefix as a strong answer", () => {
    const result = evaluateOutputText("ĠParis.ĠTheĠcapitalĠof", "Paris");
    expect(result.normalizedText).toBe("Paris. The capital of");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(true);
    expect(result.offTopic).toBe(false);
  });

  test("penalizes contradictory continuations", () => {
    const result = evaluateOutputText("ĠParis.ĠTheĠcapitalĠofĠGermanyĠisĠBerlin", "Paris");
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
    const result = evaluateOutputText("Paris is the capital", "Paris");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(true);
  });

  test("accepts a configurable public-prompt reference", () => {
    const result = evaluateOutputText(
      "Here is an implementation plan for a stable benchmark preset.",
      "benchmark",
    );
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(true);
    expect(result.offTopic).toBe(false);
    expect(result.evaluationNotes).toContain("contains benchmark");
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

describe("workload fingerprinting", () => {
  test("parses raw/prepared prompt hashes and prompt-token count from ZINC output", () => {
    const parsed = parsePromptFingerprint(
      'info(zinc): Prompt fingerprint: raw=abc123 prepared=def456 mode=chat prompt_tokens=82\ninfo(zinc): Prompt tokens (82): [1,2]',
    );
    expect(parsed).toEqual({
      rawPromptHash: "abc123",
      preparedPromptHash: "def456",
      promptTokens: 82,
    });
  });

  test("detects prepared-prompt and token-count workload drift", () => {
    const base = {
      model: "gemma4-26b-a4b-q4k-m",
      metricMode: "prefill" as const,
      promptMode: "chat",
      maxTokens: 32,
      referenceHash: "ref",
      rawPromptHash: "raw",
      preparedPromptHash: "prepared-a",
      promptTokens: 70,
    };
    const drifted = { ...base, preparedPromptHash: "prepared-b", promptTokens: 82 };
    const reason = workloadMismatchReason(base, drifted);
    expect(reason).toContain("prepared prompt hash changed");
    expect(reason).toContain("prompt tokens 70 -> 82");
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

  test("Gemma post-80 focus surfaces current profile and plateau rules", () => {
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE",
      metricMode: "prefill",
      bestTokPerSec: 88.3,
      currentBest: { tokPerSec: 88.3, containsReference: true },
      stalledCycles: 8,
      lastProfileCycle: 98,
      lastProfileOutput: [
        "info(forward):   path bytes: ssm 0.00 GiB attn 30.29 GiB dense 0.00 GiB moe-expert 22.78 GiB shared 14.50 GiB lm-head 6.57 GiB router 1.09 GiB",
        "info(forward):   prefill buckets: moe gate/up 9.65 GiB down 6.44 GiB | shared gate/up 6.84 GiB down 3.42 GiB | waits 1 commits 211.81 ms",
        "info(forward):   prefill queued prefill: requests 1 prompt_tokens 20 chunks 4 async 3 first_chunks [1,5,7,7,0,0,0,0]",
        "info(forward):   q8 hot #1: shared M=2112 K=2816 bytes=9.66 GiB calls=1642",
      ].join("\n"),
      cycles: [
        makeCycle({
          cycle: 98,
          kept: true,
          containsReference: true,
          tokPerSec: 88.3,
          description: "Collapsed Gemma26 exact-20 queued prefill tail from [1,5,7,6,1] to [1,5,7,7].",
        }),
        ...Array.from({ length: 6 }, (_, idx) => makeCycle({
          cycle: 99 + idx,
          kept: false,
          containsReference: true,
          tokPerSec: 86.0,
          description: "Retune weighted finalizer sigmoid path",
        })),
      ],
    });

    const analysis = buildGemmaPrefillPostBreakthroughAnalysis(state).join("\n");
    expect(analysis).toContain("Gemma26 M4 Prefill Post-80 Focus");
    expect(analysis).toContain("Accepted best is 88.3");
    expect(analysis).toContain("shared M=2112 K=2816");
    expect(analysis).toContain("PLATEAU RULE");
    expect(analysis).toContain("gemma26_prefill_hot");
  });

  test("Gemma post-80 focus blocks off-path route-pack churn when structural batching is disabled", () => {
    const profile = [
      "info(forward):   prefill actual path: queued-token-major default_batched=yes structural_batched=no route_layers=0 queued_chunks=12",
      "info(forward):   prefill queued prefill: requests 1 prompt_tokens 70 chunks 12 async 11 chunk_base 7 final 4 first_chunks [1,5,7,7,7,7,7,7]",
      "info(forward):   prefill queued prefill queue: async_submits 11 final_wait 728.4 ms",
      "info(forward):   q8 hot #1: shared M=2112 K=2816 bytes=34.85 GiB calls=5922",
    ].join("\n");
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE",
      metricMode: "prefill",
      bestTokPerSec: 90.8,
      currentBest: { tokPerSec: 90.6, containsReference: true },
      stalledCycles: 8,
      lastProfileOutput: profile,
      lastMetalShapesCycle: 28,
      lastMetalShapesOk: true,
      lastMetalShapesOutput: "Case moe_down_cols route-cols active-block: 1.64 ms",
      cycles: [
        makeCycle({ cycle: 10, kept: true, containsReference: true, tokPerSec: 90.8 }),
        ...Array.from({ length: 8 }, (_, idx) => makeCycle({
          cycle: 21 + idx,
          kept: idx % 2 === 0,
          containsReference: true,
          tokPerSec: idx % 2 === 0 ? 90.6 : 89.4,
          description: "Retune active-block route-pack gather barriers",
        })),
      ],
    });

    expect(gemmaPrefillActualPathIsQueuedOffPath(profile)).toBe(true);
    const analysis = buildGemmaPrefillPostBreakthroughAnalysis(state).join("\n");
    expect(analysis).toContain("Latest actual prefill path");
    expect(analysis).toContain("full batched route-pack is not executing");
    expect(analysis).toContain("structural_batched=yes");
    expect(analysis).toContain("route-pack/active-block cases as off-path");
    expect(analysis).toContain("audit the exact guard blocker");
  });

  test("Gemma post-370 focus pivots to on-path structural route-pack work", () => {
    const profile = [
      "info(forward):   prefill actual path: batched-route-pack default_batched=yes structural_batched=yes route_layers=30 queued_chunks=0",
      "info(forward):   path bytes: ssm 0.00 GiB attn 35.60 GiB dense 0.00 GiB moe-expert 44.91 GiB shared 16.95 GiB lm-head 23.38 GiB router 1.29 GiB",
      "info(forward):   prefill route pack: layers 30 slots 16800 avg_slots/layer 560.0 active_block_upper 5460 dense_dispatch_blocks 34560 active/dense 15.8%",
      "info(forward):   prefill route pack actual: samples 30 active_blocks 3400 avg/layer 113.3 min 102 max 128 actual/upper 62.3% saved_vs_upper 37.7%",
      "info(forward):   prefill route pack occupancy: full 1356 tail 2044 singleton_tail 656 padding_slots 10400 util 61.8% tail_blocks 60.1% singleton_tail 32.1%",
      "info(forward):   q8 hot #1: lm-head M=262144 K=2816 bytes=23.38 GiB calls=32",
    ].join("\n");
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE",
      metricMode: "prefill",
      bestTokPerSec: 370.1,
      currentBest: { tokPerSec: 370.1, containsReference: true },
      stalledCycles: 3,
      lastProfileOutput: profile,
      cycles: [
        makeCycle({ cycle: 90, kept: true, containsReference: true, tokPerSec: 362.3, description: "Added exact q5_1 3-route tail" }),
        makeCycle({ cycle: 97, kept: true, containsReference: true, tokPerSec: 368.4, description: "Added exact q4_k 2-route tail" }),
        makeCycle({ cycle: 98, kept: true, containsReference: true, tokPerSec: 369.1, description: "Added exact q4_k 3-route tail" }),
        makeCycle({ cycle: 99, kept: false, containsReference: true, tokPerSec: 362.9, description: "Added exact q4_k 4-route tail" }),
        makeCycle({ cycle: 100, kept: true, containsReference: true, tokPerSec: 370.1, description: "Added full 8-route q4_k path" }),
      ],
    });

    expect(gemmaPrefillActualPathIsStructuralRoutePack(profile)).toBe(true);
    const analysis = buildGemmaPrefillPostBreakthroughAnalysis(state).join("\n");
    expect(analysis).toContain("structural Gemma route-pack is live");
    expect(analysis).toContain("Latest route-pack occupancy");
    expect(analysis).toContain("q5_1 exact-6");
    expect(analysis).toContain("q4_k exact-4");
    expect(analysis).toContain("old structural guard work");
  });

  test("Gemma post-383 plateau pivots away from exhausted tail and metadata churn", () => {
    const profile = [
      "info(forward):   prefill actual path: batched-route-pack default_batched=yes structural_batched=yes route_layers=30 queued_chunks=0",
      "info(forward):   path bytes: ssm 0.00 GiB attn 35.60 GiB dense 0.00 GiB moe-expert 44.91 GiB shared 16.95 GiB lm-head 23.38 GiB router 1.29 GiB",
      "info(forward):   gpu-moe barriers/request: router 30 gate-up 990 activation 990 down 990 finalizer 4680 other 930",
      "info(forward):   prefill route pack: layers 30 slots 16800 avg_slots/layer 560.0 active_block_upper 5460 dense_dispatch_blocks 34560 active/dense 15.8%",
      "info(forward):   prefill route pack occupancy: full 1356 tail 2044 singleton_tail 656 padding_slots 10400 util 61.8% tail_blocks 60.1% singleton_tail 32.1%",
      "info(forward):   q8 hot #1: lm-head M=262144 K=2816 bytes=23.38 GiB calls=32",
      "info(forward):   q8 hot #2: shared M=2112 K=2816 bytes=11.30 GiB calls=1920",
    ].join("\n");
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE",
      metricMode: "prefill",
      bestTokPerSec: 383.6,
      currentBest: { tokPerSec: 383.6, containsReference: true },
      stalledCycles: 23,
      lastProfileOutput: profile,
      cycles: [
        makeCycle({ cycle: 124, kept: true, containsReference: true, tokPerSec: 383.6, description: "Bench coverage for production route-pack Q4_K GeGLU" }),
        makeCycle({ cycle: 141, kept: false, containsReference: true, tokPerSec: 363.3, description: "Added Gemma active-block MoE indirect Metal dispatch" }),
        makeCycle({ cycle: 142, kept: false, containsReference: true, tokPerSec: 360.4, description: "Reworked active route-pack block mapping with tiled expert scans" }),
        makeCycle({ cycle: 143, kept: false, containsReference: true, tokPerSec: 353.3, description: "Encoded active-block route counts into metadata and consumed them in kernels" }),
        makeCycle({ cycle: 145, kept: false, containsReference: true, tokPerSec: 371.0, description: "Added default-off route-pack tail histogram evidence" }),
        makeCycle({ cycle: 146, kept: false, containsReference: true, tokPerSec: 371.9, description: "Specialized Q4_K GeGLU input-row decode route / 8 -> route >> 3" }),
      ],
    });

    const analysis = buildGemmaPrefillPostBreakthroughAnalysis(state).join("\n");
    expect(analysis).toContain("POST-383 PLATEAU FACT");
    expect(analysis).toContain("do not reattempt Q4_K/Q5_1 route-tail");
    expect(analysis).toContain("Q8 LM-head/shared/attention");
    expect(analysis).toContain("gpu-moe finalizer/barrier-count");

    const pivot = buildStructuralPivotDirective(state).join("\n");
    expect(pivot).toContain("post-383 route-pack/tail/metadata family is exhausted");
    expect(pivot).toContain("public-suite validation");
  });

  test("Gemma metal-shapes route-token inference uses production prompt tokens", () => {
    expect(inferGemmaRouteTokensForMetalShapes(
      "info(forward):   prefill queued prefill: requests 1 prompt_tokens 70 chunks 12",
    )).toBe(70);

    expect(inferGemmaRouteTokensForMetalShapes(
      "info(forward):   prefill route pack: layers 30 slots 16800 avg_slots/layer 560.0 active_block_upper 5460",
    )).toBe(70);
  });

  test("Gemma plateau can refresh Metal-shapes after reverted analysis evidence", () => {
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE",
      metricMode: "prefill",
      bestTokPerSec: 383.6,
      currentBest: { tokPerSec: 383.6, containsReference: true },
      stalledCycles: 35,
      lastProfileOutput: "info(forward):   prefill actual path: batched-route-pack default_batched=yes structural_batched=yes route_layers=30 queued_chunks=0",
      lastMetalShapesCycle: 28,
      lastMetalShapesOk: true,
      lastMetalShapesOutput: "stale cycle-28 evidence",
      cycles: [
        makeCycle({ cycle: 124, kept: true, containsReference: true, tokPerSec: 383.6 }),
        makeCycle({ cycle: 158, kept: false, containsReference: true, tokPerSec: 362.1, stepKind: "analysis" }),
      ],
    });

    expect(shouldRunMetalShapesEvidence({
      state,
      cycle: 159,
      kept: false,
      containsReference: true,
      stepKind: "analysis",
      description: "Add exact-shape evidence for Gemma Q8 shared",
      selfAnalysis: "Analysis-only evidence; speed gate may revert this source change.",
      ideas: [],
    }, 1)).toBe(true);

    expect(shouldRunMetalShapesEvidence({
      state,
      cycle: 159,
      kept: false,
      containsReference: true,
      stepKind: "optimization",
      description: "Retune Q8 shared kernel",
      selfAnalysis: "No evidence step.",
      ideas: [],
    }, 1)).toBe(false);
  });

  test("Gemma plateau captures candidate evidence before reverting analysis source", () => {
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE",
      metricMode: "prefill",
      bestTokPerSec: 383.6,
      currentBest: { tokPerSec: 383.6, containsReference: true },
      stalledCycles: 50,
      lastProfileOutput: "info(forward):   prefill actual path: batched-route-pack default_batched=yes structural_batched=yes route_layers=30 queued_chunks=0",
      lastMetalShapesOutput: "stale cycle-28 evidence",
      cycles: [
        makeCycle({ cycle: 124, kept: true, containsReference: true, tokPerSec: 383.6 }),
        makeCycle({ cycle: 173, kept: false, containsReference: true, tokPerSec: 373.8, stepKind: "analysis" }),
      ],
    });

    expect(shouldRunCandidateEvidence({
      state,
      cycle: 174,
      containsReference: true,
      buildExitCode: 0,
      testExitCode: 0,
      runExitCode: 0,
      stepKind: "analysis",
      description: "Add profile-only Q8 batched GEMM token-tile accounting",
      selfAnalysis: "This analysis-only counter may be reverted by the speed gate.",
      ideas: [],
    }, 1, 1)).toEqual({ profile: true, metalShapes: true });
  });

  test("candidate evidence skips ordinary optimization churn and broken candidates", () => {
    const state = makeState({
      metricMode: "prefill",
      bestTokPerSec: 383.6,
      currentBest: { tokPerSec: 383.6, containsReference: true },
      stalledCycles: 50,
      lastProfileOutput: "info(forward):   prefill actual path: batched-route-pack default_batched=yes structural_batched=yes route_layers=30 queued_chunks=0",
      cycles: [makeCycle({ cycle: 124, kept: true, containsReference: true, tokPerSec: 383.6 })],
    });

    expect(shouldRunCandidateEvidence({
      state,
      cycle: 175,
      containsReference: true,
      buildExitCode: 0,
      testExitCode: 0,
      runExitCode: 0,
      stepKind: "optimization",
      description: "Retune Q8 shared kernel",
      selfAnalysis: "No evidence step.",
      ideas: [],
    }, 1, 1)).toEqual({ profile: false, metalShapes: false });

    expect(shouldRunCandidateEvidence({
      state,
      cycle: 175,
      containsReference: true,
      buildExitCode: 0,
      testExitCode: 1,
      runExitCode: 0,
      stepKind: "analysis",
      description: "Add exact-shape evidence for Gemma shared Q8",
      selfAnalysis: "Would be useful only if tests pass.",
      ideas: [],
    }, 1, 1)).toEqual({ profile: false, metalShapes: false });
  });

  test("Gemma plateau rejects neutral optimization churn but allows evidence work", () => {
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE",
      metricMode: "prefill",
      bestTokPerSec: 88.3,
      currentBest: { tokPerSec: 86.1, containsReference: true },
      stalledCycles: 6,
      cycles: [
        makeCycle({ cycle: 98, kept: true, containsReference: true, tokPerSec: 88.3 }),
        ...Array.from({ length: 6 }, (_, idx) => makeCycle({
          cycle: 99 + idx,
          kept: false,
          containsReference: true,
          tokPerSec: 86.0,
        })),
      ],
    });

    expect(shouldRejectPlateauNeutralKeep({
      state,
      stepKind: "optimization",
      description: "Retune weighted finalizer sigmoid cache",
      selfAnalysis: "Should be neutral to slightly faster.",
      verifyTokPerSec: 86.1,
      acceptedTokPerSec: 86.1,
      currentProgressBand: 0.25,
    })).toBe(true);

    expect(shouldRejectPlateauNeutralKeep({
      state,
      stepKind: "optimization",
      description: "Consume bench-metal-shapes gemma26_prefill_hot evidence before the next retune",
      selfAnalysis: "Adds exact-shape evidence for shared_gate_gemm.",
      verifyTokPerSec: 86.1,
      acceptedTokPerSec: 86.1,
      currentProgressBand: 0.25,
    })).toBe(false);

    expect(shouldRejectPlateauNeutralKeep({
      state,
      stepKind: "enablement",
      description: "Add validator guard reason for full batched prefill",
      selfAnalysis: "Unlocks the next structural fix.",
      verifyTokPerSec: 86.1,
      acceptedTokPerSec: 86.1,
      currentProgressBand: 0.25,
    })).toBe(false);
  });

  test("structural pivot directive appears for Gemma prefill plateau", () => {
    const state = makeState({
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Effort 11\nGemma 4 26B-A4B MoE prefill",
      metricMode: "prefill",
      bestTokPerSec: 88.3,
      currentBest: { tokPerSec: 86.1, containsReference: true },
      stalledCycles: 6,
      cycles: [
        makeCycle({ cycle: 98, kept: true, containsReference: true, tokPerSec: 88.3 }),
        ...Array.from({ length: 6 }, (_, idx) => makeCycle({
          cycle: 99 + idx,
          kept: false,
          containsReference: true,
          tokPerSec: 86.0,
          description: "Retune weighted finalizer sigmoid path",
        })),
      ],
    });

    const lines = buildStructuralPivotDirective(state).join("\n");
    expect(lines).toContain("HARD PIVOT MODE");
    expect(lines).toContain("Gemma26 M4 prefill");
    expect(lines).toContain("queued-prefill schedule");
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

  test("correctness prompt uses the active reference text", () => {
    const state = makeState();
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "contains the configured reference",
    });
    const reference = process.env.ZINC_REFERENCE_TEXT ?? "Paris";
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain(`Output MUST contain "${reference}"`);
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

// ── aggregateTimedSamples ───────────────────────────────────────────

describe("aggregateTimedSamples", () => {
  test("takes plain median without trimming below 5 samples", () => {
    const agg = aggregateTimedSamples([100, 102, 101], true);
    expect(agg.tokPerSec).toBe(101);
    expect(agg.trimmed).toBe(false);
    expect(agg.trimCount).toBe(0);
  });

  test("drops 1 high + 1 low for 5-6 samples", () => {
    const agg = aggregateTimedSamples([90, 100, 101, 102, 110], true);
    expect(agg.trimmed).toBe(true);
    expect(agg.trimCount).toBe(1);
    // After dropping 90 and 110 → median of [100,101,102] = 101
    expect(agg.tokPerSec).toBe(101);
  });

  test("drops 2 high + 2 low for 7+ samples", () => {
    const agg = aggregateTimedSamples([80, 90, 100, 101, 102, 110, 120], true);
    expect(agg.trimmed).toBe(true);
    expect(agg.trimCount).toBe(2);
    expect(agg.tokPerSec).toBe(101);
  });

  test("flags a bimodal/thermal straddle", () => {
    // median 101 with a low (98 ≤ 100.25) and high (104 ≥ 101.75) cluster
    const agg = aggregateTimedSamples([98, 101, 104], false);
    expect(agg.bimodal).toBe(true);
    expect(agg.range).toBeCloseTo(6, 5);
  });

  test("does not flag a tight cluster as bimodal", () => {
    const agg = aggregateTimedSamples([101, 101.2, 101.4], false);
    expect(agg.bimodal).toBe(false);
  });

  test("handles empty samples", () => {
    const agg = aggregateTimedSamples([], true);
    expect(agg.tokPerSec).toBeNull();
  });
});

// ── shouldConfirmCandidate ──────────────────────────────────────────

describe("shouldConfirmCandidate", () => {
  const base = {
    containsReference: true,
    ranOk: true,
    verifyTps: 102,
    bestTps: 100,
    improveBand: 2,
    bimodal: false,
    sampleCount: 5,
    confirmRuns: 6,
  };

  test("confirms a candidate sitting right on the promotion line", () => {
    // promotion line = 100 + 2 = 102; candidate at 102 is exactly borderline
    expect(shouldConfirmCandidate(base)).toBe(true);
  });

  test("confirms a clear win that is still within one band of the line", () => {
    expect(shouldConfirmCandidate({ ...base, verifyTps: 103.5 })).toBe(true);
  });

  test("does not confirm a candidate far above the promotion line", () => {
    expect(shouldConfirmCandidate({ ...base, verifyTps: 110 })).toBe(false);
  });

  test("does not confirm a clear regression far below the line", () => {
    expect(shouldConfirmCandidate({ ...base, verifyTps: 95 })).toBe(false);
  });

  test("always confirms when samples are bimodal, even far from the line", () => {
    expect(shouldConfirmCandidate({ ...base, verifyTps: 110, bimodal: true })).toBe(true);
  });

  test("never confirms incorrect output (reverted regardless of speed)", () => {
    expect(shouldConfirmCandidate({ ...base, containsReference: false })).toBe(false);
  });

  test("never confirms a crashed run", () => {
    expect(shouldConfirmCandidate({ ...base, ranOk: false })).toBe(false);
  });

  test("skips confirmation when already richly sampled", () => {
    expect(shouldConfirmCandidate({ ...base, sampleCount: 9 })).toBe(false);
  });

  test("disabled when confirmRuns is 0", () => {
    expect(shouldConfirmCandidate({ ...base, confirmRuns: 0 })).toBe(false);
  });
});

// ── recordNearMiss + buildNearMissDirective ─────────────────────────

describe("recordNearMiss", () => {
  const cand = {
    cycle: 296,
    tokPerSec: 109,
    ranOk: true,
    containsReference: false,
    acceptedTps: 102,
    description: "route-packed F32 shared-gate reuse",
    selfAnalysis: "expected +5-20 tok/s if correctness holds",
    outputText: "!!!!!!!!!!!!!!!!",
  };

  test("records a fast-but-incorrect result as a near-miss", () => {
    const state = makeState();
    expect(recordNearMiss(state, cand)).toBe(true);
    expect(state.bestIncorrect?.tokPerSec).toBe(109);
    expect(state.bestIncorrect?.cycle).toBe(296);
    expect(state.bestIncorrect?.gainPctOverAccepted).toBeCloseTo((7 / 102) * 100, 4);
  });

  test("ignores correct output", () => {
    const state = makeState();
    expect(recordNearMiss(state, { ...cand, containsReference: true })).toBe(false);
    expect(state.bestIncorrect).toBeUndefined();
  });

  test("ignores crashed runs", () => {
    const state = makeState();
    expect(recordNearMiss(state, { ...cand, ranOk: false })).toBe(false);
  });

  test("ignores results that are not meaningfully faster than accepted", () => {
    const state = makeState();
    // +1% < default 2% near-miss threshold
    expect(recordNearMiss(state, { ...cand, tokPerSec: 103, acceptedTps: 102 })).toBe(false);
  });

  test("only updates when strictly faster than the existing near-miss", () => {
    const state = makeState({ bestIncorrect: { cycle: 296, tokPerSec: 109, gainPctOverAccepted: 6.8, description: "x", selfAnalysis: "", outputText: "!!!!" } });
    expect(recordNearMiss(state, { ...cand, cycle: 319, tokPerSec: 108 })).toBe(false);
    expect(state.bestIncorrect?.cycle).toBe(296);
    expect(recordNearMiss(state, { ...cand, cycle: 322, tokPerSec: 111 })).toBe(true);
    expect(state.bestIncorrect?.cycle).toBe(322);
    expect(state.bestIncorrect?.tokPerSec).toBe(111);
  });
});

describe("buildNearMissDirective", () => {
  const state = makeState({
    bestIncorrect: {
      cycle: 296,
      tokPerSec: 109,
      gainPctOverAccepted: 6.5,
      description: "route-packed F32 shared-gate reuse",
      selfAnalysis: "expected +5-20 tok/s if correctness holds",
      outputText: "!!!!!!!!!!!!!!!!",
    },
  });

  test("emits a bisection directive with the real 35B layer-cap/validation flags the source actually parses", () => {
    const lines = buildNearMissDirective(state, 102, false).join("\n");
    expect(lines).toContain("KNOWN NEAR-MISS");
    expect(lines).toContain("109.0"); // matches fixture tokPerSec=109
    expect(lines).toContain("ZINC_QWEN36_35B_ROUTE_PACK_PREFIX_LAYERS");
    expect(lines).toContain("ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_FULL_BISECT");
    expect(lines).toContain("ZINC_QWEN36_35B_PREFILL_VALIDATE");
    expect(lines).toContain("BISECT");
  });

  test("escalates wording to a plateau redirect when stalled but family-revert count is low", () => {
    const lines = buildNearMissDirective(state, 102, true).join("\n");
    expect(lines).toContain("PLATEAU REDIRECT");
  });

  test("carries the original agent's selfAnalysis verbatim so the next cycle sees the plan", () => {
    const lines = buildNearMissDirective(state, 102, false).join("\n");
    expect(lines).toContain("Prior reasoning recorded with this near-miss");
    expect(lines).toContain("expected +5-20 tok/s"); // verbatim slice from the fixture's selfAnalysis
  });

  test("escalates to DEADLOCK mode after ≥5 route-pack family cycles all reverted", () => {
    const s = makeState({
      bestIncorrect: {
        cycle: 9,
        tokPerSec: 106.4,
        gainPctOverAccepted: 4.6,
        description: "Default-on route-pack capped at layers 0,1,2",
        selfAnalysis: "Bisect: layer 3 attention may be where F32 shared-gate diverges",
        outputText: "!!!!!!!!!!!!!!!!",
      },
      cycles: [
        makeCycle({ cycle: 10, kept: false, tokPerSec: 106.0, containsReference: false, description: "Cap route-pack at layer 0 only" }),
        makeCycle({ cycle: 11, kept: false, tokPerSec: 105.9, containsReference: false, description: "Route-pack with shared-gate scalar replay" }),
        makeCycle({ cycle: 12, kept: false, tokPerSec: 106.2, containsReference: false, description: "Route-pack prefix layers 0..1 only" }),
        makeCycle({ cycle: 13, kept: false, tokPerSec: 106.1, containsReference: false, description: "Route-pack with moe-route scatter rewrite" }),
        makeCycle({ cycle: 14, kept: false, tokPerSec: 105.8, containsReference: false, description: "Shared-gate fused materialization variant" }),
      ],
    });
    expect(countNearMissFamilyReverts(s)).toBe(5);
    const lines = buildNearMissDirective(s, 102, true).join("\n");
    expect(lines).toContain("DEADLOCK");
    expect(lines).toContain("STOP guessing");
    expect(lines).toContain("Strengthen the validator");
  });

  test("countNearMissFamilyReverts ignores cycles before the near-miss and kept/correct cycles", () => {
    const s = makeState({
      bestIncorrect: {
        cycle: 9,
        tokPerSec: 106,
        gainPctOverAccepted: 4,
        description: "route-pack",
        selfAnalysis: "",
        outputText: "!",
      },
      cycles: [
        makeCycle({ cycle: 5, kept: false, tokPerSec: 100, containsReference: false, description: "route-pack early experiment" }),
        makeCycle({ cycle: 10, kept: true, tokPerSec: 101, containsReference: true, description: "route-pack neutral keep" }),
        makeCycle({ cycle: 11, kept: false, tokPerSec: 101, containsReference: false, description: "unrelated kernel retune" }),
        makeCycle({ cycle: 12, kept: false, tokPerSec: 106, containsReference: false, description: "shared-gate variant" }),
      ],
    });
    expect(countNearMissFamilyReverts(s)).toBe(1); // only cycle 12 qualifies
  });

  test("is silent when no near-miss is recorded", () => {
    expect(buildNearMissDirective(makeState(), 102, false)).toEqual([]);
  });

  test("is silent once the accepted tree catches up to the near-miss", () => {
    expect(buildNearMissDirective(state, 109, false)).toEqual([]);
  });

  test("backfillNearMiss seeds the fastest reverted-incorrect cycle from history", () => {
    const s = makeState({
      bestTokPerSec: 102,
      cycles: [
        makeCycle({ cycle: 285, kept: true, tokPerSec: 102, containsReference: true, outputText: "Paris" }),
        makeCycle({ cycle: 296, kept: false, tokPerSec: 109.1, containsReference: false, outputText: "!!!!!!!!!!!!!!!!", description: "route-pack F32 shared-gate" }),
        makeCycle({ cycle: 319, kept: false, tokPerSec: 109.0, containsReference: false, outputText: "!!!!" }),
        // a crash should be ignored even if "fast"
        makeCycle({ cycle: 300, kept: false, tokPerSec: 120, containsReference: false, runExitCode: 139, outputText: "" }),
      ],
    });
    expect(backfillNearMiss(s)).toBe(true);
    expect(s.bestIncorrect?.cycle).toBe(296);
    expect(s.bestIncorrect?.tokPerSec).toBe(109.1);
  });

  test("backfillNearMiss is a no-op when one is already recorded", () => {
    const s = makeState({ bestIncorrect: { cycle: 1, tokPerSec: 109, gainPctOverAccepted: 6, description: "x", selfAnalysis: "", outputText: "!" } });
    expect(backfillNearMiss(s)).toBe(false);
    expect(s.bestIncorrect?.cycle).toBe(1);
  });

  test("backfillNearMiss ignores history with no qualifying revert", () => {
    const s = makeState({ bestTokPerSec: 102, cycles: [makeCycle({ cycle: 1, kept: true, tokPerSec: 102, containsReference: true })] });
    expect(backfillNearMiss(s)).toBe(false);
  });

  test("buildPrompt surfaces the near-miss directive in the optimize phase", () => {
    const result = makeResult({
      tokPerSec: 102,
      containsReference: true,
      strongAnswer: true,
      outputText: "The capital of France is Paris.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("KNOWN NEAR-MISS");
    expect(prompt).toContain("ZINC_QWEN36_35B_ROUTE_PACK_PREFIX_LAYERS");
  });
});

// ── parseDiagnosticEnv + near-miss diagnostic surfacing ─────────────

describe("parseDiagnosticEnv", () => {
  test("parses space-separated KEY=VALUE pairs", () => {
    const env = parseDiagnosticEnv("ZINC_FOO=1 ZINC_BAR=value-with-dash");
    expect(env).toEqual({ ZINC_FOO: "1", ZINC_BAR: "value-with-dash" });
  });

  test("collapses whitespace and skips malformed pairs", () => {
    const env = parseDiagnosticEnv("  ZINC_A=1   nokey nokeyhere==  ZINC_B=2  ");
    expect(env.ZINC_A).toBe("1");
    expect(env.ZINC_B).toBe("2");
    expect(env.nokey).toBeUndefined();
  });

  test("empty input yields empty record", () => {
    expect(parseDiagnosticEnv("")).toEqual({});
    expect(parseDiagnosticEnv("   ")).toEqual({});
  });
});

describe("buildNearMissDirective surfaces diagnostic output", () => {
  test("splices the latest diagnostic into the directive when present", () => {
    const state = makeState({
      bestIncorrect: {
        cycle: 9,
        tokPerSec: 106.4,
        gainPctOverAccepted: 4.6,
        description: "route-pack F32 shared-gate prefix",
        selfAnalysis: "Cap=3 hit 106.4 but broke output",
        outputText: "!!!!",
      },
      lastNearMissDiagnostic: {
        cycle: 51,
        envApplied: ["ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_FULL_BISECT", "ZINC_QWEN36_LAYER0_ROUTE_PACK_PREFILL"],
        output: "TENSOR_DIVERGE layer=3 tensor=moe_route_scatter_shared_residual_gate_f32 max_abs=2.1e-04",
      },
    });
    const lines = buildNearMissDirective(state, 102, false).join("\n");
    expect(lines).toContain("Latest near-miss diagnostic");
    expect(lines).toContain("cycle 51");
    expect(lines).toContain("ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_FULL_BISECT");
    expect(lines).toContain("TENSOR_DIVERGE layer=3");
  });

  test("omits the diagnostic section when none captured", () => {
    const state = makeState({
      bestIncorrect: {
        cycle: 9,
        tokPerSec: 106.4,
        gainPctOverAccepted: 4.6,
        description: "route-pack",
        selfAnalysis: "x",
        outputText: "!",
      },
    });
    const lines = buildNearMissDirective(state, 102, false).join("\n");
    expect(lines).not.toContain("Latest near-miss diagnostic");
  });
});

// ── noiseAwareImproveBand ──────────────────────────────────────────

describe("noiseAwareImproveBand", () => {
  test("leaves the base band untouched when samples are tight", () => {
    expect(noiseAwareImproveBand(1.0, [50.1, 50.2, 50.15])).toBe(1.0);
  });

  test("inflates the band to ~30% of the observed range when samples are noisy", () => {
    // range = 11, 30 % = 3.3 → larger than 1.0 base, so wins
    expect(noiseAwareImproveBand(1.0, [43.3, 51.0, 49.3, 47.6, 39.2])).toBeCloseTo(3.54, 1);
  });

  test("respects the base band when noise is small", () => {
    expect(noiseAwareImproveBand(2.0, [50, 50.5, 51])).toBe(2.0); // 1*0.3 < 2
  });

  test("returns the base band on insufficient samples", () => {
    expect(noiseAwareImproveBand(1.0, [50])).toBe(1.0);
    expect(noiseAwareImproveBand(1.0, [50, 51])).toBe(1.0);
  });

  test("safe with empty / identical samples", () => {
    expect(noiseAwareImproveBand(1.0, [])).toBe(1.0);
    expect(noiseAwareImproveBand(1.0, [50, 50, 50])).toBe(1.0); // range=0
  });
});

// ── buildStepKindDiversityNudge ────────────────────────────────────

describe("buildStepKindDiversityNudge", () => {
  const optCycle = (n: number) => makeCycle({ cycle: n, stepKind: "optimization", kept: false, containsReference: true });
  const anlCycle = (n: number) => makeCycle({ cycle: n, stepKind: "analysis", kept: false, containsReference: true });

  test("fires when recent window is all-optimization with no analysis/enablement", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => optCycle(i + 1));
    const lines = buildStepKindDiversityNudge({ cycles });
    expect(lines.length).toBeGreaterThan(0);
    expect(lines[0]).toContain("STEP-KIND IMBALANCE");
    expect(lines.join("\n")).toContain("microbench");
  });

  test("silent when at least one analysis or enablement cycle appears in the window", () => {
    const cycles = [optCycle(1), optCycle(2), optCycle(3), optCycle(4), optCycle(5), optCycle(6), optCycle(7), optCycle(8), optCycle(9), anlCycle(10)];
    expect(buildStepKindDiversityNudge({ cycles })).toEqual([]);
  });

  test("silent for short histories (lets fresh loops iterate)", () => {
    const cycles = [optCycle(1), optCycle(2), optCycle(3)];
    expect(buildStepKindDiversityNudge({ cycles })).toEqual([]);
  });

  test("allows one non-optimization step in the window before silencing", () => {
    // 9 of 10 optimization; the one fix shouldn't qualify as analysis/enablement
    const cycles = [
      ...Array.from({ length: 9 }, (_, i) => optCycle(i + 1)),
      makeCycle({ cycle: 10, stepKind: "fix", kept: false, containsReference: true }),
    ];
    const lines = buildStepKindDiversityNudge({ cycles });
    expect(lines.length).toBeGreaterThan(0); // fires: 9 opt + 1 fix (not analysis)
  });
});

// ── plateau stop / cooldown / finalization ─────────────────────────

describe("plateau controls", () => {
  test("auto-stop fires after a long consecutive revert streak", () => {
    const cycles = [
      makeCycle({ cycle: 83, kept: true, containsReference: true, tokPerSec: 82.1 }),
      ...Array.from({ length: 18 }, (_, idx) =>
        makeCycle({ cycle: 84 + idx, kept: false, containsReference: true, tokPerSec: 79.8 }),
      ),
    ];
    const decision = detectAutoStopForPlateau({ cycles }, { revertStreak: 18, noBestCycles: 40 });
    expect(decision.stop).toBe(true);
    expect(decision.reason).toContain("18 consecutive");
    expect(decision.bestCycle).toBe(83);
  });

  test("auto-stop fires when no promoted best appears for the configured window", () => {
    const cycles = [
      makeCycle({ cycle: 83, kept: true, containsReference: true, tokPerSec: 82.1 }),
      ...Array.from({ length: 20 }, (_, idx) =>
        makeCycle({
          cycle: 84 + idx,
          kept: idx < 5,
          containsReference: true,
          tokPerSec: idx < 5 ? 81.4 : 79.8,
        }),
      ),
    ];
    const decision = detectAutoStopForPlateau({ cycles }, { revertStreak: 99, noBestCycles: 20 });
    expect(decision.stop).toBe(true);
    expect(decision.reason).toContain("20 cycles since promoted-best");
  });

  test("classifies simd_sum and Q8 repacked families", () => {
    const families = classifyAttemptFamilies(makeCycle({
      description: "Pack two simd_sum reductions in dmmv_q8_0_repacked_k2048_nr2_qwen.metal",
    }));
    expect(families).toContain("simd_sum/reduction packing");
    expect(families).toContain("Q8/repacked shape retune");
  });

  test("classifies exhausted Gemma route-tail and metadata families", () => {
    expect(classifyAttemptFamilies(makeCycle({
      description: "Specialized Q4_K GeGLU input-row decode route / 8 -> route >> 3",
    }))).toContain("Q4_K route-tail/GeGLU micro-variants");

    expect(classifyAttemptFamilies(makeCycle({
      description: "Added exact 7-route Q5_1 active-block MoE-down tail",
    }))).toContain("Q5 down/tail variants");

    expect(classifyAttemptFamilies(makeCycle({
      description: "Added Gemma active-block MoE indirect Metal dispatch",
    }))).toContain("active-block metadata/indirect dispatch");
  });

  test("family cooldown blocks repeated reverted variants without a kept win", () => {
    const state = makeState({
      cycles: Array.from({ length: 4 }, (_, idx) =>
        makeCycle({
          cycle: 100 + idx,
          kept: false,
          containsReference: true,
          tokPerSec: 79.8,
          description: "Another simd_sum(float2) pack in a Q8 repacked sibling",
        }),
      ),
    });
    const lines = buildFamilyCooldownDirective(state, 4, 3).join("\n");
    expect(lines).toContain("FAMILY COOLDOWN");
    expect(lines).toContain("simd_sum/reduction packing");
    expect(lines).toContain("fresh profile");
  });

  test("hard cooldown rejects same-family optimization without fresh quantified evidence", () => {
    const state = makeState({
      cycles: [
        makeCycle({ cycle: 1, kept: false, stepKind: "optimization", description: "exact q4_k route tail variant" }),
        makeCycle({ cycle: 2, kept: false, stepKind: "optimization", description: "q4_k GeGLU singleton route-pack tail" }),
        makeCycle({ cycle: 3, kept: false, stepKind: "optimization", description: "active-block q4_k gate/up exact route tail" }),
      ],
    });

    const reason = cooledFamilyRejectionReason({
      state,
      stepKind: "optimization",
      description: "Add another q4_k route-pack tail fast path",
      selfAnalysis: "Same local retune without new measurements.",
      ideas: [],
      windowSize: 3,
      threshold: 3,
    });
    expect(reason).toContain("cooled-down family");
    expect(reason).toContain("Q4_K route-tail/GeGLU micro-variants");
  });

  test("hard cooldown allows same-family optimization with quantified candidate evidence", () => {
    const state = makeState({
      cycles: [
        makeCycle({ cycle: 1, kept: false, stepKind: "optimization", description: "exact q4_k route tail variant" }),
        makeCycle({ cycle: 2, kept: false, stepKind: "optimization", description: "q4_k GeGLU singleton route-pack tail" }),
        makeCycle({ cycle: 3, kept: false, stepKind: "optimization", description: "active-block q4_k gate/up exact route tail" }),
      ],
    });

    const reason = cooledFamilyRejectionReason({
      state,
      stepKind: "optimization",
      description: "Add q4_k route-pack tail fast path",
      selfAnalysis: "candidate_metal_shapes showed the exact-shape microbench improved 1.688 ms -> 1.521 ms (+9.9%).",
      ideas: [],
      windowSize: 3,
      threshold: 3,
    });
    expect(reason).toBeNull();
  });

  test("structural pivot directive appears for Qwen36 decode plateau", () => {
    const state = makeState({
      effortId: 5,
      effortFile: "MULTI_HOUR_EFFORT_5_METAL_QWEN36_LOCAL_DECODE.md",
      effortPlan: "# Effort 5\nQwen3.6 35B local Metal decode",
      currentBest: { tokPerSec: 81.1, containsReference: true },
      stalledCycles: 18,
      cycles: [
        makeCycle({ cycle: 83, kept: true, containsReference: true, tokPerSec: 82.1 }),
        ...Array.from({ length: 12 }, (_, idx) =>
          makeCycle({
            cycle: 84 + idx,
            kept: false,
            containsReference: true,
            tokPerSec: 79.8,
            description: "Agent made another simd_sum Q8 shader variant",
          }),
        ),
      ],
    });
    const lines = buildStructuralPivotDirective(state).join("\n");
    expect(lines).toContain("HARD PIVOT MODE");
    expect(lines).toContain("MoE finalizer/barrier fusion");
    expect(lines).toContain("command-buffer batching");
  });

  test("best-tree finalization restores when current accepted tree is below promoted best", () => {
    const decision = shouldFinalizeBestTree(
      {
        bestTree: { cycle: 83, tokPerSec: 82.1, commitHash: "best" },
        currentBest: { tokPerSec: 81.1, containsReference: true },
      },
      "head",
    );
    expect(decision.finalize).toBe(true);
    expect(decision.reason).toContain("cycle 83");
  });

  test("best-tree candidate prefers the latest same-band kept-correct cycle", () => {
    const candidate = bestTreeCandidateCycle({
      bestTokPerSec: 90.8,
      cycles: [
        makeCycle({ cycle: 10, kept: true, containsReference: true, tokPerSec: 90.8, commitHash: "cycle-10" }),
        makeCycle({ cycle: 20, kept: true, containsReference: true, tokPerSec: 90.75, commitHash: "cycle-20" }),
        makeCycle({ cycle: 26, kept: true, containsReference: true, tokPerSec: 90.8, commitHash: "cycle-26" }),
        makeCycle({ cycle: 28, kept: true, containsReference: true, tokPerSec: 90.6, commitHash: "cycle-28" }),
      ],
    });
    expect(candidate?.cycle).toBe(26);
    expect(candidate?.commitHash).toBe("cycle-26");
  });

  test("best-tree finalization skips when already at the promoted commit", () => {
    const decision = shouldFinalizeBestTree(
      {
        bestTree: { cycle: 83, tokPerSec: 82.1, commitHash: "best" },
        currentBest: { tokPerSec: 81.1, containsReference: true },
      },
      "best",
    );
    expect(decision.finalize).toBe(false);
  });

  test("best-tree restore updates promoted commit to the finalize commit", () => {
    const state = makeState({
      bestTree: { cycle: 83, tokPerSec: 82.1, commitHash: "cycle-83" },
    });
    applyBestTreeRestoreCommit(state, "finalize-83");
    expect(state.bestTree?.cycle).toBe(83);
    expect(state.bestTree?.tokPerSec).toBe(82.1);
    expect(state.bestTree?.commitHash).toBe("finalize-83");
  });

  test("Gemma plateau restore fires when current accepted tree fell below promoted best", () => {
    const state = makeState({
      metricMode: "prefill",
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Gemma M4 effort",
      currentBest: { tokPerSec: 87.9, containsReference: true },
      bestTokPerSec: 88.3,
      bestTree: { cycle: 98, tokPerSec: 88.3, commitHash: "best-98" },
      stalledCycles: 16,
      cycles: [
        makeCycle({ cycle: 98, kept: true, containsReference: true, tokPerSec: 88.3 }),
        makeCycle({ cycle: 106, kept: true, containsReference: true, tokPerSec: 88.0 }),
        makeCycle({ cycle: 110, kept: true, containsReference: true, tokPerSec: 87.9 }),
        makeCycle({ cycle: 111, kept: false, containsReference: true, tokPerSec: 82.5 }),
        makeCycle({ cycle: 112, kept: false, containsReference: true, tokPerSec: 86.7 }),
        makeCycle({ cycle: 113, kept: false, containsReference: true, tokPerSec: 86.2 }),
        makeCycle({ cycle: 114, kept: false, containsReference: true, tokPerSec: 86.4 }),
      ],
    });
    const decision = shouldRestorePromotedBestDuringPlateau(state, "current-head");
    expect(decision.restore).toBe(true);
    expect(decision.reason).toContain("Gemma plateau active");
    expect(decision.reason).toContain("cycle 98");
  });

  test("Gemma plateau restore skips when current accepted tree is already at promoted-best speed", () => {
    const state = makeState({
      metricMode: "prefill",
      effortId: 11,
      effortFile: "MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md",
      effortPlan: "# Gemma M4 effort",
      currentBest: { tokPerSec: 88.28, containsReference: true },
      bestTokPerSec: 88.3,
      bestTree: { cycle: 98, tokPerSec: 88.3, commitHash: "best-98" },
      stalledCycles: 16,
      cycles: [
        makeCycle({ cycle: 98, kept: true, containsReference: true, tokPerSec: 88.3 }),
        makeCycle({ cycle: 114, kept: false, containsReference: true, tokPerSec: 86.4 }),
      ],
    });
    const decision = shouldRestorePromotedBestDuringPlateau(state, "current-head");
    expect(decision.restore).toBe(false);
    expect(decision.reason).toContain("within 0.05");
  });

  test("best-tree restore path parser dedupes git log name-only output", () => {
    expect(parseRestorePathList("\nsrc/compute/forward_metal.zig\n\nsrc/compute/forward_metal.zig\nsrc/shaders/metal/a.metal\n")).toEqual([
      "src/compute/forward_metal.zig",
      "src/shaders/metal/a.metal",
    ]);
  });
});

// ── buildCrossEffortStatus ─────────────────────────────────────────

describe("buildCrossEffortStatus", () => {
  test("silent when no cross-effort baseline is set", () => {
    expect(buildCrossEffortStatus({ crossEffortBaseline: null, lastCrossEffort: null })).toEqual([]);
  });

  test("silent when baseline exists but no measurement yet", () => {
    expect(
      buildCrossEffortStatus({
        crossEffortBaseline: { metric: "prefill", tokPerSec: 100, cycle: 1 },
        lastCrossEffort: null,
      }),
    ).toEqual([]);
  });

  test("plain status when other metric is stable (Δ > -1.5%)", () => {
    const lines = buildCrossEffortStatus({
      crossEffortBaseline: { metric: "prefill", tokPerSec: 100, cycle: 1 },
      lastCrossEffort: { metric: "prefill", tokPerSec: 99.5, cycle: 5, deltaPct: -0.5 },
    }).join("\n");
    expect(lines).toContain("Cross-effort status");
    expect(lines).not.toContain("REGRESSION");
    expect(lines).not.toContain("DRIFT");
  });

  test("DRIFT warning at -1.5% to -3% (tightened from prior -2%/-5%)", () => {
    const lines = buildCrossEffortStatus({
      crossEffortBaseline: { metric: "prefill", tokPerSec: 100, cycle: 1 },
      lastCrossEffort: { metric: "prefill", tokPerSec: 98, cycle: 10, deltaPct: -2 },
    }).join("\n");
    expect(lines).toContain("CROSS-EFFORT DRIFT");
    expect(lines).toContain("Watch for further drift");
  });

  test("REGRESSION warning at <= -3% (matches Effort docs' 3% prohibition)", () => {
    const lines = buildCrossEffortStatus({
      crossEffortBaseline: { metric: "prefill", tokPerSec: 100, cycle: 1 },
      lastCrossEffort: { metric: "prefill", tokPerSec: 96.5, cycle: 15, deltaPct: -3.5 },
    }).join("\n");
    expect(lines).toContain("CROSS-EFFORT REGRESSION");
    expect(lines).toContain("HURTING the other metric");
    expect(lines).toContain("GATE it behind an env flag");
    expect(lines).toContain("3.5%");
  });

  test("decode-v4-style -3.1% drift now triggers REGRESSION (was only DRIFT before)", () => {
    // This is the exact pattern the agent ignored across cycles 70-76 of v4.
    const lines = buildCrossEffortStatus({
      crossEffortBaseline: { metric: "prefill", tokPerSec: 93.9, cycle: 30 },
      lastCrossEffort: { metric: "prefill", tokPerSec: 91, cycle: 75, deltaPct: -3.1 },
    }).join("\n");
    expect(lines).toContain("CROSS-EFFORT REGRESSION");
  });
});
