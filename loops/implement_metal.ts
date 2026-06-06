#!/usr/bin/env bun
/**
 * ZINC Metal Implementation Loop
 *
 * Autonomous loop that iteratively implements the Metal/Apple Silicon inference
 * backend. Each cycle:
 *   1. Build locally (zig build -Doptimize=ReleaseFast by default)
 *   2. Run unit tests (zig build test)
 *   3. Run inference with model (zinc -m model.gguf --prompt "..." -n N)
 *   4. Analyze output: build errors? test failures? correct tokens? tok/s?
 *   5. Spawn AI agent to make ONE implementation step
 *   6. Agent edits files → loop back to 1
 *
 * Three phases:
 *   FIX       — build errors, test failures, crashes
 *   IMPLEMENT — wire up GPU layer dispatch, produce correct tokens
 *   OPTIMIZE  — once output matches reference: improve tok/s to ≥TARGET_TOK_PER_SEC
 *
 * Usage:
 *   bun loops/implement_metal.ts                     # run indefinitely
 *   bun loops/implement_metal.ts --cycles 100        # 100 cycles max
 *   bun loops/implement_metal.ts --dry-run           # build+run only, no agent
 */

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

// ── Color & display ──────────────────────────────────────────────────

const TTY = process.stdout.isTTY ?? false;
const NO_COLOR = "NO_COLOR" in process.env;
const FORCE_COLOR = process.env.FORCE_COLOR === "1" || process.env.CLICOLOR_FORCE === "1";
const COLOR_ENABLED = !NO_COLOR && (TTY || FORCE_COLOR);

function clr(code: string, text: string): string {
  return COLOR_ENABLED ? `\x1b[${code}m${text}\x1b[0m` : text;
}

const SEP = "─".repeat(64);

// ── Constants ────────────────────────────────────────────────────────

const REPO_ROOT = resolve(import.meta.dir, "..");
const EFFORTS_DIR = resolve(REPO_ROOT, "loops", "efforts");
const RESULTS_DIR = resolve(REPO_ROOT, ".metal_optimize");
const MODEL_ID = process.env.ZINC_MODEL_ID ?? "qwen36-35b-a3b-q4k-xl";
const MODEL_PATH = process.env.ZINC_MODEL ?? null;
const TEST_PROMPT = process.env.ZINC_TEST_PROMPT ?? "The capital of France is";
const PROMPT_MODE = process.env.ZINC_PROMPT_MODE ?? "raw";
export type MetricMode = "decode" | "prefill";
const METRIC_MODE = parseMetricModeEnv("ZINC_METRIC_MODE", "decode");
const METRIC_LABEL = METRIC_MODE === "prefill" ? "prefill tok/s" : "decode tok/s";
const MAX_TOKENS = parsePositiveIntEnv("ZINC_MAX_TOKENS", 64); // Enough tokens for stable throughput measurement
const REFERENCE_TEXT = process.env.ZINC_REFERENCE_TEXT ?? "Paris"; // Expected in correct output
const TARGET_TOK_PER_SEC = parsePositiveFloatEnv("ZINC_TARGET_TOK_PER_SEC", 50);
const QWEN36_PREFILL_INTERMEDIATE_TARGET = Math.max(50, TARGET_TOK_PER_SEC);
const BENCHMARK_RUNS = parsePositiveIntEnv("ZINC_BENCHMARK_RUNS", 3); // Number of TIMED inference runs (median across them)
// Pre-rolls discarded before any timed sample. Each fresh ZINC invocation
// resets Metal residency + GPU clocks, so the FIRST few samples in a series
// land cold-GPU while later samples land warm. On M1 Max for Qwen3-8B that
// shows up as ~8.6 cold vs ~10.6 warm — a ~20% spread that masks real
// regressions. Setting BENCHMARK_WARMUPS≥1 runs throwaway inferences first
// so timed samples are more likely to start with a warm-ish GPU baseline.
// Note: a process boundary still resets some state; for fully stable
// numbers, also raise BENCHMARK_RUNS so the median rejects outliers.
const BENCHMARK_WARMUPS = parsePositiveIntEnv("ZINC_BENCHMARK_WARMUPS", 1);
// Trim the high and low samples before taking the median. Only fires when
// BENCHMARK_RUNS ≥ 5, since trimming 3 samples down to 1 just picks the
// middle anyway. Mitigates the cold/warm process-boundary noise pattern
// without requiring zinc itself to support multi-prompt batched runs.
const BENCHMARK_TRIM = parseBoolEnv("ZINC_BENCHMARK_TRIM", true);
// Sleep between consecutive timed samples. Decode benchmarks at 128 tokens/sample × 5 samples
// sustain enough GPU load on M-series that samples bimodalize from thermal throttling
// (range 7–11 tok/s observed on Effort 5 cycle 1). A short cool-down between samples
// lets the GPU clocks settle and meaningfully tightens the sample range, which directly
// improves the keep-gate's signal-to-noise ratio. Default 0 = off; set to 3000–5000 ms
// for thermal-bound benches like 35B decode.
const BENCHMARK_COOLDOWN_MS = parsePositiveIntEnv("ZINC_BENCHMARK_COOLDOWN_MS", 0);
// Reject samples that generated fewer than this many tokens before aggregating
// the median. Decode benchmarks at MAX_TOKENS=128 that EOS after a single
// token (e.g., a chat-mode "Paris" answer) report tok/s computed from a single
// generation — too noisy and not comparable to llama.cpp's `tg128` reference.
// Effort 5 cycle 27 measured 74.63 tok/s on 1-token samples; the floor of the
// sample range was 61.3, so the trimmed median was upward-biased by ~10 tok/s.
// Default 0 = off (backward-compatible); set to e.g. 16 for decode runs so
// throughput is measured over a sustained generation window.
const MIN_DECODE_TOKENS = parsePositiveIntEnv("ZINC_MIN_DECODE_TOKENS", 0);
// Cross-effort guard: when optimizing one metric (e.g., decode), periodically
// measure the OTHER metric (e.g., prefill) so a cross-effort regression surfaces
// EARLY instead of after the run completes. Effort 5 decode run cycles 4-26
// touched SSM/Q8 shaders that are shared with prefill; the regression from
// 101.9 → ~89 prefill tok/s wasn't visible until an out-of-band verification.
//
// Setup: ZINC_CROSS_EFFORT_PROMPT (different prompt for the guard),
// ZINC_CROSS_EFFORT_METRIC ("prefill" or "decode" — opposite of main),
// ZINC_CROSS_EFFORT_PROMPT_MODE ("chat" or "raw"),
// ZINC_CROSS_EFFORT_MAX_TOKENS (e.g. 16 for prefill, 64 for decode),
// ZINC_CROSS_EFFORT_EVERY (default 5 — measure every N cycles, plus cycle 1).
const CROSS_EFFORT_PROMPT = process.env.ZINC_CROSS_EFFORT_PROMPT ?? "";
const CROSS_EFFORT_METRIC: MetricMode = parseMetricModeEnv("ZINC_CROSS_EFFORT_METRIC", METRIC_MODE === "decode" ? "prefill" : "decode");
const CROSS_EFFORT_PROMPT_MODE = (process.env.ZINC_CROSS_EFFORT_PROMPT_MODE ?? "chat").trim().toLowerCase();
const CROSS_EFFORT_MAX_TOKENS = parsePositiveIntEnv("ZINC_CROSS_EFFORT_MAX_TOKENS", 16);
const CROSS_EFFORT_EVERY = parsePositiveIntEnv("ZINC_CROSS_EFFORT_EVERY", 5);
// Confirmation re-run: when a candidate lands in the noise zone around the
// promotion boundary (or its samples were flagged bimodal/THERMAL), collect
// this many EXTRA timed samples and re-aggregate over the combined set before
// deciding keep/revert. This rescues real ~1-2% wins that a 3-5 sample median
// cannot distinguish from M-series thermal jitter, without paying the extra
// runtime on every cycle. Set to 0 to disable.
const BENCHMARK_CONFIRM_RUNS = parsePositiveIntEnv("ZINC_BENCHMARK_CONFIRM_RUNS", 6);
// A fast-but-incorrect result is tracked as a "near-miss" optimization target
// only when it beats the current accepted baseline by at least this percent.
// The route-packed F32 shared-gate path repeatedly hit ~109 tok/s (+6.5%) but
// broke output to "!!!!"; recording it lets the prompt redirect the agent at
// localizing the divergent layer instead of rediscovering the same dead end.
const NEAR_MISS_MIN_GAIN_PCT = parsePositiveFloatEnv("ZINC_NEAR_MISS_MIN_GAIN_PCT", 2);
// Extra environment variables applied to a single diagnostic inference run
// after the keep verdict on cycles where `bestIncorrect` is set. The diagnostic
// run never affects the keep gate; its stdout/stderr (last 4 KB) is captured
// into `state.lastNearMissDiagnostic` and surfaced in the next cycle's prompt.
//
// This breaks the chicken-and-egg from Effort 16 run 2: the kept tree had the
// fast path default-OFF, so the route-pack validator the agent built was dead
// code — it could never run under the harness's official measurement, so
// successive cycles got no per-tensor evidence even though the validator
// existed in source. Setting e.g.
//   ZINC_NEAR_MISS_DIAGNOSTIC_ENV="ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_FULL_BISECT=1 ZINC_QWEN36_LAYER0_ROUTE_PACK_PREFILL=1"
// lets the validator fire against the kept tree without the harness reverting
// for broken output (because the diagnostic run is not graded).
//
// Format: space-separated KEY=VALUE pairs. Empty = no diagnostic.
const NEAR_MISS_DIAGNOSTIC_ENV_RAW = process.env.ZINC_NEAR_MISS_DIAGNOSTIC_ENV ?? "";
const PROFILE_EVERY = parsePositiveIntEnv("ZINC_PROFILE_EVERY", 5); // Run with --profile every N cycles
const STALL_THRESHOLD = 5; // Cycles without tok/s improvement before studying references
const RECENT_PROGRESS_WINDOW = 10;
const QWEN36_PLATEAU_STALL_CYCLES = 20;
const QWEN36_PLATEAU_WINDOW = 32;
const GEMMA_PLATEAU_STALL_CYCLES = parsePositiveIntEnv("ZINC_GEMMA_PLATEAU_STALL_CYCLES", 6);
const GEMMA_PLATEAU_WINDOW = parsePositiveIntEnv("ZINC_GEMMA_PLATEAU_WINDOW", 18);
const GEMMA_POST_BREAKTHROUGH_FLOOR = parsePositiveFloatEnv("ZINC_GEMMA_POST_BREAKTHROUGH_FLOOR", 80);
const GEMMA_POST_ROUTEPACK_PLATEAU_FLOOR = parsePositiveFloatEnv("ZINC_GEMMA_POST_ROUTEPACK_PLATEAU_FLOOR", 380);
// Optional outer-harness exact-shape Metal evidence pass. Agents are blocked
// from running `bench-metal-shapes` because it reserves the Metal GPU; the
// harness can safely run it between cycles and splice the output into the next
// prompt. Disabled by default for existing efforts. For Gemma26 prefill plateau
// work, use:
//   ZINC_METAL_SHAPES_EVERY=3
//   ZINC_METAL_SHAPES_ARGS="--case gemma26_prefill_hot --pipeline production --route-tokens 20 --iterations 80 --warmup 10"
const METAL_SHAPES_EVERY = parsePositiveIntEnv("ZINC_METAL_SHAPES_EVERY", 0);
const METAL_SHAPES_ARGS_RAW = process.env.ZINC_METAL_SHAPES_ARGS ?? "";
const AUTO_STOP_PLATEAU = parseBoolEnv("ZINC_AUTO_STOP_PLATEAU", true);
const AUTO_STOP_REVERT_STREAK = parsePositiveIntEnv("ZINC_AUTO_STOP_REVERT_STREAK", 18);
const AUTO_STOP_NO_BEST_CYCLES = parsePositiveIntEnv("ZINC_AUTO_STOP_NO_BEST_CYCLES", 20);
const STRUCTURAL_PIVOT_REVERT_STREAK = parsePositiveIntEnv("ZINC_STRUCTURAL_PIVOT_REVERT_STREAK", 8);
const STRUCTURAL_PIVOT_NO_BEST_CYCLES = parsePositiveIntEnv("ZINC_STRUCTURAL_PIVOT_NO_BEST_CYCLES", 12);
const FAMILY_COOLDOWN_WINDOW = parsePositiveIntEnv("ZINC_FAMILY_COOLDOWN_WINDOW", 12);
const FAMILY_COOLDOWN_THRESHOLD = parsePositiveIntEnv("ZINC_FAMILY_COOLDOWN_THRESHOLD", 3);
const HARD_FAMILY_COOLDOWN = parseBoolEnv("ZINC_HARD_FAMILY_COOLDOWN", true);
const WORKLOAD_RESET_ON_CHANGE = parseBoolEnv("ZINC_WORKLOAD_RESET_ON_CHANGE", true);
const FINALIZE_BEST_TREE = parseBoolEnv("ZINC_FINALIZE_BEST_TREE", true);
const TEST_TIMEOUT_MS = parsePositiveIntEnv("ZINC_TEST_TIMEOUT_MS", 120_000);
const RUN_TIMEOUT_MS = parsePositiveIntEnv("ZINC_RUN_TIMEOUT_MS", 300_000);
const STOP_ON_TARGET = parseBoolEnv("ZINC_STOP_ON_TARGET", true);
const BUILD_OPTIMIZE = process.env.ZINC_BUILD_OPTIMIZE ?? "ReleaseFast";
const LOOP_COMMIT_PATHS = ["src/", "benchmarks/", "build.zig", "build.zig.zon"];

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git fetch:*)",
  "Bash(git merge:*)",
  "Bash(git pull:*)",
  "Bash(git push:*)",
  "Bash(git rebase:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
];

// ZINC reserves the Metal GPU exclusively — only one zinc process may own it.
// On the host (e.g. the Claude agent, unlike sandboxed Codex) an agent CAN run
// the model binary, which then holds the GPU and collides with the harness's
// own exclusive measurement run, crashing the cycle with "GPU metal:0 is
// already reserved". The harness owns ALL Metal measurement, so hard-block the
// agent from launching the model binary or any GPU-reserving build step. Plain
// `zig build` and `zig build test` stay allowed (they don't reserve the GPU).
const BLOCKED_MODEL_RUN_OPS = [
  "Bash(./zig-out/bin/zinc:*)",
  "Bash(zig-out/bin/zinc:*)",
  "Bash(zig build run:*)",
  "Bash(zig build bench:*)",
  "Bash(zig build bench-metal:*)",
  "Bash(zig build bench-metal-shapes:*)",
  "Bash(zig build bench-metal-gemm-q4k:*)",
  "Bash(zig build bench-metal-dmmv-q4k:*)",
  "Bash(zig build hot-bench:*)",
];

const BLOCKED_AGENT_OPS = [...BLOCKED_GIT_OPS, ...BLOCKED_MODEL_RUN_OPS];

type AgentKind = "claude" | "codex";

function parsePositiveIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw == null || raw.trim() === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parsePositiveFloatEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw == null || raw.trim() === "") return fallback;
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseBoolEnv(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw == null || raw.trim() === "") return fallback;
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return fallback;
}

function parseMetricModeEnv(name: string, fallback: MetricMode): MetricMode {
  const raw = process.env[name];
  if (raw == null || raw.trim() === "") return fallback;
  const normalized = raw.trim().toLowerCase();
  if (normalized === "decode" || normalized === "prefill") return normalized;
  return fallback;
}

function zincModelArgs(): string[] {
  return MODEL_PATH ? ["-m", MODEL_PATH] : ["--model-id", MODEL_ID];
}

function managedModelPathForBenchmark(): string {
  if (MODEL_PATH) return MODEL_PATH;
  const home = process.env.HOME ?? "";
  return join(home, "Library", "Caches", "zinc", "models", "models", MODEL_ID, "model.gguf");
}

function zincPromptArgs(): string[] {
  const args = ["--prompt", TEST_PROMPT];
  if (PROMPT_MODE === "chat") args.push("--chat");
  return args;
}

function shortHash(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

export type WorkloadContract = {
  model: string;
  metricMode: MetricMode;
  promptMode: string;
  maxTokens: number;
  referenceHash: string;
  rawPromptHash: string;
  preparedPromptHash: string | null;
  promptTokens: number | null;
};

export function parsePromptFingerprint(output: string): { rawPromptHash: string | null; preparedPromptHash: string | null; promptTokens: number | null } {
  const fp = output.match(/Prompt fingerprint:\s*raw=([0-9a-f]+)\s+prepared=([0-9a-f]+)\s+mode=\S+\s+prompt_tokens=(\d+)/i);
  if (fp) {
    return {
      rawPromptHash: fp[1],
      preparedPromptHash: fp[2],
      promptTokens: Number.parseInt(fp[3], 10),
    };
  }
  return {
    rawPromptHash: null,
    preparedPromptHash: null,
    promptTokens: parsePrefillTokenCount(output),
  };
}

function workloadFromMeasurement(result: Pick<BuildRunResult, "runOutput" | "promptTokens">): WorkloadContract {
  const parsed = parsePromptFingerprint(result.runOutput);
  return {
    model: displayModelLabel(),
    metricMode: METRIC_MODE,
    promptMode: PROMPT_MODE,
    maxTokens: MAX_TOKENS,
    referenceHash: shortHash(REFERENCE_TEXT),
    rawPromptHash: parsed.rawPromptHash ?? shortHash(TEST_PROMPT),
    preparedPromptHash: parsed.preparedPromptHash,
    promptTokens: parsed.promptTokens ?? result.promptTokens ?? null,
  };
}

export function workloadMismatchReason(expected: WorkloadContract | null | undefined, actual: WorkloadContract | null | undefined): string | null {
  if (!expected || !actual) return null;
  const mismatches: string[] = [];
  if (expected.model !== actual.model) mismatches.push(`model ${expected.model} -> ${actual.model}`);
  if (expected.metricMode !== actual.metricMode) mismatches.push(`metric ${expected.metricMode} -> ${actual.metricMode}`);
  if (expected.promptMode !== actual.promptMode) mismatches.push(`prompt mode ${expected.promptMode} -> ${actual.promptMode}`);
  if (expected.maxTokens !== actual.maxTokens) mismatches.push(`max tokens ${expected.maxTokens} -> ${actual.maxTokens}`);
  if (expected.referenceHash !== actual.referenceHash) mismatches.push("reference text hash changed");
  if (expected.rawPromptHash !== actual.rawPromptHash) mismatches.push("raw prompt hash changed");
  if (expected.preparedPromptHash && actual.preparedPromptHash && expected.preparedPromptHash !== actual.preparedPromptHash) {
    mismatches.push("prepared prompt hash changed");
  }
  if (expected.promptTokens != null && actual.promptTokens != null && expected.promptTokens !== actual.promptTokens) {
    mismatches.push(`prompt tokens ${expected.promptTokens} -> ${actual.promptTokens}`);
  }
  return mismatches.length > 0 ? mismatches.join("; ") : null;
}

function displayModelLabel(): string {
  return MODEL_PATH ? basename(MODEL_PATH) : MODEL_ID;
}

function isGemmaRun(state?: Pick<RunState, "effortId" | "effortFile" | "effortPlan">): boolean {
  const effortText = [
    state?.effortFile ?? "",
    state?.effortPlan ?? "",
  ].join("\n").toLowerCase();
  if (effortText.trim().length > 0) return effortText.includes("gemma");
  const model = displayModelLabel().toLowerCase();
  return model.includes("gemma") ||
    state?.effortId === 11;
}

function isQwen36PrefillRun(
  state?: Pick<RunState, "effortId" | "effortFile" | "effortPlan">,
): boolean {
  if (!isQwen36LargeMoeRun(state)) return false;
  const effortText = [
    state?.effortFile ?? "",
    state?.effortPlan ?? "",
  ].join("\n").toLowerCase();
  const prefill = METRIC_MODE === "prefill" ||
    state?.effortId === 16 ||
    effortText.includes("prefill");
  return prefill;
}

function isQwen36LargeMoeRun(
  state?: Pick<RunState, "effortId" | "effortFile" | "effortPlan">,
): boolean {
  const model = displayModelLabel().toLowerCase();
  const effortText = [
    state?.effortFile ?? "",
    state?.effortPlan ?? "",
  ].join("\n").toLowerCase();
  const qwen36 = model.includes("qwen36-35b") ||
    model.includes("qwen3.6") ||
    effortText.includes("qwen36") ||
    effortText.includes("qwen 3.6") ||
    effortText.includes("qwen3.6");
  const largeMoe = model.includes("35b") ||
    effortText.includes("35b") ||
    effortText.includes("35b-a3b");
  return qwen36 && largeMoe && !isGemmaRun(state);
}

function promptTrunc(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + "..." : s;
}

function cycleMatchesWorkload(
  state: Pick<RunState, "workload">,
  cycle: Pick<CycleResult, "workload">,
): boolean {
  if (!state.workload) return true;
  if (!cycle.workload) return false;
  return workloadMismatchReason(state.workload, cycle.workload) == null;
}

function bestAcceptedTokPerSec(state: RunState, lastResult: BuildRunResult): number | null {
  const candidates: number[] = [];
  if (Number.isFinite(state.bestTokPerSec) && state.bestTokPerSec > 0) {
    candidates.push(state.bestTokPerSec);
  }
  if (lastResult.strongAnswer && lastResult.tokPerSec != null && lastResult.tokPerSec > 0) {
    candidates.push(lastResult.tokPerSec);
  }
  for (const c of state.cycles) {
    if (!cycleMatchesWorkload(state, c)) continue;
    if (c.kept && c.containsReference && c.tokPerSec != null && c.tokPerSec > 0) {
      candidates.push(c.tokPerSec);
    }
  }
  return candidates.length > 0 ? Math.max(...candidates) : null;
}

function smallAcceptedProgressBand(anchorTokPerSec: number): number {
  return Math.max(0.15, anchorTokPerSec * 0.003);
}

export function bestKeptCorrectTokPerSec(
  state: Pick<RunState, "cycles" | "bestTokPerSec" | "currentBest" | "workload">,
): number {
  const candidates: number[] = [];
  if (Number.isFinite(state.bestTokPerSec) && state.bestTokPerSec > 0) {
    candidates.push(state.bestTokPerSec);
  }
  if (
    state.currentBest?.containsReference &&
    state.currentBest.tokPerSec != null &&
    state.currentBest.tokPerSec > 0
  ) {
    candidates.push(state.currentBest.tokPerSec);
  }
  for (const c of state.cycles) {
    if (!cycleMatchesWorkload(state, c)) continue;
    if (c.kept && c.containsReference && c.tokPerSec != null && c.tokPerSec > 0) {
      candidates.push(c.tokPerSec);
    }
  }
  return candidates.length > 0 ? Math.max(...candidates) : 0;
}

export function currentAcceptedTokPerSec(state: Pick<RunState, "cycles" | "currentBest" | "bestTokPerSec" | "workload">): number {
  if (
    state.currentBest?.containsReference &&
    state.currentBest.tokPerSec != null &&
    state.currentBest.tokPerSec > 0
  ) {
    return state.currentBest.tokPerSec;
  }
  for (let idx = state.cycles.length - 1; idx >= 0; idx--) {
    const cycle = state.cycles[idx];
    if (!cycleMatchesWorkload(state, cycle)) continue;
    if (cycle.kept && cycle.containsReference && cycle.tokPerSec != null && cycle.tokPerSec > 0) {
      return cycle.tokPerSec;
    }
  }
  return Number.isFinite(state.bestTokPerSec) && state.bestTokPerSec > 0 ? state.bestTokPerSec : 0;
}

function correctResultTokPerSec(result: Pick<BuildRunResult, "containsReference" | "tokPerSec">): number {
  return result.containsReference && result.tokPerSec != null && result.tokPerSec > 0 ? result.tokPerSec : 0;
}

export function keepBaselinesForCycle(
  state: Pick<RunState, "cycles" | "bestTokPerSec" | "currentBest" | "workload">,
  cycleBaseline: Pick<BuildRunResult, "containsReference" | "tokPerSec">,
): { bestTokPerSec: number; acceptedTokPerSec: number } {
  const measuredBaseline = correctResultTokPerSec(cycleBaseline);
  const bestTokPerSec = Math.max(bestKeptCorrectTokPerSec(state), measuredBaseline);
  const acceptedTokPerSec = Math.max(currentAcceptedTokPerSec(state), measuredBaseline);
  return { bestTokPerSec, acceptedTokPerSec };
}

function normalizeStateBestTokPerSec(state: RunState): void {
  state.bestTokPerSec = Math.max(state.bestTokPerSec, bestKeptCorrectTokPerSec(state));
}

export function recentAcceptedProgress(
  state: Pick<RunState, "cycles" | "bestTokPerSec" | "currentBest" | "workload">,
  window = RECENT_PROGRESS_WINDOW,
): { start: number; end: number; delta: number; threshold: number; hasProgress: boolean } {
  const recent = state.cycles.slice(-window);
  if (recent.length === 0) {
    const current = currentAcceptedTokPerSec(state);
    const threshold = smallAcceptedProgressBand(current);
    return { start: current, end: current, delta: 0, threshold, hasProgress: false };
  }

  const priorKept = state.cycles
    .slice(0, -recent.length)
    .filter(c => cycleMatchesWorkload(state, c))
    .filter(c => c.kept && c.containsReference && c.tokPerSec != null && c.tokPerSec > 0)
    .map(c => c.tokPerSec!);
  const recentKept = recent
    .filter(c => cycleMatchesWorkload(state, c))
    .filter(c => c.kept && c.containsReference && c.tokPerSec != null && c.tokPerSec > 0)
    .map(c => c.tokPerSec!);

  const start = priorKept.length > 0
    ? Math.max(...priorKept)
    : (recentKept.length > 0 ? recentKept[0] : currentAcceptedTokPerSec(state));
  const endCandidates = [...recentKept];
  const current = currentAcceptedTokPerSec(state);
  if (current > 0) endCandidates.push(current);
  const end = endCandidates.length > 0 ? Math.max(start, ...endCandidates) : start;
  const delta = end - start;
  const threshold = smallAcceptedProgressBand(start > 0 ? start : end);
  return { start, end, delta, threshold, hasProgress: delta >= threshold };
}

function highestMatchingCycle(
  cycles: CycleResult[],
  predicate: (cycle: CycleResult) => boolean,
): CycleResult | null {
  let best: CycleResult | null = null;
  for (const cycle of cycles) {
    if (!predicate(cycle) || cycle.tokPerSec == null) continue;
    if (best == null || (cycle.tokPerSec ?? 0) > (best.tokPerSec ?? 0)) {
      best = cycle;
    }
  }
  return best;
}

function cycleSummary(cycle: CycleResult): string {
  const rate = cycle.tokPerSec == null ? "" : ` (${cycle.tokPerSec.toFixed(1)} tok/s)`;
  return `cycle ${cycle.cycle}: ${promptTrunc(cycle.description, 118)}${rate}`;
}

type ProfileEntry = { name: string; value: number };

function profileSection(line: string, marker: string): string | null {
  const idx = line.toLowerCase().indexOf(marker.toLowerCase());
  if (idx < 0) return null;
  return line.slice(idx + marker.length);
}

function parseGibEntries(section: string): ProfileEntry[] {
  const entries: ProfileEntry[] = [];
  const re = /\b([A-Za-z][A-Za-z0-9_-]*)\s+([0-9]+(?:\.[0-9]+)?)\s+GiB\b/g;
  for (const match of section.matchAll(re)) {
    entries.push({ name: match[1], value: Number.parseFloat(match[2]) });
  }
  return entries.sort((a, b) => b.value - a.value);
}

function parseNumericEntries(section: string): ProfileEntry[] {
  const entries: ProfileEntry[] = [];
  const re = /\b([A-Za-z][A-Za-z0-9_-]*)\s+([0-9]+(?:\.[0-9]+)?)\b/g;
  for (const match of section.matchAll(re)) {
    entries.push({ name: match[1], value: Number.parseFloat(match[2]) });
  }
  return entries.sort((a, b) => b.value - a.value);
}

function formatProfileEntries(entries: ProfileEntry[], unit: string, max = 5): string {
  return entries
    .slice(0, max)
    .map(e => `${e.name}=${e.value.toFixed(2)}${unit}`)
    .join(", ");
}

function cycleTags(cycle: Pick<CycleResult, "description" | "selfAnalysis" | "nextIdeas">): string[] {
  const text = [
    cycle.description,
    cycle.selfAnalysis,
    ...cycle.nextIdeas,
  ].join("\n").toLowerCase();
  const tags: string[] = [];
  if (/shader|kernel|threadgroup|simd|tg128|metal/.test(text)) tags.push("shader");
  if (/dispatch|command|encoder|barrier|commit|batch|launch/.test(text)) tags.push("dispatch");
  if (/buffer|alloc|pool|reuse|memory|private|repack/.test(text)) tags.push("memory");
  if (/fuse|fusion|merged|combined/.test(text)) tags.push("fusion");
  if (/moe|expert|router|topk|route.?pack|shared.?gate/.test(text)) tags.push("moe");
  if (/attention|flash|kv.?cache|rope/.test(text)) tags.push("attention");
  if (/ssm|delta|conv|gated.?norm|q8/.test(text)) tags.push("ssm");
  if (/half|float16|bfloat|bf16|f16/.test(text)) tags.push("precision");
  if (tags.length === 0) tags.push("other");
  return tags;
}

function categoryCounts(cycles: CycleResult[]): Array<[string, { kept: number; reverted: number }]> {
  const categories: Record<string, { kept: number; reverted: number }> = {};
  for (const c of cycles) {
    for (const tag of cycleTags(c)) {
      if (!categories[tag]) categories[tag] = { kept: 0, reverted: 0 };
      if (c.kept) categories[tag].kept++;
      else categories[tag].reverted++;
    }
  }
  return Object.entries(categories).sort((a, b) => {
    const aTotal = a[1].kept + a[1].reverted;
    const bTotal = b[1].kept + b[1].reverted;
    return bTotal - aTotal || b[1].kept - a[1].kept;
  });
}

function isRoutePackOrSharedGateWork(cycle: CycleResult): boolean {
  const text = [
    cycle.description,
    cycle.selfAnalysis,
    cycle.outputText,
    ...cycle.nextIdeas,
  ].join("\n").toLowerCase();
  return /route.?pack|shared.?gate|f32 shared|materialized.*gate|validator.*logit/.test(text);
}

function recentRoutePackCooldown(state: RunState): { active: boolean; count: number; window: number } {
  const window = 8;
  const recent = state.cycles.slice(-window);
  const count = recent.filter(c => !c.kept && isRoutePackOrSharedGateWork(c)).length;
  return { active: count >= 2, count, window };
}

function bestKeptCorrectCycle(state: Pick<RunState, "cycles" | "workload">): CycleResult | null {
  let best: CycleResult | null = null;
  for (const cycle of state.cycles) {
    if (!cycleMatchesWorkload(state, cycle)) continue;
    if (!cycle.kept || !cycle.containsReference || cycle.tokPerSec == null) continue;
    if (best == null || cycle.tokPerSec > (best.tokPerSec ?? 0)) {
      best = cycle;
    }
  }
  return best;
}

export function bestTreeCandidateCycle(state: Pick<RunState, "cycles" | "bestTokPerSec" | "workload">): CycleResult | null {
  let bestTokPerSec = Number.isFinite(state.bestTokPerSec) ? state.bestTokPerSec : 0;
  for (const cycle of state.cycles) {
    if (!cycleMatchesWorkload(state, cycle)) continue;
    if (!cycle.kept || !cycle.containsReference || cycle.tokPerSec == null || cycle.tokPerSec <= 0) continue;
    bestTokPerSec = Math.max(bestTokPerSec, cycle.tokPerSec);
  }

  let candidate: CycleResult | null = null;
  for (const cycle of state.cycles) {
    if (!cycleMatchesWorkload(state, cycle)) continue;
    if (!cycle.kept || !cycle.containsReference || cycle.tokPerSec == null || cycle.tokPerSec <= 0) continue;
    if (cycle.tokPerSec < bestTokPerSec - 0.05) continue;
    if (candidate == null || cycle.cycle > candidate.cycle) {
      candidate = cycle;
    }
  }
  return candidate;
}

function consecutiveReverts(cycles: CycleResult[]): number {
  let count = 0;
  for (let i = cycles.length - 1; i >= 0; i--) {
    if (cycles[i].kept) break;
    count++;
  }
  return count;
}

export type PlateauStopDecision = {
  stop: boolean;
  reason: string;
  consecutiveReverts: number;
  cyclesSinceBest: number;
  bestCycle: number | null;
};

export function detectAutoStopForPlateau(
  state: Pick<RunState, "cycles" | "workload">,
  opts: { revertStreak?: number; noBestCycles?: number } = {},
): PlateauStopDecision {
  const revertStreak = opts.revertStreak ?? AUTO_STOP_REVERT_STREAK;
  const noBestCycles = opts.noBestCycles ?? AUTO_STOP_NO_BEST_CYCLES;
  const best = bestKeptCorrectCycle(state);
  const last = state.cycles[state.cycles.length - 1];
  const revertCount = consecutiveReverts(state.cycles);
  const cyclesSinceBest = best && last ? Math.max(0, last.cycle - best.cycle) : 0;

  if (revertCount >= revertStreak) {
    return {
      stop: true,
      reason: `${revertCount} consecutive reverted cycles`,
      consecutiveReverts: revertCount,
      cyclesSinceBest,
      bestCycle: best?.cycle ?? null,
    };
  }

  if (best && cyclesSinceBest >= noBestCycles) {
    return {
      stop: true,
      reason: `${cyclesSinceBest} cycles since promoted-best cycle ${best.cycle}`,
      consecutiveReverts: revertCount,
      cyclesSinceBest,
      bestCycle: best.cycle,
    };
  }

  return {
    stop: false,
    reason: "",
    consecutiveReverts: revertCount,
    cyclesSinceBest,
    bestCycle: best?.cycle ?? null,
  };
}

export function classifyAttemptFamilies(cycle: CycleResult): string[] {
  const text = [
    cycle.description,
    cycle.selfAnalysis,
    cycle.outputText,
    ...cycle.nextIdeas,
  ].join("\n").toLowerCase();
  const families: string[] = [];
  if (/simd_sum|shuffle|lane[- ]parallel|float[248]|writeback/.test(text)) families.push("simd_sum/reduction packing");
  if (/q8|repack|fixed[- ]?k|tg128|k=2048|k=4096/.test(text)) families.push("Q8/repacked shape retune");
  if (/(q4[_-]?k|geglu|gate[\/ -]?up).*(tail|route[- ]?pack|active[- ]?block|input[- ]row|singleton|exact [1-7][ -]?route|route\s*\/\s*8|route\s*>>\s*3)|(?:tail|route[- ]?pack|active[- ]?block|input[- ]row).*(q4[_-]?k|geglu|gate[\/ -]?up)/.test(text)) families.push("Q4_K route-tail/GeGLU micro-variants");
  if (/(q5[_-]?(?:1|k)|moe[- ]?down|expert[- ]?down|down).*(tail|route[- ]?pack|active[- ]?block|exact [1-7][ -]?route)|(?:tail|route[- ]?pack|active[- ]?block).*(q5[_-]?(?:1|k)|moe[- ]?down|expert[- ]?down)/.test(text)) families.push("Q5 down/tail variants");
  if (/indirect.*dispatch|dispatch.*indirect|active[- ]?block.*(metadata|route[- ]?count|counts)|route[- ]?count|count metadata|tail[- ]?shape|tail histogram|occupancy stats|tiled expert scans/.test(text)) families.push("active-block metadata/indirect dispatch");
  if (/router|topk|top-?k|shared[- ]?gate|route[- ]?pack|moe[- ]?route/.test(text)) families.push("router/shared-gate");
  if (/ssm|delta|conv1?d|gated[-_ ]?norm/.test(text)) families.push("SSM recurrent/projection");
  if (/barrier|encoder|command|commit|wait|dispatch|batch/.test(text)) families.push("command/barrier scheduling");
  if (/lm[- ]?head|argmax|logits/.test(text)) families.push("final/logits tail");
  if (families.length === 0) families.push("uncategorized");
  return [...new Set(families)];
}

export function buildFamilyCooldownDirective(
  state: Pick<RunState, "cycles">,
  windowSize: number = FAMILY_COOLDOWN_WINDOW,
  threshold: number = FAMILY_COOLDOWN_THRESHOLD,
): string[] {
  const recent = state.cycles.slice(-windowSize);
  if (recent.length < threshold) return [];

  const stats = new Map<string, { kept: number; reverted: number }>();
  for (const cycle of recent) {
    for (const family of classifyAttemptFamilies(cycle)) {
      const entry = stats.get(family) ?? { kept: 0, reverted: 0 };
      if (cycle.kept) entry.kept++;
      else entry.reverted++;
      stats.set(family, entry);
    }
  }

  const active = [...stats.entries()]
    .filter(([, s]) => s.reverted >= threshold && s.kept === 0)
    .sort((a, b) => b[1].reverted - a[1].reverted);
  if (active.length === 0) return [];

  const lines: string[] = [];
  lines.push(`## ⚠ FAMILY COOLDOWN — repeated reverted variants need fresh evidence`);
  lines.push(
    `In the last ${recent.length} cycles, these families reverted repeatedly with no kept win: ` +
      active.map(([family, s]) => `${family}=${s.reverted} reverted/0 kept`).join(", ") + ".",
  );
  lines.push("- Do not try another member of a cooled-down family unless the cycle first adds or cites fresh profile/microbench/validator evidence naming the exact shader, shape, or barrier bucket.");
  lines.push("- A valid next cycle can be `@@@STEP_KIND: analysis` or `@@@STEP_KIND: enablement` if it builds the missing evidence; a same-family optimization without new evidence should be reverted by the harness.");
  return lines;
}

function hasFreshQuantifiedEvidence(text: string): boolean {
  const namesEvidence = /\b(candidate[_-]?(profile|metal[_-]?shapes)|metal-shapes|bench-metal-shapes|microbench|profile|validator|exact-shape)\b/i.test(text);
  const hasNumber = /(?:[+-]?\d+(?:\.\d+)?\s*(?:tok\/s|ms|gb\/s|%))|(?:\b\d+(?:\.\d+)?\s*x\b)/i.test(text);
  return namesEvidence && hasNumber;
}

export function cooledFamilyRejectionReason(args: {
  state: Pick<RunState, "cycles">;
  stepKind: StepKind;
  description: string;
  selfAnalysis: string;
  ideas: string[];
  windowSize?: number;
  threshold?: number;
}): string | null {
  if (!HARD_FAMILY_COOLDOWN) return null;
  if (args.stepKind !== "optimization") return null;

  const recent = args.state.cycles.slice(-(args.windowSize ?? FAMILY_COOLDOWN_WINDOW));
  if (recent.length < (args.threshold ?? FAMILY_COOLDOWN_THRESHOLD)) return null;

  const stats = new Map<string, { kept: number; reverted: number }>();
  for (const cycle of recent) {
    for (const family of classifyAttemptFamilies(cycle)) {
      const entry = stats.get(family) ?? { kept: 0, reverted: 0 };
      if (cycle.kept) entry.kept++;
      else entry.reverted++;
      stats.set(family, entry);
    }
  }
  const activeFamilies = new Set(
    [...stats.entries()]
      .filter(([, s]) => s.reverted >= (args.threshold ?? FAMILY_COOLDOWN_THRESHOLD) && s.kept === 0)
      .map(([family]) => family),
  );
  if (activeFamilies.size === 0) return null;

  const text = [
    args.description,
    args.selfAnalysis,
    ...args.ideas,
  ].join("\n");
  if (hasFreshQuantifiedEvidence(text)) return null;

  const attemptedFamilies = classifyAttemptFamilies({
    cycle: 0,
    timestamp: "",
    phase: "optimize",
    description: args.description,
    kept: false,
    tokPerSec: null,
    tokensGenerated: 0,
    containsReference: true,
    buildExitCode: 0,
    testExitCode: 0,
    runExitCode: 0,
    outputText: "",
    selfAnalysis: args.selfAnalysis,
    nextIdeas: args.ideas,
  });
  const blocked = attemptedFamilies.filter((family) => activeFamilies.has(family));
  if (blocked.length === 0) return null;
  return `cooled-down family without fresh quantified evidence: ${blocked.join(", ")}`;
}

export function buildStructuralPivotDirective(state: RunState): string[] {
  if (state.cycles.length === 0) return [];
  const stopLike = detectAutoStopForPlateau(state, {
    revertStreak: STRUCTURAL_PIVOT_REVERT_STREAK,
    noBestCycles: STRUCTURAL_PIVOT_NO_BEST_CYCLES,
  });
  const stallThreshold = isGemmaRun(state) && (state.metricMode ?? METRIC_MODE) === "prefill"
    ? GEMMA_PLATEAU_STALL_CYCLES
    : QWEN36_PLATEAU_STALL_CYCLES;
  if (!stopLike.stop && state.stalledCycles < stallThreshold) return [];

  const best = bestKeptCorrectCycle(state);
  const current = currentAcceptedTokPerSec(state);
  const lines: string[] = [];
  lines.push("## ⚠ HARD PIVOT MODE — local retunes are exhausted");
  if (best) {
    lines.push(
      `Best kept result is cycle ${best.cycle} at ${(best.tokPerSec ?? 0).toFixed(2)} ${METRIC_LABEL}; current accepted tree is ${current.toFixed(2)} ${METRIC_LABEL}; ${stopLike.cyclesSinceBest} cycles have passed without a new promoted best.`,
    );
  }
  if (stopLike.consecutiveReverts > 0) {
    lines.push(`Current tail: ${stopLike.consecutiveReverts} consecutive reverted cycles.`);
  }
  if (isQwen36LargeMoeRun(state)) {
    lines.push("- For Qwen3.6 35B, stop spending cycles on another `simd_sum`, lane-writeback, threadgroup-size, or narrow Q8 sibling variant unless a fresh profile names that exact kernel as the top remaining cost.");
    lines.push("- Pick one structural lever: MoE finalizer/barrier fusion, per-token command-buffer batching, expert dispatch/buffer fusion, or a measurement foundation that directly quantifies one of those buckets.");
  } else if (isGemmaRun(state) && (state.metricMode ?? METRIC_MODE) === "prefill") {
    if (gemmaPrefillActualPathIsQueuedOffPath(state.lastProfileOutput)) {
      lines.push("- For Gemma26 M4 prefill, the latest profile says `queued-token-major default_batched=yes structural_batched=no route_layers=0`; full route-pack is off-path.");
      lines.push("- Productive directions are the exact `structural_batched=no` guard blocker, queued-prefill chunk/wait scheduling, or a measurement counter that names why the structural path is disabled.");
    } else if (gemmaPrefillActualPathIsStructuralRoutePack(state.lastProfileOutput)) {
      lines.push("- For Gemma26 M4 prefill, the structural route-packed path is live (`structural_batched=yes`, nonzero `route_layers`). Stop auditing old structural guards or queued-token-major scheduling unless the profile falls back.");
      if (isGemmaPrefillPostRoutePackPlateau(state)) {
        lines.push("- The post-383 route-pack/tail/metadata family is exhausted. Do not spend another cycle on Q4_K/Q5_1 route-tail kernels, active-block route-count metadata, indirect dispatch, or passive tail histograms without new evidence naming an exact >1 tok/s bucket.");
        lines.push("- Productive directions are Q8 LM-head/shared/attention exact-shape work, gpu-moe finalizer/barrier-count reduction, or public-suite validation to select the next bottleneck.");
      } else {
        lines.push("- Productive directions are now on-path route-pack occupancy, q5_1/q4_k active-block tail kernels with fresh evidence, LM-head/Q8 hot shapes, or public-suite validation.");
      }
    } else {
      lines.push("- For Gemma26 M4 prefill, stop spending cycles on another weighted-finalizer or narrow Q8 threadgroup retune unless the latest profile or `bench-metal-shapes` evidence names that exact shape as underperforming.");
      lines.push("- Productive directions so far are queued-prefill schedule changes and router/RMS fusion. Prefer consuming exact-shape evidence, auditing full batched-prefill guard blockers, or validating public-suite prompt lengths.");
    }
  } else {
    lines.push("- Stop spending cycles on local arithmetic cleanup. Pick a scheduler/fusion/measurement change that can alter a named hot bucket, or explicitly mark the step as analysis.");
  }
  lines.push("- If the next change is not expected to move at least the promotion band, label it `@@@STEP_KIND: analysis` or `@@@STEP_KIND: enablement` and state the exact speed path it unlocks.");
  return lines;
}

function isGemmaPrefillPostBreakthrough(state: RunState): boolean {
  return isGemmaRun(state) &&
    (state.metricMode ?? METRIC_MODE) === "prefill" &&
    bestKeptCorrectTokPerSec(state) >= GEMMA_POST_BREAKTHROUGH_FLOOR;
}

function cyclesSinceBestKept(state: RunState): number {
  const best = bestKeptCorrectCycle(state);
  const last = state.cycles[state.cycles.length - 1];
  if (!best || !last) return 0;
  return Math.max(0, last.cycle - best.cycle);
}

function isGemmaPrefillPostRoutePackPlateau(state: RunState): boolean {
  return isGemmaPrefillPostBreakthrough(state) &&
    bestKeptCorrectTokPerSec(state) >= GEMMA_POST_ROUTEPACK_PLATEAU_FLOOR &&
    (state.stalledCycles >= GEMMA_PLATEAU_STALL_CYCLES ||
      cyclesSinceBestKept(state) >= GEMMA_PLATEAU_STALL_CYCLES ||
      consecutiveReverts(state.cycles) >= STRUCTURAL_PIVOT_REVERT_STREAK);
}

function latestGemmaPrefillActualPathLine(profile: string | null | undefined): string | null {
  if (!profile) return null;
  const matches = profile.split("\n").filter(line => /prefill actual path:/i.test(line));
  return matches.length > 0 ? matches[matches.length - 1].trim() : null;
}

export function inferGemmaRouteTokensForMetalShapes(profile: string | null | undefined): number | null {
  if (!profile) return null;

  const promptTokens = /\bprompt_tokens\s+(\d+)\b/i.exec(profile);
  if (promptTokens) {
    const parsed = Number.parseInt(promptTokens[1], 10);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }

  const avgSlots = /\bavg_slots\/layer\s+([0-9]+(?:\.[0-9]+)?)\b/i.exec(profile);
  if (avgSlots) {
    const parsed = Number.parseFloat(avgSlots[1]);
    if (Number.isFinite(parsed) && parsed > 0) {
      const routeTokens = Math.round(parsed / 8);
      if (routeTokens > 0) return routeTokens;
    }
  }

  const slots = /\blayers\s+(\d+)\s+slots\s+(\d+)\b/i.exec(profile);
  if (slots) {
    const layers = Number.parseInt(slots[1], 10);
    const totalSlots = Number.parseInt(slots[2], 10);
    if (Number.isFinite(layers) && layers > 0 && Number.isFinite(totalSlots) && totalSlots > 0) {
      const routeTokens = Math.round(totalSlots / layers / 8);
      if (routeTokens > 0) return routeTokens;
    }
  }

  return null;
}

export function gemmaPrefillActualPathIsQueuedOffPath(profile: string | null | undefined): boolean {
  const line = latestGemmaPrefillActualPathLine(profile);
  if (!line) return false;
  return /queued-token-major/i.test(line) &&
    /structural_batched=no/i.test(line) &&
    /route_layers=0/i.test(line);
}

export function gemmaPrefillActualPathIsStructuralRoutePack(profile: string | null | undefined): boolean {
  const line = latestGemmaPrefillActualPathLine(profile);
  if (!line) return false;
  const layers = /route_layers=(\d+)/i.exec(line);
  return /batched-route-pack/i.test(line) &&
    /structural_batched=yes/i.test(line) &&
    layers != null &&
    Number.parseInt(layers[1], 10) > 0;
}

export function buildGemmaPrefillPostBreakthroughAnalysis(state: RunState): string[] {
  if (!isGemmaPrefillPostBreakthrough(state)) return [];

  const best = bestKeptCorrectTokPerSec(state);
  const current = currentAcceptedTokPerSec(state);
  const cyclesSinceBest = cyclesSinceBestKept(state);
  const recent = state.cycles.slice(-GEMMA_PLATEAU_WINDOW);
  const recentKept = recent.filter(c => c.kept && c.containsReference);
  const recentReverted = recent.filter(c => !c.kept);
  const q8Retunes = recent.filter(c =>
    /q8|threadgroup|tg128|tg64|paired|repack|finalizer|sigmoid|fast::exp/i.test(`${c.description}\n${c.selfAnalysis}`)
  );
  const scheduleWins = recent.filter(c =>
    c.kept && /queued prefill|split|schedule|\[[0-9, ]+\]|tail/i.test(c.description)
  );
  const profile = state.lastProfileOutput ?? "";
  const actualPathLine = latestGemmaPrefillActualPathLine(profile);
  const routePackOffPath = gemmaPrefillActualPathIsQueuedOffPath(profile);
  const routePackStructural = gemmaPrefillActualPathIsStructuralRoutePack(profile);
  const postRoutePackPlateau = isGemmaPrefillPostRoutePackPlateau(state);
  const pathLine = profile.split("\n").find(line => /path bytes:/i.test(line));
  const prefillMoeLine = profile.split("\n").find(line => /prefill buckets: moe/i.test(line));
  const queuedLine = profile.split("\n").find(line => /prefill queued prefill:/i.test(line));
  const queueLine = profile.split("\n").find(line => /prefill queued prefill queue:/i.test(line));
  const routePackLine = profile.split("\n").find(line => /prefill route pack:/i.test(line));
  const routePackActualLine = profile.split("\n").find(line => /prefill route pack actual:/i.test(line));
  const routePackOccupancyLine = profile.split("\n").find(line => /prefill route pack occupancy:/i.test(line));
  const q8Hot = profile.split("\n").filter(line => /q8 hot #/i.test(line)).slice(0, 4);
  const evidence = state.lastMetalShapesOutput ?? "";
  const targetGap = TARGET_TOK_PER_SEC - best;

  const lines = [
    "## Gemma26 M4 Prefill Post-80 Focus",
    targetGap > 0
      ? `- Accepted best is ${best.toFixed(1)} prefill tok/s; current tree is ${current.toFixed(1)}; target is ${TARGET_TOK_PER_SEC.toFixed(1)}. Remaining gap is ${targetGap.toFixed(1)} tok/s (${((targetGap / best) * 100).toFixed(1)}%).`
      : `- Accepted best is ${best.toFixed(1)} prefill tok/s; current tree is ${current.toFixed(1)}; configured target ${TARGET_TOK_PER_SEC.toFixed(1)} is already beaten by ${(-targetGap).toFixed(1)} tok/s. Use public-suite/llama.cpp parity and the latest profile as the next bar.`,
    `- Recent window: ${recentKept.length}/${recent.length} kept, ${recentReverted.length} reverted, ${cyclesSinceBest} cycles since best. Do not optimize from pre-80 or cycle-49 assumptions.`,
  ];

  if (actualPathLine) lines.push(`- Latest actual prefill path: ${actualPathLine}`);
  if (pathLine) lines.push(`- Latest path mix: ${pathLine.trim()}`);
  if (prefillMoeLine) lines.push(`- Latest MoE/shared buckets: ${prefillMoeLine.trim()}`);
  if (queuedLine) lines.push(`- Latest queued schedule: ${queuedLine.trim()}`);
  if (queueLine) lines.push(`- Latest queued waits: ${queueLine.trim()}`);
  if (routePackLine) lines.push(`- Latest route-pack shape: ${routePackLine.trim()}`);
  if (routePackActualLine) lines.push(`- Latest active-block count: ${routePackActualLine.trim()}`);
  if (routePackOccupancyLine) lines.push(`- Latest route-pack occupancy: ${routePackOccupancyLine.trim()}`);
  if (q8Hot.length > 0) {
    lines.push(`- Latest Q8 hot shapes: ${q8Hot.map(line => line.trim().replace(/^info\(forward\):\s*/, "")).join(" | ")}`);
  }
  if (scheduleWins.length > 0) {
    const topSchedule = [...scheduleWins]
      .sort((a, b) => (b.tokPerSec ?? 0) - (a.tokPerSec ?? 0))[0];
    lines.push(`- Banked schedule win: ${cycleSummary(topSchedule)}. A new split must beat this, not merely recover an older slower split.`);
  }
  if (q8Retunes.length >= 5) {
    lines.push(`- Cooldown: ${q8Retunes.length}/${recent.length} recent cycles touched Q8/finalizer/threadgroup-style retunes. The next such optimization needs exact-shape microbench or profile evidence showing the candidate can save at least 1 tok/s.`);
  }
  if (routePackOffPath) {
    lines.push("- HARD PATH FACT: full batched route-pack is not executing in the production prefill profile (`structural_batched=no`, `route_layers=0`). Active-block/route-pack/gather kernel retunes are off-path until the outer profile flips to `structural_batched=yes` with nonzero route layers.");
    lines.push("- Valid next speed paths: emit the exact guard reason for `structural_batched=no` and fix it, or optimize the queued-token-major schedule/wait path that is actually running.");
  } else if (routePackStructural) {
    lines.push("- HARD PATH FACT: structural Gemma route-pack is live. The old `structural_batched=no` blocker is solved; do not spend another cycle on guard-audit, queued-token-major scheduling, or default-enable plumbing unless the latest profile falls back.");
    if (postRoutePackPlateau) {
      lines.push("- POST-383 PLATEAU FACT: the useful route-pack work has already been harvested. The post-best tail was dominated by reverted Q4_K GeGLU/gate-up tail variants, Q5 down/tail variants, active-block metadata/count rewrites, indirect dispatch, and passive tail histograms.");
      lines.push("- Required pivot: do not reattempt Q4_K/Q5_1 route-tail or active-block metadata/indirect-dispatch work unless fresh profile or Metal-shapes evidence names an exact non-dead bucket and a >1 tok/s budget. Prefer Q8 LM-head/shared/attention exact-shape work, gpu-moe finalizer barrier reduction, or public-suite validation.");
    } else {
      lines.push("- Current on-path waste is route-pack occupancy/tails and hot Q8/LM-head traffic. Any tail-kernel change must cite the exact tail size and avoid known regressions: q5_1 exact-6, q4_k exact-4, and alt4 block-width/profile edits.");
    }
  }
  if (evidence.trim().length > 0 && state.lastMetalShapesOk === false) {
    lines.push(`- Latest Metal-shapes evidence run failed at cycle ${state.lastMetalShapesCycle ?? "?"}. Fix that evidence path or choose a non-shape retune; do not keep adding benchmark-output changes that the harness cannot capture.`);
  } else if (evidence.trim().length > 0) {
    if (routePackOffPath) {
      lines.push(`- Latest Metal-shapes evidence is available from cycle ${state.lastMetalShapesCycle ?? "?"}, but treat route-pack/active-block cases as off-path production evidence until the actual path changes.`);
    } else {
      lines.push(`- Latest Metal-shapes evidence is available from cycle ${state.lastMetalShapesCycle ?? "?"}. Consume it before adding another microbench or shader retune.`);
    }
  } else {
    lines.push("- No Metal-shapes evidence has been captured by the outer harness yet. If targeting shared Q8, first enable `ZINC_METAL_SHAPES_EVERY` or add a default-off exact-shape case and let the harness run it.");
  }
  if (state.stalledCycles >= GEMMA_PLATEAU_STALL_CYCLES || cyclesSinceBest >= GEMMA_PLATEAU_STALL_CYCLES) {
    lines.push("- PLATEAU RULE: neutral `optimization` keeps are churn here. A speed-neutral cycle should be `analysis`/`enablement` and produce evidence the next cycle can consume.");
  }
  if (routePackOffPath) {
    lines.push(
      "- Best next moves: audit the exact guard blocker for `structural_batched=no`; fix queued-token-major chunk sizing/wait behavior; or add a profile counter that names the guard without changing kernels.",
      "- Avoid: active-block/route-pack/gather/barrier retunes, weighted-finalizer sigmoid/cache/threadgroup micro-retunes, and broad Q8 repacks unless the production profile shows that path is actually executing.",
      "",
    );
  } else if (routePackStructural) {
    if (postRoutePackPlateau) {
      lines.push(
        "- Best next moves: Q8 LM-head/shared/attention exact-shape measurement or optimization; gpu-moe finalizer/barrier-count reduction backed by the profile bucket; or full public-suite validation before narrowing the next code target.",
        "- Avoid: Q4_K/Q5_1 route-tail micro-specializations, route-count metadata rewrites, active-block indirect dispatch, passive route-tail histograms, old structural guard work, and queued-token-major schedule changes.",
        "",
      );
    } else {
      lines.push(
        "- Best next moves: measure/optimize the active `batched-route-pack` path only; focus on route-pack occupancy, q5_1/q4_k tail sizes not already reverted, LM-head/Q8 hot shapes, or public-suite validation.",
        "- Avoid: old structural guard work, queued-token-major schedule changes, q5_1 exact-6 tails, q4_k exact-4 tails, alt4 block-width changes, and broad finalizer/barrier retunes without a named profile bucket.",
        "",
      );
    }
  } else {
    lines.push(
      "- Best next moves: consume `gemma26_prefill_hot` evidence; audit why the full Gemma batched-prefill path is still not the public-suite default; or test public-suite prompt lengths before another exact-20 schedule tweak.",
      "- Avoid: weighted-finalizer sigmoid/cache/threadgroup micro-retunes, broad Q8 repacks, and route-pack semantic changes without validation-on logits parity.",
      "",
    );
  }

  return lines;
}

export function buildQwen36PrefillPlateauAnalysis(state: RunState): string[] {
  if (!isQwen36PrefillRun(state)) return [];
  if (state.stalledCycles < QWEN36_PLATEAU_STALL_CYCLES) return [];

  const recent = state.cycles.slice(-QWEN36_PLATEAU_WINDOW);
  if (recent.length === 0) return [];

  const keptCorrect = state.cycles.filter(c => c.kept && c.containsReference && c.tokPerSec != null);
  const recentKeptCorrect = recent.filter(c => c.kept && c.containsReference && c.tokPerSec != null);
  const overallBest = keptCorrect.length > 0
    ? Math.max(...keptCorrect.map(c => c.tokPerSec!))
    : bestKeptCorrectTokPerSec(state);
  const recentBest = recentKeptCorrect.length > 0
    ? Math.max(...recentKeptCorrect.map(c => c.tokPerSec!))
    : 0;
  const current = currentAcceptedTokPerSec(state);
  const topRecent = [...recentKeptCorrect]
    .sort((a, b) => (b.tokPerSec ?? 0) - (a.tokPerSec ?? 0))
    .slice(0, 3);
  const q8Retunes = recent.filter(c =>
    /q8|repack|fixed.?k|k=2048|k=4096|tg128|threadgroup/i.test(`${c.description}\n${c.selfAnalysis}`)
  );
  const neutralKept = recent.filter(c =>
    c.kept &&
    c.containsReference &&
    c.tokPerSec != null &&
    overallBest > 0 &&
    c.tokPerSec <= overallBest + 0.05
  );
  const categoryLine = categoryCounts(recent)
    .slice(0, 5)
    .map(([tag, stats]) => `${tag}=${stats.kept} kept/${stats.reverted} reverted`)
    .join(", ");

  const lines = [
    "## Qwen3.6 35B Prefill Plateau Analysis",
    `- PLATEAU MODE: ${state.stalledCycles} cycles without promoted-best improvement. Last ${recent.length} cycles: ${recent.filter(c => c.kept).length} kept, ${recent.filter(c => !c.kept).length} reverted; recent best ${recentBest.toFixed(1)} ${METRIC_LABEL}, overall best ${overallBest.toFixed(1)}, current tree ${current.toFixed(1)}.`,
  ];

  if (categoryLine) {
    lines.push(`- Recent work mix: ${categoryLine}.`);
  }
  if (q8Retunes.length >= 4) {
    lines.push(`- Cooldown: ${q8Retunes.length}/${recent.length} recent cycles touched Q8/repacked/fixed-K/TG128-style retunes. Do not spend the next cycle on another narrow shape retune unless the latest profile names that exact kernel as the top remaining cost and the change is expected to move at least 0.5 tok/s.`);
  }
  if (neutralKept.length >= Math.max(8, Math.floor(recent.length * 0.5))) {
    lines.push(`- Neutral keeps are dominating (${neutralKept.length}/${recent.length} recent cycles). In plateau mode, a correct optimization that is merely within noise is churn; prefer a structural scheduler/validator change, or make the change beat current accepted throughput by the small-progress band.`);
  }
  if (topRecent.length > 0) {
    lines.push("- Best recent evidence:");
    for (const c of topRecent) {
      lines.push(`  - ${cycleSummary(c)}`);
    }
  }
  lines.push(
    "- Required pivot: attack a larger remaining boundary (command scheduling, SSM/MoE phase fusion, route-pack correctness diff with layer/tensor max error, or an all-model coherence guard) instead of another local arithmetic cleanup.",
    "- If the next step is neutral on speed, mark it `@@@STEP_KIND: enablement` or `@@@STEP_KIND: analysis` and name the exact follow-up speed path it unlocks; unlabeled neutral optimization steps are eligible for automatic revert in plateau mode.",
    "",
  );

  return lines;
}

export function buildQwen36PrefillPostBreakthroughAnalysis(state: RunState): string[] {
  if (!isQwen36PrefillRun(state)) return [];

  const best = bestKeptCorrectTokPerSec(state);
  if (best < 60) return [];

  const breakthrough = highestMatchingCycle(
    state.cycles,
    (c) => c.kept && c.containsReference && (c.tokPerSec ?? 0) >= 60 &&
      /router_f32_topk_batched|f32 router|top-?k|topk/i.test(c.description),
  ) ?? highestMatchingCycle(
    state.cycles,
    (c) => c.kept && c.containsReference && (c.tokPerSec ?? 0) >= 60,
  );

  const profile = state.lastProfileOutput ?? "";
  const pathLine = profile.split("\n").find(line => /path bytes:/i.test(line));
  const pathEntries = pathLine
    ? parseGibEntries(profileSection(pathLine, "path bytes:") ?? pathLine)
    : [];
  const barrierLine = profile.split("\n").find(line => /barriers\/step:/i.test(line));
  const barrierEntries = barrierLine
    ? parseNumericEntries(profileSection(barrierLine, "barriers/step:") ?? barrierLine)
    : [];
  const ssmBucketLine = profile.split("\n").find(line => /prefill buckets: ssm/i.test(line));

  const recent = state.cycles.slice(-12);
  const q8Retunes = recent.filter(c =>
    /q8|repack|fixed.?k|k=2048|k=4096|tg128|threadgroup/i.test(`${c.description}\n${c.selfAnalysis}`)
  );

  const lines = [
    "## Qwen3.6 35B Post-60 Prefill Jump Focus",
  ];

  if (breakthrough) {
    lines.push(`- Banked breakthrough: ${cycleSummary(breakthrough)}. Treat this as the new floor; do not optimize from pre-cycle-231 assumptions.`);
  } else {
    lines.push(`- Banked breakthrough: accepted best is ${best.toFixed(1)} prefill tok/s. Treat this as the new floor; do not optimize from pre-60 assumptions.`);
  }

  if (pathEntries.length > 0) {
    const top = pathEntries[0];
    lines.push(`- Latest profile dominant path bytes: ${formatProfileEntries(pathEntries, " GiB")}. The next change should target \`${top.name}\` unless a fresh profile proves another bucket moved above it.`);
    if (top.name === "ssm") {
      lines.push("- After the fused F32 router/top-k win, SSM is larger than router. Do not make router/top-k the default next target unless the profile moves it back on top.");
    }
  } else {
    lines.push("- No accepted post-breakthrough profile is available. First action in the next run should be a profile-backed analysis or microbench, not a speculative kernel rewrite.");
  }

  if (barrierEntries.length > 0) {
    lines.push(`- Latest barrier pressure: ${formatProfileEntries(barrierEntries, "/step")}. Pick one barrier-heavy bucket and remove real command/barrier work; do not add passive counters.`);
  }
  if (ssmBucketLine) {
    lines.push(`- SSM detail from latest profile: ${ssmBucketLine.trim()}. Use this to split projection work from recurrent conv/delta/gated work before editing.`);
  }
  if (q8Retunes.length >= 4) {
    lines.push(`- Cooldown remains active for narrow Q8/repacked/fixed-K/TG128 retunes (${q8Retunes.length}/${recent.length} recent cycles). A small retune needs exact-shape evidence and same-cycle A/B medians.`);
  }

  lines.push(
    "- Good next swings: SSM projection/branch reuse that preserves short-prompt coherence, SSM recurrent dispatch/barrier reduction, MoE expert launch/buffer fusion, or a prefill-only exact-shape microbench tied to the current top bucket.",
    "- If touching `router_f32_topk_batched`, keep it prefill-only and cite OFF/ON medians; router work should be a narrow follow-up to cycle 231, not another broad route-pack validator pass.",
    "",
  );

  return lines;
}

function buildQwen36PrefillFocus(state: RunState, lastResult: BuildRunResult): string[] {
  if (!isQwen36PrefillRun(state)) return [];

  const best = bestAcceptedTokPerSec(state, lastResult);
  const currentAccepted = Math.max(currentAcceptedTokPerSec(state), correctResultTokPerSec(lastResult));
  const recentProgress = recentAcceptedProgress(state);
  const routeCooldown = recentRoutePackCooldown(state);
  const tokenMajorWin = highestMatchingCycle(
    state.cycles,
    (c) => c.kept && c.containsReference && /token-major|shared-gate|topk_weight_and_reduce/i.test(c.description),
  );
  const earlyCommitWin = highestMatchingCycle(
    state.cycles,
    (c) => c.kept && c.containsReference && /early graph|leading prompt chunk|commit/i.test(c.description),
  );
  const routePackFailures = state.cycles.filter((c) => {
    const text = `${c.description}\n${c.outputText}\n${c.selfAnalysis}`.toLowerCase();
    return !c.kept && !c.containsReference && /route.?pack|f32 shared.?gate|shared-gate/.test(text);
  });
  const dualQ8Failures = state.cycles.filter((c) => {
    const text = `${c.description}\n${c.outputText}\n${c.selfAnalysis}`.toLowerCase();
    return !c.kept && !c.containsReference && /dual.?q8|two-row|qkv\+z|attn_qkv\+attn_gate/.test(text);
  });
  const bangOnlyFailures = state.cycles.filter((c) => !c.kept && /^!+$/.test(c.outputText.trim()));
  const routePackDensityLine = state.lastProfileOutput
    ?.split("\n")
    .find(line => /route-pack candidate blocks/i.test(line));
  const lastKept = [...state.cycles].reverse().find(c => c.kept && c.containsReference);
  const lastKeptWasVisibilityOnly = lastKept != null &&
    /profile|validator|diff|density|visibility|counter/i.test(lastKept.description);

  const lines: string[] = [
    "## Qwen3.6 35B Prefill Target Focus",
  ];

  if (best != null) {
    const gap = QWEN36_PREFILL_INTERMEDIATE_TARGET - best;
    if (gap > 0) {
      const pct = ((gap / best) * 100).toFixed(1);
      lines.push(`- Accepted best is ${best.toFixed(1)} prefill tok/s; the next milestone is ${QWEN36_PREFILL_INTERMEDIATE_TARGET.toFixed(1)} prefill tok/s, a +${gap.toFixed(1)} tok/s (${pct}%) gap.`);
    } else {
      lines.push(`- Accepted best is ${best.toFixed(1)} prefill tok/s; keep pushing past the 50 tok/s milestone with correctness intact.`);
    }
    if (currentAccepted > 0 && Math.abs(currentAccepted - best) > 0.05) {
      lines.push(`- Current accepted tree is measuring ${currentAccepted.toFixed(1)} prefill tok/s; compare new work to this current-tree baseline, not only the promoted-best checkpoint.`);
    }
    if (recentProgress.hasProgress) {
      lines.push(`- Recent accepted movement: ${recentProgress.start.toFixed(1)} → ${recentProgress.end.toFixed(1)} prefill tok/s over the last ${RECENT_PROGRESS_WINDOW} cycles. Treat this as small real progress, not a total stall.`);
    }
  } else {
    lines.push(`- No accepted prefill baseline is known yet; establish one before chasing the ${QWEN36_PREFILL_INTERMEDIATE_TARGET.toFixed(1)} tok/s milestone.`);
  }

  if (tokenMajorWin) {
    lines.push(`- Banked win to build on: ${cycleSummary(tokenMajorWin)}. This is the current productive MoE direction.`);
  }
  if (earlyCommitWin) {
    lines.push(`- Banked dispatch win to refine: ${cycleSummary(earlyCommitWin)}. Try measured chunk-size variants before wider graph rewrites.`);
  }

  const traps: string[] = [];
  if (routePackFailures.length > 0) {
    traps.push(`${routePackFailures.length} route-packed/F32 shared-gate correctness failures`);
  }
  if (dualQ8Failures.length > 0) {
    traps.push(`${dualQ8Failures.length} dual-Q8 SSM correctness failures`);
  }
  if (bangOnlyFailures.length > 0) {
    traps.push(`${bangOnlyFailures.length} bang-only outputs`);
  }
  if (traps.length > 0) {
    lines.push(`- Measured-dead traps: ${traps.join(", ")}. Treat their tok/s as invalid until the output contains Paris.`);
  }
  if (routeCooldown.active) {
    lines.push(`- ROUTE-PACK COOLDOWN: ${routeCooldown.count} route-pack/shared-gate attempts reverted in the last ${routeCooldown.window} cycles. For the next cycle, do not edit route-pack/shared-gate validators, guards, or kernels; choose an SSM, attention, command-buffer, or exact-shape Q8 path instead.`);
  }
  if (routePackDensityLine) {
    if (routeCooldown.active) {
      lines.push(`- Latest route-pack profile: ${routePackDensityLine.trim()}. This remains useful evidence, but the cooldown above takes precedence until a real outer-harness validator log identifies a specific tensor/layer fix.`);
    } else {
      lines.push(`- Latest route-pack profile: ${routePackDensityLine.trim()}. This says the active route-pack path is worth validating; do not add another passive counter before using the existing validator/diff output.`);
    }
  }
  if (lastKeptWasVisibilityOnly) {
    lines.push(`- The last kept cycle added measurement or validation visibility (${cycleSummary(lastKept)}). The next cycle should consume that evidence to promote, fix, or abandon a default-off path.`);
  }

  lines.push(...buildQwen36PrefillPostBreakthroughAnalysis(state));
  lines.push(...buildQwen36PrefillPlateauAnalysis(state));

  lines.push(
    "- Validation gate for risky paths: `ZINC_QWEN36_LAYER0_ROUTE_PACK_PREFILL=1`, F32 shared-gate route-pack, active-block route-pack, and dual-Q8 SSM variants must stay default-off until a full active-prompt validation run keeps `Paris` and beats the accepted median.",
    "- Codex subprocesses in this harness must not run local Metal model commands such as `./zig-out/bin/zinc --model-id qwen36-35b-a3b-q4k-xl` or `ZINC_QWEN36_* ./zig-out/bin/zinc`; they fail with `Metal device not available`. Use `zig build`/`zig build test`; the outer harness owns all Metal measurement and validation runs.",
    "- Next high-leverage moves after 69.9: profile-first SSM projection/recurrent/barrier work, MoE expert launch/buffer fusion, and exact-shape router follow-ups only when the profile still names router/top-k as the blocker.",
    "- If adding a validator or microbench, make it report the exact layer, prompt-token count, max abs diff, and flag-on command so the next cycle can decide whether to promote or abandon it.",
    "",
  );

  return lines;
}

function zigBuildArgs(): string[] {
  return BUILD_OPTIMIZE === "Debug" ? ["build"] : ["build", `-Doptimize=${BUILD_OPTIMIZE}`];
}

// ── Phase detection ──────────────────────────────────────────────────

export type Phase = "fix" | "implement" | "optimize";
export type StepKind = "optimization" | "enablement" | "analysis" | "fix" | "rollback";

export type OutputEvaluation = {
  normalizedText: string;
  containsReference: boolean;
  strongAnswer: boolean;
  outputQualityScore: number;
  offTopic: boolean;
  evaluationNotes: string[];
};

export type BuildRunResult = {
  buildExitCode: number;
  buildOutput: string;
  testExitCode: number;
  testOutput: string;
  runExitCode: number | null;
  runOutput: string;
  phase: Phase;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  promptTokens?: number | null;
  promptTokenSamples?: number[];
  workload?: WorkloadContract | null;
  tokensGenerated: number;
  outputText: string;
  containsReference: boolean;
  strongAnswer: boolean;
  outputQualityScore: number;
  offTopic: boolean;
  evaluationNotes: string[];
  error: string | null;
};

export type ResultSnapshot = {
  cycle: number;
  phase: Phase;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  promptTokens?: number | null;
  promptTokenSamples?: number[];
  workload?: WorkloadContract | null;
  tokensGenerated: number;
  outputText: string;
  containsReference: boolean;
  strongAnswer: boolean;
  outputQualityScore: number;
  offTopic: boolean;
  evaluationNotes: string[];
};

export type ControllerState = {
  lastAccepted: ResultSnapshot | null;
  bestSoFar: ResultSnapshot | null;
  bestCorrect: ResultSnapshot | null;
};

type KeepDecision = {
  keep: boolean;
  improvedBestCorrect: boolean;
  reason: string;
};

export function parseTokPerSec(output: string, mode: MetricMode = "decode"): number | null {
  if (mode === "prefill") {
    const prefillRate = output.match(/Prefill(?:\s+complete)?:\s+\d+\s+tokens\s+in\s+\d+\.?\d*\s*(?:ms|s)\s*\(\s*(\d+\.?\d*)\s*tok\/s\s*\)/i);
    if (prefillRate) return parseFloat(prefillRate[1]);

    const prefillTime = output.match(/Prefill(?:\s+complete)?:\s+(\d+)\s+tokens\s+in\s+(\d+\.?\d*)\s*(ms|s)/i);
    if (prefillTime) {
      const tokens = parseInt(prefillTime[1], 10);
      let seconds = parseFloat(prefillTime[2]);
      if (prefillTime[3].toLowerCase() === "ms") seconds /= 1000;
      if (seconds > 0) return tokens / seconds;
    }

    return null;
  }

  const m = output.match(/Generated\s+(\d+)\s+tokens\s+in\s+(\d+\.?\d*)\s*(ms|s)/i);
  if (m) {
    const tokens = parseInt(m[1], 10);
    let seconds = parseFloat(m[2]);
    if (m[3] === "ms") seconds /= 1000;
    if (seconds > 0) return tokens / seconds;
  }
  const m2 = output.match(/(\d+\.?\d*)\s*tok\/s/i);
  return m2 ? parseFloat(m2[1]) : null;
}

export function parsePrefillTokenCount(output: string): number | null {
  const m = output.match(/Prefill(?:\s+complete)?:\s*(\d+)\s+tokens\s+in\s+\d+\.?\d*\s*(?:ms|s)/i);
  return m ? parseInt(m[1], 10) : null;
}

function parseTokensGenerated(output: string): number {
  const m = output.match(/Generated\s+(\d+)\s+tokens/i);
  return m ? parseInt(m[1], 10) : 0;
}

function parseOutputText(output: string): string {
  const m = output.match(/Output\s*\(\d+\s*tokens?\)\s*:\s*(.+)/i);
  return m ? m[1].trim().slice(0, 200) : "";
}

function normalizeOutputText(text: string): string {
  return text
    .replaceAll("Ġ", " ")
    .replaceAll("Ċ", "\n")
    .replace(/\s+/g, " ")
    .trim();
}

export function evaluateOutputText(text: string, referenceText: string = REFERENCE_TEXT): OutputEvaluation {
  const normalizedText = normalizeOutputText(text);
  const normalizedReference = normalizeOutputText(referenceText);
  const lower = normalizedText.toLowerCase();
  const referenceLower = normalizedReference.toLowerCase();
  const usesParisOracle = referenceLower === "paris";
  const containsReference = referenceLower.length === 0
    ? normalizedText.length > 0
    : lower.includes(referenceLower);
  const contradictoryCapitalTerms = usesParisOracle &&
    [
      "capital of germany",
      "capital of italy",
      "capital of spain",
      "capital of portugal",
      "berlin",
      "rome",
      "madrid",
      "lisbon",
    ].some((pattern) => lower.includes(pattern));
  const offTopic = containsReference && contradictoryCapitalTerms;
  const startsWithReference = referenceLower.length > 0 && lower.startsWith(referenceLower);
  const strongAnswer = containsReference && !offTopic &&
    (usesParisOracle
      ? (/^paris\b/i.test(normalizedText) || /^paris[.!?,\s]/i.test(normalizedText))
      : normalizedText.length >= 8);
  const evaluationNotes: string[] = [];
  if (offTopic) evaluationNotes.push("contains contradictory capital/country terms");
  if (containsReference && startsWithReference) {
    evaluationNotes.push(`starts with ${normalizedReference}`);
  } else if (!usesParisOracle && containsReference && normalizedReference.length > 0) {
    evaluationNotes.push(`contains ${normalizedReference}`);
  }
  const outputQualityScore = strongAnswer ? 4 : containsReference ? 1 : normalizedText ? 0 : 0;
  return {
    normalizedText,
    containsReference,
    strongAnswer,
    outputQualityScore,
    offTopic,
    evaluationNotes,
  };
}

export function detectPhase(result: BuildRunResult): Phase {
  if (result.buildExitCode !== 0) return "fix";
  if (result.testExitCode !== 0) return "fix";
  if (result.runExitCode !== 0 && result.runExitCode !== null) return "fix";
  if (result.error) return "fix";
  if (result.strongAnswer) return "optimize";
  if (result.tokensGenerated > 0) return "implement";
  return "implement";
}

function canonicalizeMemoryEntry(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`"'()[\],.:;!?-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function mergeUniqueEntries(existing: string[], incoming: string[], maxEntries: number): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  for (const entry of [...existing, ...incoming]) {
    const trimmed = entry.trim();
    if (!trimmed) continue;
    const key = canonicalizeMemoryEntry(trimmed)
      .split(" ")
      .filter(Boolean)
      .sort()
      .join(" ");
    if (!key || seen.has(key)) continue;
    seen.add(key);
    merged.push(trimmed);
    if (merged.length >= maxEntries) break;
  }
  return merged;
}

export function snapshotFromResult(cycle: number, result: BuildRunResult): ResultSnapshot {
  return {
    cycle,
    phase: result.phase,
    tokPerSec: result.tokPerSec,
    tokPerSecSamples: result.tokPerSecSamples,
    promptTokens: result.promptTokens ?? null,
    promptTokenSamples: result.promptTokenSamples ?? [],
    workload: result.workload ?? null,
    tokensGenerated: result.tokensGenerated,
    outputText: result.outputText,
    containsReference: result.containsReference,
    strongAnswer: result.strongAnswer,
    outputQualityScore: result.outputQualityScore,
    offTopic: result.offTopic,
    evaluationNotes: result.evaluationNotes,
  };
}

export function decideKeep(
  verify: BuildRunResult,
  baseline: ResultSnapshot,
  state: ControllerState,
): KeepDecision {
  const bestCorrect = state.bestCorrect;
  const baselineTokens = baseline.tokensGenerated ?? 0;

  if (bestCorrect && !verify.strongAnswer) {
    return {
      keep: false,
      improvedBestCorrect: false,
      reason: "lost short-benchmark correctness relative to accepted baseline",
    };
  }

  if (verify.strongAnswer) {
    if (!bestCorrect) {
      return {
        keep: true,
        improvedBestCorrect: true,
        reason: "first strong correct output",
      };
    }
    const bestTokPerSec = bestCorrect.tokPerSec ?? 0;
    const verifyTokPerSec = verify.tokPerSec ?? 0;
    const improvementThreshold = Math.max(0.5, bestTokPerSec * 0.02);
    if (verifyTokPerSec > bestTokPerSec + improvementThreshold) {
      return {
        keep: true,
        improvedBestCorrect: true,
        reason: "significant correct-throughput improvement",
      };
    }
    return {
      keep: false,
      improvedBestCorrect: false,
      reason: "did not beat best correct throughput",
    };
  }

  if (!bestCorrect && verify.tokensGenerated >= baselineTokens + 2) {
    return {
      keep: true,
      improvedBestCorrect: false,
      reason: "pre-correctness token-progress improvement",
    };
  }

  return {
    keep: false,
    improvedBestCorrect: false,
    reason: "no material progress",
  };
}

export function buildReflectionSummary(state: {
  cycles: Array<{
    cycle: number;
    outputText?: string;
    shortOutputText?: string;
    longOutputText?: string;
    offTopic?: boolean;
    evaluationNotes?: string[];
    decisionReason?: string;
    description?: string;
    kept?: boolean;
  }>;
}): string {
  const recentCycles = state.cycles.slice(-20);
  const total = recentCycles.length;
  const germanyDriftCount = recentCycles.filter((cycle) => {
    const text = normalizeOutputText(
      cycle.longOutputText ?? cycle.outputText ?? cycle.shortOutputText ?? "",
    ).toLowerCase();
    return text.includes("paris") && (text.includes("germany") || text.includes("berlin"));
  }).length;
  const failedCount = recentCycles.filter((cycle) => cycle.kept === false).length;

  const lines = [
    `Last 20 cycles: reviewed ${total} cycle${total === 1 ? "" : "s"}, ${failedCount} rejected.`,
  ];

  if (germanyDriftCount > 0) {
    lines.push(`Repeated failure basin: Paris->Germany list drift (${germanyDriftCount}/${total} recent cycles).`);
  }

  const paritySignals = recentCycles.filter((cycle) =>
    (cycle.evaluationNotes ?? []).some((note) => canonicalizeMemoryEntry(note).includes("contradictory capital country terms"))
  ).length;
  if (paritySignals > 0 || germanyDriftCount > 0) {
    lines.push("Prioritize parity tests around the first wrong layer or expert-down path before more speculative speed work.");
  }

  return lines.join("\n");
}

// ── Command runner ───────────────────────────────────────────────────

type RunResult = { exitCode: number; stdout: string; stderr: string };

function formatElapsed(startMs: number): string {
  const s = ((Date.now() - startMs) / 1000) | 0;
  if (s < 60) return `${s}s`;
  return `${(s / 60) | 0}m${s % 60}s`;
}

async function runCommand(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    timeout?: number;
    streamOutput?: boolean;
    stdoutLineFormatter?: (line: string) => string | null;
    stderrLineFormatter?: (line: string) => string | null;
    env?: Record<string, string>;
  } = {},
): Promise<RunResult> {
  return new Promise((res) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout,
      env: opts.env ? { ...process.env, ...opts.env } : process.env,
    });
    let stdout = "", stderr = "", lineBuffer = "", stderrLineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!opts.streamOutput) return;
      if (opts.stdoutLineFormatter) {
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const formatted = opts.stdoutLineFormatter(line);
          if (formatted != null) process.stdout.write(formatted);
        }
      } else {
        process.stdout.write(text);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (!opts.streamOutput) return;
      if (opts.stderrLineFormatter) {
        stderrLineBuffer += text;
        const lines = stderrLineBuffer.split("\n");
        stderrLineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const formatted = opts.stderrLineFormatter(line);
          if (formatted != null) process.stderr.write(formatted);
        }
      } else {
        process.stderr.write(text);
      }
    });
    child.on("error", () => res({ exitCode: -1, stdout, stderr }));
    child.on("close", (code) => {
      if (opts.streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const formatted = opts.stdoutLineFormatter(lineBuffer);
        if (formatted != null) process.stdout.write(formatted);
      }
      if (opts.streamOutput && opts.stderrLineFormatter && stderrLineBuffer.trim()) {
        const formatted = opts.stderrLineFormatter(stderrLineBuffer);
        if (formatted != null) process.stderr.write(formatted);
      }
      res({ exitCode: code ?? -1, stdout, stderr });
    });
  });
}

// ── Sample collection + aggregation ──────────────────────────────────

export type SampleAggregate = {
  tokPerSec: number | null;
  trimmed: boolean;
  trimCount: number;
  range: number;
  bimodal: boolean;
};

function medianNumber(samples: number[]): number | null {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? null;
}

/**
 * Aggregate raw timed tok/s samples into a single representative number.
 * Default is the median; with `trim` enabled the symmetric high+low extremes
 * are dropped first (1 each for 5-6 samples, 2 each for 7+). Also reports
 * whether the spread looks bimodal/thermal — the loop's accept band is tighter
 * than a >1.5 tok/s straddle, so a "kept" verdict at that noise level is
 * suspect and is what triggers a confirmation re-run.
 */
export function aggregateTimedSamples(samples: number[], trim: boolean = BENCHMARK_TRIM): SampleAggregate {
  const sorted = [...samples].sort((a, b) => a - b);
  let kept: number[] = sorted;
  let trimmed = false;
  let trimCount = 0;
  if (trim && sorted.length >= 7) {
    trimCount = 2;
    kept = sorted.slice(2, sorted.length - 2);
    trimmed = true;
  } else if (trim && sorted.length >= 5) {
    trimCount = 1;
    kept = sorted.slice(1, sorted.length - 1);
    trimmed = true;
  }
  const tokPerSec = kept.length > 0 ? kept[Math.floor(kept.length / 2)] : null;
  const range = sorted.length > 1 ? sorted[sorted.length - 1] - sorted[0] : 0;
  let bimodal = false;
  if (tokPerSec != null && sorted.length > 1) {
    const med = tokPerSec;
    const hasLow = sorted.some((s) => s <= med - 0.75);
    const hasHigh = sorted.some((s) => s >= med + 0.75);
    bimodal = range > 1.5 && hasLow && hasHigh;
  }
  return { tokPerSec, trimmed, trimCount, range, bimodal };
}

/**
 * Decide whether a candidate cycle's measurement is too close to the promotion
 * boundary (or too noisy) to trust, and therefore warrants extra samples before
 * the keep/revert verdict. Confirmation only ever helps when the change built,
 * tested, ran, and produced correct output — broken output is reverted outright
 * regardless of speed, so re-measuring it is wasted runtime.
 */
export function shouldConfirmCandidate(args: {
  containsReference: boolean;
  ranOk: boolean;
  verifyTps: number;
  bestTps: number;
  improveBand: number;
  bimodal: boolean;
  sampleCount: number;
  confirmRuns: number;
}): boolean {
  if (args.confirmRuns <= 0) return false;
  if (!args.containsReference || !args.ranOk) return false;
  if (args.verifyTps <= 0) return false;
  // Already have a robust sample count — no extra confidence to gain.
  if (args.sampleCount >= 9) return false;
  // Bimodal/thermal spread: the median picked one cluster; verify it.
  if (args.bimodal) return true;
  // Within one band of the promotion line — extra samples can flip the verdict
  // in either direction (rescue a real win just below, reject noise just above).
  const promotionLine = args.bestTps + args.improveBand;
  return Math.abs(args.verifyTps - promotionLine) <= args.improveBand;
}

// ── Build, test, and run ─────────────────────────────────────────────

/**
 * Run the built binary `runs` times and collect parsed tok/s samples. Used for
 * both the in-cycle benchmark and the optional confirmation re-run. Does not
 * build or test — the caller guarantees `./zig-out/bin/zinc` is current.
 */
const sleep = (ms: number) => new Promise<void>((res) => setTimeout(res, ms));

async function collectTimedSamples(
  maxTokens: number,
  runs: number,
  label: string,
): Promise<{ samples: number[]; promptTokenSamples: number[]; lastRun: RunResult; lastCombined: string; droppedShort: number }> {
  const samples: number[] = [];
  const promptTokenSamples: number[] = [];
  let lastRun: RunResult = { exitCode: -1, stdout: "", stderr: "" };
  let lastCombined = "";
  let droppedShort = 0;
  for (let sample = 0; sample < runs; sample++) {
    if (sample > 0 && BENCHMARK_COOLDOWN_MS > 0) {
      // Inter-sample cool-down: let GPU clocks settle before the next timed run
      // so we measure sustained throughput, not the thermal slope.
      await sleep(BENCHMARK_COOLDOWN_MS);
    }
    const run = await runCommand(
      "./zig-out/bin/zinc",
      [...zincModelArgs(), ...zincPromptArgs(), "-n", String(maxTokens)],
      { timeout: RUN_TIMEOUT_MS },
    );
    lastRun = run;
    lastCombined = run.stderr + run.stdout;
    if (run.exitCode !== 0) break; // crash — no point running more samples
    const tps = parseTokPerSec(lastCombined, METRIC_MODE);
    const promptTokens = parsePrefillTokenCount(lastCombined);
    const tokens = parseTokensGenerated(lastCombined);
    // Reject samples that didn't generate enough tokens to give a stable
    // throughput estimate. A model that EOS'd after 1 token reports a
    // single-iteration tok/s (noisy + not comparable to llama.cpp tg128).
    // Only applies in decode mode — prefill is timed off prompt ingest, not
    // generation length.
    if (
      METRIC_MODE === "decode" &&
      MIN_DECODE_TOKENS > 0 &&
      tps != null &&
      tokens < MIN_DECODE_TOKENS
    ) {
      droppedShort++;
      console.log(clr("1;33", `    ${label} ${sample + 1}/${runs}: ${tps.toFixed(2)} ${METRIC_LABEL} DROPPED (generated only ${tokens} tokens; min ${MIN_DECODE_TOKENS})`));
      continue;
    }
    if (tps != null) {
      samples.push(tps);
      if (promptTokens != null) promptTokenSamples.push(promptTokens);
      console.log(clr("2", `    ${label} ${sample + 1}/${runs}: ${tps.toFixed(2)} ${METRIC_LABEL}${promptTokens != null ? ` prompt=${promptTokens}tok` : ""}${tokens > 0 ? ` (${tokens} tok)` : ""}`));
    }
  }
  return { samples, promptTokenSamples, lastRun, lastCombined, droppedShort };
}

/**
 * Inflate the promotion band when measured sample noise dwarfs it.
 *
 * Without this, on thermally-noisy decode runs the band can be ~1 tok/s while
 * sample spread is ~10 tok/s, so a "+1.0 tok/s improvement" verdict is mostly
 * noise. The verdict ratchets the recorded best upward on luck rather than on
 * real progress (Effort 16 cycles 1-24 saw 0.21→0.30 inflated this way before
 * the original noise band was tightened).
 *
 * Heuristic: require the candidate to beat best by at least ~30% of the
 * observed range. Pure ranges of 0-2 tok/s leave the base band untouched
 * (clean signal); ranges of 7-11 tok/s lift the band to ~2-3 tok/s, which
 * is the smallest improvement that survives ~95% of the observed jitter.
 * Returns the larger of the original band and this noise-aware floor.
 */
export function noiseAwareImproveBand(baseBand: number, samples: number[]): number {
  if (samples.length < 3) return baseBand;
  const sorted = [...samples].sort((a, b) => a - b);
  const range = sorted[sorted.length - 1] - sorted[0];
  if (range <= 0) return baseBand;
  return Math.max(baseBand, range * 0.3);
}

async function buildTestRun(maxTokens: number): Promise<BuildRunResult> {
  const buildArgs = zigBuildArgs();
  console.log(clr("1;33", `  🔨 Building (${buildArgs.join(" ")})...`));
  const build = await runCommand("zig", buildArgs, { timeout: 120_000 });

  if (build.exitCode !== 0) {
    return {
      buildExitCode: build.exitCode,
      buildOutput: build.stderr + build.stdout,
      testExitCode: -1,
      testOutput: "",
      runExitCode: null,
      runOutput: "",
      phase: "fix",
      tokPerSec: null,
      tokPerSecSamples: [],
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
      offTopic: false,
      evaluationNotes: [],
      error: "Build failed",
    };
  }
  console.log(clr("1;32", "  ✅ Build OK"));

  console.log(clr("1;33", "  🧪 Testing..."));
  const test = await runCommand("zig", ["build", "test"], { timeout: TEST_TIMEOUT_MS });

  if (test.exitCode !== 0) {
    return {
      buildExitCode: 0,
      buildOutput: build.stderr,
      testExitCode: test.exitCode,
      testOutput: test.stderr + test.stdout,
      runExitCode: null,
      runOutput: "",
      phase: "fix",
      tokPerSec: null,
      tokPerSecSamples: [],
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
      offTopic: false,
      evaluationNotes: [],
      error: "Tests failed",
    };
  }
  console.log(clr("1;32", "  ✅ Tests OK"));

  if (MODEL_PATH && !existsSync(MODEL_PATH)) {
    console.log(clr("1;33", "  ⚠ Model not found, skipping inference run"));
    return {
      buildExitCode: 0,
      buildOutput: build.stderr,
      testExitCode: 0,
      testOutput: "",
      runExitCode: null,
      runOutput: "",
      phase: "implement",
      tokPerSec: null,
      tokPerSecSamples: [],
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
      offTopic: false,
      evaluationNotes: [],
      error: null,
    };
  }

  const warmupLabel = BENCHMARK_WARMUPS > 0 ? `${BENCHMARK_WARMUPS} warmup + ` : "";
  console.log(clr("1;33", `  🚀 Running inference (${maxTokens} tokens, ${warmupLabel}${BENCHMARK_RUNS} samples, ${PROMPT_MODE} prompt)...`));
  const tokPerSecSamples: number[] = [];
  const promptTokenSamples: number[] = [];
  let lastRun: RunResult = { exitCode: -1, stdout: "", stderr: "" };
  let lastCombined = "";

  // Pre-roll throwaway runs to prime caches (file mmap, Metal pipeline,
  // OS page cache) before the timed samples. These do NOT preserve GPU
  // clock state across process boundaries — that's a deeper limitation —
  // but they at least surface model-load failures here instead of mid-
  // measurement, and warm the OS file cache so timing starts from a
  // consistent state.
  for (let warmup = 0; warmup < BENCHMARK_WARMUPS; warmup++) {
    const run = await runCommand(
      "./zig-out/bin/zinc",
      [...zincModelArgs(), ...zincPromptArgs(), "-n", String(maxTokens)],
      { timeout: RUN_TIMEOUT_MS },
    );
    lastRun = run;
    lastCombined = run.stderr + run.stdout;
    if (run.exitCode !== 0) break; // crash — surface immediately, don't pretend to measure
    const tps = parseTokPerSec(lastCombined, METRIC_MODE);
    const promptTokens = parsePrefillTokenCount(lastCombined);
    if (tps != null) {
      console.log(clr("2", `    warmup ${warmup + 1}/${BENCHMARK_WARMUPS}: ${tps.toFixed(2)} ${METRIC_LABEL}${promptTokens != null ? ` prompt=${promptTokens}tok` : ""} (discarded)`));
    }
  }

  // If a warmup crashed, skip the timed loop and surface the failure below.
  const warmupCrashed = BENCHMARK_WARMUPS > 0 && lastRun.exitCode !== 0;

  // Aggregate samples. Default is median of all timed samples. With
  // BENCHMARK_TRIM, drop symmetric high+low extremes before taking the
  // median:
  //   * 5–6 samples:  drop 1 high + 1 low  → median of remaining 3–4
  //   * 7+   samples: drop 2 high + 2 low  → median of remaining ≥3
  // The 2-trim variant kicks in for overnight runs (BENCHMARK_RUNS=7),
  // where Effort 14 logged bimodal samples on M1 Max — e.g. [42, 45]
  // sets where a single-trim still picked the wrong cluster. Symmetric
  // 2-trim from 7 samples leaves the middle 3, which is robust to the
  // cold-GPU outlier and the occasional too-warm outlier together.
  let droppedShortSamples = 0;
  if (!warmupCrashed) {
    const timed = await collectTimedSamples(maxTokens, BENCHMARK_RUNS, "sample");
    tokPerSecSamples.push(...timed.samples);
    promptTokenSamples.push(...timed.promptTokenSamples);
    lastRun = timed.lastRun;
    lastCombined = timed.lastCombined;
    droppedShortSamples += timed.droppedShort;
  }
  if (droppedShortSamples > 0) {
    console.log(clr("1;33", `    ⚠ Dropped ${droppedShortSamples} sample(s) with fewer than ZINC_MIN_DECODE_TOKENS=${MIN_DECODE_TOKENS} generated tokens — measurement floor for sustained decode`));
  }

  const agg = aggregateTimedSamples(tokPerSecSamples);
  const tokPerSec = agg.tokPerSec;
  const promptTokens = medianNumber(promptTokenSamples);
  const tokensGenerated = parseTokensGenerated(lastCombined);
  const outputText = parseOutputText(lastCombined);
  const evaluation = evaluateOutputText(outputText);
  const workload = workloadFromMeasurement({ runOutput: lastCombined, promptTokens });

  if (tokPerSec != null && tokPerSecSamples.length > 1) {
    const aggLabel = agg.trimmed
      ? `trimmed median (drop ${agg.trimCount} high + ${agg.trimCount} low)`
      : "median";
    const flag = agg.bimodal ? " ⚠ THERMAL" : "";
    console.log(clr("1;36", `    ${aggLabel}: ${tokPerSec.toFixed(2)} ${METRIC_LABEL} [${tokPerSecSamples.map(s => s.toFixed(1)).join(", ")}] range=${agg.range.toFixed(1)}${flag}`));
  }

  const result: BuildRunResult = {
    buildExitCode: 0,
    buildOutput: build.stderr,
    testExitCode: 0,
    testOutput: "",
    runExitCode: lastRun.exitCode,
    runOutput: lastCombined,
    phase: "implement",
    tokPerSec,
    tokPerSecSamples,
    promptTokens,
    promptTokenSamples,
    workload,
    tokensGenerated,
    outputText,
    containsReference: evaluation.containsReference,
    strongAnswer: evaluation.strongAnswer,
    outputQualityScore: evaluation.outputQualityScore,
    offTopic: evaluation.offTopic,
    evaluationNotes: evaluation.evaluationNotes,
    error: lastRun.exitCode !== 0 ? `Runtime exit code ${lastRun.exitCode}` : null,
  };
  result.phase = detectPhase(result);
  return result;
}

// ── Agent stream formatters ──────────────────────────────────────────

type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDelta: boolean;
};

type CodexStreamState = {
  startedCommandIds: Set<string>;
};

function coerceDisplayText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    const parts = value.map((entry) => coerceDisplayText(entry)).filter((entry) => entry.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(value, null, 2); } catch { return ""; }
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const parts = [
      record.text,
      record.message,
      record.output,
      record.stdout,
      record.stderr,
      record.content,
      record.result,
      record.summary,
      record.output_text,
    ].map((entry) => coerceDisplayText(entry)).filter((entry) => entry.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(record, null, 2); } catch { return ""; }
  }
  return "";
}

function formatToolInput(name: string, jsonBuf: string): string {
  let input: Record<string, unknown> = {};
  try { input = JSON.parse(jsonBuf); } catch { return ""; }
  const out: string[] = [];
  const MAX_DIFF = 5;

  if (name === "edit") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
    const oldLines = ((input.old_string as string | undefined) ?? "").split("\n");
    const newLines = ((input.new_string as string | undefined) ?? "").split("\n");
    for (const l of oldLines.slice(0, MAX_DIFF)) out.push(clr("31", `   - ${l}`));
    if (oldLines.length > MAX_DIFF) out.push(clr("2", `   - … (${oldLines.length - MAX_DIFF} more)`));
    for (const l of newLines.slice(0, MAX_DIFF)) out.push(clr("32", `   + ${l}`));
    if (newLines.length > MAX_DIFF) out.push(clr("2", `   + … (${newLines.length - MAX_DIFF} more)`));
  } else if (name === "write") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const lineCount = ((input.content as string | undefined) ?? "").split("\n").length;
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")} (${lineCount} lines)`));
  } else if (name === "bash") {
    const cmd = (input.command as string | undefined) ?? "?";
    out.push(clr("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "…" : cmd}`));
  } else if (name === "read") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
  } else if (name === "grep") {
    const pattern = (input.pattern as string | undefined) ?? "?";
    out.push(clr("2", ` → /${pattern}/`));
  } else if (name === "glob") {
    out.push(clr("2", ` → ${(input.pattern as string | undefined) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

function formatClaudeStreamLine(rawLine: string, state: ClaudeStreamState): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return null; }

  if (event.type === "stream_event") {
    const e = event.event as Record<string, unknown> | undefined;
    if (!e) return null;

    if (e.type === "content_block_start") {
      const block = e.content_block as Record<string, unknown> | undefined;
      if (block?.type === "tool_use") {
        state.currentToolName = (block.name as string) ?? "tool";
        state.currentBlockIsToolUse = true;
        state.inputJsonBuffer = "";
        state.inTextBlock = false;
        return `\n${clr("33", `🔧 ${state.currentToolName}`)}`;
      }
      if (block?.type === "text") {
        state.inTextBlock = true;
        state.currentBlockIsToolUse = false;
        return COLOR_ENABLED ? "\n\x1b[96m" : "\n";
      }
      state.inTextBlock = false;
      state.currentBlockIsToolUse = false;
      return null;
    }
    if (e.type === "content_block_delta") {
      const delta = e.delta as Record<string, unknown> | undefined;
      if (delta?.type === "input_json_delta") {
        state.inputJsonBuffer += (delta.partial_json as string) ?? "";
        return null;
      }
      if (delta?.type === "text_delta" && state.inTextBlock) {
        state.sawTextDelta = true;
        return delta.text as string;
      }
      return null;
    }
    if (e.type === "content_block_stop") {
      if (state.currentBlockIsToolUse) {
        state.currentBlockIsToolUse = false;
        const detail = formatToolInput(state.currentToolName ?? "", state.inputJsonBuffer);
        state.inputJsonBuffer = "";
        return detail || null;
      }
      if (state.inTextBlock) {
        state.inTextBlock = false;
        return COLOR_ENABLED ? "\x1b[0m\n" : "\n";
      }
      return null;
    }
    return null;
  }

  if (event.type === "user") {
    const result = event.tool_use_result as Record<string, unknown> | undefined;
    if (result) return clr("32", "   ☑ accepted") + "\n";
    return null;
  }

  if (event.type === "assistant") {
    const msg = event.message as Record<string, unknown> | undefined;
    if (!msg) return null;
    const content = msg.content;
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const block of content) {
        const b = block as Record<string, unknown>;
        if (b?.type === "text" && typeof b.text === "string" && b.text.trim())
          parts.push(b.text);
      }
      const text = parts.join("\n");
      if (!text.trim() || state.sawTextDelta) {
        state.sawTextDelta = false;
        return null;
      }
      return clr("96", text) + "\n";
    }
    return null;
  }
  return null;
}

function formatCodexJsonEvent(
  rawLine: string,
  state: CodexStreamState,
): string | null | undefined {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return undefined; }

  const eventType = typeof event.type === "string" ? event.type : "";
  if (eventType === "thread.started" || eventType === "turn.started" || eventType === "turn.completed")
    return null;
  if (eventType === "error") {
    const message = coerceDisplayText(event.message);
    return message ? `${clr("31", message)}\n` : null;
  }
  if (!eventType.startsWith("item.")) return null;

  const item = event.item as Record<string, unknown> | undefined;
  if (!item) return null;

  const itemType = typeof item.type === "string" ? item.type : "";
  const itemId = typeof item.id === "string" ? item.id : "";
  const phase = eventType.slice("item.".length);

  if (itemType === "reasoning" && phase === "completed") {
    const text = coerceDisplayText(item.summary ?? item.text ?? item.message ?? item.content);
    return text ? `${clr("2", `thinking: ${text}`)}\n` : null;
  }

  if (itemType === "command_execution") {
    const input = item.input as Record<string, unknown> | undefined;
    const cmd = coerceDisplayText(item.command ?? input?.command ?? "").trim();
    const output = coerceDisplayText(item.aggregated_output ?? item.output ?? item.stdout ?? "");
    const exitCode = typeof item.exit_code === "number" ? item.exit_code : null;
    const startedAlready = itemId ? state.startedCommandIds.has(itemId) : false;

    if (phase === "started") {
      if (itemId) state.startedCommandIds.add(itemId);
      return cmd
        ? `\n${clr("33", "🔧 bash")}\n${clr("2", `   $ ${cmd}`)}\n`
        : `\n${clr("33", "🔧 bash")}\n`;
    }

    let out = "";
    if (!startedAlready) {
      out += `\n${clr("33", "🔧 bash")}\n`;
      if (cmd) out += clr("2", `   $ ${cmd}`) + "\n";
    }
    if (phase === "completed") {
      const lines = output.split("\n").filter((line) => line.trim());
      const tail = lines.slice(-3);
      const statusColor = exitCode === 0 ? "32" : exitCode == null ? "33" : "31";
      const statusText = exitCode === 0
        ? "   ☑ accepted"
        : exitCode == null
          ? "   ⚠ completed"
          : `   ✖ exit ${exitCode}`;
      const body = tail.length > 0
        ? (lines.length > 3 ? clr("2", "   …\n") : "") +
          tail.map((line) => clr("2", `   ${line.trim()}`)).join("\n") +
          "\n"
        : "";
      out += `${clr(statusColor, statusText)}\n${body}`;
    }
    if (itemId) state.startedCommandIds.delete(itemId);
    return out || null;
  }

  if (itemType === "file_change" && phase === "completed") {
    const changesSource = [item.changes, item.file_changes, item.files];
    const changes = changesSource.flatMap((value) => {
      if (!Array.isArray(value)) return [];
      return value.map((entry) => {
        const change = entry as Record<string, unknown>;
        return {
          path: coerceDisplayText(change.path ?? change.file_path ?? "?"),
          action: coerceDisplayText(change.change_type ?? change.kind ?? ""),
        };
      });
    });
    if (changes.length === 0) return null;
    let out = `\n${clr("35", "📝 file change")}\n`;
    for (const change of changes.slice(0, 6)) {
      out += clr("2", `   ${change.action ? `${change.action}: ` : ""}${change.path}`) + "\n";
    }
    return out;
  }

  if (itemType === "agent_message" && phase === "completed") {
    const text = coerceDisplayText(item.text ?? item.message ?? item.output_text ?? item.content);
    return text ? `${clr("96", text)}\n` : null;
  }

  if (itemType === "error" && phase === "completed") {
    const text = coerceDisplayText(item.text ?? item.message ?? item.content);
    return text ? `${clr("33", text)}\n` : null;
  }

  return null;
}

function formatCodexStreamLine(rawLine: string, state: CodexStreamState): string | null {
  const jsonFormatted = formatCodexJsonEvent(rawLine, state);
  if (jsonFormatted !== undefined) return jsonFormatted;
  if (!rawLine.trim()) return "\n";
  if (rawLine === "thinking") return `${clr("2", rawLine)}\n`;
  if (rawLine === "codex") return `${clr("1;35", rawLine)}\n`;
  if (rawLine.includes("still running")) return `${clr("2", rawLine)}\n`;
  return `${clr("2", `[codex] ${rawLine}`)}\n`;
}

function formatCodexStderrLine(rawLine: string): string | null {
  if (!rawLine.trim()) return "\n";
  return `${clr("2", `[codex] ${rawLine}`)}\n`;
}

// ── Agent invocation ─────────────────────────────────────────────────

// Reasoning-effort knob for the Claude agent. The CLI accepts
// low|medium|high|xhigh|max; default high. Override via ZINC_CLAUDE_EFFORT
// (set ZINC_CLAUDE_EFFORT=max for the top tier).
const CLAUDE_EFFORT = process.env.ZINC_CLAUDE_EFFORT ?? "high";

function buildClaudeArgs(prompt: string, model?: string): string[] {
  const args = [
    "-p",
    "--verbose",
    "--output-format", "stream-json",
    "--include-partial-messages",
    `--disallowed-tools=${BLOCKED_AGENT_OPS.join(",")}`,
    "--permission-mode", "bypassPermissions",
    "--effort", CLAUDE_EFFORT,
  ];
  if (model) args.push("--model", model);
  args.push(prompt);
  return args;
}

// Match the reasoning-effort knob `loops/optimize_perf.ts` uses for Codex.
// `xhigh` is the top tier; override via ZINC_CODEX_REASONING_EFFORT if a cycle
// needs something cheaper.
const CODEX_REASONING_EFFORT = process.env.ZINC_CODEX_REASONING_EFFORT ?? "xhigh";
const CODEX_MODEL = process.env.ZINC_CODEX_MODEL ?? "gpt-5.5";

function buildCodexArgs(prompt: string, model?: string): string[] {
  const args = [
    "exec",
    "-c",
    `model_reasoning_effort="${CODEX_REASONING_EFFORT}"`,
    "--skip-git-repo-check",
    "--json",
    "--color", "never",
    "--sandbox", "workspace-write",
    "--cd", REPO_ROOT,
  ];
  args.push("--model", model ?? CODEX_MODEL);
  args.push(prompt);
  return args;
}

async function resetCycleToPreHash(preHash: string): Promise<void> {
  await runCommand("git", ["reset", "--hard", preHash]);
  // `git reset --hard` leaves untracked files behind. Agents often add new
  // shaders/benchmarks while exploring; if a cycle is reverted, those files
  // must not leak into the next pre-cycle checkpoint and become "accepted".
  await runCommand("git", ["clean", "-fd", "--", ...LOOP_COMMIT_PATHS]).catch(() => {});
}

async function runAgent(agent: AgentKind, prompt: string, model?: string): Promise<RunResult> {
  const label = agent === "codex" ? "Codex" : "Claude";
  console.log(clr("1;34", SEP));
  console.log(clr("1;34", `  🧠 AGENT PROMPT (${label})`));
  console.log(clr("1;34", SEP));
  const lines = prompt.split("\n");
  for (const line of lines.slice(0, 20)) process.stdout.write(clr("2", line) + "\n");
  if (lines.length > 20) process.stdout.write(clr("2", `… (${lines.length - 20} more lines)\n`));
  console.log(clr("1;34", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(clr("2", `\n⏳ agent running (${formatElapsed(startedAt)})...\n`));
  }, 30_000);

  console.log(clr("1;36", SEP));
  console.log(clr("1;36", `  💬 AGENT RESPONSE (${label})`));
  console.log(clr("1;36", SEP));

  let result: RunResult;
  if (agent === "codex") {
    const streamState: CodexStreamState = {
      startedCommandIds: new Set(),
    };
    result = await runCommand("codex", buildCodexArgs(prompt, model), {
      streamOutput: true,
      timeout: 1_800_000, // 30 min
      stdoutLineFormatter: (line) => formatCodexStreamLine(line, streamState),
      stderrLineFormatter: formatCodexStderrLine,
    });
  } else {
    const streamState: ClaudeStreamState = {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDelta: false,
    };
    result = await runCommand("claude", buildClaudeArgs(prompt, model), {
      streamOutput: true,
      timeout: 1_800_000, // 30 min
      stdoutLineFormatter: (line) => formatClaudeStreamLine(line, streamState),
    });
  }

  clearInterval(heartbeat);
  console.log(clr("1;36", SEP));
  console.log(clr("1;32", `  ✅ ${label} done in ${formatElapsed(startedAt)}`));
  return result;
}

// ── Near-miss tracking + correctness-debug directive ─────────────────

/**
 * Record a fast-but-incorrect measurement as the standing near-miss target if
 * it is faster than the one already tracked. Only candidates that built,
 * tested, and RAN (no crash) but produced wrong output qualify — a crash or a
 * slower-than-accepted result is not a speedup worth chasing. Returns true if
 * `state.bestIncorrect` was updated.
 */
export function recordNearMiss(
  state: RunState,
  candidate: {
    cycle: number;
    tokPerSec: number | null;
    ranOk: boolean;
    containsReference: boolean;
    acceptedTps: number;
    description: string;
    selfAnalysis: string;
    outputText: string;
  },
): boolean {
  if (candidate.containsReference) return false; // correct output isn't a near-miss
  if (!candidate.ranOk) return false; // crash/build/test failures aren't speedups
  const tps = candidate.tokPerSec ?? 0;
  if (tps <= 0 || candidate.acceptedTps <= 0) return false;
  const gainPct = ((tps - candidate.acceptedTps) / candidate.acceptedTps) * 100;
  if (gainPct < NEAR_MISS_MIN_GAIN_PCT) return false; // not meaningfully faster
  if (state.bestIncorrect && state.bestIncorrect.tokPerSec >= tps) return false;
  state.bestIncorrect = {
    cycle: candidate.cycle,
    tokPerSec: tps,
    gainPctOverAccepted: gainPct,
    description: candidate.description,
    selfAnalysis: candidate.selfAnalysis,
    outputText: candidate.outputText.slice(0, 80),
  };
  return true;
}

/**
 * One-time migration for resumed runs whose state.json predates
 * `bestIncorrect`. Scans cycle history for the fastest cycle that built,
 * tested, and ran but was reverted for wrong output, and seeds the near-miss
 * target from it so the redirect kicks in on the very next cycle. Gain is
 * measured against the best kept-correct throughput in the same history. No-op
 * if a near-miss is already recorded.
 */
export function backfillNearMiss(state: RunState): boolean {
  if (state.bestIncorrect) return false;
  const acceptedTps = bestKeptCorrectTokPerSec(state);
  if (acceptedTps <= 0) return false;
  let best: CycleResult | null = null;
  for (const c of state.cycles) {
    if (c.kept) continue;
    if (c.containsReference) continue;
    if (c.buildExitCode !== 0 || c.testExitCode !== 0) continue;
    if (c.runExitCode !== 0) continue; // skip crashes/skipped runs
    if (c.tokPerSec == null) continue;
    const gainPct = ((c.tokPerSec - acceptedTps) / acceptedTps) * 100;
    if (gainPct < NEAR_MISS_MIN_GAIN_PCT) continue;
    if (!best || (c.tokPerSec ?? 0) > (best.tokPerSec ?? 0)) best = c;
  }
  if (!best || best.tokPerSec == null) return false;
  state.bestIncorrect = {
    cycle: best.cycle,
    tokPerSec: best.tokPerSec,
    gainPctOverAccepted: ((best.tokPerSec - acceptedTps) / acceptedTps) * 100,
    description: best.description,
    selfAnalysis: best.selfAnalysis ?? "",
    outputText: (best.outputText ?? "").slice(0, 80),
  };
  return true;
}

/**
 * Build the CORRECTNESS-DEBUG section that redirects the agent at a recorded
 * near-miss: a change that proved a speedup is reachable but broke output. The
 * directive is forceful — it tells the agent to STOP guessing new speed ideas
 * and instead localize the divergent layer with the bisection/validation env
 * flags the model already supports. The wording escalates once the run has
 * stalled, which doubles as the active plateau-escape redirect.
 */
/** Regex that flags a cycle as a "route-pack family" attempt — the family
 * the near-miss belongs to. Used to count how many cycles have already tried
 * to chase the same near-miss without making it correct, so the directive can
 * escalate from "go bisect" to "STOP guessing, run the validator". */
const NEAR_MISS_ROUTE_PACK_RE = /\b(route[- ]?pack(ed|ing)?|shared[- ]?gate|moe[- ]?route|prefix[- ]?layers?)\b/i;

/** Cycles since `bestIncorrect` was recorded that attempted the same family
 * but reverted with broken output (so they discovered nothing new and the
 * agent should change tactic). */
export function countNearMissFamilyReverts(state: RunState): number {
  const nm = state.bestIncorrect;
  if (!nm) return 0;
  let count = 0;
  for (const c of state.cycles) {
    if (c.cycle < nm.cycle) continue; // only count attempts AT OR AFTER the near-miss
    if (c.kept) continue;
    if (c.containsReference) continue;
    if (c.tokPerSec == null) continue;
    const text = `${c.description} ${c.selfAnalysis ?? ""}`;
    if (NEAR_MISS_ROUTE_PACK_RE.test(text)) count++;
  }
  return count;
}

/**
 * If the recent N cycles are dominated by `optimization` step-kinds with no
 * `analysis` / `enablement` foundation work, return a directive telling the
 * agent to build evidence/tooling rather than try another kernel variant.
 *
 * Triggered late enough to let an obviously-fast-and-fresh loop iterate
 * (don't nag in the first 8 cycles), and only when the agent has demonstrably
 * NOT been building tools. Designed for the Effort 5 cycle-10-25 pattern:
 * 16 consecutive optimization reverts targeting noise-bound infrastructure
 * because no microbench was ever built for the hot Q4_K/Q8_0 SSM shapes.
 */
export function buildStepKindDiversityNudge(
  state: Pick<RunState, "cycles">,
  windowSize: number = 10,
): string[] {
  const recent = state.cycles.slice(-windowSize);
  if (recent.length < 8) return [];
  let optCount = 0, otherCount = 0;
  for (const c of recent) {
    if (c.stepKind === "optimization") optCount++;
    else if (c.stepKind === "analysis" || c.stepKind === "enablement") otherCount++;
  }
  // Require ~all recent cycles to be optimization AND zero analysis/enablement.
  if (optCount < recent.length - 1) return [];
  if (otherCount > 0) return [];

  const lines: string[] = [];
  lines.push(`## ⚠ STEP-KIND IMBALANCE — ${optCount}/${recent.length} recent cycles were "optimization", 0 were "analysis"/"enablement"`);
  lines.push("");
  lines.push("Pattern: you keep proposing kernel/threadgroup/barrier retunes, almost all reverted by the noise-band. The harness has zero new exact-shape evidence to back the next attempt — every cycle is a blind shot in a regime where sample range exceeds the keep band.");
  lines.push("");
  lines.push("Before another optimization, do ONE of:");
  lines.push("1. **Add an exact-shape microbench** for the actual hot tensors (e.g. extend `benchmarks/metal_q8_shapes.zig` with the LM head / SSM qkv / ssm_out / shared-gate shapes from the latest profile). Default-off, foundation keep allowed at 0 % impact.");
  lines.push("2. **Add a per-kernel timing probe** behind an env flag that prints µs/dispatch for the top-5 hot kernels under `--profile`. Default-off.");
  lines.push("3. **Read the latest profile output below** and write down the ONE smallest-named-bucket hypothesis that explains the gap to llama.cpp — name the kernel + shape + estimated savings, then either implement that fix (not a retune) OR add the microbench that would confirm/reject it.");
  lines.push("");
  lines.push("Do NOT propose another threadgroup sweep, barrier-scope change, or hazard-tracking flag this cycle. Those have measured neutral across the recent window — the issue is lack of evidence, not lack of kernel candidates.");
  return lines;
}

export function buildNearMissDirective(
  state: RunState,
  acceptedBestTps: number,
  stalled: boolean,
): string[] {
  const nm = state.bestIncorrect;
  if (!nm) return [];
  // Only surface while the near-miss is still meaningfully ahead of where the
  // accepted tree already is — once a correct change matches/exceeds it, the
  // target is moot.
  if (nm.tokPerSec <= acceptedBestTps + Math.max(0.3, acceptedBestTps * 0.005)) return [];

  const familyReverts = countNearMissFamilyReverts(state);
  const escalate = familyReverts >= 5; // agent has already tried this family ≥5 times

  const lines: string[] = [];
  lines.push(
    escalate
      ? `## ★★★★ NEAR-MISS DEADLOCK — ${familyReverts} cycles attempted the ${nm.tokPerSec.toFixed(1)} ${METRIC_LABEL} family with NO correctness fix. STOP guessing variants. Localize the divergence FIRST.`
      : stalled
        ? `## ★★★ PLATEAU REDIRECT — chase the proven ${nm.tokPerSec.toFixed(1)} ${METRIC_LABEL} near-miss`
        : `## ★ KNOWN NEAR-MISS — a proven +${nm.gainPctOverAccepted.toFixed(1)}% speedup is correctness-blocked`,
  );
  lines.push(
    `Cycle ${nm.cycle} reached **${nm.tokPerSec.toFixed(1)} ${METRIC_LABEL}** (current accepted ≈ ${acceptedBestTps.toFixed(1)}, +${nm.gainPctOverAccepted.toFixed(1)}%) but was REVERTED because output broke to "${nm.outputText}".`,
  );
  lines.push(`That change: ${trunc(nm.description, 200)}`);
  // Carry the original agent's rationale forward verbatim (longer trunc).
  // Cycle 9 of the previous run captured the full bisection plan and named
  // the suspected divergence site in its own self-analysis — that evidence is
  // far more useful to the next agent than the directive alone.
  if (nm.selfAnalysis) {
    lines.push("");
    lines.push("Prior reasoning recorded with this near-miss (the agent who made it explained its plan):");
    lines.push(`> ${trunc(nm.selfAnalysis, 900).replace(/\n+/g, " ")}`);
  }
  lines.push("");
  lines.push(
    "This is the single highest-value lever in the run. The speedup is REAL and reachable — the only blocker is a numerical divergence somewhere in the route-pack / F32 shared-gate scatter family. Do NOT re-attempt blind variants of the same idea; that has already failed repeatedly.",
  );
  lines.push("");
  lines.push("**Validator infrastructure already exists in the tree** (built by prior cycles). Use these env flags — they are real and parsed by `src/compute/forward_metal.zig` / `forward.zig`:");
  lines.push("- `ZINC_QWEN36_35B_ROUTE_PACK_PREFIX_LAYERS=<N>` — cap how many leading SSM/attention layers use the fast path (bisection knob).");
  lines.push("- `ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_FULL=1` and `ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_FULL_BISECT=1` — run the route-pack path AND the reference per token; emit per-layer/per-tensor max-abs/L2 diffs.");
  lines.push("- `ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_LAYER=<L>` — pin the validator to one layer for a focused report.");
  lines.push("- `ZINC_QWEN36_35B_PREFILL_VALIDATE=1` + `ZINC_QWEN36_35B_PREFILL_VALIDATE_LAYER=<L>` + `ZINC_QWEN36_35B_PREFILL_VALIDATE_TOKENS=<N>` — broader per-layer logits/intermediate diff.");
  lines.push("- `ZINC_QWEN36_35B_SSM_PREFILL_PROJ` / `ZINC_QWEN36_35B_SSM_PROJ_VALIDATE_LAYER=<L>` — SSM-projection-specific validator.");
  lines.push("");
  if (escalate) {
    lines.push("**You are in DEADLOCK MODE.** Do not propose another route-pack variant. The single useful action this cycle is one of:");
    lines.push("(a) **Strengthen the validator** so it prints the first diverging *tensor name* and *layer index* in a single line you can grep for (today the validator reports diffs but the loop can't pin the failure to one tensor without you reading the output). Default-off; this is a foundation keep allowed at 0 % impact per Rule 4 of the effort.");
    lines.push("(b) **Read the most recent profile output and prior cycles' self-analyses below** and write down the smallest hypothesis that explains why the route-pack F32 shared-gate scatter produces `!!!!` — name the candidate tensor (e.g. `moe_route_scatter_shared_residual_gate_f32` scalar inputs, `router_f32_topk_batched_shared_gate` output, or the SSM-out → MoE handoff at non-zero start-layer) and the ONE small numerical fix you'd try.");
    lines.push("(c) If you have already done (a) and (b) in past cycles and the validator reports a specific divergent tensor, **fix that tensor's numerics this cycle** (e.g. match the slow-path reduction order, replace f32 accumulation with f64, fix a quant scale mismatch). One source change, one tensor.");
    lines.push("");
    lines.push("Pick exactly one of (a), (b), (c). Do NOT also retune kernel threadgroups or refactor unrelated paths in the same cycle — those have measured neutral across the recent stall window.");
  } else {
    lines.push("Plan for THIS cycle (pick one):");
    lines.push("1. Reproduce the fast path under a capped prefix (`ZINC_QWEN36_35B_ROUTE_PACK_PREFIX_LAYERS=<N>`) and binary-search the largest N where output still contains \"Paris\". The first layer beyond that N is where the fast path diverges. Report the cap value + Paris/no-Paris in `@@@SELF_ANALYSIS`.");
    lines.push("2. Run the existing validator at the suspect layer (`ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_FULL_BISECT=1 ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_LAYER=<L>`) and report the first diverging tensor name and max-abs diff. (The OUTER HARNESS runs the model — describe the change as wiring the validator into the prefill path and leave the actual run to the harness.)");
    lines.push("3. Fix the divergence at its source (match the validated slow-path scalar / reduction order) and KEEP the cap-based partial enablement if full enablement still drifts — partial is still a real win.");
  }
  // If the harness has captured a diagnostic run (validator output under
  // ZINC_NEAR_MISS_DIAGNOSTIC_ENV), splice the last 2 KB of it into the
  // directive. This is the evidence prior cycles lacked.
  const diag = state.lastNearMissDiagnostic;
  if (diag && diag.output) {
    lines.push("");
    lines.push(`**Latest near-miss diagnostic** (cycle ${diag.cycle}, env: ${diag.envApplied.join(", ")}). Read this BEFORE choosing your change — it shows what the validator saw with the fast path enabled:`);
    lines.push("```");
    lines.push(diag.output.slice(-2000));
    lines.push("```");
  }

  lines.push("");
  lines.push("Report which layer/tensor diverged (or which env knob isolated it) in `@@@SELF_ANALYSIS` even if you don't fully fix it this cycle — that evidence is what unblocks the next cycle.");
  return lines;
}

// ── Prompt builder ───────────────────────────────────────────────────

const trunc = (s: string, max: number) => (s.length > max ? s.slice(0, max) + "…" : s);

export function buildPrompt(state: RunState, lastResult: BuildRunResult): string {
  const { cycles, failedApproaches, phase } = state;

  const historyBlock = cycles.length > 0
    ? cycles.slice(-15).map(h => {
        const desc = trunc(h.description, 70);
        return `  #${h.cycle}: [${h.phase}] ${desc} → ${h.kept ? "KEPT" : "REVERTED"}${h.tokPerSec != null ? ` (${h.tokPerSec.toFixed(1)} tok/s)` : ""}${h.containsReference ? " ✅CORRECT" : ""}`;
      }).join("\n")
    : "  (none yet)";

  const failedBlock = failedApproaches.length > 0
    ? failedApproaches.slice(-20).map((f, n) => `  ${n + 1}. ${trunc(f, 120)}`).join("\n")
    : "  (none yet)";

  const ideasBlock = state.ideas.length > 0
    ? state.ideas.slice(-15).map((idea, i) => `  ${i + 1}. ${trunc(idea, 120)}`).join("\n")
    : "  (none yet)";

  const buildOut = lastResult.buildOutput.slice(-2000);
  const testOut = lastResult.testOutput.slice(-2000);
  const runOut = lastResult.runOutput.slice(-3000);

  // Build diagnosis based on phase
  const diagnosis: string[] = [];
  const correctnessReference = REFERENCE_TEXT.trim().length > 0
    ? REFERENCE_TEXT.trim()
    : "non-empty coherent output";

  if (lastResult.buildExitCode !== 0) {
    diagnosis.push("## Status: BUILD FAILURE");
    diagnosis.push("Fix the compilation error shown below. Do NOT attempt performance work until it compiles.");
  } else if (lastResult.testExitCode !== 0) {
    diagnosis.push("## Status: TEST FAILURE");
    diagnosis.push("Fix the failing test. All 27+ Metal tests must pass before any perf work.");
  } else if (lastResult.runExitCode !== 0 && lastResult.runExitCode !== null) {
    diagnosis.push(`## Status: RUNTIME CRASH (exit code ${lastResult.runExitCode})`);
    if (/already reserved|reserved by another zinc/i.test(lastResult.runOutput)) {
      // Environmental, not a code bug: another zinc process holds the GPU.
      diagnosis.push("This is NOT a code bug. The Metal GPU is exclusively reserved by another zinc process (a stale/leftover instance, or an agent that ran the model itself). The model loaded fine; nothing in the source caused this.");
      diagnosis.push("DO NOT edit code to 'fix' this and DO NOT run the model yourself. Make NO source change this cycle — emit @@@STEP_KIND: analysis and explain that the GPU was occupied. The operator must free the GPU (kill the stray zinc) before the harness can measure. The harness will re-measure next cycle once the GPU is free.");
    } else {
      diagnosis.push("Build and tests pass but ZINC crashes during inference. Fix the crash first.");
    }
  } else if (!lastResult.containsReference) {
    diagnosis.push(`## Status: CORRECTNESS REGRESSION — output doesn't contain "${correctnessReference}"`);
    diagnosis.push(`Output text: "${lastResult.outputText}"`);
    diagnosis.push("The previous optimization broke correctness. You MUST restore correct output first.");
    diagnosis.push("Read the git diff to see what changed and revert the problematic part.");
  } else if (lastResult.tokPerSec != null && lastResult.tokPerSec < TARGET_TOK_PER_SEC) {
    const current = lastResult.tokPerSec;
    const gap = TARGET_TOK_PER_SEC - current;
    const pctNeeded = ((gap / current) * 100).toFixed(0);
    const currentAccepted = Math.max(currentAcceptedTokPerSec(state), correctResultTokPerSec(lastResult));
    const bestKept = Math.max(bestKeptCorrectTokPerSec(state), correctResultTokPerSec(lastResult));
    diagnosis.push(`## Status: CORRECT OUTPUT — ${current.toFixed(2)} ${METRIC_LABEL} → target ≥${TARGET_TOK_PER_SEC}`);
    if (lastResult.promptTokens != null) {
      diagnosis.push(`Benchmark prompt shape: ${lastResult.promptTokens} prompt tokens${lastResult.promptTokenSamples?.length ? ` [${lastResult.promptTokenSamples.join(", ")}]` : ""}. Treat this as part of the workload contract; do not overfit one exact boundary without checking nearby public prompt lengths.`);
    }
    if (state.workload) {
      diagnosis.push(
        `Locked workload: model=${state.workload.model}, mode=${state.workload.promptMode}, metric=${state.workload.metricMode}, max_tokens=${state.workload.maxTokens}, prompt_tokens=${state.workload.promptTokens ?? "?"}, raw_prompt_hash=${state.workload.rawPromptHash}, prepared_prompt_hash=${state.workload.preparedPromptHash ?? "unknown"}, reference_hash=${state.workload.referenceHash}.`,
      );
      diagnosis.push("Do not change prompt preparation, chat-template behavior, correctness oracle text, or prompt token count as a performance optimization. Such changes are workload changes, not kernel throughput wins, and the harness will reject them as incomparable.");
    }
    diagnosis.push(`Gap: ${gap.toFixed(1)} ${METRIC_LABEL} (need ${pctNeeded}% improvement)`);
    if (currentAccepted > 0 || bestKept > 0) {
      diagnosis.push(`Accepted baseline: current tree ${currentAccepted.toFixed(2)} ${METRIC_LABEL}; highest kept-correct ${bestKept.toFixed(2)} ${METRIC_LABEL}.`);
    }
    if (state.bestTokPerSec > 0 && bestKept > state.bestTokPerSec + 0.05) {
      diagnosis.push(`Note: saved promoted-best checkpoint is ${state.bestTokPerSec.toFixed(2)} ${METRIC_LABEL}; use the kept-correct/current-tree baseline above for comparisons.`);
    }
    diagnosis.push(`Output: "${trunc(lastResult.outputText, 80)}"`);
    if (lastResult.tokPerSecSamples.length > 1) {
      diagnosis.push(`Benchmark samples: [${lastResult.tokPerSecSamples.map(s => s.toFixed(1)).join(", ")}] ${METRIC_LABEL}`);
      const sampleMin = Math.min(...lastResult.tokPerSecSamples);
      const sampleMax = Math.max(...lastResult.tokPerSecSamples);
      const sampleRange = sampleMax - sampleMin;
      if (sampleRange > Math.max(2.0, current * 0.2)) {
        diagnosis.push(`Benchmark variance warning: sample range ${sampleRange.toFixed(1)} tok/s is too wide for reliable direction. Do not optimize from the low sample; compare against accepted best and profile evidence.`);
      }
    }
  } else {
    diagnosis.push(`## Status: TARGET REACHED — ${lastResult.tokPerSec?.toFixed(1)} ${METRIC_LABEL} ≥${TARGET_TOK_PER_SEC}`);
    diagnosis.push("Performance target met!");
  }

  // Known near-miss redirect. When a prior change proved a speedup is reachable
  // but broke output, point the agent at localizing the divergent layer instead
  // of rediscovering the dead end. Only surface while currently correct (we're
  // optimizing, not fixing) and the near-miss still leads the accepted tree.
  const stalledNow = state.stalledCycles >= STALL_THRESHOLD;
  if (lastResult.containsReference) {
    const acceptedForNearMiss = Math.max(
      currentAcceptedTokPerSec(state),
      bestKeptCorrectTokPerSec(state),
      correctResultTokPerSec(lastResult),
    );
    const nearMissLines = buildNearMissDirective(state, acceptedForNearMiss, stalledNow);
    if (nearMissLines.length > 0) {
      diagnosis.push("");
      diagnosis.push(...nearMissLines);
    }
  }

  // Step-kind diversity nudge. When recent cycles are all optimization shots
  // and no analysis/enablement work was done, push back. Effort 5 cycles 10-25
  // drifted into noise-bound barrier/encoder retunes (10+ reverts) because the
  // agent never built an exact-shape microbench for the actual hot tensors;
  // surface the imbalance so the next cycle considers foundation work.
  const diversityLines = buildStepKindDiversityNudge(state);
  if (diversityLines.length > 0) {
    diagnosis.push("");
    diagnosis.push(...diversityLines);
  }

  const familyCooldownLines = buildFamilyCooldownDirective(state);
  if (familyCooldownLines.length > 0) {
    diagnosis.push("");
    diagnosis.push(...familyCooldownLines);
  }

  const structuralPivotLines = buildStructuralPivotDirective(state);
  if (structuralPivotLines.length > 0) {
    diagnosis.push("");
    diagnosis.push(...structuralPivotLines);
  }

  // Cross-effort guard status. When a baseline exists, surface the latest
  // measurement so the agent sees whether their changes are quietly regressing
  // the other metric. Escalated wording when regression ≥ 5%.
  const crossLines = buildCrossEffortStatus(state);
  if (crossLines.length > 0) {
    diagnosis.push("");
    diagnosis.push(...crossLines);
  }

  // Stall warning
  const recentProgress = recentAcceptedProgress(state);
  if (state.stalledCycles >= STALL_THRESHOLD && recentProgress.hasProgress) {
    diagnosis.push("");
    diagnosis.push(`## Limited Progress — accepted baseline moved ${recentProgress.start.toFixed(2)} → ${recentProgress.end.toFixed(2)} ${METRIC_LABEL} recently`);
    diagnosis.push("");
    diagnosis.push("Do not treat this as a clean stall, but the gain is still too small for the target gap.");
    diagnosis.push("Use the latest profile/validator evidence to convert the current default-off or measurement-only path into a default-on correctness-preserving speedup.");
    diagnosis.push("Avoid adding another passive probe unless it answers the exact blocker exposed by the previous one.");
  } else if (state.stalledCycles >= STALL_THRESHOLD) {
    diagnosis.push("");
    diagnosis.push(`## ⚠ STALL — ${state.stalledCycles} cycles without meaningful improvement. STUDY THE REFERENCES.`);
    diagnosis.push("");
    diagnosis.push("Guessing is not working. Before making ANY more changes, you MUST study how");
    diagnosis.push("production Metal inference engines solve this exact problem:");
    diagnosis.push("");
    const llamaMetal = existsSync("/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal")
      ? "/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal"
      : "/tmp/llama.cpp/ggml/src/ggml-metal";
    const vllmMoe = existsSync("/Users/zolotukhin/Workplace/vllm/vllm/model_executor/layers/fused_moe")
      ? "/Users/zolotukhin/Workplace/vllm/vllm/model_executor/layers/fused_moe"
      : "/tmp/vllm/vllm/model_executor/layers/fused_moe";
    diagnosis.push("### Step 1: Read llama.cpp Metal backend");
    if (llamaMetal.startsWith("/tmp/")) {
      diagnosis.push("```bash");
      diagnosis.push("git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp");
      diagnosis.push("```");
    }
    diagnosis.push("Read these files if they exist in the local checkout:");
    diagnosis.push(`- \`${llamaMetal}/ggml-metal-context.m\` — graph command-buffer scheduling and commit/wait policy`);
    diagnosis.push(`- \`${llamaMetal}/ggml-metal-ops.cpp\` — op-level encoder barriers, fusion, and mul_mat_id dispatch`);
    diagnosis.push(`- \`${llamaMetal}/ggml-metal.metal\` — Q8/Q4 matvec and routed matmul kernels`);
    diagnosis.push("Some llama.cpp checkouts no longer have `ggml-metal.m`; do not waste a cycle rediscovering that rename.");
    diagnosis.push("After reading the current files once, cite the specific function you are adapting and move to a measured code change.");
    diagnosis.push("- Look at how they batch command buffers, manage encoders, and choose per-shape Q8 paths");
    diagnosis.push("- Note how many commitAndWait calls happen per token (likely 1)");
    diagnosis.push("");
    diagnosis.push("### Step 2: Read vLLM MoE packing");
    if (vllmMoe.startsWith("/tmp/")) {
      diagnosis.push("```bash");
      diagnosis.push("git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm");
      diagnosis.push("```");
    }
    diagnosis.push("Read these files:");
    diagnosis.push(`- \`${vllmMoe}\` — topk -> align/pack -> grouped expert flow`);
    diagnosis.push("- Look at which ideas require many prompt tokens; do not force those into single-token decode");
    diagnosis.push("");
    diagnosis.push("### Step 3: Apply what you learned");
    diagnosis.push("Identify the SPECIFIC technique from llama.cpp or vLLM that addresses our bottleneck,");
    diagnosis.push("then implement it. Cite which file/function you're adapting from in @@@DESCRIPTION.");
    diagnosis.push("Do NOT repeat variations of previously failed approaches.");
    diagnosis.push("If the local Codex subprocess cannot initialize Metal, do not spend the cycle retrying direct `./zig-out/bin/zinc` or Metal microbenchmarks; the outer harness owns the Metal measurement gate.");
  } else if (state.stalledCycles >= 3 && !recentProgress.hasProgress) {
    diagnosis.push("");
    diagnosis.push(`## Note: ${state.stalledCycles}/${STALL_THRESHOLD} cycles without improvement — will switch to reference study soon`);
  }

  // Reflection summary from recent cycles
  const reflectionSummary = cycles.length >= 5
    ? buildReflectionSummary({ cycles: cycles as any })
    : null;

  const phaseLabel = phase === "fix" ? "FIX" : phase === "implement" ? "IMPLEMENT" : "OPTIMIZE";

  // For optimize phase, use a focused prompt
  const isOptimize = phase === "optimize" || (lastResult.strongAnswer && lastResult.tokPerSec != null);

  const sections: string[] = [
    `# ZINC Metal ${phaseLabel} Task`,
    "",
  ];

  // When the loop is driven by a `--effort N` doc, inline its full text near
  // the top of the prompt. This gives the agent the analysis, benchmark
  // contract, and step ordering from the plan — the loop still owns cycle
  // history, diagnostics, and the build/test/run gate.
  if (state.effortPlan && state.effortPlan.trim().length > 0) {
    sections.push(
      `## Current Effort Plan (${state.effortFile ?? `effort ${state.effortId}`})`,
      "",
      "You are executing the multi-hour plan below. Pick the next",
      "unfinished step from the plan's Execution Order, implement ONE",
      "focused change for that step, and let the loop measure the",
      "result. Do not redo steps already completed in Cycle History.",
      "",
      "```markdown",
      state.effortPlan.trim(),
      "```",
      "",
    );
  }

  const gemmaRun = isGemmaRun(state);
  const modelContext = gemmaRun ? [
    "## Model (Gemma 4 26B-A4B MoE Q4_K_M)",
    "- 30 layers, all current profile steps are attention + Gemma MoE (`mix/step: attn 30.0 gpu-moe 30.0`).",
    "- hidden_dim=2816, n_heads=16, n_kv_heads=8, vocab=262144.",
    "- MoE FFN: 128 experts, 8 active per token, intermediate=704, shared expert=2112.",
    "- Hot request profile after cycle 49: q8_0 52.56 GiB, q4_k 13.96 GiB, q5_1 9.31 GiB.",
    "- Hot path bytes: attn 31.16 GiB, moe-expert 23.26 GiB, shared 14.83 GiB, lm-head 6.57 GiB.",
    "",
  ] : [
    "## Model (Qwen3.6-35B-A3B, Q4_K, 20.8 GB)",
    "- 40 layers: every 4th is full attention (layers 3,7,11,...,39), rest are SSM/delta-net.",
    "- MoE FFN: 256 experts, 8 active per token, + shared expert.",
    "- head_dim=256, hidden_dim=2048, n_heads=16, n_kv_heads=2.",
    "- Active parameters per token: ~3B (due to MoE sparsity).",
    "- Effective working set per decode step: ~1.7 GB at Q4_K.",
    "",
  ];

  sections.push(
    ...diagnosis,
    "",
    "## Hardware",
    "- Mac Studio M4 Max, 64 GB unified memory, 40-core GPU, 546 GB/s bandwidth",
    "- Apple GPU family: Apple9 (M4), simdgroup_matrix = true, bfloat = true",
    "- macOS, Metal compute only",
    "",
    ...modelContext,
  );

  if (isOptimize) {
    // Optimization-specific sections
    sections.push(
      "## Bandwidth Analysis",
      "- Memory BW: 546 GB/s theoretical, ~480 GB/s achievable",
      ...(gemmaRun ? [
        "- Gemma 26B current public-suite baseline is about 30 tok/s decode and 32-34 tok/s prefill on M4.",
        "- Treat broad Q8 threadgroup/repack work as low-probability unless an exact-shape benchmark proves the candidate wins first.",
        "- The highest-leverage work is Gemma MoE route coverage, grouped expert execution, and prefill/decode isolation per Effort 11.",
      ] : [
        "- Working set per token: ~1.7 GB (only active experts + attention layers)",
        "- Theoretical BW-limited decode: ~280 tok/s (480 / 1.7)",
        `- Current: ${lastResult.tokPerSec?.toFixed(1)} tok/s → ${((lastResult.tokPerSec ?? 0) / 280 * 100).toFixed(0)}% of theoretical BW limit`,
        "- This means MOST time is lost to dispatch overhead, sync, or compute bottlenecks — NOT bandwidth",
      ]),
      "",
      "## Baseline Reference",
      gemmaRun ? "- Use the loaded Effort 11 baseline and public-suite numbers; do not cite obsolete cycle-49 numbers as the current Gemma target." : "- llama.cpp Metal on this machine: 72.93 tok/s decode (tg128)",
      `- ZINC target: ≥${TARGET_TOK_PER_SEC} ${METRIC_LABEL}`,
      `- ZINC current: ${lastResult.tokPerSec?.toFixed(1)} ${METRIC_LABEL}`,
      "",
      "## Optimization Targets (pick ONE per cycle)",
      "",
      "### 1. Reduce command buffer submissions",
      "Each commitAndWait() is a CPU-GPU sync point (~50-100μs overhead).",
      "Ideal: ONE command buffer submit per decode step. Batch all 40 layers into",
      "a single command buffer with barriers between dependent dispatches.",
      "Check how many commits happen per token in forward_metal.zig's decode loop.",
      "",
      "### 2. Minimize Metal encoder recreation",
      "mtl_barrier() creates a new compute command encoder. Each encoder switch costs ~10-30μs.",
      "Only barrier when there is a true data dependency. Adjacent dispatches to different",
      "buffers do NOT need a barrier.",
      "",
      "### 3. MoE expert dispatch batching",
      "With 8 active experts per token, if each expert is a separate dispatch, that's 8 small",
      "dispatches per layer × 30 MoE layers = 240 small dispatches. Each has launch overhead.",
      "Consider: batch multiple experts into one dispatch with offset indexing, or fuse gate+up.",
      "",
      "### 4. Threadgroup size tuning",
      "M4 Max: max 1024 threads per threadgroup, 32 SIMD width.",
      "DMMV shaders for Q4_K: check if threadgroup size matches the row count.",
      "Undersized threadgroups → low occupancy. Oversized → register pressure.",
      "",
      "### 5. Use half/bfloat for intermediates",
      "M4 has 2x throughput for bfloat16 vs float32 in compute.",
      "If intermediate buffers (hidden state, norm output) can use half precision,",
      "this halves bandwidth and doubles ALU throughput for those stages.",
      "",
      "### 6. Fused kernels",
      "RMSNorm + first DMMV could be fused to avoid writing norm_buf to memory.",
      "SwiGLU (gate * silu(up)) is another fusion candidate.",
      "Each fused kernel saves one global memory round-trip.",
      "",
      "### 7. Pipeline state object caching",
      "If getPipeline() does dictionary lookup per dispatch, cache the PSO pointers",
      "for hot paths (called 40× per token).",
      "",
    );
  } else {
    // Fix/implement sections (legacy path, kept for correctness regressions)
    sections.push(
      "## Project Structure",
      "```",
      "src/compute/forward_metal.zig — Metal inference engine (THE MAIN FILE)",
      "src/metal/   — shim.h, shim.m (ObjC C API), device.zig, buffer.zig, command.zig",
      "src/shaders/metal/ — MSL compute shaders (dmmv_q4k, flash_attn, rms_norm_mul, etc.)",
      "```",
      "",
      "## Key Reference: forward.zig (Vulkan version)",
      "The Vulkan `decodeStep()` at src/compute/forward.zig shows the exact layer dispatch",
      "sequence. Read it for the correct order of operations, tensor names, and dimensions.",
      "",
    );
  }

  sections.push(
    "## Key Files to Edit",
    "- src/compute/forward_metal.zig — decode loop, dispatch sequence, buffer management",
    "- src/metal/command.zig — command buffer management, barrier implementation",
    "- src/metal/shim.m — ObjC shim: mtl_dispatch, mtl_barrier, mtl_commit",
    "- src/shaders/metal/*.metal — shader source (threadgroup sizes, occupancy)",
    "",
  );

  // Profile output if available
  if (state.lastProfileOutput) {
    sections.push(
      `## Profile Output (cycle ${state.lastProfileCycle})`,
      "Use this to identify the actual hotspots. Focus optimization on the slowest phases.",
      "```",
      state.lastProfileOutput.slice(-3000),
      "```",
      "",
    );
  }

  if (state.lastMetalShapesOutput) {
    const metalShapesFailed = state.lastMetalShapesOk === false;
    sections.push(
      `## Metal Shapes Evidence (${metalShapesFailed ? "failed " : ""}cycle ${state.lastMetalShapesCycle})`,
      metalShapesFailed
        ? "Outer-harness exact-shape benchmark failed. Fix this evidence path or avoid relying on it before proposing another same-shape shader retune."
        : "Outer-harness exact-shape benchmark output. Agents must consume this before proposing another same-shape shader retune.",
      "```",
      state.lastMetalShapesOutput.slice(-3000),
      "```",
      "",
    );
  }

  // Build/test/run output
  if (lastResult.buildOutput) {
    sections.push("## Build Output (last 2000 chars)", "```", buildOut, "```", "");
  }
  if (lastResult.testOutput) {
    sections.push("## Test Output (last 2000 chars)", "```", testOut, "```", "");
  }
  if (lastResult.runOutput) {
    sections.push("## Run Output (last 3000 chars)", "```", runOut, "```", "");
  }

  // Reflection
  if (reflectionSummary) {
    sections.push("## Reflection (auto-analysis of recent cycles)", reflectionSummary, "");
  }

  // Self-review summaries from periodic reviews
  if (state.reviewSummaries && state.reviewSummaries.length > 0) {
    // Include the latest review
    sections.push(state.reviewSummaries[state.reviewSummaries.length - 1], "");
  }

  const qwen36PrefillFocus = buildQwen36PrefillFocus(state, lastResult);
  if (qwen36PrefillFocus.length > 0) {
    sections.push(...qwen36PrefillFocus);
  }

  const gemmaPrefillFocus = buildGemmaPrefillPostBreakthroughAnalysis(state);
  if (gemmaPrefillFocus.length > 0) {
    sections.push(...gemmaPrefillFocus);
  }

  sections.push(
    "## Cycle History",
    historyBlock,
    "",
    "## Failed Approaches (DO NOT repeat these)",
    failedBlock,
    "",
    "## Ideas",
    ideasBlock,
    "",
    "## Rules",
    "1. Make ONE focused change per cycle. Measure, don't guess.",
    `2. CORRECTNESS IS SACRED. Output MUST contain "${correctnessReference}". Speed without correctness = instant revert.`,
    "3. All 27+ tests must continue passing.",
    "4. Do NOT modify src/vulkan/, loops/, or .env.",
    "5. Do NOT run git push, git pull, git fetch, git merge, git rebase, git reset, git checkout, or git restore. The harness owns git commits/reverts.",
    "6. Zig 0.15.2 API: ArrayList is unmanaged (pass allocator to append/deinit).",
    "7. MSL shaders use 'main0' as entry point (SPIRV-Cross convention).",
    "8. Metal push constants go in buffer[n_bufs] (see shim.m mtl_dispatch).",
    "9. The Metal command pattern: beginCommand → dispatch → barrier → dispatch → commitAndWait.",
    "10. UMA advantage: all buffers are SharedMode — cpu_ptr gives direct CPU access to GPU data.",
    "11. Read the profile output and run output BEFORE deciding what to optimize.",
    "12. Prefer changes to forward_metal.zig and shaders. Avoid refactoring infrastructure.",
    "13. NEVER run the model yourself: not `./zig-out/bin/zinc ...`, not `zig build run/bench/bench-metal*/hot-bench`, not any command that loads the model on the GPU. ZINC reserves the Metal GPU exclusively, so your run would collide with the harness's own measurement and crash the cycle (\"GPU metal:0 is already reserved\"). The OUTER HARNESS owns all measurement, profiling, and validation runs. You may ONLY run `zig build` and `zig build test`. A \"RUNTIME CRASH\" reported below is the harness's measurement, not something you reproduce or fix by running the model.",
    "",
    "## Output Format",
    "After making your change, print these 4 lines:",
    "@@@DESCRIPTION: <one-line summary>",
    "@@@STEP_KIND: <optimization|enablement|analysis|fix|rollback>",
    "@@@SELF_ANALYSIS: <why this approach and what you expect, with estimated tok/s impact>",
    "@@@NEXT_IDEAS: <comma-separated ideas for future cycles>",
  );

  return sections.join("\n");
}

// ── State ────────────────────────────────────────────────────────────

export type CycleResult = {
  cycle: number;
  timestamp: string;
  phase: Phase;
  description: string;
  kept: boolean;
  tokPerSec: number | null;
  tokensGenerated: number;
  promptTokens?: number | null;
  promptTokenSamples?: number[];
  workload?: WorkloadContract | null;
  containsReference: boolean;
  buildExitCode: number;
  testExitCode: number;
  runExitCode: number | null;
  outputText: string;
  error?: string;
  stepKind?: StepKind;
  selfAnalysis: string;
  nextIdeas: string[];
  /// Accepted git commit after this cycle's keep commit. Null/absent for
  /// reverted cycles and older state files.
  commitHash?: string | null;
  /// True when this kept cycle promoted the all-time best throughput, not just
  /// a neutral/foundation keep.
  promotedBest?: boolean;
};

export type BestTree = {
  cycle: number;
  tokPerSec: number;
  commitHash: string;
};

/**
 * The fastest measurement the loop has ever seen that built, tested, ran, and
 * was REVERTED purely because the output was wrong. This is a standing
 * optimization target: it proves a speedup is reachable, so the prompt redirects
 * the agent at localizing the divergent layer rather than re-deriving the same
 * correctness-breaking idea from scratch (the route-packed ~109 tok/s dead end).
 */
export type NearMiss = {
  cycle: number;
  tokPerSec: number;
  gainPctOverAccepted: number;
  description: string;
  selfAnalysis: string;
  outputText: string;
};

export type RunState = {
  runId: string;
  cycles: CycleResult[];
  failedApproaches: string[];
  ideas: string[];
  phase: Phase;
  metricMode?: MetricMode | null;
  /// Locked measurement workload contract. A run's accepted performance
  /// baselines are comparable only while model, prompt mode, raw/prepared
  /// prompt hashes, prompt token count, max tokens, metric, and correctness
  /// reference remain stable.
  workload?: WorkloadContract | null;
  currentBest: { tokPerSec: number | null; containsReference: boolean } | null;
  stalledCycles: number;
  bestTokPerSec: number;
  /// Git commit for the fastest promoted-best tree. Used at loop end to restore
  /// the worktree if later neutral keeps left HEAD slower than the peak.
  bestTree?: BestTree | null;
  /// Fastest reverted-for-correctness measurement; see {@link NearMiss}.
  bestIncorrect?: NearMiss | null;
  lastProfileOutput: string | null;
  lastProfileCycle: number | null;
  /// Optional outer-harness exact-shape Metal benchmark output. This is
  /// separate from `lastProfileOutput`: profile measures whole-model request
  /// behavior, while Metal-shapes isolates hot kernels the agent is blocked
  /// from benchmarking directly.
  lastMetalShapesOutput?: string | null;
  lastMetalShapesCycle?: number | null;
  lastMetalShapesOk?: boolean | null;
  /// Most recent near-miss diagnostic run (see ZINC_NEAR_MISS_DIAGNOSTIC_ENV).
  /// Captured AFTER the keep verdict, runs the binary with extra env applied so
  /// validators / probes that the kept tree leaves default-off still fire and
  /// produce per-tensor/per-layer evidence for the next cycle.
  lastNearMissDiagnostic?: { cycle: number; envApplied: string[]; output: string } | null;
  /// Cross-effort guard: first measurement of the OTHER metric in this run,
  /// frozen as the regression baseline. Set on cycle 1 (or the first cross-
  /// effort measurement); all subsequent cross-effort tok/s are compared to it.
  crossEffortBaseline?: { metric: MetricMode; tokPerSec: number; cycle: number } | null;
  /// Latest cross-effort measurement + delta vs baseline. Surfaced in the next
  /// cycle's prompt so the agent can see whether their changes regressed the
  /// other metric — and react before the loop ends.
  lastCrossEffort?: { cycle: number; metric: MetricMode; tokPerSec: number; deltaPct: number } | null;
  reviewSummaries: string[];
  /// Optional multi-hour effort doc (raw markdown) spliced into every agent
  /// prompt. Loaded via `--effort N`, which finds `MULTI_HOUR_EFFORT_N_*.md`
  /// in `loops/efforts`. Null means run in the stock FIX/IMPLEMENT/OPTIMIZE mode.
  effortPlan?: string | null;
  effortId?: number | null;
  effortFile?: string | null;
};

async function loadState(runDir: string): Promise<RunState | null> {
  const p = join(runDir, "state.json");
  if (!existsSync(p)) return null;
  return JSON.parse(await readFile(p, "utf8")) as RunState;
}

/**
 * Resolve `--effort N` to a `MULTI_HOUR_EFFORT_N_*.md` filename in
 * `loops/efforts`. Returns `{ file, plan }` on success, null if no matching
 * doc exists.
 */
async function loadEffortPlan(effort: number): Promise<{ file: string; plan: string } | null> {
  const prefix = `MULTI_HOUR_EFFORT_${effort}_`;
  if (!existsSync(EFFORTS_DIR)) return null;
  const matches = readdirSync(EFFORTS_DIR).filter(
    (name) => name.startsWith(prefix) && name.endsWith(".md"),
  );
  if (matches.length === 0) return null;
  if (matches.length > 1) {
    console.error(
      clr("1;33", `  ⚠ Multiple effort docs match ${prefix}*.md: ${matches.join(", ")}. Using ${matches[0]}.`),
    );
  }
  const file = matches[0];
  const plan = await readFile(join(EFFORTS_DIR, file), "utf8");
  return { file, plan };
}

async function saveState(runDir: string, state: RunState): Promise<void> {
  await writeFile(join(runDir, "state.json"), JSON.stringify(state, null, 2));
}

function measuredWorkload(result: BuildRunResult): WorkloadContract | null {
  if (result.buildExitCode !== 0 || result.testExitCode !== 0) return null;
  if (result.runExitCode !== 0) return null;
  return result.workload ?? null;
}

function resetPerformanceBaselinesToMeasurement(state: RunState, result: BuildRunResult): void {
  state.currentBest = result.containsReference && result.tokPerSec != null
    ? { tokPerSec: result.tokPerSec, containsReference: true }
    : null;
  state.bestTokPerSec = result.containsReference && result.tokPerSec != null ? result.tokPerSec : 0;
  state.bestTree = null;
  state.bestIncorrect = null;
  state.stalledCycles = 0;
  state.crossEffortBaseline = null;
  state.lastCrossEffort = null;
}

async function syncWorkloadBeforeCycle(runDir: string, state: RunState, result: BuildRunResult): Promise<void> {
  const workload = measuredWorkload(result);
  if (!workload) return;

  if (!state.workload) {
    state.workload = workload;
    if (state.cycles.length > 0 && WORKLOAD_RESET_ON_CHANGE) {
      resetPerformanceBaselinesToMeasurement(state, result);
      console.log(clr("1;33", `  ◎ Workload contract locked for legacy state; reset performance baselines to ${result.tokPerSec?.toFixed(2) ?? "n/a"} ${METRIC_LABEL}`));
    } else {
      console.log(clr("1;36", `  ◎ Workload contract locked: prompt=${workload.promptTokens ?? "?"} tokens raw=${workload.rawPromptHash} prepared=${workload.preparedPromptHash ?? "unknown"}`));
    }
    await saveState(runDir, state);
    return;
  }

  const mismatch = workloadMismatchReason(state.workload, workload);
  if (!mismatch) return;

  if (!WORKLOAD_RESET_ON_CHANGE) {
    console.log(clr("1;31", `  ⚠ Workload changed but reset is disabled: ${mismatch}`));
    return;
  }

  state.workload = workload;
  resetPerformanceBaselinesToMeasurement(state, result);
  console.log(clr("1;33", `  ◎ Workload changed (${mismatch}); reset performance baselines to current measurement ${result.tokPerSec?.toFixed(2) ?? "n/a"} ${METRIC_LABEL}`));
  await saveState(runDir, state);
}

export function shouldFinalizeBestTree(
  state: Pick<RunState, "bestTree" | "currentBest">,
  currentHead: string,
): { finalize: boolean; reason: string } {
  const bestTree = state.bestTree;
  if (!bestTree || !bestTree.commitHash) return { finalize: false, reason: "no promoted-best commit recorded" };
  if (currentHead.trim() === bestTree.commitHash.trim()) return { finalize: false, reason: "already at promoted-best commit" };
  const currentTps = state.currentBest?.containsReference ? (state.currentBest.tokPerSec ?? 0) : 0;
  if (currentTps >= bestTree.tokPerSec - 0.05) {
    return { finalize: false, reason: "current tree is within 0.05 tok/s of promoted best" };
  }
  return {
    finalize: true,
    reason: `current tree ${currentTps.toFixed(2)} is below promoted-best cycle ${bestTree.cycle} at ${bestTree.tokPerSec.toFixed(2)}`,
  };
}

export function shouldRestorePromotedBestDuringPlateau(
  state: RunState,
  currentHead: string,
): { restore: boolean; reason: string } {
  const finalize = shouldFinalizeBestTree(state, currentHead);
  if (!finalize.finalize) return { restore: false, reason: finalize.reason };

  const cyclesSinceBest = cyclesSinceBestKept(state);
  if (
    isGemmaPrefillPostBreakthrough(state) &&
    (state.stalledCycles >= GEMMA_PLATEAU_STALL_CYCLES || cyclesSinceBest >= GEMMA_PLATEAU_STALL_CYCLES)
  ) {
    return {
      restore: true,
      reason: `Gemma plateau active (${state.stalledCycles} stalled, ${cyclesSinceBest} cycles since best); ${finalize.reason}`,
    };
  }

  const plateauStop = detectAutoStopForPlateau(state);
  if (plateauStop.stop) {
    return {
      restore: true,
      reason: `plateau auto-stop condition met (${plateauStop.reason}); ${finalize.reason}`,
    };
  }

  return { restore: false, reason: `plateau restore not active; ${finalize.reason}` };
}

export function parseRestorePathList(raw: string): string[] {
  return [...new Set(
    raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0),
  )];
}

async function loopCyclePathsChangedSince(commitHash: string): Promise<string[]> {
  const out = await runCommand("git", [
    "log",
    "--format=",
    "--name-only",
    "--grep",
    "^metal-loop: cycle-",
    `${commitHash}..HEAD`,
    "--",
    ...LOOP_COMMIT_PATHS,
  ]).catch(() => null);
  return parseRestorePathList(out?.stdout ?? "");
}

async function backfillBestTreeCommitFromGit(state: RunState): Promise<void> {
  const best = bestTreeCandidateCycle(state);
  if (!best || best.tokPerSec == null) return;
  if (
    state.bestTree?.commitHash &&
    state.bestTree.cycle >= best.cycle &&
    state.bestTree.tokPerSec >= best.tokPerSec - 0.05
  ) {
    const stillOnBranch = await runCommand("git", ["merge-base", "--is-ancestor", state.bestTree.commitHash, "HEAD"]).then(() => true).catch(() => false);
    if (stillOnBranch) return;
  }
  const grep = `metal-loop: cycle-${best.cycle} `;
  const found = await runCommand("git", ["log", "--format=%H", "--grep", grep, "-1"]).catch(() => null);
  const commitHash = found?.stdout.trim() || best.commitHash;
  if (commitHash) {
    state.bestTree = { cycle: best.cycle, tokPerSec: best.tokPerSec, commitHash };
  }
}

export function applyBestTreeRestoreCommit(
  state: Pick<RunState, "bestTree">,
  commitHash: string | null | undefined,
): void {
  if (!state.bestTree?.commitHash || !commitHash) return;
  state.bestTree = { ...state.bestTree, commitHash };
}

async function restoreBestTree(runDir: string, state: RunState, reason: string): Promise<void> {
  if (!state.bestTree?.commitHash) return;

  const best = state.bestTree;
  console.log(clr("1;35", `  ↩ Restoring promoted-best tree from cycle ${best.cycle} (${best.tokPerSec.toFixed(2)} ${METRIC_LABEL}): ${reason}`));
  const restorePaths = await loopCyclePathsChangedSince(best.commitHash);
  if (restorePaths.length === 0) {
    console.log(clr("2", "  best-tree restore skipped: no post-best loop cycle paths changed"));
    state.currentBest = { tokPerSec: best.tokPerSec, containsReference: true };
    await saveState(runDir, state);
    return;
  }
  console.log(clr("2", `  restoring ${restorePaths.length} loop-touched path(s): ${restorePaths.join(", ")}`));
  await runCommand("git", ["restore", "--source", best.commitHash, "--staged", "--worktree", "--", ...restorePaths]);
  const status = await runCommand("git", ["status", "--porcelain", "--", ...restorePaths]);
  if (status.stdout.trim().length > 0) {
    await runCommand("git", ["add", "-A", ...restorePaths]).catch(() => {});
    const committed = await runCommand("git", ["commit", "-m", `metal-loop: finalize best tree from cycle ${best.cycle} (${best.tokPerSec.toFixed(1)} ${METRIC_LABEL})`]).catch(() => null);
    if (committed) {
      const head = await runCommand("git", ["rev-parse", "HEAD"]).catch(() => null);
      applyBestTreeRestoreCommit(state, head?.stdout.trim());
    }
  }
  state.currentBest = { tokPerSec: best.tokPerSec, containsReference: true };
  await saveState(runDir, state);
}

async function finalizeBestTreeIfNeeded(runDir: string, state: RunState): Promise<void> {
  if (!FINALIZE_BEST_TREE) return;
  await backfillBestTreeCommitFromGit(state);
  if (!state.bestTree?.commitHash) return;
  const head = await runCommand("git", ["rev-parse", "HEAD"]).catch(() => null);
  const currentHead = head?.stdout.trim() ?? "";
  const decision = shouldFinalizeBestTree(state, currentHead);
  if (!decision.finalize) {
    console.log(clr("2", `  best-tree finalize skipped: ${decision.reason}`));
    return;
  }
  await restoreBestTree(runDir, state, decision.reason);
}

async function restorePromotedBestDuringPlateauIfNeeded(runDir: string, state: RunState): Promise<void> {
  if (!FINALIZE_BEST_TREE) return;
  await backfillBestTreeCommitFromGit(state);
  if (!state.bestTree?.commitHash) return;
  const head = await runCommand("git", ["rev-parse", "HEAD"]).catch(() => null);
  const currentHead = head?.stdout.trim() ?? "";
  const decision = shouldRestorePromotedBestDuringPlateau(state, currentHead);
  if (!decision.restore) return;
  await restoreBestTree(runDir, state, decision.reason);
}

async function runProfileBenchmark(): Promise<string> {
  console.log(clr("1;33", "  📊 Profiling run (--profile)..."));
  const run = await runCommand(
    "./zig-out/bin/zinc",
    [...zincModelArgs(), ...zincPromptArgs(), "-n", String(Math.min(MAX_TOKENS, 32)), "--profile"],
    { timeout: RUN_TIMEOUT_MS },
  );
  const combined = (run.stderr + run.stdout).slice(-4000);
  console.log(clr("2", "    profile captured"));
  return combined;
}

function splitShellWords(raw: string): string[] {
  return raw.trim().split(/\s+/).filter((s) => s.length > 0);
}

async function runMetalShapesBenchmark(state?: RunState): Promise<string> {
  const modelPath = managedModelPathForBenchmark();
  if (!existsSync(modelPath)) {
    throw new Error(`managed model path for bench-metal-shapes not found: ${modelPath}`);
  }

  const inferredRouteTokens = inferGemmaRouteTokensForMetalShapes(state?.lastProfileOutput);
  const extraArgs = METAL_SHAPES_ARGS_RAW.trim().length > 0
    ? splitShellWords(METAL_SHAPES_ARGS_RAW)
    : [
        "--case", "gemma26_prefill_hot",
        "--pipeline", "production",
        "--route-tokens", String(inferredRouteTokens ?? Math.max(1, MAX_TOKENS + 4)),
        "--iterations", "80",
        "--warmup", "10",
      ];

  console.log(clr("1;35", `  🔬 Metal-shapes evidence run (${extraArgs.join(" ")})...`));
  const run = await runCommand(
    "zig",
    ["build", "bench-metal-shapes", "--", "-m", modelPath, ...extraArgs],
    { timeout: RUN_TIMEOUT_MS },
  );
  const combined = (run.stderr + run.stdout).slice(-6000);
  if (run.exitCode !== 0) {
    throw new Error(combined);
  }
  console.log(clr("2", `    metal-shapes captured (${combined.length} bytes)`));
  return combined;
}

function evidenceIntentText(args: {
  description: string;
  selfAnalysis: string;
  ideas: string[];
}): string {
  return [
    args.description,
    args.selfAnalysis,
    ...args.ideas,
  ].join("\n").toLowerCase();
}

function explicitlyRequestsMetalShapesEvidence(text: string): boolean {
  return /\b(bench-metal-shapes|metal-shapes|gemma26_prefill_hot|microbench|shared_gate_gemm|shared_up_gemm|shared_down_gemm|moe_down_cols|moe_gate_up_geglu|exact-shape)\b/.test(text);
}

function explicitlyRequestsProfileEvidence(text: string): boolean {
  return /\b(--profile|profile-only|profile counter|profile counters|profile line|profile lines|profile evidence|dispatch counter|dispatch counters|barrier counter|barrier counters|timing counter|token-tile accounting|q8 gemm #)\b/.test(text);
}

export type CandidateEvidenceDecision = {
  profile: boolean;
  metalShapes: boolean;
};

export function shouldRunCandidateEvidence(args: {
  state: RunState;
  cycle: number;
  containsReference: boolean;
  buildExitCode: number;
  testExitCode: number;
  runExitCode: number | null;
  stepKind: StepKind;
  description: string;
  selfAnalysis: string;
  ideas: string[];
}, profileEvery: number = PROFILE_EVERY, metalShapesEvery: number = METAL_SHAPES_EVERY): CandidateEvidenceDecision {
  const none = { profile: false, metalShapes: false };
  if (!args.containsReference) return none;
  if (args.buildExitCode !== 0 || args.testExitCode !== 0 || args.runExitCode !== 0) return none;

  const text = evidenceIntentText(args);
  const explicitProfileRequest = explicitlyRequestsProfileEvidence(text);
  const explicitShapesRequest = explicitlyRequestsMetalShapesEvidence(text);
  const evidenceStep = args.stepKind === "analysis" || args.stepKind === "enablement" ||
    explicitProfileRequest || explicitShapesRequest;
  if (!evidenceStep) return none;

  return {
    profile: explicitProfileRequest ||
      (profileEvery > 0 && (args.cycle === 1 || args.cycle % profileEvery === 0)),
    metalShapes: shouldRunMetalShapesEvidence({
      state: args.state,
      cycle: args.cycle,
      kept: false,
      containsReference: true,
      stepKind: args.stepKind,
      description: args.description,
      selfAnalysis: args.selfAnalysis,
      ideas: args.ideas,
    }, metalShapesEvery),
  };
}

export function shouldRunMetalShapesEvidence(args: {
  state: RunState;
  cycle: number;
  kept: boolean;
  containsReference: boolean;
  stepKind: StepKind;
  description: string;
  selfAnalysis: string;
  ideas: string[];
}, metalShapesEvery: number = METAL_SHAPES_EVERY): boolean {
  if (metalShapesEvery <= 0) return false;
  if (!args.containsReference) return false;

  const text = evidenceIntentText(args);
  const explicitShapesRequest = explicitlyRequestsMetalShapesEvidence(text);
  const evidenceStep = args.stepKind === "analysis" || args.stepKind === "enablement" || explicitShapesRequest;
  if (!args.kept && !evidenceStep) return false;

  if (
    gemmaPrefillActualPathIsQueuedOffPath(args.state.lastProfileOutput) &&
    (args.state.lastMetalShapesOutput ?? "").trim().length > 0 &&
    !explicitShapesRequest
  ) {
    return false;
  }

  if (args.cycle % metalShapesEvery === 0) {
    return args.kept || evidenceStep;
  }

  return isGemmaPrefillPostBreakthrough(args.state) &&
    (args.state.stalledCycles >= GEMMA_PLATEAU_STALL_CYCLES ||
      cyclesSinceBestKept(args.state) >= GEMMA_PLATEAU_STALL_CYCLES ||
      explicitShapesRequest);
}

/**
 * Parse the ZINC_NEAR_MISS_DIAGNOSTIC_ENV string ("KEY1=VALUE1 KEY2=VALUE2") into
 * a {KEY: VALUE} record. Empty / malformed pairs are skipped. Exported for tests.
 */
export function parseDiagnosticEnv(raw: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const pair of raw.split(/\s+/).filter((s) => s.length > 0)) {
    const eq = pair.indexOf("=");
    if (eq <= 0) continue;
    const key = pair.slice(0, eq).trim();
    const value = pair.slice(eq + 1).trim();
    if (key && value) out[key] = value;
  }
  return out;
}

/**
 * Run the model once with extra env vars applied (typically to flip on a
 * default-off validator) and return the truncated combined output. The result
 * is NOT scored against the keep gate — it exists only to surface evidence in
 * the next cycle's prompt. Safe to call on the kept tree because the diagnostic
 * env never persists past this subprocess.
 */
async function runNearMissDiagnostic(
  maxTokens: number,
  extraEnv: Record<string, string>,
): Promise<string> {
  const keys = Object.keys(extraEnv);
  console.log(clr("1;35", `  🔬 Near-miss diagnostic run (env: ${keys.join(", ")})...`));
  const run = await runCommand(
    "./zig-out/bin/zinc",
    [...zincModelArgs(), ...zincPromptArgs(), "-n", String(Math.min(maxTokens, 32))],
    { timeout: RUN_TIMEOUT_MS, env: extraEnv },
  );
  const combined = (run.stderr + run.stdout).slice(-4000);
  console.log(clr("2", `    diagnostic captured (${combined.length} bytes)`));
  return combined;
}

/**
 * Cross-effort guard: run one inference using the OTHER metric/prompt mode
 * (e.g. prefill during a decode-focused run) and return the parsed tok/s for
 * that metric. Used to surface cross-effort regression EARLY without affecting
 * the main keep gate.
 */
async function runCrossEffortCheck(): Promise<number | null> {
  if (!CROSS_EFFORT_PROMPT) return null;
  console.log(clr("1;35", `  🔬 Cross-effort check: measuring ${CROSS_EFFORT_METRIC} with ${CROSS_EFFORT_PROMPT_MODE} prompt mode...`));
  const args = [
    ...zincModelArgs(),
    "--prompt", CROSS_EFFORT_PROMPT,
    ...(CROSS_EFFORT_PROMPT_MODE === "chat" ? ["--chat"] : []),
    "-n", String(CROSS_EFFORT_MAX_TOKENS),
  ];
  const run = await runCommand("./zig-out/bin/zinc", args, { timeout: RUN_TIMEOUT_MS });
  if (run.exitCode !== 0) {
    console.log(clr("1;33", "    ⚠ Cross-effort check crashed; skipping"));
    return null;
  }
  const combined = run.stderr + run.stdout;
  const tps = parseTokPerSec(combined, CROSS_EFFORT_METRIC);
  if (tps == null) {
    console.log(clr("1;33", "    ⚠ Cross-effort check produced no parseable tok/s; skipping"));
    return null;
  }
  console.log(clr("2", `    cross-effort ${CROSS_EFFORT_METRIC}: ${tps.toFixed(2)} tok/s`));
  return tps;
}

/**
 * Build the cross-effort status section spliced into every prompt while a
 * cross-effort baseline is set. Highlights regression vs the baseline so the
 * agent can react before the run ends with a silent cross-effort loss.
 */
export function buildCrossEffortStatus(state: Pick<RunState, "crossEffortBaseline" | "lastCrossEffort">): string[] {
  const last = state.lastCrossEffort;
  const base = state.crossEffortBaseline;
  if (!last || !base) return [];
  const lines: string[] = [];
  // Thresholds tightened to match the Effort 5/16 docs' explicit "no cross-
  // effort regression >3%" rule. Decode v4 had drift at -3 to -4% for 6 cycles
  // but the agent never acknowledged it in selfAnalysis — DRIFT at -2% was
  // too soft. REGRESSION at -3% now triggers forceful "gate the suspect fusion"
  // guidance instead of just "watch for further drift."
  const severe = last.deltaPct <= -3;
  const moderate = last.deltaPct <= -1.5 && !severe;
  const tag = severe
    ? "## ⚠⚠ CROSS-EFFORT REGRESSION — your changes are HURTING the other metric"
    : moderate
      ? "## ⚠ CROSS-EFFORT DRIFT — the other metric is sliding"
      : "## Cross-effort status";
  lines.push(tag);
  lines.push(
    `Other metric: **${last.metric}** measured ${last.tokPerSec.toFixed(2)} tok/s at cycle ${last.cycle} (baseline ${base.tokPerSec.toFixed(2)} from cycle ${base.cycle}, Δ ${last.deltaPct >= 0 ? "+" : ""}${last.deltaPct.toFixed(1)}%).`,
  );
  if (severe) {
    lines.push("");
    lines.push(`A change earlier in this run regressed ${last.metric} by ${Math.abs(last.deltaPct).toFixed(1)}%. The Effort 5/16 docs explicitly prohibit cross-effort regressions >3%. Before another ${METRIC_MODE}-targeting change:`);
    lines.push("- Read the Cycle History below and identify which KEPT cycle(s) likely touched shared kernels (SSM, Q8 dispatch, fused-norm, RoPE/KV).");
    lines.push(`- If a single recent fusion is the suspect, GATE it behind an env flag rather than landing another optimization on top. That preserves the ${METRIC_MODE} win while restoring ${last.metric}.`);
    lines.push("- Do not roll out another shader fusion this cycle if its hot path is also used by " + last.metric + " — instead, propose a default-off variant.");
  } else if (moderate) {
    lines.push("Watch for further drift. If the next cross-effort check goes below -5%, the harness will treat this as a hard regression.");
  }
  return lines;
}

const REVIEW_EVERY = 10;

export function buildSelfReview(state: RunState): string {
  const recent = state.cycles.slice(-REVIEW_EVERY);
  if (recent.length === 0) return "";

  const kept = recent.filter(c => c.kept);
  const tpsValues = kept.filter(c => c.tokPerSec != null).map(c => c.tokPerSec!);
  const rawTpsStart = recent[0].tokPerSec ?? 0;
  const rawTpsEnd = recent[recent.length - 1].tokPerSec ?? rawTpsStart;
  const rawDelta = rawTpsEnd - rawTpsStart;
  const priorKeptTps = state.cycles
    .slice(0, -recent.length)
    .filter(c => c.kept && c.tokPerSec != null)
    .map(c => c.tokPerSec!);
  const acceptedStart = priorKeptTps.length > 0
    ? Math.max(...priorKeptTps)
    : (tpsValues.length > 0 ? tpsValues[0] : rawTpsStart);
  const acceptedEnd = tpsValues.length > 0
    ? Math.max(acceptedStart, ...tpsValues)
    : acceptedStart;
  const acceptedDelta = acceptedEnd - acceptedStart;
  const smallProgressThreshold = smallAcceptedProgressBand(acceptedStart);

  const lines: string[] = [
    `## Self-Review (last ${recent.length} cycles)`,
    "",
    `- Kept: ${kept.length}/${recent.length} changes`,
    `- Accepted best movement: ${acceptedStart.toFixed(1)} → ${acceptedEnd.toFixed(1)} (${acceptedDelta >= 0 ? "+" : ""}${acceptedDelta.toFixed(1)})`,
    `- Raw measured movement, including reverted candidates: ${rawTpsStart.toFixed(1)} → ${rawTpsEnd.toFixed(1)} (${rawDelta >= 0 ? "+" : ""}${rawDelta.toFixed(1)})`,
    `- Best tok/s in window: ${tpsValues.length > 0 ? Math.max(...tpsValues).toFixed(1) : "N/A"}`,
    "",
    "### What's working vs not:",
  ];

  for (const [cat, stats] of categoryCounts(recent).sort((a, b) => b[1].kept - a[1].kept)) {
    const total = stats.kept + stats.reverted;
    const rate = ((stats.kept / total) * 100).toFixed(0);
    const indicator = stats.kept > stats.reverted ? "✅" : stats.kept === 0 ? "❌" : "⚠";
    lines.push(`  ${indicator} ${cat}: ${stats.kept}/${total} kept (${rate}% success)`);
  }

  lines.push("");
  if (kept.length === 0) {
    lines.push("### ⚠ No accepted progress — strategic pivot required:");
    lines.push("- Do NOT treat faster reverted candidates as progress");
    lines.push("- Stop doubling down on categories with 0 kept changes");
    lines.push("- Build missing measurement/microbench coverage before another kernel retune");
  } else if (acceptedDelta < smallProgressThreshold) {
    lines.push("### ⚠ Low progress in accepted changes — strategic pivot recommended:");
    lines.push("- STOP trying small variations of what already failed");
    lines.push("- Focus on categories with >50% success rate above");
    lines.push("- If no category is working, the bottleneck is elsewhere — profile first");
  } else if (acceptedDelta < 1) {
    lines.push(`### Small accepted progress (+${acceptedDelta.toFixed(1)} tok/s). Use the evidence from kept measurement/validator cycles before adding another probe.`);
  } else {
    lines.push(`### Progress is positive in accepted changes (+${acceptedDelta.toFixed(1)} tok/s). Double down only on kept categories.`);
  }

  // Most impactful kept changes
  const impactful = kept
    .filter(c => c.tokPerSec != null)
    .sort((a, b) => (b.tokPerSec ?? 0) - (a.tokPerSec ?? 0))
    .slice(0, 3);
  if (impactful.length > 0) {
    lines.push("");
    lines.push("### Top performing changes:");
    for (const c of impactful) {
      lines.push(`  - ${c.description} → ${c.tokPerSec?.toFixed(1)} tok/s`);
    }
  }

  return lines.join("\n");
}

function extractAgentText(stdout: string): string {
  const lines = stdout.split("\n");
  const texts: string[] = [];
  for (const line of lines) {
    try {
      const evt = JSON.parse(line);
      if (evt?.type === "assistant") {
        const content = evt?.message?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (block?.type === "text" && typeof block.text === "string") {
              texts.push(block.text);
            }
          }
        }
      }
      if (evt?.type === "item.completed" && evt?.item?.type === "agent_message") {
        const text = coerceDisplayText(
          evt.item.text ?? evt.item.message ?? evt.item.output_text ?? evt.item.content,
        );
        if (text.trim()) texts.push(text);
      }
    } catch { /* not JSON */ }
  }
  return texts.join("\n");
}

function inferStepKind(description: string, selfAnalysis: string): StepKind {
  const text = `${description}\n${selfAnalysis}`.toLowerCase();
  if (/\b(rollback|revert|measured[- ]dead)\b/.test(text)) return "rollback";
  if (/\b(fix|build failure|test failure|correctness regression|crash)\b/.test(text)) return "fix";
  if (/\b(analysis|profile|measure|microbench|counter|diagnos|validator|diff|instrument)\b/.test(text)) return "analysis";
  if (/\b(enablement|plumbing|infrastructure|harness|scaffold|guard|sweep|coherence)\b/.test(text)) return "enablement";
  return "optimization";
}

function parseStepKind(raw: string | undefined, description: string, selfAnalysis: string): StepKind {
  const normalized = raw?.trim().toLowerCase();
  if (
    normalized === "optimization" ||
    normalized === "enablement" ||
    normalized === "analysis" ||
    normalized === "fix" ||
    normalized === "rollback"
  ) {
    return normalized;
  }
  return inferStepKind(description, selfAnalysis);
}

export function shouldRejectPlateauNeutralKeep(args: {
  state: RunState;
  stepKind: StepKind;
  description: string;
  selfAnalysis: string;
  verifyTokPerSec: number;
  acceptedTokPerSec: number;
  currentProgressBand: number;
}): boolean {
  if (args.verifyTokPerSec >= args.acceptedTokPerSec + args.currentProgressBand) return false;
  if (args.stepKind !== "optimization") return false;

  const text = `${args.description}\n${args.selfAnalysis}`.toLowerCase();
  if (/\b(profile|measure|microbench|counter|diagnos|validator|validate|diff|instrument|coherence|harness|bench-metal-shapes|metal-shapes|exact-shape)\b/.test(text)) {
    return false;
  }

  if (isQwen36PrefillRun(args.state)) {
    return args.state.stalledCycles >= QWEN36_PLATEAU_STALL_CYCLES;
  }

  if (isGemmaPrefillPostBreakthrough(args.state)) {
    return args.state.stalledCycles >= GEMMA_PLATEAU_STALL_CYCLES ||
      cyclesSinceBestKept(args.state) >= GEMMA_PLATEAU_STALL_CYCLES;
  }

  return false;
}

export function shouldRejectQwen36PlateauNeutralKeep(args: {
  state: RunState;
  stepKind: StepKind;
  description: string;
  selfAnalysis: string;
  verifyTokPerSec: number;
  acceptedTokPerSec: number;
  currentProgressBand: number;
}): boolean {
  return shouldRejectPlateauNeutralKeep(args);
}

function shouldPreservePromotedBestStallPressure(
  state: RunState,
  verifyTokPerSec: number,
  bestTokPerSec: number,
): boolean {
  return isGemmaPrefillPostBreakthrough(state) && verifyTokPerSec < bestTokPerSec;
}

// ── Main loop ────────────────────────────────────────────────────────

function findLatestRunDir(): string | null {
  if (!existsSync(RESULTS_DIR)) return null;
  const entries = readdirSync(RESULTS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name)
    .sort()
    .reverse();
  for (const entry of entries) {
    const stateFile = join(RESULTS_DIR, entry, "state.json");
    if (existsSync(stateFile)) return join(RESULTS_DIR, entry);
  }
  return null;
}

/**
 * Move existing run history aside instead of deleting it. A bare (non-`--resume`)
 * launch used to `rmSync` the whole `.metal_optimize/` directory — that wiped a
 * 340-cycle Effort-16 run that took ~12h to produce because the operator
 * forgot `--resume`. Archiving makes a bare launch recoverable: the prior
 * `state.json` survives under `.metal_optimize.archive/<timestamp>/<run-id>/`
 * and can be moved back if needed. Cheap rename, no copy.
 */
function cleanupOldRuns(): void {
  if (!existsSync(RESULTS_DIR)) return;
  // Fast-path: directory is empty — nothing to archive.
  let entries: string[] = [];
  try {
    entries = readdirSync(RESULTS_DIR);
  } catch {
    // If we cannot read it, fall back to the old destructive behaviour so a
    // corrupt directory cannot wedge the launch indefinitely.
    rmSync(RESULTS_DIR, { recursive: true, force: true });
    return;
  }
  if (entries.length === 0) {
    rmSync(RESULTS_DIR, { recursive: true, force: true });
    return;
  }
  const archiveRoot = `${RESULTS_DIR}.archive`;
  mkdirSync(archiveRoot, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const dest = join(archiveRoot, stamp);
  renameSync(RESULTS_DIR, dest);
  console.log(clr("2", `  Archived old runs (${entries.length} dir(s)) to ${dest}`));
}

async function main() {
  const args = process.argv.slice(2);
  let maxCycles = 999;
  let dryRun = false;
  let agent: AgentKind = "codex";
  let model: string | undefined;
  let resume = false;
  let effort: number | null = null;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--agent": {
        const value = args[++i];
        if (value !== "claude" && value !== "codex") {
          console.error(`Invalid --agent: ${value}. Use claude or codex.`);
          process.exit(1);
        }
        agent = value;
        break;
      }
      case "--model":
        model = args[++i];
        break;
      case "--cycles":
        maxCycles = parseInt(args[++i] ?? "999", 10);
        break;
      case "--dry-run":
        dryRun = true;
        break;
      case "--resume":
        resume = true;
        break;
      case "--effort": {
        const parsed = parseInt(args[++i] ?? "", 10);
        if (!Number.isFinite(parsed) || parsed <= 0) {
          console.error("Invalid --effort value; expected a positive integer.");
          process.exit(1);
        }
        effort = parsed;
        break;
      }
      case "--help":
        console.log([
          "Usage: bun loops/implement_metal.ts [options]",
          "",
          "Options:",
          "  --agent <claude|codex>  Agent to use (default: codex)",
          "  --model <name>          Model override for selected agent",
          "  --cycles N              Max cycles (default: 999)",
          "  --dry-run               Build+run only, no agent",
          "  --resume                Resume the most recent run",
          "  --effort N              Load MULTI_HOUR_EFFORT_N_*.md from loops/efforts",
          "                          and splice it into every agent prompt",
        ].join("\n"));
        process.exit(0);
    }
  }

  // Resolve the effort plan once so missing-file errors surface before the
  // build loop starts. Kept null in resume mode when no --effort is passed so
  // the saved state's effortPlan wins.
  let effortBundle: { file: string; plan: string } | null = null;
  if (effort != null) {
    effortBundle = await loadEffortPlan(effort);
    if (!effortBundle) {
      console.error(
        clr(
          "1;31",
          `No MULTI_HOUR_EFFORT_${effort}_*.md found in ${EFFORTS_DIR}. ` +
            "Create one or drop the --effort flag.",
        ),
      );
      process.exit(1);
    }
    console.log(
      clr("1;36", `  Effort ${effort}: loaded ${effortBundle.file} (${effortBundle.plan.length} chars)`),
    );
  }

  const agentLabel = agent === "codex" ? "Codex" : "Claude";

  // Resume or fresh start
  let runId: string;
  let runDir: string;
  let state: RunState;
  let startCycle: number;

  if (resume) {
    const latestDir = findLatestRunDir();
    if (!latestDir) {
      console.error("No previous run found to resume.");
      process.exit(1);
    }
    const loaded = await loadState(latestDir);
    if (!loaded) {
      console.error(`No state.json in ${latestDir}`);
      process.exit(1);
    }
    state = loaded;
    // Backfill fields that may not exist in older state files
    state.reviewSummaries ??= [];
    state.metricMode ??= METRIC_MODE;
    state.workload ??= null;
    state.stalledCycles ??= 0;
    state.bestTokPerSec ??= 0;
    state.lastProfileOutput ??= null;
    state.lastProfileCycle ??= null;
    state.lastMetalShapesOutput ??= null;
    state.lastMetalShapesCycle ??= null;
    state.lastMetalShapesOk ??= null;
    state.effortPlan ??= null;
    state.effortId ??= null;
    state.effortFile ??= null;
    state.bestTree ??= null;
    state.bestIncorrect ??= null;
    state.lastNearMissDiagnostic ??= null;
    state.crossEffortBaseline ??= null;
    state.lastCrossEffort ??= null;
    normalizeStateBestTokPerSec(state);
    await backfillBestTreeCommitFromGit(state);
    // Seed the near-miss target from history for runs predating bestIncorrect.
    if (backfillNearMiss(state) && state.bestIncorrect) {
      console.log(clr("1;35", `  ◎ Backfilled near-miss from cycle ${state.bestIncorrect.cycle}: ${state.bestIncorrect.tokPerSec.toFixed(1)} ${METRIC_LABEL} (+${state.bestIncorrect.gainPctOverAccepted.toFixed(1)}% over best kept-correct, output broke to "${state.bestIncorrect.outputText}")`));
    }
    // Re-read the effort doc from disk every resume so an edited plan
    // reaches the next agent invocation without losing saved history.
    if (effortBundle) {
      state.effortPlan = effortBundle.plan;
      state.effortId = effort;
      state.effortFile = effortBundle.file;
    } else if (state.effortId != null) {
      const refreshed = await loadEffortPlan(state.effortId);
      if (refreshed) {
        state.effortPlan = refreshed.plan;
        state.effortFile = refreshed.file;
      }
    }
    runId = state.runId;
    runDir = latestDir;
    startCycle = state.cycles.length + 1;
    await restorePromotedBestDuringPlateauIfNeeded(runDir, state);
    console.log(clr("1;36", "╔══════════════════════════════════════════════════════════════╗"));
    console.log(clr("1;36", "║  ZINC Metal Optimization Loop — RESUMING                     ║"));
    console.log(clr("1;36", `║  Target: ≥${TARGET_TOK_PER_SEC} ${METRIC_LABEL}  |  Model: ${displayModelLabel().slice(0, 35)}  ║`));
    console.log(clr("1;36", `║  Run: ${runId}  |  Resuming from cycle ${startCycle}            ║`));
    console.log(clr("1;36", "╚══════════════════════════════════════════════════════════════╝"));
    console.log(`  Agent: ${clr("1", agentLabel)}${model ? ` (${model})` : ""}`);
    console.log(`  Previous cycles: ${state.cycles.length}, best kept-correct: ${bestKeptCorrectTokPerSec(state).toFixed(2)} ${METRIC_LABEL}, current accepted: ${currentAcceptedTokPerSec(state).toFixed(2)} ${METRIC_LABEL}`);
    console.log(`  Results: ${clr("2", runDir)}`);
    if (state.effortFile) {
      console.log(`  Effort: ${clr("1;36", `#${state.effortId}`)} → ${state.effortFile}`);
    }
  } else {
    // Fresh start — clean up old data
    cleanupOldRuns();
    runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    runDir = join(RESULTS_DIR, runId);
    await mkdir(runDir, { recursive: true });
    startCycle = 1;
    state = {
      runId,
      cycles: [],
      failedApproaches: [],
      ideas: [],
      phase: "optimize",
      metricMode: METRIC_MODE,
      workload: null,
      currentBest: null,
      stalledCycles: 0,
      bestTokPerSec: 0,
      bestTree: null,
      lastProfileOutput: null,
      lastProfileCycle: null,
      lastMetalShapesOutput: null,
      lastMetalShapesCycle: null,
      lastMetalShapesOk: null,
      lastNearMissDiagnostic: null,
      crossEffortBaseline: null,
      lastCrossEffort: null,
      reviewSummaries: [],
      effortPlan: effortBundle?.plan ?? null,
      effortId: effort,
      effortFile: effortBundle?.file ?? null,
    };
    console.log(clr("1;36", "╔══════════════════════════════════════════════════════════════╗"));
    console.log(clr("1;36", "║  ZINC Metal Optimization Loop                                ║"));
    console.log(clr("1;36", `║  Target: ≥${TARGET_TOK_PER_SEC} ${METRIC_LABEL}  |  Model: ${displayModelLabel().slice(0, 35)}  ║`));
    console.log(clr("1;36", `║  Run: ${runId}  |  Max cycles: ${maxCycles}               ║`));
    console.log(clr("1;36", "╚══════════════════════════════════════════════════════════════╝"));
    console.log(`  Agent: ${clr("1", agentLabel)}${model ? ` (${model})` : ""}`);
    console.log(`  Results: ${clr("2", runDir)}`);
    if (state.effortFile) {
      console.log(`  Effort: ${clr("1;36", `#${state.effortId}`)} → ${state.effortFile}`);
    }
  }

  for (let cycle = startCycle; cycle <= maxCycles; cycle++) {
    console.log(clr("1;35", "\n" + "═".repeat(64)));
    console.log(clr("1;35", `  CYCLE ${cycle}`));
    console.log(clr("1;35", "═".repeat(64)));

    // Live-reload the effort doc each cycle so edits during a long run
    // take effect on the next cycle without needing --resume. The doc
    // is re-spliced into every agent prompt anyway; only the in-memory
    // copy needs to refresh.
    if (state.effortId != null) {
      const refreshed = await loadEffortPlan(state.effortId);
      if (refreshed && refreshed.plan !== state.effortPlan) {
        state.effortPlan = refreshed.plan;
        state.effortFile = refreshed.file;
        console.log(clr("1;36", `  Effort plan reloaded (${refreshed.file}, ${refreshed.plan.length} chars)`));
      }
    }

    const cycleDir = join(runDir, `cycle-${String(cycle).padStart(3, "0")}`);
    await mkdir(cycleDir, { recursive: true });

    // Always use full token count for stable benchmarking
    const currentMaxTokens = MAX_TOKENS;

    // Step 1: Build + Test + Run
    const result = await buildTestRun(currentMaxTokens);
    await syncWorkloadBeforeCycle(runDir, state, result);
    state.phase = result.phase;

    await writeFile(join(cycleDir, "build.log"), result.buildOutput);
    await writeFile(join(cycleDir, "test.log"), result.testOutput);
    await writeFile(join(cycleDir, "run.log"), result.runOutput);

    // Display status
    if (result.buildExitCode !== 0) {
      console.log(clr("1;31", `  ❌ BUILD FAILED`));
    } else if (result.testExitCode !== 0) {
      console.log(clr("1;31", `  ❌ TESTS FAILED`));
    } else if (result.runExitCode !== 0 && result.runExitCode !== null) {
      console.log(clr("1;31", `  ❌ CRASH (exit ${result.runExitCode})`));
    } else {
      const refTag = result.containsReference ? clr("1;32", " ✅CORRECT") : clr("1;33", " ❌WRONG");
      const tpsTag = result.tokPerSec ? ` ${result.tokPerSec.toFixed(1)} ${METRIC_LABEL}` : "";
      console.log(clr("1;32", `  ✅ ${result.tokensGenerated} tokens${tpsTag}`) + refTag);
      if (result.outputText) console.log(clr("2", `  Output: "${result.outputText.slice(0, 80)}"`));
    }

    if (dryRun) {
      console.log(clr("1;33", "  (dry-run mode — skipping agent)"));
      continue;
    }

    // Step 2: Git snapshot
    await runCommand("git", ["add", "-A", ...LOOP_COMMIT_PATHS]).catch(() => {});
    await runCommand("git", ["commit", "--allow-empty", "-m", `metal-loop: pre-cycle-${cycle}`]).catch(() => {});
    const preCommit = await runCommand("git", ["rev-parse", "HEAD"]);
    const preHash = preCommit.stdout.trim();

    // Step 3: Run agent
    const prompt = buildPrompt(state, result);
    await writeFile(join(cycleDir, "prompt.md"), prompt);

    const agentResult = await runAgent(agent, prompt, model);
    await writeFile(join(cycleDir, "agent.log"), agentResult.stdout + agentResult.stderr);

    // Extract markers
    const agentText = extractAgentText(agentResult.stdout);
    const lastChars = agentText.slice(-3000);
    const descMatch = lastChars.match(/@@@DESCRIPTION:\s*(.+)/im);
    const stepKindMatch = lastChars.match(/@@@STEP_KIND:\s*(.+)/im);
    const analysisMatch = lastChars.match(/@@@SELF_ANALYSIS:\s*(.+)/im);
    const ideasMatch = lastChars.match(/@@@NEXT_IDEAS:\s*(.+)/im);
    const description = descMatch?.[1]?.trim() ?? "Agent made changes";
    const selfAnalysis = analysisMatch?.[1]?.trim() ?? "";
    const stepKind = parseStepKind(stepKindMatch?.[1], description, selfAnalysis);
    const newIdeas = ideasMatch?.[1]?.split(",").map(s => s.trim()).filter(s => s.length > 3) ?? [];

    // Step 4: Verify
    console.log(clr("1;33", "\n  📊 Verifying..."));
    const verify = await buildTestRun(currentMaxTokens);
    await writeFile(join(cycleDir, "verify.log"), JSON.stringify(verify, null, 2));

    // Keep/revert decision — tight for optimization
    let kept = false;
    let promotedBest = false;
    let acceptedCommitHash: string | null = null;
    const baselines = keepBaselinesForCycle(state, result);
    const bestTps = baselines.bestTokPerSec;
    const acceptedTps = baselines.acceptedTokPerSec;
    // Proportional bands scale with real accepted performance, but use
    // two different anchors: promotion compares with the best kept
    // correct result, while neutral keeps compare with the current
    // accepted baseline measured at the start of this cycle. This avoids
    // the Effort 16 cycle-106/107 drift where a stale best=43.4 allowed
    // a 44.0 tree to keep 43.2 tok/s regressions as "within noise".
    //
    // The neutral band is intentionally tighter than the promotion band:
    // a correct foundation step can be kept when it is flat, but not when
    // it materially slows the current tree.
    // Promotion band scales with best, then inflates when measured sample
    // spread dwarfs it (decode benchmarks regularly bimodalize from M-series
    // thermal throttling; a tiny 1-tok band against a 10-tok spread accepts
    // mostly noise). See `noiseAwareImproveBand`.
    const improveBand = noiseAwareImproveBand(
      Math.max(0.3, bestTps * 0.02),
      verify.tokPerSecSamples,
    );
    const noiseBand = Math.max(0.25, acceptedTps * 0.01);
    const currentProgressBand = smallAcceptedProgressBand(acceptedTps);

    // Confirmation re-run: when a correct candidate sits in the noise zone
    // around the promotion boundary (or its samples were bimodal/THERMAL),
    // collect extra timed samples and re-aggregate over the combined set
    // before the verdict. This rescues real ~1-2% wins from M-series thermal
    // jitter without paying the cost on clearly-keep or clearly-revert cycles.
    // The tree is unchanged here, so no rebuild/retest is needed. Only the
    // throughput estimate is refined — the correctness verdict from the cycle
    // run stands.
    const preConfirmAgg = aggregateTimedSamples(verify.tokPerSecSamples);
    const preConfirmWorkloadMismatch = workloadMismatchReason(state.workload, measuredWorkload(verify));
    if (
      !preConfirmWorkloadMismatch &&
      shouldConfirmCandidate({
        containsReference: verify.containsReference,
        ranOk: verify.runExitCode === 0,
        verifyTps: verify.tokPerSec ?? 0,
        bestTps,
        improveBand,
        bimodal: preConfirmAgg.bimodal,
        sampleCount: verify.tokPerSecSamples.length,
        confirmRuns: BENCHMARK_CONFIRM_RUNS,
      })
    ) {
      console.log(clr("1;33", `  🔁 Confirmation re-run (${BENCHMARK_CONFIRM_RUNS} extra samples; candidate ${(verify.tokPerSec ?? 0).toFixed(2)} near promotion line ${(bestTps + improveBand).toFixed(2)}${preConfirmAgg.bimodal ? ", bimodal" : ""})...`));
      const confirm = await collectTimedSamples(currentMaxTokens, BENCHMARK_CONFIRM_RUNS, "confirm");
      if (confirm.samples.length > 0) {
        verify.tokPerSecSamples = [...verify.tokPerSecSamples, ...confirm.samples];
        verify.promptTokenSamples = [...(verify.promptTokenSamples ?? []), ...confirm.promptTokenSamples];
        verify.promptTokens = medianNumber(verify.promptTokenSamples);
        const merged = aggregateTimedSamples(verify.tokPerSecSamples);
        if (merged.tokPerSec != null) {
          const flag = merged.bimodal ? " ⚠ STILL BIMODAL" : "";
          console.log(clr("1;36", `    confirmed: ${(verify.tokPerSec ?? 0).toFixed(2)} → ${merged.tokPerSec.toFixed(2)} ${METRIC_LABEL} over ${verify.tokPerSecSamples.length} samples [${verify.tokPerSecSamples.map(s => s.toFixed(1)).join(", ")}]${flag}`));
          verify.tokPerSec = merged.tokPerSec;
          verify.workload = workloadFromMeasurement(verify);
          await writeFile(join(cycleDir, "verify.log"), JSON.stringify(verify, null, 2));
        }
      }
    }
    const verifyTps = verify.tokPerSec ?? 0;
    const verifyWorkloadMismatch = workloadMismatchReason(state.workload, measuredWorkload(verify));
    const cooldownRejectReason = cooledFamilyRejectionReason({
      state,
      stepKind,
      description,
      selfAnalysis,
      ideas: newIdeas,
    });

    let candidateProfileCaptured = false;
    let candidateMetalShapesCaptured = false;
    let candidateMetalShapesAttempted = false;
    const candidateEvidence = !verifyWorkloadMismatch ? shouldRunCandidateEvidence({
      state,
      cycle,
      containsReference: verify.containsReference,
      buildExitCode: verify.buildExitCode,
      testExitCode: verify.testExitCode,
      runExitCode: verify.runExitCode,
      stepKind,
      description,
      selfAnalysis,
      ideas: newIdeas,
    }) : { profile: false, metalShapes: false };
    if (candidateEvidence.profile || candidateEvidence.metalShapes) {
      const header = [
        `CANDIDATE EVIDENCE cycle ${cycle} (${stepKind})`,
        "Captured before the keep/revert verdict while this candidate source tree was still checked out.",
        "If the cycle was later reverted, these counters describe the reverted candidate, not current HEAD.",
        `Description: ${description}`,
        "",
      ].join("\n");

      if (candidateEvidence.profile) {
        try {
          state.lastProfileOutput = header + (await runProfileBenchmark());
          state.lastProfileCycle = cycle;
          candidateProfileCaptured = true;
          await writeFile(join(cycleDir, "candidate_profile.log"), state.lastProfileOutput);
        } catch {
          console.log(clr("1;33", "  ⚠ Candidate profile run failed, continuing"));
        }
      }

      if (candidateEvidence.metalShapes) {
        candidateMetalShapesAttempted = true;
        try {
          state.lastMetalShapesOutput = header + (await runMetalShapesBenchmark(state));
          state.lastMetalShapesCycle = cycle;
          state.lastMetalShapesOk = true;
          candidateMetalShapesCaptured = true;
          await writeFile(join(cycleDir, "candidate_metal_shapes.log"), state.lastMetalShapesOutput);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          state.lastMetalShapesOutput = `${header}FAILED candidate metal-shapes evidence run:\n${msg}`;
          state.lastMetalShapesCycle = cycle;
          state.lastMetalShapesOk = false;
          await writeFile(join(cycleDir, "candidate_metal_shapes_failed.log"), state.lastMetalShapesOutput);
          console.log(clr("1;33", `  ⚠ Candidate Metal-shapes evidence run failed, continuing: ${msg.slice(-500)}`));
        }
      }
    }

    if (verifyWorkloadMismatch) {
      console.log(clr("1;31", `  ↩ REVERTING — workload changed (${verifyWorkloadMismatch})`));
      await resetCycleToPreHash(preHash);
      state.failedApproaches.push(`${description} — changed benchmark workload: ${verifyWorkloadMismatch}`);
      state.stalledCycles++;
    } else if (verify.buildExitCode !== 0 || verify.testExitCode !== 0) {
      // Build or test broken → revert
      console.log(clr("1;31", `  ↩ REVERTING — ${verify.buildExitCode !== 0 ? "build" : "tests"} broken`));
      await resetCycleToPreHash(preHash);
      state.failedApproaches.push(`${description} — broke ${verify.buildExitCode !== 0 ? "build" : "tests"}`);
      state.stalledCycles++;
    } else if (verify.runExitCode !== 0 && verify.runExitCode !== null) {
      // Crash → revert
      console.log(clr("1;31", `  ↩ REVERTING — runtime crash`));
      await resetCycleToPreHash(preHash);
      state.failedApproaches.push(`${description} — runtime crash`);
      state.stalledCycles++;
    } else if (!verify.containsReference && state.currentBest?.containsReference) {
      // Lost correctness → always revert
      console.log(clr("1;31", `  ↩ REVERTING — lost correctness (output: "${verify.outputText.slice(0, 60)}")`));
      await resetCycleToPreHash(preHash);
      state.failedApproaches.push(`${description} — broke correctness`);
      state.stalledCycles++;
    } else if (cooldownRejectReason) {
      console.log(clr("1;31", `  ↩ REVERTING — ${cooldownRejectReason}`));
      await resetCycleToPreHash(preHash);
      state.failedApproaches.push(`${description} — ${cooldownRejectReason}`);
      state.stalledCycles++;
    } else if (verify.containsReference && verifyTps > bestTps + improveBand) {
      // Meaningful speed improvement with correct output
      kept = true;
      promotedBest = true;
      state.bestTokPerSec = verifyTps;
      state.stalledCycles = 0;
      console.log(clr("1;32", `  ✅ KEPT — ${verifyTps.toFixed(2)} ${METRIC_LABEL} (best was ${bestTps.toFixed(2)}, +${(verifyTps - bestTps).toFixed(2)}; band +${improveBand.toFixed(2)})`));
    } else if (verify.containsReference && verifyTps >= acceptedTps - noiseBand) {
      // Within noise band, correct output — keep the change but DO NOT
      // advance bestTokPerSec. Advancing on noise creates a one-way
      // ratchet that pretends throughput improved when it did not
      // (Effort 12 cycles 1-24 went 0.21 → 0.30 this way, all noise).
      if (shouldRejectPlateauNeutralKeep({
        state,
        stepKind,
        description,
        selfAnalysis,
        verifyTokPerSec: verifyTps,
        acceptedTokPerSec: acceptedTps,
        currentProgressBand,
      })) {
        console.log(clr("1;31", `  ↩ REVERTING — plateau mode rejects neutral ${stepKind} keep at ${verifyTps.toFixed(2)} ${METRIC_LABEL}; current ${acceptedTps.toFixed(2)}`));
        await resetCycleToPreHash(preHash);
        state.failedApproaches.push(`${description} — plateau-neutral ${stepKind} did not beat current ${acceptedTps.toFixed(1)} ${METRIC_LABEL}`);
        state.stalledCycles++;
      } else {
        kept = true;
        if (verifyTps >= acceptedTps + currentProgressBand) {
          if (shouldPreservePromotedBestStallPressure(state, verifyTps, bestTps)) {
            state.stalledCycles++;
            console.log(clr("1;32", `  ↑ KEPT — current accepted improved ${acceptedTps.toFixed(2)} → ${verifyTps.toFixed(2)} ${METRIC_LABEL}, but remains below promoted best ${bestTps.toFixed(2)}; plateau pressure preserved`));
          } else {
            state.stalledCycles = 0;
            console.log(clr("1;32", `  ↑ KEPT — current accepted improved ${acceptedTps.toFixed(2)} → ${verifyTps.toFixed(2)} ${METRIC_LABEL} (below promotion band +${improveBand.toFixed(2)})`));
          }
        } else {
          state.stalledCycles++;
          console.log(clr("1;33", `  ≈ KEPT — ${verifyTps.toFixed(2)} ${METRIC_LABEL} (within ${noiseBand.toFixed(2)} of current ${acceptedTps.toFixed(2)}; best ${bestTps.toFixed(2)} unchanged)`));
        }
      }
    } else if (verify.containsReference && !state.currentBest?.containsReference) {
      // Gained correctness for the first time
      kept = true;
      promotedBest = true;
      state.bestTokPerSec = verifyTps;
      state.stalledCycles = 0;
      console.log(clr("1;32", `  ✅ KEPT — gained correct output! ${verifyTps.toFixed(2)} ${METRIC_LABEL}`));
    } else {
      // Regressed speed or no correctness
      console.log(clr("1;31", `  ↩ REVERTING — ${verifyTps.toFixed(2)} ${METRIC_LABEL} < current ${acceptedTps.toFixed(2)} (regressed ${(acceptedTps - verifyTps).toFixed(2)}; band -${noiseBand.toFixed(2)})`));
      await resetCycleToPreHash(preHash);
      state.failedApproaches.push(`${description} — regressed from current ${acceptedTps.toFixed(1)} to ${verifyTps.toFixed(1)} ${METRIC_LABEL}`);
      state.stalledCycles++;
    }

    // Track a fast-but-incorrect revert as the standing near-miss target so
    // the next prompt redirects the agent at localizing the divergent layer
    // rather than rediscovering the same correctness-breaking idea. No-op for
    // correct or slower-than-accepted results.
    if (
      recordNearMiss(state, {
        cycle,
        tokPerSec: verify.tokPerSec,
        ranOk: verify.runExitCode === 0,
        containsReference: verify.containsReference,
        acceptedTps,
        description,
        selfAnalysis,
        outputText: verify.outputText,
      })
    ) {
      console.log(clr("1;35", `  ◎ NEAR-MISS recorded — ${(verify.tokPerSec ?? 0).toFixed(1)} ${METRIC_LABEL} (+${state.bestIncorrect!.gainPctOverAccepted.toFixed(1)}% over accepted) but output broke; will redirect future cycles to localize the divergent layer.`));
    }

    if (kept) {
      state.currentBest = {
        tokPerSec: verify.tokPerSec,
        containsReference: verify.containsReference,
      };
      await runCommand("git", ["add", "-A", ...LOOP_COMMIT_PATHS]).catch(() => {});
      await runCommand("git", ["commit", "-m", `metal-loop: cycle-${cycle} ${description} (${verifyTps.toFixed(1)} ${METRIC_LABEL})`]).catch(() => {});
      const acceptedHead = await runCommand("git", ["rev-parse", "HEAD"]).catch(() => null);
      acceptedCommitHash = acceptedHead?.stdout.trim() ?? null;
      if (promotedBest && acceptedCommitHash && verify.tokPerSec != null) {
        state.bestTree = {
          cycle,
          tokPerSec: verify.tokPerSec,
          commitHash: acceptedCommitHash,
        };
      }
    }

    // Periodic profiling run (after verify, so we profile the current accepted state)
    // Also profile on cycle 1 so the agent has data from the start
    if (!candidateProfileCaptured && (cycle === 1 || cycle % PROFILE_EVERY === 0) && kept && verify.containsReference) {
      try {
        state.lastProfileOutput = await runProfileBenchmark();
        state.lastProfileCycle = cycle;
        await writeFile(join(cycleDir, "profile.log"), state.lastProfileOutput);
      } catch {
        console.log(clr("1;33", "  ⚠ Profile run failed, continuing"));
      }
    }

    if (!candidateMetalShapesCaptured && !(candidateMetalShapesAttempted && !kept) && shouldRunMetalShapesEvidence({
      state,
      cycle,
      kept,
      containsReference: verify.containsReference,
      stepKind,
      description,
      selfAnalysis,
      ideas: newIdeas,
    })) {
      try {
        state.lastMetalShapesOutput = await runMetalShapesBenchmark(state);
        state.lastMetalShapesCycle = cycle;
        state.lastMetalShapesOk = true;
        await writeFile(join(cycleDir, "metal_shapes.log"), state.lastMetalShapesOutput);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        state.lastMetalShapesOutput = `FAILED metal-shapes evidence run:\n${msg}`;
        state.lastMetalShapesCycle = cycle;
        state.lastMetalShapesOk = false;
        await writeFile(join(cycleDir, "metal_shapes_failed.log"), state.lastMetalShapesOutput);
        console.log(clr("1;33", `  ⚠ Metal-shapes evidence run failed, continuing: ${msg.slice(-500)}`));
      }
    }

    // Near-miss diagnostic run. Only fires when (a) ZINC_NEAR_MISS_DIAGNOSTIC_ENV
    // is set, and (b) bestIncorrect exists and is still meaningfully ahead of the
    // accepted tree (i.e. the directive will fire for the next cycle). Skipped on
    // build/test failures so we never burn time diagnosing an unbuildable tree.
    const diagnosticEnv = parseDiagnosticEnv(NEAR_MISS_DIAGNOSTIC_ENV_RAW);
    const diagEnvKeys = Object.keys(diagnosticEnv);
    if (
      diagEnvKeys.length > 0 &&
      state.bestIncorrect &&
      verify.buildExitCode === 0 &&
      verify.testExitCode === 0 &&
      state.bestIncorrect.tokPerSec >
        Math.max(currentAcceptedTokPerSec(state), bestKeptCorrectTokPerSec(state)) +
          Math.max(0.3, bestKeptCorrectTokPerSec(state) * 0.005)
    ) {
      try {
        const diagOutput = await runNearMissDiagnostic(currentMaxTokens, diagnosticEnv);
        state.lastNearMissDiagnostic = {
          cycle,
          envApplied: diagEnvKeys,
          output: diagOutput,
        };
        await writeFile(join(cycleDir, "near_miss_diagnostic.log"), diagOutput);
      } catch {
        console.log(clr("1;33", "  ⚠ Near-miss diagnostic run failed, continuing"));
      }
    }

    // Cross-effort guard. When ZINC_CROSS_EFFORT_PROMPT is set, measure the
    // OTHER metric (e.g. prefill during a decode run) every CROSS_EFFORT_EVERY
    // cycles (plus cycle 1 for the baseline). Surfaces regression early.
    if (
      CROSS_EFFORT_PROMPT &&
      verify.buildExitCode === 0 &&
      verify.testExitCode === 0 &&
      (cycle === 1 || cycle % CROSS_EFFORT_EVERY === 0)
    ) {
      try {
        const xtps = await runCrossEffortCheck();
        if (xtps != null) {
          if (!state.crossEffortBaseline) {
            state.crossEffortBaseline = { metric: CROSS_EFFORT_METRIC, tokPerSec: xtps, cycle };
            console.log(clr("1;36", `  ◎ Cross-effort baseline locked: ${CROSS_EFFORT_METRIC} = ${xtps.toFixed(2)} tok/s (cycle ${cycle})`));
          }
          const baseline = state.crossEffortBaseline.tokPerSec;
          const deltaPct = ((xtps - baseline) / baseline) * 100;
          state.lastCrossEffort = { cycle, metric: CROSS_EFFORT_METRIC, tokPerSec: xtps, deltaPct };
          const flag = deltaPct <= -3 ? " ⚠⚠ REGRESSION" : deltaPct <= -1.5 ? " ⚠ drift" : "";
          console.log(clr(deltaPct <= -3 ? "1;31" : deltaPct <= -1.5 ? "1;33" : "1;36", `  ◎ Cross-effort ${CROSS_EFFORT_METRIC}: ${xtps.toFixed(2)} tok/s (baseline ${baseline.toFixed(2)}, Δ ${deltaPct >= 0 ? "+" : ""}${deltaPct.toFixed(1)}%)${flag}`));
        }
      } catch {
        console.log(clr("1;33", "  ⚠ Cross-effort check failed, continuing"));
      }
    }

    // Update state
    const cycleResult: CycleResult = {
      cycle,
      timestamp: new Date().toISOString(),
      phase: verify.phase,
      description,
      kept,
      tokPerSec: verify.tokPerSec,
      tokensGenerated: verify.tokensGenerated,
      promptTokens: verify.promptTokens ?? null,
      promptTokenSamples: verify.promptTokenSamples ?? [],
      workload: verify.workload ?? null,
      containsReference: verify.containsReference,
      buildExitCode: verify.buildExitCode,
      testExitCode: verify.testExitCode,
      runExitCode: verify.runExitCode,
      outputText: verify.outputText,
      selfAnalysis,
      stepKind,
      nextIdeas: newIdeas,
      commitHash: acceptedCommitHash,
      promotedBest,
    };

    state.cycles.push(cycleResult);
    for (const idea of newIdeas) {
      if (!state.ideas.includes(idea)) state.ideas.push(idea);
    }

    // Self-review every REVIEW_EVERY cycles
    if (state.cycles.length > 0 && state.cycles.length % REVIEW_EVERY === 0) {
      console.log(clr("1;35", `\n  🔍 Self-review (${state.cycles.length} cycles completed)...`));
      const review = buildSelfReview(state);
      state.reviewSummaries.push(review);
      console.log(clr("2", review));
    }

    normalizeStateBestTokPerSec(state);
    await restorePromotedBestDuringPlateauIfNeeded(runDir, state);
    await saveState(runDir, state);

    // Status summary
    console.log(clr("2", `  stall=${state.stalledCycles} best-kept=${bestKeptCorrectTokPerSec(state).toFixed(2)} ${METRIC_LABEL} current=${currentAcceptedTokPerSec(state).toFixed(2)} target=${TARGET_TOK_PER_SEC}`));

    // Check if we're done
    if (STOP_ON_TARGET && verify.containsReference && verify.tokPerSec != null && verify.tokPerSec >= TARGET_TOK_PER_SEC) {
      console.log(clr("1;32", "\n" + "=".repeat(64)));
      console.log(clr("1;32", `  TARGET REACHED: ${verify.tokPerSec.toFixed(1)} ${METRIC_LABEL} >= ${TARGET_TOK_PER_SEC} with correct output!`));
      console.log(clr("1;32", "=".repeat(64)));
      break;
    }

    if (AUTO_STOP_PLATEAU) {
      const plateauStop = detectAutoStopForPlateau(state);
      if (plateauStop.stop) {
        console.log(clr("1;35", "\n" + "=".repeat(64)));
        console.log(clr("1;35", `  PLATEAU AUTO-STOP: ${plateauStop.reason}`));
        if (plateauStop.bestCycle != null) {
          console.log(clr("1;35", `  Best promoted cycle: ${plateauStop.bestCycle}; cycles since best: ${plateauStop.cyclesSinceBest}; consecutive reverts: ${plateauStop.consecutiveReverts}`));
        }
        console.log(clr("1;35", "=".repeat(64)));
        await finalizeBestTreeIfNeeded(runDir, state);
        await saveState(runDir, state);
        break;
      }
    }
  }

  await finalizeBestTreeIfNeeded(runDir, state);

  console.log(clr("1;36", `\nLoop complete. Results: ${runDir}`));
  console.log(clr("1;36", `Total cycles: ${state.cycles.length}`));
  console.log(clr("1;36", `Kept: ${state.cycles.filter(c => c.kept).length}`));
  console.log(clr("1;36", `Best kept-correct: ${bestKeptCorrectTokPerSec(state).toFixed(2)} ${METRIC_LABEL}; current accepted: ${currentAcceptedTokPerSec(state).toFixed(2)} ${METRIC_LABEL} (target: ${TARGET_TOK_PER_SEC}), correct=${state.currentBest?.containsReference ?? false}`));
}

if (import.meta.main) {
  main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}
