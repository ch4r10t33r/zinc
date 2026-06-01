#!/usr/bin/env bun
/**
 * ZINC_RT Autopilot — Overnight A/B Optimization Loop
 *
 * Goal: make ZINC_RT (our own GPU runtime, replacing Vulkan) beat the
 * Vulkan backend on tok/s for the validated catalog model on RDNA4.
 *
 * Vulkan is a permanently supported peer backend (selected via
 * `-Dbackend=vulkan`), NOT a deprecation target. The agent must keep it
 * building and passing tests every cycle. This loop's job is to make
 * `zinc_rt` faster than `vulkan` — not to remove `vulkan`.
 *
 * Each cycle:
 *   1. rsync local source → remote RDNA4 node
 *   2. Build BOTH binaries:
 *        zig build -Dbackend=vulkan -Doptimize=ReleaseFast
 *        zig build -Dbackend=zinc_rt        -Doptimize=ReleaseFast
 *      If `-Dbackend=zinc_rt` doesn't yet exist (we're at M0), we still
 *      benchmark legacy and mark zinc_rt as "not yet built".
 *   3. Head-to-head benchmark: run each binary N times under flock,
 *      take median tok/s, parse output coherence.
 *   4. Compute ratio = zinc_rt_tps / vulkan_tps. Goal: > 1.0.
 *   5. Spawn agent (claude or codex). Agent reads docs/ZINC_RT_DESIGN.md,
 *      decides what milestone we're at, makes ONE focused change.
 *   6. Re-build, re-benchmark.
 *   7. Keep change if it improves the ratio (or unlocks coherent output);
 *      revert otherwise.
 *
 * Phases:
 *   MIGRATE — ZINC_RT build broken, runtime broken, output incoherent,
 *             or running at <60% of Vulkan. Agent focuses on getting
 *             ZINC_RT to a working, correct, competitive state.
 *   OPTIMIZE — ZINC_RT is coherent and within 60% of Vulkan. Agent now
 *             focuses on beating Vulkan.
 *
 * Reboot support:
 *   The agent may request a node reboot by emitting "@@@REBOOT" at the
 *   end of its output (e.g. after a kernel module reload, driver swap,
 *   sysctl change that requires reboot). The loop sends `reboot` over
 *   SSH and polls until the node is back (up to 6 minutes), then
 *   continues the cycle's verification step.
 *
 * Persistence:
 *   Results land in .zinc_rt_autopilot/<runId>/. State file is JSON;
 *   `--resume <dir>` picks up where a previous overnight run left off.
 *
 * Usage:
 *   bun loops/zinc_rt_autopilot.ts                       # forever, codex default
 *   bun loops/zinc_rt_autopilot.ts --cycles 30
 *   bun loops/zinc_rt_autopilot.ts --agent claude
 *   bun loops/zinc_rt_autopilot.ts --dry-run             # bench-only sanity
 *   bun loops/zinc_rt_autopilot.ts --resume <dir>
 *   bun loops/zinc_rt_autopilot.ts --runs-per-binary 5
 *   bun loops/zinc_rt_autopilot.ts --model-path /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
 *   bun loops/zinc_rt_autopilot.ts --target-ratio 1.05   # stop when zinc_rt ≥1.05× vulkan
 *
 * Designed to be safe to run overnight unattended. Survives:
 *   - GPU lock contention (uses flock)
 *   - SSH flakes (retries with backoff)
 *   - Node reboots (agent-requested or unprompted)
 *   - Broken builds (treated as MIGRATE-phase signal)
 *   - Agent crashes (one cycle skipped, loop continues)
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import {
  isGarbageString,
  formatElapsed,
} from "./optimize_llm_tps";
import {
  parseTokPerSec,
  parsePrefillTokPerSec,
  parseTokensGenerated,
  isGarbageOutput,
  isCoherentText,
  parseBandwidthUtil,
  parseEffectiveBW,
} from "./optimize_zinc";

const ZINC_RT_DIRECT_DECODE_EFFORT =
  "loops/efforts/MULTI_HOUR_EFFORT_15_ZINC_RT_DIRECT_DECODE_120TPS.md";

// ── Color & display ──────────────────────────────────────────────────

const TTY = process.stdout.isTTY ?? false;
const NO_COLOR = "NO_COLOR" in process.env;
const FORCE_COLOR =
  process.env.FORCE_COLOR === "1" || process.env.CLICOLOR_FORCE === "1";
const COLOR_ENABLED = !NO_COLOR && (TTY || FORCE_COLOR);

function clr(code: string, text: string): string {
  return COLOR_ENABLED ? `\x1b[${code}m${text}\x1b[0m` : text;
}

const SEP = "─".repeat(72);
const DSEP = "═".repeat(72);

// ── Constants ────────────────────────────────────────────────────────

const REPO_ROOT = resolve(import.meta.dir, "..");
let PROJECT_ROOT = REPO_ROOT;
let RESULTS_DIR = resolve(REPO_ROOT, ".zinc_rt_autopilot");

function loadEnv(): Record<string, string> {
  const envPath = join(REPO_ROOT, ".env");
  const vars: Record<string, string> = {};
  if (existsSync(envPath)) {
    const content = require("fs").readFileSync(envPath, "utf8") as string;
    for (const line of content.split("\n")) {
      const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.+?)\s*$/);
      if (m) vars[m[1]] = m[2];
    }
  }
  return vars;
}

const ENV = loadEnv();
const ZINC_HOST = process.env.ZINC_HOST ?? ENV.ZINC_HOST ?? "127.0.0.1";
const ZINC_PORT = Number(process.env.ZINC_PORT ?? ENV.ZINC_PORT ?? "22");
const ZINC_USER = process.env.ZINC_USER ?? ENV.ZINC_USER ?? "root";
let REMOTE_ZINC_DIR = "/root/zinc";
const DEFAULT_MODEL = "/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";
const DEFAULT_PROMPT = "The capital of France is";
const CLAUDE_MODEL = process.env.ZINC_CLAUDE_MODEL ?? "claude-opus-4-7[1m]";
const CLAUDE_EFFORT = process.env.ZINC_CLAUDE_EFFORT ?? "max";
const CODEX_MODEL = process.env.ZINC_CODEX_MODEL ?? "gpt-5.5";
const CODEX_REASONING_EFFORT = process.env.ZINC_CODEX_REASONING_EFFORT ?? "xhigh";

const REMOTE_VULKAN_BIN = "./zig-out/bin/zinc-vulkan";
const REMOTE_ZINC_RT_BIN = "./zig-out/bin/zinc-zinc_rt";
const REMOTE_ENV_PREFIX = "export PATH=/root/.bun/bin:$PATH;";

// Threshold above which ZINC_RT is considered "competitive" and we leave
// MIGRATE phase. 0.6 = within 60% of Vulkan tok/s.
const COMPETITIVE_THRESHOLD = 0.6;
// Required ratio improvement to keep a change in OPTIMIZE phase.
const RATIO_IMPROVEMENT_KEEP = 0.01; // +1% ratio
const ABS_TPS_IMPROVEMENT_KEEP = 0.5; // or +0.5 tok/s, whichever is larger
const MIGRATE_REL_TPS_IMPROVEMENT_KEEP = 0.02; // M0 accepts smaller real gains
const MIGRATE_MIN_ABS_TPS_IMPROVEMENT_KEEP = 0.15;
const FOUNDATION_MAX_TPS_REGRESSION = 0.03; // validation keeps may be flat, not slower
const HARD_PIVOT_STALL_CYCLES = 8;
const AGENT_TIMEOUT_MS = 3_600_000; // 1h — M0 layer-lowering cycles can be substantial
const VERIFICATION_REPAIR_ATTEMPTS = 1;

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
  "Bash(git push:*)",
];

const BLOCKED_FILE_OPS = [
  "Edit(.env)",
  "Write(.env)",
  "Edit(.zinc_rt_autopilot/*)",
  "Write(.zinc_rt_autopilot/*)",
];

// docs/ZINC_RT_DESIGN.md is the contract — but we allow agent to ANNOTATE
// it with achieved milestone numbers per §25.2.
// We do NOT block writes to docs/ZINC_RT_DESIGN.md.

type AgentKind = "claude" | "codex";

// ── Phase model ──────────────────────────────────────────────────────

export type Phase = "migrate" | "optimize";

export type BenchmarkResult = {
  /** Median decode tok/s across the runs. Null if no run produced metrics. */
  decodeTps: number | null;
  /** Median prefill tok/s across the runs. */
  prefillTps: number | null;
  /** All decode tok/s samples that parsed. */
  decodeSamples: number[];
  /** All prefill tok/s samples that parsed. */
  prefillSamples: number[];
  /** Build exit code (last build) — 0 ok, !=0 failed, -1 not built yet. */
  buildExitCode: number;
  /** Last build stdout/stderr (truncated to 6KB). */
  buildOutput: string;
  /** Last run stdout/stderr (truncated to 6KB). */
  runOutput: string;
  /** Run exit code (last run, or null if not run). */
  runExitCode: number | null;
  /** True if the decoded output looks coherent. */
  coherentText: boolean;
  /** True if output is repetitive/garbage. */
  garbageOutput: boolean;
  /** Tokens generated in last run. */
  tokensGenerated: number;
  /** Modeled bandwidth utilization (from output, if reported). */
  bandwidthUtil: number | null;
  /** Modeled effective bandwidth GB/s. */
  effectiveBW: number | null;
  /** Error message if benchmarking failed entirely. */
  error: string | null;
  /** Whether `-Dbackend=zinc_rt` is even recognized by build.zig (false at M0). */
  backendFlagRecognized: boolean;
};

export type ABBenchmark = {
  vulkan: BenchmarkResult;
  zinc_rt: BenchmarkResult;
  /** zinc_rt.decodeTps / vulkan.decodeTps. Null if either side missing. */
  ratio: number | null;
  /** Same for prefill. */
  prefillRatio: number | null;
};

// ── Command runner ───────────────────────────────────────────────────

type RunResult = { exitCode: number; stdout: string; stderr: string };

async function runCommand(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    streamOutput?: boolean;
    timeout?: number;
    stdoutLineFormatter?: (line: string) => string | null;
  } = {},
): Promise<RunResult> {
  const streamOutput = opts.streamOutput ?? false;
  return new Promise((res, rej) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? PROJECT_ROOT,
      env: opts.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout,
    });
    let stdout = "",
      stderr = "",
      lineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!streamOutput) return;
      if (opts.stdoutLineFormatter) {
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const f = opts.stdoutLineFormatter(line);
          if (f !== null) process.stdout.write(f);
        }
      } else {
        process.stdout.write(text);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (streamOutput) process.stderr.write(text);
    });
    child.on("error", rej);
    child.on("close", (code) => {
      if (streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const f = opts.stdoutLineFormatter(lineBuffer);
        if (f !== null) process.stdout.write(f);
      }
      res({ exitCode: code ?? 1, stdout, stderr });
    });
  });
}

// ── SSH & rsync ──────────────────────────────────────────────────────

async function ssh(command: string, timeout = 120_000): Promise<string> {
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-o", "StrictHostKeyChecking=no",
      "-o", "ConnectTimeout=10",
      "-o", "ServerAliveInterval=15",
      "-p", String(ZINC_PORT),
      `${ZINC_USER}@${ZINC_HOST}`,
      command,
    ],
    { streamOutput: false, timeout },
  );
  if (exitCode !== 0 && !stderr.includes("Warning")) {
    throw new Error(`SSH failed (${exitCode}): ${stderr.slice(0, 500)}`);
  }
  return stdout.trim();
}

async function sshSafe(command: string, timeout = 120_000): Promise<string | null> {
  try {
    return await ssh(command, timeout);
  } catch {
    return null;
  }
}

async function rsyncToRemote(): Promise<void> {
  console.log(clr("2", "  rsync → remote..."));
  const { exitCode, stderr } = await runCommand(
    "rsync",
    [
      "-az", "--delete",
      "-e", `ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no`,
      "--exclude", ".zig-cache",
      "--exclude", "zig-out",
      "--exclude", "zig-cache",
      "--exclude", "node_modules",
      "--exclude", ".git",
      "--exclude", ".zinc_optimize",
      "--exclude", ".llm_optimize",
      "--exclude", ".zinc_rt_autopilot",
      "--exclude", ".zig-api-cache",
      "--exclude", ".DS_Store",
      "--exclude", "site",
      `${PROJECT_ROOT}/`,
      `${ZINC_USER}@${ZINC_HOST}:${REMOTE_ZINC_DIR}/`,
    ],
    { timeout: 180_000 },
  );
  if (exitCode !== 0) {
    throw new Error(`rsync failed: ${stderr.slice(0, 500)}`);
  }
}

// ── Reboot ───────────────────────────────────────────────────────────

async function rebootRemote(): Promise<boolean> {
  console.log(clr("1;33", "\n  🔁 Reboot requested by agent — sending reboot over SSH..."));
  // `nohup ... &` + immediate disconnect; we don't wait for the reply
  await sshSafe("nohup sh -c 'sleep 1 && reboot' >/dev/null 2>&1 &", 15_000);
  // Brief sleep so the node actually goes down before we start polling
  await new Promise((r) => setTimeout(r, 8_000));

  const deadline = Date.now() + 6 * 60 * 1000; // 6 min
  let attempts = 0;
  while (Date.now() < deadline) {
    attempts++;
    process.stdout.write(clr("2", `\r  Waiting for node to come back (attempt ${attempts})...`));
    const probe = await sshSafe("echo back", 8_000);
    if (probe && probe.includes("back")) {
      process.stdout.write("\n");
      console.log(clr("1;32", `  ✅ Node back online after ${attempts} attempts.`));
      // Give services another 10s to settle
      await new Promise((r) => setTimeout(r, 10_000));
      return true;
    }
    await new Promise((r) => setTimeout(r, 8_000));
  }
  process.stdout.write("\n");
  console.log(clr("1;31", "  ❌ Node did not come back within 6 minutes."));
  return false;
}

// ── Remote build for each backend ────────────────────────────────────

type BuildSummary = {
  exitCode: number;
  output: string;
  /** True if the build.zig recognized the -Dbackend flag (only false at M0). */
  flagRecognized: boolean;
};

async function remoteBuild(backend: "vulkan" | "zinc_rt"): Promise<BuildSummary> {
  console.log(clr("2", `  build -Dbackend=${backend}...`));
  // We pass -Dno-shaders=true for zinc_rt builds once they don't need legacy shaders.
  // For now, we always build the shader set so the vulkan path remains
  // testable; the zinc_rt build can compile additionally without harming Vulkan.
  const cmd = `${REMOTE_ENV_PREFIX} cd ${REMOTE_ZINC_DIR} && (zig build -Doptimize=ReleaseFast -Dbackend=${backend} 2>&1; echo __EXIT__=$?)`;
  const { stdout, stderr } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      cmd,
    ],
    { streamOutput: true, timeout: 360_000 },
  );
  const combined = stdout + (stderr ? `\n${stderr}` : "");
  const exitMatch = combined.match(/__EXIT__=(\d+)/);
  const exitCode = exitMatch ? parseInt(exitMatch[1], 10) : 1;
  // build.zig that doesn't know -Dbackend can print "invalid option:
  // -Dbackend", "unknown option 'backend'", or similar depending on Zig.
  const backendFlagMissing =
    /invalid option:\s+-Dbackend\b/i.test(combined) ||
    /unknown option ['"]?backend['"]?/i.test(combined) ||
    /unknown build option: ['"]?backend['"]?/i.test(combined);
  const flagRecognized = !backendFlagMissing;

  // Rename the produced binary so vulkan vs zinc_rt builds don't overwrite each other.
  if (exitCode === 0) {
    await sshSafe(`cd ${REMOTE_ZINC_DIR} && cp -f ./zig-out/bin/zinc ${backend === "vulkan" ? REMOTE_VULKAN_BIN : REMOTE_ZINC_RT_BIN}`, 10_000);
  }

  return { exitCode, output: combined.slice(-6000), flagRecognized };
}

async function remoteTest(): Promise<{ passed: boolean; output: string }> {
  console.log(clr("2", "  zig build test..."));
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      `${REMOTE_ENV_PREFIX} cd ${REMOTE_ZINC_DIR} && zig build test --summary all 2>&1`,
    ],
    { streamOutput: false, timeout: 180_000 },
  );
  const out = stdout + (stderr ? `\n${stderr}` : "");
  const passMatch = out.match(/(\d+)\/\d+\s+tests\s+passed/i);
  if (passMatch) console.log(clr("2", `  ✓ ${passMatch[0]}`));
  if (exitCode !== 0) console.log(clr("1;31", "  ❌ Tests failed"));
  return { passed: exitCode === 0, output: out.slice(-3000) };
}

// ── Benchmark protocol ───────────────────────────────────────────────

async function runOnce(
  binary: string,
  modelPath: string,
  prompt: string,
  envFlags: string,
): Promise<{ exitCode: number; output: string }> {
  // flock serializes GPU access on the remote node.
  const inner = `${envFlags} timeout 90 ${binary} -m ${modelPath} --prompt "${prompt}" 2>&1`;
  const wrapped = `cd ${REMOTE_ZINC_DIR} && flock /tmp/zinc-gpu.lock -c '${inner.replace(/'/g, "'\\''")}'`;
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      wrapped,
    ],
    { streamOutput: false, timeout: 240_000 },
  );
  return { exitCode, output: stdout + (stderr ? `\n${stderr}` : "") };
}

function median(xs: number[]): number | null {
  if (xs.length === 0) return null;
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/**
 * Benchmark one binary. We do `runsPerBinary` independent runs (warmup
 * is the first run, included in samples but typically slowest). Take
 * the median of samples that produced a tok/s number AND coherent text.
 *
 * Each run is fully cold: flock prevents overlap with other instances,
 * but we don't reuse a server process — so cache state is essentially
 * a fresh load every time.
 */
async function benchmarkBinary(
  binary: string,
  modelPath: string,
  prompt: string,
  envFlags: string,
  runsPerBinary: number,
): Promise<BenchmarkResult> {
  const decodeSamples: number[] = [];
  const prefillSamples: number[] = [];
  let lastOutput = "";
  let lastExit: number | null = null;
  let coherentSeen = false;
  let garbageSeen = false;
  let lastTokensGenerated = 0;
  let lastBwUtil: number | null = null;
  let lastEffBW: number | null = null;

  for (let i = 0; i < runsPerBinary; i++) {
    const tag = `[${i + 1}/${runsPerBinary}]`;
    process.stdout.write(clr("2", `    ${tag} ${binary.split("/").pop()}... `));
    const res = await runOnce(binary, modelPath, prompt, envFlags);
    lastOutput = res.output;
    lastExit = res.exitCode;

    if (res.exitCode !== 0) {
      process.stdout.write(clr("1;31", `crash (exit ${res.exitCode})\n`));
      continue;
    }

    const decode = parseTokPerSec(res.output);
    const prefill = parsePrefillTokPerSec(res.output);
    if (decode != null) decodeSamples.push(decode);
    if (prefill != null) prefillSamples.push(prefill);
    const coherent = isCoherentText(res.output);
    const garbage = isGarbageOutput(res.output);
    if (coherent) coherentSeen = true;
    if (garbage && !coherent) garbageSeen = true;
    lastTokensGenerated = parseTokensGenerated(res.output);
    const bw = parseBandwidthUtil(res.output);
    const eff = parseEffectiveBW(res.output);
    if (bw != null) lastBwUtil = bw;
    if (eff != null) lastEffBW = eff;

    const decodeStr = decode != null ? `${decode.toFixed(1)} tok/s` : "—";
    const prefillStr = prefill != null ? `${prefill.toFixed(1)} pref` : "";
    const cohStr = coherent ? clr("32", " ✓") : (garbage ? clr("31", " ✗") : clr("33", " ?"));
    process.stdout.write(`${decodeStr}${prefillStr ? " / " + prefillStr : ""}${cohStr}\n`);
  }

  return {
    decodeTps: median(decodeSamples),
    prefillTps: median(prefillSamples),
    decodeSamples,
    prefillSamples,
    buildExitCode: 0,
    buildOutput: "",
    runOutput: lastOutput.slice(-6000),
    runExitCode: lastExit,
    coherentText: coherentSeen,
    garbageOutput: garbageSeen && !coherentSeen,
    tokensGenerated: lastTokensGenerated,
    bandwidthUtil: lastBwUtil,
    effectiveBW: lastEffBW,
    error: lastExit !== 0 ? `Last run exit ${lastExit}` : null,
    backendFlagRecognized: true,
  };
}

async function fullAB(
  modelPath: string,
  prompt: string,
  runsPerBinary: number,
): Promise<ABBenchmark> {
  // Build vulkan first
  const vkBuild = await remoteBuild("vulkan");
  let vulkan: BenchmarkResult;
  if (vkBuild.exitCode !== 0) {
    vulkan = {
      decodeTps: null,
      prefillTps: null,
      decodeSamples: [],
      prefillSamples: [],
      buildExitCode: vkBuild.exitCode,
      buildOutput: vkBuild.output,
      runOutput: "",
      runExitCode: null,
      coherentText: false,
      garbageOutput: false,
      tokensGenerated: 0,
      bandwidthUtil: null,
      effectiveBW: null,
      error: "vulkan build failed",
      backendFlagRecognized: true,
    };
    console.log(clr("1;31", "  ❌ vulkan build failed — A/B incomplete"));
  } else {
    vulkan = await benchmarkBinary(
      REMOTE_VULKAN_BIN,
      modelPath,
      prompt,
      "RADV_PERFTEST=coop_matrix",
      runsPerBinary,
    );
    vulkan.buildExitCode = vkBuild.exitCode;
    vulkan.buildOutput = vkBuild.output;
  }

  // Build zinc_rt
  const rtBuild = await remoteBuild("zinc_rt");
  let zinc_rt: BenchmarkResult;
  if (!rtBuild.flagRecognized) {
    // build.zig doesn't yet know about -Dbackend=zinc_rt — we're at M0
    console.log(clr("1;33", "  ⚠ -Dbackend=zinc_rt flag not recognized by build.zig — pre-M0 state"));
    zinc_rt = {
      decodeTps: null,
      prefillTps: null,
      decodeSamples: [],
      prefillSamples: [],
      buildExitCode: rtBuild.exitCode,
      buildOutput: rtBuild.output,
      runOutput: "",
      runExitCode: null,
      coherentText: false,
      garbageOutput: false,
      tokensGenerated: 0,
      bandwidthUtil: null,
      effectiveBW: null,
      error: "zinc_rt backend flag not yet recognized",
      backendFlagRecognized: false,
    };
  } else if (rtBuild.exitCode !== 0) {
    zinc_rt = {
      decodeTps: null,
      prefillTps: null,
      decodeSamples: [],
      prefillSamples: [],
      buildExitCode: rtBuild.exitCode,
      buildOutput: rtBuild.output,
      runOutput: "",
      runExitCode: null,
      coherentText: false,
      garbageOutput: false,
      tokensGenerated: 0,
      bandwidthUtil: null,
      effectiveBW: null,
      error: "zinc_rt build failed",
      backendFlagRecognized: true,
    };
    console.log(clr("1;31", "  ❌ zinc_rt build failed"));
  } else {
    // Use ZINC_RT_TIER if set, else let auto-detect pick (T2 on 6.16+, T1 on older)
    const tier = process.env.ZINC_RT_TIER ?? ENV.ZINC_RT_TIER ?? "";
    const envFlags = tier ? `RADV_PERFTEST=coop_matrix ZINC_RT_TIER=${tier}` : "RADV_PERFTEST=coop_matrix";
    zinc_rt = await benchmarkBinary(
      REMOTE_ZINC_RT_BIN,
      modelPath,
      prompt,
      envFlags,
      runsPerBinary,
    );
    zinc_rt.buildExitCode = rtBuild.exitCode;
    zinc_rt.buildOutput = rtBuild.output;
  }

  const ratio =
    vulkan.decodeTps != null &&
      vulkan.decodeTps > 0 &&
      zinc_rt.decodeTps != null
      ? zinc_rt.decodeTps / vulkan.decodeTps
      : null;
  const prefillRatio =
    vulkan.prefillTps != null &&
      vulkan.prefillTps > 0 &&
      zinc_rt.prefillTps != null
      ? zinc_rt.prefillTps / vulkan.prefillTps
      : null;

  return { vulkan, zinc_rt, ratio, prefillRatio };
}

// ── Phase detection ──────────────────────────────────────────────────

export function detectPhase(ab: ABBenchmark): Phase {
  // If zinc_rt doesn't yet build or run, we're migrating.
  if (!ab.zinc_rt.backendFlagRecognized) return "migrate";
  if (ab.zinc_rt.buildExitCode !== 0) return "migrate";
  if (ab.zinc_rt.runExitCode !== 0 && ab.zinc_rt.runExitCode !== null) return "migrate";
  if (!ab.zinc_rt.coherentText) return "migrate";
  if (detectZincRtExecutionMode(ab.zinc_rt) !== "direct") return "migrate";
  // If we're well below vulkan, still migrating
  if (ab.ratio != null && ab.ratio < COMPETITIVE_THRESHOLD) return "migrate";
  return "optimize";
}

// ── Claude stream formatter ──────────────────────────────────────────

type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDeltaInCurrentMessage: boolean;
};

const MAX_DIFF_LINES = 8;

function formatToolInput(toolName: string, inputJson: string): string {
  let input: Record<string, unknown> = {};
  try { input = JSON.parse(inputJson) as Record<string, unknown>; } catch { /* partial */ }
  const name = toolName.toLowerCase();
  const out: string[] = [];

  if (name === "edit") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
    const oldLines = ((input.old_string as string | undefined) ?? "").split("\n");
    const newLines = ((input.new_string as string | undefined) ?? "").split("\n");
    for (const l of oldLines.slice(0, MAX_DIFF_LINES)) out.push(clr("31", `   - ${l}`));
    if (oldLines.length > MAX_DIFF_LINES) out.push(clr("2", `   - … (${oldLines.length - MAX_DIFF_LINES} more)`));
    for (const l of newLines.slice(0, MAX_DIFF_LINES)) out.push(clr("32", `   + ${l}`));
    if (newLines.length > MAX_DIFF_LINES) out.push(clr("2", `   + … (${newLines.length - MAX_DIFF_LINES} more)`));
  } else if (name === "write") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const lineCount = ((input.content as string | undefined) ?? "").split("\n").length;
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")} (${lineCount} lines)`));
  } else if (name === "bash") {
    const cmd = (input.command as string | undefined) ?? "?";
    out.push(clr("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "…" : cmd}`));
  } else if (name === "read") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const offset = input.offset != null ? ` @line ${input.offset}` : "";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}${offset}`));
  } else if (name === "grep") {
    const pattern = (input.pattern as string | undefined) ?? "?";
    const path = (input.path as string | undefined) ?? "";
    out.push(clr("2", ` → /${pattern}/${path ? ` in ${path.split("/").slice(-2).join("/")}` : ""}`));
  } else if (name === "glob") {
    out.push(clr("2", ` → ${(input.pattern as string | undefined) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

function coerceDisplayText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    const parts = value.map((e) => coerceDisplayText(e)).filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(value, null, 2); } catch { return ""; }
  }
  if (typeof value === "object") {
    const r = value as Record<string, unknown>;
    const parts = [r.text, r.message, r.output, r.stdout, r.stderr, r.content, r.result, r.output_text]
      .map((e) => coerceDisplayText(e)).filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(r, null, 2); } catch { return ""; }
  }
  return "";
}

function formatToolResult(result: Record<string, unknown>): string {
  const file = result.file as Record<string, unknown> | undefined;
  if (file) return clr("32", `   ☑ accepted`) + clr("2", `  (${file.numLines ?? "?"} lines)`) + "\n";
  const content = coerceDisplayText(result.content);
  if (!content.trim()) return clr("32", "   ☑ accepted") + "\n";
  const lines = content.split("\n").filter((l) => l.trim());
  const tail = lines.slice(-3);
  const ellipsis = lines.length > 3 ? clr("2", "   …\n") : "";
  const body = tail.map((l) => clr("2", `   ${l.trim()}`)).join("\n");
  return clr("32", "   ☑ accepted") + "\n" + ellipsis + body + "\n";
}

function formatClaudeStreamLine(rawLine: string, state: ClaudeStreamState): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return rawLine + "\n"; }

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
        state.sawTextDeltaInCurrentMessage = true;
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
    if (result) return formatToolResult(result);
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
      if (!text.trim()) return null;
      if (state.sawTextDeltaInCurrentMessage) {
        state.sawTextDeltaInCurrentMessage = false;
        return null;
      }
      return clr("96", text) + "\n";
    }
    return null;
  }
  return null;
}

// ── Codex stream formatter ──────────────────────────────────────────

function formatCodexStreamLine(rawLine: string): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try {
    event = JSON.parse(rawLine) as Record<string, unknown>;
  } catch {
    return rawLine.trim().startsWith("@@@") ? `${clr("96", rawLine.trim())}\n` : null;
  }

  const type = typeof event.type === "string" ? event.type : "";
  if (type === "error") {
    const text = coerceDisplayText(event.message ?? event.content);
    return text ? `${clr("31", text)}\n` : null;
  }
  if (type === "item.completed") {
    const item = event.item as Record<string, unknown> | undefined;
    if (item?.type === "agent_message") {
      const text = coerceDisplayText(item.text ?? item.message ?? item.output_text ?? item.content);
      return text ? `${clr("96", text)}\n` : null;
    }
    if (item?.type === "reasoning") {
      const text = coerceDisplayText(item.summary ?? item.text ?? item.message ?? item.content);
      return text ? `${clr("2", `thinking: ${text}`)}\n` : null;
    }
  }
  return null;
}

function extractAgentText(stdout: string): string {
  const texts: string[] = [];
  for (const line of stdout.split("\n")) {
    if (!line.trim()) continue;
    try {
      const event = JSON.parse(line) as Record<string, unknown>;
      const type = event.type;
      if (type === "assistant") {
        const content = (event.message as Record<string, unknown> | undefined)?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            const text = (block as Record<string, unknown>)?.text;
            if (typeof text === "string" && text.trim()) texts.push(text);
          }
        }
      } else if (type === "message" || type === "agent") {
        const text = coerceDisplayText(event.content ?? event.message);
        if (text.trim()) texts.push(text);
      } else if (type === "item.completed") {
        const item = event.item as Record<string, unknown> | undefined;
        if (item?.type === "agent_message") {
          const text = coerceDisplayText(item.text ?? item.message ?? item.output_text ?? item.content);
          if (text.trim()) texts.push(text);
        }
      }
    } catch {
      if (line.trim().startsWith("@@@")) texts.push(line.trim());
    }
  }
  return texts.join("\n");
}

// ── Agent invocation ─────────────────────────────────────────────────

function buildClaudeArgs(prompt: string): string[] {
  return [
    "-p",
    "--verbose",
    "--output-format", "stream-json",
    "--include-partial-messages",
    `--disallowed-tools=${[...BLOCKED_GIT_OPS, ...BLOCKED_FILE_OPS].join(",")}`,
    "--permission-mode", "bypassPermissions",
    "--model", CLAUDE_MODEL,
    "--effort", CLAUDE_EFFORT,
    prompt,
  ];
}

function buildCodexArgs(prompt: string): string[] {
  return [
    "exec",
    "-c",
    `model_reasoning_effort="${CODEX_REASONING_EFFORT}"`,
    "--skip-git-repo-check",
    "--json",
    "--color", "never",
    "--sandbox", "danger-full-access",
    "--cd", REPO_ROOT,
    "--model", CODEX_MODEL,
    prompt,
  ];
}

async function runAgent(
  agent: AgentKind,
  prompt: string,
): Promise<RunResult> {
  const label = agent === "codex" ? "Codex" : "Claude";
  console.log(clr("1;34", SEP));
  console.log(clr("1;34", `  🧠 PROMPT (${label})`));
  console.log(clr("1;34", SEP));
  const promptLines = prompt.split("\n");
  for (const line of promptLines.slice(0, 18))
    process.stdout.write(clr("2", line) + "\n");
  if (promptLines.length > 18)
    process.stdout.write(clr("2", `… (${promptLines.length - 18} more lines)\n`));
  console.log(clr("1;34", SEP));

  console.log(clr("1;36", SEP));
  console.log(clr("1;36", `  💬 RESPONSE (${label})`));
  console.log(clr("1;36", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(
      clr("2", `\n⏳ still running (${formatElapsed(startedAt)} elapsed)...\n`),
    );
  }, 30_000);

  const claudeState: ClaudeStreamState = {
    currentToolName: null,
    currentBlockIsToolUse: false,
    inputJsonBuffer: "",
    inTextBlock: false,
    sawTextDeltaInCurrentMessage: false,
  };

  const result =
    agent === "codex"
      ? await runCommand("codex", buildCodexArgs(prompt), {
        streamOutput: true,
        timeout: AGENT_TIMEOUT_MS,
        stdoutLineFormatter: formatCodexStreamLine,
      })
      : await runCommand("claude", buildClaudeArgs(prompt), {
        streamOutput: true,
        timeout: AGENT_TIMEOUT_MS,
        stdoutLineFormatter: (line) => formatClaudeStreamLine(line, claudeState),
      });

  clearInterval(heartbeat);
  console.log(clr("1;36", SEP));
  console.log(clr("1;32", `  ✅ ${label} done in ${formatElapsed(startedAt)}`));
  console.log(clr("1;36", SEP));

  return result;
}

// ── Prompt builder ───────────────────────────────────────────────────

function bestRatioOfState(state: RunState): number | null {
  let best: number | null = null;
  for (const c of state.cycles) {
    if (c.kept && c.afterAB.ratio != null) {
      if (best == null || c.afterAB.ratio > best) best = c.afterAB.ratio;
    }
  }
  return best;
}

function bestZincRtTpsOfState(state: RunState): number | null {
  let best: number | null = null;
  for (const c of state.cycles) {
    if (c.kept && c.afterAB.zinc_rt.decodeTps != null) {
      if (best == null || c.afterAB.zinc_rt.decodeTps > best)
        best = c.afterAB.zinc_rt.decodeTps;
    }
  }
  return best;
}

function consecutiveReverts(cycles: CycleResult[], maxWindow = HARD_PIVOT_STALL_CYCLES): number {
  let count = 0;
  for (let i = cycles.length - 1; i >= Math.max(0, cycles.length - maxWindow); i--) {
    if (!cycles[i].kept) count++;
    else break;
  }
  return count;
}

function hasM1ValidationSignal(result: BenchmarkResult): boolean {
  const out = result.runOutput;
  if (!out) return false;
  return (
    /ZINC_RT M1.*?(passed|success|retired)/i.test(out) ||
    /T2 UMQ admission passed/i.test(out) ||
    /USERQ_(CREATE|FREE).*?(ok|success|passed)/i.test(out) ||
    /UMQ.*?(queue|fence|nop|write_data).*?(ok|success|passed|retired)/i.test(out) ||
    /protected[- ]fence.*?(ok|success|passed|retired)/i.test(out)
  );
}

export type ZincRtExecutionMode =
  | "missing"
  | "cpu"
  | "cpu_after_admission"
  | "host_assisted"
  | "vulkan_compat"
  | "direct"
  | "unknown";

export function isZincRtBenchmarkShortcut(output: string): boolean {
  return (
    /\bpath clamped decode budget\b/i.test(output) ||
    /\bdecode MoE top-k lowered to 0\b/i.test(output) ||
    /\bdecode LM-head row scan capped\b/i.test(output) ||
    /\bshortcut_free\s*=\s*0\b/i.test(output) ||
    /\bbenchmark_shortcuts\s*=\s*(?!none\b)[a-z0-9_,:-]+/i.test(output)
  );
}

function hasHostAssistedEvidence(output: string): boolean {
  return (
    /\bhost-assisted\b/i.test(output) ||
    /\bhost_assisted\b/i.test(output) ||
    /\bmodel_execution\s*=\s*hybrid_direct_compute\b/i.test(output) ||
    /\bmodel_execution\s*=\s*host_assisted_[a-z0-9_:-]+\b/i.test(output) ||
    /\breal_model_slice\s*=\s*0\b/i.test(output) ||
    isZincRtBenchmarkShortcut(output)
  );
}

function hasDirectComputeEvidence(output: string): boolean {
  if (hasHostAssistedEvidence(output)) return false;
  if (/\breal_model_slice\s*=\s*0\b/i.test(output)) return false;
  if (/\bmodel_execution\s*=\s*direct\b/i.test(output)) {
    return /\breal_model_slice\s*=\s*1\b/i.test(output) || hasDirectModelSliceEvidence(output);
  }
  return hasDirectModelSliceEvidence(output);
}

function hasDirectModelSliceEvidence(output: string): boolean {
  const match = output.match(/\bdirect_compute_kind\s*=\s*([a-z0-9_:-]+)/i);
  if (match == null) return false;
  const kind = match[1].toLowerCase();
  if (/^(argmax|rms_norm_elem0|argmax_rms_norm_elem0)$/.test(kind)) return false;
  if (!/(router|dmmv|matvec|lm_head|moe|expert|ssm|qkv|ffn)/.test(kind)) return false;
  return (
    /\bconsumed_gpu_compute_value\s*=\s*1\b/i.test(output) ||
    /\bconsumed_gpu_model_value\s*=\s*1\b/i.test(output)
  );
}

function hasScalarFallbackEvidence(output: string): boolean {
  return (
    /\bmodel_execution\s*=\s*cpu_fallback\b/i.test(output) ||
    /\bforward_zinc_rt\b.*\bM1 scalar\b/i.test(output) ||
    /\bforward_zinc_rt\b.*\bscalar path\b/i.test(output) ||
    /\bM1 scalar\b/i.test(output) ||
    /execution_tier\s*=\s*t_cpu\b/i.test(output) ||
    /\bforward_zinc_rt\b.*\bT-CPU\b/i.test(output) ||
    /\bM0 T-CPU\b/i.test(output)
  );
}

function hasDirectAdmissionEvidenceInOutput(output: string): boolean {
  return (
    /T1 KFD.*admission passed/i.test(output) ||
    /AMDGPU CS.*retired/i.test(output) ||
    /PM4.*WRITE_DATA.*retired/i.test(output) ||
    /direct_token_boundary\s*=/i.test(output) ||
    /T2 UMQ admission passed/i.test(output) ||
    /USERQ_(CREATE|FREE).*?(ok|success|passed)/i.test(output)
  );
}

export function detectZincRtExecutionMode(resultOrOutput: Pick<BenchmarkResult, "runOutput"> | string): ZincRtExecutionMode {
  const out = typeof resultOrOutput === "string" ? resultOrOutput : resultOrOutput.runOutput;
  if (!out.trim()) return "missing";

  if (/\bmodel_execution\s*=\s*vulkan_compat\b/i.test(out)) {
    return "vulkan_compat";
  }

  // This is the critical state in current ZINC_RT: queue admission and PM4
  // smokes pass, but model tokens still come from forward_zinc_rt's CPU path.
  if (/execution_tier\s*=\s*t_cpu_after_admission\b/i.test(out)) {
    return "cpu_after_admission";
  }

  // A direct tier label or COPY_DATA model-value gate is not enough: the
  // current 47-48 tok/s path still logs `M1 scalar` while using PM4 only for
  // copies. Call it direct only once a real compute slice is benchmark-visible.
  if (hasScalarFallbackEvidence(out)) {
    return hasDirectAdmissionEvidenceInOutput(out) ? "cpu_after_admission" : "cpu";
  }

  if (hasHostAssistedEvidence(out)) {
    return "host_assisted";
  }

  if (hasDirectComputeEvidence(out)) {
    return "direct";
  }

  if (/execution_tier\s*=\s*(t1_pm4|t2_umq|t_metal|t_intel|t_cuda|direct|gpu)\b/i.test(out)) {
    return "unknown";
  }

  return "unknown";
}

function zincRtExecutionModeLabel(mode: ZincRtExecutionMode): string {
  switch (mode) {
    case "cpu_after_admission":
      return "CPU fallback after direct-queue admission";
    case "cpu":
      return "CPU fallback";
    case "host_assisted":
      return "host-assisted CPU decode with direct probes";
    case "vulkan_compat":
      return "Vulkan compatibility tier";
    case "direct":
      return "direct runtime tier";
    case "missing":
      return "no run output";
    case "unknown":
      return "unknown";
  }
}

function isZincRtCpuExecuted(result: BenchmarkResult): boolean {
  const mode = detectZincRtExecutionMode(result);
  return mode === "cpu" || mode === "cpu_after_admission" || mode === "host_assisted";
}

function isZincRtDirectExecuted(result: BenchmarkResult): boolean {
  return detectZincRtExecutionMode(result) === "direct";
}

function hasDirectAdmissionEvidence(result: BenchmarkResult): boolean {
  return hasDirectAdmissionEvidenceInOutput(result.runOutput);
}

function hasNewM1ValidationSignal(before: ABBenchmark, after: ABBenchmark): boolean {
  return !hasM1ValidationSignal(before.zinc_rt) && hasM1ValidationSignal(after.zinc_rt);
}

function hasNewDirectExecutionSignal(before: ABBenchmark, after: ABBenchmark): boolean {
  return !isZincRtDirectExecuted(before.zinc_rt) && isZincRtDirectExecuted(after.zinc_rt);
}

function hasNewDirectModelSliceSignal(before: ABBenchmark, after: ABBenchmark): boolean {
  return !hasDirectModelSliceEvidence(before.zinc_rt.runOutput) && hasDirectModelSliceEvidence(after.zinc_rt.runOutput);
}

function foundationPerformanceEnvelopeOk(before: ABBenchmark, after: ABBenchmark): boolean {
  if (after.zinc_rt.runExitCode !== 0 || !after.zinc_rt.coherentText) return false;
  if (before.vulkan.coherentText && !after.vulkan.coherentText) return false;
  const beforeRt = before.zinc_rt.decodeTps;
  const afterRt = after.zinc_rt.decodeTps;
  if (beforeRt == null || afterRt == null) return true;
  return afterRt >= beforeRt * (1 - FOUNDATION_MAX_TPS_REGRESSION);
}

function revertReason(cycle: CycleResult): string {
  if (cycle.error) return cycle.error;
  if (cycle.afterAB.zinc_rt.runExitCode !== 0 && cycle.afterAB.zinc_rt.runExitCode != null) {
    return `zinc_rt exit ${cycle.afterAB.zinc_rt.runExitCode}`;
  }
  if (cycle.beforeAB.zinc_rt.coherentText && !cycle.afterAB.zinc_rt.coherentText) {
    return "lost coherent zinc_rt output";
  }
  if (cycle.beforeAB.vulkan.coherentText && !cycle.afterAB.vulkan.coherentText) {
    return "lost coherent vulkan output";
  }
  if (cycle.beforeAB.ratio != null && cycle.afterAB.ratio != null) {
    const delta = cycle.afterAB.ratio - cycle.beforeAB.ratio;
    return `flat/regressed ratio (${delta >= 0 ? "+" : ""}${delta.toFixed(4)})`;
  }
  return "no measurable progress";
}

function formatFailedApproach(cycle: CycleResult): string {
  return `#${cycle.cycle} ${cycle.description} — ${revertReason(cycle)}`;
}

function buildGapAnalysis(state: RunState, before: ABBenchmark): string {
  const lines: string[] = [];
  const mode = detectZincRtExecutionMode(before.zinc_rt);
  const recent = state.cycles.slice(-12);
  const markerAttempts = recent.filter((c) =>
    /pm4|kfd|amdgpu|cs|fence|doorbell|write_data|stream_out|marker|admission|umq|userq/i.test(c.description),
  ).length;
  const scalarAttempts = recent.filter((c) =>
    /t-cpu|scalar|cpu|q4_0|q8_0|top-k|topk|moe|lm_head|thread|worker|dot|requant|matvec/i.test(c.description),
  ).length;
  const t2Failures = state.failedApproaches.filter((f) => /t2|umq|userq/i.test(f)).length;
  const bestRt = bestZincRtTpsOfState(state);

  if (mode === "cpu_after_admission") {
    lines.push("- Execution gap: direct queue admission is working, but generated tokens still come from `execution_tier=t_cpu_after_admission` / `forward_zinc_rt M0 T-CPU`.");
  } else if (mode === "cpu") {
    lines.push("- Execution gap: generated tokens still come from the T-CPU path.");
  } else if (mode === "host_assisted") {
    lines.push("- Execution gap: current ZINC_RT tokens are host-assisted. The direct PM4/CS snippets are boundary probes, while the hot decode matvecs still run in `forward_zinc_rt` CPU code.");
  }

  if (hasDirectAdmissionEvidence(before.zinc_rt) && mode !== "direct") {
    lines.push("- PM4/KFD gap: admission and fence/WRITE_DATA smoke tests are no longer the blocker; the missing step is replacing a CPU-produced model value with a GPU-produced value in the exercised forward path.");
  }

  if (isZincRtBenchmarkShortcut(before.zinc_rt.runOutput)) {
    lines.push("- Benchmark-quality gap: current zinc_rt output uses a shortcut such as top-k=0, a capped LM-head row scan, or a clamped decode budget. Treat tok/s as a bring-up signal, not proof that the model path can beat Vulkan.");
  }

  if (recent.length > 0 && markerAttempts >= Math.max(4, Math.ceil(recent.length / 3))) {
    lines.push(`- Cycle-pattern gap: ${markerAttempts}/${recent.length} recent cycles were PM4/UMQ/admission/marker work. More markers are unlikely to move tok/s unless they execute a real model slice.`);
  }

  if (recent.length > 0 && scalarAttempts >= Math.max(4, Math.ceil(recent.length / 3)) && mode !== "direct") {
    lines.push(`- CPU-local gap: ${scalarAttempts}/${recent.length} recent cycles still attacked the CPU fallback. That can improve the benchmark number, but it does not move ZINC_RT toward M1 direct execution.`);
  }

  if (t2Failures >= 4) {
    lines.push(`- UMQ gap: ${t2Failures} recorded failed approaches involve T2/UMQ/USERQ. Do not pivot to T2 again unless the cycle first proves kernel/user_queue and USERQ_CREATE availability on this node.`);
  }

  if (bestRt != null && before.zinc_rt.decodeTps != null && mode !== "direct") {
    lines.push(`- Keep-threshold gap: best CPU-fallback zinc_rt sample is ${bestRt.toFixed(1)} tok/s; current sample is ${before.zinc_rt.decodeTps.toFixed(1)} tok/s. Treat smaller moves as noise unless they beat the best checkpoint by a material margin.`);
  }

  return lines.length > 0 ? lines.map((line) => `  ${line}`).join("\n") : "  (none detected)";
}

function buildPrompt(state: RunState, before: ABBenchmark, phase: Phase): string {
  const trunc = (s: string, max: number) =>
    s && s.length > max ? s.slice(0, max) + "…" : s;

  const ratioStr = before.ratio != null
    ? `${before.ratio.toFixed(3)} (${(before.ratio * 100).toFixed(1)}%)`
    : "n/a";
  const vkTps = before.vulkan.decodeTps?.toFixed(1) ?? "n/a";
  const rtTps = before.zinc_rt.decodeTps?.toFixed(1) ?? "n/a";
  const vkCoh = before.vulkan.coherentText ? "✓" : "✗";
  const rtCoh = before.zinc_rt.coherentText ? "✓" : "✗";
  const rtExecutionMode = detectZincRtExecutionMode(before.zinc_rt);
  const rtExecutionLabel = zincRtExecutionModeLabel(rtExecutionMode);

  const historyBlock =
    state.cycles.length > 0
      ? state.cycles
        .slice(-12)
        .map((h) => {
          const desc = trunc(h.description, 64);
          const rt = h.afterAB.zinc_rt.decodeTps?.toFixed(1) ?? "—";
          const vk = h.afterAB.vulkan.decodeTps?.toFixed(1) ?? "—";
          const ratio = h.afterAB.ratio?.toFixed(2) ?? "—";
          const agentTag = h.agent ? `[${h.agent}] ` : "";
          return `  #${h.cycle}: ${agentTag}[${h.phase}] ${desc} → ${h.kept ? "KEPT" : "REVERTED"} (rt=${rt}, vk=${vk}, ratio=${ratio})`;
        })
        .join("\n")
      : "  (none yet)";

  const ideasBlock = state.ideas.length > 0
    ? state.ideas.slice(-12).map((i, n) => `  ${n + 1}. ${trunc(i, 110)}`).join("\n")
    : "  (none yet)";

  const failedBlock = state.failedApproaches.length > 0
    ? state.failedApproaches.slice(-15).map((f, n) => `  ${n + 1}. ${trunc(f, 320)}`).join("\n")
    : "  (none yet)";

  const bestRatio = bestRatioOfState(state);
  const bestRt = bestZincRtTpsOfState(state);
  const gapAnalysis = buildGapAnalysis(state, before);

  // Decide milestone hint based on observable state
  let milestoneHint = "";
  if (!before.zinc_rt.backendFlagRecognized) {
    milestoneHint = `**Current milestone: PRE-M0.** \`build.zig\` does not yet accept \`-Dbackend=zinc_rt\`. Your first job is to add that build option. See §6.2 of \`docs/ZINC_RT_DESIGN.md\` — under the new tree layout, the build needs to wire \`src/zinc_rt/\` and route \`src/gpu/interface.zig\` based on the selected backend. For now, \`src/zinc_rt/\` may be empty — that is fine; the build can produce a binary that errors out at runtime saying "ZINC_RT not yet implemented". The point of this step is to make the build flag exist so subsequent cycles can fill in the implementation.`;
  } else if (before.zinc_rt.buildExitCode !== 0) {
    milestoneHint = `**Current milestone: M0 (T-CPU bring-up).** Build flag works but compilation fails. Fix the build errors shown below. T-CPU (\`src/zinc_rt/ring/cpu.zig\`) is the first tier to land — a pure Zig implementation of the IR, used as the validation oracle. It does not need to be fast; it needs to be correct and to build.`;
  } else if (before.zinc_rt.runExitCode !== 0 && before.zinc_rt.runExitCode !== null) {
    milestoneHint = `**Current milestone: M0 (T-CPU bring-up).** Build OK but runtime crashes. Fix the crash. Most likely a missing IR op implementation, a buffer-binding bug, or a panic in \`forward_zinc_rt.zig\`.`;
  } else if (isZincRtBenchmarkShortcut(before.zinc_rt.runOutput)) {
    milestoneHint = `**Current milestone: M1 host-assisted shortcut gap (not M2/M3 yet).** ZINC_RT is producing a tok/s number, but the exercised path is still benchmark-shortcut host code. The current output contains at least one of:
  - \`decode MoE top-k lowered to 0\`
  - \`decode LM-head row scan capped\`
  - \`path clamped decode budget\`

This means the ratio is a bring-up scalar, not evidence that the full decode path is competitive. Do not chase continuous batching, descriptor churn, tiny PM4 probes, or RDNA opcode tuning until a real model slice runs on the GPU and the prompt consumes it.

Read \`${ZINC_RT_DIRECT_DECODE_EFFORT}\` before changing code. The measured hot path is CPU \`matvecRawDirectSerial\`; the useful next step is to lower a real DMMV/router/LM-head row range behind T1/T2 and consume that GPU-produced value, or emit a measured-dead report with remote evidence.`;
  } else if (!before.zinc_rt.coherentText) {
    milestoneHint = `**Current milestone: M0/M1 (correctness).** ZINC_RT runs but output is not coherent. This is a forward-pass correctness bug, exactly the class of issue T-CPU exists to catch (§15 of the design). Either:
  - your T-CPU implementation of one or more IR ops is wrong, or
  - the IR being emitted by \`forward_zinc_rt.zig\` doesn't match what \`forward.zig\` (legacy) does for the same inputs.
Compare against \`forward.zig\` op-by-op. Use the existing CPU reference in \`forward.zig::runDecodeStepReference\` as a second oracle.`;
  } else if (rtExecutionMode !== "direct") {
    const admissionText = hasDirectAdmissionEvidence(before.zinc_rt)
      ? "T1/KFD/AMDGPU-CS admission and fence smokes are already benchmark-visible."
      : "Direct queue admission is not yet benchmark-visible.";
    milestoneHint = `**Current milestone: M1 execution gap (not M2 yet).** ZINC_RT is coherent, but the exercised model path is still **${rtExecutionLabel}**. ${admissionText}

The ratio is therefore not proof that M1 decode execution exists. Do not chase M2 descriptor churn, submission-count tuning, or CPU scalar matvec/dequant/threading work while the output has \`model_execution=cpu_fallback\`, \`M1 scalar\`, or \`direct_compute_ops=0\`.

Copy gates such as \`direct_token_boundary=amdgpu_cs_copy_data\`, copied scalar model values, \`argmax_top2\`, and \`rms_norm_elem0\` prove PM4/CS dataflow, not full shader execution. The next accepted step should expose a benchmark-visible model slice such as \`direct_compute_kind=router_row_range\`, \`dmmv_row_range\`, or full/validated LM-head row range, and the prompt output must consume that value.

Next useful work must do one of these:
  - lower the smallest real model slice behind T1/T2 and verify the prompt output uses that GPU-computed value;
  - wire one router/DMMV/LM-head row-range verifier whose result is consumed by current generation;
  - or emit a measured-dead report with remote evidence explaining exactly why direct execution cannot advance on this node.`;
  } else if (before.ratio != null && before.ratio < 0.3) {
    milestoneHint = `**Current milestone: M0→M1 transition.** ZINC_RT is coherent but very slow (ratio ${(before.ratio * 100).toFixed(0)}% of Vulkan). This usually means you're still on T-CPU. To move to M1, implement the T2 UMQ tier (\`src/zinc_rt/ring/umq.zig\`) — it's the easier of the two AMD direct-submission paths and ships on kernel 6.16+. See §14 of the design doc.`;
  } else if (before.ratio != null && before.ratio < COMPETITIVE_THRESHOLD) {
    milestoneHint = `**Current milestone: M1→M2 transition.** ZINC_RT is running but trails Vulkan at ${(before.ratio * 100).toFixed(0)}%. Likely fixes:
  - Reduce per-token submits (target: 1 submit per decode token).
  - Eliminate descriptor-set churn (use the push-constant-only path from §11.3).
  - Add the T1 PM4 KFD path if not already; M2 ships T1 alongside T2 (§13).`;
  } else if (before.ratio != null && before.ratio < 1.0) {
    milestoneHint = `**Current milestone: M2→M3.** ZINC_RT is close — ${(before.ratio * 100).toFixed(0)}% of Vulkan. To cross 100%, you need one of:
  - Continuous-batching scheduler (§18) — even with one slot, the slot-table + zero-recompile graph reduces per-token overhead vs Vulkan's command-buffer model.
  - Paged KV v2 (§19) — GPU-resident metadata, fewer host roundtrips.
  - Chunked prefill if prefill is in the benchmark path.`;
  } else if (before.ratio != null && before.ratio < 1.2) {
    milestoneHint = `**Current milestone: M3→M4.** ZINC_RT now BEATS Vulkan at ${(before.ratio * 100).toFixed(0)}%. To widen the gap toward the M4 target (1.5× of today, 220 tok/s on Qwen 3.6 35B-A3B):
  - Start porting top-time kernels from SPIR-V to RDNA4 GAS asm under \`src/zinc_rt/isa/gfx1201/\` (§11.4, §17).
  - WMMA path on prefill if prefill is in the benchmark.`;
  } else {
    milestoneHint = `**Current milestone: M4→M5.** ZINC_RT is at ${(before.ratio * 100).toFixed(0)}% of Vulkan. The remaining win comes from the megakernel (§21) — collapsing the entire forward pass into one persistent kernel. This is unmapped territory on RDNA4. See §21 for the staged approach.`;
  }

  // Phase-specific diagnosis
  const diagnosis: string[] = [];
  if (phase === "migrate") {
    diagnosis.push("## Status: MIGRATE — getting ZINC_RT to a competitive working state");
    diagnosis.push("");
    diagnosis.push(`vulkan tok/s: **${vkTps}** ${vkCoh}`);
    diagnosis.push(`zinc_rt tok/s:       **${rtTps}** ${rtCoh}`);
    diagnosis.push(`ratio (zinc_rt / vulkan): **${ratioStr}**`);
    if (before.zinc_rt.error) {
      diagnosis.push(`error: ${before.zinc_rt.error}`);
    }
    diagnosis.push("");
    diagnosis.push(milestoneHint);
  } else {
    diagnosis.push("## Status: OPTIMIZE — beating Vulkan on RDNA4");
    diagnosis.push("");
    diagnosis.push(`vulkan tok/s: **${vkTps}**`);
    diagnosis.push(`zinc_rt tok/s:       **${rtTps}**`);
    diagnosis.push(`ratio: **${ratioStr}**`);
    if (bestRatio != null) {
      diagnosis.push(`best ratio so far: **${bestRatio.toFixed(3)}** (${(bestRatio * 100).toFixed(1)}%)`);
    }
    if (bestRt != null) {
      diagnosis.push(`best zinc_rt tok/s so far: **${bestRt.toFixed(1)}**`);
    }
    if (before.zinc_rt.bandwidthUtil != null) {
      diagnosis.push(`zinc_rt modeled BW: ${before.zinc_rt.effectiveBW?.toFixed(0) ?? "?"} GB/s (${before.zinc_rt.bandwidthUtil.toFixed(0)}% of 576 GB/s)`);
    }
    diagnosis.push("");
    diagnosis.push(milestoneHint);
  }

  // Stall detection
  const stallCount = consecutiveReverts(state.cycles);
  const nonDirectExecuted = !isZincRtDirectExecuted(before.zinc_rt);
  const hardPivot =
    phase === "migrate" &&
    stallCount >= HARD_PIVOT_STALL_CYCLES &&
    before.zinc_rt.coherentText &&
    nonDirectExecuted;

  // Build/run output for debugging
  const buildOut = (phase === "migrate" && before.zinc_rt.buildExitCode !== 0)
    ? before.zinc_rt.buildOutput.slice(-3500)
    : before.vulkan.buildExitCode !== 0
      ? before.vulkan.buildOutput.slice(-2500)
      : "";
  const runOut = before.zinc_rt.runOutput
    ? before.zinc_rt.runOutput.slice(-3000)
    : before.vulkan.runOutput.slice(-2000);

  return [
    `# ZINC_RT ${phase.toUpperCase()} Task`,
    "",
    ...diagnosis,
    "",
    "## The Goal",
    "",
    "Make `zinc_rt` (our own GPU runtime, see `docs/ZINC_RT_DESIGN.md`) beat `vulkan` tok/s on the validated catalog model.",
    "",
    "Every cycle this loop benchmarks BOTH backends head-to-head on the same RDNA4 node, same prompt, same model. The ratio (zinc_rt / vulkan) is the north star only after the run proves a real model slice is GPU-produced and consumed. Tiny PM4 probes such as top-2 argmax or one RMS element are validation signals, not proof that the decode path can beat Vulkan. Above 1.0 means we're winning only when benchmark shortcuts are absent.",
    "",
    "## Current Deep Profile",
    "",
    `Read \`${ZINC_RT_DIRECT_DECODE_EFFORT}\` before editing performance code.`,
    "- 2026-05-16 clean-HEAD profile: default host-assisted zinc_rt is ~79 tok/s with 4K LM-head, top-k=0 after prefill, and an 8-token decode clamp.",
    "- Full-vocab LM-head drops the same path to ~60 tok/s. Extending decode to 256 tokens drops to ~68 tok/s and repeats `city of the city`, so the 8-token run overstates quality and steady-state speed.",
    "- Decode-only `perf record` on a 256-token scratch run: ~78% of samples are `forward_zinc_rt.matvecRawDirectSerial`; ~4% are `runSsmHeadRange`. The current bottleneck is CPU matvec, not GPU opcode quality.",
    "- The native gfx1201 snippets in `src/zinc_rt/ring/cs.zig` are one-wave `argmax_top2` and `rms_norm_elem0` probes. Tuning those opcodes cannot move the model above 120 tok/s.",
    "",
    "## The Design (you MUST read this first)",
    "",
    "`docs/ZINC_RT_DESIGN.md` is the contract. It defines:",
    "- The six tiers (T1 PM4, T2 UMQ, T-CPU, T-Metal, T-Intel, T-CUDA)",
    "- The IR (Appendix B)",
    "- The kernel ABI (§11)",
    "- The submission model (§12)",
    "- The 8-milestone plan (§25)",
    "- The source-tree layout (§6.2, Appendix C)",
    "",
    "Do NOT redesign. Follow the design. If the design has a gap that blocks your work, surface it in `@@@SELF_ANALYSIS` and propose a minimal extension — do not unilaterally change the architecture.",
    "",
    "## Hardware (remote RDNA4 node)",
    "",
    "- GPU: AMD Radeon AI PRO R9700 (RDNA4 / gfx1201, 32 GB VRAM, 576 GB/s)",
    "- 64 CUs, wave64 optimal, 32 KB L1/CU, 6 MB L2",
    "- VK_KHR_cooperative_matrix 16x16x16 available",
    "- RADV driver (Mesa), `RADV_PERFTEST=coop_matrix` set for both backends",
    "- System glslc: shaderc 2023.8 (Ubuntu 24.04) — newer versions break RADV",
    "- Zig 0.15.2",
    "- Kernel: check with `ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST 'uname -r'`",
    "  - ≥ 6.16: T2 UMQ may be available, but kernel version is not sufficient. Also check `/sys/module/amdgpu/parameters/user_queue` and a real USERQ_CREATE probe.",
    "  - 6.14–6.15: T1 KFD direct only",
    "  - If benchmark output says `execution_tier=t_cpu_after_admission`, direct queue admission is only a smoke path; tokens are still CPU-produced.",
    "",
    "## What you can modify",
    "",
    "- Any source, shader, build, benchmark, test, docs, or loop file needed to make progress.",
    "- You may modify Vulkan, Metal, shared model/tokenizer code, build.zig, tests, docs, and ZINC_RT files if the change is justified by the benchmark goal.",
    "- You may use low-level RDNA/GAS/assembly work when the path is coherent, shortcut-free, and profiling shows a real GPU model kernel is the bottleneck.",
    "",
    "## What you must NOT touch",
    "",
    "- Never run `git push` or otherwise update any remote branch. The harness/user owns publishing.",
    "- Never use destructive git/worktree commands (`git reset`, `git checkout`, `git restore`, `git clean`, `git stash`, `git revert`). The harness owns rollback.",
    "- Never edit secrets (`.env`) or delete/modify `.zinc_rt_autopilot/` run-state.",
    "- Do not delete tests to make a build pass; fix the code or add better tests.",
    "",
    "These restrictions are enforced by the harness where possible; outcome gates below are the real contract.",
    "",
    "**Vulkan is not going away.** You may change Vulkan/shared code, but the Vulkan baseline must still build, pass tests, run coherently, and remain a fair comparison. If your change breaks or cheats the baseline, the loop reverts it.",
    "",
    "## A/B Benchmark Results This Cycle",
    "",
    "```",
    `vulkan: decode=${vkTps} tok/s, prefill=${before.vulkan.prefillTps?.toFixed(1) ?? "n/a"}, coherent=${vkCoh}, exit=${before.vulkan.runExitCode}`,
    `zinc_rt:       decode=${rtTps} tok/s, prefill=${before.zinc_rt.prefillTps?.toFixed(1) ?? "n/a"}, coherent=${rtCoh}, exit=${before.zinc_rt.runExitCode}`,
    `ratio:         ${ratioStr}`,
    `zinc_rt exec:  ${rtExecutionLabel}`,
    `flag recog:    ${before.zinc_rt.backendFlagRecognized}`,
    "```",
    "",
    ...(buildOut ? [
      "## Build output (truncated, last 3500 chars)",
      "```",
      buildOut,
      "```",
      "",
    ] : []),
    ...(runOut ? [
      "## Run output (truncated, last 3000 chars)",
      "```",
      runOut,
      "```",
      "",
    ] : []),
    "## Optimization History (last 12 kept/reverted cycles)",
    historyBlock,
    "",
    "## Controller Diagnosis From Past Cycles",
    gapAnalysis,
    "",
    "## Failed Approaches — DO NOT REPEAT",
    failedBlock,
    "",
    "If a failed approach appears here, do not implement the same idea under a different name. Either address the recorded failure mode directly or pick a different path.",
    "",
    "## Ideas accumulated from past cycles",
    ideasBlock,
    "",
    ...(stallCount >= 3 ? [
      `## ⚠ STALL DETECTED — ${stallCount} consecutive cycles reverted`,
      "Your recent attempts haven't moved the needle. STOP making incremental tweaks.",
      "Step back. Re-read `docs/ZINC_RT_DESIGN.md`. Pick a different milestone direction.",
      "Concretely:",
      ...(nonDirectExecuted ? [
        `  - The benchmark is still ${rtExecutionLabel}. Do not call this M2 work yet.`,
        "  - Stop adding PM4/UMQ/admission/copy markers unless the cycle runs a real compute slice.",
        "  - Stop doing CPU matvec/dequant/threading tweaks unless they are a prerequisite for a direct-execution correctness gate.",
        "  - Prefer the smallest real executed slice: one GPU argmax over scalar-produced logits, one router/top-k value, or one verified DMMV row range that feeds the current prompt result.",
      ] : [
        "  - If you've been hand-tweaking shaders, switch to reducing submission count.",
        "  - If you've been on one direct tier and it is measured-dead, switch tiers only after proving the node supports the alternative UAPI.",
      ]),
      "  - Run `bun loops/zinc_rt_autopilot.ts --dry-run` locally to inspect the current benchmark output without spending an agent cycle.",
      "",
    ] : []),
    ...(hardPivot ? [
      "## HARD PIVOT MODE",
      "",
      "The loop is coherent but stalled while the benchmark remains on the wrong execution tier. This cycle must not be another ordinary tweak.",
      "",
      "Allowed outcomes for this cycle:",
      "- Produce a new benchmark-visible direct-execution signal: the first real decode slice behind a retired T1/T2 fence, with a GPU-produced value consumed by the prompt result.",
      "- Or make no code change and emit `@@@DESCRIPTION: measured-dead: <specific blocker>` with evidence from the remote node or UAPI docs.",
      "",
      "Forbidden in hard pivot mode:",
      "- T-CPU matvec/dequant/threading/fusion optimizations when the CPU fallback is already coherent.",
      "- Logging-only UMQ/PM4 code that does not change the current benchmark output.",
      "- Another preflight/admission probe unless the before/after output changes from failed/missing to passed and no prior kept cycle already exposes that class of signal.",
      "",
    ] : []),
    "## Rules",
    "",
    `1. Make ONE focused change toward beating Vulkan. ${phase === "migrate" ? "Correctness/coherent output > tok/s — fix the broken path first." : "Tok/s > everything else."}`,
    ...(phase === "migrate" ? [
      "   MIGRATE progress must be exercised by the current benchmark or by a new harness-visible validation gate.",
      "   Do not add standalone future packet/building-block code unless this cycle also wires it into executed forward progress or a failing/passing validation gate.",
      "   If the verified graph is not lowered, prefer lowering the smallest real executed slice over adding another verifier.",
    ] : []),
    "2. Before printing the final `@@@` markers, self-verify changed files on the remote. Run repo-root `.env` + rsync + the exact build gates:",
    "   `set -a; source .env; set +a`",
    `   \`rsync -az --delete --exclude '.git' --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' --exclude '.zinc_rt_autopilot' -e "ssh -p $ZINC_PORT" . "$ZINC_USER@$ZINC_HOST:${REMOTE_ZINC_DIR}/"\``,
    `   \`ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" 'cd ${REMOTE_ZINC_DIR} && zig build test --summary all && zig build -Doptimize=ReleaseFast -Dbackend=vulkan && zig build -Doptimize=ReleaseFast -Dbackend=zinc_rt'\``,
    "   If any gate fails, fix it before final output. Do not leave trivial compile errors for the harness rollback.",
    "3. Both build flags MUST stay buildable: `-Dbackend=vulkan` and `-Dbackend=zinc_rt`.",
    "4. Output bit-equality is desirable but not strictly required during MIGRATE. Coherent output comes before absolute performance. In OPTIMIZE phase, zinc_rt MUST produce coherent, shortcut-free text: no top-k=0 decode, no capped LM-head row scan unless explicitly validating row-range quality, and no clamped decode-budget result counted as a win.",
    "5. If you need to reboot the remote node (e.g. you reloaded the amdgpu kernel module, changed sysctl, etc.), emit `@@@REBOOT` on its own line at the end of your output. The loop will reboot and continue.",
    "6. You may SSH into the remote node to inspect the synced ZINC source, compare against the existing Vulkan backend, run probes, inspect /sys, check kernel version, etc. Load repo-root `.env` in the same shell first:",
    "   `set -a; source .env; set +a; ssh -p \"$ZINC_PORT\" \"$ZINC_USER@$ZINC_HOST\" '<cmd>'`",
    "   If SSH says `Operation not permitted`, report it as a sandbox/network permission problem, not as missing RDNA access.",
    "7. You may edit any repo files except the explicit forbidden items above, but you may not push to remote or destructively rewrite the worktree.",
    "8. Zig 0.15.2 API: `ArrayList` is unmanaged (pass allocator), `StringHashMap` → `StringHashMapUnmanaged`, `File.stdout()`, writer takes a buffer arg.",
    "9. RDNA4: wave64 optimal, workgroup_size = 64, shared memory ≤ 64 KB, `RADV_PERFTEST=coop_matrix` enables wmma.",
    "",
    "## Output Format",
    "",
    "After making your change, print these lines at the very end of your output:",
    "@@@DESCRIPTION: <one-line summary of what you changed>",
    "@@@SELF_ANALYSIS: <why this approach; what you expect the ratio to do>",
    "@@@NEXT_IDEAS: <comma-separated ideas for future cycles>",
    "@@@MILESTONE: <M0|M1|M2|M3|M4|M5|M6 — which milestone this change belongs to>",
    "@@@REBOOT (optional — emit ONLY if remote reboot is required)",
  ].join("\n");
}

// ── Cycle types ──────────────────────────────────────────────────────

type CycleResult = {
  cycle: number;
  timestamp: string;
  phase: Phase;
  /** Which agent drove this cycle (codex or claude). */
  agent: AgentKind;
  description: string;
  kept: boolean;
  /** Before agent ran. */
  beforeAB: ABBenchmark;
  /** After agent's changes were applied and verified. */
  afterAB: ABBenchmark;
  selfAnalysis: string;
  nextIdeas: string[];
  milestoneTag: string;
  rebooted: boolean;
  error?: string;
};

type RunState = {
  runId: string;
  cycles: CycleResult[];
  failedApproaches: string[];
  ideas: string[];
  phase: Phase;
  bestRatio: number | null;
  bestZincRtTps: number | null;
  bestVulkanTps: number | null;
};

async function loadState(runDir: string): Promise<RunState | null> {
  const p = join(runDir, "state.json");
  if (!existsSync(p)) return null;
  return JSON.parse(await readFile(p, "utf8")) as RunState;
}

async function saveState(runDir: string, state: RunState): Promise<void> {
  await mkdir(runDir, { recursive: true });
  await writeFile(join(runDir, "state.json"), JSON.stringify(state, null, 2));
}

async function writeCycleFile(cycleDir: string, name: string, contents: string): Promise<void> {
  await mkdir(cycleDir, { recursive: true });
  await writeFile(join(cycleDir, name), contents);
}

function summariseAB(label: string, ab: ABBenchmark): string {
  const vkTps = ab.vulkan.decodeTps?.toFixed(1) ?? "—";
  const rtTps = ab.zinc_rt.decodeTps?.toFixed(1) ?? "—";
  const ratio = ab.ratio?.toFixed(3) ?? "—";
  const vkCoh = ab.vulkan.coherentText ? "✓" : "✗";
  const rtCoh = ab.zinc_rt.coherentText ? "✓" : "✗";
  return `${label}: vk=${vkTps}${vkCoh} rt=${rtTps}${rtCoh} ratio=${ratio}`;
}

type VerificationFailure = {
  kind: string;
  output: string;
};

type VerificationAttempt = {
  after: ABBenchmark;
  verificationError: string | null;
  repairFailure: VerificationFailure | null;
};

function failureOutputTail(output: string): string {
  const text = output.trim();
  return text ? text.slice(-5000) : "(no output captured)";
}

function detectRepairableVerificationFailure(after: ABBenchmark): VerificationFailure | null {
  if (after.vulkan.buildExitCode !== 0) {
    return { kind: "vulkan build failed", output: failureOutputTail(after.vulkan.buildOutput) };
  }
  if (after.zinc_rt.backendFlagRecognized && after.zinc_rt.buildExitCode !== 0) {
    return { kind: "zinc_rt build failed", output: failureOutputTail(after.zinc_rt.buildOutput) };
  }
  if (after.vulkan.runExitCode !== null && after.vulkan.runExitCode !== 0) {
    return { kind: "vulkan benchmark run failed", output: failureOutputTail(after.vulkan.runOutput) };
  }
  if (after.zinc_rt.runExitCode !== null && after.zinc_rt.runExitCode !== 0) {
    return { kind: "zinc_rt benchmark run failed", output: failureOutputTail(after.zinc_rt.runOutput) };
  }
  return null;
}

async function verifyCandidate(
  before: ABBenchmark,
  modelPath: string,
  prompt: string,
  runsPerBinary: number,
): Promise<VerificationAttempt> {
  try {
    await rsyncToRemote();
    const testRes = await remoteTest();
    if (!testRes.passed) {
      console.log(clr("1;31", "  ❌ Tests broken — repair pass required before rollback"));
      const failure = {
        kind: "remote zig build test failed",
        output: failureOutputTail(testRes.output),
      };
      return {
        after: before,
        verificationError: "Tests failed after agent's changes",
        repairFailure: failure,
      };
    }

    const after = await fullAB(modelPath, prompt, runsPerBinary);
    console.log(clr("1;36", `  ${summariseAB("AFTER", after)}`));

    const failure = detectRepairableVerificationFailure(after);
    if (failure) {
      console.log(clr("1;31", `  ❌ ${failure.kind} — repair pass required before rollback`));
      return {
        after,
        verificationError: `${failure.kind} after agent's changes`,
        repairFailure: failure,
      };
    }

    return { after, verificationError: null, repairFailure: null };
  } catch (e) {
    const message = String(e);
    console.log(clr("1;31", `  ❌ Verification failed: ${message}`));
    return {
      after: before,
      verificationError: message,
      repairFailure: null,
    };
  }
}

function buildRepairPrompt(failure: VerificationFailure): string {
  return [
    "# ZINC_RT Verification Repair Task",
    "",
    "The previous agent change failed the harness verification gate. Do not start a new optimization idea.",
    "Make the smallest repair needed so the existing change builds/tests/runs, preserving the original intent where possible.",
    "",
    "## Failure To Fix",
    "",
    failure.kind,
    "",
    "```",
    failure.output,
    "```",
    "",
    "## Required Scope",
    "",
    "- Fix compile/test/runtime breakage only.",
    "- Do not add a new benchmark idea while repairing this failure.",
    "- Do not run `git push` or destructive git commands.",
    "- Do not edit `.env` or `.zinc_rt_autopilot/` run-state.",
    "",
    "## Required Self-Verification Before Final Output",
    "",
    "Run these from the repo root after your repair:",
    "",
    "```bash",
    "set -a; source .env; set +a",
    `rsync -az --delete --exclude '.git' --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' --exclude '.zinc_rt_autopilot' -e "ssh -p $ZINC_PORT" . "$ZINC_USER@$ZINC_HOST:${REMOTE_ZINC_DIR}/"`,
    `ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" 'cd ${REMOTE_ZINC_DIR} && zig build test --summary all && zig build -Doptimize=ReleaseFast -Dbackend=vulkan && zig build -Doptimize=ReleaseFast -Dbackend=zinc_rt'`,
    "```",
    "",
    "If the self-verification fails, keep fixing until it passes or clearly report the blocker.",
    "",
    "End with:",
    "@@@DESCRIPTION: repaired verification failure: <short summary>",
    "@@@SELF_ANALYSIS: <what broke and why the fix is minimal>",
    "@@@NEXT_IDEAS: <comma-separated ideas, or none>",
    "@@@MILESTONE: <M0|M1|M2|M3|M4|M5|M6>",
  ].join("\n");
}

async function runRepairPass(
  agent: AgentKind,
  cycleDir: string,
  failure: VerificationFailure,
): Promise<void> {
  console.log(clr("1;33", "\n  🛠 Verification repair pass"));
  const prompt = buildRepairPrompt(failure);
  await writeCycleFile(cycleDir, "repair_prompt.md", prompt);
  const repair = await runAgent(agent, prompt);
  await writeCycleFile(cycleDir, "repair_stdout.txt", repair.stdout);
  await writeCycleFile(cycleDir, "repair_stderr.txt", repair.stderr);
}

// ── Cycle runner ─────────────────────────────────────────────────────

async function runCycle(
  runDir: string,
  state: RunState,
  agent: AgentKind,
  modelPath: string,
  prompt: string,
  runsPerBinary: number,
): Promise<CycleResult> {
  const cycleNum = state.cycles.length + 1;
  const cycleDir = join(runDir, `cycle-${String(cycleNum).padStart(3, "0")}`);
  await mkdir(cycleDir, { recursive: true });

  console.log(clr("1;35", "\n" + DSEP));
  console.log(clr("1;35", `  CYCLE ${cycleNum}`));
  console.log(clr("1;35", DSEP));

  // Step 1: rsync
  try {
    await rsyncToRemote();
  } catch (e) {
    console.log(clr("1;31", `  ❌ rsync failed: ${e}`));
    const empty: ABBenchmark = {
      vulkan: emptyResult(),
      zinc_rt: emptyResult(),
      ratio: null,
      prefillRatio: null,
    };
    const cycleResult: CycleResult = {
      cycle: cycleNum,
      timestamp: new Date().toISOString(),
      phase: state.phase,
      agent,
      description: "rsync failed",
      kept: false,
      beforeAB: empty,
      afterAB: empty,
      selfAnalysis: "",
      nextIdeas: [],
      milestoneTag: "",
      rebooted: false,
      error: String(e),
    };
    await writeCycleFile(cycleDir, "result.json", JSON.stringify(cycleResult, null, 2));
    return cycleResult;
  }

  // Step 2: BEFORE A/B benchmark
  console.log(clr("1;33", "\n  📊 Baseline A/B benchmark"));
  const before = await fullAB(modelPath, prompt, runsPerBinary);
  state.phase = detectPhase(before);
  console.log(clr("1;36", `  ${summariseAB("BEFORE", before)} (phase=${state.phase})`));
  await writeCycleFile(cycleDir, "before.json", JSON.stringify(before, null, 2));

  // Step 3: Git checkpoint
  await runCommand("git", ["add", "-A", "src/", "build.zig", "build.zig.zon", "benchmarks/", "docs/ZINC_RT_DESIGN.md", "loops/zinc_rt_autopilot.ts", "loops/efforts/"], { cwd: PROJECT_ROOT }).catch(() => { });
  await runCommand("git", ["commit", "--allow-empty", "-m", `zinc_rt-autopilot: pre-cycle-${cycleNum} checkpoint`], { cwd: PROJECT_ROOT }).catch(() => { });
  const preCommit = await runCommand("git", ["rev-parse", "HEAD"], { cwd: PROJECT_ROOT });
  const preHash = preCommit.stdout.trim();

  // Step 4: Agent
  const promptText = buildPrompt(state, before, state.phase);
  await writeCycleFile(cycleDir, "prompt.md", promptText);

  const agentResult = await runAgent(agent, promptText);
  await writeCycleFile(cycleDir, "agent_stdout.txt", agentResult.stdout);
  await writeCycleFile(cycleDir, "agent_stderr.txt", agentResult.stderr);

  const assembledText = extractAgentText(agentResult.stdout);
  const lastChars = assembledText.slice(-3500);

  const descMatch =
    lastChars.match(/^@@@DESCRIPTION:\s*(.+)/im) ??
    lastChars.match(/^DESCRIPTION:\s*(.+)/im);
  const analysisMatch =
    lastChars.match(/^@@@SELF_ANALYSIS:\s*(.+)/im) ??
    lastChars.match(/^SELF_ANALYSIS:\s*(.+)/im);
  const ideasMatch =
    lastChars.match(/^@@@NEXT_IDEAS:\s*(.+)/im) ??
    lastChars.match(/^NEXT_IDEAS:\s*(.+)/im);
  const milestoneMatch =
    lastChars.match(/^@@@MILESTONE:\s*(.+)/im) ??
    lastChars.match(/^MILESTONE:\s*(.+)/im);
  const rebootRequested = /^@@@REBOOT\b/im.test(lastChars);

  const rawDesc = descMatch?.[1]?.trim() ?? "";
  let description = rawDesc && !isGarbageString(rawDesc) ? rawDesc : "";
  if (!description) {
    try {
      const diff = await runCommand("git", ["diff", "--stat", preHash, "HEAD"], { cwd: PROJECT_ROOT });
      const files = diff.stdout.split("\n")
        .filter((l) => l.includes("|"))
        .map((l) => l.trim().split("|")[0].trim().split("/").pop())
        .filter(Boolean)
        .slice(0, 3);
      description = files.length > 0
        ? `Modified ${files.join(", ")}`
        : "Agent made changes";
    } catch {
      description = "Agent made changes";
    }
  }
  const selfAnalysis = analysisMatch?.[1]?.trim() ?? "";
  const newIdeas = ideasMatch?.[1]
    ?.split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 3 && s.length < 120 && !isGarbageString(s)) ?? [];
  const milestoneTag = milestoneMatch?.[1]?.trim().slice(0, 16) ?? "";

  // Step 5: Reboot if requested
  let rebooted = false;
  if (rebootRequested) {
    rebooted = await rebootRemote();
    if (!rebooted) {
      console.log(clr("1;31", "  ❌ Reboot failed — verifying without reboot completion (may fail)"));
    }
  }

  // Step 6: Verify — rsync + test + A/B
  console.log(clr("1;33", "\n  📊 Verification A/B benchmark"));
  let verification = await verifyCandidate(before, modelPath, prompt, runsPerBinary);
  let after: ABBenchmark = verification.after;
  let verificationError: string | null = verification.verificationError;

  for (
    let repairAttempt = 0;
    verification.repairFailure && repairAttempt < VERIFICATION_REPAIR_ATTEMPTS;
    repairAttempt += 1
  ) {
    console.log(clr("1;33", `  ⚠ ${verification.repairFailure.kind}; asking agent to repair before reverting (${repairAttempt + 1}/${VERIFICATION_REPAIR_ATTEMPTS})`));
    await runRepairPass(agent, cycleDir, verification.repairFailure);
    console.log(clr("1;33", "\n  📊 Verification A/B benchmark after repair"));
    verification = await verifyCandidate(before, modelPath, prompt, runsPerBinary);
    after = verification.after;
    verificationError = verification.verificationError;
  }

  await writeCycleFile(cycleDir, "after.json", JSON.stringify(after, null, 2));

  // Step 7: Decide
  let keep = false;
  let keepReason = "";

  if (verificationError) {
    keepReason = `verification error: ${verificationError}`;
    keep = false;
  } else if (state.phase === "migrate") {
    const beforeCpuExecuted = isZincRtCpuExecuted(before.zinc_rt);
    const migrateBestRt = bestZincRtTpsOfState(state);
    // MIGRATE: keep if we made meaningful progress
    if (!before.zinc_rt.backendFlagRecognized && after.zinc_rt.backendFlagRecognized) {
      keep = true;
      keepReason = "build flag now recognized";
    } else if (before.zinc_rt.buildExitCode !== 0 && after.zinc_rt.buildExitCode === 0) {
      keep = true;
      keepReason = "zinc_rt build now passes";
    } else if (
      after.zinc_rt.buildExitCode === 0 &&
      (before.zinc_rt.runExitCode !== 0 && before.zinc_rt.runExitCode !== null) &&
      after.zinc_rt.runExitCode === 0
    ) {
      keep = true;
      keepReason = "zinc_rt runtime now succeeds";
    } else if (!before.zinc_rt.coherentText && after.zinc_rt.coherentText) {
      keep = true;
      keepReason = "zinc_rt output now coherent";
    } else if (
      before.ratio == null && after.ratio != null
    ) {
      keep = true;
      keepReason = "zinc_rt now produces tok/s metric";
    } else if (
      hasNewM1ValidationSignal(before, after) &&
      foundationPerformanceEnvelopeOk(before, after)
    ) {
      keep = true;
      keepReason = "new benchmark-visible M1 validation signal";
    } else if (
      hasNewDirectModelSliceSignal(before, after) &&
      after.zinc_rt.runExitCode === 0 &&
      after.zinc_rt.coherentText
    ) {
      keep = true;
      keepReason = "new consumed direct model-slice signal";
    } else if (
      hasNewDirectExecutionSignal(before, after) &&
      after.zinc_rt.coherentText
    ) {
      keep = true;
      keepReason = "zinc_rt now executes through a direct runtime tier";
    } else if (
      !beforeCpuExecuted &&
      before.ratio != null && after.ratio != null &&
      after.ratio >= before.ratio + RATIO_IMPROVEMENT_KEEP &&
      after.zinc_rt.coherentText
    ) {
      keep = true;
      keepReason = `ratio improved ${before.ratio.toFixed(3)} → ${after.ratio.toFixed(3)}`;
    } else if (
      !beforeCpuExecuted &&
      after.zinc_rt.coherentText &&
      after.zinc_rt.decodeTps != null && before.zinc_rt.decodeTps != null &&
      after.zinc_rt.decodeTps >= before.zinc_rt.decodeTps + ABS_TPS_IMPROVEMENT_KEEP
    ) {
      keep = true;
      keepReason = `zinc_rt tok/s improved ${before.zinc_rt.decodeTps.toFixed(1)} → ${after.zinc_rt.decodeTps.toFixed(1)}`;
    } else if (
      after.zinc_rt.coherentText &&
      after.zinc_rt.decodeTps != null && before.zinc_rt.decodeTps != null &&
      after.zinc_rt.decodeTps >= Math.max(
        before.zinc_rt.decodeTps,
        beforeCpuExecuted ? (migrateBestRt ?? before.zinc_rt.decodeTps) : before.zinc_rt.decodeTps,
      ) + Math.max(
        beforeCpuExecuted ? 1.0 : MIGRATE_MIN_ABS_TPS_IMPROVEMENT_KEEP,
        Math.max(
          before.zinc_rt.decodeTps,
          beforeCpuExecuted ? (migrateBestRt ?? before.zinc_rt.decodeTps) : before.zinc_rt.decodeTps,
        ) * MIGRATE_REL_TPS_IMPROVEMENT_KEEP,
      )
    ) {
      keep = true;
      const baseline = Math.max(
        before.zinc_rt.decodeTps,
        beforeCpuExecuted ? (migrateBestRt ?? before.zinc_rt.decodeTps) : before.zinc_rt.decodeTps,
      );
      keepReason = `migrate tok/s improved past ${baseline.toFixed(2)} → ${after.zinc_rt.decodeTps.toFixed(2)}`;
    }

    // Quality gate
    if (keep && before.zinc_rt.coherentText && !after.zinc_rt.coherentText) {
      keep = false;
      keepReason = "regressed coherent output";
    }
    // Don't break vulkan
    if (keep && before.vulkan.coherentText && !after.vulkan.coherentText) {
      keep = false;
      keepReason = "broke vulkan output coherence (vulkan must keep working)";
    }
  } else {
    // OPTIMIZE: keep if ratio improved by at least RATIO_IMPROVEMENT_KEEP
    const minAbsImprovement = Math.max(
      ABS_TPS_IMPROVEMENT_KEEP,
      (before.zinc_rt.decodeTps ?? 0) * 0.01,
    );
    if (
      before.ratio != null && after.ratio != null &&
      after.ratio >= before.ratio + RATIO_IMPROVEMENT_KEEP &&
      after.zinc_rt.coherentText &&
      after.zinc_rt.decodeTps != null && before.zinc_rt.decodeTps != null &&
      after.zinc_rt.decodeTps >= before.zinc_rt.decodeTps + minAbsImprovement
    ) {
      keep = true;
      keepReason = `ratio +${((after.ratio - before.ratio) * 100).toFixed(1)}pp, tok/s +${(after.zinc_rt.decodeTps - before.zinc_rt.decodeTps).toFixed(1)}`;
    }
    // Quality gates
    if (keep && before.zinc_rt.coherentText && !after.zinc_rt.coherentText) {
      keep = false;
      keepReason = "broke output coherence";
    }
    if (keep && before.vulkan.coherentText && !after.vulkan.coherentText) {
      keep = false;
      keepReason = "broke vulkan path";
    }
  }

  console.log(
    clr(keep ? "1;32" : "1;31",
      `  → ${keep ? "✅ KEEPING" : "❌ REVERTING"}  (${keepReason || "no improvement"})`,
    ),
  );

  if (!keep) {
    console.log(clr("2", `  reset to ${preHash.slice(0, 8)}`));
    await runCommand("git", ["reset", "--hard", preHash], { cwd: PROJECT_ROOT }).catch(() => { });
  } else {
    await runCommand("git", ["add", "-A", "src/", "build.zig", "build.zig.zon", "benchmarks/", "docs/ZINC_RT_DESIGN.md", "loops/zinc_rt_autopilot.ts", "loops/efforts/"], { cwd: PROJECT_ROOT }).catch(() => { });
    await runCommand(
      "git", ["commit", "--allow-empty", "-m", `zinc_rt-autopilot: ${description}`],
      { cwd: PROJECT_ROOT },
    ).catch(() => { });

    // Update bests
    if (after.ratio != null && (state.bestRatio == null || after.ratio > state.bestRatio)) {
      state.bestRatio = after.ratio;
    }
    if (after.zinc_rt.decodeTps != null && (state.bestZincRtTps == null || after.zinc_rt.decodeTps > state.bestZincRtTps)) {
      state.bestZincRtTps = after.zinc_rt.decodeTps;
    }
    if (after.vulkan.decodeTps != null && (state.bestVulkanTps == null || after.vulkan.decodeTps > state.bestVulkanTps)) {
      state.bestVulkanTps = after.vulkan.decodeTps;
    }
  }

  const cycleResult: CycleResult = {
    cycle: cycleNum,
    timestamp: new Date().toISOString(),
    phase: state.phase,
    agent,
    description,
    kept: keep,
    beforeAB: before,
    afterAB: after,
    selfAnalysis,
    nextIdeas: newIdeas,
    milestoneTag,
    rebooted,
    error: verificationError ?? undefined,
  };
  await writeCycleFile(cycleDir, "result.json", JSON.stringify(cycleResult, null, 2));
  return cycleResult;
}

function emptyResult(): BenchmarkResult {
  return {
    decodeTps: null,
    prefillTps: null,
    decodeSamples: [],
    prefillSamples: [],
    buildExitCode: -1,
    buildOutput: "",
    runOutput: "",
    runExitCode: null,
    coherentText: false,
    garbageOutput: false,
    tokensGenerated: 0,
    bandwidthUtil: null,
    effectiveBW: null,
    error: "not run",
    backendFlagRecognized: false,
  };
}

// ── Main ─────────────────────────────────────────────────────────────

type CliOptions = {
  agent: AgentKind;
  /** When true, alternate between codex and claude per cycle. */
  alternate: boolean;
  cycles: number;
  modelPath: string;
  prompt: string;
  runsPerBinary: number;
  dryRun: boolean;
  resumeDir?: string;
  targetRatio: number | null;
  targetTps: number | null;
};

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    agent: "codex",
    alternate: false,
    cycles: Infinity,
    modelPath: DEFAULT_MODEL,
    prompt: DEFAULT_PROMPT,
    runsPerBinary: 3,
    dryRun: false,
    targetRatio: null,
    targetTps: null,
  };
  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case "--agent":
        opts.agent = argv[++i] as AgentKind;
        break;
      case "--alternate":
        opts.alternate = true;
        break;
      case "--cycles":
        opts.cycles = parseInt(argv[++i], 10);
        break;
      case "--model-path":
        opts.modelPath = argv[++i];
        break;
      case "--prompt":
        opts.prompt = argv[++i];
        break;
      case "--runs-per-binary":
        opts.runsPerBinary = parseInt(argv[++i], 10);
        break;
      case "--dry-run":
        opts.dryRun = true;
        break;
      case "--resume":
        if (argv[i + 1] && !argv[i + 1].startsWith("--")) {
          opts.resumeDir = argv[++i];
        } else {
          opts.resumeDir = "latest";
        }
        break;
      case "--target-ratio":
        opts.targetRatio = parseFloat(argv[++i]);
        break;
      case "--target-tps":
        opts.targetTps = parseFloat(argv[++i]);
        break;
      case "--help":
      case "-h":
        console.log([
          "Usage: bun loops/zinc_rt_autopilot.ts [options]",
          "",
          "Agent options (both codex and claude are fully supported):",
          "  --agent <claude|codex>     Agent to use (default: codex)",
          "  --alternate                Round-robin between codex and claude per",
          "                             cycle. Useful overnight to get diverse",
          "                             ideas from both models.",
          "",
          "Cycle / run options:",
          "  --cycles N                 Max cycles (default: infinite — runs until",
          "                             killed or --target-* reached)",
          "  --model-path <path>        GGUF model on remote node",
          "                             (default: " + DEFAULT_MODEL + ")",
          "  --prompt <s>               Benchmark prompt (default: \"" + DEFAULT_PROMPT + "\")",
          "  --runs-per-binary N        Runs per backend per benchmark (default: 3,",
          "                             median used)",
          "  --dry-run                  A/B benchmark only, no agent",
          "  --resume [dir]             Resume from .zinc_rt_autopilot/<runId>/",
          "                             Bare --resume resumes the latest run with state.json",
          "",
          "Stop conditions:",
          "  --target-ratio R           Exit when zinc_rt/vulkan ratio ≥ R",
          "                             (e.g. 1.05 = beat vulkan by 5%)",
          "  --target-tps T             Exit when zinc_rt tok/s ≥ T",
          "",
          "Environment overrides (read from process env or .env):",
          "  ZINC_HOST, ZINC_PORT, ZINC_USER     — remote node SSH",
          "  ZINC_RT_TIER                        — force T1/T2/T-CPU (default: auto)",
          "  ZINC_CLAUDE_MODEL                   — claude model (default: " + CLAUDE_MODEL + ")",
          "  ZINC_CLAUDE_EFFORT                  — claude effort (default: " + CLAUDE_EFFORT + ")",
          "  ZINC_CODEX_MODEL                    — codex model (default: " + CODEX_MODEL + ")",
          "  ZINC_CODEX_REASONING_EFFORT         — codex effort (default: " + CODEX_REASONING_EFFORT + ")",
          "",
          "Each cycle:",
          "  1. rsync, build BOTH `-Dbackend=vulkan` AND `-Dbackend=zinc_rt`",
          "  2. benchmark each N times under flock, take median tok/s",
          "  3. spawn agent (codex or claude) with the design doc + last cycle's results",
          "  4. agent edits files; re-build, re-benchmark, and gets one repair pass",
          "     before rollback if tests/builds/runs fail",
          "  5. keep change if ratio improved, coherence was unlocked, or an M1",
          "     validation gate became benchmark-visible without a material slowdown",
          "  6. continue until --cycles or --target-* reached",
          "",
          "Vulkan is a permanent supported backend, not a deletion target. Every",
          "milestone keeps `-Dbackend=vulkan` building and passing tests. The loop",
          "auto-reverts any change that breaks the Vulkan build.",
        ].join("\n"));
        process.exit(0);
    }
  }
  return opts;
}

async function resolveResumeDir(requested: string): Promise<string> {
  if (requested !== "latest") {
    return requested.startsWith("/") || requested.startsWith(".")
      ? resolve(requested)
      : join(RESULTS_DIR, requested);
  }

  const entries = await readdir(RESULTS_DIR, { withFileTypes: true }).catch(() => []);
  const runDirs = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter((name) => existsSync(join(RESULTS_DIR, name, "state.json")))
    .sort();
  if (runDirs.length === 0) {
    throw new Error("--resume requested but no .zinc_rt_autopilot run with state.json exists");
  }
  return join(RESULTS_DIR, runDirs[runDirs.length - 1]);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  RESULTS_DIR = resolve(REPO_ROOT, ".zinc_rt_autopilot");

  const runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const runDir = opts.resumeDir ? await resolveResumeDir(opts.resumeDir) : join(RESULTS_DIR, runId);
  await mkdir(runDir, { recursive: true });

  console.log(clr("1;35", DSEP));
  console.log(clr("1;35", "  ZINC_RT AUTOPILOT — overnight A/B vs vulkan"));
  console.log(clr("1;35", DSEP));
  console.log(`  Remote:    ${clr("1", `${ZINC_USER}@${ZINC_HOST}:${ZINC_PORT}`)}`);
  console.log(`  RemoteDir: ${clr("1", REMOTE_ZINC_DIR)}`);
  console.log(`  Model:     ${clr("1", opts.modelPath)}`);
  console.log(`  Prompt:    ${clr("1", JSON.stringify(opts.prompt))}`);
  const agentDetail = opts.alternate
    ? `alternating (codex ${CODEX_MODEL}/${CODEX_REASONING_EFFORT} ↔ claude ${CLAUDE_MODEL}/${CLAUDE_EFFORT})`
    : opts.agent === "claude"
      ? `claude (${CLAUDE_MODEL}, effort=${CLAUDE_EFFORT})`
      : `codex (${CODEX_MODEL}, effort=${CODEX_REASONING_EFFORT})`;
  console.log(`  Agent:     ${clr("1", agentDetail)}`);
  console.log(`  Cycles:    ${opts.cycles === Infinity ? "infinite" : String(opts.cycles)}`);
  console.log(`  Runs/bin:  ${opts.runsPerBinary}`);
  if (opts.targetRatio != null) console.log(`  TargetR:   ${opts.targetRatio.toFixed(3)}`);
  if (opts.targetTps != null) console.log(`  TargetT:   ${opts.targetTps.toFixed(1)} tok/s`);
  console.log(`  RunDir:    ${runDir}`);
  console.log(clr("1;35", DSEP));

  // Pre-checks
  console.log(clr("2", "\n  SSH check..."));
  try {
    const osInfo = await ssh("uname -a", 15_000);
    console.log(clr("1;32", `  ✓ ${osInfo.slice(0, 96)}`));
  } catch (e) {
    console.error(clr("31", `\n  ❌ Cannot reach remote node: ${e}`));
    process.exit(1);
  }

  try {
    const zigVer = await ssh("zig version", 10_000);
    console.log(clr("1;32", `  ✓ zig ${zigVer}`));
  } catch {
    console.error(clr("31", "  ❌ zig not found on remote"));
    process.exit(1);
  }

  try {
    const kver = await ssh("uname -r", 10_000);
    console.log(clr("1;32", `  ✓ kernel ${kver}`));
    const major = parseInt(kver.split(".")[0] ?? "0", 10);
    const minor = parseInt(kver.split(".")[1] ?? "0", 10);
    if (major < 6 || (major === 6 && minor < 16)) {
      console.log(clr("1;33", `  ⚠ kernel < 6.16 → T2 UMQ not available; T1 PM4 only`));
    } else {
      console.log(clr("1;32", `  ✓ kernel ≥ 6.16 → T2 UMQ available`));
    }
  } catch {
    console.log(clr("1;33", "  ⚠ could not detect kernel version"));
  }

  // Ensure remote dir exists
  await sshSafe(`mkdir -p ${REMOTE_ZINC_DIR}`, 10_000);

  // Dry run
  if (opts.dryRun) {
    console.log(clr("1;33", "\n  DRY RUN: rsync + build both + benchmark"));
    await rsyncToRemote();
    const ab = await fullAB(opts.modelPath, opts.prompt, opts.runsPerBinary);
    console.log(clr("1;33", "\n  Result:"));
    console.log("  " + summariseAB("AB", ab));
    if (ab.ratio != null) {
      const verdict = ab.ratio >= 1.0 ? "WINNING ✓" : "behind";
      console.log(clr(ab.ratio >= 1.0 ? "1;32" : "1;33", `  → ratio ${ab.ratio.toFixed(3)} (${verdict})`));
    }
    return;
  }

  // Load or create state
  let state = await loadState(runDir);
  if (!state) {
    state = {
      runId,
      cycles: [],
      failedApproaches: [],
      ideas: [],
      phase: "migrate",
      bestRatio: null,
      bestZincRtTps: null,
      bestVulkanTps: null,
    };
    await saveState(runDir, state);
  } else {
    console.log(clr("1;33", `\n  Resuming from cycle ${state.cycles.length}`));
  }

  // Main loop
  let cyclesDone = 0;
  let consecutiveSSHFailures = 0;

  while (cyclesDone < opts.cycles) {
    // SSH health check
    try {
      await ssh("echo ok", 15_000);
      consecutiveSSHFailures = 0;
    } catch {
      consecutiveSSHFailures++;
      if (consecutiveSSHFailures >= 3) {
        console.error(
          clr("31", `\n  ❌ SSH unreachable ${consecutiveSSHFailures}x. Waiting 5 min...`),
        );
        await new Promise((r) => setTimeout(r, 300_000));
        continue;
      }
      console.log(clr("33", `  SSH failed (${consecutiveSSHFailures}/3), retry in 60s...`));
      await new Promise((r) => setTimeout(r, 60_000));
      continue;
    }

    // --alternate: pick codex on even cycles, claude on odd cycles (or vice
    // versa depending on what the user set as --agent). Default is plain
    // opts.agent for every cycle.
    const cycleAgent: AgentKind = opts.alternate
      ? (state.cycles.length % 2 === 0
        ? (opts.agent === "codex" ? "codex" : "claude")
        : (opts.agent === "codex" ? "claude" : "codex"))
      : opts.agent;

    if (opts.alternate) {
      console.log(clr("1;36", `\n  Agent this cycle: ${cycleAgent}`));
    }

    const cycleResult = await runCycle(runDir, state, cycleAgent, opts.modelPath, opts.prompt, opts.runsPerBinary);
    state.cycles.push(cycleResult);

    // Track failed approaches
    if (!cycleResult.kept && cycleResult.description !== "Agent made changes" && !cycleResult.description.includes("rsync")) {
      const desc = formatFailedApproach(cycleResult).slice(0, 420);
      if (!isGarbageString(desc)) {
        state.failedApproaches.push(desc);
      }
      if (state.failedApproaches.length > 25) {
        state.failedApproaches = state.failedApproaches.slice(-25);
      }
    }

    // Merge new ideas with dedup
    for (const idea of cycleResult.nextIdeas) {
      if (isGarbageString(idea)) continue;
      const words = new Set(idea.toLowerCase().split(/\s+/).filter((w) => w.length > 3));
      const isDupe = state.ideas.some((existing) => {
        const ew = new Set(existing.toLowerCase().split(/\s+/).filter((w) => w.length > 3));
        if (ew.size === 0 || words.size === 0) return false;
        let overlap = 0;
        for (const w of words) if (ew.has(w)) overlap++;
        return overlap / Math.min(words.size, ew.size) > 0.6;
      });
      if (!isDupe) state.ideas.push(idea);
    }
    if (state.ideas.length > 25) state.ideas = state.ideas.slice(-25);

    // Cap cycle history (keep all kept + last 50)
    if (state.cycles.length > 80) {
      const kept = state.cycles.filter((c) => c.kept);
      const recent = state.cycles.slice(-50);
      const seen = new Set<number>();
      const merged: CycleResult[] = [];
      for (const c of [...kept, ...recent]) {
        if (!seen.has(c.cycle)) {
          seen.add(c.cycle);
          merged.push(c);
        }
      }
      state.cycles = merged.sort((a, b) => a.cycle - b.cycle);
    }

    await saveState(runDir, state);

    // Summary
    const keptCount = state.cycles.filter((c) => c.kept).length;
    console.log(clr("1;35", "\n" + DSEP));
    console.log(clr("1;35", `  AFTER ${state.cycles.length} CYCLES`));
    console.log(clr("1;35", `  Phase: ${state.phase.toUpperCase()}`));
    if (state.bestRatio != null) {
      const winning = state.bestRatio >= 1.0;
      console.log(clr(winning ? "1;32" : "1;33",
        `  Best ratio: ${state.bestRatio.toFixed(3)} (${(state.bestRatio * 100).toFixed(1)}%) ${winning ? "WINNING ✓" : ""}`));
    }
    if (state.bestZincRtTps != null) {
      console.log(clr("1;35", `  Best zinc_rt:   ${state.bestZincRtTps.toFixed(1)} tok/s`));
    }
    if (state.bestVulkanTps != null) {
      console.log(clr("1;35", `  Best vulkan:    ${state.bestVulkanTps.toFixed(1)} tok/s`));
    }
    console.log(clr("1;35", `  Kept: ${keptCount}/${state.cycles.length}`));
    console.log(clr("1;35", DSEP));

    // Target-reached early exit
    if (opts.targetRatio != null && state.bestRatio != null && state.bestRatio >= opts.targetRatio) {
      console.log(clr("1;32", `\n  🏁 Target ratio ${opts.targetRatio.toFixed(3)} reached. Stopping.`));
      break;
    }
    if (opts.targetTps != null && state.bestZincRtTps != null && state.bestZincRtTps >= opts.targetTps) {
      console.log(clr("1;32", `\n  🏁 Target ${opts.targetTps.toFixed(1)} tok/s reached. Stopping.`));
      break;
    }

    cyclesDone++;
    await new Promise((r) => setTimeout(r, 3_000));
  }

  console.log(clr("1;32", "\n" + DSEP));
  console.log(clr("1;32", "  ZINC_RT AUTOPILOT — RUN COMPLETE"));
  console.log(clr("1;32", `  Cycles: ${state.cycles.length} | Kept: ${state.cycles.filter((c) => c.kept).length}`));
  if (state.bestRatio != null) {
    console.log(clr("1;32", `  Best ratio: ${state.bestRatio.toFixed(3)} (${(state.bestRatio * 100).toFixed(1)}%)`));
  }
  if (state.bestZincRtTps != null && state.bestVulkanTps != null) {
    console.log(clr("1;32", `  Best zinc_rt: ${state.bestZincRtTps.toFixed(1)}    Best vulkan: ${state.bestVulkanTps.toFixed(1)}`));
  }
  console.log(clr("1;32", `  Results: ${runDir}`));
  console.log(clr("1;32", DSEP));
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(clr("31", `\nFatal error: ${err.message ?? err}`));
    process.exit(1);
  });
}
