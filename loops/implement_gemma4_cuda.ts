#!/usr/bin/env bun
/**
 * ZINC gemma4 CUDA Implementation Loop
 *
 * Autonomous loop that iteratively implements the **gemma4** forward pass for the
 * CUDA backend (`src/compute/forward_cuda.zig` is currently qwen35/qwen36-only),
 * so the last 2 catalog models (gemma4-31b dense, gemma4-26b-a4b MoE) generate
 * coherent text on the 4090 — completing the 5/5 catalog. See the plan in
 * `loops/efforts/MULTI_HOUR_EFFORT_22_CUDA_GEMMA4.md`.
 *
 * The CUDA toolchain + GPUs live on the remote "agent-zinc" box (the Mac can't
 * build CUDA), so each cycle:
 *   1. rsync local src/ → box
 *   2. build the CUDA zinc exe ON THE BOX (zig build -Dbackend=cuda)
 *   3. run the target gemma4 model ON THE BOX (zinc -m <gguf> --prompt …)
 *   4. analyze: build error? unsupported-arch? incoherent? coherent (≈ reference)?
 *   5. spawn ONE Claude agent to make a single focused implementation step
 *      (it edits files locally; the loop owns the canonical build/run/validate)
 *   6. loop back to 1 until the output is coherent, then advance to the MoE model
 *
 * Phases: FIX (build/crash) → IMPLEMENT (produce coherent tokens) → MOE (wire the
 * gemma4 MoE for the 26b). "Coherent" = the greedy output contains REFERENCE_TEXT
 * (e.g. "Paris") AND has no NaN/garbage; the agent must verify token-for-token vs
 * llama.cpp via the per-layer-diff method before declaring done.
 *
 * Usage:
 *   bun loops/implement_gemma4_cuda.ts                 # run until coherent (or MAX_CYCLES)
 *   bun loops/implement_gemma4_cuda.ts --cycles 30
 *   bun loops/implement_gemma4_cuda.ts --dry-run       # sync+build+run only, no agent
 *
 * Env (defaults match the agent-zinc box; see memory `agent-zinc-access`):
 *   ZINC_SSH        ssh target            (agent-zinc@100.67.129.14)
 *   ZINC_SSH_PORT   ssh port              (2222)
 *   ZINC_SSH_KEY    identity file         (~/.ssh/id_ed25519_agent-zinc)
 *   ZINC_GPU        4090 UUID             (GPU-e59a6fce-1961-bafe-927c-06c0149f2370)
 *   ZINC_BOX_REPO   repo path on box      (workspace/zinc)
 *   ZINC_GEMMA      gemma4 gguf on box    (workspace/models/gemma-4-31B-it-Q4_K_M.gguf)
 *   ZINC_GEMMA_MOE  gemma4 MoE gguf       (workspace/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf)
 *   ZINC_REFERENCE  expected substring    (Paris)
 *   ZINC_PROMPT     test prompt           (The capital of France is)
 */

import { spawn } from "node:child_process";
import { resolve } from "node:path";

// ── config ───────────────────────────────────────────────────────────
const REPO_ROOT = resolve(import.meta.dir, "..");
const SSH_TARGET = process.env.ZINC_SSH ?? "agent-zinc@100.67.129.14";
const SSH_PORT = process.env.ZINC_SSH_PORT ?? "2222";
const SSH_KEY = process.env.ZINC_SSH_KEY ?? `${process.env.HOME}/.ssh/id_ed25519_agent-zinc`;
const GPU = process.env.ZINC_GPU ?? "GPU-e59a6fce-1961-bafe-927c-06c0149f2370";
const BOX_REPO = process.env.ZINC_BOX_REPO ?? "workspace/zinc";
const GEMMA = process.env.ZINC_GEMMA ?? "workspace/models/gemma-4-31B-it-Q4_K_M.gguf";
const GEMMA_MOE = process.env.ZINC_GEMMA_MOE ?? "workspace/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf";
const REFERENCE = process.env.ZINC_REFERENCE ?? "Paris";
const PROMPT = process.env.ZINC_PROMPT ?? "The capital of France is";
const EFFORT_FILE = "loops/efforts/MULTI_HOUR_EFFORT_22_CUDA_GEMMA4.md";
const CLAUDE_EFFORT = process.env.ZINC_CLAUDE_EFFORT ?? "high";
const SEP = "─".repeat(72);

const SSH = ["-i", SSH_KEY, "-p", SSH_PORT, "-o", "IdentitiesOnly=yes",
  "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"];

function log(s: string) { process.stdout.write(s + "\n"); }

// Run a local command, capture combined stdout+stderr.
function run(cmd: string, args: string[], opts: { timeout?: number; stream?: boolean } = {}): Promise<{ code: number; out: string }> {
  return new Promise((res) => {
    const child = spawn(cmd, args, { cwd: REPO_ROOT });
    let out = "";
    const onData = (d: Buffer) => { const s = d.toString(); out += s; if (opts.stream) process.stdout.write(s); };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    const to = opts.timeout ? setTimeout(() => child.kill("SIGKILL"), opts.timeout) : null;
    child.on("close", (code) => { if (to) clearTimeout(to); res({ code: code ?? -1, out }); });
    child.on("error", (e) => { if (to) clearTimeout(to); res({ code: -1, out: out + String(e) }); });
  });
}

const ssh = (remoteCmd: string, timeout = 600_000) =>
  run("ssh", [...SSH, SSH_TARGET, "bash", "-lc", remoteCmd], { timeout, stream: false });

// ── cycle steps ──────────────────────────────────────────────────────
async function syncToBox(): Promise<void> {
  await run("rsync", ["-az", "--delete", "-e", `ssh ${SSH.join(" ")}`,
    "src/", `${SSH_TARGET}:${BOX_REPO}/src/`]);
  // keep build.zig in sync too (new targets/kernels may need it)
  await run("rsync", ["-az", "-e", `ssh ${SSH.join(" ")}`, "build.zig", `${SSH_TARGET}:${BOX_REPO}/build.zig`]);
}

async function buildOnBox(): Promise<{ ok: boolean; errors: string }> {
  const r = await ssh(`cd ~/${BOX_REPO} && CUDA_VISIBLE_DEVICES=${GPU} timeout 320 ~/zig-0.15.2/zig build -Dbackend=cuda -Dshaders=false 2>&1`, 360_000);
  const errors = r.out.split("\n").filter((l) => /error:|panic:/.test(l)).join("\n");
  return { ok: r.code === 0 && errors === "", errors: errors || (r.code !== 0 ? r.out.slice(-2000) : "") };
}

async function runGemma(modelPath: string): Promise<string> {
  const r = await ssh(`cd ~/${BOX_REPO} && CUDA_VISIBLE_DEVICES=${GPU} timeout 240 zig-out/bin/zinc -m ~/${modelPath} --prompt ${JSON.stringify(PROMPT)} -n 24 2>&1 | tail -20`, 280_000);
  return r.out;
}

type Status = "BUILD_FAIL" | "CRASH" | "INCOHERENT" | "COHERENT";
function classify(build: { ok: boolean }, runOut: string): Status {
  if (!build.ok) return "BUILD_FAIL";
  if (/panic|error:|out of memory|UnsupportedArchitecture|non-finite/i.test(runOut)) return "CRASH";
  // pull the generated text after "Output"
  const m = runOut.match(/Output[^:]*:\s*([\s\S]*)$/i);
  const gen = (m ? m[1] : runOut).trim();
  if (gen.includes(REFERENCE)) return "COHERENT";
  return "INCOHERENT";
}

function buildPrompt(status: Status, build: { errors: string }, runOut: string, cycle: number): string {
  return [
    `You are implementing the gemma4 forward pass for ZINC's CUDA backend (cycle ${cycle}).`,
    `GOAL: make ${GEMMA} (gemma4 dense) generate coherent text on the 4090 via ZINC CUDA`,
    `(greedy output should contain "${REFERENCE}" for the prompt "${PROMPT}"), then the gemma4 MoE.`,
    ``,
    `THE PLAN + ARCH FINDINGS ARE IN: ${EFFORT_FILE} — read it first. Key: gemma4 has scaled`,
    `embeddings, per-layer/altup embeddings, sliding-window attention (period ~6), per-head q/k`,
    `norm, 4–6 norms/layer, gemma RMSNorm (1+weight), GeGLU, final-logit softcap/bias; the 26b`,
    `adds MoE. forward_cuda.zig is qwen35/qwen36-only and rejects gemma with UnsupportedArchitecture.`,
    ``,
    `CURRENT STATE (status=${status}):`,
    status === "BUILD_FAIL" ? `BUILD ERRORS:\n${build.errors}` : `LAST RUN OUTPUT:\n${runOut}`,
    ``,
    `MAKE EXACTLY ONE focused implementation step toward coherent gemma4, then STOP:`,
    `- FIX: if BUILD_FAIL/CRASH, fix the build/crash.`,
    `- IMPLEMENT: extend loader_cuda.zig (+ ModelConfig) for the gemma4 config; add needed kernels`,
    `  (gelu/geglu, gemma-norm, SWA-masked attention) to src/shaders/cuda/kernels.cu; build a`,
    `  forwardGemma path in forward_cuda.zig (branch on architecture == .gemma).`,
    `- VALIDATE correctness with the PER-LAYER-DIFF method (src/dbg_cuda.zig residual dump vs a`,
    `  llama.cpp eval-callback l_out-N reference) — argmax can match while a mid-layer is wrong`,
    `  (that was the qwen35 gate bug). Reference impl: ~/workspace/llama.cpp/src/models/gemma4.cpp.`,
    ``,
    `CONSTRAINTS:`,
    `- Edit files LOCALLY in this repo (the loop syncs src/ to the box and owns the canonical`,
    `  build + run + GPU validation after you — do not rely on building CUDA on this Mac).`,
    `- For your own inspection you MAY ssh to the box: ssh ${SSH.join(" ")} ${SSH_TARGET}`,
    `  (repo ~/${BOX_REPO}, gemma at ~/${GEMMA}, build: \`CUDA_VISIBLE_DEVICES=${GPU} ~/zig-0.15.2/zig build -Dbackend=cuda -Dshaders=false\`).`,
    `- ADDITIVE only; never break the qwen35/qwen36 path (run scripts/validate_cuda_decode.sh to confirm).`,
    `- Commit your change (git) with a clear message and update ${EFFORT_FILE} (a cycle log line).`,
    `- Keep it to ONE step; the loop will rebuild + re-run gemma and report back next cycle.`,
  ].join("\n");
}

async function runAgent(prompt: string): Promise<void> {
  log(`\x1b[1;34m${SEP}\n  🧠 AGENT (claude, effort=${CLAUDE_EFFORT})\n${SEP}\x1b[0m`);
  const args = ["-p", "--permission-mode", "bypassPermissions", "--effort", CLAUDE_EFFORT, prompt];
  await run("claude", args, { timeout: 1_800_000, stream: true });
}

// ── main ─────────────────────────────────────────────────────────────
async function main() {
  const argv = process.argv.slice(2);
  const dryRun = argv.includes("--dry-run");
  const cyclesIdx = argv.indexOf("--cycles");
  const maxCycles = cyclesIdx >= 0 ? parseInt(argv[cyclesIdx + 1] ?? "50", 10) : 50;

  log(`\x1b[1;35m${SEP}\n  ZINC gemma4 CUDA loop — target: ${GEMMA}  ref:"${REFERENCE}"\n${SEP}\x1b[0m`);

  let target = GEMMA;
  let phase = "DENSE";
  for (let cycle = 1; cycle <= maxCycles; cycle++) {
    log(`\n\x1b[1;33m▶ Cycle ${cycle}/${maxCycles}  [${phase}]\x1b[0m`);
    await syncToBox();
    const build = await buildOnBox();
    let runOut = "";
    if (build.ok) runOut = await runGemma(target);
    const status = build.ok ? classify(build, runOut) : "BUILD_FAIL";
    log(`  status=\x1b[1m${status}\x1b[0m`);
    if (build.ok) log(runOut.split("\n").slice(-6).map((l) => "    " + l).join("\n"));
    else log("    " + build.errors.split("\n").slice(0, 6).join("\n    "));

    if (status === "COHERENT") {
      log(`\x1b[1;32m  ✅ ${target} coherent!\x1b[0m`);
      if (phase === "DENSE") { phase = "MOE"; target = GEMMA_MOE; log(`  → advancing to gemma4 MoE: ${target}`); continue; }
      log(`\x1b[1;32m  🎉 both gemma4 models coherent — 5/5 catalog on the 4090.\x1b[0m`);
      return;
    }
    if (dryRun) { log("  (--dry-run: skipping agent)"); return; }
    await runAgent(buildPrompt(status, build, runOut, cycle));
  }
  log(`\x1b[1;31m  ⛔ reached ${maxCycles} cycles without full coherence.\x1b[0m`);
}

main().catch((e) => { console.error(e); process.exit(1); });
