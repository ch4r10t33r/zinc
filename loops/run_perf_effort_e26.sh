#!/usr/bin/env bash
# Effort 26 (beat-llama: prefill graphs + MoE decode) autonomous loop. Twin of
# run_perf_effort.sh but pinned to the RTX 5090 and the ~/zinc-e26 box dir, and
# it runs in its OWN git worktree — so it advances ALONGSIDE the Effort-25 decode
# loop (4090, ~/zinc-e25, the main checkout) without contending on GPU, box dir,
# or working tree. Each cycle spawns a fresh `claude -p` that reads
# MULTI_HOUR_EFFORT_26_BEAT_LLAMA.md, lands ONE validated increment, commits to a
# perf/e26-* branch, appends to the cycle log, then stops.
#
#   bash loops/run_perf_effort_e26.sh loops/efforts/MULTI_HOUR_EFFORT_26_BEAT_LLAMA.md
#   stop with:  pkill -f run_perf_effort_e26
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
EFFORT="${1:?usage: run_perf_effort_e26.sh <effort-file.md>}"
[ -f "$EFFORT" ] || { echo "no effort file: $EFFORT"; exit 1; }
MAX=${MAX_CYCLES:-40}
LOG="/tmp/perf_effort_$(basename "$EFFORT" .md).log"

PROMPT="You are autonomously advancing a scoped CUDA perf EFFORT on the ZINC inference engine (Zig), in ${ROOT} (a DEDICATED git worktree — work HERE, never in /Users/stepan/Workspace/zinc, which the parallel Effort-25 decode loop owns), then STOPPING after ONE validated increment.

STEP 0 — READ: your memory (MEMORY.md + project_zinc_cuda_backend.md + project_cuda_perf_blog.md) AND the effort file ${EFFORT}. It holds the targets (BEAT llama.cpp on every prefill + MoE-decode row where ZINC trails), the LEVER (gemma prefill is LAUNCH-bound, ~10% util → CUDA GRAPHS over the prefill chain, NOT faster GEMM kernels — Effort 24 proved the tensor-core GEMM is end-to-end NEUTRAL), the HARD RULES, the validation contract, and the cycle log. Honor all of it.

HARD CONSTRAINTS for THIS effort (they OVERRIDE the generic playbook): pin the RTX 5090 — export CUDA_VISIBLE_DEVICES=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350 and run validate_catalog with ZINC_GPU=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350. The Effort-25 loop owns the 4090 (GPU-e59a6fce-...) and ~/zinc-e25 — do NOT touch them. Use the isolated box dir ~/zinc-e26 (never ~/workspace/zinc). Isolated-cache builds (ZIG_LOCAL_CACHE_DIR+ZIG_GLOBAL_CACHE_DIR; verify the binary md5 actually changed or you are measuring stale code). DO NOT async gemma decode (proven regression). Box gotchas: PREFILL/DECODE tok/s prints to STDERR (2>&1); always 'nohup CMD >FILE 2>&1 &' on the box and poll FILE; util-gate A/B via --query-gpu=utilization.gpu; gemma-31B reloads 18GB/call.

THIS CYCLE: pick the next un-done target from the effort file (or continue an in-progress perf/e26-* branch you find), implement ONE focused change, build with isolated caches, run scripts/validate_catalog.sh (ZINC_GPU=the 5090 UUID) — it MUST stay 5/5 token-correct; if correctness breaks, REVERT and document why. Measure an interleaved A/B and compare zinc tok/s vs llama.cpp on the SAME 5090 + same gguf (the BAR is BEAT llama — use ~/workspace/llama.cpp/build/bin/llama-bench or the perf suite for the baseline). If it is a VALIDATED WIN, commit ONLY this change to perf/e26-<short-target> and push it (NOT main); append a dated entry to the effort file's cycle log AND memory. If NEGATIVE, revert the code and log the finding (negatives are valuable). Clean up box scratch dirs. STOP — do not loop yourself.

NEVER: break catalog correctness, commit unvalidated code or a swept working tree (commit ONLY your scoped change), push to main, async gemma, disturb the 4090/Effort-25 work or /Users/stepan/Workspace/zinc, or trust a single boost-noisy measurement."

echo "=== e26 beat-llama loop: $EFFORT  (5090, root=$ROOT, max $MAX cycles)  $(date) ===" | tee -a "$LOG"
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i + 1))
  echo "===== e26 cycle $i / $MAX  —  $(date) =====" | tee -a "$LOG"
  claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG"
  echo "===== e26 cycle $i done — $(date); sleeping 60s =====" | tee -a "$LOG"
  sleep 60
done
echo "=== e26 loop finished after $i cycles $(date) ===" | tee -a "$LOG"
