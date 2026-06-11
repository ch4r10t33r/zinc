#!/usr/bin/env bash
# Run a SCOPED CUDA perf effort with an autonomous loop. Each cycle spawns a
# fresh `claude -p` that reads the effort file, lands ONE validated increment
# scoped to that effort, commits only that change to a perf/* branch, appends to
# the effort's cycle log, then stops. Continuity via the effort file + git
# branches + memory (not a single long context). Safe by construction:
# validate-before-commit, branches-not-main, 4090-pinned, isolated-cache builds.
#
#   bash loops/run_perf_effort.sh loops/efforts/MULTI_HOUR_EFFORT_23_GEMMA_ATTN_FUSION.md
#   stop with:  pkill -f run_perf_effort
set -u
cd "$(dirname "$0")/.." || exit 1
EFFORT="${1:?usage: run_perf_effort.sh <effort-file.md>}"
[ -f "$EFFORT" ] || { echo "no effort file: $EFFORT"; exit 1; }
MAX=${MAX_CYCLES:-40}
LOG="/tmp/perf_effort_$(basename "$EFFORT" .md).log"

PROMPT="You are autonomously advancing a scoped CUDA perf EFFORT on the ZINC inference engine (Zig), in /Users/stepan/Workspace/zinc, then STOPPING after ONE validated increment.

STEP 0 — READ: your memory (MEMORY.md + the project_zinc_cuda_backend.md it points to) AND the effort file: ${EFFORT}. It holds the targets, the HARD RULES (isolated-cache builds with ZIG_LOCAL_CACHE_DIR+ZIG_GLOBAL_CACHE_DIR and a verified hash change; DO NOT async gemma — boost-saturated, proven regression; 4090-pinned by UUID; isolated box dir ~/zinc-eNN, never ~/workspace/zinc; do not disturb the parallel 5090 work; interleaved back-to-back A/B for boost noise), the validation contract, and the cycle log. Honor all of it.

THIS CYCLE: pick the next un-done target from the effort file (or continue an in-progress perf/* branch you find), implement ONE focused change, build with ISOLATED caches (verify the binary hash changed or you are measuring stale code), run scripts/validate_catalog.sh — it MUST stay 5/5 token-correct (the fused kernel must be bit-equivalent); if correctness breaks, REVERT and document why. Measure an interleaved A/B vs the pre-cycle binary. If it is a VALIDATED WIN, commit ONLY this change to perf/e<N>-<short-target> and push it (NOT main); append a dated entry to the effort file's cycle log AND memory. If NEGATIVE, revert the code and log the finding (negatives are valuable). Clean up box scratch dirs. STOP — do not loop yourself.

NEVER: break catalog correctness, commit unvalidated code or a swept working tree (commit only your scoped change), push to main, async gemma, disturb the 5090 work, or trust a single boost-noisy measurement."

echo "=== perf-effort loop: $EFFORT  (max $MAX cycles)  $(date) ===" | tee -a "$LOG"
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i + 1))
  echo "===== cycle $i / $MAX  —  $(date) =====" | tee -a "$LOG"
  claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG"
  echo "===== cycle $i done — $(date); sleeping 60s =====" | tee -a "$LOG"
  sleep 60
done
echo "=== effort loop finished after $i cycles $(date) ===" | tee -a "$LOG"
