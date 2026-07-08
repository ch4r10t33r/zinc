#!/usr/bin/env bash
# Effort 30 — CUDA RTX 5090 PREFILL autonomous loop. Each cycle spawns a fresh
# `claude -p` (clean context) that reads the effort file + memory, lands ONE
# validated prefill increment to a perf/e30-* branch, then stops. Self-recovering
# (a crashed cycle doesn't stop the loop). Run from a DEDICATED worktree so it
# never disturbs the main checkout a parallel loop may own.
#
#   git worktree add ~/Workspace/zinc-e30 <main>
#   cd ~/Workspace/zinc-e30
#   nohup bash loops/run_prefill_e30.sh loops/efforts/MULTI_HOUR_EFFORT_30_CUDA_5090_PREFILL.md > /tmp/e30.log 2>&1 &
#   watch: tail -f /tmp/e30.log      stop: pkill -f run_prefill_e30
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
EFFORT="${1:?usage: run_prefill_e30.sh <effort-file.md>}"
[ -f "$EFFORT" ] || { echo "no effort file: $EFFORT"; exit 1; }
MAX=${MAX_CYCLES:-40}          # ~10-13h of cycles
SLEEP=${CYCLE_SLEEP:-60}
LOG="/tmp/perf_effort_e30.log"

PROMPT="You are autonomously advancing a scoped CUDA PREFILL perf effort on the ZINC inference engine (Zig) in ${ROOT} — a DEDICATED git worktree; work HERE, and use \`git worktree add\` for any branch work (the main checkout /Users/stepan/Workspace/zinc may be owned by a parallel loop and \`git checkout\` can abort). Land ONE validated increment, then STOP (do not loop yourself).

STEP 0 — READ: your memory (MEMORY.md + project_effort26_beat_llama.md + project_cuda_perf_blog.md) AND the effort file ${EFFORT}. It has the current state (the coalesced-attention win is DONE, +53% dense gemma-31b prefill), the measure-FIRST discipline, the phase profiles, the candidate levers (priority order), the DEAD ENDS (do not re-litigate), and the HARD RULES. Honor all of it.

THIS CYCLE:
1. If a perf/e30-* branch is mid-flight, continue it; else pick the next un-done lever from the effort file (flash-attention for dense is highest-EV; profile-gate anything new — the effort file shows how). Do NOT re-attempt a documented dead end.
2. Implement ONE focused change in this worktree.
3. Build on the box: rsync this whole worktree to zincbox:~/zinc-harvest/ (single-source './ dest/', never multi-source+--delete), then '~/zig-0.15.2/zig build -Dbackend=cuda -Dshaders=false -Doptimize=ReleaseFast'. Pin the 5090: CUDA_VISIBLE_DEVICES=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350.
4. CORRECTNESS GATE (validate_catalog is unusable in the box tree): confirm your change is token-identical to the shipped default via greedy '--prompt \"<real text>\" --raw -n 20' on gemma-26b AND gemma-31b, ≥2 real prompts — compare the 'Output (...)' line (STDERR, 2>&1). If tokens diverge, REVERT and log why.
5. MEASURE: interleaved ABBA A/B (≥4 rounds, drop the cold first round) of your change vs default, zinc prefill tok/s on the 5090; the box has ~±10% boost noise so require a consistent multi-round win. For 'beat llama', the bar is ~/workspace/llama.cpp/build/bin/llama-bench pp on the same gguf.
6. If a VALIDATED WIN: commit ONLY your scoped change to perf/e30-<short-target> via a git worktree, push it (NEVER main); append a dated one-liner to ${EFFORT}'s cycle log AND to project_effort26_beat_llama memory. If NEGATIVE: revert the code, log the finding in the effort file (negatives are valuable). Clean box scratch.

TIMEOUT DISCIPLINE (this cycle is killed at 50min): wrap EVERY box zinc run in 'timeout 200' so a buggy GPU-hanging kernel (e.g. a bad __syncthreads/OOB) cannot stall you — if a run returns EMPTY output, that kernel crashed/hung: REVERT it immediately, do not retry. Prefer isolated microbenches ('dbg_cuda gemm M K T') over full model reloads where possible.

MEMORY CONTINUITY (critical — the prior run re-litigated because cycles skipped this): BEFORE you finish — win, negative, OR running low on time — you MUST append a dated one-liner (target + verdict + branch) to BOTH project_effort26_beat_llama memory AND ${EFFORT}'s cycle log, so the next cycle does not repeat your work. Read those FIRST and never re-attempt a documented dead end (flash-attention is DEAD — do not rebuild it).

NEVER: break token-correctness, commit unvalidated code or a swept working tree, push to main, disturb other loops' worktrees/GPUs, or trust a single boost-noisy measurement. STOP after one increment. CONVERGENCE SELF-STOP: if your HONEST assessment is that the effort has CONVERGED — every lever is shipped or documented-dead and no valid autonomous 50-min increment remains — AND the cycle log already shows a prior cycle reaching the same STOP/converged conclusion, then instead of appending YET ANOTHER redundant confirmation, write the file /tmp/e30_converged containing a one-line reason; the driver halts the loop when that file exists. Only write it when genuinely converged — a real new lever means keep working, do NOT write it."

echo "=== e30 prefill loop START $(date -u +%FT%TZ) — 5090, root=$ROOT, max $MAX ===" | tee -a "$LOG"
i=0
rm -f /tmp/e30_converged   # fresh start clears any stale convergence sentinel
while [ "$i" -lt "$MAX" ]; do
  if [ -f /tmp/e30_converged ]; then
    echo "=== e30 HALT: convergence sentinel present — $(head -1 /tmp/e30_converged 2>/dev/null) ($(date -u +%FT%TZ)) ===" | tee -a "$LOG"
    break
  fi
  i=$((i + 1))
  echo "===== e30 cycle $i / $MAX — $(date -u +%FT%TZ) =====" | tee -a "$LOG"
  # macOS has no `timeout`; perl alarm survives exec → hard-kills a cycle at 50min.
  perl -e 'alarm shift; exec @ARGV' 3000 claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG" \
    || echo "(cycle $i exited nonzero/timed-out — self-recovering)" | tee -a "$LOG"
  # kill any box zinc/build a hung cycle may have orphaned before the next cycle
  ssh -o BatchMode=yes -o ConnectTimeout=8 zincbox 'pkill -f "zig-out/bin/zinc" 2>/dev/null; pkill -f "zig build" 2>/dev/null' >/dev/null 2>&1 || true
  echo "===== e30 cycle $i done — $(date -u +%FT%TZ); sleeping ${SLEEP}s =====" | tee -a "$LOG"
  sleep "$SLEEP"
done
echo "=== e30 prefill loop FINISHED after $i cycles $(date -u +%FT%TZ) ===" | tee -a "$LOG"
