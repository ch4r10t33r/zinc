#!/usr/bin/env bash
# loops/supervise_perf.sh — auto-restarting supervisor for optimize_perf.ts.
#
# optimize_perf.ts runs N cycles then exits, and a single transient exit
# (bun crash, agent-subprocess hard exit, OOM, SSH storm) otherwise halts a
# multi-hour effort. This wrapper treats any exit as restartable: it re-runs
# the loop with --resume (falling back to a fresh baseline on the first
# iteration / when no state exists), so the effort progresses until killed.
#
# Usage:
#   nohup bun loops/supervise_perf.sh 25 opencode 40 >> /tmp/zinc_rdna2_loop.log 2>&1 &
#   loops/supervise_perf.sh 17 codex 30          # effort 17, codex, 30 cycles/iter
#   ZINC_RDNA_NODE=rdna2 loops/supervise_perf.sh 25 opencode
#
# Args:  $1 = effort id (default 25)
#        $2 = agent    (default opencode)
#        $3 = cycles per iteration (default 40)
# Env:   ZINC_RDNA_NODE (default rdna2)
#        ZINC_PERF_LOOP_LOG (default /tmp/zinc_rdna2_loop.log)
#        plus the usual ZINC_OPENCODE_MODEL / ZINC_OPENCODE_VARIANT / .env keys
# Stop:  pkill -f supervise_perf.sh  (and the bun optimize_perf.ts child it owns)

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Worktree-local .env (gitignored) — present in the runnable worktree, not in a
# fresh clone. Optional; the loop also reads process env directly.
if [ -f "$REPO/.env" ]; then
  set -a
  . "$REPO/.env"
  set +a
fi
export ZINC_RDNA_NODE="${ZINC_RDNA_NODE:-rdna2}"

EFFORT="${1:-25}"
AGENT="${2:-opencode}"
CYCLES="${3:-40}"
STATE="$REPO/.perf_optimize/effort_${EFFORT}_state.json"
LOG="${ZINC_PERF_LOOP_LOG:-/tmp/zinc_rdna2_loop.log}"

echo "[supervisor] effort=$EFFORT agent=$AGENT node=$ZINC_RDNA_NODE cycles/iter=$CYCLES"
echo "[supervisor] log=$LOG"
echo "[supervisor] state=$STATE"

while true; do
  if [ -f "$STATE" ]; then
    echo "[supervisor] $(date '+%F %T') — resume (prior state present)"
    bun loops/optimize_perf.ts --effort "$EFFORT" --agent "$AGENT" --cycles "$CYCLES" --resume
  else
    echo "[supervisor] $(date '+%F %T') — fresh start (no prior state)"
    bun loops/optimize_perf.ts --effort "$EFFORT" --agent "$AGENT" --cycles "$CYCLES"
  fi
  ec=$?
  # A normal exit (cycles exhausted) returns 0 and is still restartable — the
  # goal is "run until killed", not "stop after N cycles". Any non-zero is a
  # crash; --resume on the next iteration picks up from the last committed
  # cycle boundary (state is written only after a keep/revert decision, so a
  # mid-cycle crash never corrupts it).
  echo "[supervisor] $(date '+%F %T') — loop exited code=$ec; restarting in 15s"
  sleep 15
done
