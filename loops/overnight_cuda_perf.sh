#!/usr/bin/env bash
# Overnight autonomous CUDA perf deep-dive. Each cycle spawns a fresh `claude -p`
# session (clean context) that lands ONE validated perf increment, then stops;
# continuity is carried by git branches + memory. Safe by construction: each
# cycle validates 5/5 token-correctness before committing, commits only to
# perf/* branches (never main), and pins the 4090.
#
# Run from the repo root:  bash loops/overnight_cuda_perf.sh
# Stop in the morning with:  pkill -f overnight_cuda_perf
set -u
cd "$(dirname "$0")/.." || exit 1
PROMPT="$(cat loops/overnight_cuda_perf_prompt.txt)"
MAX=${MAX_CYCLES:-50}          # safety backstop (~8-12h of cycles)
LOG=/tmp/overnight_cuda_perf.log
echo "=== overnight CUDA perf loop started $(date) (max $MAX cycles) ===" | tee -a "$LOG"
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i + 1))
  echo "===== cycle $i / $MAX  —  $(date) =====" | tee -a "$LOG"
  claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG"
  echo "===== cycle $i done — $(date); sleeping 60s =====" | tee -a "$LOG"
  sleep 60
done
echo "=== overnight loop finished after $i cycles $(date) ===" | tee -a "$LOG"
