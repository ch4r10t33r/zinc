#!/usr/bin/env bash
# Effort 31 — CUDA RTX 5090 DECODE autonomous loop. Each cycle spawns a fresh
# `claude -p` (clean context) that reads the effort file + memory, lands ONE
# validated decode increment to a perf/e31-* branch, then stops. Hardened:
# perl-alarm 50min hang-timeout, orphan-kill between cycles, convergence self-stop.
# Run from a DEDICATED worktree so it never disturbs the main checkout.
#
#   git worktree add ~/Workspace/zinc-e31 origin/main
#   cd ~/Workspace/zinc-e31
#   nohup bash loops/run_decode_e31.sh loops/efforts/MULTI_HOUR_EFFORT_31_CUDA_5090_DECODE.md > /tmp/e31_loop.log 2>&1 &
#   watch: tail -f /tmp/e31_loop.log      stop: pkill -f run_decode_e31
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
EFFORT="${1:?usage: run_decode_e31.sh <effort-file.md>}"
[ -f "$EFFORT" ] || { echo "no effort file: $EFFORT"; exit 1; }
MAX=${MAX_CYCLES:-25}
SLEEP=${CYCLE_SLEEP:-60}
LOG="/tmp/perf_effort_e31.log"

PROMPT="You are autonomously advancing a scoped CUDA DECODE perf effort on the ZINC inference engine (Zig) in ${ROOT} — a DEDICATED git worktree; work HERE, and use \`git worktree add\` for any branch work (the main checkout may be owned by a parallel loop and \`git checkout\` can abort). Land ONE validated increment, then STOP (do not loop yourself).

STEP 0 — READ: your memory (MEMORY.md + project_5090_decode_gap.md + project_effort26_beat_llama.md) AND the effort file ${EFFORT}. It has the MEASURED decode gaps (MoE worst: gemma-26b 32%, qwen-a3b 39% of llama), the root cause (decode is COMPUTE/LATENCY-bound not launch-bound → CUDA GRAPHS ARE NEUTRAL/DEAD; the LM head is only 7%; the LAYERS are 90% at ~8% BW), the sole viable lever (STATIC KERNEL FUSION of adjacent single-block launches — the Effort-27 playbook), and the HARD RULES. Honor all of it. Do NOT re-attempt graphs or LM-head work (dead).

THIS CYCLE:
1. If a perf/e31-* branch is mid-flight, continue it; else pick the next un-done fusion adjacency from the effort file. SCAN the decode hot path (forward_cuda_gemma.zig moeFfnBlock/attentionLayer/tail, forward_cuda.zig for qwen) for two ADJACENT single-block rms/elementwise launches (same or chained buffers) that can fuse into ONE kernel, block-count preserved → bit-identical.
2. Implement ONE focused fusion in this worktree.
3. Build on the box: rsync this whole worktree to zincbox:~/zinc-harvest/ (single-source './ dest/'), then '~/zig-0.15.2/zig build -Dbackend=cuda -Dshaders=false -Doptimize=ReleaseFast'. Pin the 5090: CUDA_VISIBLE_DEVICES=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350.
4. CORRECTNESS GATE: fusion is usually BIT-identical (same kernels reordered) → require token-IDENTICAL output to the shipped default on gemma-26b AND qwen (greedy '--prompt \"<real text>\" --raw -n 20', compare the 'Output(...)' line, STDERR 2>&1). A DIFFER = REVERT.
5. MEASURE DECODE: interleaved ABBA A/B (≥4 rounds, drop the cold first), zinc decode tok/s (parse 'Generated N tokens in … — X tok/s'); use a RAMBLE prompt (repeat a sentence ~8×) so the model generates ≥100 tokens (short prompts EOS early). Decode is boost-noisy (±10%) → require a consistent multi-round win.
6. If a VALIDATED WIN: commit ONLY your scoped change to perf/e31-<short-target> via a git worktree, push it (NEVER main); append a dated one-liner to ${EFFORT}'s cycle log AND to project_5090_decode_gap memory. If NEGATIVE: revert, log the finding (negatives are valuable).

TIMEOUT: wrap EVERY box zinc run in 'timeout 200' (a decode kernel bug can hang the GPU); empty output = crashed → revert, don't retry. CONVERGENCE SELF-STOP: if genuinely converged (no valid 50-min fusion increment remains) AND a prior cycle already said so, write /tmp/e31_converged (one-line reason) and STOP — do not append redundant confirmations.

NEVER: break token-correctness, commit unvalidated code or a swept working tree, push to main, disturb other loops' worktrees/GPUs, or trust a single boost-noisy measurement. STOP after one increment."

rm -f /tmp/e31_converged
echo "=== e31 decode loop START $(date -u +%FT%TZ) — 5090, root=$ROOT, max $MAX ===" | tee -a "$LOG"
i=0
while [ "$i" -lt "$MAX" ]; do
  if [ -f /tmp/e31_converged ]; then
    echo "=== e31 HALT: convergence sentinel — $(head -1 /tmp/e31_converged 2>/dev/null) ($(date -u +%FT%TZ)) ===" | tee -a "$LOG"
    break
  fi
  i=$((i + 1))
  echo "===== e31 cycle $i / $MAX — $(date -u +%FT%TZ) =====" | tee -a "$LOG"
  perl -e 'alarm shift; exec @ARGV' 3000 claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG" \
    || echo "(cycle $i exited nonzero/timed-out — self-recovering)" | tee -a "$LOG"
  ssh -o BatchMode=yes -o ConnectTimeout=8 zincbox 'pkill -f "zig-out/bin/zinc" 2>/dev/null; pkill -f "zig build" 2>/dev/null' >/dev/null 2>&1 || true
  echo "===== e31 cycle $i done — $(date -u +%FT%TZ); sleeping ${SLEEP}s =====" | tee -a "$LOG"
  sleep "$SLEEP"
done
echo "=== e31 decode loop FINISHED after $i cycles $(date -u +%FT%TZ) ===" | tee -a "$LOG"
