#!/usr/bin/env bash
# Effort 28 (multi-tenant + continuous request-batching on CUDA) autonomous loop.
# UNLIKE the perf loops (run_perf_effort_e26/4090.sh, which land an independent
# A/B perf tweak per cycle on a fresh perf/* branch), this is a MULTI-CYCLE BUILD
# on ONE persistent branch `feat/e28-batching`: each cycle reads the effort file's
# CURRENT STATE, advances the next increment/sub-step, validates it (batched==serial
# token-identical + catalog 5/5 on the serial path), commits to feat/e28-batching,
# updates CURRENT STATE + the cycle log, then stops. Runs in its own worktree
# (/Users/stepan/Workspace/zinc-e28) and owns the now-dedicated agent-zinc box.
#
#   bash loops/run_perf_effort_e28.sh loops/efforts/MULTI_HOUR_EFFORT_28_CUDA_BATCHING.md
#   stop with:  pkill -f run_perf_effort_e28
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
EFFORT="${1:?usage: run_perf_effort_e28.sh <effort-file.md>}"
[ -f "$EFFORT" ] || { echo "no effort file: $EFFORT"; exit 1; }
MAX=${MAX_CYCLES:-40}
LOG="/tmp/perf_effort_$(basename "$EFFORT" .md).log"

PROMPT="You are autonomously building MULTI-TENANT SERVING + CONTINUOUS (request-level) BATCHING into the ZINC CUDA backend (Zig), in ${ROOT} (a DEDICATED git worktree on the PERSISTENT branch feat/e28-batching — work HERE, never in /Users/stepan/Workspace/zinc [main checkout], …/zinc-e26, or …/zinc-e27), then STOPPING after ONE validated increment.

STEP 0 — READ: your memory (MEMORY.md + project_zinc_cuda_backend.md + project_effort28_batching.md if present) AND the effort file ${EFFORT} IN FULL. It is the spec: the honest baseline (server is thread-per-request but GPU-serialized behind generation_mutex; CUDA forward is single-sequence; src/scheduler/* exists but is UNWIRED), the KEY INSIGHT (batched DECODE ≈ the batched PREFILL that already works in prefillBatched — the qkv/o/ffn GEMMs are already B-row-capable; the ONLY single-sequence assumptions are gemma_attention_batched's 'seq_len = t+1' over a shared KV (kernels.cu:3150) and the kv_write position — generalize those to per-sequence positions[b] + a per-sequence KV slot offset), the ORDERED DEPENDENT increments, the validation contract, the HARD RULES, and especially the '## CURRENT STATE' pointer (done-so-far + exact next step) and the cycle log. Honor all of it. This is a BUILD, not a perf A/B — increments DEPEND on each other; do them in order; reuse prefillBatched/BatchScratch, do NOT write a serving engine from scratch.

HARD CONSTRAINTS (override the generic playbook): persistent branch feat/e28-batching (accumulate increments; push each cycle; NEVER main; NEVER a fresh perf/* branch). Batching is ADDITIVE — the production single-sequence decodeStep/prefillBatched/server path MUST stay the default and stay correct at every commit; add batched paths behind a NEW entrypoint (e.g. a 'dbg_cuda batch' harness / a decodeBatch fn) until increment 3 deliberately flips serving over. Pin the RTX 5090: export CUDA_VISIBLE_DEVICES=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350 and run validate_catalog + measurements with ZINC_GPU=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350. The box is DEDICATED to this effort now (e26/e27 perf loops were stopped); the 4090 (GPU-e59a6fce-…) is also free for a parallel build/validate. Isolated box dir ~/zinc-e28 (rsync source there; never ~/workspace/zinc). Isolated-cache builds (ZIG_LOCAL_CACHE_DIR+ZIG_GLOBAL_CACHE_DIR; verify the binary md5 actually changed or you are measuring stale code). DO NOT async gemma decode (proven regression; orthogonal to batching). Box gotchas: DECODE/PREFILL/GEN_IDS tok/s print to STDERR (2>&1); always 'nohup CMD >FILE 2>&1 &' on the box and poll FILE; gemma-31B reloads 18GB/call (~45s) and can WEDGE WSL2 sshd → keep B and max_ctx modest, batch box work; kill by PID not pkill-self-match.

THIS CYCLE: from '## CURRENT STATE' pick the exact next sub-step (or continue what you find in-progress on feat/e28-batching), implement ONE focused increment, build with isolated caches (md5 changed?). VALIDATE per the contract: (1) the production SERIAL path must stay scripts/validate_catalog.sh 5/5 token-correct (ZINC_GPU=5090 UUID); (2) for batched work, the batched output must be TOKEN-IDENTICAL to N separate single-sequence runs (the dbg_cuda batch proof, mixed positions). If correctness breaks, FIX or REVERT and document why. If the increment is VALIDATED, commit ONLY this change to feat/e28-batching and push it; then UPDATE the effort file's '## CURRENT STATE' (done-so-far + exact next step + new risks) AND append a dated cycle-log entry AND your memory. If BLOCKED (box wedged) or the step is a NEGATIVE, log the finding, leave the tree clean, and update CURRENT STATE. Clean up box scratch dirs. STOP — do not loop yourself.

NEVER: break the serial catalog correctness, leave the production single-sequence path broken, commit unvalidated/swept code (commit ONLY your scoped change), push to main, async gemma decode, disturb /Users/stepan/Workspace/zinc or …/zinc-e26 or …/zinc-e27, commit host/IP/port, or trust a single boost-noisy throughput measurement."

echo "=== e28 batching build loop: $EFFORT  (5090, root=$ROOT, branch feat/e28-batching, max $MAX cycles)  $(date) ===" | tee -a "$LOG"
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i + 1))
  echo "===== e28 cycle $i / $MAX  —  $(date) =====" | tee -a "$LOG"
  claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG"
  echo "===== e28 cycle $i done — $(date); sleeping 60s =====" | tee -a "$LOG"
  sleep 60
done
echo "=== e28 loop finished after $i cycles $(date) ===" | tee -a "$LOG"
