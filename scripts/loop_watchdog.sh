#!/usr/bin/env bash
# loop_watchdog.sh — keep an autonomous perf loop alive: restart its driver on
# EXIT *or* HANG. The plain restart-on-exit supervisor misses hangs — a wedged
# `claude -p` blocks the driver without it ever exiting, so nothing restarts and
# the GPU sits idle for hours (observed: e27 frozen ~5h on one cycle). This
# watchdog watches the driver's tee-LOG mtime and force-restarts a stale cycle.
#
#   WORKTREE=/Users/stepan/Workspace/zinc-e27 \
#   DRIVER=run_perf_effort_4090.sh \
#   EFFORT=loops/efforts/MULTI_HOUR_EFFORT_27_CUDA_4090_DECODE.md \
#   LOG=/tmp/perf_effort_MULTI_HOUR_EFFORT_27_CUDA_4090_DECODE.log \
#   nohup bash scripts/loop_watchdog.sh >/tmp/e27_watchdog.log 2>&1 &
#
# Env: WORKTREE, DRIVER (script under loops/), EFFORT (path arg to the driver),
#      LOG (driver tee-log to watch), STALE_SECS (default 2700=45m — keep > the
#      slowest legit silent cycle so a long A/B isn't false-killed), MAX_CYCLES
#      (200), CHECK_SECS (300). Stop: pkill -f loop_watchdog (then kill the driver).
set -u
WORKTREE="${WORKTREE:?}"; DRIVER="${DRIVER:?}"; EFFORT="${EFFORT:?}"; LOG="${LOG:?}"
STALE_SECS="${STALE_SECS:-2700}"; MAX_CYCLES="${MAX_CYCLES:-200}"; CHECK_SECS="${CHECK_SECS:-300}"
wlog(){ echo "[watchdog $(date '+%m-%d %H:%M:%S')] $*"; }
start_driver() {
  ( cd "$WORKTREE" \
      && git fetch origin -q 2>/dev/null \
      && git checkout -q -f --detach origin/main 2>/dev/null
    MAX_CYCLES="$MAX_CYCLES" exec bash "loops/$DRIVER" "$EFFORT" ) &
  echo $!
}
PID=$(start_driver); wlog "started driver pid $PID (worktree $WORKTREE, stale>${STALE_SECS}s)"
while true; do
  sleep "$CHECK_SECS"
  now=$(date +%s); mt=$(stat -f %m "$LOG" 2>/dev/null || echo "$now"); age=$((now - mt))
  alive=0; kill -0 "$PID" 2>/dev/null && alive=1
  if [ "$alive" -eq 0 ] || [ "$age" -gt "$STALE_SECS" ]; then
    wlog "RESTART (driver_alive=$alive log_age=${age}s > ${STALE_SECS}s) — killing pid $PID + children"
    pkill -P "$PID" 2>/dev/null; kill -9 "$PID" 2>/dev/null; sleep 5
    PID=$(start_driver); wlog "new driver pid $PID"
  fi
done
