#!/usr/bin/env bash
#
# gpu_monitor.sh — live utilisation / temperature monitor for the remote CUDA node.
#
# Polls `nvidia-smi` on a remote CUDA node over a multiplexed SSH connection
# and renders a refreshing dashboard. The node name must come from GPU_NODE or
# --node so public checkouts do not carry private SSH aliases.
#
# Usage:
#   scripts/gpu_monitor.sh                 # live dashboard, refresh every 2s
#   scripts/gpu_monitor.sh -n 1            # refresh every 1s
#   scripts/gpu_monitor.sh --once          # one snapshot, then exit (good for scripts/cron)
#   scripts/gpu_monitor.sh --log gpu.csv   # live dashboard + append timestamped CSV here
#   scripts/gpu_monitor.sh --no-color      # disable ANSI colour
#
# Remote logging (runs detached ON the node, survives SSH disconnect / laptop sleep):
#   scripts/gpu_monitor.sh --remote-log start    # start a background CSV logger on the node
#   scripts/gpu_monitor.sh --remote-log status   # running? row count + latest sample
#   scripts/gpu_monitor.sh --remote-log tail     # last 15 logged rows
#   scripts/gpu_monitor.sh --remote-log fetch    # scp the node's CSV down to here
#   scripts/gpu_monitor.sh --remote-log stop     # stop the logger (CSV stays on the node)
#
# Env:
#   GPU_NODE   ssh host/alias to poll
#
set -euo pipefail

NODE="${GPU_NODE:-}"
INTERVAL=2
ONCE=0
LOGFILE=""
USE_COLOR=1
REMOTE_LOG=""

usage() { sed -n '3,24p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--interval) INTERVAL="${2:?missing seconds}"; shift 2 ;;
    --once)        ONCE=1; shift ;;
    --log)         LOGFILE="${2:?missing file}"; shift 2 ;;
    --node)        NODE="${2:?missing node}"; shift 2 ;;
    --no-color)    USE_COLOR=0; shift ;;
    --remote-log)
      case "${2:-}" in
        start|stop|status|fetch|tail) REMOTE_LOG="$2"; shift 2 ;;
        *)                            REMOTE_LOG="start"; shift 1 ;;
      esac ;;
    -h|--help)     usage 0 ;;
    [0-9]*)        INTERVAL="$1"; shift ;;          # bare number => interval
    *)             echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$NODE" ]]; then
  echo "missing GPU node; set GPU_NODE=<ssh-host-or-alias> or pass --node" >&2
  exit 2
fi

# Colour palette (empty when disabled / not a TTY).
if [[ "$USE_COLOR" == 1 && -t 1 ]]; then
  RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'
  CYN=$'\033[0;36m'; DIM=$'\033[2m';    BLD=$'\033[1m'; RST=$'\033[0m'
else
  RED=""; GRN=""; YEL=""; CYN=""; DIM=""; BLD=""; RST=""; USE_COLOR=0
fi

# Multiplexed SSH: first call opens a master, the rest reuse it (cheap per-tick),
# and the master auto-reconnects if the tailnet path flaps. Keep ControlPath short
# and out of $TMPDIR — macOS $TMPDIR overruns the 104-char Unix-socket limit.
CTL="/tmp/.gpumon-$(id -u)-%C"
# Keepalive is deliberately lenient (60s, not 10s): when the node is busy (e.g. a
# 35B model loaded), responses lag and a tight ServerAlive kills the mux master
# mid-command ("broken pipe"). 60s tolerates a slow node without false drops.
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15
     -o ServerAliveInterval=10 -o ServerAliveCountMax=6
     -o ControlMaster=auto -o ControlPersist=30 -o ControlPath="$CTL")
SCP=(scp -o BatchMode=yes -o ConnectTimeout=15
     -o ControlMaster=auto -o ControlPersist=30 -o ControlPath="$CTL")

QUERY='index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit'
REMOTE_CMD="nvidia-smi --query-gpu=${QUERY} --format=csv,noheader,nounits"

cleanup() {
  [[ "$USE_COLOR" == 1 ]] && printf '\033[?25h' >&2   # restore cursor
  "${SSH[@]}" -O exit "$NODE" >/dev/null 2>&1 || true  # close the master
}
trap cleanup EXIT
trap 'exit 130' INT TERM   # turn Ctrl-C into a real exit so the EXIT trap fires once

# render_rows: read CSV on stdin, print an aligned, coloured table body.
render_rows() {
  awk -F', *' -v RED="$RED" -v GRN="$GRN" -v YEL="$YEL" -v CYN="$CYN" \
              -v DIM="$DIM" -v RST="$RST" '
    {
      idx=$1; name=$2; util=$3+0; memu=$5+0; memt=$6+0; temp=$7+0; pdraw=$8+0; plim=$9+0;
      sub(/^NVIDIA /,"",name); sub(/^GeForce /,"",name);
      w=10; f=int(util/10 + 0.5); if(f>w)f=w; if(f<0)f=0;
      bar=""; for(i=0;i<f;i++) bar=bar "#"; for(i=f;i<w;i++) bar=bar ".";
      uc = (util>=80?RED:(util>=40?YEL:GRN));
      tc = (temp>=80?RED:(temp>=60?YEL:GRN));
      mp = (memt>0?int(memu*100/memt):0);
      printf " %-2s %-9s [%s%s%s] %s%3d%%%s   %s%3d C%s   %5.1f/%5.1f GiB %3d%%   %4d/%4d W\n",
        idx, name, uc, bar, RST, uc, util, RST, tc, temp, RST, memu/1024, memt/1024, mp, pdraw, plim;
    }'
}

header() {
  printf '%s #  GPU        Utilisation   Temp     Memory                  Power%s\n' "$BLD" "$RST"
}

# Append timestamped rows to the CSV log (writes a header row on first use).
log_rows() {
  local data="$1" ts
  ts="$(date '+%Y-%m-%dT%H:%M:%S')"
  if [[ ! -s "$LOGFILE" ]]; then
    echo "timestamp,host,index,name,util_gpu,util_mem,mem_used_mib,mem_total_mib,temp_c,power_w,power_limit_w" >"$LOGFILE"
  fi
  printf '%s\n' "$data" | awk -F', *' -v ts="$ts" -v h="$NODE" 'BEGIN{OFS=","}
    { print ts, h, $1, $2, $3, $4, $5, $6, $7, $8, $9 }' >>"$LOGFILE"
}

frame() {                       # one poll; echoes raw CSV. Falls back to a direct
                                # (non-mux) connection if the control master is wedged
                                # — happens when the node is too memory-starved to
                                # stand up a multiplexed session but plain ssh still works.
  local out
  out="$("${SSH[@]}" "$NODE" "$REMOTE_CMD" 2>/dev/null)"
  [ -n "$out" ] && { printf '%s\n' "$out"; return 0; }
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$NODE" "$REMOTE_CMD" 2>/dev/null
}

# --- remote logging: a detached nvidia-smi logger that lives ON the node -------
# State on the node is under ~/.gpu_monitor/ (run.sh, logger.pid, thermals.csv).
# The logger runs in its own session with stdio detached, so it survives this SSH
# session closing and your laptop sleeping. Manage via start/stop/status/tail/fetch.
remote_log() {
  local action="$1"
  case "$action" in
    start)
      local logger supervisor lb64 sb64
      # Single-quoted so nothing expands locally; shipped to the node as base64.
      # run.sh = the sampler (append-mode, never truncates). supervisor.sh keeps it
      # alive: if the sampler dies (OOM, transient kill), it's respawned within ~30s.
      # The supervisor persists because the WSL VM has multi-day uptime; `start` is
      # idempotent so a cron/keepalive can re-ensure the supervisor itself.
      logger='#!/usr/bin/env bash
RDIR="$HOME/.gpu_monitor"; CSV="$RDIR/thermals.csv"; PIDF="$RDIR/logger.pid"
INT="${1:-2}"
echo $$ > "$PIDF"
[ -s "$CSV" ] || echo "timestamp,index,name,util_gpu,util_mem,mem_used_mib,mem_total_mib,temp_c,power_w,power_limit_w" > "$CSV"
while :; do
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits >> "$CSV" 2>/dev/null
  sleep "$INT"
done'
      supervisor='#!/usr/bin/env bash
RDIR="$HOME/.gpu_monitor"; SPID="$RDIR/supervisor.pid"; LPID="$RDIR/logger.pid"
echo $$ > "$SPID"
INT="${1:-2}"
launch() { if command -v setsid >/dev/null 2>&1; then setsid bash "$RDIR/run.sh" "$INT" >/dev/null 2>&1 </dev/null & else nohup bash "$RDIR/run.sh" "$INT" >/dev/null 2>&1 </dev/null & fi; }
while :; do
  if [ ! -f "$LPID" ] || ! kill -0 "$(cat "$LPID" 2>/dev/null)" 2>/dev/null; then launch; fi
  sleep 30
done'
      lb64="$(printf '%s' "$logger" | base64 | tr -d '\n')"
      sb64="$(printf '%s' "$supervisor" | base64 | tr -d '\n')"
      "${SSH[@]}" "$NODE" "
        RDIR=\"\$HOME/.gpu_monitor\"; mkdir -p \"\$RDIR\"; SPID=\"\$RDIR/supervisor.pid\";
        echo \"$lb64\" | base64 -d > \"\$RDIR/run.sh\";
        echo \"$sb64\" | base64 -d > \"\$RDIR/supervisor.sh\";
        if [ -f \"\$SPID\" ] && kill -0 \"\$(cat \"\$SPID\" 2>/dev/null)\" 2>/dev/null; then
          echo \"already running (supervisor pid \$(cat \"\$SPID\"))\"; exit 0; fi
        if command -v setsid >/dev/null 2>&1; then
          setsid bash \"\$RDIR/supervisor.sh\" $INTERVAL >/dev/null 2>&1 </dev/null &
        else
          nohup bash \"\$RDIR/supervisor.sh\" $INTERVAL >/dev/null 2>&1 </dev/null &
        fi
        sleep 1.5;
        if [ -f \"\$SPID\" ] && kill -0 \"\$(cat \"\$SPID\" 2>/dev/null)\" 2>/dev/null; then
          echo \"started (supervisor \$(cat \"\$SPID\"), logger \$(cat \"\$RDIR/logger.pid\" 2>/dev/null || echo ?), every ${INTERVAL}s, auto-respawn) -> \$RDIR/thermals.csv\";
        else echo \"failed to start\" >&2; exit 1; fi" ;;
    stop)
      # Kill the supervisor FIRST, otherwise it respawns the logger we just killed.
      "${SSH[@]}" "$NODE" "
        RDIR=\"\$HOME/.gpu_monitor\"; CSV=\"\$RDIR/thermals.csv\";
        for w in supervisor logger; do PF=\"\$RDIR/\$w.pid\";
          if [ -f \"\$PF\" ]; then P=\"\$(cat \"\$PF\")\";
            kill \"\$P\" 2>/dev/null; sleep 0.2; kill -9 \"\$P\" 2>/dev/null; rm -f \"\$PF\"; echo \"stopped \$w (pid \$P)\"; fi
        done;
        [ -f \"\$CSV\" ] && echo \"logged \$(( \$(wc -l < \"\$CSV\") - 1 )) rows -> \$CSV\" || true" ;;
    status)
      "${SSH[@]}" "$NODE" "
        RDIR=\"\$HOME/.gpu_monitor\"; CSV=\"\$RDIR/thermals.csv\"; sup=stopped; log=stopped;
        [ -f \"\$RDIR/supervisor.pid\" ] && kill -0 \"\$(cat \"\$RDIR/supervisor.pid\" 2>/dev/null)\" 2>/dev/null && sup=running;
        [ -f \"\$RDIR/logger.pid\" ] && kill -0 \"\$(cat \"\$RDIR/logger.pid\" 2>/dev/null)\" 2>/dev/null && log=running;
        echo \"supervisor=\$sup  logger=\$log\";
        if [ -f \"\$CSV\" ]; then echo \"rows=\$(( \$(wc -l < \"\$CSV\") - 1 ))  size=\$(du -h \"\$CSV\" | cut -f1)  path=\$CSV\"; echo \"latest:\"; tail -n 2 \"\$CSV\"; else echo \"(no CSV yet)\"; fi" ;;
    tail)
      "${SSH[@]}" "$NODE" "tail -n 15 \"\$HOME/.gpu_monitor/thermals.csv\" 2>/dev/null || echo '(no CSV yet)'" ;;
    fetch)
      local dest="gpu_thermals_${NODE//[^A-Za-z0-9._-]/_}.csv"
      if "${SCP[@]}" "$NODE:.gpu_monitor/thermals.csv" "$dest" 2>/dev/null; then
        echo "fetched -> $dest ($(( $(wc -l < "$dest") - 1 )) rows)"
      else echo "fetch failed — no CSV on $NODE yet? start it: $0 --remote-log start" >&2; return 1; fi ;;
    *) echo "unknown --remote-log action: $action (start|stop|status|tail|fetch)" >&2; return 2 ;;
  esac
}

set +e                          # the watch loop tolerates transient SSH failures

# --- remote-log dispatch (does its work over SSH, then exits) -----------------
if [[ -n "$REMOTE_LOG" ]]; then
  remote_log "$REMOTE_LOG"
  exit $?
fi

# --- one-shot mode -----------------------------------------------------------
if [[ "$ONCE" == 1 ]]; then
  out="$(frame)"
  if [[ -z "$out" ]]; then echo "gpu_monitor: no response from $NODE" >&2; exit 1; fi
  header
  printf '%s\n' "$out" | render_rows
  [[ -n "$LOGFILE" ]] && log_rows "$out"
  exit 0
fi

# --- live dashboard ----------------------------------------------------------
printf '\033[?25l' >&2          # hide cursor for a stable refresh
fails=0
while true; do
  out="$(frame)"
  printf '\033[H\033[J'         # cursor home + clear to end of screen
  if [[ -n "$out" ]]; then
    fails=0
    printf '%sGPU monitor%s  %s  %s%s%s\n\n' \
      "$BLD" "$RST" "$NODE" "$DIM" "$(date '+%Y-%m-%d %H:%M:%S')" "$RST"
    header
    printf '%s\n' "$out" | render_rows
    [[ -n "$LOGFILE" ]] && log_rows "$out"
    printf '\n%severy %ss · Ctrl-C to quit%s\n' "$DIM" "$INTERVAL" "$RST"
  else
    fails=$((fails+1))
    printf '%s⚠ no response from %s (retry %d)%s\n' "$YEL" "$NODE" "$fails" "$RST"
    printf '%schecking the tailnet path… is %s reachable? `ssh %s true`%s\n' \
      "$DIM" "$NODE" "$NODE" "$RST"
  fi
  sleep "$INTERVAL"
done
