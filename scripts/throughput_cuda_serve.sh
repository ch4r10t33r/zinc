#!/usr/bin/env bash
# Effort 28 increment 3 (3c) — CUDA serving AGGREGATE-THROUGHPUT gate (headline).
#
# Drives the running `zinc` CUDA server (one GPU worker thread + per-request SSE,
# src/server/cuda_serve.zig + runCudaServe) with B = 1,2,4,8 CONCURRENT clients
# and measures aggregate DECODE throughput, to prove request-batching scales
# aggregate tok/s past the single-stream (serialized) baseline. Decode is
# launch/bandwidth-bound (~7-12% util single-stream), so batching B sequences
# reads the same weights ONCE and amortizes the same launches across B tokens →
# expect ~linear scaling toward compute-bound.
#
# Headline metric = SERVER-SIDE pure-decode tok/s from the /stats counters (added
# this cycle): the worker times each batched `decodeBatch` step and sums batch
# occupancy, so over a phase  Δdecode_tokens / Δdecode_wall_s = aggregate decode
# tok/s and Δdecode_tokens / Δdecode_steps = mean batch occupancy (→ B when the
# batch fills). This excludes prefill/SSE/idle and isn't perturbed by external
# curl-start jitter. We ALSO print the external end-to-end aggregate (B*NG/wall)
# as a sanity cross-check. GATE: median srv decode tok/s at B>1 strictly exceeds
# the B=1 baseline (real batching win), and mean occupancy ≈ B (batch fills).
#
# Run ON the box from the repo root. 5090-pinned. The server reloads gemma-31b
# (~17GB, ~2min) ONCE then all phases hit it. Budget-only EOS → every request
# runs exactly NG decode tokens (clean, equal-length measurement).
set -u
GPU=${ZINC_GPU:-GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350}
ZIG=${ZIG:-$HOME/zig-0.15.2/zig}
MD=${ZINC_MODELS:-$HOME/workspace/models}
MODEL=${ZINC_SERVE_MODEL:-$MD/gemma-4-31B-it-Q4_K_M.gguf}
PORT=${ZINC_SERVE_PORT:-8190}
# Net-fenced box: reach the local server via 127.99.x (plain 127.0.0.1 is dropped).
HOST=${ZINC_SERVE_HOST:-127.99.0.1}
NG=${ZINC_TP_NG:-96}
NSLOTS=${ZINC_TP_NSLOTS:-8}
CTX=${ZINC_TP_CTX:-1024}
ROUNDS=${ZINC_TP_ROUNDS:-3}
BLIST=${ZINC_TP_BLIST:-"1 2 4 8"}
# Short fixed raw-token-id prompt (debug-ids mode) so decode dominates + every
# client is identical length.
PROMPT=${ZINC_TP_PROMPT:-"651,2134,573,1496"}
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-/tmp/e28gc}
T=/tmp/zinc_tp_gate; mkdir -p "$T"; rm -f "$T"/*.out "$T"/*.log "$T"/results.txt 2>/dev/null

echo "=== build zinc (server binary) ==="
$ZIG build -Dbackend=cuda -Dshaders=false >"$T/build.log" 2>&1 || { echo "ZINC BUILD FAIL"; grep -E 'error:' "$T/build.log" | head; exit 1; }
ZINC=$(ls -t zig-out/bin/zinc 2>/dev/null | head -1)
[ -x "$ZINC" ] || { echo "no zinc binary"; exit 1; }
echo "zinc md5: $(md5sum "$ZINC" | cut -d' ' -f1)"

echo "=== start server (port $PORT, slots $NSLOTS, ctx $CTX, NG $NG, budget-only EOS) ==="
CUDA_VISIBLE_DEVICES=$GPU ZINC_GPU=$GPU ZINC_SERVE_DEBUG_IDS=1 ZINC_SCHED_EOS=4294967295 \
  "$ZINC" -m "$MODEL" -p "$PORT" -c "$CTX" --parallel "$NSLOTS" -n "$NG" \
  >"$T/server.log" 2>&1 &
SRV=$!
trap '[ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT
ready=0
for _ in $(seq 1 360); do
  if curl -s --max-time 3 "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || { echo "server died during startup:"; tail -20 "$T/server.log"; exit 1; }
  sleep 1
done
[ "$ready" = 1 ] || { echo "server not ready in time:"; tail -20 "$T/server.log"; exit 1; }
echo "server ready (pid $SRV)"

stat()    { curl -s --max-time 5 "http://$HOST:$PORT/stats"; }
gen_one() { curl -N -s --max-time 300 -X POST "http://$HOST:$PORT/v1/completions" \
              -H 'Content-Type: application/json' \
              -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":$NG}" >/dev/null; }

echo "=== warmup ==="
gen_one; gen_one
echo "warmup /stats: $(stat)"

phase() { # $1=B  -> appends "B srv_tps occ ext_tps" to results.txt + prints a line
  local B=$1
  local s0; s0=$(stat)
  local t0; t0=$(date +%s.%N)
  local pids=()
  local i
  for ((i=0; i<B; i++)); do gen_one & pids+=($!); done
  wait "${pids[@]}"
  local t1; t1=$(date +%s.%N)
  local s1; s1=$(stat)
  python3 - "$s0" "$s1" "$t0" "$t1" "$B" "$NG" "$T/results.txt" <<'PY'
import sys, json
s0=json.loads(sys.argv[1]); s1=json.loads(sys.argv[2])
t0=float(sys.argv[3]); t1=float(sys.argv[4]); B=int(sys.argv[5]); NG=int(sys.argv[6]); rf=sys.argv[7]
dtok=s1['decode_tokens']-s0['decode_tokens']
dwall=(s1['decode_wall_ns']-s0['decode_wall_ns'])/1e9
dstep=s1['decode_steps']-s0['decode_steps']
ext=t1-t0
srv=dtok/dwall if dwall>0 else 0.0
occ=dtok/dstep if dstep>0 else 0.0
extt=(B*NG)/ext if ext>0 else 0.0
print(f"  B={B:>2}  srv_decode_tok/s={srv:8.1f}  occ={occ:5.2f}  ext_agg_tok/s={extt:8.1f}  (dtok={dtok} dsteps={dstep} dwall={dwall:.3f}s ext={ext:.3f}s)")
open(rf,'a').write(f"{B} {srv:.3f} {occ:.3f} {extt:.3f}\n")
PY
}

for r in $(seq 1 "$ROUNDS"); do
  echo "=== round $r ==="
  for B in $BLIST; do phase "$B"; done
done

echo "=== SUMMARY (median over $ROUNDS rounds) ==="
python3 - "$T/results.txt" "$BLIST" <<'PY'
import sys, statistics as st
rows={}
for ln in open(sys.argv[1]):
    b,srv,occ,ext=ln.split()
    rows.setdefault(int(b),[]).append((float(srv),float(occ),float(ext)))
blist=[int(x) for x in sys.argv[2].split()]
base=None; ok=True; occ_ok=True
print(f"  {'B':>3} {'srv_tok/s(med)':>15} {'occ(med)':>9} {'ext_tok/s(med)':>15} {'scale_vs_B1':>12}")
for b in blist:
    vs=rows.get(b,[])
    if not vs: continue
    srv=st.median(v[0] for v in vs); occ=st.median(v[1] for v in vs); ext=st.median(v[2] for v in vs)
    if base is None: base=srv
    scale=srv/base if base>0 else 0
    print(f"  {b:>3} {srv:>15.1f} {occ:>9.2f} {ext:>15.1f} {scale:>11.2f}x")
    if b>1 and srv<=base: ok=False
    if b>1 and occ < b*0.6: occ_ok=False  # batch should fill to ≥60% of B
print()
print(f"TP_GATE_SCALING:{'PASS' if ok else 'FAIL'} (B>1 srv decode tok/s > B=1 baseline)")
print(f"TP_GATE_OCCUPANCY:{'PASS' if occ_ok else 'FAIL'} (mean batch occupancy ≥ 0.6*B)")
print(f"TP_GATE:{'PASS' if (ok and occ_ok) else 'FAIL'}")
PY
