# Effort 25 - RDNA2 (RX 9070 XT) Qwen 3.5 9B prefill recovery

> **Status:** active · seeded from Effort 17 findings + a node-specific cycle-0
> diagnosis (Mesa 26.0-devel DP4a regression). Cycle-0 fix already landed
> (commit on `rdna4-907xt-qwen9b-prefill-dp4a-fix`): prefill 6.8 -> ~171 tok/s.

Date: 2026-06-13

This is the **RDNA2-node** (consumer RX 9070 XT) sibling of Effort 17 (which
targets the Radeon AI PRO R9700 / RDNA1). The model, architecture family,
and primary metric are the same (Qwen 3.5 9B dense SSM+attention hybrid,
site-aligned decode-extended Long Coding Plan prefill). The node differs, and
the largest cycle-0 win was a **node-specific driver regression** rather than
a batched-prefill generalization.

## Target node (RDNA2 alias)

- Alias: `rdna2` (select with `ZINC_RDNA_NODE=rdna2`)
- GPU: **AMD Radeon RX 9070 XT** (`1002:7550`, GFX1201, Navi 48 = RDNA4, 64 CUs, 16 GB VRAM)
- Driver: **RADV, Mesa 26.0.0-devel** (the 9070 XT is new hardware that
  *requires* Mesa 26.x; older Mesa does not support GFX1201. This is NOT
  downgradeable to the 25.0.7 the R9700 node is pinned to.)
- Vulkan 1.4.330, `shaderIntegerDotProduct = true`, `cooperative_matrix = yes`
- Host: AMD Ryzen 9 3900X (Zen 2, 24 threads) — older than the R9700 node's Zen 5
- Managed model: `qwen35-9b-q4k-m` at
  `~/Library/Caches/zinc` layout → on node `/root/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf`
- Env keys (gitignored `.env`): `ZINC_RDNA2_HOST/PORT/USER/REMOTE_DIR/REMOTE_MODEL/SERVER_PORT/REMOTE_LOG`

```bash
# Always source .env and select the rdna2 node before any benchmark.
set -a; source .env; set +a
export ZINC_RDNA_NODE=rdna2
```

## Agent environment (opencode on the macOS host) — READ BEFORE ANY SHELL COMMAND

This effort is driven by `opencode run` (GLM-5.2). The agent process runs on
the **macOS host**, not on the RDNA2 node. That changes a few things the
generic cycle rules get wrong:

- **No `timeout` / `gtimeout` on macOS.** Do not wrap commands in `timeout`;
  it is `command not found` here and wastes a turn. Use the shell tool's own
  per-command `timeout` parameter instead (see next bullet).
- **opencode's shell tool kills any command at 30 s by default.** Builds,
  rsync, shader compiles, and `ssh ... zig build` all exceed that. When you
  invoke the shell tool for ANY of those, pass a larger per-command timeout
  (e.g. 300000 ms); otherwise the command is terminated mid-run and you get
  misleading partial output. Long one-shot diagnostics belong in a backgrounded
  `nohup ... &` that you then poll, not a single foreground shell call.
- **The model GGUF lives ONLY on the remote node**
  (`$ZINC_RDNA2_REMOTE_MODEL`). Do NOT run `strings`/`grep`/`cat`/`stat` on
  local `*.gguf` paths — there is no local copy; those commands return empty
  and waste turns. Inspect the model via `ssh` instead.
- **The controller does the authoritative remote sync + shader compile + build
  + 3-sample benchmark + coherence sweep AFTER you return.** So your in-cycle
  build obligation is just a sanity check: a LOCAL `zig build
  -Doptimize=ReleaseFast` (macOS skips GPU/shader work, ~10-30 s) to catch
  syntax/type errors is sufficient. You do NOT need to run the full remote
  build inline; if you do, background it and raise the shell-tool timeout.
- Editing is LOCAL (this worktree). The controller rsyncs to the node. Keep
  edits confined to `src/`, `build.zig`, shaders, and docs; the controller
  only reverts those paths.

## llama.cpp hardware ceiling on THIS node (measured 2026-06-13)

llama.cpp is installed at `/root/llama.cpp/build-vulkan/bin/`. Build 9a532ae,
Vulkan, RADV. On the 9070 XT it reports `matrix cores: none`, `int dot: 0`
(llama-side), but still achieves:

| scenario | prompt toks | llama.cpp prefill | llama.cpp decode |
|---|---:|---:|---:|
| pp512 / tg256 | 512 / 256 | **973.49 tok/s** | **21.74 tok/s** |

Two facts that frame the whole effort:

1. **Decode is not the problem here.** ZINC decode (~39.6 tok/s) already beats
   llama.cpp (21.7 tok/s) by ~75% on this node — a wider margin than on the
   R9700. The low absolute decode number is a card/driver characteristic that
   hits both runtimes equally; it is NOT a ZINC regression. **Do not optimize
   decode.**
2. **Prefill is the prize.** llama.cpp reaches 973 tok/s, so the hardware can
   prefill fast. ZINC's gap to close is prefill-only.

## Cycle-0 result (already landed)

**Root cause of the original ~6.8 tok/s prefill:** the int8 DP4a dense
gate+up+SwiGLU shader (`mul_mm_q4k_gate_up_swiglu_full_dp4a`) runs ~25x
slower on Mesa 26.0.0-devel + cooperative-matrix than the non-DP4a branchless
Q4_K path. A 64-token prefill spent 97% of its time (9095 ms) in that one
shader; the branchless path does the same gateup matmul in ~0 ms with
identical output. The same DP4a shader is fast on RDNA4 *without* cooperative
matrix (R9700 / Mesa 25.0.7).

**Fix (committed, `qwenDenseFfnDp4aEnabled` in `src/compute/forward.zig`):**
default the 9B to the branchless `mul_mm_q4k_gate_up_swiglu_full` path when
`amd_rdna4` + `cooperative_matrix` are both exposed. Scoped to the 9B; the
27B and other models are untouched. `ZINC_QWEN_DENSE_FFN_DP4A=1` forces DP4a
back on for re-testing.

**Measured after the fix (9070 XT, Qwen3.5-9B Q4_K_M, default no env):**

| shape | before | after |
|---|---:|---:|
| prefill 64 t (decode-extended / primary) | 6.8 tok/s | **~171 tok/s** |
| prefill 326 t | 5.8 tok/s | **~153 tok/s** |
| prefill 781 t | — | ~142 tok/s |
| decode 256 t | 39.6 tok/s | 39.6 tok/s (unchanged) |

RDNA1 (R9700, coopmat=no) is **unaffected** — DP4a stays on, 421 tok/s.

## Current phase budget (64-token Long Coding Plan, after cycle-0 fix)

From `ZINC_PREFILL_PROFILE=1` (totals over the prefill, ~373 ms):

```
GPU phases totals: attn=40.5  ssm=163.5  dense_ffn=142.9  tail=2.3  embed=0.0
SSM subphases:     proj=73.3 qkv=59.1 z=22.4 conv=37.0 delta=26.5 gnorm=37.4 out=36.7 ...
dense_ffn:          gateup=68.0 (branchless, fast)  down_matmul=60.1 (generic)  residual_acc=74.5
```

The two big buckets are now **SSM (~163 ms)** and **dense_ffn (~143 ms)**,
roughly tied. Within them the largest single sub-phases are:

- `dense_ffn residual_acc` = ~74 ms — a standalone `dispatchScaleAcc`
  (`hidden += 1.0 * down_out`) launched as its own dispatch + barrier, per
  layer. For the bytes touched this is dispatch/launch-overhead-bound.
- SSM `proj` (~73 ms) + `qkv` (~59 ms) — the SSM-layer wqkv/z projections on
  the generic batched matmul path.
- `dense_ffn down_matmul` = ~60 ms (generic; DP4a down is off via the cycle-0
  gate, which is correct here).

## Measurement contract

The controller benchmark is the public Long Coding Plan prompt in **raw** mode
(same as Effort 17's decode-extended scenario):

```text
Write an implementation plan for adding a stable benchmark preset to a
local LLM CLI. Include the command shape, warmup policy, metrics to
collect, failure handling, llama.cpp comparison, and how the site should
display prefill, decode, latency, and overall prompt+decode throughput.

Plan:
1.
```

- Model: `qwen35-9b-q4k-m` (managed cache; do NOT copy ad-hoc GGUFs)
- Prompt mode: raw. Primary metric: **ZINC prefill tok/s**.
- Generation cap in loop: 8 tokens (prefill is the metric).
- Published comparison cap: 256 tokens (same prompt text).
- **llama.cpp target on this node: 973 tok/s prefill** (pp512). ZINC is at
  ~171, so the remaining gap is ~5.7x — substantially structural
  (batched-GEMM / dispatch-overhead work), not a single toggle.

A useful keep must:

1. Improve primary prefill tok/s over the best accepted checkpoint.
2. Preserve coherent output (the Long Coding Plan answer begins
   "Command shape: `benchmark --preset <name>` ...").
3. Preserve the decode lead over llama.cpp (do not regress decode).
4. Not regress 174-token / 322-token prefill while helping 64-token.

## Candidate next levers (ordered by expected value)

### Track A - Eliminate the standalone dense-FFN residual add (~74 ms)

`dispatchScaleAcc(hidden += down_out)` runs as its own dispatch + barrier per
layer (`src/compute/forward.zig`, the `dense_ffn_residual_acc` phase). For the
bytes touched it is launch-overhead-bound. Options:

1. Fuse the residual into the dense-down projection epilogue (a down+acc
   shader variant that writes `hidden[i] += result[i]`). Removes 32 dispatches
   + 32 barriers. Highest payoff, real shader work — add the `.comp` to
   `build.zig` shader install and prove a clean remote build loads it.
2. Make `scale_accumulate.comp` process more elements per workgroup (stride
   loop) so far fewer WGs are dispatched. Smaller change, shared by all
   `dispatchScaleAcc` callers — must measure prefill AND decode so the 9B
   decode path is not regressed.

Either way: measure flag OFF/ON in the same cycle and confirm final output is
bit-for-bit the same text.

### Track B - SSM projection batching / GEMM tuning (~132 ms combined)

SSM `proj` (~73 ms) + `qkv` (~59 ms) are the largest single sub-phases and run
on the generic batched matmul. This is the structural GEMM gap to llama.cpp.
Treat as multi-cycle: collect paired `RADV_DEBUG=shaderstats` for the active
SSM projection shader before editing, and prefer layer-major batching of
wqkv/z (token-order recurrence preserved) over single-shader tile tweaks.

### Track C - Generic dense-down (~60 ms)

`dense_ffn down_matmul` is on the generic path (DP4a down is intentionally off
here). If Track A's down+acc fusion lands, this bucket is partially absorbed.
Otherwise this needs the same shaderstats-first approach as Track B.

## Known traps / do-not-repeat (node-specific)

- **Do not re-enable DP4a dense FFN for the 9B on this node.** It is the
  cycle-0 regression (`ZINC_QWEN_DENSE_FFN_DP4A=1` reproduces 6.8 tok/s).
  Only revisit if Mesa 26.0 *stable* ships and a paired measurement shows the
  DP4a shader is fast again.
- **Do not gate the SSM DP4a functions expecting a win.** Tested cycle-0:
  adding the same coopmat+rdna4 guard to `qwenDenseSsmOutDp4aEnabled` /
  `qwenDenseSsmProjDp4aEnabled` / `qwenDenseProjectionDp4aEnabled` moved SSM
  165.8 -> 163.5 ms (noise). The 9B SSM projections were NOT on the DP4a path.
- **Do not sweep `ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS`.** Forcing prefix
  layers 3/8/16/31 changed nothing (the layer-major prefix path already
  engages at prefix=3; the cost is inside it, not in layer count).
- **Do not optimize decode.** It already beats llama.cpp by ~75% on this node.
- **Do not trust the "Modeled decode bandwidth" line** — its `576 GB/s
  theoretical` is hardcoded for the R9700 and the MB/token model is inflated
  on this node. Use the wall-clock `tok/s` lines.
- **Do not run on RDNA1 concurrently.** An autonomous `zinc_rt_autopilot` loop
  owns RDNA1 and overwrites `zig-out/bin/zinc` with its ZINC_RT build. This
  effort targets RDNA2 only (`ZINC_RDNA_NODE=rdna2`).
- Do not add a `.comp` without adding it to `build.zig` shader install and
  proving a clean remote build actually loads it (macOS cannot compile
  shaders — glslc runs on the node).
- Do not keep a flag-gated optimization without paired OFF/ON measurement in
  the same cycle.

## First-cycle requirement for any kernel change

Before editing a shader or dispatch, capture a fresh `ZINC_PREFILL_PROFILE=1`
run on RDNA2 and confirm which sub-phase you are moving. If the phase labels
are missing for a bucket you intend to attack, add only the missing labels as
a foundation step and re-measure before any "optimization" commit.

```bash
set -a; source .env; set +a
export ZINC_RDNA_NODE=rdna2
# sync, build, profile
rsync -az --delete --exclude '.git' --exclude '.env' --exclude '.zig-cache' \
  --exclude 'zig-out' --exclude 'node_modules' --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_RDNA2_PORT" . $ZINC_RDNA2_USER@$ZINC_RDNA2_HOST:/root/zinc/
ssh -p $ZINC_RDNA2_PORT $ZINC_RDNA2_USER@$ZINC_RDNA2_HOST \
  'cd /root/zinc && zig build -Doptimize=ReleaseFast && \
   RADV_PERFTEST=coop_matrix ZINC_PREFILL_PROFILE=1 ./zig-out/bin/zinc \
     --model-id qwen35-9b-q4k-m --prompt "$(cat /tmp/lcp.txt)"'
```
