# Benchmarking ZINC

How to reproduce ZINC's published numbers, run an ad-hoc llama.cpp baseline, and use the per-kernel hot-bench. Apple Silicon notes at the bottom.

## Published benchmarks (the canonical numbers)

The site at [zolotukhin.ai/zinc/benchmarks](https://zolotukhin.ai/zinc/benchmarks) is generated from `tools/performance_suite.mjs`. Every published run pairs ZINC against llama.cpp on the same hardware, same model file, and the same four-scenario matrix (`core`, `context-medium`, `context-long`, `decode-extended`). Those scenarios are real workload prompts: quick chat, coding review, incident-context QA, and longer coding-plan generation. The full run is what gets pushed to `site/src/data/zinc-performance.json` and Cloudflare Pages picks it up on every push to `main`.

Remote machine details are intentionally supplied by `.env` or CLI flags. Do not commit public hostnames, public IP addresses, private SSH ports, or one-off benchmark node aliases into this document or into published site data.

```bash
# Metal target (Apple Silicon, runs locally)
bun tools/performance_suite.mjs --target metal

# RDNA4 target (remote node from .env). Published RDNA runs use the
# Vulkan backend; ZINC_RT is a separate bring-up runtime and must be
# requested explicitly with --rdna-backend zinc_rt.
bun tools/performance_suite.mjs --target rdna --rdna-sync --rdna-build --rdna-start-llama --rdna-backend vulkan

# Pick a specific RDNA node when .env defines ZINC_RDNA1_* and ZINC_RDNA2_*.
bun tools/performance_suite.mjs --target rdna --rdna-node rdna2 --rdna-sync --rdna-build --rdna-backend vulkan

# Intel Arc target (separate remote node from .env)
bun tools/performance_suite.mjs --target intel --intel-sync --intel-build --intel-start-llama

# Everything in one run
bun tools/performance_suite.mjs --target all --rdna-sync --rdna-build --rdna-start-llama
```

Defaults are 1 warmup + 3 measured runs per scenario; pass `--runs N --warmup M` to override. Pass `--no-site-write` to leave `site/src/data/zinc-performance.json` alone (the run still emits a `/tmp` artifact via `--output`). The suite stamps `provenance.zinc.version` from `git describe --dirty`; when `--rdna-sync` is used, provenance comes from the local tree that was synced, not from the remote workdir's stale `.git` metadata. Commit (or stash) before publishing — a `-dirty` tag means reviewers cannot reproduce the exact tree.

RDNA suite runs sync into `/root/zinc-bench` by default. Keep that checkout isolated from optimization loops and ad-hoc experiments; sharing `/root/zinc` can overwrite `zig-out/bin/zinc` mid-suite and invalidate later model rows. The suite verifies `zinc --version` reports the requested backend before RDNA measurements, and before every ZINC scenario, so a stale or overwritten binary fails loudly instead of publishing mixed Vulkan/ZINC_RT data.

`bun tools/performance_suite.mjs --help` lists every flag (target subset, model filtering, baseline binary overrides, remote workdir / libc / model-root overrides, etc.).

Current RDNA publish runs cover Qwen 3.5 9B Q4_K_M, Qwen 3.6 27B Q4_K_M, and Qwen 3.6 35B-A3B UD Q4_K_XL. The small-Qwen row is `Qwen3.5-9B-Q4_K_M.gguf`, not the older Qwen 3 8B GGUF.

## Ad-hoc llama.cpp baseline on the RDNA4 node

Use this when you want a one-off llama.cpp number outside the perf suite — for example, to validate a Mesa or driver change before doing a full publish run.

**Model on the RDNA4 test node**: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.4 GiB, MoE 35B/3B active).

The reference setup on the RDNA4 test node:

### Test node setup (critical for reproducing the baseline)

```bash
# 1. Mesa must be 25.0.7 (25.2.8 causes ~14% RADV regression)
dpkg -l mesa-vulkan-drivers  # should show 25.0.7-0ubuntu0.24.04.2
# Pinned in /etc/apt/preferences.d/mesa-pin to prevent auto-upgrade

# 2. GECC disabled (amdgpu.ras_enable=0 in /etc/default/grub)
cat /sys/module/amdgpu/parameters/ras_enable  # should show 0

# 3. RADV_PERFTEST=coop_matrix set in llama-server.service
#    Without this, cooperative matrix is disabled → scalar fallback

# 4. llama.cpp must be recorded from the benchmark binary:
#    /root/llama.cpp/build/bin/llama-server --version
#    The published artifact records the exact version and commit under
#    provenance.llama_cpp. Rebuild and republish when changing it.

# 5. Server flags (in /etc/systemd/system/llama-server.service):
#    -ngl 99 --device Vulkan0 --parallel 4 -c 32768
#    -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --mlock --flash-attn on
```

### Measure llama.cpp

```bash
source .env

# Start server (if not running)
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "systemctl start llama-server && sleep 15"

# Warmup + 3 benchmark runs via OpenAI API
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST '
  curl -s http://localhost:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" > /dev/null
  for i in 1 2 3; do
    out=$(curl -s http://localhost:8088/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":256,\"stream\":false}" \
    )
    gen=$(printf "%s" "$out" | jq -r ".timings.predicted_per_second // 0")
    prompt=$(printf "%s" "$out" | jq -r ".timings.prompt_per_second // 0")
    printf "Run %d: gen %s tok/s | prompt %s tok/s\n" "$i" "$gen" "$prompt"
  done
'
# Treat this as a one-off sanity check. The canonical values are the
# medians in site/src/data/zinc-performance.json and on /zinc/benchmarks.
```

## Measure ZINC (CLI)

```bash
source .env

# Sync source to test node
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

# Build and run
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is'"

# Key output lines:
#   info(forward): Prefill complete: N tokens in X ms (Y tok/s)
#   info(forward): Generated N tokens in X ms — Y tok/s (Z ms/tok)
```

## Measure ZINC (HTTP)

Use the HTTP benchmarks for end-to-end API latency, queueing behavior, or to compare the chat endpoint against the raw completions path.

Caveats:

1. Bench a clean node. Other `zinc`, `llama-server`, and `llama-cli` processes on the RDNA4 host contaminate latency and throughput.
2. `POST /v1/chat/completions` is an end-user latency benchmark, not a pure decode-throughput benchmark. The chat route applies templates and stop handling, so many prompts stop after only a handful of tokens.
3. Use `POST /v1/completions` for sustained HTTP decode throughput.
4. ZINC server generation is still serialized. With `concurrency > 1`, aggregate throughput stays roughly flat while per-request latency grows because requests queue behind one active decode.

Clean-server setup:

```bash
source .env

# 1. Stop stale GPU users on the test node.
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  pkill -f 'zig-out/bin/zinc' || true; \
  pkill -f 'llama-server' || true; \
  pkill -f 'llama-cli' || true"

# 2. Sync, build, and restart one clean ZINC server on :9090.
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && zig build -Doptimize=ReleaseFast && \
  nohup env RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --port 9090 >/tmp/zinc_9090.log 2>&1 < /dev/null &"

# 3. Wait for health.
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  until curl -fsS http://127.0.0.1:9090/health >/dev/null; do sleep 1; done; \
  curl -sS http://127.0.0.1:9090/health"
```

Chat-endpoint latency matrix:

```bash
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  /root/.bun/bin/bun tools/benchmark_api.mjs \
    --base http://127.0.0.1:9090/v1 \
    --mode chat \
    --output /tmp/zinc_api_chat_benchmark.json"
```

Raw sustained throughput:

```bash
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  /root/.bun/bin/bun tools/benchmark_api.mjs \
    --base http://127.0.0.1:9090/v1 \
    --mode raw \
    --output /tmp/zinc_api_raw_benchmark.json"
```

## Latest single-stream reference

Current numbers for every (target, model, scenario) combination live on the dashboard: [zolotukhin.ai/zinc/benchmarks](https://zolotukhin.ai/zinc/benchmarks). The dashboard is regenerated by `tools/performance_suite.mjs` on every publish run and reports prefill, decode, and a combined prompt-plus-decode "overall" ratio against the same-machine llama.cpp baseline.

## Hot-bench: per-kernel microbenchmarks

Use the dedicated microbenchmark when whole-model decode says "MoE", "shared expert", or `ssm_delta_net` is hot and you need exact per-kernel numbers plus `RADV_DEBUG=shaderstats` feedback.

Caveat: hot-bench rotates across multiple buffer sets to reduce cache-hot bias, but treat its GB/s as a kernel-comparison signal, not a final whole-model DRAM bandwidth number.

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --iterations 200 --warmup 25"
```

Single case + shader stats:

```bash
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  RADV_DEBUG=shaderstats zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --case ssm_delta"
```

Available cases: `q8_router`, `q8_shared_gate_up`, `q8_shared_down`, `q8_ssm_out`, `ssm_delta`.

## Troubleshooting

If llama.cpp baseline drops below ~100 tok/s, check in order:

1. **Mesa version** — `dpkg -l mesa-vulkan-drivers` must show 25.0.7 (not 25.2.8).
2. **GECC** — `cat /sys/module/amdgpu/parameters/ras_enable` must show 0.
3. **coop_matrix** — server log must show `matrix cores: KHR_coopmat`.
4. **Reboot** — Mesa/driver changes need a reboot to take full effect.
5. **DPM stuck low** — long-running GPU processes can hold the R9700 in low-DPM. A reboot restores peak clocks. (Effort 10 baselines were corrupted by this for ~22 days.)
6. **Dirty benchmark node** — stop stray `zinc` / `llama-*` processes before comparing runs.
7. **Wrong endpoint for the question** — `/v1/chat/completions` for chat latency and queueing, `/v1/completions` for sustained HTTP decode throughput.
8. **Early chat stops** — if chat completions are ending after a handful of tokens, change the prompt or switch to `/v1/completions`.
