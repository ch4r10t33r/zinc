# Development Guide

Everything you need to build, test, debug, and contribute to ZINC.

If you just want to run inference, see [Getting Started](/zinc/docs/getting-started/). This page is for people who want to modify the engine, add features, fix bugs, or understand the internals.

## Prerequisites

| Tool | Version | What it does |
|------|---------|-------------|
| **Zig** | 0.15.2+ | Host compiler for all Zig source |
| **Bun** | Latest | TypeScript test runner, site builder, API benchmarks |
| **glslc** | shaderc 2023.8 | Compiles GLSL compute shaders to SPIR-V (Linux/AMD only) |
| **Vulkan SDK** | 1.3+ | Runtime for AMD GPU dispatch (Linux only) |
| **Xcode CLI Tools** | Latest | Metal compiler and frameworks (macOS only) |

On macOS (Apple Silicon):

```bash
brew install zig bun
xcode-select --install
```

On Linux (AMD GPU):

```bash
# Zig: https://ziglang.org/download/
# Bun: https://bun.sh
sudo apt install libvulkan-dev glslc
```

## Build

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc

# Debug build (fast compile, slow runtime)
zig build

# Release build (slow compile, fast runtime — use for benchmarking)
zig build -Doptimize=ReleaseFast
```

The binary lands at `./zig-out/bin/zinc`. Shaders are compiled from `src/shaders/*.comp` to SPIR-V during the build.

## Test

```bash
# Full test suite (Zig unit tests + Bun TypeScript tests)
zig build test

# Just the TypeScript tests
bun test

# With integration smoke tests (requires a running server + model files)
ZINC_API_BASE_URL=http://localhost:8080/v1 zig build test -Dfull-tests=true
```

The test suite covers:
- Zig unit tests across all modules (tokenizer, routes, catalog, forward pass, etc.)
- Chat UI rendering, markdown, thinking logic, repetition detection
- API benchmark tool validation
- Site build and Zig API documentation generation
- Optional: OpenAI SDK compatibility, Qwen model smoke tests

## Debug Flags

| Flag | What it does |
|------|-------------|
| `--debug` | Enable verbose logging (same as `ZINC_DEBUG=1`) |
| `--profile` | Enable per-dispatch GPU profiling (Vulkan only) |
| `ZINC_DEBUG=1` | Environment variable alternative to `--debug` |
| `ZINC_PREFILL_PROFILE=1` | Per-phase prefill timing (dense FFN, SSM, attention, projections) |
| `ZINC_BATCHED_PREFILL=0` | Disable RDNA batched prefill; useful for isolating regressions |
| `ZINC_FA_SPLIT_K=<n>` | Override the scalar flash-attention split-K count |
| `ZINC_TOOL_CALLING=0` | Disable OpenAI-compatible tool calling on `/v1/chat/completions` |
| `RADV_PERFTEST=coop_matrix` | Enable cooperative matrix on RDNA4 (recommended) |
| `RADV_DEBUG=shaders` | Dump compiled Vulkan shaders |
| `RADV_DEBUG=shaderstats` | Show VGPR/occupancy/spill stats |

For the full set of `ZINC_*` knobs (fusion gates, per-architecture experiments), grep the codebase: `rg 'getEnvVar.*ZINC_' src/`.

Example debug run:

```bash
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "Hello" -n 64
```

## Project Structure

```
src/
├── main.zig                     # CLI entry, arg parsing, server startup, chat subcommand
├── compute/
│   ├── forward.zig              # Vulkan inference engine — prefill + decode loop
│   ├── forward_metal.zig        # Metal inference engine — prefill + decode loop
│   ├── dmmv.zig                 # DMMV dispatch (quantized matmul-vec)
│   ├── elementwise.zig          # Fused elementwise ops (RMS norm, SwiGLU, etc.)
│   ├── attention.zig            # Flash attention dispatch
│   ├── argmax.zig               # Argmax / sampling dispatch
│   └── graph.zig                # Decode graph builder and exporter
├── model/
│   ├── tokenizer.zig            # BPE tokenizer, chat templates, thinking toggle
│   ├── catalog.zig              # Managed model catalog with thinking_stable flag
│   ├── gguf.zig                 # GGUF file parser and tensor metadata
│   ├── loader.zig               # Model loader (Vulkan — mmap + DMA to VRAM)
│   ├── loader_metal.zig         # Model loader (Metal — zero-copy mmap)
│   ├── architecture.zig         # Architecture detection (Qwen, MoE, SSM, etc.)
│   ├── config.zig               # Model configuration from GGUF metadata
│   └── managed.zig              # Managed model download, install, activation
├── server/
│   ├── routes.zig               # OpenAI-compatible API, streaming, stop detection
│   ├── chat.html                # Built-in chat UI (embedded at compile time)
│   ├── http.zig                 # HTTP server and connection handling
│   ├── http.zig                 # HTTP server base (request parsing, response writing)
│   ├── routes.zig               # OpenAI-compatible request handlers and streaming
│   ├── model_manager.zig        # Hot model switching and catalog view
│   ├── model_manager_metal.zig  # Metal-specific model manager extensions
│   ├── model_manager_runtime.zig # Runtime abstraction for model manager
│   ├── runtime.zig              # Backend runtime dispatch (Vulkan vs Metal)
│   ├── tool_format.zig          # OpenAI tool-calling output formatting
│   └── chat.html                # Static chat UI served from /chat
├── vulkan/
│   ├── instance.zig             # Vulkan instance and device init
│   ├── pipeline.zig             # Compute pipeline and shader loading
│   ├── buffer.zig               # GPU buffer allocation and transfers
│   ├── command.zig              # Command buffer recording and submission
│   ├── gpu_detect.zig           # GPU vendor/capability detection
│   └── vk.zig                   # Vulkan C API bindings
├── metal/
│   ├── device.zig               # Metal device init and capability query
│   ├── pipeline.zig             # MSL compute pipeline compilation
│   ├── buffer.zig               # Metal buffer management
│   ├── command.zig              # Command buffer and encoder
│   ├── c.zig                    # Metal C API bindings
│   ├── shim.h                   # Objective-C shim header
│   └── shim.m                   # Objective-C shim implementation
├── gpu/
│   └── interface.zig            # Backend abstraction (Vulkan vs Metal)
├── scheduler/
│   ├── scheduler.zig            # Request scheduling
│   ├── kv_cache.zig             # KV cache management
│   └── request.zig              # Request state
├── diagnostics.zig              # --check system diagnostics (Vulkan)
├── diagnostics_metal.zig        # --check system diagnostics (Metal)
├── regression_tests.zig         # Regression test fixtures
├── shaders/
│   ├── *.comp                   # GLSL compute shaders (Vulkan/SPIR-V)
│   └── metal/*.metal            # MSL compute shaders (Apple Silicon)
site/                            # zolotukhin.ai Astro site
docs/                            # Technical documentation (published to site)
tools/                           # API benchmark, standalone utilities
specs/                           # Feature specifications and plans
benchmarks/                      # GPU microbenchmarks (bandwidth, dispatch, Metal)
scripts/                         # Deployment scripts
tests/                           # TypeScript test files
loops/                           # Self-improving optimization loop
```

## Graph Export

ZINC can export the decode computation graph for debugging and visualization:

```bash
# JSON report with node/edge lists, op counts, and critical path
./zig-out/bin/zinc -m model.gguf --graph-report graph.json

# Graphviz DOT for visualization
./zig-out/bin/zinc -m model.gguf --graph-dot graph.dot
dot -Tsvg graph.dot -o graph.svg

# Inspect with jq
cat graph.json | jq '.summary'
cat graph.json | jq '.nodes[] | select(.op_type == "dmmv") | {name, quant, rows, cols}'
```

These flags are available via `--help-all`.

## Benchmarking

### CLI decode throughput

```bash
# Single-stream decode (the primary metric)
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "The capital of France is" -n 128
```

### API benchmarks

```bash
# Chat endpoint matrix (short/medium/long prompts, concurrency 1/2/4)
bun tools/benchmark_api.mjs --base http://localhost:8080 --mode chat

# Raw completions throughput
bun tools/benchmark_api.mjs --base http://localhost:8080 --mode raw

# Serving concurrency gate (expected to fail until continuous batching lands)
bun tools/benchmark_api.mjs \
  --base http://localhost:8080 \
  --mode concurrency \
  --min-c4-aggregate-scale 2.5 \
  --max-c4-p95-latency-multiplier 2.0
```

The concurrency gate compares each `concurrency=4` scenario against its
matching `concurrency=1` baseline. A serialized server typically shows
aggregate completion throughput near `1x` and p95 latency near `4x`; continuous
batching should raise aggregate throughput materially without linear p95 growth.

### Public performance output

```bash
# Generate the benchmark artifact that powers /zinc/benchmarks.
# Legacy "both" runs RDNA + Metal; use "all" to include the Intel Arc node.
bun tools/performance_suite.mjs --target both --rdna-sync --rdna-build --rdna-start-llama --output /tmp/zinc-performance.json
bun tools/performance_suite.mjs --target intel --output /tmp/zinc-intel-performance.json
```

The generated target entries include the exact ZINC git version/commit for the machine that produced each target and the llama.cpp binary version/commit used for the baseline on that target. The public performance page renders the same provenance so benchmark rows are tied to a concrete source tree and baseline build.

Use the RDNA suite's server-vs-server path for headline comparisons. One-shot CLI runs are diagnostics; do not compare them against a warmed baseline server when deciding whether a target is beaten.

### Hot decode kernel microbenchmarks

```bash
# Measure individual GPU kernel performance
zig build hot-bench -Doptimize=ReleaseFast
./zig-out/bin/zinc-hot-bench --shader-dir zig-out/share/zinc/shaders
```

For detailed tuning guidance, see [RDNA4 Tuning Guide](/zinc/docs/rdna4-tuning/), the [AMD GPU Reference](/zinc/docs/amd-gpu-reference/), and the [Intel GPU Reference](/zinc/docs/intel-gpu-reference/).

## RDNA4 Test Node

For AMD GPU testing, the project uses remote RDNA4 nodes. Environment setup:

```bash
# .env file in repo root
# Optional selector for machines with more than one RDNA node.
# The selector maps rdna1 -> ZINC_RDNA1_* and rdna2 -> ZINC_RDNA2_*.
ZINC_RDNA_NODE=rdna1

ZINC_RDNA1_HOST=<ip-or-hostname>
ZINC_RDNA1_PORT=<ssh-port>
ZINC_RDNA1_USER=root

ZINC_RDNA2_HOST=<ip-or-hostname>
ZINC_RDNA2_PORT=<ssh-port>
ZINC_RDNA2_USER=root

# Compatibility fallback for older scripts and ad-hoc shell commands.
ZINC_HOST=<active-rdna-ip-or-hostname>
ZINC_PORT=<active-rdna-ssh-port>
ZINC_USER=<active-rdna-ssh-user>

# Optional overrides consumed by scripts/deploy_rdna4_server.sh
ZINC_REMOTE_DIR=/root/zinc                 # remote checkout path
ZINC_REMOTE_MODEL=/root/models/<file>.gguf # model file the deployed server loads
ZINC_SERVER_PORT=9090                      # remote server port
ZINC_REMOTE_LOG=/tmp/zinc_<port>.log       # remote log file
```

Node-specific deploy overrides can use the same prefix, for example
`ZINC_RDNA2_REMOTE_MODEL`, `ZINC_RDNA2_SERVER_PORT`, and
`ZINC_RDNA2_REMOTE_LOG`.

Intel Arc benchmark nodes use separate keys so they do not override the RDNA defaults:

```bash
ZINC_INTEL_HOST=<ip>
ZINC_INTEL_PORT=<ssh-port>
ZINC_INTEL_USER=<ssh-user>
ZINC_INTEL_WORKDIR=/home/<ssh-user>/zinc-gpu-loop
ZINC_INTEL_XDG_CACHE_HOME=/home/<ssh-user>/.cache
ZINC_INTEL_MODEL_ROOT=/home/<ssh-user>/.cache/zinc/models/models
```

Deploy and test:

```bash
# Full deploy: sync → build → restart → health check
bash scripts/deploy_rdna4_server.sh

# Skip steps as needed
bash scripts/deploy_rdna4_server.sh --no-build --no-restart
```

The deploy script includes a retry health check (30 attempts, 1s apart) to handle model loading time.

## Key Architecture Decisions

Before making changes to these areas, understand the existing design:

- **Compute graph IR** — the decode graph is built from GGUF metadata, not hand-coded
- **Model architectures** — Qwen3, Qwen3.5, and Gemma 4 (dense transformer, MoE, and SSM hybrid)
- **GGUF parsing** — zero-copy mmap with DMA to GPU VRAM
- **Vulkan init** — single-device, single-queue, push-constant dispatch
- **Metal init** — default system device, zero-copy `newBufferWithBytesNoCopy`

## Contributing

See [CONTRIBUTING.md](https://github.com/zolotukhin/zinc/blob/main/CONTRIBUTING.md) for development expectations, workflow, and what makes a good contribution.

See [Code of Conduct](https://github.com/zolotukhin/zinc/blob/main/CODE_OF_CONDUCT.md) for community standards.

## Further Reading

- [Getting Started](/zinc/docs/getting-started/) — first run, model download, basic usage
- [Running ZINC](/zinc/docs/running-zinc/) — CLI reference, server mode, managed models
- [API Reference](/zinc/docs/api/) — OpenAI-compatible HTTP endpoints
- [Technical Specification](/zinc/docs/spec/) — architecture, kernels, scheduler
- [RDNA4 Tuning Guide](/zinc/docs/rdna4-tuning/) — performance profiling and optimization
- [AMD GPU Reference](/zinc/docs/amd-gpu-reference/) — RDNA3/RDNA4 hardware details
- [Intel GPU Reference](/zinc/docs/intel-gpu-reference/) — Arc B-series hardware, memory bandwidth, and Xe2 opcode notes
- [Apple Silicon Reference](/zinc/docs/apple-silicon-reference/) — M1–M5 capabilities
- [Apple Metal Reference](/zinc/docs/apple-metal-reference/) — MSL kernel optimization
