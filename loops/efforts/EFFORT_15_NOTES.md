# Effort 15 — running notes

Local machine: Apple M1 Max, 32 GPU cores, 32 GB UMA, MTLGPUFamilyApple7.
Model: `gemma4-26b-a4b-q4k-m` (Gemma 4 26B-A4B MoE, ~4B active, Q4_K_M,
16.9 GB on disk). Chat mode only — raw completion is not a valid coherence
gate for this model.

Effort doc: `MULTI_HOUR_EFFORT_15_METAL_GEMMA_M1.md`.

## Phase 0 — Baseline (TODO: fill in on first cycle)

Build: `zig build -Doptimize=ReleaseFast` at HEAD `<commit>`.

### ZINC baseline (canonical prompt, chat, n=128)

- decode tok/s (3-run median): TODO
- prefill tok/s (3-run median): TODO
- coherence: chat output contains `Paris` — TODO confirm

### Profile

TODO — paste `--profile` dispatch-ms / dispatch-byte breakdown. Separate the
decode-only slice from the ~9 s chat prefill.

### llama.cpp baseline

- decode tok/s (3-run median): TODO
- comparator binary supports Gemma 4: TODO confirm (rebuild if not)
- gap vs ZINC: TODO (absolute + percentage)

## Phase 1 — Hottest decode kernel

TODO

## Cycle log

TODO — one section per cycle: change, microbench result, whole-model result,
verdict (KEEP / REVERT), lesson.

## Do-not-retry list

TODO — populate as reverts accumulate. Also see Effort 11's "Known dead ends"
(measured on M4 — re-test on M1 only with a fresh `bench-metal-shapes`
number).

## Outcome (fill in on exit)

- starting decode tok/s: TODO
- ending decode tok/s: TODO
- decode tok/s at each accepted cycle: TODO
- attempted-and-reverted changes with reasons: TODO
- single most useful learning: TODO
