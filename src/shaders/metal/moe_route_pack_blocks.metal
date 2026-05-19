#include <metal_stdlib>

using namespace metal;

struct Params {
    uint n_tokens;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint ids_stride;
};

#define NUM_COLS 4u

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device uint* counts [[buffer(2)]],
    device uint* ids [[buffer(3)]],
    device atomic_uint* active_block_count [[buffer(4)]],
    device uint* active_blocks [[buffer(5)]],
    uint expert_id [[thread_position_in_threadgroup]]
) {
    if (expert_id == 0u) {
        atomic_store_explicit(active_block_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (expert_id >= p.n_experts) {
        return;
    }

    uint n = 0u;
    for (uint token = 0u; token < p.n_tokens; token++) {
        device const uint* row = routing + token * p.routing_stride;
        for (uint slot = 0u; slot < p.k; slot++) {
            if (row[slot] == expert_id) {
                ids[expert_id * p.ids_stride + n] = token * p.k + slot;
                n++;
            }
        }
    }

    counts[expert_id] = n;

    for (uint packed_base = 0u; packed_base < n; packed_base += NUM_COLS) {
        const uint block_id = atomic_fetch_add_explicit(active_block_count, 1u, memory_order_relaxed);
        active_blocks[block_id] = expert_id | ((packed_base / NUM_COLS) << 16u);
    }
}
