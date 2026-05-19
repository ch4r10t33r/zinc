#include <metal_stdlib>

using namespace metal;

struct Params {
    uint n_tokens;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint ids_stride;
};

#define NUM_COLS 8u

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],
    device uint* ids [[buffer(3)]],
    device atomic_uint* active_block_count [[buffer(4)]],
    device uint* active_blocks [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid == 0u) {
        atomic_store_explicit(active_block_count, 0u, memory_order_relaxed);
    }
    if (tid < p.n_experts) {
        atomic_store_explicit(counts + tid, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint expert_id = tid;
    if (expert_id >= p.n_experts) {
        return;
    }

    const uint total_routes = p.n_tokens * p.k;
    uint count = 0u;
    for (uint route = 0u; route < total_routes; route++) {
        const uint token = route / p.k;
        const uint slot = route - token * p.k;
        const uint route_expert = routing[token * p.routing_stride + slot];
        if (route_expert != expert_id) {
            continue;
        }

        if (count < p.ids_stride) {
            ids[expert_id * p.ids_stride + count] = route;
            if ((count % NUM_COLS) == 0u) {
                const uint block_id = atomic_fetch_add_explicit(active_block_count, 1u, memory_order_relaxed);
                active_blocks[block_id] = expert_id | ((count / NUM_COLS) << 16u);
            }
        }
        count++;
    }

    atomic_store_explicit(counts + expert_id, count, memory_order_relaxed);
}
