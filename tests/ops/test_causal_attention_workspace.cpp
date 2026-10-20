// The attention workspace reservation must cover every call the route gates can select.
//
// The planner runs on the configured capacity while a launch passes the keys it actually sees, so
// the two envelopes differ. A route gated on the visible-key count therefore has to be reserved
// for the last key count at which it still runs, not only for the capacity: the Volta flash prefill
// route keeps a staging budget, and a capacity past that budget once left every call of such a
// configuration unreserved, which surfaced as `std::bad_alloc` inside the first attention layer
// (report: `text/layers/3 prefill columns=1920: std::bad_alloc` at `--max-context 200000`).
//
// This checks the invariant directly: the capacity-planned reservation at some capacity is never
// smaller than the reservation a launch needs at any visible key count below that capacity.

#include "ninfer/ops/softmax_attention.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using ninfer::ops::AttentionHeadGeometry;
using ninfer::ops::CausalAttentionExecutionEnvelope;
using ninfer::KvCacheStorage;
using ninfer::ops::causal_softmax_attention_workspace_capacity_bytes;

// The 27B geometry the report ran: 16 query heads over 2 key/value heads, head dim 256, one
// prefill chunk of 1920 rows.
constexpr AttentionHeadGeometry kGeometry{256, 16, 2};
constexpr std::int32_t kChunkTokens = 1920;

std::size_t planned_bytes(KvCacheStorage storage, std::uint32_t capacity) {
    const CausalAttentionExecutionEnvelope envelope{.min_visible_keys = 1,
                                                     .max_visible_keys = capacity,
                                                     .small_prefill    = true};
    return causal_softmax_attention_workspace_capacity_bytes(kGeometry, storage, envelope, 1,
                                                             kChunkTokens, kChunkTokens);
}

std::size_t launch_bytes(KvCacheStorage storage, std::uint32_t visible) {
    const CausalAttentionExecutionEnvelope envelope{.min_visible_keys = visible,
                                                     .max_visible_keys = visible,
                                                     .small_prefill    = true};
    return causal_softmax_attention_workspace_capacity_bytes(kGeometry, storage, envelope, 1,
                                                             kChunkTokens, kChunkTokens);
}

int failures = 0;

void expect_covered(KvCacheStorage storage, const char* name, std::uint32_t capacity,
                    std::vector<std::uint32_t> visibles) {
    const std::size_t plan = planned_bytes(storage, capacity);
    for (const std::uint32_t visible : visibles) {
        const std::size_t need = launch_bytes(storage, visible);
        if (need > plan) {
            std::printf("FAIL %s: capacity %u plans %.1f MiB, a call at %u visible keys needs "
                        "%.1f MiB\n",
                        name, capacity, static_cast<double>(plan) / 1048576.0, visible,
                        static_cast<double>(need) / 1048576.0);
            ++failures;
            continue;
        }
        std::printf("ok   %s: capacity %u plans %.1f MiB >= %.1f MiB at %u keys\n", name, capacity,
                    static_cast<double>(plan) / 1048576.0,
                    static_cast<double>(need) / 1048576.0, visible);
    }
}

} // namespace

int main() {
    // The flash prefill route's staging budget is 131,072 visible keys. Capacities at and past it
    // are the cases that broke; a short call under any of them must still fit the plan.
    const std::vector<std::uint32_t> visibles{1024, 4096, 33280, 83000, 131072};
    for (const std::uint32_t capacity : {131072U, 200000U, 262144U}) {
        expect_covered(KvCacheStorage::Int8Group64, "int8", capacity, visibles);
        expect_covered(KvCacheStorage::BFloat16, "bf16", capacity, visibles);
    }
    if (failures != 0) {
        std::printf("%d reservations are smaller than a call they must cover\n", failures);
        return 1;
    }
    std::printf("every plan covers its calls\n");
    return 0;
}
