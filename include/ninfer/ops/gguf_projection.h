#pragma once

// GGUF fused-projection part descriptor. A fused input projection over GGUF matrices has logical
// parts that may sit in different parents of different block types. Each part is rows of one
// parent landing at `row` of output `output`: for a GDN projection 0 = q/k/v and 1 = z; for an
// attention projection 0 = query, 1 = gate, 2 = key and 3 = value.

#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cstdint>
#include <vector>

namespace ninfer::ops {

struct GgufProjectionPart {
    Weight weight;
    std::int32_t output = 0;
    std::int32_t row    = 0;
};

struct GgufProjectionWeights {
    std::vector<GgufProjectionPart> parts;
    LinearPolicy policy = LinearPolicy::A16Only;
};

} // namespace ninfer::ops
