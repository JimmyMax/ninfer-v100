#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

#ifdef NINFER_SM70
// Ported from the V100 fork ("perf(sm70): restore long-context attention scaling"): the small-T
// decode policy asks for roughly one split per 480 cached tokens, and the former 85-split ceiling
// started binding near 41K, serializing progressively more KV work into each CTA. Volta's supported
// context is 262,144 tokens, which the 560-split ceiling covers without constraining the policy.
inline constexpr int kSmallTMaximumBaseSplits = 560;
#else
inline constexpr int kSmallTMaximumBaseSplits = 85;
#endif

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = kSmallTMaximumBaseSplits * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;

} // namespace ninfer::ops
