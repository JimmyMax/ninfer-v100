#include "ops/linear/nvfp4/nvfp4_volta_sm70.h"

#if defined(NINFER_TIER_VOLTA)
#    include "ops/linear/nvfp4/nvfp4_launch.h"

namespace ninfer::ops::detail {

bool nvfp4_volta_takes(const Weight& w, std::int32_t tokens) noexcept {
    if (w.qtype != QType::NVFP4) { return false; }
    // 存放布局（BlockScaleK16M128x4，artifact 原样）与加载时重排后的 QPN 布局都接受；
    // QPN 内核自己按 w.layout 选择 prepacked / 未 prepacked 的变体。
    if (!is_block_scale_nvfp4(w.layout)) {
        return false;
    }
    if (tokens < 1 || tokens > kNvfp4VoltaQpnMaxTokens) { return false; }
    return nvfp4_volta_qpn_supported(w.n, w.k, tokens);
}

void nvfp4_volta_project(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_nvfp4_volta_qpn(x, w, out, stream);
}

} // namespace ninfer::ops::detail
#endif
