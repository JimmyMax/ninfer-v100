#include "ops/linear/nvfp4/nvfp4_volta_sm70.h"

#if defined(NINFER_TIER_VOLTA)
#    include "ops/linear/nvfp4/nvfp4_launch.h"

#    include <cuda_fp16.h>

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

void nvfp4_volta_project_staged(const Tensor& x, const Weight& w, Tensor& out,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    const std::size_t bytes =
        static_cast<std::size_t>(w.k) * static_cast<std::size_t>(x.ne[1]) * sizeof(half);
    const std::size_t aligned_used = (workspace.used() + 255) / 256 * 256;
    if (workspace.capacity() >= aligned_used && workspace.capacity() - aligned_used >= bytes) {
        auto scope             = workspace.scope();
        const DeviceSpan stage = workspace.alloc_bytes(bytes, 256);
        nvfp4_volta_stage_bf16_fp16_sm70(x, stage.data, stream);
        launch_nvfp4_volta_qpn_fp16(x, stage.data, w, out, stream);
        return;
    }
    launch_nvfp4_volta_qpn(x, w, out, stream);
}

} // namespace ninfer::ops::detail
#endif
