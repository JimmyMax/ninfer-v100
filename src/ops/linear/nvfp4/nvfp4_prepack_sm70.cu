#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"

#include "artifact/permute_nvfp4_qpn.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cstdint>

namespace ninfer::ops::detail {

// 本仓库 V100 增量（docs/v100/PLAN.md）：加载时的原位重排实现放在 artifact 模块
// （artifact/permute_nvfp4_qpn.cu），因为模型挂起后重新上传也要走同一段置换；这里只是把
// Weight 视图的两个平面交给它。Weight 的 qdata/scales 是同一份载荷的两个平面。
void nvfp4_prepack_qpn_sm70(Weight& weight, cudaStream_t stream) {
    (void)validate_nvfp4_weight(weight, "NVFP4 QPN prepack");
    artifact::permute_nvfp4_to_qpn_planes(
        weight.n, weight.k, const_cast<std::byte*>(static_cast<const std::byte*>(weight.qdata)),
        const_cast<std::byte*>(static_cast<const std::byte*>(weight.scales)), stream);
    weight.layout = QuantLayout::VoltaQpnPrepacked;
}

} // namespace ninfer::ops::detail
