#include "ops/linear/fp8/fp8_prepack_sm70.h"

#include "artifact/permute_fp8_qpn.h"
#include "ops/linear/fp8/fp8_format.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// 本仓库 V100 增量（docs/v100/PLAN.md）：加载时的原位重排实现在 artifact 模块
// （artifact/permute_fp8_qpn.cu），因为模型挂起后重新上传也要走同一段置换；这里只是把 Weight
// 视图的码字平面交给它。尺度平面原位不动（QPN8 按输出行读一个 bf16 尺度）。
void fp8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream) {
    (void)validate_fp8_weight(weight, "FP8 QPN prepack");
    if (weight.n <= 0 || weight.k <= 0 || (weight.n % 32) != 0 || (weight.k % 128) != 0 ||
        (weight.k / 128) < 8) {
        throw std::invalid_argument(
            "FP8 QPN prepack: shape must tile N32 K128 with at least eight k blocks");
    }
    artifact::permute_fp8_to_qpn_planes(
        weight.n, weight.k, const_cast<std::byte*>(static_cast<const std::byte*>(weight.qdata)),
        stream);
    weight.layout = QuantLayout::VoltaQpnPrepacked;
}

} // namespace ninfer::ops::detail
