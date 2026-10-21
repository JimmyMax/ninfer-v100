#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/nvfp4/nvfp4_dispatch_unified.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "core/layout.h"
#if defined(NINFER_TIER_VOLTA)
#    include "ops/linear/nvfp4/nvfp4_launch.h" // 本仓库 V100 增量：kNvfp4VoltaQpnMaxTokens
#endif
#if defined(NINFER_TIER_AMPERE)
#    include "ops/linear/bf16/bf16_dispatch.h" // Ampere 宽 T A16 路线的 GEMM 族
#    include "ops/linear/nvfp4/nvfp4_a16_wide_sm86.h"
#endif
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144, &kNvfp4N5120K17408};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}
} // namespace

static std::size_t nvfp4_linear_workspace_capacity_bytes_own(std::int32_t n, std::int32_t k,
                                                             LinearPolicy policy,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    return allows_a4(policy) && shape.uses_a4(min_tokens, max_tokens)
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
#if defined(NINFER_TIER_AMPERE)
    // A16 族的宽 T（t > 32）在表检查之前被宽路线接管（见 nvfp4_dispatch），容量必须同源：
    // 只占 BF16 暂存平面，GEMM 直接写调用方的 out，无其它工作区。
    if (!allows_a4(policy) && max_tokens >= kNvfp4A16WideFirstToken) {
        (void)resolve_shape(n, k, policy);
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(nvfp4_a16_wide_staged_bytes(n, k), 256);
        return layout.peak_bytes(1);
    }
#endif
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return linear_route_table(LinearRouteFamily::Nvfp4, width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_linear_workspace_capacity_bytes(n, k, policy, first, last)
                       : nvfp4_linear_workspace_capacity_bytes_own(n, k, policy, first, last);
        });
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
#if defined(NINFER_TIER_AMPERE)
    // A16 族的宽 T（t > 32）：把 NVFP4 平面精确解码为 BF16 暂存，交给既有 BF16 GEMM 族
    // （宽 T 上就是 cuBLAS 张量核心）。形状路线的 32 列分块循环在 32 以下仍然更省带宽，
    // 所以只在 32 以上接管。
    if (!allows_a4(policy) && x.ne[1] >= kNvfp4A16WideFirstToken) {
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 A16 wide linear requires caller workspace");
        }
        validate_nvfp4_weight(weight, "nvfp4 linear");
        auto scope = workspace->scope();
        const DeviceSpan stage =
            workspace->alloc_bytes(nvfp4_a16_wide_staged_bytes(weight.n, weight.k), 256);
        nvfp4_a16_wide_staged(weight, stage.data, stream);
        Weight staged = weight;
        staged.qtype  = QType::BF16;
        staged.qdata  = stage.data;
        bf16_dispatch(x, staged, out, policy, stream);
        return;
    }
#endif
#if defined(NINFER_TIER_VOLTA)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：加载时重排过的权重（VoltaQpnPrepacked）只有两条
    // 消费路径——QPN（t <= 32，见 linear.cpp 的钩子）与 CUTLASS（t > 32，见 linear_swiglu /
    // linear_add 的钩子）。落到这里的宽 T 说明调用方把重排后的权重喂给了没有片段序解码器的
    // 算子；明确报错，而不是把片段序当行主序算出垃圾。
    if (weight.layout == QuantLayout::VoltaQpnPrepacked &&
        x.ne[1] > kNvfp4VoltaQpnMaxTokens) {
        throw std::invalid_argument(
            "nvfp4 linear: a prepacked weight has no wide-T decoder on this path");
    }
#endif
    if (x.ne[1] > 0 &&
        linear_route_table(LinearRouteFamily::Nvfp4, x.ne[1]) == LinearRouteTable::Unified) {
        unified::nvfp4_dispatch(x, weight, out, policy, workspace, stream);
        return;
    }
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a4(policy) || !shape.uses_a4(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
