#include "core/device.h" // 本仓库 V100 增量：CUDA_CHECK
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_shapes_unified.h"
#include "ops/linear/fp8/fp8_format.h"
#if defined(NINFER_SM70)
#    include "core/layout.h"                 // 本仓库 V100 增量
#    include "ops/linear/fp8/fp8_cutlass_sm70.h"
#    include "ops/linear/fp8/fp8_launch.h"   // 本仓库 V100 增量：QPN 声明
#endif
#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
#if defined(NINFER_SM70)
// 本仓库 V100 增量（docs/v100/PLAN.md）：宽 T 的 CUTLASS 路线要把权重整块反量化成 FP16
// （n*k*2 字节）。当这块暂存会主导 arena 时（1.27 GB 的输出头会变成 2.54 GB，而引擎按最宽
// 上下文预留它），改用分块 QPN：t <= 32 时它本来就只读一遍权重，而输出头在 prefill 里只算
// 最后几行。阈值只决定暂存从哪儿来，两条路线都认预打包布局。
constexpr std::uint64_t kFp8CutlassWeightLimitBytes = 256ULL << 20;

[[nodiscard]] bool fp8_wide_uses_cutlass(std::int32_t n, std::int32_t k) noexcept {
    return static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k) * 2U <=
           kFp8CutlassWeightLimitBytes;
}
#endif
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
                         &kFp8N5120K6144,  &kFp8N5120K17408, &kFp8N248320K5120};
const std::array kUnifiedShapes{&unified::kFp8N14336K5120, &unified::kFp8N16384K5120,
                                &unified::kFp8N34816K5120, &unified::kFp8N5120K6144,
                                &unified::kFp8N5120K17408, &unified::kFp8N248320K5120};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                    LinearRouteTable table) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : table == LinearRouteTable::Unified ? kUnifiedShapes : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}
} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：QPN 支持的 FP8 形状在 sm_70 上由 QPN（t<=32，
    // 零暂存）与 CUTLASS（t>32，自带暂存；暂存过大的形状仍走分块 QPN）拥有全部宽度。容量
    // 必须来自同一决策而不是 route table——测试会强制 unified 表，而钩子不受表影响。
    if (fp8_volta_qpn_takes(policy, n, k, std::min(max_tokens, kFp8VoltaQpnMaxTokens))) {
        if (max_tokens <= kFp8VoltaQpnMaxTokens || !fp8_wide_uses_cutlass(n, k)) { return 0; }
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(DType::BF16, {n, max_tokens});
        return layout.peak_bytes(1) + fp8_cutlass_sm70_workspace_bytes(n, k, max_tokens);
    }
#endif
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return linear_route_table(LinearRouteFamily::Fp8, width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) -> std::size_t {
            const auto& shape = resolve_shape(n, k, policy, table);
            if (!allows_a8(policy) || !shape.uses_a8(first, last)) { return 0; }
            return fp8_a8_workspace_capacity_bytes(
                last, k, shape.partial_capacity_bytes ? shape.partial_capacity_bytes(last) : 0);
        });
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：QPN 支持的 FP8 形状在 sm_70 上全部宽度绕过 route
    // table：t<=32 走 QPN（读 BF16 激活，零暂存），t>32 走 CUTLASS 宽 T（按需解量化 + FP16
    // tensor-core GEMM）。必须在 route-table 检查之前拦截。
    if (fp8_volta_qpn_takes(policy, weight.n, weight.k,
                                std::min(x.ne[1], kFp8VoltaQpnMaxTokens))) {
        if (x.ne[1] <= kFp8VoltaQpnMaxTokens || !fp8_wide_uses_cutlass(weight.n, weight.k)) {
            // QPN 直接写进 out，按 32 列分块（t <= 32 时只有一块，等价于解码宽度的那一次）。
            for (std::int32_t token_begin = 0; token_begin < x.ne[1];
                 token_begin += kFp8VoltaQpnMaxTokens) {
                const std::int32_t active =
                    std::min(kFp8VoltaQpnMaxTokens, x.ne[1] - token_begin);
                auto* input = static_cast<std::uint8_t*>(x.data) +
                              static_cast<std::int64_t>(token_begin) * weight.k *
                                  sizeof(std::uint16_t);
                auto* output = static_cast<std::uint8_t*>(out.data) +
                               static_cast<std::int64_t>(token_begin) * weight.n *
                                   sizeof(std::uint16_t);
                Tensor input_chunk(input, DType::BF16, {weight.k, active});
                Tensor output_chunk(output, DType::BF16, {weight.n, active});
                launch_fp8_volta_qpn(input_chunk, weight, output_chunk, stream);
            }
            return;
        }
        if (workspace == nullptr) {
            throw std::invalid_argument("fp8 Volta linear prefill requires workspace");
        }
        auto scope       = workspace->scope();
        Tensor projected = workspace->alloc(DType::BF16, {weight.n, x.ne[1]});
        fp8_cutlass_sm70_launch(x, weight, projected, *workspace, stream);
        // CUTLASS 路线把结果写在 projected 平面（它需要一块不属于调用方的输出缓冲），
        // 没有 epilogue 可以顺带写出去，所以这里拷回 out。
        CUDA_CHECK(cudaMemcpyAsync(out.data, projected.data,
                                   static_cast<std::size_t>(weight.n) * x.ne[1] *
                                       sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }
#endif
    const auto& shape = resolve_shape(weight.n, weight.k, policy,
                                      linear_route_table(LinearRouteFamily::Fp8, x.ne[1]));
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k,
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(x.ne[1]) : 0);
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
