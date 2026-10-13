#include "ops/linear/common/route_table.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan_unified.h"
#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#if defined(NINFER_SM70)
// 本仓库 V100 增量（docs/v100/PLAN.md）：Volta QPN 的 residual 输出与激活暂存。
#    include "ops/linear/nvfp4/nvfp4_launch.h"
#    include "ops/linear/nvfp4/nvfp4_volta_sm70.h"
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
        allows_a8_int(policy)) {
        return Nvfp4LinearAddRoute::A16;
    }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    const std::int32_t first_w4a4 = input_rows == 6144 ? 7 : 8;
    return tokens >= first_w4a4 ? Nvfp4LinearAddRoute::W4A4 : Nvfp4LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    constexpr std::int32_t kChunk = 32;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            nvfp4_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            nvfp4_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

} // namespace

static std::size_t nvfp4_linear_add_workspace_capacity_bytes_own(std::int32_t output_rows,
                                                                 std::int32_t input_rows,
                                                                 LinearPolicy policy,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
#if defined(NINFER_SM70)
    // 本仓库 V100 增量：解码宽度上的 NVFP4 down 投影走 QPN 的 residual 路径，
    // 需要一块 fp16 激活暂存（k * tokens * 2 字节，256 字节对齐）。
    if (max_tokens <= kNvfp4VoltaQpnMaxTokens &&
        nvfp4_volta_qpn_supported(output_rows, input_rows, max_tokens)) {
        return static_cast<std::size_t>(input_rows) * static_cast<std::size_t>(max_tokens) * 2 +
               256;
    }
#endif
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearAddRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return fused_route_table("unified/nvfp4_linear_add", width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_linear_add_workspace_capacity_bytes(
                             output_rows, input_rows, unified_policy(policy), first, last)
                       : nvfp4_linear_add_workspace_capacity_bytes_own(output_rows, input_rows,
                                                                       policy, first, last);
        });
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：解码/校验宽度上的 NVFP4 down 投影
    // （5120 x 17408，每层 89 M 参数）。先做一次 fp16 激活暂存，再用 QPN 的
    // residual 输出策略一次性完成 out += x·wᵀ（无临时投影、无单独 residual_add）。
    if (x.ne[1] <= kNvfp4VoltaQpnMaxTokens &&
        nvfp4_volta_qpn_supported(weight.n, weight.k, x.ne[1])) {
        const std::size_t stage_bytes =
            static_cast<std::size_t>(x.ne[0]) * static_cast<std::size_t>(x.ne[1]) * 2;
        const std::size_t aligned_used = (workspace.used() + 255) / 256 * 256;
        if (workspace.capacity() >= aligned_used &&
            workspace.capacity() - aligned_used >= stage_bytes) {
            auto stage_scope       = workspace.scope();
            const DeviceSpan stage = workspace.alloc_bytes(stage_bytes, 256);
            nvfp4_volta_stage_bf16_fp16_sm70(x, stage.data, stream);
            launch_nvfp4_volta_qpn_fp16_residual(x, stage.data, weight, residual, stream);
            return;
        }
    }
#endif
    if (fused_route_table("unified/nvfp4_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::nvfp4_linear_add_dispatch(x, weight, residual, unified_policy(policy), workspace,
                                           stream);
        return;
    }
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
    auto scope                       = workspace.scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_w4a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
