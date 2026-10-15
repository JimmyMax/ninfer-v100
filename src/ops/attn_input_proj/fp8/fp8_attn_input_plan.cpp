#include "ops/linear/common/route_table.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan_unified.h"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"
#if defined(NINFER_SM70)
#    include "ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.h" // 本仓库 V100 增量
#    include "ops/linear/fp8/fp8_launch.h"                        // 本仓库 V100 增量：QPN
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8AttnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8AttnInputRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 attn_input_proj: unsupported policy");
    }
    return tokens >= 5 ? Fp8AttnInputRoute::A8 : Fp8AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1)
        fp8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= kFp8AttnInputLastSimtT)
        fp8_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= kFp8AttnInputLastSmallMmaT)
        fp8_attn_input_a16_small_mma_launch(x, weight, q, gate, k, v, stream);
    else
        fp8_attn_input_a16_gemm_launch(x, weight, q, gate, k, v, stream);
}

} // namespace

static std::size_t fp8_attn_input_workspace_capacity_bytes_own(LinearPolicy policy,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8AttnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N14336K5120::kInputRows)
               : 0;
}

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：本 op 的 14336x5120 在 sm_70 上由 QPN
    // （t<=32，读 BF16 激活，零暂存——上游 A16 解码契约如此）与 CUTLASS（t>=33，自带暂存）
    // 拥有全部宽度。容量来自同一决策而不是 route table——测试会强制 unified 表，而钩子不受
    // 表影响。
    if (fp8_volta_qpn_takes(policy, Fp8N14336K5120::kOutputRows, Fp8N14336K5120::kInputRows,
                                std::min(max_tokens, kFp8VoltaQpnMaxTokens))) {
        // 本仓库 V100 增量：t<=32 的 QPN 现在要一块 k x t 的 FP16 激活暂存区（见
        // fp8_qpn_stage_bytes 的实测收益）；更宽走 CUTLASS，它是自带暂存的，两者取大。
        const std::size_t stage = fp8_qpn_stage_bytes(
            Fp8N14336K5120::kInputRows, std::min(max_tokens, kFp8VoltaQpnMaxTokens));
        return max_tokens <= kFp8VoltaQpnMaxTokens
                   ? stage
                   : std::max(fp8_attn_input_cutlass_workspace_bytes(max_tokens), stage);
    }
#endif
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return fused_route_table("unified/fp8_attn_input", width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::fp8_attn_input_workspace_capacity_bytes(policy, first, last)
                       : fp8_attn_input_workspace_capacity_bytes_own(policy, first, last);
        });
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：注意力投影是这个模型 FP8 权重的主要所在地
    // （每层一个 14336x5120）。sm_70 上 t>=33 走 CUTLASS 宽 T，t<=32 走 QPN。必须在
    // route-table 检查之前拦截。
    if (fp8_volta_qpn_takes(policy, weight.n, weight.k,
                                std::min<std::int32_t>(x.ne[1], kFp8VoltaQpnMaxTokens))) {
        if (x.ne[1] >= 33) {
            if (workspace == nullptr) {
                throw std::invalid_argument("fp8 Volta attention prefill requires workspace");
            }
            fp8_attn_input_cutlass_sm70_launch(x, weight, q, gate, k, v, *workspace, stream);
            return;
        }
        // 本仓库 V100 增量：激活一次性暂存为 FP16 再交给 QPN——同形状同预打包布局下实测
        // 545 -> 741 GB/s（T=5），内核里的逐元素 bf16->fp16 转换就是那 1.4x 的来源。
        if (workspace != nullptr) {
            auto stage_scope       = workspace->scope();
            const DeviceSpan stage = workspace->alloc_bytes(
                fp8_qpn_stage_bytes(weight.k, x.ne[1]), 256);
            fp8_stage_bf16_activation_sm70(x, stage.data, stream);
            launch_fp8_attn_input_volta_qpn(x, weight, q, stage.data, gate, k, v, stream);
            return;
        }
        launch_fp8_attn_input_volta_qpn(x, weight, q, nullptr, gate, k, v, stream);
        return;
    }
#endif
    if (fused_route_table("unified/fp8_attn_input", x.ne[1]) == LinearRouteTable::Unified) {
        unified::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }
    if (resolve_route(policy, x.ne[1]) == Fp8AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    fp8_attn_input_a8_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
