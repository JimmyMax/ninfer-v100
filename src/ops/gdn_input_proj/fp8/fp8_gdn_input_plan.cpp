#include "ops/linear/common/route_table.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan_unified.h"
#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"
#if defined(NINFER_SM70)
#    include "ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.h" // 本仓库 V100 增量
#    include "ops/linear/fp8/fp8_launch.h"                        // 本仓库 V100 增量：QPN
#endif

#include <cstdint>
#include <cuda_fp16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8GdnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8GdnInputRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    return tokens >= 17 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

} // namespace

static std::size_t fp8_gdn_input_workspace_capacity_bytes_own(LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N16384K5120::kInputRows)
               : 0;
}

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：本 op 的 16384x5120 在 sm_70 上由 QPN（t<=32，
    // 零暂存）与 CUTLASS（t>=33）拥有全部宽度；容量来自同一决策而不是 route table。
    if (fp8_volta_qpn_takes(policy, Fp8N16384K5120::kOutputRows, Fp8N16384K5120::kInputRows,
                                std::min(max_tokens, kFp8VoltaQpnMaxTokens))) {
        return max_tokens <= kFp8VoltaQpnMaxTokens
                   ? 0
                   : fp8_gdn_input_cutlass_workspace_bytes(max_tokens);
    }
#endif
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return fused_route_table("unified/fp8_gdn_input", width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::fp8_gdn_input_workspace_capacity_bytes(policy, first, last)
                       : fp8_gdn_input_workspace_capacity_bytes_own(policy, first, last);
        });
}

void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：预打包过的权重只能由布局感知的路线消费。
    // GDN 输入投影（16384x5120）在解码/验证宽度上走 QPN：上游把 W=4..6 交给物化路线，
    // 其中 5..6 落到只有 ~10% 带宽的 fp8_a16_ksplit_mma（48 层 x 0.89 ms = 42.8 ms/轮，
    // 占满 77 ms 轮的 56%），QPN 只要 ~0.30 ms。更宽由上层 CUTLASS / 旧阶梯接管。
    if (is_volta_qpn_prepacked_fp8(weight.qtype, weight.layout) &&
        x.ne[1] <= kFp8VoltaQpnMaxTokens) {
        // 加载时预留的 FP16 激活暂存区（见 fp8_launch.h）：同形状同布局实测 548 -> 768 GB/s。
        // 暂存区缺失（未预留）时退回 bf16 直读路径，不影响正确性。
        if (fp8_gdn_qpn_stage_buffer() != nullptr) {
            void* const stage = fp8_gdn_qpn_stage_buffer();
            fp8_stage_bf16_activation_sm70(x, stage, stream);
            launch_fp8_gdn_input_volta_qpn(x, weight, qkv, z, stage, stream);
            return;
        }
        launch_fp8_gdn_input_volta_qpn(x, weight, qkv, z, nullptr, stream);
        return;
    }
#endif
    if (x.ne[1] == 1) {
        fp8_gdn_input_decode_launch(x, weight, qkv, z, stream);
    } else {
        fp8_gdn_input_matrix_launch(x, weight, qkv, z, stream);
    }
}

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    fp8_gdn_input_a8_launch(x, weight, qkv, z, scratch, stream);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：sm_70 上 QPN 支持的 GDN 输入投影 t>=33 走
    // CUTLASS 宽 T，t<=32 走 QPN（零暂存）。必须在 route-table 检查之前拦截。
    if (fp8_volta_qpn_takes(policy, weight.n, weight.k,
                                std::min<std::int32_t>(x.ne[1], kFp8VoltaQpnMaxTokens))) {
        if (x.ne[1] >= 33) {
            if (workspace == nullptr) {
                throw std::invalid_argument("fp8 Volta GDN prefill requires workspace");
            }
            fp8_gdn_input_cutlass_sm70_launch(x, weight, qkv, z, *workspace, stream);
            return;
        }
        // 与 a16 dispatch 一致：用加载时预留的固定暂存区把激活暂存成 FP16。
        if (fp8_gdn_qpn_stage_buffer() != nullptr) {
            void* const stage = fp8_gdn_qpn_stage_buffer();
            fp8_stage_bf16_activation_sm70(x, stage, stream);
            launch_fp8_gdn_input_volta_qpn(x, weight, qkv, z, stage, stream);
            return;
        }
        launch_fp8_gdn_input_volta_qpn(x, weight, qkv, z, nullptr, stream);
        return;
    }
#endif
    if (fused_route_table("unified/fp8_gdn_input", x.ne[1]) == LinearRouteTable::Unified) {
        unified::fp8_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }
    if (resolve_route(policy, x.ne[1]) == Fp8GdnInputRoute::A16) {
        fp8_gdn_input_a16_dispatch(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    fp8_gdn_input_a8_dispatch(x, weight, qkv, z, *workspace, stream);
}

} // namespace ninfer::ops::detail
