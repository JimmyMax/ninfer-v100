#include "ops/linear/common/route_table.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan_unified.h"
#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#if defined(NINFER_SM70)
#    include "core/layout.h"          // 本仓库 V100 增量：WorkspaceLayoutBuilder
#    include "ops/linear/fp8/fp8_launch.h"
#endif
#include "ops/linear/fp8/fp8_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearSwiGluRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 linear_swiglu: T must be positive"); }
    // Integer-A8 policies (groupwise-int weights) never select the FP8 A8 path.
    if (policy == LinearPolicy::A16Only || allows_a8_int(policy)) { return Fp8LinearSwiGluRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 linear_swiglu admits only A16 or A8");
    }
    return tokens == 1 || tokens >= 3 ? Fp8LinearSwiGluRoute::A8 : Fp8LinearSwiGluRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kOutputRows = Fp8N34816K5120::kOutputRows / 2;
    constexpr std::int32_t kChunk      = 4;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * kOutputRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {kOutputRows, active});
        if (active == 1) {
            fp8_linear_swiglu_decode_launch(input_chunk, weight, output_chunk, stream);
        } else {
            fp8_linear_swiglu_small_t_launch(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

static std::size_t fp8_linear_swiglu_workspace_capacity_bytes_own(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    const bool interval_uses_a8 = allows_a8(policy) && (min_tokens == 1 || max_tokens >= 3);
    return interval_uses_a8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N34816K5120::kInputRows)
               : 0;
}

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：sm_70 上 t<=32 走拆分式 QPN（gate/up 的 FP32 暂存
    // 是真实需求，也正是上游 A16 契约要求的"激活前保持 FP32"）。容量来自同一决策而不是
    // route table——测试会强制 unified 表，而钩子不受表影响。
    if (min_tokens <= kFp8VoltaQpnMaxTokens &&
        fp8_volta_qpn_takes(policy, Fp8N34816K5120::kOutputRows, Fp8N34816K5120::kInputRows,
                            std::min(max_tokens, kFp8VoltaQpnMaxTokens))) {
        constexpr std::int32_t kHalf = 17408; // Fp8N34816K5120::kOutputRows / 2
        const std::int32_t chunk     = std::min(max_tokens, kFp8VoltaQpnMaxTokens);
        WorkspaceLayoutBuilder split_layout;
        (void)split_layout.alloc_bytes(static_cast<std::size_t>(kHalf) * chunk * sizeof(float),
                                       256);
        (void)split_layout.alloc_bytes(static_cast<std::size_t>(kHalf) * chunk * sizeof(float),
                                       256);
        (void)split_layout.alloc_bytes(
            static_cast<std::size_t>(Fp8N34816K5120::kInputRows) * chunk * sizeof(std::uint16_t),
            256);
        // 宽 T（t>=33）交回上游阶梯：本 op 的 A16 契约要求激活之前保持 FP32，而这里的
        // 拆分路线在 t<=32 用 FP32 平面满足它；单次 CUTLASS 只给出 BF16 的 gate_up 平面，
        // 过不了上游的判据。上游宽 T 阶梯不需要暂存，所以顶层查询（min_tokens > 32）与
        // 落入宽区的单 T 查询都返回 0（见上面的 min_tokens 判断）。
        return split_layout.peak_bytes(1);
    }
#endif
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return fused_route_table("unified/fp8_linear_swiglu", width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::fp8_linear_swiglu_workspace_capacity_bytes(unified_policy(policy),
                                                                             first, last)
                       : fp8_linear_swiglu_workspace_capacity_bytes_own(policy, first, last);
        });
}

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：FP8 SwiGLU 在 sm_70 上 t<=32 走拆分式 QPN
    // （fork 实测拆分快于融合内核；gate/up 的 FP32 暂存是这条路线的一部分），t>=33 走
    // CUTLASS 宽 T。必须在 route-table 检查之前拦截。
    if (weight.n == Fp8N34816K5120::kOutputRows && x.ne[1] <= kFp8VoltaQpnMaxTokens &&
        fp8_volta_qpn_takes(policy, Fp8N34816K5120::kOutputRows, weight.k, x.ne[1])) {
        if (x.ne[1] > kFp8VoltaQpnMaxTokens) { return; } // 宽 T 交回上游阶梯（见容量处的说明）
        constexpr std::int32_t kHalf = Fp8N34816K5120::kOutputRows / 2;
        auto scope                  = workspace.scope();
        const DeviceSpan gate       = workspace.alloc_bytes(
            static_cast<std::size_t>(kHalf) * x.ne[1] * sizeof(float), 256);
        const DeviceSpan up = workspace.alloc_bytes(
            static_cast<std::size_t>(kHalf) * x.ne[1] * sizeof(float), 256);
        const DeviceSpan activation = workspace.alloc_bytes(
            static_cast<std::size_t>(weight.k) * x.ne[1] * sizeof(std::uint16_t), 256);
        fp8_linear_swiglu_qpn_split_launch(x, weight, out,
                                           reinterpret_cast<float*>(gate.data),
                                           reinterpret_cast<float*>(up.data), activation.data,
                                           stream);
        return;
    }
#endif
    if (fused_route_table("unified/fp8_linear_swiglu", x.ne[1]) == LinearRouteTable::Unified) {
        unified::fp8_linear_swiglu_dispatch(x, weight, out, unified_policy(policy), workspace,
                                            stream);
        return;
    }
    if (resolve_route(policy, x.ne[1]) == Fp8LinearSwiGluRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    fp8_linear_swiglu_a8_launch(x, weight, out, workspace, stream);
}

} // namespace ninfer::ops::detail
