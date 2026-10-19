#include "ops/linear/common/route_table.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h"
#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#if defined(NINFER_TIER_VOLTA)
#    include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h" // 本仓库 V100 增量：宽 T prefill 路径
#endif

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedW4A4,
    LinearW4A4Post,
    TmaFusedW4A4,
};

constexpr std::int32_t kFusedMaxTokens = 128;

// A ragged width is worth the fused route once its partial M tile is not half of the grid. While
// the grid is two tiles the fused kernel and the public linear + silu_mul composition are close
// enough on an RTX 5090 that the sign of the difference does not survive a second session, so
// that band keeps the composition it already had.
constexpr std::int32_t kRaggedTmaFloor = 2 * kNvfp4TmaBlockM;
// The capacity function walks one step down from this floor to find the widest width the
// composition still serves. That walk is only correct while the floor is at least one whole tile:
// below that the composition owns widths above the floor as well, and one step does not reach
// them.
static_assert(kRaggedTmaFloor >= kNvfp4TmaBlockM,
              "a ragged floor inside the first M tile would leave the composition band above it "
              "out of the workspace capacity");

Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 linear_swiglu: invalid compute policy");
    }
    if (!allows_a4(policy)) { // A16Only, AllowA8 and the integer-A8 policies
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        // 本仓库 V100 增量（docs/v100/PLAN.md）：sm_70 上 t = 2..32 由 dispatch 顶部的 QPN
        // 拆分钩子接管；更宽的 prefill 宽度不会到这里（wrapper 把它交回 sm_70 路线，见
        // linear_swiglu.cpp 的排除条件），故这里保持上游的注册边界。
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        // Deliberately verbose. This throw fires during *runtime planning*, before a token
        // is processed, so on sm_86 it is the entire user-visible symptom of an artifact
        // that cannot be loaded at all -- and "registered only through T=16" gives an
        // operator nothing to act on. It took a full-suite run and a bisect of two
        // deliberate decisions to work out why; the message now carries that.
        throw std::invalid_argument(
            "nvfp4 linear_swiglu: the A16 route is registered only through T=16, and on sm_80/86/89 A16 is the only policy available for NVFP4 weights (no NVFP4 tensor-core path), so this artifact cannot serve a prefill chunk wider than 16 columns on this architecture. Use a groupwise-int artifact here; an sm_120a build runs NVFP4 through AllowA4 at any width.");
    }
    if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
    if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    if (tokens <= kFusedMaxTokens) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
    // This route dispatches its own fused kernel rather than a Linear shape's, so it carries its
    // own condition - a whole tile, or a ragged width with enough behind it; the call site below
    // forces the matching scale layout either way.
    if (tokens >= kNvfp4TmaBlockM &&
        ((tokens % kNvfp4TmaBlockM) == 0 || tokens >= kRaggedTmaFloor)) {
        return Nvfp4LinearSwiGluRoute::TmaFusedW4A4;
    }
    return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
}

struct Nvfp4LinearSwiGluWorkspace {
    Tensor projected;
    DeviceSpan linear;
};

template <class Allocator>
Nvfp4LinearSwiGluWorkspace allocate_baseline_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4LinearSwiGluWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {Nvfp4N34816K5120::kOutputRows, tokens}, 256);
    const std::size_t linear_bytes = linear_workspace_capacity_bytes(
        QType::NVFP4, Nvfp4N34816K5120::kOutputRows, Nvfp4N34816K5120::kInputRows,
        LinearPolicy::AllowA4, tokens, tokens);
    out.linear = allocator.alloc_bytes(linear_bytes, 256);
    return out;
}

template <class Allocator>
Nvfp4W4a4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_w4a4_workspace(allocator, tokens, Nvfp4N34816K5120::kInputRows);
}

std::size_t baseline_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_baseline_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

#if defined(NINFER_TIER_VOLTA)
// 本仓库 V100 增量：sm_70 的 NVFP4 gate_up QPN 拆分暂存（t <= 32）——gate fp32 + 激活 fp16。
// 容量用与派发相同的 layout 序列算出，保证“查询即执行高水位”。
std::size_t sm70_split_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(static_cast<std::size_t>(Nvfp4N34816K5120::kOutputRows / 2) *
                                 static_cast<std::size_t>(tokens) * 4,
                             256);
    (void)layout.alloc_bytes(
        static_cast<std::size_t>(Nvfp4N34816K5120::kInputRows) * static_cast<std::size_t>(tokens) *
            2,
        256);
    return layout.peak_bytes(1);
}

// 宽 T（t > 32）的 CUTLASS 路线暂存：projected BF16 平面 + FP16 权重/激活暂存。
std::size_t sm70_wide_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {Nvfp4N34816K5120::kOutputRows, tokens}, 256);
    (void)layout.alloc_bytes(nvfp4_cutlass_sm70_workspace_bytes(Nvfp4N34816K5120::kOutputRows,
                                                               Nvfp4N34816K5120::kInputRows,
                                                               tokens),
                             256);
    return layout.peak_bytes(1);
}
#endif

} // namespace

static std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes_own(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if (!allows_a4(policy)) {
#if defined(NINFER_TIER_VOLTA)
        // 本仓库 V100 增量：A16 族在 t <= 32 上由 QPN 拆分接管（见公共容量函数）；
        // 更宽的宽度不会调用到本函数。
        return sm70_split_workspace_bytes(
            std::min(max_tokens, kNvfp4VoltaQpnMaxTokens));
#else
        return 0;
#endif
    }
    if (max_tokens <= 4) {
        return 0;
    }

    std::size_t maximum = 0;
    if (min_tokens <= kFusedMaxTokens && max_tokens >= 5) {
        maximum = fused_workspace_bytes(std::min(max_tokens, kFusedMaxTokens));
    }
    // From the ragged floor upward the TMA route owns every width, so the widest it can be asked
    // for is the interval's own maximum; below the floor it owns only whole tiles, and the guard
    // then rejects an interval that contains none.
    const std::int32_t largest_fused =
        max_tokens >= kRaggedTmaFloor ? max_tokens : max_tokens - (max_tokens % kNvfp4TmaBlockM);
    if (largest_fused >= std::max(min_tokens, kNvfp4TmaBlockM)) {
        maximum = std::max(maximum, fused_workspace_bytes(largest_fused));
    }

    // The composition keeps the gap below the floor, so it no longer has to be sized for the
    // interval's maximum. Capping at the floor puts the search where the only TMA widths are whole
    // tiles, and no two consecutive widths are both whole tiles, so one step is always enough.
    std::int32_t last_baseline = std::min(max_tokens, kRaggedTmaFloor - 1);
    if (resolve_route(policy, last_baseline) == Nvfp4LinearSwiGluRoute::TmaFusedW4A4) {
        --last_baseline;
    }
    if (last_baseline >= std::max(min_tokens, kFusedMaxTokens + 1)) {
        maximum = std::max(maximum, baseline_workspace_bytes(last_baseline));
    }
    return maximum;
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
#if defined(NINFER_TIER_VOLTA)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：sm_70 的 A16 族在 t <= 32 上被 QPN 拆分接管
    // （表检查之前），容量必须在这里短路；更宽的 prefill 宽度仍走上游路线与上游容量。
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    if (!allows_a4(policy)) {
        // 窄带（t <= 32）与宽带（t > 32）的路线不同，跨带取两者最大；单宽度查询时恰好落在
        // 自己那一段，保持“查询即执行高水位”。
        std::size_t need = 0;
        if (min_tokens <= kNvfp4VoltaQpnMaxTokens) {
            need = sm70_split_workspace_bytes(std::min(max_tokens, kNvfp4VoltaQpnMaxTokens));
        }
        if (max_tokens > kNvfp4VoltaQpnMaxTokens) {
            need = std::max(need, sm70_wide_workspace_bytes(max_tokens));
        }
        return need;
    }
#endif
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return fused_route_table("unified/nvfp4_linear_swiglu", width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_linear_swiglu_workspace_capacity_bytes(
                             unified_policy(policy), first, last)
                       : nvfp4_linear_swiglu_workspace_capacity_bytes_own(policy, first, last);
        });
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
#if defined(NINFER_TIER_VOLTA)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：解码/校验宽度（t = 1..32）上的 NVFP4 gate_up
    // （34816 x 5120，每层 178 M 参数，占模型 NVFP4 质量的三分之二）走 fork 的 QPN 拆分路线：
    // 两个独立 QPN2 pass（gate / up 各半）写 fp32 暂存，再由合并内核做 silu(gate)*up。
    // fork 实测拆分 371.6 GB/s vs 融合内核 225.4 GB/s，故不用融合单内核。
    if (!allows_a4(policy) && nvfp4_linear_swiglu_qpn_split_supported(weight.k, x.ne[1])) {
        constexpr std::int32_t kIntermediate = Nvfp4N34816K5120::kOutputRows / 2;
        const std::size_t gate_bytes = static_cast<std::size_t>(kIntermediate) *
                                       static_cast<std::size_t>(x.ne[1]) * sizeof(float);
        const std::size_t act_bytes  = static_cast<std::size_t>(weight.k) *
                                       static_cast<std::size_t>(x.ne[1]) * sizeof(std::uint16_t);
        const std::size_t aligned_used = (workspace.used() + 255) / 256 * 256;
        if (workspace.capacity() >= aligned_used &&
            workspace.capacity() - aligned_used >= gate_bytes + act_bytes) {
            auto scope              = workspace.scope();
            const DeviceSpan gate   = workspace.alloc_bytes(gate_bytes, 256);
            const DeviceSpan act    = workspace.alloc_bytes(act_bytes, 256);
            nvfp4_linear_swiglu_qpn_split_launch(x, weight, out,
                                                 reinterpret_cast<float*>(gate.data), act.data,
                                                 stream);
            return;
        }
    }
    // 更宽的 prefill 宽度（t > 32）：一次性解量化 NVFP4 为 FP16，CUTLASS sm_70 GEMM 复用再
    // silu_mul。同样在表检查之前接管，容量才能与执行同源。
    // （fork 在这里的解量化用 fp16 移位技巧，实测过不了 A16 判据；本仓库改成与 sm_70 dequant
    //   同构的精确解码，见 nvfp4_cutlass_sm70.cu 的注释。）
    if (!allows_a4(policy) && x.ne[1] > kNvfp4VoltaQpnMaxTokens) {
        auto scope       = workspace.scope();
        Tensor projected = workspace.alloc(DType::BF16, {Nvfp4N34816K5120::kOutputRows, x.ne[1]}, 256);
        nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
        constexpr std::int32_t kIntermediate = Nvfp4N34816K5120::kOutputRows / 2;
        silu_mul(projected.slice(0, 0, kIntermediate),
                 projected.slice(0, kIntermediate, kIntermediate), out, stream);
        return;
    }
#endif
    if (fused_route_table("unified/nvfp4_linear_swiglu", x.ne[1]) == LinearRouteTable::Unified) {
        unified::nvfp4_linear_swiglu_dispatch(x, weight, out, unified_policy(policy), workspace,
                                              stream);
        return;
    }
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedW4A4:
        nvfp4_linear_swiglu_w4a4_launch(x, weight, out, workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedW4A4: {
        auto scope                       = workspace.scope();
        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        launch_nvfp4_w4a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled, stream);
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_linear_swiglu_w4a4_tma(
            scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            x.ne[1], alpha, stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::LinearW4A4Post:
        break;
    }

    auto scope                         = workspace.scope();
    Nvfp4LinearSwiGluWorkspace scratch = allocate_baseline_workspace(workspace, x.ne[1]);
    WorkspaceArena linear_workspace(scratch.linear);
    linear(x, weight, scratch.projected, LinearPolicy::AllowA4, linear_workspace, stream);
    constexpr std::int32_t kIntermediate = Nvfp4N34816K5120::kOutputRows / 2;
    silu_mul(scratch.projected.slice(0, 0, kIntermediate),
             scratch.projected.slice(0, kIntermediate, kIntermediate), out, stream);
}

} // namespace ninfer::ops::detail
