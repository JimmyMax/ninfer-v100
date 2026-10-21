#include "ops/linear/common/route_table.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h"
#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#if defined(NINFER_TIER_VOLTA)
#    include "ops/linear/nvfp4/nvfp4_launch.h"
#    include "ops/linear/nvfp4/nvfp4_volta_sm70.h"
#endif
#if defined(NINFER_TIER_AMPERE)
#    include "ops/linear/bf16/bf16_dispatch.h" // Ampere 宽 T A16 路线的 GEMM 族
#    include "ops/linear/nvfp4/nvfp4_a16_wide_sm86.h"
#endif

#include <algorithm>
#include <cstdint>

#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4AttnInputRoute::A16;
    }
    if (!allows_a4(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return tokens >= 4 ? Nvfp4AttnInputRoute::W4A4 : Nvfp4AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kChunk  = 32;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor query_chunk(query, DType::BF16, {kQRows, active});
        Tensor gate_chunk(output_gate, DType::BF16, {kQRows, active});
        Tensor key_chunk(key, DType::BF16, {kKvRows, active});
        Tensor value_chunk(value, DType::BF16, {kKvRows, active});
        if (active == 1) {
            nvfp4_attn_input_decode_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                           value_chunk, stream);
        } else {
            nvfp4_attn_input_small_t_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                            value_chunk, stream);
        }
    }
}

} // namespace

static std::size_t nvfp4_attn_input_workspace_capacity_bytes_own(LinearPolicy policy,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows)
               : 0;
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
#if defined(NINFER_TIER_AMPERE)
    // A16 族的宽 T（t >= 33）在表检查之前被宽路线接管（见派发）：只占整面 BF16 暂存，四个
    // 目的地直接吃 GEMM 的输出。t <= 32 留在 32 列分块循环，不占工作区。
    if ((policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        max_tokens >= kNvfp4A16WideFirstToken) {
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(nvfp4_a16_wide_staged_bytes(Nvfp4N14336K5120::kOutputRows,
                                                             Nvfp4N14336K5120::kInputRows),
                                 256);
        return layout.peak_bytes(1);
    }
#endif
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return fused_route_table("unified/nvfp4_attn_input", width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_attn_input_workspace_capacity_bytes(policy, first, last)
                       : nvfp4_attn_input_workspace_capacity_bytes_own(policy, first, last);
        });
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
#if defined(NINFER_TIER_VOLTA)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：解码/校验宽度上 14336x5120 走 Volta QPN，
    // 不再落到 sm_70 的 SIMT 解码路线（实测纯解码比 v1.5 慢 12% 的来源之一）。宽 T 的
    // prefill 维持原有 A16/W4A4 路线，读的是同一份存放布局，因此加载期不需要重排。
    if (nvfp4_volta_takes(weight, x.ne[1])) {
        launch_nvfp4_attn_input_volta_qpn(x, weight, q, gate, k, v, stream);
        return;
    }
#endif
#if defined(NINFER_TIER_AMPERE)
    // A16 族的宽 T（t >= 33）：14336x5120 整面精确解码为 BF16 暂存，四个行段 GEMM 直接写
    // q / k / gate / v（父行顺序 [query; key; gate; value]）。t <= 32 留在下面的 32 列
    // 分块循环。
    if ((policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        x.ne[1] >= kNvfp4A16WideFirstToken) {
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 A16 wide attn_input_proj requires caller workspace");
        }
        auto scope = workspace->scope();
        const DeviceSpan stage =
            workspace->alloc_bytes(nvfp4_a16_wide_staged_bytes(weight.n, weight.k), 256);
        nvfp4_a16_wide_staged(weight, stage.data, stream);
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        const auto sub = [&](std::int32_t row_begin, std::int32_t rows) {
            Weight staged = weight;
            staged.qtype  = QType::BF16;
            staged.qdata  = static_cast<const std::uint8_t*>(stage.data) +
                            static_cast<std::uint64_t>(row_begin) *
                                static_cast<std::uint64_t>(weight.k) *
                                sizeof(__nv_bfloat16);
            staged.n      = rows;
            return staged;
        };
        bf16_dispatch(x, sub(0, kQRows), q, policy, stream);
        bf16_dispatch(x, sub(kQRows, kKvRows), k, policy, stream);
        bf16_dispatch(x, sub(kQRows + kKvRows, kQRows), gate, policy, stream);
        bf16_dispatch(x, sub(2 * kQRows + kKvRows, kKvRows), v, policy, stream);
        return;
    }
#endif
    if (fused_route_table("unified/nvfp4_attn_input", x.ne[1]) == LinearRouteTable::Unified) {
        unified::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
