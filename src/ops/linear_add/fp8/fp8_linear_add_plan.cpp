#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#include "ops/linear/fp8/fp8_launch.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearAddRoute : std::uint8_t {
    A16,
#ifdef NINFER_VOLTA_BUILD
    QpnResidual,
#endif
    A8,
};

Fp8LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != Fp8Residual6144Geometry::kOutputRows ||
        (input_rows != Fp8Residual6144Geometry::kInputRows &&
         input_rows != Fp8Residual17408Geometry::kInputRows)) {
        throw std::invalid_argument("fp8 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) {
#ifdef NINFER_VOLTA_BUILD
        return Fp8LinearAddRoute::QpnResidual;
#else
        return Fp8LinearAddRoute::A16;
#endif
    }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear_add: unsupported policy");
    }
    const std::int32_t first_a8 = input_rows == Fp8Residual6144Geometry::kInputRows ? 22 : 25;
    if (tokens >= first_a8) { return Fp8LinearAddRoute::A8; }
#ifdef NINFER_VOLTA_BUILD
    return Fp8LinearAddRoute::QpnResidual;
#else
    return Fp8LinearAddRoute::A16;
#endif
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kFp8LastSmallT) {
        const std::int32_t active = std::min(kFp8LastSmallT, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            fp8_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            fp8_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

#ifdef NINFER_VOLTA_BUILD
template <class Allocator>
Tensor allocate_projected(Allocator& allocator, std::int32_t output_rows, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {output_rows, tokens}, 256);
}

void launch_qpn_residual(const Tensor& x, const Weight& weight, Tensor& residual,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[1] <= 32) {
        const std::size_t stage_bytes =
            static_cast<std::size_t>(x.ne[0]) * static_cast<std::size_t>(x.ne[1]) * 2;
        const std::size_t aligned_used = (workspace.used() + 255) / 256 * 256;
        if (workspace.capacity() >= aligned_used && workspace.capacity() - aligned_used >= stage_bytes) {
            auto scope             = workspace.scope();
            const DeviceSpan stage = workspace.alloc_bytes(stage_bytes, 256);
            fp8_stage_bf16_activation_sm70(x, stage.data, stream);
            fp8_linear_add_qpn_fp16_launch(x, stage.data, weight, residual, stream);
            return;
        }
        fp8_linear_add_qpn_launch(x, weight, residual, stream);
        return;
    }
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected(workspace, weight.n, x.ne[1]);
    fp8_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    residual_add(projected, residual, stream);
}

std::size_t qpn_residual_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                         std::int32_t tokens) {
    // fp16 staging of the activation for the QPN route (T <= 32).
    WorkspaceLayoutBuilder stage_layout;
    (void)stage_layout.alloc_bytes(
        static_cast<std::size_t>(input_rows) * static_cast<std::size_t>(std::min(tokens, 32)) * 2,
        256);
    const std::size_t stage_bytes = stage_layout.peak_bytes(1);
    WorkspaceLayoutBuilder layout;
    if (tokens <= 32) { return stage_bytes; }
    (void)allocate_projected(layout, output_rows, tokens);
    const std::size_t linear_bytes =
        fp8_cutlass_sm70_workspace_bytes(output_rows, input_rows, tokens);
    (void)layout.alloc_bytes(linear_bytes, 256);
    return std::max(layout.peak_bytes(1), stage_bytes);
}
#endif

} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    const Fp8LinearAddRoute route = resolve_route(output_rows, input_rows, policy, max_tokens);
    if (route == Fp8LinearAddRoute::A8) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Fp8LinearAddRoute::QpnResidual) {
        return qpn_residual_workspace_bytes(output_rows, input_rows, max_tokens);
    }
#endif
    return 0;
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Fp8LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Fp8LinearAddRoute::QpnResidual) {
        launch_qpn_residual(x, weight, residual, workspace, stream);
        return;
    }
#endif
    fp8_linear_add_a8_launch(x, weight, residual, workspace, stream);
}

} // namespace ninfer::ops::detail
