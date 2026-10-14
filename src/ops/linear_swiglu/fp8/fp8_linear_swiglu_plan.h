#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream);

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

// 本仓库 V100 增量（docs/v100/PLAN.md）：sm_70 的拆分式 QPN 路线（gate/up 的 FP32 暂存 +
// FP16 激活暂存由调用方提供）。
#if defined(NINFER_SM70)
void fp8_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        float* gate_scratch, float* up_scratch,
                                        void* activation_scratch, cudaStream_t stream);
[[nodiscard]] bool fp8_linear_swiglu_qpn_split_supported(std::int32_t k, std::int32_t t) noexcept;
#endif

} // namespace ninfer::ops::detail
