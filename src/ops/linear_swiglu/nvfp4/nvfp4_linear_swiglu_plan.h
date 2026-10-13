#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#if defined(NINFER_SM70)
#    include "ops/linear/nvfp4/nvfp4_launch.h" // 本仓库 V100 增量：QPN 常量与声明
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

#if defined(NINFER_SM70)
// 本仓库 V100 增量（docs/v100/PLAN.md）：fork 的 Volta QPN 融合 SwiGLU。
// 权重是 artifact 原样的 gate_up (34816 x k)；输出是 silu(gate) * up 的 (t, 17408)。
void nvfp4_linear_swiglu_volta_qpn_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
[[nodiscard]] bool nvfp4_linear_swiglu_volta_qpn_supported(std::int32_t k,
                                                           std::int32_t t) noexcept;

void nvfp4_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          float* gate_scratch, void* activation_scratch,
                                          cudaStream_t stream);
[[nodiscard]] bool nvfp4_linear_swiglu_qpn_split_supported(std::int32_t k,
                                                           std::int32_t t) noexcept;
#endif

} // namespace ninfer::ops::detail
