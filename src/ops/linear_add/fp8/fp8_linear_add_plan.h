#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                  std::int32_t input_rows,
                                                                  LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream);
void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);
void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream);
#ifdef NINFER_VOLTA_BUILD
void fp8_linear_add_qpn_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                               cudaStream_t stream);
// Same as fp8_linear_add_qpn_launch with the activation pre-staged to fp16 (x_fp16 holds
// x.ne[0] * x.ne[1] halves). Converting inside the kernel costs 8 cvt per 16-byte activation
// load and made the residual shapes run at ~360 GB/s instead of ~530.
void fp8_linear_add_qpn_fp16_launch(const Tensor& x, const void* x_fp16, const Weight& weight,
                                    Tensor& residual, cudaStream_t stream);
#endif

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
