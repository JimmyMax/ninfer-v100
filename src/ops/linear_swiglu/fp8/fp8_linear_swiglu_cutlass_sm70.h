#pragma once

// Volta (sm_70) tensor-core path for the FP8 MLP gate/up GEMM, via NVIDIA CUTLASS's Sm70
// (FP16 mma.sync.m8n8k4) template GEMM. Two-phase: dequantize the row-scaled E4M3 weights
// (code byte * per-output-row BF16 scale) to a scratch FP16 buffer, cast bf16 activations to
// FP16, then run CUTLASS's stock Gemm with a direct bf16 epilogue output. Mirrors
// q4_linear_swiglu_cutlass_sm70.cu exactly -- see docs/v100.md for why groupwise's
// dequant-once-then-CUTLASS route beats every fused-dequant kernel at wide T.

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t fp8_linear_swiglu_cutlass_workspace_bytes(std::int32_t gate_up_rows, std::int32_t k,
                                                       std::int32_t cols);

void fp8_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& gate_up_out,
                                           WorkspaceArena& ws, cudaStream_t stream);

// 本仓库 V100 增量（docs/v100/fp8-swiglu-wide-t-*.md）：宽 T（t>=33）的 FP8 SwiGLU。与上面共用同一
// 个 sm_70 mma 内核，但 CUTLASS 的输出是 FP32 平面，行缩放在 FP32 里做，SwiGLU 直接从该平面读出
// 并写回 BF16 的 out([n/2, t]，token-major)。契约上与上游 A16 阶梯一致（激活前保持 FP32）。
std::size_t fp8_linear_swiglu_cutlass_f32_workspace_bytes(std::int32_t gate_up_rows,
                                                          std::int32_t k, std::int32_t cols);

void fp8_linear_swiglu_cutlass_sm70_launch_f32(const Tensor& x, const Weight& w, Tensor& out,
                                               WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
