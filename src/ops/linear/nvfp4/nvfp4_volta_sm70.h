#pragma once

// 本仓库 V100 增量（docs/v100/PLAN.md）：NVFP4 投影在 sm_70 上的 Volta QPN 路由。
//
// 上游的 Volta NVFP4 路线（ops/linear/sm70 的 small-T 解码器）实测只有 85 GB/s / DRAM 9.5%，
// 而同一形状的 fp8/q4/q5/q8 都在 480–620 GB/s。这里在 QType::NVFP4 的分发前接上 fork 的
// QPN 内核（nvfp4_volta_qpn_gemm.cuh），它按 Volta 片段顺序直接消费 NVFP4 码。
//
// 阶段 1 只覆盖解码/校验宽度（tokens <= kNvfp4VoltaQpnMaxTokens = 32）；prefill 的宽 T
// 路径见 nvfp4_volta_mma_gemm.cuh（阶段 2）。

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if defined(NINFER_SM70)
// 这个权重在这个宽度上是否走 Volta QPN：NVFP4 + 可读布局 + 解码宽度 + 形状受 QPN 支持。
[[nodiscard]] bool nvfp4_volta_takes(const Weight& w, std::int32_t tokens) noexcept;

// 单个 destination 的 QPN 投影（out = x · wᵀ，contiguous BF16 输出）。
void nvfp4_volta_project(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

// BF16 → FP16 的激活暂存（QPN 的 fp16 激活变体与 residual 输出策略需要）。
void nvfp4_volta_stage_bf16_fp16_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
