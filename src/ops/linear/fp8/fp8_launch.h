#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {
using Fp8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// 本仓库 V100 增量（docs/v100/PLAN.md）：FP8 的 Volta QPN（quadpair-split-N）路线。
// QPN 在 sm_70 上拥有受支持形状的全部宽度（宽 T 分块成 32 列）；预打包布局见
// fp8_prepack_sm70.h。
#if defined(NINFER_TIER_VOLTA)
inline constexpr std::int32_t kFp8VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kFp8VoltaQpnMaxTokens   = 32;
void fp8_stage_bf16_activation_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream);
void launch_fp8_volta_qpn(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream);
void launch_fp8_volta_qpn_fp16(const Tensor& x, const Weight& weight, const void* x_fp16,
                               Tensor& out, cudaStream_t stream);
[[nodiscard]] bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                           std::int32_t t) noexcept;

// 本仓库 V100 增量（docs/v100/PLAN.md）：GDN 输入投影（16384x5120）的 QPN 路线需要一块
// k x 32 的 FP16 激活暂存区（固定 320 KiB）。它不向 arena 记账（容量函数不知道权重布局），
// 而是在**模型加载时**（任何 CUDA graph 捕获之前）预留一次，运行期只读指针——捕获中绝不
// 调用 cudaMalloc，否则捕获会被 cudaErrorStreamCaptureInvalidated 打断。
// 实测同一形状/布局下：bf16 激活 548 GB/s vs 暂存 fp16 768 GB/s。
[[nodiscard]] void* fp8_gdn_qpn_stage_reserve();
[[nodiscard]] void* fp8_gdn_qpn_stage_buffer();

// 本仓库 V100 增量（docs/v100/PLAN.md）：QPN 读“先暂存好的 FP16 激活”比在内核里对 BF16 激活
// 做逐元素转换快 24–51%（实测 14336x5120 T=5：545 → 741 GB/s；16384x5120：548 → 768；
// 5120x17408：502 → 759；只有词汇表形状 248320x5120 无收益，故不暂存）。暂存区尺寸是
// k x t 的 FP16，256 字节对齐；调用方从自己的 arena 借，容量函数必须把它算进高水位。
[[nodiscard]] inline std::size_t fp8_qpn_stage_bytes(std::int32_t k, std::int32_t t) {
    if (k <= 0 || t <= 0) { return 0; }
    const std::size_t bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(t) * 2;
    return (bytes + 255) / 256 * 256;
}

// 本仓库 V100 增量：Volta QPN/CUTLASS 只在 A16 家族策略下接管——AllowA8/AllowA4 是上游自己
// 实现并测试的张量核路线（sm_70 上没有 FP8 张量核，那些路线会自行拒绝或走量化激活），不该被
// Volta 路线顶掉：两者的数值契约不同（A8 的激活是被量化过的）。
[[nodiscard]] inline bool fp8_volta_qpn_takes(LinearPolicy policy, std::int32_t n, std::int32_t k,
                                              std::int32_t t) noexcept {
    return !allows_a8(policy) && !allows_a4(policy) && fp8_volta_qpn_supported(n, k, t);
}
#endif
}
