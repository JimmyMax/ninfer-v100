#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
using Fp8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// 本仓库 V100 增量（docs/v100/PLAN.md）：FP8 的 Volta QPN（quadpair-split-N）路线。
// QPN 在 sm_70 上拥有受支持形状的全部宽度（宽 T 分块成 32 列）；预打包布局见
// fp8_prepack_sm70.h。
#if defined(NINFER_SM70)
inline constexpr std::int32_t kFp8VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kFp8VoltaQpnMaxTokens   = 32;
void fp8_stage_bf16_activation_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream);
void launch_fp8_volta_qpn(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream);
void launch_fp8_volta_qpn_fp16(const Tensor& x, const Weight& weight, const void* x_fp16,
                               Tensor& out, cudaStream_t stream);
[[nodiscard]] bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                           std::int32_t t) noexcept;

// 本仓库 V100 增量：Volta QPN/CUTLASS 只在 A16 家族策略下接管——AllowA8/AllowA4 是上游自己
// 实现并测试的张量核路线（sm_70 上没有 FP8 张量核，那些路线会自行拒绝或走量化激活），不该被
// Volta 路线顶掉：两者的数值契约不同（A8 的激活是被量化过的）。
[[nodiscard]] inline bool fp8_volta_qpn_takes(LinearPolicy policy, std::int32_t n, std::int32_t k,
                                              std::int32_t t) noexcept {
    return !allows_a8(policy) && !allows_a4(policy) && fp8_volta_qpn_supported(n, k, t);
}
#endif
}
