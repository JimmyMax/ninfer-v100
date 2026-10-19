#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"
#if defined(NINFER_TIER_VOLTA)
#    include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh" // 本仓库 V100 增量
#endif

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule        = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    auto* output          = static_cast<__nv_bfloat16*>(residual.data);
    const Fp8ContiguousOutput destination{output, Geometry::kOutputRows};
    const Fp8GemvIdentityRows rows{};
    const Fp8AddResidualEpilogue epilogue{output, Geometry::kOutputRows};
    fp8_gemv_kernel<Geometry, Schedule, Fp8ContiguousOutput, Fp8GemvIdentityRows, false,
                    Fp8AddResidualEpilogue><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), destination, rows, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream) {
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launch<Fp8N5120K6144>(x, weight, residual, stream);
        return;
    case Fp8GeometryId::N5120K17408:
        launch<Fp8N5120K17408>(x, weight, residual, stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}

#if defined(NINFER_TIER_VOLTA)
// 本仓库 V100 增量（docs/v100/PLAN.md）：QPN 内核 + 残差 epilogue（投影结果直接加进 residual）。
void fp8_linear_add_qpn_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                               cudaStream_t stream) {
    launch_fp8_volta_qpn_with_output(
        x, weight,
        Fp8ResidualOutput{static_cast<__nv_bfloat16*>(residual.data), weight.n}, weight.n, stream);
}

void fp8_linear_add_qpn_fp16_launch(const Tensor& x, const void* x_fp16, const Weight& weight,
                                    Tensor& residual, cudaStream_t stream) {
    launch_fp8_volta_qpn_with_fp16_activation(
        x, weight, static_cast<const half*>(x_fp16),
        Fp8ResidualOutput{static_cast<__nv_bfloat16*>(residual.data), weight.n}, weight.n, stream);
}
#endif

} // namespace ninfer::ops::detail
