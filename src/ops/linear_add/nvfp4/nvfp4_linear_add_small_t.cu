#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_simt.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
#if defined(NINFER_TIER_AMPERE)
    // t<=5 的 17408 几何（down 投影）与 swiglu small_t 同病：TokenPacked 把激活常驻寄存器，
    // sm_89 上占用率被压到 ~40%（ncu 实测 85 寄存器/线程）。SharedPhase 走 shared 暂存，
    // 8 warp 提高延迟隐藏。6144 几何与其它 t 不变；Volta 层级保持原调度（QPN 先接管，
    // 本文件仍被编译，保持 sm_70 二进制不变）。
    static constexpr bool kSharedPhaseSmallT =
        ActiveTokens <= 5 && Geometry::kInputRows == 17408;
    using Schedule = Nvfp4SimtSchedule<
        (ActiveTokens <= 16 && ActiveTokens >= (Geometry::kInputRows == 6144 ? 14 : 8)) ? 16
            : (kSharedPhaseSmallT ? 8 : 4),
        1, 2, (ActiveTokens >= 17 && ActiveTokens <= 20) ? 8 : 16, ActiveTokens, 1,
        kSharedPhaseSmallT ? Nvfp4SimtActivationAccess::SharedPhase
                           : Nvfp4SimtActivationAccess::TokenPacked,
        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous,
        1>;
#else
    using Schedule = Nvfp4SimtSchedule<
        (ActiveTokens <= 16 && ActiveTokens >= (Geometry::kInputRows == 6144 ? 14 : 8)) ? 16 : 4, 1,
        2, (ActiveTokens >= 17 && ActiveTokens <= 20) ? 8 : 16, ActiveTokens, 1,
        Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default,
        1, Nvfp4SimtBlockOrder::RowsContiguous, 1>;
#endif
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const float inverse       = 1.0F / weight.weight_scale_divisor;
    auto* output              = static_cast<__nv_bfloat16*>(residual.data);
    nvfp4_simt_kernel<Geometry, ActiveTokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, 2 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
constexpr auto make_launchers() {
    return make_launchers<Geometry>(std::make_index_sequence<32 - 2 + 1>{});
}

constexpr auto kResidual6144Launchers  = make_launchers<Nvfp4N5120K6144>();
constexpr auto kResidual17408Launchers = make_launchers<Nvfp4N5120K17408>();

} // namespace

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - 2);
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        kResidual6144Launchers[index](x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        kResidual17408Launchers[index](x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
