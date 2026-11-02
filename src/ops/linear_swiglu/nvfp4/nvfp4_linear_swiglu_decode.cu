#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N34816K5120;
using Schedule =
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>;

constexpr int kIntermediate = Geometry::kOutputRows / 2;
static_assert(Schedule::kRowsPerWarp == 2);
static_assert((Schedule::kWarpsPerCta % 4) == 0);
static_assert((128 % Schedule::kWarpsPerCta) == 0);
static_assert((kIntermediate % Schedule::kWarpsPerCta) == 0);

__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kMinBlocksPerSm) void nvfp4_linear_swiglu_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                                                const std::
                                                                    uint8_t* __restrict__ codes,
                                                                const std::
                                                                    uint8_t* __restrict__ scales,
                                                                float inverse_weight_divisor,
                                                                __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4GemvSharedStorage<Geometry, Schedule> shared;
    // e2m1 decode LUT in shared memory: 16 entries (4-bit code -> float value). Replaces the
    // hardware intrinsic (a 4-6 instruction ALU sequence on Ada) with a single shared load.
    __shared__ float e2m1_lut[16];
    if (threadIdx.x < 16) {
        const float values[16] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
                                  -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
        e2m1_lut[threadIdx.x]   = values[threadIdx.x];
    }
    __syncthreads();
    constexpr int kCtasPerM128                    = 128 / Schedule::kWarpsPerCta;
    const int block                               = static_cast<int>(blockIdx.x);
    const int m_tile                              = block / kCtasPerM128;
    const int cta_in_tile                         = block - m_tile * kCtasPerM128;
    const int lane                                = static_cast<int>(threadIdx.x) & 31;
    const int warp                                = static_cast<int>(threadIdx.x) >> 5;
    const int flat_pair                           = cta_in_tile * Schedule::kWarpsPerCta + warp;
    const int rmod                                = flat_pair >> 2;
    const int quartile                            = flat_pair & 3;
    const int gate_row                            = m_tile * 128 + rmod + quartile * 32;
    const int parent_rows[Schedule::kRowsPerWarp] = {gate_row, gate_row + kIntermediate};

    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    constexpr int kPhases         = Geometry::kInputRows / kValuesPerPhase;
    constexpr int kGroupsPerLane  = Schedule::kValuesPerLane / 16;
    const auto* activation_pairs  = reinterpret_cast<const std::uint32_t*>(x);
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        float coefficients[Schedule::kRowsPerWarp][kGroupsPerLane];
        Nvfp4CodePack<Schedule::kValuesPerLane> row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            load_nvfp4_coefficients<Geometry, Schedule>(
                scales, shared, parent_rows[local_row], warp * Schedule::kRowsPerWarp + local_row,
                phase, lane, inverse_weight_divisor, coefficients[local_row]);
            const std::int64_t code_offset =
                static_cast<std::int64_t>(parent_rows[local_row]) * Geometry::kCodeBytesPerRow +
                phase * (kValuesPerPhase / 2) + lane * (Schedule::kValuesPerLane / 2);
            row_codes[local_row] = load_nvfp4_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(
                codes + code_offset);
        }
#pragma unroll
        for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
            const int activation_index =
                phase * (kValuesPerPhase / 2) + lane * Schedule::kPairsPerLane + pair;
            const float2 activation = bf16x2_bits_to_float2(activation_pairs[activation_index]);
            const int group         = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                const float code_x        = e2m1_lut[packed & 0x0f];
                const float code_y        = e2m1_lut[packed >> 4];
                const float coefficient   = coefficients[local_row][group];
                constexpr int kChainMask  = Schedule::kAccumulatorChains - 1;
                const int chain           = (2 * pair) & kChainMask;
                accumulators[local_row][chain] =
                    fmaf(code_x * coefficient, activation.x, accumulators[local_row][chain]);
                accumulators[local_row][(chain + 1) & kChainMask] = fmaf(
                    code_y * coefficient, activation.y,
                    accumulators[local_row][(chain + 1) & kChainMask]);
            }
        }
    }

    float gate = 0.0F;
    float up   = 0.0F;
#pragma unroll
    for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
        gate += accumulators[0][chain];
        up += accumulators[1][chain];
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
    if (lane == 0) { out[gate_row] = __float2bfloat16_rn(silu(gate) * up); }
}

} // namespace

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    constexpr int kBlocks = kIntermediate / Schedule::kWarpsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_linear_swiglu_decode_kernel<<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
