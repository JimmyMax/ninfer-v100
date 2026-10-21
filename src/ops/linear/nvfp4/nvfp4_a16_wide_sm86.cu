#include "ops/linear/nvfp4/nvfp4_a16_wide_sm86.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kThreads = 256;

__device__ __forceinline__ float decode_e4m3_scale_exact(std::uint8_t value) {
    __nv_fp8_e4m3 scale;
    scale.__x = value;
    return float(scale);
}

// 8 个 E2M1 码（一个 uint32）精确解码并缩放为 8 个 BF16：硬件转换到 float2，fp32 乘系数，
// 一对一次 __float22bfloat162_rn——与 nvfp4_cutlass_sm70.cu 的 FP16 版同一结构，数值语义
// 相同（E2M1 值与 E4M3 尺度在 BF16 里精确可表示，乘积只做一次舍入）。
__device__ __forceinline__ void decode_e2m1_word_bf16(std::uint32_t packed, float coefficient,
                                                     __nv_bfloat16 (&out)[8]) {
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        __nv_fp4x2_e2m1 pair;
        pair.__x             = static_cast<std::uint8_t>((packed >> (8 * byte)) & 0xffu);
        const float2 decoded = static_cast<float2>(pair);
        const float2 scaled = make_float2(decoded.x * coefficient, decoded.y * coefficient);
        *reinterpret_cast<__nv_bfloat162*>(out + 2 * byte) = __float22bfloat162_rn(scaled);
    }
}

// 与 sm_70 的 dequant_nvfp4_row_to_fp16 同一分段所有权与 BlockScaleK16M128x4 尺度 swizzle，
// 输出换为 BF16。一个线程持有一个 K16 尺度组的一半（8 个 k 值）：读 4 字节码字、1 字节
// E4M3 尺度，写 16 字节 BF16。
__global__ void dequant_nvfp4_row_to_bf16(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales, int n, int k,
                                          float inverse_weight_divisor,
                                          __nv_bfloat16* __restrict__ out) {
    const std::int64_t segment = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int segments_per_row = k / 8;
    const std::int64_t count   = static_cast<std::int64_t>(n) * segments_per_row;
    if (segment >= count) { return; }
    const int row         = static_cast<int>(segment / segments_per_row);
    const int row_segment = static_cast<int>(
        segment - static_cast<std::int64_t>(row) * segments_per_row);
    const int group      = row_segment / 2;
    const int group_half = row_segment & 1;
    const int scale_tile    = group / 4;
    const int scale_lane    = group & 3;
    const int row_inner     = row & 127;
    const int scales_per_m128 = k / 64;
    const std::int64_t scale_offset =
        static_cast<std::int64_t>((row / 128) * scales_per_m128 + scale_tile) * 512 +
        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
    const int byte_begin = group * 8 + group_half * 4;
    const std::uint8_t* source = codes + static_cast<std::int64_t>(row) * (k / 2) + byte_begin;
    const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(source);
    const float coefficient =
        decode_e4m3_scale_exact(scales[scale_offset]) * inverse_weight_divisor;
    __nv_bfloat16 values[8];
    decode_e2m1_word_bf16(packed, coefficient, values);
    auto* destination = reinterpret_cast<uint4*>(
        out + static_cast<std::int64_t>(row) * k + byte_begin * 2);
    *destination = *reinterpret_cast<const uint4*>(values);
}
} // namespace

std::size_t nvfp4_a16_wide_staged_bytes(std::int32_t n, std::int32_t k) {
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * sizeof(__nv_bfloat16);
}

void nvfp4_a16_wide_staged(const Weight& weight, void* plane, cudaStream_t stream) {
    // dense 契约：单除数平面、16 字节对齐的码/尺度指针、完整的几何。存放布局是
    // BlockScaleK16M128x4（码平面 + swizzled 尺度平面）；Volta QPN 重排布局需要片段序
    // 解码器，这里不支持。
    (void)validate_nvfp4_weight(weight, "nvfp4 a16 wide staged");
    if (weight.layout != QuantLayout::BlockScaleK16M128x4) {
        throw std::invalid_argument(
            "nvfp4 a16 wide staged: only the stored block-scale layout is supported");
    }
    if (plane == nullptr) {
        throw std::invalid_argument("nvfp4 a16 wide staged: null staging plane");
    }
    const std::int32_t n = weight.n;
    const std::int32_t k = weight.k;
    const std::int64_t count = static_cast<std::int64_t>(n) * (k / 8);
    const std::uint32_t blocks = static_cast<std::uint32_t>((count + kThreads - 1) / kThreads);
    dequant_nvfp4_row_to_bf16<<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), n, k,
        1.0F / weight.weight_scale_divisor, static_cast<__nv_bfloat16*>(plane));
}

} // namespace ninfer::ops::detail
