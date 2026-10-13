#include "artifact/permute_nvfp4_qpn.h"

#include "artifact/schema.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace ninfer::artifact {
namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string("failed to ") + operation + ": " +
                            cudaGetErrorString(status));
    }
}

__device__ __forceinline__ std::uint8_t qpn_nibble(const std::uint8_t* row, int k) {
    const std::uint8_t byte = row[k >> 1];
    return (k & 1) == 0 ? byte & 0x0fu : byte >> 4;
}

// 一个 lane 写一个 32 行 × 16 k 的片段（8 字节）。片段内的行序是 QPN2 的 mma 片段映射
// （quadpair 的 8 行 × 4 组），k 序是 E2M1 的相邻码字对：artifact 的码字按 k 相邻打包，
// 而片段一次消费 (k, k+4) 对，所以 order[] 把 16 个 k 位置重排成那 8 对。
__global__ void prepack_qpn_kernel(const std::uint8_t* __restrict__ input_codes,
                                   const std::uint8_t* __restrict__ input_scales,
                                   std::uint8_t* __restrict__ output_codes,
                                   std::uint8_t* __restrict__ output_scales, int n, int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int groups         = k / 16;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * groups * 32;
    if (index >= count) { return; }

    const int lane                 = static_cast<int>(index % 32);
    const int group                = static_cast<int>((index / 32) % groups);
    const int tile                 = static_cast<int>(index / (static_cast<std::int64_t>(groups) * 32));
    const int row                  = tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    const std::uint8_t* source_row = input_codes + static_cast<std::int64_t>(row) * (k / 2);
    constexpr int order[16]        = {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15};
    std::uint8_t* destination      = output_codes + index * 8;
#pragma unroll
    for (int byte = 0; byte < 8; ++byte) {
        const int low_k  = group * 16 + order[2 * byte];
        const int high_k = group * 16 + order[2 * byte + 1];
        destination[byte] = static_cast<std::uint8_t>(qpn_nibble(source_row, low_k) |
                                                      (qpn_nibble(source_row, high_k) << 4));
    }

    // 尺度：同一个 128 行 × 4 组的 swizzle tile，但片段按 (tile, group) 线性排。
    const int scale_tile      = group / 4;
    const int scale_lane      = group & 3;
    const int row_inner       = row & 127;
    const int scales_per_m128 = k / 64;
    const std::int64_t scale_offset =
        static_cast<std::int64_t>((row / 128) * scales_per_m128 + scale_tile) * 512 +
        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
    output_scales[index] = input_scales[scale_offset];
}

} // namespace

bool nvfp4_qpn_prepack_supported(const WeightGeometry& geometry) noexcept {
    if (geometry.format != QType::NVFP4 ||
        geometry.layout != QuantLayout::BlockScaleK16M128x4) {
        return false;
    }
    if (geometry.shape.size() != 2) { return false; }
    const std::int64_t n = static_cast<std::int64_t>(geometry.shape[0]);
    const std::int64_t k = static_cast<std::int64_t>(geometry.shape[1]);
    return n > 0 && k > 0 && (n % 128) == 0 && (k % 64) == 0 && (k / 64) >= 8;
}

namespace {

// 在调用方的流上原地置换：读完整份输入到暂存、再写回（两个平面上都有读写交叠）。
void prepack_in_place(std::int32_t n, std::int32_t k, std::byte* codes, std::byte* scales,
                      cudaStream_t stream) {
    const auto code_bytes  = static_cast<std::size_t>(n) * static_cast<std::size_t>(k) / 2;
    const auto scale_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(k) / 16;
    std::byte* scratch = nullptr;
    check_cuda(cudaMalloc(&scratch, code_bytes + scale_bytes), "allocate QPN prepack scratch");
    auto* packed_codes  = reinterpret_cast<std::uint8_t*>(scratch);
    auto* packed_scales = packed_codes + code_bytes;
    const std::int64_t tuples = static_cast<std::int64_t>(n / 32) * (k / 16) * 32;
    prepack_qpn_kernel<<<static_cast<int>((tuples + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        reinterpret_cast<const std::uint8_t*>(codes),
        reinterpret_cast<const std::uint8_t*>(scales), packed_codes, packed_scales, n, k);
    check_cuda(cudaGetLastError(), "launch the QPN prepack");
    check_cuda(cudaMemcpyAsync(codes, packed_codes, code_bytes, cudaMemcpyDeviceToDevice, stream),
               "copy the prepacked codes");
    check_cuda(cudaMemcpyAsync(scales, packed_scales, scale_bytes, cudaMemcpyDeviceToDevice, stream),
               "copy the prepacked scales");
    check_cuda(cudaStreamSynchronize(stream), "finish the QPN prepack");
    check_cuda(cudaFree(scratch), "free the QPN prepack scratch");
}

} // namespace

void permute_nvfp4_to_qpn_planes(std::int32_t n, std::int32_t k, std::byte* codes,
                                 std::byte* scales, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (n % 128) != 0 || (k % 64) != 0 || (k / 64) < 8) {
        throw ArtifactError("an object without the NVFP4 QPN layout cannot be prepacked");
    }
    prepack_in_place(n, k, codes, scales, stream);
}

void permute_nvfp4_to_qpn(const WeightGeometry& geometry, std::byte* data, cudaStream_t stream) {
    if (!nvfp4_qpn_prepack_supported(geometry)) {
        throw ArtifactError("an object without the NVFP4 QPN layout cannot be prepacked");
    }
    const std::int32_t n = static_cast<std::int32_t>(geometry.shape[0]);
    const std::int32_t k = static_cast<std::int32_t>(geometry.shape[1]);
    prepack_in_place(n, k, data, data + geometry.scale_offset, stream);
}

} // namespace ninfer::artifact
