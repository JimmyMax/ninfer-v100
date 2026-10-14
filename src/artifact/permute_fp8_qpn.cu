#include "artifact/permute_fp8_qpn.h"

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

// 目标位置：QPN8 的流是 [tile(N/32)][block(K/128)][slice(8)][lane(32)] x 16 字节。行序与 NVFP4
// 的 QPN2 相同（quadpair 的 8 行 x 4 组映射），只是这里一行只有一段 16 字节的切片；下面的
// index 正是那个流位置，lane 由 index 反推出它要搬的那一行。
//
// 一个线程搬一个 16 字节切片：读行主序的 (row, block*128 + slice*16)，写到流里的第 index 段。
__global__ void prepack_qpn_kernel(const std::uint8_t* __restrict__ input,
                                   std::uint8_t* __restrict__ output, int n, int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * (k / 128) * 8 * 32;
    if (index >= count) { return; }

    const int lane  = static_cast<int>(index % 32);
    const int slice = static_cast<int>((index / 32) % 8);
    const int block = static_cast<int>((index / (32 * 8)) % (k / 128));
    const int tile  = static_cast<int>(index / (static_cast<std::int64_t>(k / 128) * 8 * 32));
    const int row   = tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);

    const std::uint8_t* source = input + static_cast<std::int64_t>(row) * k + block * 128 + slice * 16;
    *reinterpret_cast<uint4*>(output + index * 16) = *reinterpret_cast<const uint4*>(source);
}

// 在调用方的流上原地置换：读完整份输入到暂存、再写回（码字平面内部有读写交叠）。尺度平面
// 不动。
void prepack_in_place(std::int32_t n, std::int32_t k, std::byte* codes, cudaStream_t stream) {
    const auto code_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    std::byte* scratch    = nullptr;
    check_cuda(cudaMalloc(&scratch, code_bytes), "allocate the FP8 QPN prepack scratch");
    const std::int64_t tuples =
        static_cast<std::int64_t>(n / 32) * (k / 128) * 8 * 32;
    prepack_qpn_kernel<<<static_cast<int>((tuples + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        reinterpret_cast<const std::uint8_t*>(codes), reinterpret_cast<std::uint8_t*>(scratch), n, k);
    check_cuda(cudaGetLastError(), "launch the FP8 QPN prepack");
    check_cuda(cudaMemcpyAsync(codes, scratch, code_bytes, cudaMemcpyDeviceToDevice, stream),
               "copy the prepacked FP8 codes");
    check_cuda(cudaStreamSynchronize(stream), "finish the FP8 QPN prepack");
    check_cuda(cudaFree(scratch), "free the FP8 QPN prepack scratch");
}

} // namespace

bool fp8_qpn_prepack_supported(const WeightGeometry& geometry) noexcept {
    if (geometry.format != QType::FP8_E4M3FN_ROW_BF16 ||
        geometry.layout != QuantLayout::RowScale) {
        return false;
    }
    if (geometry.shape.size() != 2) { return false; }
    const std::int64_t n = static_cast<std::int64_t>(geometry.shape[0]);
    const std::int64_t k = static_cast<std::int64_t>(geometry.shape[1]);
    return n > 0 && k > 0 && (n % 32) == 0 && (k % 128) == 0 && (k / 128) >= 8;
}

void permute_fp8_to_qpn_planes(std::int32_t n, std::int32_t k, std::byte* codes,
                               cudaStream_t stream) {
    if (n <= 0 || k <= 0 || (n % 32) != 0 || (k % 128) != 0 || (k / 128) < 8) {
        throw ArtifactError("an object without the FP8 QPN layout cannot be prepacked");
    }
    prepack_in_place(n, k, codes, stream);
}

void permute_fp8_to_qpn(const WeightGeometry& geometry, std::byte* data, cudaStream_t stream) {
    if (!fp8_qpn_prepack_supported(geometry)) {
        throw ArtifactError("an object without the FP8 QPN layout cannot be prepacked");
    }
    prepack_in_place(static_cast<std::int32_t>(geometry.shape[0]),
                     static_cast<std::int32_t>(geometry.shape[1]), data, stream);
}

} // namespace ninfer::artifact
