#include "ops/linear/fp8/fp8_prepack_sm70.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/fp8/fp8_format.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Destination is [N/32 tile][K/128 block][8 sixteen-byte slices][lane].  At consumption,
// one warp therefore issues one coalesced 512-byte transaction sequence for each slice while
// every lane still receives the same 128 adjacent K values as the portable row-major layout.
__global__ void prepack_fp8_qpn_kernel(const std::uint8_t* __restrict__ input,
                                       std::uint8_t* __restrict__ output, int n, int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int blocks         = k / 128;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * blocks * 8 * 32;
    if (index >= count) { return; }

    const int lane  = static_cast<int>(index % 32);
    const int slice = static_cast<int>((index / 32) % 8);
    const int block = static_cast<int>((index / (32 * 8)) % blocks);
    const int tile  = static_cast<int>(index / (static_cast<std::int64_t>(blocks) * 8 * 32));
    const int row   = tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);

    const auto* source = reinterpret_cast<const uint4*>(
        input + static_cast<std::int64_t>(row) * k + block * 128 + slice * 16);
    reinterpret_cast<uint4*>(output)[index] = *source;
}

} // namespace

void fp8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream) {
    const Fp8WeightGeometry geometry = validate_fp8_weight(weight, "FP8 QPN prepack");
    if (weight.n % 32 != 0 || weight.k % 128 != 0) {
        throw std::invalid_argument("FP8 QPN prepack: shape must tile N32 K128");
    }
    DeviceBuffer scratch(static_cast<std::size_t>(geometry.code_plane_bytes));
    const std::int64_t tuples = static_cast<std::int64_t>(weight.n / 32) * (weight.k / 128) * 8 * 32;
    prepack_fp8_qpn_kernel<<<static_cast<int>((tuples + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), static_cast<std::uint8_t*>(scratch.p),
        weight.n, weight.k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.qdata), scratch.p,
                               static_cast<std::size_t>(geometry.code_plane_bytes),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    weight.layout = QuantLayout::VoltaQpnPrepacked;
}

} // namespace ninfer::ops::detail
