#include "artifact/permute_q6k_dense.h"

#include "artifact/schema.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace ninfer::artifact {
namespace {

constexpr int kThreads    = 256;
constexpr int kBlockElems = 256;
constexpr int kBlockBytes = 210;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string("failed to ") + operation + ": " +
                            cudaGetErrorString(status));
    }
}

// u32 from a 2-byte-aligned address (ql/qh words are 210*b + even offsets).
__device__ __forceinline__ std::uint32_t u32_a2(const std::uint8_t* p) {
    const auto* h = reinterpret_cast<const std::uint16_t*>(p);
    return std::uint32_t(h[0]) | (std::uint32_t(h[1]) << 16);
}

// One 32-value slice per thread. The standard superblock holds the slice's 32 six-bit codes as
// 16 low nibbles in ql (4 bits each) plus 16 high fields in qh (2 bits each); the dense stream
// packs the same 32 codes into 192 contiguous bits, so code 10 straddles words 0/1 and code 21
// straddles words 1/2.
__global__ void q6k_dense_permute_kernel(const std::uint8_t* __restrict__ src,
                                         std::uint8_t* __restrict__ dst, int n, int k) {
    const int slices_per_row = k / 32;
    const std::int64_t total = std::int64_t(n) * slices_per_row;
    const int row_bytes      = k / kBlockElems * kBlockBytes;
    const int scale_off      = 3 * k / 4;
    const int d_off          = scale_off + k / 16;
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += std::int64_t(gridDim.x) * blockDim.x) {
        const int s              = int(i % slices_per_row);
        const int row            = int(i / slices_per_row);
        const std::uint8_t* rsrc = src + std::int64_t(row) * row_bytes;
        std::uint8_t* rdst        = dst + std::int64_t(row) * row_bytes;
        const int b = s / 8, u = s % 8, h = u / 4, q = u % 4;
        const std::uint8_t* blk = rsrc + b * kBlockBytes;
        const std::uint8_t* ql  = blk + 64 * h + 32 * (q & 1);
        const std::uint8_t* qh  = blk + 128 + 32 * h;
        const std::uint32_t qlw[8] = {u32_a2(ql),     u32_a2(ql + 4),  u32_a2(ql + 8),
                                      u32_a2(ql + 12), u32_a2(ql + 16), u32_a2(ql + 20),
                                      u32_a2(ql + 24), u32_a2(ql + 28)};
        const std::uint32_t qhw[8] = {u32_a2(qh),     u32_a2(qh + 4),  u32_a2(qh + 8),
                                      u32_a2(qh + 12), u32_a2(qh + 16), u32_a2(qh + 20),
                                      u32_a2(qh + 24), u32_a2(qh + 28)};
        std::uint64_t W[3] = {0, 0, 0};
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const std::uint32_t lo   = (qlw[j >> 2] >> ((j & 3) * 8 + 4 * (q >> 1))) & 0xFu;
            const std::uint32_t hi   = (qhw[j >> 2] >> ((j & 3) * 8 + 2 * q)) & 3u;
            const std::uint32_t code = (hi << 4) | lo;
            const int bit6           = 6 * j;
            const int w              = bit6 >> 6;
            const int sb             = bit6 & 63;
            W[w] |= std::uint64_t(code) << sb;
            if (sb > 58) { W[w + 1] |= std::uint64_t(code) >> (64 - sb); }
        }
        std::uint32_t* out_w = reinterpret_cast<std::uint32_t*>(rdst + 24 * s);
        out_w[0] = std::uint32_t(W[0]);
        out_w[1] = std::uint32_t(W[0] >> 32);
        out_w[2] = std::uint32_t(W[1]);
        out_w[3] = std::uint32_t(W[1] >> 32);
        out_w[4] = std::uint32_t(W[2]);
        out_w[5] = std::uint32_t(W[2] >> 32);
        rdst[scale_off + 2 * s]     = blk[192 + 8 * h + 2 * q];
        rdst[scale_off + 2 * s + 1] = blk[192 + 8 * h + 2 * q + 1];
        if (u == 0) {
            rdst[d_off + 2 * b]     = blk[208];
            rdst[d_off + 2 * b + 1] = blk[209];
        }
    }
}

} // namespace

bool q6k_dense_prepack_supported(const WeightGeometry& geometry) noexcept {
    if (geometry.format != QType::GGUF_Q6_K || geometry.layout != QuantLayout::GgufBlocks) {
        return false;
    }
    if (geometry.shape.size() != 2) { return false; }
    const std::int64_t n = geometry.shape[0];
    const std::int64_t k = geometry.shape[1];
    // The dense value stream must be 8-byte aligned at every row start: row_bytes = 210 * k/256
    // is a multiple of 8 exactly when k/256 is a multiple of 4.
    return n > 0 && k > 0 && (k % 1024) == 0;
}

namespace {

void permute_in_place(std::int32_t n, std::int32_t k, std::byte* data, cudaStream_t stream) {
    const std::int64_t row_bytes = std::int64_t(k) / kBlockElems * kBlockBytes;
    const std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(row_bytes);
    std::byte* scratch = nullptr;
    check_cuda(cudaMalloc(&scratch, bytes), "allocate the Q6_K dense prepack scratch");
    const std::int64_t total = std::int64_t(n) * (k / 32);
    const int blocks         =
        int(std::min<std::int64_t>((total + kThreads - 1) / kThreads, 8192));
    q6k_dense_permute_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const std::uint8_t*>(data), reinterpret_cast<std::uint8_t*>(scratch), n,
        k);
    check_cuda(cudaGetLastError(), "launch the Q6_K dense prepack");
    check_cuda(cudaMemcpyAsync(data, scratch, bytes, cudaMemcpyDeviceToDevice, stream),
               "copy the dense prepacked Q6_K rows");
    check_cuda(cudaStreamSynchronize(stream), "finish the Q6_K dense prepack");
    check_cuda(cudaFree(scratch), "free the Q6_K dense prepack scratch");
}

} // namespace

void permute_q6k_to_dense(const WeightGeometry& geometry, std::byte* data, cudaStream_t stream) {
    if (!q6k_dense_prepack_supported(geometry)) {
        throw ArtifactError("an object without the GgufBlocks Q6_K layout cannot be dense-prepacked");
    }
    permute_in_place(static_cast<std::int32_t>(geometry.shape[0]),
                     static_cast<std::int32_t>(geometry.shape[1]), data, stream);
}

} // namespace ninfer::artifact
