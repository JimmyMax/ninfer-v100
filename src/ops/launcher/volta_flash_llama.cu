// ninfer::ops::detail - direct launch of the vendored llama.cpp flash-attention kernel.
//
// The sm_70 prefill route stages the paged cache itself (ops/launcher/gqa_attention_volta_flash.cu)
// and then needs the kernel launched with launch_fattn's mma_f16 grid and shared-memory
// configuration and nothing else. This translation unit is the only one that sees the vendored
// tree: third_party/llama_cpp_fattn defines CUDA_CHECK, WARP_SIZE and friends, names that also
// exist under src/ops, so CMake scopes its include directory to this file.
//
// Why it exists next to the older bridge (ops/linear/gguf/ggml_fattn.cu, which drives the newer
// ggml fattn in third_party/ggml-quants): measured on W=2048/ctx=16384 int8, the newer ggml kernel
// costs 1.8 ns per query-key pair here against 0.75 ns for this pinned kernel, which is 2.4x -- the
// whole remaining prefill gap to the upstream reference build.

#include "fattn-mma-f16.cuh" // vendored; must precede the ninfer headers

#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Spelled out rather than taken from gqa_attention_decode.cuh's kGqaHeadDim: including that header
// would pull in ops/common/mma.cuh and ops/common/warp.cuh, whose WARP_SIZE collides with the
// vendored common.cuh. Guarded by the static_asserts in VoltaFlashTiling below.
constexpr int kHeadDim = 256;
constexpr int kDKQ     = 256;
constexpr int kDV      = 256;

// The kernel picks ncols2 from the GQA ratio, and this table mirrors that choice per registered
// geometry:
//   24q/4kv -> ratio 6: 6 % 8 != 0, 6 % 4 != 0, then DKQ <= 256 and 6 % 2 == 0 -> ncols2 = 2
//   16q/2kv -> ratio 8: 8 % 8 == 0                                            -> ncols2 = 8
// ncols1 is then chosen so ncols1*ncols2 == 32, the smallest tile Volta compiles.
template <typename Geometry>
struct VoltaFlashTiling;

template <>
struct VoltaFlashTiling<CausalD256H24Kv4> {
    static constexpr int ncols2 = 2;
    static constexpr int ncols1 = 32;
};

template <>
struct VoltaFlashTiling<CausalD256H16Kv2> {
    static constexpr int ncols2 = 8;
    static constexpr int ncols1 = 4;
};

template <typename Geometry>
struct VoltaFlashParams {
    static constexpr int kQHeads  = Geometry::QHeads;
    static constexpr int kKVHeads = Geometry::KVHeads;
    static constexpr int kGroup   = Geometry::GroupSize;
    static constexpr int kNcols2  = VoltaFlashTiling<Geometry>::ncols2;
    static constexpr int kNcols1  = VoltaFlashTiling<Geometry>::ncols1;
    static constexpr int kNcols   = kNcols1 * kNcols2;

    static_assert(kQHeads / kKVHeads == kGroup, "gqa ratio mismatch");
    static_assert(kGroup % kNcols2 == 0, "ncols2 must divide the GQA group");
    static_assert(kNcols >= 32, "Volta compiles no tile smaller than 32 columns");
};

struct FlashLaunchConfig {
    int nthreads      = 0;
    int nwarps        = 0;
    int nbatch_fa     = 0;
    std::size_t nbytes_shared = 0;
    int blocks_per_sm = 0;
    int nsm           = 0;
};

// One cached config per geometry: the shared-memory and occupancy figures depend only on the
// tiling, so they are resolved once per instantiation.
template <typename Geometry>
const FlashLaunchConfig& flash_launch_config() {
    using P = VoltaFlashParams<Geometry>;

    static const FlashLaunchConfig config = [] {
        FlashLaunchConfig c;

        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        c.nsm = prop.multiProcessorCount;

        const int cc = prop.major * 100 + prop.minor * 10;

        const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2(kDKQ, kDV, P::kNcols, cc);
        const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2(kDKQ, kDV, P::kNcols, cc);
        const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(kDKQ, kDV, P::kNcols, cc);
        const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg(kDKQ, kDV, P::kNcols, cc);
        const int nstages =
            ggml_cuda_fattn_mma_get_nstages(kDKQ, kDV, P::kNcols1, P::kNcols2, cc);

        c.nthreads  = ggml_cuda_fattn_mma_get_nthreads(kDKQ, kDV, P::kNcols, cc);
        c.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(kDKQ, kDV, P::kNcols, cc);
        c.nwarps    = c.nthreads / WARP_SIZE;

        const int cols_per_warp = std::min(P::kNcols, get_cols_per_warp(cc));

        const std::size_t shared_KV_1stage =
            std::size_t(c.nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const std::size_t shared_KV_2stage =
            std::size_t(c.nbatch_fa) * (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const std::size_t shared_Q  = std::size_t(P::kNcols) * (kDKQ / 2 + 4) * sizeof(half2);
        const std::size_t shared_mask =
            std::size_t(P::kNcols1) * (c.nbatch_fa / 2 + 4) * sizeof(half2);
        const std::size_t shared_combine =
            std::size_t(c.nwarps) * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
        const std::size_t shared_KV = nstages <= 1 ? shared_KV_1stage : shared_KV_2stage;

        c.nbytes_shared = std::max(shared_combine, Q_in_reg
                                                       ? std::max(shared_Q, shared_KV + shared_mask)
                                                       : shared_Q + shared_KV + shared_mask);

        auto kernel = flash_attn_ext_f16<kDKQ, kDV, P::kNcols1, P::kNcols2, false, false>;
        cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(c.nbytes_shared));
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c.blocks_per_sm, kernel, c.nthreads,
                                                      c.nbytes_shared);
        if (c.blocks_per_sm <= 0) { c.blocks_per_sm = 1; }
        return c;
    }();
    return config;
}

// One Q-block through the vendored kernel plus its stream-K fixup.
template <typename Geometry>
void launch_flash_block(const float* q_f32, const half* k_f16, const half* v_f16, const half* mask,
                        float* dst, float2* dst_meta, int tokens, int n_kv, int mask_stride,
                        float scale, cudaStream_t stream) {
    using P               = VoltaFlashParams<Geometry>;
    constexpr int kQHeads = P::kQHeads;
    constexpr int kKVHeads = P::kKVHeads;
    constexpr int kGroup   = P::kGroup;
    constexpr int kNcols1  = P::kNcols1;
    constexpr int kNcols2  = P::kNcols2;

    const FlashLaunchConfig& c = flash_launch_config<Geometry>();

    const int ntiles_x     = (tokens + kNcols1 - 1) / kNcols1;
    const int ntiles_z_gqa = (kGroup + kNcols2 - 1) / kNcols2;
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * kKVHeads;
    const int ntiles_KV    = (n_kv + c.nbatch_fa - 1) / c.nbatch_fa;

    const int max_blocks = c.blocks_per_sm * c.nsm;

    const int raw     = std::min(max_blocks, ntiles_KV * ntiles_dst);
    const int rounded = (raw / ntiles_dst) * ntiles_dst;
    const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    const int nblocks = loss <= 5 ? rounded : raw;

    const dim3 block_dim(WARP_SIZE, c.nwarps, 1);
    const dim3 blocks_num(nblocks, 1, 1);

    const std::int32_t nb01 = kDKQ * kQHeads * sizeof(float);
    const std::int32_t nb02 = kDKQ * sizeof(float);
    const std::int32_t nb11 = kDKQ * kKVHeads * sizeof(half);
    const std::int32_t nb12 = kDKQ * sizeof(half);
    const std::int32_t nb31 = mask_stride * sizeof(half);

    const uint3 ne01_fd = init_fastdiv_values(tokens);

    flash_attn_ext_f16<kDKQ, kDV, kNcols1, kNcols2, false, false>
        <<<blocks_num, block_dim, c.nbytes_shared, stream>>>(
            reinterpret_cast<const char*>(q_f32), reinterpret_cast<const char*>(k_f16),
            reinterpret_cast<const char*>(v_f16), reinterpret_cast<const char*>(mask), nullptr,
            nullptr, dst, dst_meta, scale, 0.0f, 1.0f, 1.0f, /*n_head_log2=*/16u, 0.0f, kDKQ,
            ne01_fd, kQHeads, 1, nb01, nb02, 0, kDKQ, n_kv, kKVHeads, 1, nb11, nb12, 0, nb11, nb12,
            0, tokens, 1, 1, nb31, 0, 0);

    if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
        const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
        const uint3 fd2 = init_fastdiv_values(ntiles_x);

        flash_attn_stream_k_fixup_uniform<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned) ntiles_dst, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kKVHeads, nblocks, kGroup, nblocks / ntiles_dst,
                fd0, fd1, fd2);
    } else if (ntiles_dst % nblocks != 0) {
        const int total_work = ntiles_KV * ntiles_dst;

        const uint3 fd_k_j_z_ne12 =
            init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd_k_j_z = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
        const uint3 fd_k_j   = init_fastdiv_values(ntiles_KV * ntiles_x);
        const uint3 fd_k     = init_fastdiv_values(ntiles_KV);

        flash_attn_stream_k_fixup_general<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned) nblocks, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kGroup, total_work, fd_k_j_z_ne12, fd_k_j_z, fd_k_j,
                fd_k);
    }
}

template <typename Geometry>
std::size_t meta_elements_impl(std::int32_t tokens) {
    using P                    = VoltaFlashParams<Geometry>;
    const FlashLaunchConfig& c = flash_launch_config<Geometry>();
    const int ntiles_x         = (tokens + P::kNcols1 - 1) / P::kNcols1;
    const int ntiles_dst       = ntiles_x * ((P::kGroup + P::kNcols2 - 1) / P::kNcols2) * P::kKVHeads;
    const int max_blocks       = c.blocks_per_sm * c.nsm;
    // Upper bound: the launch never uses more blocks than this.
    const int nblocks = std::max(max_blocks, ntiles_dst);
    return static_cast<std::size_t>(nblocks) * P::kNcols * (2 + kDV / 2);
}

} // namespace

std::size_t volta_flash_llama_meta_elements(std::int32_t q_heads, std::int32_t tokens) {
    if (q_heads == CausalD256H24Kv4::QHeads) {
        return meta_elements_impl<CausalD256H24Kv4>(tokens);
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        return meta_elements_impl<CausalD256H16Kv2>(tokens);
    }
    throw std::invalid_argument("volta flash llama: unsupported Q head geometry");
}

void volta_flash_llama_launch(const float* q_f32, const half* k_f16, const half* v_f16,
                             const half* mask, float* out_f32, float2* meta,
                             std::int32_t q_heads, std::int32_t kv_heads, std::int32_t tokens,
                             std::int32_t n_kv, std::int32_t mask_stride, float scale,
                             cudaStream_t stream) {
    (void)kv_heads;
    if (q_heads == CausalD256H24Kv4::QHeads) {
        launch_flash_block<CausalD256H24Kv4>(q_f32, k_f16, v_f16, mask, out_f32, meta, tokens,
                                             n_kv, mask_stride, scale, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        launch_flash_block<CausalD256H16Kv2>(q_f32, k_f16, v_f16, mask, out_f32, meta, tokens,
                                             n_kv, mask_stride, scale, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    throw std::invalid_argument("volta flash llama: unsupported Q head geometry");
}

} // namespace ninfer::ops::detail
