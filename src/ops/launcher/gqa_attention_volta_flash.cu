// ninfer::ops::detail - Volta (sm_70) flash-attention prefill route.
//
// Drives the vendored llama.cpp tensor-core flash attention (third_party/ggml-quants, reached
// through ops/linear/gguf/ggml_fattn.h) from the cached causal attention op. The kernel reads K/V
// as strided FP16 and Q as strided FP32, while this repo stores BF16 Q/out and a paged BF16 or
// INT8-G64 KV cache. Rather than inject page lookups into the kernel's hot loop, this file stages
// per layer: gather the visible key range into a contiguous FP16 buffer once, then per Q-block
// rotate/convert Q, build the causal mask from the device-side positions and run the kernel. The
// caller appends K/V to the pages first, exactly like the sm_70 prompt route does.
//
// Why it exists: the int8 prompt route evaluates QK^T/PV on CUDA cores (DP4A), the flash kernel on
// tensor cores. At 27K prefill the upstream reference build spends 3.93 s in flash attention
// against this line's 7.28 s of prompt attention, and this route is what closes that gap when it is
// affordable.
//
// It is OPT-IN (NINFER_ATTN_VOLTA_FLASH=1, and NINFER_ATTN_VOLTA_FLASH_INT8=1 for the int8 cache)
// because at the shapes this engine actually calls it is not yet a win: prefill is chunked, and
// every call has to stage its whole visible prefix, so the staging and mask traffic repeat per
// chunk instead of once per layer. Correctness is not in question -- the int8 and bf16 prompt
// oracles both pass with the route forced on.
//
// Cost: the gathered planes are `2 * round_up(keys) * kv_heads * 256 * 2` bytes, so the route only
// applies while the visible key range stays inside kFlashMaximumVisibleKeys; past that the prompt
// route runs and no staging is allocated at all.

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/linear/gguf/ggml_fattn.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim = 256;

// The kernel reads whole key tiles (FATTN_KQ_STRIDE = 256 in the vendored launcher), so the key
// extent handed to it is padded up. Padding rows are gathered as zero and fully masked, so they
// contribute nothing.
constexpr int kKeyPad = 256;

// Query rows per flash launch. Bounding the block bounds the mask that has to be built and lets an
// early block attend over a shorter key range than a later one.
constexpr int kFlashQBlockTokens = 128;

// The mask covers whole rows of 8 halves, and the bridge wants `slots % 256 == 0`.
constexpr int kFlashMaskRowPad = 8;

// Staging cap: the gathered K/V planes cost 2 bytes per (key, kv head, dim), so 32K keys at
// kv_heads = 4 come to 134 MiB. Longer key ranges keep the prompt route (and allocate nothing).
constexpr std::int32_t kFlashMaximumVisibleKeys = 32768;

constexpr int round_up_keys(int n) { return ((n + kKeyPad - 1) / kKeyPad) * kKeyPad; }

bool volta_flash_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_ATTN_VOLTA_FLASH");
        return value != nullptr && value[0] != '0';
    }();
    return enabled;
}

// The INT8 cache is correct (the int8 prompt oracle passes) but slower than this line's int8 prompt
// route at prefill widths: staging the paged int8 planes into contiguous FP16 plus the shared flash
// launcher costs more than the prompt kernel's direct DP4A read of the pages (42.2 ms vs 41.3 ms at
// W=2048/ctx=16384, 807 vs 906 tok/s end to end). It stays opt-in until the launcher is leaner.
bool volta_flash_int8_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_ATTN_VOLTA_FLASH_INT8");
        return value != nullptr && value[0] != '0';
    }();
    return enabled;
}

// `table_rows` is null for the single-sequence entry, where the first (only) table row is used.
__device__ __forceinline__ const std::int32_t* select_block_table(
        const std::int32_t* block_tables, const std::int32_t* table_rows, int logical_pages) {
    const std::int32_t row = table_rows == nullptr ? 0 : table_rows[0];
    return block_tables + static_cast<std::int64_t>(row) * logical_pages;
}

// Paged cache -> contiguous FP16 in the layout the bridge wants: element (d, key, head) at
// head * (n_kv_padded * kHeadDim) + key * kHeadDim + d.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_kernel(
        const __nv_bfloat16* __restrict__ k_pages, const half* __restrict__ v_pages,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded) {
    const int key = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (key >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (key + static_cast<std::int64_t>(n_kv_padded) * h);

    // Padding rows are written as zero rather than left uninitialised: the mask gives them weight
    // zero, and 0 * NaN would still be NaN.
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t src = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);

    k_out[dst] = __float2half(__bfloat162float(k_pages[src]));
    v_out[dst] = v_pages[src];
}

// Paged INT8-G64 -> contiguous FP16. Dequantizing once here keeps the flash kernel as the
// already-qualified arithmetic boundary.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_i8_kernel(
        const std::int8_t* __restrict__ k_pages, const std::int8_t* __restrict__ v_pages,
        const half* __restrict__ k_scales, const half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded) {
    const int key = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (key >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (key + static_cast<std::int64_t>(n_kv_padded) * h);
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t code = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);
    const int group         = d / kKVCacheInt8Group;
    const std::int64_t scale =
        paged_kv_element_offset<kKVCacheInt8Groups, kKVHeads>(block_table, h, key, group);
    k_out[dst] = __float2half(static_cast<float>(k_pages[code]) * __half2float(k_scales[scale]));
    v_out[dst] = __float2half(static_cast<float>(v_pages[code]) * __half2float(v_scales[scale]));
}

// BF16 [256, QHeads, T] -> FP32, same element order.
__global__ void volta_flash_convert_q_kernel(const __nv_bfloat16* __restrict__ q,
                                             float* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __bfloat162float(q[i]);
}

// INT8 K is stored in the rotated basis, so Q must cross the same transform before the FP16 flash
// kernel evaluates QK.
__launch_bounds__(256) __global__ void volta_flash_convert_q_i8_kernel(
        const __nv_bfloat16* __restrict__ q, float* __restrict__ out, int rows) {
    constexpr int kWarps = 8;
    const int row = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows) { return; }
    float values[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int d  = lane + 32 * item;
        values[item] = __bfloat162float(q[static_cast<std::int64_t>(row) * kHeadDim + d]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int d = lane + 32 * item;
        out[static_cast<std::int64_t>(row) * kHeadDim + d] = values[item];
    }
}

// FP32 -> BF16 on the way back out, same element order.
__global__ void volta_flash_convert_out_kernel(const float* __restrict__ in,
                                               __nv_bfloat16* __restrict__ out,
                                               std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __float2bfloat16(in[i]);
}

// Causal mask for one Q-block, built from the device-side positions rather than inferred from the
// envelope, so a non-sequential position layout cannot silently produce a wrong mask. The valid
// column count is read on the device too: `valid_columns` is a device tensor and dereferencing it
// from the host would read device memory.
__global__ void volta_flash_build_mask_kernel(const std::int32_t* __restrict__ positions,
                                              const std::int32_t* __restrict__ valid_columns,
                                              int width, int token_begin,
                                              half* __restrict__ mask, int n_kv,
                                              int mask_stride) {
    const int row = blockIdx.x;
    const int declared = valid_columns == nullptr ? width : valid_columns[0];
    const int valid_rows = declared < width ? declared : width;
    const std::int32_t last_visible =
        token_begin + row < valid_rows ? positions[token_begin + row] : -1;

    half* mask_row = mask + static_cast<std::int64_t>(row) * mask_stride;
    for (int key = threadIdx.x; key < n_kv; key += blockDim.x) {
        mask_row[key] = key <= last_visible ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

bool volta_flash_kv_heads(std::int32_t q_heads, std::int32_t& kv_heads) {
    if (q_heads == CausalD256H24Kv4::QHeads) {
        kv_heads = CausalD256H24Kv4::KVHeads;
        return true;
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        kv_heads = CausalD256H16Kv2::KVHeads;
        return true;
    }
    return false;
}

template <typename Geometry>
void volta_flash_launch_impl(const Tensor& q, const Tensor& positions,
                             const Tensor& valid_columns, const Tensor& table_rows, float scale,
                             std::int64_t visible_keys, const PagedKVBatchLayerView& cache,
                             Tensor& out, Tensor& k_gathered, Tensor& v_gathered, Tensor& mask,
                             Tensor& q_f32, Tensor& out_f32, void* flash_workspace,
                             std::size_t flash_workspace_bytes, cudaStream_t stream) {
    constexpr int kQHeads  = Geometry::QHeads;
    constexpr int kKVHeads = Geometry::KVHeads;

    const std::int32_t width         = q.ne[2];
    const std::int32_t n_kv_padded   = round_up_keys(static_cast<int>(visible_keys));
    const std::int32_t logical_pages = cache.block_tables.ne[0];

    auto* block_tables = static_cast<const std::int32_t*>(cache.block_tables.data);
    auto* rows         = static_cast<const std::int32_t*>(table_rows.data);
    auto* position_ptr = static_cast<const std::int32_t*>(positions.data);

    // Gather the visible key range paged -> contiguous FP16, once for every Q-block of this layer.
    if (cache.storage == KvCacheStorage::Int8Group64) {
        volta_flash_gather_kv_i8_kernel<kKVHeads>
            <<<dim3(n_kv_padded, kKVHeads), kHeadDim, 0, stream>>>(
                static_cast<const std::int8_t*>(cache.k_pages.data),
                static_cast<const std::int8_t*>(cache.v_pages.data),
                static_cast<const half*>(cache.k_scale_pages.data),
                static_cast<const half*>(cache.v_scale_pages.data), block_tables, rows,
                logical_pages, static_cast<half*>(k_gathered.data),
                static_cast<half*>(v_gathered.data), static_cast<std::int32_t>(visible_keys),
                n_kv_padded);
    } else {
        volta_flash_gather_kv_kernel<kKVHeads>
            <<<dim3(n_kv_padded, kKVHeads), kHeadDim, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(cache.k_pages.data),
                static_cast<const half*>(cache.v_pages.data), block_tables, rows, logical_pages,
                static_cast<half*>(k_gathered.data), static_cast<half*>(v_gathered.data),
                static_cast<std::int32_t>(visible_keys), n_kv_padded);
    }
    CUDA_CHECK(cudaGetLastError());

    const bool rotated  = cache.storage == KvCacheStorage::Int8Group64;
    auto* q_f32_ptr     = static_cast<float*>(q_f32.data);
    auto* out_f32_ptr   = static_cast<float*>(out_f32.data);
    auto* mask_ptr      = static_cast<half*>(mask.data);
    auto* valid_ptr     = valid_columns.data == nullptr
                              ? nullptr
                              : static_cast<const std::int32_t*>(valid_columns.data);

    // The keys already in the cache before this call: the called rows are the tail of the
    // sequence, so the first row's position is `visible_keys - width`. Every Q-block has to be
    // given a key extent that includes that prefix, otherwise the block cannot see it.
    const std::int32_t prefix = std::max<std::int32_t>(0, static_cast<std::int32_t>(visible_keys) - width);
    for (std::int32_t begin = 0; begin < width; begin += kFlashQBlockTokens) {
        const std::int32_t tokens = std::min<std::int32_t>(kFlashQBlockTokens, width - begin);
        const std::int32_t n_kv   = round_up_keys(prefix + begin + tokens);
        const std::int32_t rows_padded =
            ((tokens + kFlashMaskRowPad - 1) / kFlashMaskRowPad) * kFlashMaskRowPad;

        const auto* q_begin = static_cast<const __nv_bfloat16*>(q.data) +
                              static_cast<std::int64_t>(begin) * kQHeads * kHeadDim;
        if (rotated) {
            const int rows = tokens * kQHeads;
            volta_flash_convert_q_i8_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(q_begin, q_f32_ptr,
                                                                               rows);
        } else {
            const std::int64_t count = static_cast<std::int64_t>(tokens) * kQHeads * kHeadDim;
            const int blocks         = static_cast<int>((count + 255) / 256);
            volta_flash_convert_q_kernel<<<blocks, 256, 0, stream>>>(q_begin, q_f32_ptr, count);
        }

        volta_flash_build_mask_kernel<<<rows_padded, 256, 0, stream>>>(position_ptr, valid_ptr,
                                                                      width, begin, mask_ptr,
                                                                      n_kv, n_kv);

        const std::size_t out_bytes =
            static_cast<std::size_t>(tokens) * kQHeads * kHeadDim * sizeof(float);
        CUDA_CHECK(cudaMemsetAsync(out_f32_ptr, 0, out_bytes, stream));

        gguf::FlashAttentionArgs args;
        args.head_dim        = kHeadDim;
        args.query_heads     = kQHeads;
        args.kv_heads        = kKVHeads;
        args.tokens          = tokens;
        args.q               = q_f32_ptr;
        args.q_token_stride  = static_cast<std::int64_t>(kQHeads) * kHeadDim;
        args.k               = k_gathered.data;
        args.v               = v_gathered.data;
        args.row_bytes       = static_cast<std::int64_t>(kHeadDim) * 2;
        // The gathered planes are laid out for the whole visible range; only the first `slots`
        // keys of each head are read.
        args.head_bytes      = static_cast<std::int64_t>(kHeadDim) * n_kv_padded * 2;
        args.slots           = n_kv;
        args.mask            = mask_ptr;
        args.mask_row_halves = n_kv;
        args.out             = out_f32_ptr;
        args.scale           = scale;
        args.kv_max          = nullptr;
        gguf::flash_attention(args, flash_workspace, flash_workspace_bytes, stream);

        const std::int64_t count = static_cast<std::int64_t>(tokens) * kQHeads * kHeadDim;
        const int blocks         = static_cast<int>((count + 255) / 256);
        volta_flash_convert_out_kernel<<<blocks, 256, 0, stream>>>(
            out_f32_ptr,
            static_cast<__nv_bfloat16*>(out.data) +
                static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
            count);
    }
    // Rows past valid_columns keep the zero the caller's contract requires.
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::int32_t causal_attention_volta_flash_q_block_tokens() { return kFlashQBlockTokens; }

std::int32_t causal_attention_volta_flash_maximum_visible_keys() {
    return kFlashMaximumVisibleKeys;
}

bool causal_attention_volta_flash_applies(std::int32_t q_heads, std::int32_t width,
                                          std::int32_t batch_size, KvCacheStorage storage,
                                          std::int32_t head_dim,
                                          std::int64_t max_visible_keys) {
    if (!volta_flash_enabled()) { return false; }
    if (head_dim != kHeadDim) { return false; }
    std::int32_t kv_heads = 0;
    if (!volta_flash_kv_heads(q_heads, kv_heads)) { return false; }
    if (batch_size != 1) { return false; }
    // BFloat16 by default; the INT8 cache is correct but slower here, so it is opt-in. Both still
    // go through the width and visible-key gates below.
    if (storage == KvCacheStorage::Int8Group64) {
        if (!volta_flash_int8_enabled()) { return false; }
    } else if (storage != KvCacheStorage::BFloat16) {
        return false;
    }
    if (max_visible_keys > kFlashMaximumVisibleKeys) { return false; }
    return width >= kFlashQBlockTokens;
}

std::int32_t causal_attention_volta_flash_key_extent(std::int64_t max_visible_keys) {
    const std::int32_t keys = static_cast<std::int32_t>(std::max<std::int64_t>(kKeyPad, max_visible_keys));
    const std::int32_t rounded = round_up_keys(keys);
    return rounded < kFlashMaximumVisibleKeys ? rounded : kFlashMaximumVisibleKeys;
}

std::size_t causal_attention_volta_flash_gather_elements(std::int32_t q_heads,
                                                         std::int64_t max_visible_keys) {
    std::int32_t kv_heads = 0;
    if (!volta_flash_kv_heads(q_heads, kv_heads)) { return 0; }
    const std::int64_t keys = causal_attention_volta_flash_key_extent(max_visible_keys);
    return static_cast<std::size_t>(2) * static_cast<std::size_t>(keys) *
           static_cast<std::size_t>(kv_heads) * kHeadDim;
}

std::size_t causal_attention_volta_flash_block_elements(std::int32_t q_heads,
                                                        std::int64_t max_visible_keys) {
    std::int32_t kv_heads = 0;
    if (!volta_flash_kv_heads(q_heads, kv_heads)) { return 0; }
    const std::int64_t keys = causal_attention_volta_flash_key_extent(max_visible_keys);
    const std::int64_t mask = keys * (kFlashQBlockTokens + kFlashMaskRowPad);
    const std::int64_t f32 =
        static_cast<std::int64_t>(2) * q_heads * kHeadDim * kFlashQBlockTokens;
    return static_cast<std::size_t>(mask + f32);
}

std::size_t causal_attention_volta_flash_stream_elements(std::int32_t q_heads) {
    std::int32_t kv_heads = 0;
    if (!volta_flash_kv_heads(q_heads, kv_heads)) { return 0; }
    // The stream-K partials are float2, so the bridge's byte count is rounded up to float2 here.
    const std::size_t bytes = gguf::flash_attention_workspace_bytes(
        kHeadDim, q_heads, kv_heads, kFlashQBlockTokens);
    return (bytes + sizeof(float2) - 1) / sizeof(float2);
}

void causal_attention_volta_flash_launch(const Tensor& q, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& table_rows,
                                         float scale, std::int64_t visible_keys,
                                         const PagedKVBatchLayerView& cache, Tensor& out,
                                         Tensor& k_gathered, Tensor& v_gathered, Tensor& mask,
                                         Tensor& q_f32, Tensor& out_f32, void* flash_workspace,
                                         std::size_t flash_workspace_bytes, cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        volta_flash_launch_impl<CausalD256H24Kv4>(q, positions, valid_columns, table_rows, scale,
                                                  visible_keys, cache, out, k_gathered,
                                                  v_gathered, mask, q_f32, out_f32,
                                                  flash_workspace, flash_workspace_bytes, stream);
        return;
    }
    if (q.ne[1] == CausalD256H16Kv2::QHeads) {
        volta_flash_launch_impl<CausalD256H16Kv2>(q, positions, valid_columns, table_rows, scale,
                                                  visible_keys, cache, out, k_gathered,
                                                  v_gathered, mask, q_f32, out_f32,
                                                  flash_workspace, flash_workspace_bytes, stream);
        return;
    }
    throw std::invalid_argument("causal_attention volta flash: unsupported Q head geometry");
}

} // namespace ninfer::ops::detail
