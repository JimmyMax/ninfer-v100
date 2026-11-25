#pragma once

// 本仓库 V100 增量（恢复自 1.7.2，bd052b2b9）：Q6_K 密集逐行布局（QuantLayout::VoltaQ6KDense）
// 的全部设备端消费路径。行是 [192*nblk 个 6-bit 码字][k/16 个 int8 切片尺度][2*nblk 个 half d]，
// 元素 e 在位 6e；一个 32 值切片就是 24 B = 3 个 64-bit 字。该布局只在 Volta rank 上由
// storage_trades 产生，所以这里的 kernel 用普通 launch（无 PDL dependent 需求）。
//
// 依赖 ggml_bridge_vec.cuh 的 Slice/VecArgs/Decoder/dot4/vec_store/slice_value/store8/
// kVecWarps/device_facts/check，由包含者先引入或本头文件自己引入。

#include "ggml_bridge_vec.cuh"

namespace ninfer::ops::gguf::detail {
namespace vec {

// Q6_K 是偏置 32 编码（值 = 码字 - 32），符号技巧产出的 s.w 与超块解码逐位相同，点积式不变。
__device__ __forceinline__ void decode_q6k_dense(const std::uint8_t* row, int s, int k, Slice& sl) {
    const float d     = half_at(row + 3 * k / 4 + k / 16 + 2 * (s / 8));
    const auto* sc    = reinterpret_cast<const std::int8_t*>(row + 3 * k / 4 + 2 * s);
    sl.f[0]           = d * float(sc[0]);
    sl.f[1]           = d * float(sc[1]);
    const auto* w     = reinterpret_cast<const std::uint64_t*>(row + 24 * s);
    const std::uint64_t W0 = w[0], W1 = w[1], W2 = w[2];
    auto chunk        = [&](int i) -> std::uint64_t {
        switch (i) {
        case 0: return W0 & 0x0000000000FFFFFFull;
        case 1: return (W0 >> 24) & 0x0000000000FFFFFFull;
        case 2: return ((W0 >> 48) | (W1 << 16)) & 0x0000000000FFFFFFull;
        case 3: return (W1 >> 8) & 0x0000000000FFFFFFull;
        case 4: return (W1 >> 32) & 0x0000000000FFFFFFull;
        case 5: return ((W1 >> 56) | (W2 << 8)) & 0x0000000000FFFFFFull;
        case 6: return (W2 >> 16) & 0x0000000000FFFFFFull;
        default: return (W2 >> 40) & 0x0000000000FFFFFFull;
        }
    };
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const std::uint64_t C = chunk(i);
        const std::uint32_t v0 = C & 63u;
        const std::uint32_t v1 = (C >> 6) & 63u;
        const std::uint32_t v2 = (C >> 12) & 63u;
        const std::uint32_t v3 = (C >> 18) & 63u;
        const std::uint32_t v  = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
        sl.w[i] = int(v ^ 0xE0E0E0E0u ^ ((v & 0x20202020u) * 6u));
    }
}

// kernel<Q6_K, T, false, 0> 的密集布局版——lane 直接取 32 值切片 s = lane + 32*it，
// 行内没有超块偏移。其余（累加、shfl 归约、存储）与超块版相同，保证位级一致。
template <int T>
__launch_bounds__(kVecWarps * 32) __global__ void kernel_q6k_dense(VecArgs p) {
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int k      = p.k;
    const int slices = k / 32;
    constexpr int kRows = 2;
    for (int row0 = (blockIdx.x * kVecWarps + warp) * kRows; row0 < p.rows;
         row0 += gridDim.x * kVecWarps * kRows) {
        const std::uint8_t* rowp[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            rowp[r] = p.weight + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes;
        }
        float acc[kRows][T];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        const auto slice = [&](int it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);
                d[j]            = ds.x;
                const auto* q   = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                Slice sl;
                decode_q6k_dense(rowp[r], s, k, sl);
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    acc[r][j] +=
                        d[j] * (sl.f[0] * float(dot4(sl.w, a[j], 0)) +
                                sl.f[1] * float(dot4(sl.w + 4, a[j] + 4, 0)));
                }
            }
        };
        for (int it = 0; lane + 32 * it < slices; ++it) { slice(it); }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                if (row0 + r < p.rows) {
#pragma unroll
                    for (int j = 0; j < T; ++j) { vec_store(p.out, row0 + r, j, acc[r][j]); }
                }
            }
        }
    }
}

template <int T>
void launch_q6k_dense(const VecArgs& args, cudaStream_t stream) {
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel_q6k_dense<T>,
                                                            kVecWarps * 32, 0),
              "vector occupancy");
        return std::max(blocks, 1);
    }();
    const int per_block = kVecWarps * 2;
    const int groups    = (args.rows + per_block - 1) / per_block;
    const int blocks    = std::min(groups, resident * device_facts().sm_count);
    kernel_q6k_dense<T><<<blocks, kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "vector product launch");
}

// Q6_K 密集布局的 slice dequantize（sm70 预填充与 embedding gather 的快路径）。
template <typename dst_t>
__launch_bounds__(256) __global__ void slice_dequantize_q6k_dense_kernel(
    const std::uint8_t* __restrict__ weight, std::int64_t row_bytes,
    const std::int32_t* __restrict__ row_ids, int rows, int k, dst_t* __restrict__ out,
    std::int64_t out_row_stride) {
    const int slices         = k / 32;
    const std::int64_t total = std::int64_t(rows) * slices;
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += std::int64_t(gridDim.x) * blockDim.x) {
        const int row         = int(i / slices);
        const int slice       = int(i % slices);
        const std::int64_t at = row_ids != nullptr ? row_ids[row] : row;
        Slice s;
        decode_q6k_dense(weight + at * row_bytes, slice, k, s);
        dst_t* y = out + std::int64_t(row) * out_row_stride + 32 * std::int64_t(slice);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v[8];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                v[t] = slice_value<GGML_TYPE_Q6_K>(s, 2 * q + t / 4, t % 4);
            }
            store8(y + 8 * q, v);
        }
    }
}

} // namespace vec

// Q6_K 密集布局（QuantLayout::VoltaQ6KDense）的向量路线。几何条件比超块版多一条
// k % 1024 == 0（行字节 210*k/256 必须是 8 的倍数，64-bit 码字读取才对齐）。融合 [gate; up] 的
// Q6_K 父矩阵不会进入密集布局（storage_trades 只选非融合角色），所以这里没有 Fused 形态。
template <ggml_type type>
void vec_launch_dense(const VecArgs& args, int columns, cudaStream_t stream) {
    static_assert(type == GGML_TYPE_Q6_K);
    using D = vec::Decoder<type>;
    if (args.k % D::kBlockElems != 0 || args.row_bytes % D::kBlockBytes != 0 ||
        args.row_bytes < std::int64_t(args.k / D::kBlockElems) * D::kBlockBytes ||
        args.k % 1024 != 0 || args.rows <= 0) {
        throw std::invalid_argument("gguf dense vector product: unsupported geometry");
    }
    switch (columns) {
    case 1: return vec::launch_q6k_dense<1>(args, stream);
    case 2: return vec::launch_q6k_dense<2>(args, stream);
    case 3: return vec::launch_q6k_dense<3>(args, stream);
    case 4: return vec::launch_q6k_dense<4>(args, stream);
    case 5: return vec::launch_q6k_dense<5>(args, stream);
    case 6: return vec::launch_q6k_dense<6>(args, stream);
    case 7: return vec::launch_q6k_dense<7>(args, stream);
    case 8: return vec::launch_q6k_dense<8>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf dense vector product: columns must be 1..8");
}

// Q6_K 密集布局的 slice dequantize。几何条件同 slice_dequantize 加 k % 1024 == 0。
template <typename dst_t>
void slice_dequantize_q6k_dense(const void* weight, std::int64_t row_bytes, int k,
                                const std::int32_t* row_ids, int rows, dst_t* out,
                                std::int64_t out_row_stride, cudaStream_t stream) {
    if (k % vec::Decoder<GGML_TYPE_Q6_K>::kBlockElems != 0 || k % 1024 != 0 || rows <= 0 ||
        out_row_stride % 8 != 0 || reinterpret_cast<std::uintptr_t>(out) % 16 != 0) {
        throw std::invalid_argument("gguf dense slice dequantize: unsupported geometry");
    }
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks, vec::slice_dequantize_q6k_dense_kernel<dst_t>, 256, 0),
              "slice dequantize occupancy");
        return std::max(blocks, 1);
    }();
    const std::int64_t threads = std::int64_t(rows) * (k / 32);
    const int blocks = int(std::min<std::int64_t>((threads + 255) / 256,
                                                  std::int64_t(resident) * device_facts().sm_count));
    vec::slice_dequantize_q6k_dense_kernel<dst_t><<<blocks, 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight), row_bytes, row_ids, rows, k, out, out_row_stride);
    check(cudaGetLastError(), "dense slice dequantize launch");
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_Q6K_DENSE_INSTANCE                                                              \
    template void ninfer::ops::gguf::detail::vec_launch_dense<GGML_TYPE_Q6_K>(                     \
        const ninfer::ops::gguf::detail::VecArgs&, int, cudaStream_t);                             \
    template void ninfer::ops::gguf::detail::slice_dequantize_q6k_dense<half>(                       \
        const void*, std::int64_t, int, const std::int32_t*, int, half*, std::int64_t, cudaStream_t); \
    template void ninfer::ops::gguf::detail::slice_dequantize_q6k_dense<__nv_bfloat16>(             \
        const void*, std::int64_t, int, const std::int32_t*, int, __nv_bfloat16*, std::int64_t, cudaStream_t); \
    template void ninfer::ops::gguf::detail::slice_dequantize_q6k_dense<float>(                       \
        const void*, std::int64_t, int, const std::int32_t*, int, float*, std::int64_t, cudaStream_t)
