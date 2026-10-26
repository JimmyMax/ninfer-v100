#pragma once

#include "core/weight_view.h"

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace ninfer::artifact {
// 本仓库 V100 增量（docs/v100/PLAN.md）。
//
// Volta 的向量解码内核按 32 值切片读 Q6_K：标准超块序每切片要 16 个 2-byte 读取（ql 4-bit 低
// 半 + qh 2-bit 高半，均 2-byte 对齐），而 sm_70 的向量内核是延迟受限的，这正是 Q6_K 停在
// ~550 GB/s、Q4_K 到 ~800 GB/s 的差距来源。密集布局把同一行的码字流成 6-bit 连续位流
// （元素 e 在位 6e），切片就是 24 B = 3 个 64-bit 字，尺度/d 平面各挪到行尾。载荷尺寸不变
// （210 B/超块），所以它是纯重排：在设备上、每次上传之后原地施加（与 QPN 同一位置），
// `.ninfer` 永远不声明它。
[[nodiscard]] bool q6k_dense_prepack_supported(const WeightGeometry& geometry) noexcept;

// 原地重排一个 GgufBlocks 的 Q6_K 对象。几何必须满足 q6k_dense_prepack_supported。
void permute_q6k_to_dense(const WeightGeometry& geometry, std::byte* data, cudaStream_t stream);

} // namespace ninfer::artifact
