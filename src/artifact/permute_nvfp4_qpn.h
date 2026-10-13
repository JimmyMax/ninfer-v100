#pragma once

#include "core/weight_view.h"

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace ninfer::artifact {

// 本仓库 V100 增量（docs/v100/PLAN.md）。
//
// Volta QPN2 消费 NVFP4 码字/尺度的顺序不是 artifact 里的行主序：它按 [N/32 tile][K/16
// group][lane32][8B] 读一个 32 行 × 16 k 的片段，尺度平面沿同一 tile 走 128 行的
// BlockScaleK16M128x4 swizzle。这个重排是尺寸不变的片段局部置换，所以在设备上、每次上传之后
// 原地施加（与 permute_row_split 同一位置、同一理由），artifact 本身永远不声明它。
[[nodiscard]] bool nvfp4_qpn_prepack_supported(const WeightGeometry& geometry) noexcept;

// 原地置换码字与尺度两个平面。几何必须满足 nvfp4_qpn_prepack_supported。
void permute_nvfp4_to_qpn(const WeightGeometry& geometry, std::byte* data, cudaStream_t stream);

// 同上，但载荷载在相邻的两个平面上（ops 的 Weight 视图把 qdata/scales 分开持有）。
void permute_nvfp4_to_qpn_planes(std::int32_t n, std::int32_t k, std::byte* codes,
                                 std::byte* scales, cudaStream_t stream);

} // namespace ninfer::artifact
