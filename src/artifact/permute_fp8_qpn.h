#pragma once

#include "core/weight_view.h"

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace ninfer::artifact {

// 本仓库 V100 增量（docs/v100/PLAN.md）。
//
// FP8（行尺度）的码字平面在 artifact 里是行主序：一行 k 个 E4M3 码字。Volta QPN8 内核按
// [N/32 tile][K/128 block][8 个 16 字节切片][lane32] 读它——一个 lane 在一个切片里拿 16 个
// 连续 k，于是 32 个 lane 合起来一次读满 512 字节的连续线。
//
// 与 NVFP4 的 QPN 置换一样，这是一个尺寸不变的片段局部置换，所以在设备上、每次上传之后原地
// 施加（与 permute_row_split 同一位置、同一理由），artifact 本身永远不声明它。尺度平面原位
// 不动：QPN8 按输出行读一个 bf16 尺度，行序没有变。
[[nodiscard]] bool fp8_qpn_prepack_supported(const WeightGeometry& geometry) noexcept;

// 原地置换码字平面（artifact 布局把码字放在载荷起始处）。几何必须满足
// fp8_qpn_prepack_supported。
void permute_fp8_to_qpn(const WeightGeometry& geometry, std::byte* data, cudaStream_t stream);

// 同上，但载荷由调用方直接给出（ops 的 Weight 视图把码字与尺度分开持有）。
void permute_fp8_to_qpn_planes(std::int32_t n, std::int32_t k, std::byte* codes,
                               cudaStream_t stream);

} // namespace ninfer::artifact
