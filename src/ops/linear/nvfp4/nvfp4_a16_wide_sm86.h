#pragma once

// Ampere 档（sm_80/86/89）的 NVFP4 宽 T A16 路线（本仓库增量）。sm_86 上的 NVFP4 权重没有
// 张量核心路线（W4A4 需要 sm_120a），A16 的专用融合内核只注册到小 T；上游在 sm_86 上因此
// 连制品都加载不了（swiglu 规划在 T>16 上抛出）。本路线与 sm_70 的 CUTLASS 宽 T 策略同构：
// 把存放布局的 NVFP4 权重平面一次性精确解码为 BF16 暂存平面，交给既有的 BF16 GEMM 族
// （bf16_dispatch，宽 T 上就是 cuBLAS BF16 张量核心），再按各融合算子追加 silu_mul /
// residual_add / 多目的地行段。解码是精确的：硬件 fp4x2 与 E4M3 转换、fp32 乘逆除数、一次
// 舍入——与 sm_70 宽 T 路线同一数值语义（E2M1 值与 E4M3 尺度在 BF16 里精确可表示）。
//
// 只有 Ampere 档构建编译本文件（CMake 的 NINFER_ARCH_AMPERE 条件）；Volta/Turing/Blackwell
// 构建看不到它，V100 线的对象与二进制逐位不变。
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// A16 宽路线的接管宽度：t 在此之上走“解码到 BF16 暂存 + BF16 GEMM”，之下留在各算子既有的
// 小 T 专用路线（融合内核 / 32 列分块循环）。与 sm_70 的 QPN（≤32）/ CUTLASS（>32）分界一致：
// 分块循环每次只读存放的码/尺度平面（~100 MB 量级），宽路线多付整面 BF16 暂存的写与读，
// 32 列之下永远是循环更省。
inline constexpr std::int32_t kNvfp4A16WideFirstToken = 33;

// BF16 暂存平面的字节数：n * k * 2。GEMM 的输出直接写调用方的 out 或工作区投影平面，
// 不另占暂存。
[[nodiscard]] std::size_t nvfp4_a16_wide_staged_bytes(std::int32_t n, std::int32_t k);

// 把整个 NVFP4 平面（存放的 BlockScaleK16M128x4 布局：码平面 + swizzled 尺度平面）精确解码
// 进 `plane`（BF16 行主序 [n][k]，16 字节对齐）。除数按 dense 路线的契约整面取
// weight_scale_divisor；QPN 重排布局与堆叠多除数平面不支持（dense 的 A16 路线同样不接受它们）。
void nvfp4_a16_wide_staged(const Weight& weight, void* plane, cudaStream_t stream);

} // namespace ninfer::ops::detail
