#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// 本仓库 V100 增量（docs/v100/PLAN.md）：fork 的 fp16 移位技巧（decode_e2m1_word_shift +
// unshuffle + rebias）要在 fp16 里做两次舍入，而且中间值落在次正规区（2^(e-15)*(1+M/2)，
// 小至 2^-15）会丢尾位。实测把 A16 激活档位的结果推到 2 个 bf16 ULP 之外，过不了上游
// “判据只属于激活档位、与实现路径无关”的契约（相对 L2 3.3e-3）。这里改成与 sm_70
// dequant 路径同构的精确解码：硬件转换到 float2，fp32 乘法，一次舍入到 fp16。
__device__ __forceinline__ float decode_e4m3_scale_exact(std::uint8_t value) {
    __nv_fp8_e4m3 scale;
    scale.__x = value;
    return float(scale);
}

// 把 8 个 E2M1 码（一个 uint32）精确解码并缩放为 8 个 fp16。
__device__ __forceinline__ void decode_e2m1_word_exact(std::uint32_t packed, float coefficient,
                                                      half (&out)[8]) {
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        __nv_fp4x2_e2m1 pair;
        pair.__x             = static_cast<std::uint8_t>((packed >> (8 * byte)) & 0xffu);
        const float2 decoded = static_cast<float2>(pair);
        // 本仓库 V100 增量：两个值一次 __float22half2_rn——与逐元素 __float2half 同为
        // round-to-nearest，数值完全一致，但少一半 cvt 指令（prefill 的反量化是权重受限的）。
        const float2 scaled = make_float2(decoded.x * coefficient, decoded.y * coefficient);
        *reinterpret_cast<half2*>(out + 2 * byte) = __float22half2_rn(scaled);
    }
}

// 把预打包片段的 4 字节（8 个 k 值）精确解码为 8 个 fp16。
//
// 片段里的字节不是按 k 顺序排的：加载时重排把每个 16-k 组切成两个 4 字节半组，半组内相邻的 k
// 对交错存放 —— byte0 存 (k+0, k+2)、byte1 存 (k+4, k+6)、byte2 存 (k+1, k+3)、
// byte3 存 (k+5, k+7)（见 artifact/permute_nvfp4_qpn.cu 的 order[]）。fork 的移位技巧恰好
// 还原了这个顺序（v0 = (byte0 低位, byte2 低位) = (k+0, k+1)…），所以它不需要 unshuffle；
// 顺序读字节则会把 k 读成置换后的次序，权重全错。
__device__ __forceinline__ void decode_qpn_e2m1_word_exact(std::uint32_t packed,
                                                           float coefficient, half (&out)[8]) {
    // Each source byte holds one even and one odd k value, so one fp4x2 conversion serves two
    // outputs; decoding a byte per output (the earlier form) repeated every conversion and every
    // multiply twice. The scaled pairs land in the shuffle order (k+0,k+2), (k+1,k+3) and are
    // interleaved into k order by half-register moves. Values and rounding are unchanged: fp32
    // multiply, one RN conversion to fp16.
    constexpr int qpn_byte[4] = {0, 2, 1, 3};
    half2 scaled[4];
#pragma unroll
    for (int half_group = 0; half_group < 4; ++half_group) {
        __nv_fp4x2_e2m1 pair;
        pair.__x = static_cast<std::uint8_t>((packed >> (8 * qpn_byte[half_group])) & 0xffu);
        const float2 decoded = static_cast<float2>(pair);
        scaled[half_group] =
            __float22half2_rn(make_float2(decoded.x * coefficient, decoded.y * coefficient));
    }
    auto* pairs = reinterpret_cast<half2*>(out);
    pairs[0] = __halves2half2(__low2half(scaled[0]), __low2half(scaled[1]));
    pairs[1] = __halves2half2(__high2half(scaled[0]), __high2half(scaled[1]));
    pairs[2] = __halves2half2(__low2half(scaled[2]), __low2half(scaled[3]));
    pairs[3] = __halves2half2(__high2half(scaled[2]), __high2half(scaled[3]));
}

// The artifact stores adjacent E2M1 values in each code byte and one E4M3 scale per K16 group.
// Scales use the BlockScaleK16M128x4 swizzle. Materializing once is intentionally a wide-T
// strategy: CUTLASS can reuse the resulting FP16 matrix across every token tile instead of
// decoding the packed weights again for each tile.
__global__ void dequant_nvfp4_row_to_fp16(const std::uint8_t* __restrict__ codes,
                                           const std::uint8_t* __restrict__ scales, int n, int k,
                                           float inverse_weight_divisor,
                                           cutlass::half_t* __restrict__ out) {
    // A thread owns half of one K16 scale group. The former byte-per-thread,
    // two-dimensional launch created hundreds of thousands of tiny blocks for
    // gate_up; this preserves the exact native artifact decode while matching
    // the prepacked decoder's eight-output work granularity.
    const std::int64_t segment = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int segments_per_row = k / 8;
    const std::int64_t count   = static_cast<std::int64_t>(n) * segments_per_row;
    if (segment >= count) { return; }
    const int row        = static_cast<int>(segment / segments_per_row);
    const int row_segment = static_cast<int>(segment -
        static_cast<std::int64_t>(row) * segments_per_row);
    const int group      = row_segment / 2;
    const int group_half = row_segment & 1;
    const int scale_tile    = group / 4;
    const int scale_lane    = group & 3;
    const int row_inner     = row & 127;
    const int scales_per_m128 = k / 64;
    const std::int64_t scale_offset =
        static_cast<std::int64_t>((row / 128) * scales_per_m128 + scale_tile) * 512 +
        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
    const int byte_begin = group * 8 + group_half * 4;
    const std::uint8_t* source = codes + static_cast<std::int64_t>(row) * (k / 2) + byte_begin;
    const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(source);
    const float coefficient =
        decode_e4m3_scale_exact(scales[scale_offset]) * inverse_weight_divisor;
    half values[8];
    decode_e2m1_word_exact(packed, coefficient, values);
    auto* destination = reinterpret_cast<uint4*>(
        out + static_cast<std::int64_t>(row) * k + byte_begin * 2);
    *destination = *reinterpret_cast<const uint4*>(values);
}

__global__ void dequant_nvfp4_qpn_to_fp16(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales, int n, int k,
                                          float inverse_weight_divisor,
                                          cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int segment  = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int segments_per_row = k / 8;
    if (row >= n || segment >= segments_per_row) { return; }

    const int group      = segment / 2;
    const int group_half = segment & 1;
    const int local_row  = row & 31;
    const int qp         = local_row / 8;
    const int r          = local_row & 7;
    const int lane       = (qp << 2) | (r & 3) | ((r & 4) << 2);
    const int groups     = k / 16;
    const std::int64_t tuple =
        (static_cast<std::int64_t>(row / 32) * groups + group) * 32 + lane;
    const std::uint8_t* packed = codes + tuple * 8 + group_half * 4;
    const std::uint32_t packed_word = *reinterpret_cast<const std::uint32_t*>(packed);
    const float coefficient =
        decode_e4m3_scale_exact(scales[tuple]) * inverse_weight_divisor;
    half values[8];
    decode_qpn_e2m1_word_exact(packed_word, coefficient, values);
    auto* destination = reinterpret_cast<uint4*>(
        out + static_cast<std::int64_t>(row) * k + segment * 8);
    *destination = *reinterpret_cast<const uint4*>(values);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    // 本仓库 V100 增量：一对一次，减半转换指令（bf16->fp32 精确，fp32->fp16 RN）。
    const std::int64_t pair = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t i    = pair * 2;
    if (i + 1 < count) {
        const float2 widened = make_float2(__bfloat162float(in[i]), __bfloat162float(in[i + 1]));
        auto* destination    = reinterpret_cast<half2*>(out + i);
        *destination         = __float22half2_rn(widened);
    } else if (i < count) {
        out[i] = cutlass::half_t(__bfloat162float(in[i]));
    }
}

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA          = cutlass::half_t;
using ElementInputB          = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;
using LayoutInputA           = cutlass::layout::RowMajor;
using LayoutInputB           = cutlass::layout::ColumnMajor;
using LayoutOutput           = cutlass::layout::RowMajor;
using MMAOp                  = cutlass::arch::OpClassTensorOp;
using SmArch                 = cutlass::arch::Sm70;
using ShapeMMAThreadBlock    = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp           = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp             = cutlass::gemm::GemmShape<8, 8, 4>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
    ElementComputeEpilogue>;
constexpr int kNumStages = 2;
using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp, SwizzleThreadBlock, kNumStages>;

template <class Allocator>
struct CutlassWorkspace {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t n,
                                                       std::int32_t k, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, n});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_workspace_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes); }
    return out;
}

} // namespace

std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                               std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(arguments);
    (void)allocate_cutlass_workspace(layout, n, k, cols, gemm_workspace_bytes);
    return layout.peak_bytes(1);
}

void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments sizing_arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(sizing_arguments);

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch =
        allocate_cutlass_workspace(ws, n, k, cols, gemm_workspace_bytes);
    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    constexpr int kThreads = 256;
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        const dim3 grid(static_cast<unsigned>((k / 8 + kThreads - 1) / kThreads),
                        static_cast<unsigned>(n), 1u);
        dequant_nvfp4_qpn_to_fp16<<<grid, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), n, k,
            1.0F / w.weight_scale_divisor, w_fp16);
    } else {
        const std::int64_t segments = static_cast<std::int64_t>(n) * (k / 8);
        const int blocks = static_cast<int>((segments + kThreads - 1) / kThreads);
        dequant_nvfp4_row_to_fp16<<<blocks, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), n, k,
            1.0F / w.weight_scale_divisor, w_fp16);
    }
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t x_count = static_cast<std::int64_t>(cols) * k;
    // 每个线程处理一对（半分元素），所以块数是元素对数的一半向上取整。
    const int x_blocks         = static_cast<int>(((x_count + 1) / 2 + 255) / 256);
    bf16_to_fp16_kernel<<<x_blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                      x_fp16, x_count);
    CUDA_CHECK(cudaGetLastError());

    Gemm gemm_op;
    typename Gemm::Arguments arguments{
        problem_size, {x_fp16, k}, {w_fp16, k}, {static_cast<ElementOutput*>(out.data), n},
        {static_cast<ElementOutput*>(out.data), n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS initialize failed");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS gemm() failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
