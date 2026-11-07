// EXPERIMENT-11: CUTLASS Turing GEMM for FP16×FP16→FP32
// 修正：A=X (M×K row-major), B=W (N×K col-major), C=out (N×M col-major)
// CUTLASS: C[M,N] = A[M,K] × B[K,N]

#include "ops/linear/sm70/sm70_gemm.h"

#include "core/device.h"
#include "ops/common/device_tier.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::sm70 {
namespace {

using ElementInput       = cutlass::half_t;
using ElementOutput      = float;
using ElementAccumulator = float;
using ElementCompute     = float;

// A = X: [M, K] row-major (M=tokens, K=5120, lda=K)
// B = W: [N, K] column-major (N=rows, K=5120, ldb=K)
// C = out: [M, N] column-major (M=tokens, N=rows, ldc=N)
using LayoutInputA = cutlass::layout::RowMajor;      // X: tokens × K
using LayoutInputB = cutlass::layout::ColumnMajor;   // W: N × K (col-major of K×N row-major)
using LayoutOutput = cutlass::layout::ColumnMajor;   // out: tokens × rows

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 4, ElementAccumulator, ElementCompute>;

using GemmKernel = cutlass::gemm::device::Gemm<
    ElementInput, LayoutInputA,
    ElementInput, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,    // stages
    8,    // AlignmentA
    8,    // AlignmentB
    false // SplitKSerial
>;

void* g_workspace = nullptr;
std::size_t g_workspace_bytes = 0;

void ensure_workspace(std::size_t bytes) {
    if (g_workspace_bytes < bytes) {
        if (g_workspace) { cudaFree(g_workspace); }
        cudaMalloc(&g_workspace, bytes);
        g_workspace_bytes = bytes;
    }
}

} // namespace

void gemm_cutlass(const __half* w, std::int32_t rows, std::int32_t k, const __half* x,
                  std::int32_t tokens, float* out, std::int64_t ldc, cudaStream_t stream) {
    // C[M,N] = A[M,K] × B[K,N]
    // A = X: [tokens, k] row-major, lda = k
    // B = W: [rows, k] col-major = W[k, rows] row-major, ldb = k
    // C = out: [tokens, rows] col-major, ldc = ldc (传入值 = rows)
    GemmKernel gemm_op;

    const cutlass::gemm::GemmCoord problem(tokens, rows, k);

    const typename GemmKernel::Arguments args(
        problem,
        {reinterpret_cast<const ElementInput*>(x), k},     // A = X
        {reinterpret_cast<const ElementInput*>(w), k},     // B = W
        {out, static_cast<int>(ldc)},                       // C
        {out, static_cast<int>(ldc)},                       // D = C
        {1.0f, 0.0f});

    const std::size_t workspace_bytes = GemmKernel::get_workspace_size(args);
    ensure_workspace(workspace_bytes);

    const cutlass::Status status = gemm_op(args, g_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM failed: " +
                                 std::to_string(static_cast<int>(status)));
    }
}

} // namespace ninfer::ops::detail::sm70
