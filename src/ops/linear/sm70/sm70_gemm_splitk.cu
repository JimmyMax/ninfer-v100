// EXPERIMENT-11: 手动 split-K GEMM（Turing）
// 思路：把 K 维拆成 S 段，每段用 cuBLAS GEMM 算部分和，再用一个 reduce kernel 求和
// 这样 grid = S × (M/128) × (N/128)，SM 利用率翻倍

#include "ops/linear/sm70/sm70_gemm.h"

#include "core/device.h"
#include "ops/common/device_tier.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::detail::sm70 {

int g_turing_splitk = []() {
    const char* env = getenv("NINFER_SPLITK");
    return env ? atoi(env) : 1;
}();

namespace {

// 手动 split-K GEMM：C = A × B，A=[K,N] row-major，B=[K,M] col-major，C=[N,M] col-major
// 把 K 拆成 S 段：C = sum_s(A_s × B_s)
// 每段用 cuBLAS，最后 reduce

struct SplitKWorkspace {
    float* partial = nullptr;   // [S, N, M] 每段的部分和
    std::size_t partial_elems = 0;
    cublasHandle_t blas = nullptr;
    int split_k = 1;
};

SplitKWorkspace& workspace() {
    static SplitKWorkspace ws;
    return ws;
}

void ensure_splitk_workspace(int split_k, int rows, int tokens, cudaStream_t stream) {
    auto& ws = workspace();
    std::size_t need = (std::size_t)split_k * rows * tokens;
    if (ws.partial_elems < need) {
        if (ws.partial) { cudaFree(ws.partial); }
        cudaMalloc(&ws.partial, need * sizeof(float));
        ws.partial_elems = need;
    }
    if (!ws.blas) { cublasCreate(&ws.blas); }
    cublasSetStream(ws.blas, stream);
    ws.split_k = split_k;
}

// reduce kernel：sum partial[s][i] over s
__global__ void split_k_reduce(const float* __restrict__ partial, float* __restrict__ out,
                               int split_k, long total) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    float sum = 0.f;
    for (int s = 0; s < split_k; s++) {
        sum += partial[s * total + i];
    }
    out[i] = sum;
}

} // namespace

void gemm_splitk(const __half* w, std::int32_t rows, std::int32_t k, const __half* x,
                 std::int32_t tokens, float* out, std::int64_t ldc, cudaStream_t stream,
                 int split_k) {
    // C[N,M] = A[K,N]^T × B[K,M]，A=w row-major，B=x col-major
    // 拆 K 为 split_k 段
    auto& ws = workspace();
    ensure_splitk_workspace(split_k, rows, tokens, stream);

    const int k_per_split = k / split_k;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 每段 cuBLAS GEMM
    for (int s = 0; s < split_k; s++) {
        const __half* w_s = w + (long)s * k_per_split * rows;
        const __half* x_s = x + (long)s * k_per_split * tokens;
        float* out_s = ws.partial + (long)s * rows * tokens;

        cublasStatus_t st = cublasGemmEx(
            ws.blas, CUBLAS_OP_T, CUBLAS_OP_N,
            rows, tokens, k_per_split,
            &alpha, w_s, CUDA_R_16F, k_per_split,
            x_s, CUDA_R_16F, k_per_split,
            &beta, out_s, CUDA_R_32F, rows,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("splitK cuBLAS failed: " + std::to_string((int)st));
        }
    }

    // reduce
    long total = (long)rows * tokens;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    split_k_reduce<<<(unsigned)blocks, threads, 0, stream>>>(
        ws.partial, out, split_k, total);
}

} // namespace ninfer::ops::detail::sm70
