#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_volta_sm70.h"

#if defined(NINFER_TIER_VOLTA)
#    include <cuda_bf16.h>
#    include <cuda_fp16.h>

namespace ninfer::ops::detail {
namespace {

__global__ void nvfp4_volta_stage_bf16_fp16_kernel(const __nv_bfloat16* __restrict__ input,
                                                   half* __restrict__ output,
                                                   std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = __float2half(__bfloat162float(input[i])); }
}

} // namespace

// 本仓库 V100 增量（docs/v100/PLAN.md）：QPN 的 fp16 激活暂存。
// fork 把它放在 ops/linear/fp8/fp8_volta_qpn_gemm.cu（与 FP8 QPN 同一文件）；
// 这里单独成文件，只搬这一支，避免把整个 FP8 QPN 也拖进来。
void nvfp4_volta_stage_bf16_fp16_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream) {
    const std::int64_t count = x.numel();
    nvfp4_volta_stage_bf16_fp16_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<half*>(fp16_out), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
#endif
