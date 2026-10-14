#include "core/device.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_SM70

namespace {
using S = Fp8VoltaQpnSchedule;
static_assert(kFp8VoltaQpnRowsPerTile == S::kRowsPerTile,
              "dispatch mirrors the tile height for host code");

__global__ void fp8_stage_bf16_activation_kernel(const __nv_bfloat16* __restrict__ input,
                                                 half* __restrict__ output,
                                                 std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = __float2half(__bfloat162float(input[i])); }
}
} // namespace

void fp8_stage_bf16_activation_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream) {
    const std::int64_t count = x.numel();
    fp8_stage_bf16_activation_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<half*>(fp16_out), count);
    CUDA_CHECK(cudaGetLastError());
}

bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // A lane consumes whole 128-byte lines, and the CTA's SPLITK warps split K by whole lines.
    // 8 is the floor every shipped SPLITK choice (8 or 16) divides down to (see the launcher).
    if (k % S::kKPerBlock != 0) { return false; }
    if (k / S::kKPerBlock < 8) { return false; }
    // Two 8-row A tiles cover T=9..16 in one pass over the weights. The W8 sibling rejected its
    // two-tile form because the 32x8 fused route already absorbed those rows for free; FP8 has no
    // 32x8 route, so here the alternative is chunking -- reading the whole weight twice -- and the
    // second tile wins easily.
    //
    // The portable layout historically kept T=1 on the slightly faster SIMT decoder. Volta
    // artifacts are now prepacked for QPN, so the row-major decoder is no longer a legal consumer;
    // QPN owns that width as well as verification.
    return t <= 4 * S::kRowsPerTile;
}

void launch_fp8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    launch_fp8_volta_qpn_with_output(x, w, Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data),
                                                               n},
                                     n, stream);
}

void launch_fp8_volta_qpn_fp16(const Tensor& x, const Weight& w, const void* x_fp16, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    launch_fp8_volta_qpn_with_fp16_activation(
        x, w, static_cast<const half*>(x_fp16),
        Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), n}, n, stream);
}

#endif // NINFER_SM70

} // namespace ninfer::ops::detail
