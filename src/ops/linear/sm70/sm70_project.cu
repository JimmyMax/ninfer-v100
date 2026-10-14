#include "ops/linear/sm70/sm70_project.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.cuh"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/sm70/sm70_gemm.cuh"
#include "ops/linear/sm70/sm70_integer.h"
#include "ops/linear/sm70/sm70_small.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail::sm70 {
namespace {

// plane [t][2 * count]: gate rows then up rows of one slab.
__global__ void swiglu_epilogue_kernel(const float* __restrict__ plane, std::int32_t count,
                                       std::int32_t tokens, std::int32_t row0, std::int32_t rows,
                                       RowFactor gate_factor, std::int32_t gate_first,
                                       RowFactor up_factor, std::int32_t up_first,
                                       __nv_bfloat16* __restrict__ out) {
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= count) { return; }
    const float gate_scale = gate_factor(gate_first + row0 + r);
    const float up_scale   = up_factor(up_first + row0 + r);
    for (std::int32_t t = static_cast<std::int32_t>(blockIdx.y); t < tokens;
         t += static_cast<std::int32_t>(gridDim.y)) {
        const float* column = plane + static_cast<std::size_t>(t) * (2 * count);
        out[static_cast<std::size_t>(t) * rows + row0 + r] =
            __float2bfloat16_rn(silu(column[r] * gate_scale) * (column[count + r] * up_scale));
    }
}

} // namespace

std::size_t workspace_bytes(std::int32_t max_rows, std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(activation_bytes(k, tokens));
    (void)layout.alloc_bytes(slab_bytes(max_rows, k, tokens));
    return layout.peak_bytes(1);
}

Activation prepare(const Tensor& x, std::int32_t max_rows, WorkspaceArena& workspace,
                   cudaStream_t stream) {
    if (x.dtype != DType::BF16 || x.data == nullptr || !x.is_contiguous() || x.ne[2] != 1 ||
        x.ne[3] != 1 || max_rows <= 0) {
        throw std::invalid_argument("sm70 projection: x must be a contiguous BF16 [K, T]");
    }
    Activation out;
    out.k        = x.ne[0];
    out.tokens   = x.ne[1];
    out.max_rows = max_rows;
    auto* f16 =
        static_cast<__half*>(workspace.alloc_bytes(activation_bytes(out.k, out.tokens)).data);
    out.scratch = workspace.alloc_bytes(slab_bytes(max_rows, out.k, out.tokens)).data;
    convert_activation(static_cast<const __nv_bfloat16*>(x.data), out.k, out.tokens, f16, stream);
    out.data = f16;
    return out;
}

void project(const Activation& x, const Weight& w, std::span<const Destination> destinations,
             cudaStream_t stream) {
    if (w.k != x.k || w.n > x.max_rows || destinations.empty() ||
        destinations.size() > kMaxDestinations || !dequantizable(w)) {
        throw std::invalid_argument("sm70 projection: unsupported weight or destinations");
    }
    DestinationEpilogue epilogue;
    for (const Destination& d : destinations) { epilogue.set[epilogue.count++] = d; }
    epilogue.factor       = row_factor(w);
    const auto dequantize = [&](std::int32_t row0, std::int32_t count, __half* dst) {
        dequantize_rows(w, row0, count, dst, stream);
    };
    product(x.data, x.tokens, w.n, w.k, dequantize, epilogue, x.scratch, stream);
}

void swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
            cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const bool joined       = up == nullptr;
    if (out.dtype != DType::BF16 || out.ne[1] != x.tokens || gate.k != x.k ||
        (joined ? gate.n != 2 * rows : (gate.n != rows || up->n != rows || up->k != x.k)) ||
        !dequantizable(gate) || (!joined && !dequantizable(*up))) {
        throw std::invalid_argument("sm70 swiglu: unsupported weights or output");
    }
    const std::int32_t slab = std::max(1, slab_rows(x.max_rows, x.k) / 2);
    auto* weights           = static_cast<__half*>(x.scratch);
    auto* plane             = reinterpret_cast<float*>(
        static_cast<std::byte*>(x.scratch) +
        (static_cast<std::size_t>(slab_rows(x.max_rows, x.k)) * x.k * sizeof(__half) + 255) / 256 *
            256);
    const Weight& up_weight     = joined ? gate : *up;
    const std::int32_t up_first = joined ? rows : 0;
    constexpr int kThreads      = 256;
    for (std::int32_t row0 = 0; row0 < rows; row0 += slab) {
        const std::int32_t count = std::min(slab, rows - row0);
        dequantize_rows(gate, row0, count, weights, stream);
        dequantize_rows(up_weight, up_first + row0, count,
                        weights + static_cast<std::size_t>(count) * x.k, stream);
        gemm(weights, 2 * count, x.k, x.data, x.tokens, plane, 2 * count, stream);
        const dim3 grid(static_cast<unsigned>((count + kThreads - 1) / kThreads),
                        static_cast<unsigned>(std::min<std::int32_t>(x.tokens, 65535)));
        swiglu_epilogue_kernel<<<grid, kThreads, 0, stream>>>(
            plane, count, x.tokens, row0, rows, row_factor(gate), 0, row_factor(up_weight),
            up_first, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }
}

bool covered_format(QType format, std::int32_t k) {
    // The small-T route's chunks: one T2 G128 group, 64 values of the other formats.
    const std::int32_t chunk = format == QType::T2_G128_FP16 ? 128 : 64;
    if (!dequantizable_format(format) || k % chunk != 0) { return false; }
#if defined(NINFER_SM70)
    // 本仓库 V100 增量（docs/v100/PLAN.md）：QPN 支持的 FP8 形状由 QPN/CUTLASS 内核拥有全部宽度
    // （与 fork 一致），sm_70 的 SIMT 反量化路线不再认领——否则加载时重排
    // （VoltaQpnPrepacked）的权重会被按行主序误读。
    if (format == QType::FP8_E4M3FN_ROW_BF16 && k % 128 == 0 && k / 128 >= 8) { return false; }
#endif
    return true;
}

bool covers(const Weight& w, std::int32_t tokens) {
    return enabled() && tokens >= 1 && covered_format(w.qtype, w.k) && dequantizable(w);
}

std::size_t covered_bytes(QType format, std::int32_t max_rows, std::int32_t k,
                          std::int32_t min_tokens, std::int32_t max_tokens, LinearPolicy policy) {
    std::size_t bytes = 0;
    if (small_takes(min_tokens)) {
        bytes = small_workspace_bytes(k, std::min(max_tokens, cublas_min_tokens() - 1));
    }
    if (integer_format(format, k, min_tokens, policy)) {
        bytes = std::max(bytes, integer_bytes(k, std::min(max_tokens, kIntegerMaxTokens)));
    }
    if (takes(max_tokens)) { bytes = std::max(bytes, workspace_bytes(max_rows, k, max_tokens)); }
    return bytes;
}

Activation prepare_for(const Tensor& x, std::int32_t max_rows, WorkspaceArena& workspace,
                       cudaStream_t stream, bool integer) {
    if (takes(x.ne[1])) { return prepare(x, max_rows, workspace, stream); }
    Activation out = small_prepare(x, workspace, stream);
    if (integer) { integer_prepare(x, workspace, stream, out); }
    return out;
}

bool direct_tokens(std::int32_t tokens) {
    return small_takes(tokens) && tokens <= kSmallDirectTokens;
}

Activation prepare_direct(const Tensor& x) {
    if (!direct_tokens(x.ne[1])) {
        throw std::invalid_argument(
            "sm70 projection: a direct activation is at most eight columns");
    }
    return small_direct(x);
}

void project_for(const Activation& x, const Weight& w, std::span<const Destination> destinations,
                 cudaStream_t stream) {
    if (x.codes != nullptr && w.qtype == QType::T2_G128_FP16) {
        integer_project(x, w, destinations, stream);
    } else if (takes(x.tokens)) {
        project(x, w, destinations, stream);
    } else {
        small_project(x, w, destinations, stream);
    }
}

void swiglu_for(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                cudaStream_t stream) {
    if (x.codes != nullptr && gate.qtype == QType::T2_G128_FP16) {
        integer_swiglu(x, gate, up, out, stream);
    } else if (takes(x.tokens)) {
        swiglu(x, gate, up, out, stream);
    } else {
        small_swiglu(x, gate, up, out, stream);
    }
}

} // namespace ninfer::ops::detail::sm70
