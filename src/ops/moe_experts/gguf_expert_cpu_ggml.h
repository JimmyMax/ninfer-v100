#pragma once

#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::ops {

struct GgufCpuMatrix;

// GGUF i-quant expert rows through ggml-cpu's vecdot kernels (third_party/ggml-quants, the
// same llama.cpp commit as the GPU bridge): int8 x int8 dots with ggml's Q8_K activations for
// the i-quant gate/up and Q8_0 for the IQ4_NL down, in ggml's SIMD rounding order. A fast
// approximation of gguf_cpu_rows, not bit-identical to the GPU or the exact CPU path.
enum class GgmlActLayout { Q8K, Q8_0 };

[[nodiscard]] bool gguf_ggml_supported(QType format) noexcept;

// BF16 activations quantized into ggml's Q8_K / Q8_0 row layouts, one row per token.
class GgmlCpuActivation {
public:
    GgmlCpuActivation() = default;
    GgmlCpuActivation(int token_capacity, int k);

    // Quantizes `tokens` BF16 columns, `stride` values apart, into the given layout.
    void prepare_bf16(GgmlActLayout layout, const std::uint16_t* values, std::size_t stride,
                      int tokens);

    [[nodiscard]] int tokens() const noexcept { return tokens_; }
    [[nodiscard]] int k() const noexcept { return k_; }
    [[nodiscard]] GgmlActLayout layout() const noexcept { return layout_; }
    [[nodiscard]] const void* codes(int token) const noexcept;
    [[nodiscard]] std::size_t row_bytes() const noexcept;

private:
    int capacity_ = 0;
    int k_        = 0;
    int tokens_   = 0;
    GgmlActLayout layout_ = GgmlActLayout::Q8K;
    std::size_t row_bytes_ = 0;
    std::vector<std::uint8_t> data_;
    std::vector<float> scratch_;
};

// y[t * y_stride + r] = row r . x[t] for rows [row_begin, row_end) and all prepared tokens.
// Throws std::invalid_argument for a format gguf_ggml_supported does not cover.
void gguf_cpu_rows_ggml(const GgufCpuMatrix& w, const GgmlCpuActivation& x, int row_begin,
                        int row_end, float* y, std::size_t y_stride);

}  // namespace ninfer::ops
