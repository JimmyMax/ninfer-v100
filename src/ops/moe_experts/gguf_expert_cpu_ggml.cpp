#include "ops/moe_experts/gguf_expert_cpu_ggml.h"

#include "ops/moe_experts/gguf_expert_cpu.h"  // full GgufCpuMatrix definition

#include "ggml-quants.h"  // NInfer: vendored (block types, quantize_row_*_ref)
#include "quants.h"       // NInfer: vendored (ggml_vec_dot_*)

#include <bit>
#include <stdexcept>

namespace ninfer::ops {

// BF16 -> FP32. A BF16 value's 1/8/7 bit fields sit exactly in FP32's 1/8/23, so the
// exponent-normal range is a plain left shift; BF16 subnormals (below 2^-126) are not
// bit-shiftable, so they decode through their value (2^-133 per mantissa unit).
static float bf16_to_float(std::uint16_t bits) noexcept {
    if ((bits & 0x7F80U) == 0) {  // subnormal or zero
        if (bits == 0) { return 0.0f; }
        return static_cast<float>(bits & 0x7FU) * 1.1754943508e-40f;  // 2^-133
    }
    std::uint32_t v = std::uint32_t(bits) << 16;
    return std::bit_cast<float>(v);
}

bool gguf_ggml_supported(QType format) noexcept {
    switch (format) {
    case QType::GGUF_IQ2_XXS:
    case QType::GGUF_IQ2_XS:
    case QType::GGUF_IQ2_S:
    case QType::GGUF_IQ3_XXS:
    case QType::GGUF_IQ3_S:
    case QType::GGUF_IQ4_XS:
    case QType::GGUF_IQ4_NL:
    case QType::GGUF_Q8_0: return true;
    default: return false;
    }
}

GgmlCpuActivation::GgmlCpuActivation(int token_capacity, int k)
    : capacity_(token_capacity), k_(k) {
    if (capacity_ < 1 || k_ < 32 || k_ % 32 != 0) {
        throw std::invalid_argument("GGUF ggml CPU activation: bad shape");
    }
}

void GgmlCpuActivation::prepare_bf16(GgmlActLayout layout, const std::uint16_t* values,
                                     std::size_t stride, int tokens) {
    if (tokens > capacity_) { throw std::invalid_argument("GGUF ggml CPU activation: too many tokens"); }
    if (layout == GgmlActLayout::Q8K && k_ % 256 != 0) {
        throw std::invalid_argument("GGUF ggml CPU activation: k must be a multiple of 256 for Q8_K");
    }
    layout_    = layout;
    // The vecdot kernels read the in-memory block layouts: block_q8_K keeps a float d and
    // int16 bsums (292 bytes per 256 values), block_q8_0 its FP16 d (36 bytes per 32).
    row_bytes_ = layout == GgmlActLayout::Q8K ? std::size_t(k_) / 256 * sizeof(block_q8_K)
                                              : std::size_t(k_) / 32 * sizeof(block_q8_0);
    data_.resize(std::size_t(capacity_) * row_bytes_);
    scratch_.resize(k_);
    for (int t = 0; t < tokens; ++t) {
        float* x = scratch_.data();
        const std::uint16_t* v = values + std::size_t(t) * stride;
        for (int i = 0; i < k_; ++i) { x[i] = bf16_to_float(v[i]); }
        if (layout == GgmlActLayout::Q8K) {
            quantize_row_q8_K_ref(x, reinterpret_cast<block_q8_K*>(data_.data() + std::size_t(t) * row_bytes_), k_);
        } else {
            quantize_row_q8_0_ref(x, reinterpret_cast<block_q8_0*>(data_.data() + std::size_t(t) * row_bytes_), k_);
        }
    }
    tokens_ = tokens;
}

const void* GgmlCpuActivation::codes(int token) const noexcept {
    return data_.data() + std::size_t(token) * row_bytes_;
}

std::size_t GgmlCpuActivation::row_bytes() const noexcept { return row_bytes_; }

void gguf_cpu_rows_ggml(const GgufCpuMatrix& w, const GgmlCpuActivation& x, int row_begin,
                        int row_end, float* y, std::size_t y_stride) {
    if (!gguf_ggml_supported(w.format) || w.k != x.k() || !w.data || row_begin < 0 || row_end > w.rows ||
        row_begin > row_end || x.tokens() == 0) {
        throw std::invalid_argument("GGUF ggml CPU expert: mismatched matrix, activation or rows");
    }
    const int rows = row_end - row_begin;
    const int ntok = x.tokens();
    switch (w.format) {
    case QType::GGUF_IQ2_XXS:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_iq2_xxs_q8_K(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                          w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    case QType::GGUF_IQ2_XS:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_iq2_xs_q8_K(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                         w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    case QType::GGUF_IQ2_S:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_iq2_s_q8_K(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                        w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    case QType::GGUF_IQ3_XXS:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_iq3_xxs_q8_K(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                          w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    case QType::GGUF_IQ3_S:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_iq3_s_q8_K(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                        w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    case QType::GGUF_IQ4_XS:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_iq4_xs_q8_K(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                         w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    case QType::GGUF_IQ4_NL:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_iq4_nl_q8_0(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                         w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    case QType::GGUF_Q8_0:
        for (int t = 0; t < ntok; ++t) {
            const void* act = x.codes(t);
            for (int r = 0; r < rows; ++r) {
                ggml_vec_dot_q8_0_q8_0(w.k, y + (std::size_t(t) * y_stride + row_begin + r), 1,
                                       w.data + std::size_t(row_begin + r) * w.row_bytes, 1, act, 1, 1);
            }
        }
        break;
    default:
        throw std::invalid_argument("GGUF ggml CPU expert: unsupported format");
    }
}

}  // namespace ninfer::ops
