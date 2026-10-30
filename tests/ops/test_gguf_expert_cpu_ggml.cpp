// The fast ggml-cpu arithmetic of the GGUF CPU experts (gguf_expert_cpu_ggml): q8_K / q8_0
// activations through ggml's AVX2 vecdot kernels, against (1) an FP64 oracle from independently
// decoded stored blocks and (2) the exact q8_1 path, at the Flash-Next expert shapes, for every
// format the fast path supports. The q8 activation bounds the error: relative L2 under 3% and
// the worst output within 10% of the oracle's peak, like the exact path's int8 activation.
#include "ops/moe_experts/gguf_expert_cpu.h"
#include "ops/moe_experts/gguf_expert_cpu_ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-quants.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::ops;

void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

std::uint16_t half_bits(float value) {
    const auto word = std::bit_cast<std::uint32_t>(value);
    const int exponent = int((word >> 23) & 255) - 127 + 15;
    const std::uint32_t mantissa = ((word & 0x7fffff) + 0x1000) >> 13;
    return static_cast<std::uint16_t>((exponent << 10) + mantissa);
}

std::uint16_t bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fff + ((word >> 16) & 1);
    return static_cast<std::uint16_t>(word >> 16);
}

struct Matrix {
    QType format;
    int rows, k;
    std::int64_t row_bytes;
    std::vector<std::uint8_t> bytes;
    std::vector<double> values; // decoded independently from the stored bytes
};

// One stored block of `format` to its FP64 values. The i-quant decodes follow ggml-quants.c's
// dequantize_row_* (llama.cpp a25c9865, MIT); the exact path's oracles use the same formulas.
void reference_block(QType format, const std::uint8_t* bytes, double* y) {
    if (format == QType::GGUF_Q8_0) {
        const auto* x     = reinterpret_cast<const block_q8_0*>(bytes);
        const double d    = exact_half(x->d);
        for (int j = 0; j < QK8_0; ++j) { y[j] = d * static_cast<std::int8_t>(x->qs[j]); }
        return;
    }
    if (format == QType::GGUF_IQ4_NL) {
        const auto* x  = reinterpret_cast<const block_iq4_nl*>(bytes);
        const double d = exact_half(x->d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j]              = d * kvalues_iq4nl[x->qs[j] & 0xf];
            y[j + QK4_NL / 2] = d * kvalues_iq4nl[x->qs[j] >> 4];
        }
        return;
    }
    if (format == QType::GGUF_IQ4_XS) {
        const auto* x      = reinterpret_cast<const block_iq4_xs*>(bytes);
        const std::uint8_t* qs = x->qs;
        const double d     = exact_half(x->d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0]  = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
        return;
    }
    if (format == QType::GGUF_IQ2_XS) {
        const auto* x      = reinterpret_cast<const block_iq2_xs*>(bytes);
        const double d     = exact_half(x->d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const double db[2] = {d * (0.5 + (x->scales[ib32] & 0xf)) * 0.25,
                                  d * (0.5 + (x->scales[ib32] >> 4)) * 0.25};
            for (int l = 0; l < 4; ++l) {
                const auto* grid = reinterpret_cast<const std::uint8_t*>(
                    iq2xs_grid + (x->qs[4 * ib32 + l] & 511));
                const std::uint8_t signs = ksigns_iq2xs[x->qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.0 : 1.0);
                }
                y += 8;
            }
        }
        return;
    }
    if (format == QType::GGUF_IQ2_XXS) {
        const auto* x = reinterpret_cast<const block_iq2_xxs*>(bytes);
        const double d = exact_half(x->d);
        std::uint32_t aux32[2];
        const std::uint8_t* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::memcpy(aux32, x->qs + 4 * ib32, 2 * sizeof(std::uint32_t));
            const double db = d * (0.5 + (aux32[1] >> 28)) * 0.25;
            for (int l = 0; l < 4; ++l) {
                const auto* grid = reinterpret_cast<const std::uint8_t*>(iq2xxs_grid + aux8[l]);
                const std::uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0 : 1.0);
                }
                y += 8;
            }
        }
        return;
    }
    if (format == QType::GGUF_IQ2_S) {
        const auto* x             = reinterpret_cast<const block_iq2_s*>(bytes);
        const double d            = exact_half(x->d);
        const std::uint8_t* qs    = x->qs;
        const std::uint8_t* qh    = x->qh;
        const std::uint8_t* signs = qs + QK_K / 8;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const double db[2] = {d * (0.5 + (x->scales[ib32] & 0xf)) * 0.25,
                                  d * (0.5 + (x->scales[ib32] >> 4)) * 0.25};
            for (int l = 0; l < 4; ++l) {
                const double dl = db[l / 2];
                const auto* grid = reinterpret_cast<const std::uint8_t*>(
                    iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
        return;
    }
    if (format == QType::GGUF_IQ3_XXS) {
        const auto* x       = reinterpret_cast<const block_iq3_xxs*>(bytes);
        const double d      = exact_half(x->d);
        const std::uint8_t* qs = x->qs;
        const std::uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(aux32));
            const double db = d * (0.5 + (aux32 >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
                    iq3xxs_grid + qs[2 * l + 0]);
                const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
                    iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.0 : 1.0);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 8;
        }
        return;
    }
    if (format == QType::GGUF_IQ3_S) {
        const auto* x             = reinterpret_cast<const block_iq3_s*>(bytes);
        const double d            = exact_half(x->d);
        const std::uint8_t* qs    = x->qs;
        const std::uint8_t* qh    = x->qh;
        const std::uint8_t* signs = x->signs;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const double db1 = d * (1 + 2 * (x->scales[ib32 / 2] & 0xf));
            const double db2 = d * (1 + 2 * (x->scales[ib32 / 2] >> 4));
            for (int half = 0; half < 2; ++half) {
                const double db = half == 0 ? db1 : db2;
                for (int l = 0; l < 4; ++l) {
                    const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
                        iq3s_grid + (qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256)));
                    const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
                        iq3s_grid + (qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.0 : 1.0);
                        y[j + 4] = db * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
                    }
                    y += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
        return;
    }
    throw std::runtime_error("reference_block: unhandled format");
}

Matrix make(QType format, int rows, int k, std::mt19937& rng) {
    const auto [group, block] = gguf_block_shape(format);
    Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}};
    m.bytes.resize(std::size_t(rows) * m.row_bytes);
    for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
    std::uniform_real_distribution<float> scale(
        format == QType::GGUF_Q8_0 ? 0.002F : 0.0005F,
        format == QType::GGUF_Q8_0 ? 0.05F : 0.004F);
    m.values.resize(std::size_t(rows) * k);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const auto d    = half_bits(scale(rng));
            const std::uint16_t signed_d = (rng() % 4 == 0) ? std::uint16_t(d | 0x8000) : d;
            std::memcpy(p, &signed_d, 2);
            reference_block(format, p, m.values.data() + std::size_t(r) * k + std::size_t(b) * group);
        }
    }
    return m;
}

GgmlActLayout layout_for(QType format) {
    return (format == QType::GGUF_IQ4_NL || format == QType::GGUF_Q8_0) ? GgmlActLayout::Q8_0
                                                                        : GgmlActLayout::Q8K;
}

void check(const Matrix& m, int tokens, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> x(std::size_t(tokens) * m.k);
    for (auto& v : x) { v = bf16(normal(rng)); }
    std::vector<double> oracle(std::size_t(tokens) * m.rows);
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < m.rows; ++r) {
            double sum = 0;
            for (int i = 0; i < m.k; ++i) {
                sum += m.values[std::size_t(r) * m.k + i] *
                       double(std::bit_cast<float>(std::uint32_t(x[std::size_t(t) * m.k + i]) << 16));
            }
            oracle[std::size_t(t) * m.rows + r] = sum;
        }
    }
    const GgufCpuMatrix w{m.format, m.bytes.data(), m.row_bytes, m.rows, m.k};
    // The exact path where it covers the format (IQ2_XXS and IQ2_XS are fast-path only):
    // the reference product, and the yardstick for the fast path's drift.
    std::vector<float> exact(oracle.size());
    if (gguf_cpu_supported(m.format)) {
        GgufCpuActivation a(16, m.k);
        a.prepare_bf16(m.format, x.data(), m.k, tokens);
        gguf_cpu_rows(w, a, 0, m.rows, exact.data(), m.rows, GgufCpuBackend::Scalar);
    }
    // The fast path.
    GgmlCpuActivation fa(16, m.k);
    fa.prepare_bf16(layout_for(m.format), x.data(), m.k, tokens);
    std::vector<float> y(oracle.size()), again(oracle.size());
    // Two row ranges, as the workers split them, then the whole matrix again.
    gguf_cpu_rows_ggml(w, fa, 0, m.rows / 3, y.data(), m.rows);
    gguf_cpu_rows_ggml(w, fa, m.rows / 3, m.rows, y.data(), m.rows);
    gguf_cpu_rows_ggml(w, fa, 0, m.rows, again.data(), m.rows);
    double error = 0, norm = 0, peak = 0, worst = 0, diff = 0, diff_norm = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        require(std::isfinite(y[i]), "nonfinite output");
        require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(again[i]),
                "a row split changed the result or a repeat differed");
        error += (double(y[i]) - oracle[i]) * (double(y[i]) - oracle[i]);
        norm  += oracle[i] * oracle[i];
        peak   = std::max(peak, std::abs(oracle[i]));
        worst  = std::max(worst, std::abs(double(y[i]) - oracle[i]));
        if (gguf_cpu_supported(m.format)) {
            diff      += (double(y[i]) - exact[i]) * (double(y[i]) - exact[i]);
            diff_norm += exact[i] * exact[i];
        }
    }
    const double relative = std::sqrt(error / norm);
    const double vs_exact = gguf_cpu_supported(m.format) ? std::sqrt(diff / diff_norm) : 0;
    std::cout << "ggml " << int(m.format) << " " << m.rows << "x" << m.k << " T=" << tokens << ": "
              << "relative L2 " << relative << ", max error/peak " << worst / peak << ", vs exact "
              << vs_exact << '\n';
    require(relative < 0.03 && worst / peak < 0.1, "error exceeds the q8-activation bound");
    require(vs_exact < 0.05, "the fast path strays from the exact path");
}

} // namespace

int main() {
    try {
        std::mt19937 rng(20261030);
        // The gate and up projections: k = 2560 (ten 256-value blocks).
        for (const QType format : {QType::GGUF_IQ2_XXS, QType::GGUF_IQ2_XS, QType::GGUF_IQ2_S,
                                   QType::GGUF_IQ3_XXS, QType::GGUF_IQ3_S, QType::GGUF_IQ4_XS,
                                   QType::GGUF_IQ4_NL, QType::GGUF_Q8_0}) {
            require(gguf_ggml_supported(format), "a format is not marked supported");
            const Matrix gate = make(format, 640, 2560, rng);
            for (const int tokens : {1, 2, 4, 8, 9}) { check(gate, tokens, rng); }
            // The down projection: k = 640, a multiple of 32 but not 256 (Q8_0 activations only).
            if (layout_for(format) == GgmlActLayout::Q8_0) {
                const Matrix down = make(format, 2560, 640, rng);
                for (const int tokens : {1, 4, 8}) { check(down, tokens, rng); }
            }
        }
        // Uncovered formats refuse instead of silently falling back.
        for (const QType format : {QType::GGUF_Q2_0, QType::GGUF_Q4_K, QType::GGUF_Q6_K}) {
            require(!gguf_ggml_supported(format), "an uncovered format is marked supported");
        }
        bool refused = false;
        try {
            GgmlCpuActivation a(1, 2560);
            std::vector<std::uint16_t> x(2560 * 2);
            a.prepare_bf16(GgmlActLayout::Q8K, x.data(), 2560, 2);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "an activation above its capacity was accepted");
        refused = false;
        try { GgmlCpuActivation a(1, 100); } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "a k that is not a multiple of 32 was accepted");
        refused = false;
        try {
            GgmlCpuActivation a(1, 640);
            std::vector<std::uint16_t> x(640 * 2);
            a.prepare_bf16(GgmlActLayout::Q8K, x.data(), 640, 1);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "a Q8_K activation with k not a multiple of 256 was accepted");
        std::cout << "gguf_expert_cpu_ggml: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gguf_expert_cpu_ggml: " << error.what() << '\n';
        return 1;
    }
}
