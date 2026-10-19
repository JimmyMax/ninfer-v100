// GGUF CPU expert rows (Q2_0, Q8_0, IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) against FP64 from
// independently decoded stored blocks, at the Flash-Next expert shapes, for every backend this
// host supports. The i-quants and IQ4_NL are decoded by ggml's reference dequantizers. Two
// oracles: the exact product bounds the int8 activation's error, and the q8_1 product (the
// activation quantized per 32 values as the GPU does, d = amax / 127 rounded to binary16, codes
// round(v / d)) must agree to FP32 summation, so the CPU computes the GPU's arithmetic; every
// vector backend must give the scalar products' bits (test_moe_experts_gguf compares those with the
// GPU's). SiLU (ops/common/silu.h) against FP64, and the BF16 middle of every backend.
#include "ops/common/silu.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
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
    // Positive normal values only, rounded to nearest by truncating a rounded mantissa.
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

// The binary16 value nearest `value` (ties to even), from its exponent's spacing in FP64.
double nearest_half(float value) {
    const double v = std::abs(double(value));
    if (v == 0) { return 0; }
    const int exponent   = std::max(std::ilogb(v), -14); // below 2^-14 the spacing stays 2^-24
    const double spacing = std::ldexp(1.0, exponent - 10);
    const double rounded = std::nearbyint(v / spacing) * spacing; // FE_TONEAREST: ties to even
    return value < 0 ? -rounded : rounded;
}

struct Matrix {
    QType format;
    int rows, k;
    std::int64_t row_bytes;
    std::vector<std::uint8_t> bytes;
    std::vector<double> values; // decoded independently from the stored bytes
};

// ggml-quants.c's dequantize_row_iq2_s, _iq3_xxs, _iq3_s, _iq4_nl and _iq4_xs (llama.cpp
// a25c9865, MIT), with FP64 output: one stored block of `format` to its values.
void reference_block(QType format, const std::uint8_t* bytes, double* y) {
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
                const auto* grid1 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 0]);
                const auto* grid2 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 1]);
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
    // IQ3_S
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
}

Matrix make(QType format, int rows, int k, std::mt19937& rng) {
    if (format != QType::GGUF_Q2_0 && format != QType::GGUF_Q8_0) {
        // Every byte pattern is a valid i-quant or IQ4_NL block; d is set to a sensible size.
        const auto [group, block] = gguf_block_shape(format);
        Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}};
        m.bytes.resize(std::size_t(rows) * m.row_bytes);
        for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
        std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
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
    const int group = format == QType::GGUF_Q2_0 ? 64 : 32;
    const int block = format == QType::GGUF_Q2_0 ? 18 : 34;
    Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}};
    m.bytes.resize(std::size_t(rows) * m.row_bytes);
    std::uniform_real_distribution<float> scale(0.002F, 0.05F);
    for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            std::uint8_t* p     = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const auto d        = half_bits(scale(rng));
            const std::uint16_t signed_d = (rng() % 4 == 0) ? std::uint16_t(d | 0x8000) : d;
            std::memcpy(p, &signed_d, 2);
        }
    }
    m.values.resize(std::size_t(rows) * k);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            const std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const double d        = exact_half(unsigned(p[0]) | (unsigned(p[1]) << 8));
            for (int j = 0; j < group; ++j) {
                const double code = format == QType::GGUF_Q2_0
                                        ? double(((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1)
                                        : double(static_cast<std::int8_t>(p[2 + j]));
                m.values[std::size_t(r) * k + std::size_t(b) * group + j] = d * code;
            }
        }
    }
    return m;
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
    // q8_1 per 32 values, and the product it gives with every slice's dot exact in FP64.
    std::vector<double> q8(oracle.size()), magnitude(oracle.size());
    std::vector<int> codes(m.k);
    std::vector<double> scales(m.k / 32);
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < m.k / 32; ++s) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::abs(std::bit_cast<float>(
                                          std::uint32_t(x[std::size_t(t) * m.k + 32 * s + i]) << 16)));
            }
            const float d = amax / 127.0F;
            scales[s]     = nearest_half(d);
            for (int i = 0; i < 32; ++i) {
                const float v = std::bit_cast<float>(std::uint32_t(x[std::size_t(t) * m.k + 32 * s + i]) << 16);
                codes[32 * s + i] = amax == 0 ? 0 : int(std::lround(v / d));
            }
        }
        for (int r = 0; r < m.rows; ++r) {
            double sum = 0, size = 0;
            for (int s = 0; s < m.k / 32; ++s) {
                double dot = 0;
                for (int i = 32 * s; i < 32 * s + 32; ++i) {
                    dot += m.values[std::size_t(r) * m.k + i] * codes[i];
                }
                sum += scales[s] * dot;
                size += std::abs(scales[s] * dot);
            }
            q8[std::size_t(t) * m.rows + r]        = sum;
            magnitude[std::size_t(t) * m.rows + r] = size;
        }
    }
    GgufCpuActivation a(16, m.k);
    a.prepare_bf16(m.format, x.data(), m.k, tokens);
    const GgufCpuMatrix w{m.format, m.bytes.data(), m.row_bytes, m.rows, m.k};
    std::vector<float> scalar(oracle.size());
    gguf_cpu_rows(w, a, 0, m.rows, scalar.data(), m.rows, GgufCpuBackend::Scalar);
    for (const auto backend : {GgufCpuBackend::Scalar, GgufCpuBackend::Avx2}) {
        if (!gguf_cpu_backend_available(backend)) { continue; }
        std::vector<float> y(oracle.size()), again(oracle.size());
        // Two row ranges, as workers split them.
        gguf_cpu_rows(w, a, 0, m.rows / 3, y.data(), m.rows, backend);
        gguf_cpu_rows(w, a, m.rows / 3, m.rows, y.data(), m.rows, backend);
        gguf_cpu_rows(w, a, 0, m.rows, again.data(), m.rows, backend);
        double error = 0, norm = 0, peak = 0, worst = 0, q8_drift = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            q8_drift = std::max(q8_drift, std::abs(double(y[i]) - q8[i]) / (magnitude[i] + 1e-30));
            require(std::isfinite(y[i]), "nonfinite output");
            require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(again[i]),
                    "a row split changed the result or a repeat differed");
            require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(scalar[i]),
                    "a vector backend's bits differ from the scalar products'");
            error += (double(y[i]) - oracle[i]) * (double(y[i]) - oracle[i]);
            norm += oracle[i] * oracle[i];
            peak  = std::max(peak, std::abs(oracle[i]));
            worst = std::max(worst, std::abs(double(y[i]) - oracle[i]));
        }
        const double relative = std::sqrt(error / norm);
        std::cout << gguf_cpu_backend_name(backend) << " format " << int(m.format) << " "
                  << m.rows << "x" << m.k << " T=" << tokens << ": relative L2 " << relative
                  << ", max error/peak " << worst / peak << ", vs q8_1 " << q8_drift << '\n';
        require(relative < 0.03 && worst / peak < 0.1, "error exceeds the int8-activation bound");
        require(q8_drift < 2e-5, "the product is not the GPU's q8_1 arithmetic");
    }
}

// Units in the last place of |exact| as a float, between `got` and FP64's `exact`.
double ulps(float got, double exact) {
    const float near = static_cast<float>(exact);
    const double ulp = std::nextafter(std::abs(near), std::numeric_limits<float>::infinity()) - std::abs(near);
    return std::abs(double(got) - exact) / ulp;
}

double silu64(double x) { return x / (1.0 + std::exp(-x)); }

void check_silu(std::mt19937& rng) {
    // Within four units of FP64 (the exponential's polynomial, the sum and the quotient); where
    // e^-x passes the float range silu is a zero of x's sign.
    std::vector<float> x = {0.0F, -0.0F, 1e-30F, -1e-30F, 1e-3F, -1e-3F, 0.5F, -0.5F, 1.0F, -1.0F,
                            20.0F, -20.0F, 87.0F, -87.0F, 87.3365479F, -87.3365479F, 88.0F,
                            -88.0F, 88.7228394F, -88.7228394F, 100.0F, -100.0F, 3e38F, -3e38F};
    std::normal_distribution<float> normal(0.0F, 4.0F);
    std::uniform_real_distribution<float> wide(-87.0F, 87.0F);
    for (int i = 0; i < 200000; ++i) { x.push_back(i % 2 ? normal(rng) : wide(rng)); }
    double worst = 0;
    for (const float v : x) {
        const float got     = silu(v);
        const double exact  = silu64(double(v));
        require(std::isfinite(got), "nonfinite SiLU");
        if (v <= -88.7228394F) {
            require(got == 0 && std::signbit(got), "SiLU past e^-x's range");
            continue;
        }
        worst = std::max(worst, ulps(got, exact));
    }
    std::cout << "silu: " << x.size() << " values, worst " << worst << " ulp\n";
    require(worst <= 4, "SiLU strays from FP64");
    // The BF16 middle: every backend gives the same words, each within one BF16 unit of FP64's.
    std::vector<float> gate(x.begin(), x.begin() + 100003), up(gate.size());
    std::normal_distribution<float> unit(0.0F, 1.0F);
    for (auto& v : up) { v = unit(rng); }
    std::vector<std::uint16_t> scalar(gate.size());
    gguf_cpu_swiglu_bf16(gate.data(), up.data(), gate.size(), scalar.data(), GgufCpuBackend::Scalar);
    for (std::size_t i = 0; i < gate.size(); ++i) {
        if (std::abs(gate[i]) >= 88.0F) { continue; } // the backends' agreement covers these
        const double exact = silu64(double(gate[i])) * double(up[i]);
        const double got   = double(std::bit_cast<float>(std::uint32_t(scalar[i]) << 16));
        const double step  = std::ldexp(1.0, std::ilogb(std::max(std::abs(exact), 1e-38)) - 7);
        require(std::abs(got - exact) <= step, "the BF16 middle strays from FP64");
    }
    for (const auto backend : {GgufCpuBackend::Avx2}) {
        if (!gguf_cpu_backend_available(backend)) { continue; }
        std::vector<std::uint16_t> y(gate.size());
        gguf_cpu_swiglu_bf16(gate.data(), up.data(), gate.size(), y.data(), backend);
        require(y == scalar, "a vector backend's BF16 middle differs from the scalar one");
        std::cout << gguf_cpu_backend_name(backend) << " middle: " << y.size() << " words equal\n";
    }
}

} // namespace

int main() {
    try {
        std::mt19937 rng(20261009);
        check_silu(rng);
        for (const QType format : {QType::GGUF_Q2_0, QType::GGUF_Q8_0, QType::GGUF_IQ4_NL}) {
            const Matrix gate = make(format, 640, 2560, rng);
            const Matrix down = make(format, 2560, 640, rng);
            for (const int tokens : {1, 2, 3, 4, 5, 8, 9}) {
                check(gate, tokens, rng);
                check(down, tokens, rng);
            }
        }
        // The i-quants hold the gate and up projections (k = 2560, ten 256-value blocks).
        for (const QType format : {QType::GGUF_IQ2_S, QType::GGUF_IQ3_XXS, QType::GGUF_IQ3_S,
                                   QType::GGUF_IQ4_XS}) {
            const Matrix gate = make(format, 640, 2560, rng);
            for (const int tokens : {1, 2, 3, 4, 5, 8, 9}) { check(gate, tokens, rng); }
            require(gguf_cpu_supported(format), "an i-quant is not supported");
        }
        require(!gguf_cpu_supported(QType::GGUF_IQ2_XS), "an unimplemented format is supported");
        bool refused = false;
        try {
            GgufCpuActivation a(1, 2560);
            std::vector<std::uint16_t> x(2560 * 2);
            a.prepare_bf16(QType::GGUF_Q2_0, x.data(), 2560, 2);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "an activation above its capacity was accepted");
        std::cout << "gguf_expert_cpu: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gguf_expert_cpu: " << error.what() << '\n';
        return 1;
    }
}
