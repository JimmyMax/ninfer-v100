#include "ops/moe_experts/gguf_expert_cpu.h"

#include "ops/common/silu.h"
#include "ops/moe_experts/gguf_cpu_tables.h"

#include <immintrin.h>

#include <cstring>

// The scalar products' arithmetic (gguf_expert_cpu.cpp) eight slices at a time. Group g holds
// slices 8 g .. 8 g + 7, which are GPU lanes 8 (g mod 4) .. 8 (g mod 4) + 7: one vector of each
// token's four holds their partial sums, and groups g and g + 4 add to it in slice order.

namespace ninfer::ops::gguf_cpu_detail {
namespace {

inline float half_at(const std::uint8_t* p) {
    std::uint16_t bits;
    std::memcpy(&bits, p, 2);
    // The vector F16C form: MSVC does not declare the scalar _cvtsh_ss.
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// Unsigned codes times signed activations: eight int32 lanes, each the sum of four adjacent
// products.
inline __m256i dot_u8s8(__m256i codes, __m256i a) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes, a), _mm256_set1_epi16(1));
}

inline std::uint32_t load32(const std::uint8_t* p) {
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
}

// 32 sign bits, bit j for value j, as 32 bytes of -1 (bit set) or +1: ggml's bit-selector
// expansion (each byte broadcast to eight lanes, masked by its bit, compared).
inline __m256i sign_bytes(std::uint32_t bits) {
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                            2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i select = _mm256_set1_epi64x(0x8040201008040201LL);
    const __m256i set    = _mm256_and_si256(
        _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(bits)), spread), select);
    return _mm256_or_si256(_mm256_cmpeq_epi8(set, select), _mm256_set1_epi8(1));
}

// ggml's 16 IQ4_NL values for 16 bytes of two 4-bit indices: the low nibbles are values 0..15,
// the high ones 16..31.
inline __m256i iq4_values(const std::uint8_t* q) {
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
    const __m128i nib   = _mm_set1_epi8(15);
    return _mm256_set_m128i(_mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), nib)),
                            _mm_shuffle_epi8(table, _mm_and_si128(bytes, nib)));
}

// A group's eight slices: per slice magnitudes (unsigned) and signs (bytes whose sign is the
// weight's, never zero where the weight is not), so that its products are
// maddubs(magnitude, sign(a, sign)), and its factors (decode_slice's). A slice past the row's end
// has factors 0 and no codes.
struct Group {
    __m256i magnitude[8], sign[8];
    alignas(32) float factor[2][8];
};

// The integer dots of a group's eight slices for one token, lane i slice i's: its first 16 values
// and its last 16 (IQ2_S's halves, which have their own factors), or its whole dot in `first`.
struct Dots {
    __m256i first, second;
};

// The dots from each slice's eight int32 lanes of four products each (lanes 0..3: values 0..15).
// hadd sums adjacent lanes of two slices within each 128-bit half, so after two rounds the low half
// holds lanes 0..3 of four slices and the high half their lanes 4..7; the permutes put slices 0..7
// in order.
inline Dots halves8(const __m256i* u) {
    const __m256i h0123 =
        _mm256_hadd_epi32(_mm256_hadd_epi32(u[0], u[1]), _mm256_hadd_epi32(u[2], u[3]));
    const __m256i h4567 =
        _mm256_hadd_epi32(_mm256_hadd_epi32(u[4], u[5]), _mm256_hadd_epi32(u[6], u[7]));
    return {_mm256_permute2x128_si256(h0123, h4567, 0x20),
            _mm256_permute2x128_si256(h0123, h4567, 0x31)};
}

// The warp butterfly over a token's 32 lanes (four vectors of eight): lane l takes lane l + 16,
// then + 8, + 4, + 2, + 1.
inline float butterfly8(const __m256* lanes) {
    const __m256 c = _mm256_add_ps(_mm256_add_ps(lanes[0], lanes[2]), _mm256_add_ps(lanes[1], lanes[3]));
    const __m128 d = _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));
    const __m128 e = _mm_add_ps(d, _mm_movehl_ps(d, d));
    return _mm_cvtss_f32(_mm_add_ss(e, _mm_shuffle_ps(e, e, 1)));
}

// Eight per-slice values from `from`: the live ones, zero for slices past the row's end (a
// group of a 640-value row holds four).
inline __m256 live_floats(const float* from, int live) {
    if (live == 8) { return _mm256_loadu_ps(from); }
    alignas(32) float to[8];
    for (int i = 0; i < 8; ++i) { to[i] = i < live ? from[i] : 0.0F; }
    return _mm256_load_ps(to);
}

inline __m256i live_ints(const std::int32_t* from, int live) {
    if (live == 8) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from)); }
    alignas(32) std::int32_t to[8];
    for (int i = 0; i < 8; ++i) { to[i] = i < live ? from[i] : 0; }
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(to));
}

constexpr int kMaxTokens = 16;

// Every format's rows: `decode(row, g, live, group)` fills group g's slices (live of them) and
// `dot(t, g, live, group)` gives token t's integer dots of them (Dots for IQ2_S's halves, else the
// whole dots).
template <bool kHalves, class Decode, class Dot>
void rows_in_lane_order(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin,
                        int row_end, float* y, std::size_t y_stride, const Decode& decode,
                        const Dot& dot) {
    const int tokens = x.tokens();
    const int slices = w.k / 32;
    const int groups = (slices + 7) / 8;
    __m256 acc[kMaxTokens][4];
    Group group;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        for (int t = 0; t < tokens; ++t) {
            for (__m256& v : acc[t]) { v = _mm256_setzero_ps(); }
        }
        for (int g = 0; g < groups; ++g) {
            const int live = slices - 8 * g < 8 ? slices - 8 * g : 8;
            decode(r, g, live, group);
            const __m256 factor0 = _mm256_load_ps(group.factor[0]);
            for (int t = 0; t < tokens; ++t) {
                const __m256 a = live_floats(x.scales(t) + 8 * g, live);
                __m256 product;
                if constexpr (kHalves) {
                    const Dots d       = dot(t, g, live, group);
                    const __m256 inner = _mm256_fmadd_ps(
                        factor0, _mm256_cvtepi32_ps(d.first),
                        _mm256_mul_ps(_mm256_load_ps(group.factor[1]), _mm256_cvtepi32_ps(d.second)));
                    product = _mm256_mul_ps(a, inner);
                } else {
                    product = _mm256_mul_ps(_mm256_mul_ps(factor0, a),
                                            _mm256_cvtepi32_ps(dot(t, g, live, group)));
                }
                acc[t][g & 3] = _mm256_add_ps(acc[t][g & 3], product);
            }
        }
        for (int t = 0; t < tokens; ++t) { y[std::size_t(t) * y_stride + row] = butterfly8(acc[t]); }
    }
}

// An i-quant group is one 256-value block.
template <QType F>
inline void decode_iq(const std::uint8_t* block, Group& out) {
    const float d = half_at(block);
    for (int sub = 0; sub < 8; ++sub) {
        if constexpr (F == QType::GGUF_IQ2_S) {
            const std::uint8_t* qs = block + 2 + 4 * sub;
            const unsigned qh      = block[66 + sub];
            const unsigned scale   = block[74 + sub];
            out.magnitude[sub]     = _mm256_set_epi64x(
                static_cast<long long>(iq2s_grid[qs[3] | ((qh << 2) & 0x300U)]),
                static_cast<long long>(iq2s_grid[qs[2] | ((qh << 4) & 0x300U)]),
                static_cast<long long>(iq2s_grid[qs[1] | ((qh << 6) & 0x300U)]),
                static_cast<long long>(iq2s_grid[qs[0] | ((qh << 8) & 0x300U)]));
            out.sign[sub]      = sign_bytes(load32(block + 34 + 4 * sub));
            out.factor[0][sub] = d * float(2 * (scale & 15U) + 1) * 0.125F;
            out.factor[1][sub] = d * float(2 * (scale >> 4) + 1) * 0.125F;
        } else if constexpr (F == QType::GGUF_IQ3_XXS) {
            const std::uint8_t* qs  = block + 2 + 8 * sub;
            const std::uint32_t aux = load32(block + 66 + 4 * sub);
            out.magnitude[sub]      = _mm256_set_epi32(
                static_cast<int>(iq3xxs_grid[qs[7]]), static_cast<int>(iq3xxs_grid[qs[6]]),
                static_cast<int>(iq3xxs_grid[qs[5]]), static_cast<int>(iq3xxs_grid[qs[4]]),
                static_cast<int>(iq3xxs_grid[qs[3]]), static_cast<int>(iq3xxs_grid[qs[2]]),
                static_cast<int>(iq3xxs_grid[qs[1]]), static_cast<int>(iq3xxs_grid[qs[0]]));
            const std::uint32_t bits = std::uint32_t(ksigns_iq2xs[aux & 127U]) |
                                       std::uint32_t(ksigns_iq2xs[(aux >> 7) & 127U]) << 8 |
                                       std::uint32_t(ksigns_iq2xs[(aux >> 14) & 127U]) << 16 |
                                       std::uint32_t(ksigns_iq2xs[(aux >> 21) & 127U]) << 24;
            out.sign[sub]      = sign_bytes(bits);
            out.factor[0][sub] = d * float(2 * (aux >> 28) + 1) * 0.25F;
            out.factor[1][sub] = 0.0F;
        } else if constexpr (F == QType::GGUF_IQ3_S) {
            const std::uint8_t* qs = block + 2 + 8 * sub;
            const unsigned qh      = block[66 + sub];
            const unsigned scale   = (block[106 + sub / 2] >> (4 * (sub & 1))) & 15U;
            out.magnitude[sub]     = _mm256_set_epi32(
                static_cast<int>(iq3s_grid[qs[7] | ((qh << 1) & 256U)]),
                static_cast<int>(iq3s_grid[qs[6] | ((qh << 2) & 256U)]),
                static_cast<int>(iq3s_grid[qs[5] | ((qh << 3) & 256U)]),
                static_cast<int>(iq3s_grid[qs[4] | ((qh << 4) & 256U)]),
                static_cast<int>(iq3s_grid[qs[3] | ((qh << 5) & 256U)]),
                static_cast<int>(iq3s_grid[qs[2] | ((qh << 6) & 256U)]),
                static_cast<int>(iq3s_grid[qs[1] | ((qh << 7) & 256U)]),
                static_cast<int>(iq3s_grid[qs[0] | ((qh << 8) & 256U)]));
            out.sign[sub]      = sign_bytes(load32(block + 74 + 4 * sub));
            out.factor[0][sub] = d * float(2 * scale + 1);
            out.factor[1][sub] = 0.0F;
        } else {
            static_assert(F == QType::GGUF_IQ4_XS);
            std::uint16_t high;
            std::memcpy(&high, block + 2, 2);
            const unsigned low  = (block[4 + sub / 2] >> (4 * (sub & 1))) & 15U;
            const __m256i value = iq4_values(block + 8 + 16 * sub);
            out.magnitude[sub]  = _mm256_sign_epi8(value, value);
            out.sign[sub]       = value; // no IQ4_NL value is zero
            out.factor[0][sub] = d * float(int(low | (((unsigned(high) >> (2 * sub)) & 3U) << 4)) - 32);
            out.factor[1][sub] = 0.0F;
        }
    }
}

template <QType F>
void iq_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride) {
    constexpr int bytes   = gguf_block_shape(F).bytes;
    constexpr bool halves = F == QType::GGUF_IQ2_S;
    const auto decode     = [](const std::uint8_t* row, int g, int, Group& group) {
        decode_iq<F>(row + std::size_t(g) * bytes, group);
    };
    const auto dot = [&x](int t, int g, int, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i));
            u[i]            = dot_u8s8(group.magnitude[i], _mm256_sign_epi8(v, group.sign[i]));
        }
        const Dots d = halves8(u);
        if constexpr (halves) {
            return d;
        } else {
            return _mm256_add_epi32(d.first, d.second);
        }
    };
    rows_in_lane_order<halves>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

// Q2_0: a 64-value block is two slices, so a group is four blocks. Its 16 code bytes hold values
// 4 i + L at bits 2 L of byte i; the activations' lane order stores value 4 i + L at position
// 16 L + i, so the codes of L = 0, 1 (and 2, 3) pair with the block's first (second) 32 positions.
// Of a block's int32 lanes, 0, 1, 4 and 5 then belong to its first slice and 2, 3, 6 and 7 to its
// second, which hadd pairs up as halves8 does. Codes c stand for c - 1: each slice's sum of
// activation codes comes off its dot.
void q2_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride) {
    const auto decode = [](const std::uint8_t* row, int g, int live, Group& group) {
        const __m128i three = _mm_set1_epi8(3);
        for (int i = 0; i < 8; ++i) {
            group.factor[0][i] = i < live ? half_at(row + std::size_t(4 * g + i / 2) * 18) : 0.0F;
        }
        for (int b = 0; 2 * b < live; ++b) {
            const __m128i q =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + std::size_t(4 * g + b) * 18 + 2));
            group.magnitude[2 * b] =
                _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 2), three), _mm_and_si128(q, three));
            group.magnitude[2 * b + 1] = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 6), three),
                                                          _mm_and_si128(_mm_srli_epi16(q, 4), three));
        }
    };
    const auto dot = [&x](int t, int g, int live, const Group& group) {
        __m256i v[4];
        for (int b = 0; b < 4; ++b) {
            if (2 * b >= live) {
                v[b] = _mm256_setzero_si256();
                continue;
            }
            const std::int8_t* a = x.codes(t) + (4 * g + b) * 64;
            v[b] = _mm256_add_epi32(
                dot_u8s8(group.magnitude[2 * b], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))),
                dot_u8s8(group.magnitude[2 * b + 1],
                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32))));
        }
        const __m256i h01 = _mm256_hadd_epi32(v[0], v[1]), h23 = _mm256_hadd_epi32(v[2], v[3]);
        const __m256i dots = _mm256_add_epi32(_mm256_permute2x128_si256(h01, h23, 0x20),
                                              _mm256_permute2x128_si256(h01, h23, 0x31));
        return _mm256_sub_epi32(dots, live_ints(x.sums(t) + 8 * g, live));
    };
    rows_in_lane_order<false>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

// Q8_0's signed codes or IQ4_NL's values: a slice is a block.
template <bool kNl>
void block32_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                  float* y, std::size_t y_stride) {
    constexpr int bytes = kNl ? 18 : 34;
    const auto decode   = [](const std::uint8_t* row, int g, int live, Group& group) {
        for (int i = 0; i < 8; ++i) {
            if (i >= live) {
                group.factor[0][i] = 0.0F;
                group.magnitude[i] = group.sign[i] = _mm256_setzero_si256();
                continue;
            }
            const std::uint8_t* block = row + std::size_t(8 * g + i) * bytes;
            const __m256i codes =
                kNl ? iq4_values(block + 2)
                    : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 2));
            group.magnitude[i] = _mm256_sign_epi8(codes, codes);
            group.sign[i]      = codes;
            group.factor[0][i] = half_at(block);
        }
    };
    const auto dot = [&x](int t, int g, int live, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            // A slice past the row's end has no codes: its sign bytes zero the activation.
            const __m256i v = i < live ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i))
                                       : _mm256_setzero_si256();
            u[i] = dot_u8s8(group.magnitude[i], _mm256_sign_epi8(v, group.sign[i]));
        }
        const Dots d = halves8(u);
        return _mm256_add_epi32(d.first, d.second);
    };
    rows_in_lane_order<false>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

} // namespace

void rows_avx2(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
               float* y, std::size_t y_stride) {
    if (x.tokens() > kMaxTokens) { return rows_scalar(w, x, row_begin, row_end, y, y_stride); }
    switch (w.format) {
    case QType::GGUF_IQ2_S: return iq_rows<QType::GGUF_IQ2_S>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ3_XXS:
        return iq_rows<QType::GGUF_IQ3_XXS>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ3_S: return iq_rows<QType::GGUF_IQ3_S>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ4_XS:
        return iq_rows<QType::GGUF_IQ4_XS>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q2_0: return q2_rows(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ4_NL: return block32_rows<true>(w, x, row_begin, row_end, y, y_stride);
    default: return block32_rows<false>(w, x, row_begin, row_end, y, y_stride);
    }
}

// silu.h eight values at a time: the same operations in the same order, so the same bits.
void swiglu_avx2(const float* gate, const float* up, std::size_t count, std::uint16_t* out) {
    const __m256 one = _mm256_set1_ps(1.0F);
    std::size_t i    = 0;
    alignas(32) float products[8];
    for (; i + 8 <= count; i += 8) {
        const __m256 x = _mm256_loadu_ps(gate + i);
        const __m256 v = _mm256_xor_ps(x, _mm256_set1_ps(-0.0F)); // e^v, v = -x
        const __m256 n = _mm256_round_ps(_mm256_mul_ps(v, _mm256_set1_ps(1.44269504F)),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-0.693145751953125F), v);
        r        = _mm256_fmadd_ps(n, _mm256_set1_ps(-1.42860677e-6F), r);
        __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(1.9875691500e-4F), r, _mm256_set1_ps(1.3981999507e-3F));
        p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3F));
        p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2F));
        p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1F));
        p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1F));
        p        = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r);
        p        = _mm256_add_ps(p, one);
        // n is -126 .. 128 where the polynomial's value is kept; 2^128 is 2^127 times 2.
        const __m256i k = _mm256_cvttps_epi32(n);
        const __m256i clamped =
            _mm256_min_epi32(_mm256_max_epi32(k, _mm256_set1_epi32(-126)), _mm256_set1_epi32(127));
        const __m256 power = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_add_epi32(clamped, _mm256_set1_epi32(127)), 23));
        const __m256 beyond = _mm256_castsi256_ps(_mm256_cmpgt_epi32(k, _mm256_set1_epi32(127)));
        __m256 e = _mm256_mul_ps(_mm256_mul_ps(p, power), _mm256_blendv_ps(one, _mm256_set1_ps(2.0F), beyond));
        e = _mm256_blendv_ps(e, _mm256_setzero_ps(),
                             _mm256_cmp_ps(v, _mm256_set1_ps(-87.3365479F), _CMP_LT_OQ));
        e = _mm256_blendv_ps(e, _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000)),
                             _mm256_cmp_ps(v, _mm256_set1_ps(88.7228394F), _CMP_GE_OQ));
        e = _mm256_blendv_ps(e, v, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        const __m256 silu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_store_ps(products, _mm256_mul_ps(silu, _mm256_loadu_ps(up + i)));
        for (int j = 0; j < 8; ++j) { out[i + j] = float_to_bf16(products[j]); }
    }
    for (; i < count; ++i) { out[i] = float_to_bf16(ops::silu(gate[i]) * up[i]); }
}

} // namespace ninfer::ops::gguf_cpu_detail
