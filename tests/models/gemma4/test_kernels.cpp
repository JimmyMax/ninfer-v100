// Gemma 4's own decode kernels against FP64 oracles:
//  - decode_attention over an F16 cache, both head geometries (D 256 with two query heads per KV
//    head, D 512 with eight), linear and sliding-window ring caches, one and many splits, short
//    and long contexts and two sequences in one batch; the BF16 rows and the q8_1 rows; and keys
//    derived from the values (K = V blocks: RoPE(V * k_norm), proportional over 64 of 256 pairs).
//  - route_select: the logits, the top-k ids and weights and the expert grouping of the pairs.

#include "core/arena.h"
#include "models/gemma4/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace k = ninfer::models::gemma4::kernels;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

struct AttentionCase {
    int head_dim, query_heads, kv_heads, ring, window, splits;
    std::vector<int> positions; // one per batch row
    bool derived = false;       // keys derived from the values
    int group    = 1;           // rows per sequence: a verification's consecutive positions
};

void attention(const AttentionCase& c, std::mt19937& rng) {
    const int d = c.head_dim, b = int(c.positions.size());
    const int max_pos = *std::max_element(c.positions.begin(), c.positions.end());
    const int slots   = c.ring != 0 ? c.ring : max_pos + 1;
    const std::int64_t head_stride = std::int64_t(slots) * d;
    const std::int64_t seq_stride  = head_stride * c.kv_heads;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__half> kc(seq_stride * b), vc(seq_stride * b);
    for (auto& v : kc) { v = __float2half(0.25f * normal(rng)); }
    for (auto& v : vc) { v = __float2half(normal(rng)); }
    std::vector<float> q(std::size_t(b) * c.query_heads * d);
    for (auto& v : q) { v = normal(rng); }
    std::vector<std::int32_t> seqs(b), pos(c.positions.begin(), c.positions.end());
    for (int r = 0; r < b; ++r) { seqs[r] = r / c.group; }
    // Derived keys: k_norm gammas near one, Gemma's full-attention frequencies (theta 1e6, the
    // first 64 pairs turn), and the oracle's keys RoPE(V * k_norm) rounded to F16 from the stored V.
    std::vector<float> k_norm(d), inv_freq(d / 2, 0.0f);
    for (auto& w : k_norm) { w = 1.0f + 0.25f * normal(rng); }
    for (int i = 0; i < 64 && i < d / 2; ++i) {
        inv_freq[i] = float(1.0 / std::pow(1e6, double(2 * i) / double(d)));
    }
    if (c.derived) {
        for (int r = 0; r < b; ++r) {
            for (int kv = 0; kv < c.kv_heads; ++kv) {
                for (int j = 0; j < slots; ++j) {
                    const std::int64_t at = r * seq_stride + kv * head_stride + std::int64_t(j) * d;
                    for (int i = 0; i < d / 2; ++i) {
                        float co = 1.0f, si = 0.0f;
                        if (inv_freq[i] != 0.0f) {
                            const float angle = float(j) * inv_freq[i];
                            co                = std::cos(angle);
                            si                = std::sin(angle);
                        }
                        const double n0 = double(__half2float(vc[at + i])) * k_norm[i];
                        const double n1 = double(__half2float(vc[at + i + d / 2])) * k_norm[i + d / 2];
                        kc[at + i]         = __float2half(float(n0 * co - n1 * si));
                        kc[at + i + d / 2] = __float2half(float(n1 * co + n0 * si));
                    }
                }
            }
        }
    }
    DeviceBuffer dw(k_norm.size() * 4), df(inv_freq.size() * 4), dr(std::size_t(slots) * 64 * 8);
    dw.copy_from_host(k_norm.data(), k_norm.size() * 4);
    df.copy_from_host(inv_freq.data(), inv_freq.size() * 4);
    k::rope_table(static_cast<const float*>(df.p), 64, slots, static_cast<float2*>(dr.p), nullptr);

    DeviceBuffer dk(kc.size() * 2), dv(vc.size() * 2), dq(q.size() * 4), ds(b * 4), dp(b * 4);
    dk.copy_from_host(kc.data(), kc.size() * 2);
    dv.copy_from_host(vc.data(), vc.size() * 2);
    dq.copy_from_host(q.data(), q.size() * 4);
    ds.copy_from_host(seqs.data(), b * 4);
    dp.copy_from_host(pos.data(), b * 4);
    const std::size_t floats =
        k::decode_attention_workspace_floats(b, c.query_heads, d, c.splits);
    DeviceBuffer partial(floats * 4), out(std::size_t(b) * c.query_heads * d * 2);
    const int width = c.query_heads * d;
    DeviceBuffer q8(std::size_t(b) * width + std::size_t(b) * (width / 32) * 4 + 64);

    k::DecodeAttentionArgs a;
    a.q = static_cast<const float*>(dq.p);
    a.k_cache = c.derived ? nullptr : static_cast<const __half*>(dk.p);
    a.k_norm = static_cast<const float*>(dw.p);
    a.rope = static_cast<const float2*>(dr.p);
    a.rope_pairs = 64;
    a.v_cache = static_cast<const __half*>(dv.p);
    a.seq_stride = seq_stride;
    a.head_stride = head_stride;
    a.sequences = static_cast<const std::int32_t*>(ds.p);
    a.positions = static_cast<const std::int32_t*>(dp.p);
    a.batch = b;
    a.query_heads = c.query_heads;
    a.kv_heads = c.kv_heads;
    a.head_dim = d;
    a.ring = c.ring;
    a.window = c.window;
    a.splits = c.splits;
    a.scale = 1.0f;
    a.partial = static_cast<float*>(partial.p);
    a.out = static_cast<__nv_bfloat16*>(out.p);
    k::decode_attention(a, nullptr);
    check(cudaDeviceSynchronize(), "decode_attention");
    std::vector<__nv_bfloat16> got(std::size_t(b) * width);
    out.copy_to_host(got.data(), got.size() * 2);
    // The q8_1 form of the same rows.
    a.out   = nullptr;
    a.out_q.qs = static_cast<std::int8_t*>(q8.p);
    a.out_q.ds = reinterpret_cast<half2*>(static_cast<std::uint8_t*>(q8.p) + std::size_t(b) * width);
    k::decode_attention(a, nullptr);
    check(cudaDeviceSynchronize(), "decode_attention q8");
    std::vector<std::int8_t> got_q(std::size_t(b) * width);
    std::vector<half2> got_ds(std::size_t(b) * (width / 32));
    check(cudaMemcpy(got_q.data(), a.out_q.qs, got_q.size(), cudaMemcpyDeviceToHost), "q8 qs");
    check(cudaMemcpy(got_ds.data(), a.out_q.ds, got_ds.size() * 4, cudaMemcpyDeviceToHost), "q8 ds");

    const int group = c.query_heads / c.kv_heads;
    double num = 0, den = 0, qnum = 0;
    for (int r = 0; r < b; ++r) {
        const int p  = c.positions[r];
        const int lo = c.ring != 0 ? std::max(0, p - c.window + 1) : 0;
        for (int h = 0; h < c.query_heads; ++h) {
            const int kv = h / group;
            const float* qh = q.data() + (std::size_t(r) * c.query_heads + h) * d;
            std::vector<double> score;
            double m = -1e300;
            for (int j = lo; j <= p; ++j) {
                const int slot = c.ring != 0 ? j % c.ring : j;
                const __half* kr = kc.data() + seqs[r] * seq_stride + kv * head_stride + std::int64_t(slot) * d;
                double s = 0;
                for (int i = 0; i < d; ++i) { s += double(qh[i]) * __half2float(kr[i]); }
                score.push_back(s);
                m = std::max(m, s);
            }
            double l = 0;
            for (auto& s : score) { s = std::exp(s - m); l += s; }
            for (int i = 0; i < d; ++i) {
                double o = 0;
                for (int j = lo; j <= p; ++j) {
                    const int slot = c.ring != 0 ? j % c.ring : j;
                    o += score[j - lo] *
                         __half2float(vc[seqs[r] * seq_stride + kv * head_stride + std::int64_t(slot) * d + i]);
                }
                o /= l;
                const std::size_t at = std::size_t(r) * width + std::size_t(h) * d + i;
                const double e = double(__bfloat162float(got[at])) - o;
                num += e * e;
                den += o * o;
                const float dq = __low2float(got_ds[at / 32]);
                const double eq = double(got_q[at]) * dq - o;
                qnum += eq * eq;
            }
        }
    }
    const double rms = std::sqrt(num / den), qrms = std::sqrt(qnum / den);
    const std::string name = "decode_attention D" + std::to_string(d) + " ring " + std::to_string(c.ring) +
                             " splits " + std::to_string(c.splits) + " pos " + std::to_string(max_pos) +
                             (c.derived ? " derived keys" : "") +
                             (c.group > 1 ? " rows of one sequence " + std::to_string(c.group) : "");
    const bool ok = rms < 6e-3 && qrms < 1.5e-2;
    std::cout << (ok ? "OK   " : "FAIL ") << name << " rms=" << rms << " q8 rms=" << qrms << '\n';
    failures += ok ? 0 : 1;
}

void route(int tokens, std::mt19937& rng) {
    constexpr int kHidden = 2816, kExperts = 128, kTopK = 8;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(std::size_t(tokens) * kHidden), router(std::size_t(kExperts) * kHidden),
        scale(kExperts);
    for (auto& v : x) { v = normal(rng) * 0.02f; }
    for (auto& v : router) { v = normal(rng); }
    std::uniform_real_distribution<float> unit(0.5f, 1.5f);
    for (auto& v : scale) { v = unit(rng); }
    const int pairs = tokens * kTopK;
    DeviceBuffer dx(x.size() * 4), dr(router.size() * 4), dsc(scale.size() * 4),
        logits(std::size_t(tokens) * kExperts * 4), ids(pairs * 4), weights(pairs * 4),
        routing((kExperts + 1 + pairs + kExperts + 1) * 4);
    dx.copy_from_host(x.data(), x.size() * 4);
    dr.copy_from_host(router.data(), router.size() * 4);
    dsc.copy_from_host(scale.data(), scale.size() * 4);
    auto* base = static_cast<std::int32_t*>(routing.p);
    k::route_select(static_cast<const float*>(dx.p), kHidden, tokens, static_cast<const float*>(dr.p),
                    static_cast<const float*>(dsc.p), kExperts, kTopK, static_cast<float*>(logits.p),
                    static_cast<std::int32_t*>(ids.p), static_cast<float*>(weights.p), base,
                    base + kExperts + 1, base + kExperts + 1 + pairs,
                    base + kExperts + 1 + pairs + kExperts, nullptr);
    check(cudaDeviceSynchronize(), "route_select");
    std::vector<std::int32_t> got_ids(pairs), arrays(kExperts + 1 + pairs + kExperts + 1);
    std::vector<float> got_w(pairs);
    ids.copy_to_host(got_ids.data(), pairs * 4);
    weights.copy_to_host(got_w.data(), pairs * 4);
    routing.copy_to_host(arrays.data(), arrays.size() * 4);
    const std::int32_t* bounds = arrays.data();
    const std::int32_t* sorted = bounds + kExperts + 1;
    const std::int32_t* active = sorted + pairs;
    const std::int32_t count   = active[kExperts];
    bool ok = true;
    for (int t = 0; t < tokens; ++t) {
        std::vector<std::pair<double, int>> l(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            double s = 0;
            for (int i = 0; i < kHidden; ++i) {
                s += double(x[std::size_t(t) * kHidden + i]) * router[std::size_t(e) * kHidden + i];
            }
            l[e] = {s, e};
        }
        std::sort(l.begin(), l.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        double sum = 0;
        for (int j = 0; j < kTopK; ++j) { sum += std::exp(l[j].first - l[0].first); }
        // Near ties can order differently in FP32; compare as sets with their weights.
        std::map<int, double> want;
        for (int j = 0; j < kTopK; ++j) {
            want[l[j].second] = std::exp(l[j].first - l[0].first) / sum * scale[l[j].second];
        }
        for (int j = 0; j < kTopK; ++j) {
            const int e = got_ids[t * kTopK + j];
            if (!want.contains(e) || std::abs(want[e] - got_w[t * kTopK + j]) > 1e-4) { ok = false; }
        }
    }
    // The grouping: every pair once, under its own expert, active experts ascending with pairs.
    std::set<int> seen;
    int listed = 0;
    for (int e = 0; e < kExperts; ++e) {
        const int n = bounds[e + 1] - bounds[e];
        if (n < 0) { ok = false; }
        if (n > 0) {
            if (listed >= count || active[listed] != e) { ok = false; }
            ++listed;
        }
        for (int i = bounds[e]; i < bounds[e + 1]; ++i) {
            if (got_ids[sorted[i]] != e || !seen.insert(sorted[i]).second) { ok = false; }
        }
    }
    ok = ok && listed == count && int(seen.size()) == pairs && bounds[0] == 0 && bounds[kExperts] == pairs;
    std::cout << (ok ? "OK   " : "FAIL ") << "route_select T=" << tokens << '\n';
    failures += ok ? 0 : 1;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(77);
        // Sliding layers: D 256, 16 query heads over 8, a 1024-token window in a ring of exactly
        // the window (as a sequence keeps it) and in a wider one.
        attention({256, 16, 8, 1024, 1024, 16, {5}}, rng);
        attention({256, 16, 8, 1024, 1024, 16, {700, 1900}}, rng);
        attention({256, 16, 8, 1280, 1024, 1, {3000}}, rng);
        // A verification's rows: groups of one sequence's consecutive positions sharing a block.
        attention({256, 16, 8, 1280, 1024, 16, {3000, 3001, 3002, 3003}, false, 4}, rng);
        attention({256, 16, 8, 1280, 1024, 1, {500, 501, 2000, 2001}, false, 2}, rng);
        attention({256, 16, 8, 1280, 1024, 16, {0, 1, 2, 3}, false, 4}, rng);
        attention({256, 16, 8, 1280, 1024, 16, {1022, 1023, 1024}, false, 3}, rng);
        // Full layers: D 512, 16 query heads over 2, a linear cache.
        attention({512, 16, 2, 0, 0, 123, {0}}, rng);
        attention({512, 16, 2, 0, 0, 123, {40, 4999}}, rng);
        attention({512, 16, 2, 0, 0, 1, {777}}, rng);
        attention({512, 16, 2, 0, 0, 123, {0}, true}, rng);
        attention({512, 16, 2, 0, 0, 123, {40, 4999}, true}, rng);
        attention({512, 16, 2, 0, 0, 1, {777}, true}, rng);
        for (const int tokens : {1, 3, 8}) { route(tokens, rng); }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma 4 kernels\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma 4 kernels: " << error.what() << '\n';
        return 1;
    }
}
