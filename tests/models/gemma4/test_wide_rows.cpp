// 本仓库 V100 增量（docs/v100/gemma4-wide-rows-2026-10-17.md）：Gemma 4 行内核的 FP64 oracle。
//
// 背景：rms_norm / attn_residual / ffn_combine 原先每线程负责一个 float4，因此 hidden/4 不能超过
// 1024 线程（一行一个 block，hidden <= 4096）。dense 的 Gemma 4 31B hidden = 5376 装不下，于是
// 新增了按 blockDim.x 步进的宽行内核，只在 hidden/4 > 1024 时分派。这个测试用同一份随机输入同时
// 跑窄（2816）与宽（5376）两条路径，两者都必须与同一个 FP64 参考一致——这样宽行实现既有独立的
// 正确性证据，也保证它与既有路径数值等价（窄行内核本身逐字未改）。

#include "models/gemma4/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace k = ninfer::models::gemma4::kernels;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

struct Device {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Device(std::size_t n) : bytes(n) {
        if (cudaMalloc(&p, n) != cudaSuccess) { throw std::runtime_error("cudaMalloc"); }
    }
    ~Device() { if (p != nullptr) { cudaFree(p); } }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const void* src) { cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice); }
    void get(void* dst) const { cudaMemcpy(dst, p, bytes, cudaMemcpyDeviceToHost); }
};

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = __float2bfloat16(v[i]); }
    return out;
}

std::vector<float> from_bf16(const std::vector<__nv_bfloat16>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = __bfloat162float(v[i]); }
    return out;
}

double rms(const std::vector<double>& row, double eps) {
    double ss = 0;
    for (const double v : row) { ss += v * v; }
    return std::sqrt(ss / double(row.size()) + eps);
}

bool close(double got, double want, double rel, double abs) {
    return std::fabs(got - want) <= abs + rel * std::fabs(want);
}

// 每个 hidden 跑一遍三个内核，和 FP64 参考比。
void run(int hidden, std::mt19937& rng) {
    constexpr int kTokens = 4;
    const double eps      = 1e-6;
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> x(std::size_t(kTokens) * hidden), w(hidden, 1.0f);
    for (auto& v : x) { v = normal(rng); }
    for (int i = 0; i < hidden; ++i) { w[i] = 0.5f + 0.01f * float(i % 7); }

    // ---- rms_norm ----
    {
        std::vector<__nv_bfloat16> out(std::size_t(kTokens) * hidden);
        Device dx(x.size() * 4), dw(w.size() * 4), dout(out.size() * 2);
        dx.put(x.data());
        dw.put(w.data());
        k::rms_norm(static_cast<const float*>(dx.p), static_cast<const float*>(dw.p), hidden,
                    kTokens, float(eps), static_cast<__nv_bfloat16*>(dout.p), nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("sync"); }
        dout.get(out.data());
        const auto got = from_bf16(out);
        int bad = 0;
        double worst = 0;
        for (int t = 0; t < kTokens; ++t) {
            std::vector<double> row(x.begin() + std::size_t(t) * hidden,
                                    x.begin() + std::size_t(t + 1) * hidden);
            const double r = rms(row, eps);
            for (int i = 0; i < hidden; ++i) {
                const double want = row[i] / r * double(w[i]);
                const double d    = std::fabs(got[std::size_t(t) * hidden + i] - want);
                worst = std::max(worst, d);
                if (!close(got[std::size_t(t) * hidden + i], want, 2e-2, 2e-2)) { ++bad; }
            }
        }
        expect(bad == 0, "rms_norm hidden=" + std::to_string(hidden) + " (worst " +
                             std::to_string(worst) + ")");
    }

    // ---- attn_residual ----
    {
        std::vector<float> h(std::size_t(kTokens) * hidden), o(std::size_t(kTokens) * hidden),
            w_post(hidden), ffn_norm(hidden);
        for (auto& v : h) { v = normal(rng); }
        for (auto& v : o) { v = normal(rng); }
        for (auto& v : w_post) { v = 0.5f + 0.01f * float(int(&v - w_post.data()) % 5); }
        for (auto& v : ffn_norm) { v = 0.5f + 0.01f * float(int(&v - ffn_norm.data()) % 5); }
        const auto ob  = to_bf16(o);
        const auto h0  = h;
        std::vector<__nv_bfloat16> mlp(std::size_t(kTokens) * hidden);
        Device dh(h.size() * 4), dob(ob.size() * 2), dwp(w_post.size() * 4),
            dfn(ffn_norm.size() * 4), dmlp(mlp.size() * 2);
        dh.put(h.data());
        dob.put(ob.data());
        dwp.put(w_post.data());
        dfn.put(ffn_norm.data());
        k::AttnResidualArgs a;
        a.h        = static_cast<float*>(dh.p);
        a.o        = static_cast<const __nv_bfloat16*>(dob.p);
        a.w_post   = static_cast<const float*>(dwp.p);
        a.ffn_norm = static_cast<const float*>(dfn.p);
        a.mlp_in   = static_cast<__nv_bfloat16*>(dmlp.p);
        a.hidden   = hidden;
        a.tokens   = kTokens;
        a.eps      = float(eps);
        k::attn_residual(a, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("sync"); }
        dh.get(h.data());
        dmlp.get(mlp.data());
        const auto obf = from_bf16(ob);
        const auto mf  = from_bf16(mlp);
        int bad_h = 0, bad_mlp = 0;
        for (int t = 0; t < kTokens; ++t) {
            const std::size_t base = std::size_t(t) * hidden;
            std::vector<double> orow(obf.begin() + base, obf.begin() + base + hidden);
            const double ro = rms(orow, eps);
            std::vector<double> hrow(hidden);
            for (int i = 0; i < hidden; ++i) {
                hrow[i] = double(h0[base + i]) + orow[i] / ro * double(w_post[i]);
                if (!close(h[base + i], hrow[i], 1e-4, 1e-4)) { ++bad_h; }
            }
            const double rh = rms(hrow, eps);
            for (int i = 0; i < hidden; ++i) {
                const double want = hrow[i] / rh * double(ffn_norm[i]);
                if (!close(mf[base + i], want, 2e-2, 2e-2)) { ++bad_mlp; }
            }
        }
        expect(bad_h == 0, "attn_residual h hidden=" + std::to_string(hidden));
        expect(bad_mlp == 0, "attn_residual mlp hidden=" + std::to_string(hidden));
    }

    // ---- ffn_combine（dense：有 w_dense、无 MoE）----
    {
        std::vector<float> h(std::size_t(kTokens) * hidden), dense(std::size_t(kTokens) * hidden),
            w_dense(hidden), w_post(hidden), next_norm(hidden), scalar{1.25f};
        for (auto& v : h) { v = normal(rng); }
        for (auto& v : dense) { v = normal(rng); }
        for (int i = 0; i < hidden; ++i) {
            w_dense[i]   = 0.5f + 0.01f * float(i % 5);
            w_post[i]    = 0.5f + 0.01f * float(i % 3);
            next_norm[i] = 0.5f + 0.01f * float(i % 4);
        }
        const auto db = to_bf16(dense);
        const auto h0 = h;
        std::vector<__nv_bfloat16> next_out(std::size_t(kTokens) * hidden);
        Device dh(h.size() * 4), ddb(db.size() * 2), dwd(w_dense.size() * 4),
            dwp(w_post.size() * 4), dnn(next_norm.size() * 4), dsc(4), dno(next_out.size() * 2);
        dh.put(h.data());
        ddb.put(db.data());
        dwd.put(w_dense.data());
        dwp.put(w_post.data());
        dnn.put(next_norm.data());
        dsc.put(scalar.data());
        k::FfnCombineArgs a;
        a.h         = static_cast<float*>(dh.p);
        a.dense     = static_cast<const __nv_bfloat16*>(ddb.p);
        a.w_dense   = static_cast<const float*>(dwd.p);
        a.w_post    = static_cast<const float*>(dwp.p);
        a.next_norm = static_cast<const float*>(dnn.p);
        a.scalar    = static_cast<const float*>(dsc.p);
        a.next_out  = static_cast<__nv_bfloat16*>(dno.p);
        a.hidden    = hidden;
        a.tokens    = kTokens;
        a.eps       = float(eps);
        k::ffn_combine(a, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("sync"); }
        dh.get(h.data());
        dno.get(next_out.data());
        const auto df = from_bf16(db);
        const auto nf = from_bf16(next_out);
        int bad_h = 0, bad_next = 0;
        for (int t = 0; t < kTokens; ++t) {
            const std::size_t base = std::size_t(t) * hidden;
            const std::vector<double> drow(df.begin() + base, df.begin() + base + hidden);
            const double rd = rms(drow, eps);
            std::vector<double> yrow(hidden);
            for (int i = 0; i < hidden; ++i) {
                yrow[i] = drow[i] / rd * double(w_dense[i]);
            }
            const double ry = rms(yrow, eps);
            std::vector<double> hrow(hidden);
            for (int i = 0; i < hidden; ++i) {
                hrow[i] =
                    (double(h0[base + i]) + yrow[i] / ry * double(w_post[i])) * double(scalar[0]);
                if (!close(h[base + i], hrow[i], 1e-4, 1e-4)) { ++bad_h; }
            }
            const double rh = rms(hrow, eps);
            for (int i = 0; i < hidden; ++i) {
                const double want = hrow[i] / rh * double(next_norm[i]);
                if (!close(nf[base + i], want, 2e-2, 2e-2)) { ++bad_next; }
            }
        }
        expect(bad_h == 0, "ffn_combine h hidden=" + std::to_string(hidden));
        expect(bad_next == 0, "ffn_combine next hidden=" + std::to_string(hidden));
    }
}

} // namespace

int main() {
    int device = 0;
    if (cudaGetDeviceCount(&device) != cudaSuccess || device == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(20261017u);
        // 2816 = Gemma 4 26B-A4B 的 hidden（窄行路径），5376 = dense 31B（宽行路径）。
        for (const int hidden : {2816, 5376}) { run(hidden, rng); }
    } catch (const std::exception& e) {
        std::cerr << "unexpected exception: " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma 4 wide-row kernels\n";
    return failures == 0 ? 0 : 1;
}
