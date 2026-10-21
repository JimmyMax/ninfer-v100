// Standalone benchmark of the GGUF vector GEMV path (ops/linear/gguf vec kernel) on synthetic
// blocks with production layouts. Measures ms and effective GB/s per (type, rows, k, t).
//
// Build (in the sm70 buildenv container):
//   nvcc -O3 -arch=sm_70 -std=c++20 \
//     -I/root -I/root/third_party/ggml-quants/include -I/root/third_party/ggml-quants/src \
//     gguf_gemv_bench.cpp /root/src/ops/linear/gguf/ggml_bridge.cu -o gguf_gemv_bench

#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gguf = ninfer::ops::gguf;
using ninfer::QType;
using ninfer::Weight;

static gguf::GgmlType ggml_of(QType q) {
    switch (q) {
    case QType::GGUF_Q8_0:    return gguf::GgmlType::Q8_0;
    case QType::GGUF_Q2_K:    return gguf::GgmlType::Q2_K;
    case QType::GGUF_Q3_K:    return gguf::GgmlType::Q3_K;
    case QType::GGUF_Q4_K:    return gguf::GgmlType::Q4_K;
    case QType::GGUF_Q5_K:    return gguf::GgmlType::Q5_K;
    case QType::GGUF_Q6_K:    return gguf::GgmlType::Q6_K;
    case QType::GGUF_IQ2_XXS: return gguf::GgmlType::IQ2_XXS;
    case QType::GGUF_IQ2_XS:  return gguf::GgmlType::IQ2_XS;
    case QType::GGUF_IQ2_S:   return gguf::GgmlType::IQ2_S;
    case QType::GGUF_IQ3_XXS: return gguf::GgmlType::IQ3_XXS;
    case QType::GGUF_IQ3_S:   return gguf::GgmlType::IQ3_S;
    case QType::GGUF_IQ1_S:   return gguf::GgmlType::IQ1_S;
    case QType::GGUF_IQ1_M:   return gguf::GgmlType::IQ1_M;
    case QType::GGUF_IQ4_NL:  return gguf::GgmlType::IQ4_NL;
    case QType::GGUF_IQ4_XS:  return gguf::GgmlType::IQ4_XS;
    case QType::GGUF_Q4_0:    return gguf::GgmlType::Q4_0;
    case QType::GGUF_Q5_0:    return gguf::GgmlType::Q5_0;
    case QType::GGUF_Q2_0:    return gguf::GgmlType::Q2_0;
    }
    throw std::runtime_error("harness: unknown qtype");
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

struct Case {
    QType qtype;
    const char* name;
    std::vector<int> half_fields;
    int block_bytes = 0, block_elems = 0;
};

std::vector<std::uint8_t> random_blocks(const Case& c, int rows, int k, std::mt19937& rng) {
    const std::int64_t count = std::int64_t(rows) * (k / c.block_elems);
    std::vector<std::uint8_t> bytes(count * c.block_bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * c.block_bytes;
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
    }
    return bytes;
}

struct DeviceBuffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit DeviceBuffer(std::size_t n) : bytes(n) {
        check(cudaMalloc(&p, bytes), "alloc");
    }
    ~DeviceBuffer() {
        if (p) { cudaFree(p); }
    }
    void copy_from_host(const void* src, std::size_t n) {
        check(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice), "h2d");
    }
};

double time_product(const Case& c, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int t, DeviceBuffer& out, cudaStream_t stream) {
    gguf::VectorOutput vo;
    vo.bf16          = static_cast<__nv_bfloat16*>(out.p);
    vo.column_stride = rows;
    vo.epilogue      = gguf::Epilogue::Store;
    // Warmup + timed iterations.
    for (int i = 0; i < 5; ++i) {
        gguf::vector_product(ggml_of(c.qtype), weight, row_bytes, rows, k, activation, t, vo, stream);
    }
    check(cudaStreamSynchronize(stream), "sync");
    std::vector<double> times;
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    for (int i = 0; i < 30; ++i) {
        check(cudaEventRecord(a, stream), "rec");
        gguf::vector_product(ggml_of(c.qtype), weight, row_bytes, rows, k, activation, t, vo, stream);
        check(cudaEventRecord(b, stream), "rec");
        check(cudaEventSynchronize(b), "esync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, a, b), "elapsed");
        times.push_back(ms);
    }
    check(cudaEventDestroy(a), "ev");
    check(cudaEventDestroy(b), "ev");
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int device = 0;
    check(cudaSetDevice(device), "set device");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, device), "props");
    std::printf("device %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    std::mt19937 rng(20261021);
    const std::vector<Case> cases = {
        {QType::GGUF_IQ3_S, "iq3_s", {0}},
        {QType::GGUF_IQ4_XS, "iq4_xs", {0}},
        {QType::GGUF_IQ3_XXS, "iq3_xxs", {0}},
        {QType::GGUF_Q4_K, "q4_k", {0, 2}},
        {QType::GGUF_Q6_K, "q6_k", {208}},
    };
    // (rows, k) as in the Qwen3.8 27B artifact: MLP gate/up, down, GDN projections.
    const std::vector<std::pair<int, int>> shapes = {
        {17408, 5120},   // MLP gate/up (IQ3_S / IQ4_XS / Q4_K / IQ3_XXS)
        {5120, 17408},   // MLP down
        {6144, 5120},    // GDN out / small
        {2048, 10240},   // GDN input projection (narrow)
    };
    const int t_list[] = {1, 5};
    const std::size_t max_act = std::size_t(10240) * 8;

    std::printf("%-8s %-12s %-10s %-4s %12s %12s\n", "type", "shape", "bytes", "t", "ms", "GB/s");
    for (const auto& c0 : cases) {
        Case c = c0;
        const auto shape = gguf::block_shape(ggml_of(c.qtype));
        c.block_bytes = shape.bytes;
        c.block_elems = shape.elements;
        for (const auto& s : shapes) {
            const int rows = s.first, k = s.second;
            if (k % c.block_elems != 0) { continue; }
            std::printf("preparing %s %dx%d\n", c.name, rows, k);
            const std::int64_t row_bytes = std::int64_t(k / c.block_elems) * c.block_bytes;
            std::mt19937 brng(rng());
            auto host = random_blocks(c, rows, k, brng);
            DeviceBuffer blocks(host.size() + 4096);
            check(cudaMemset(static_cast<char*>(blocks.p) + host.size(), 0, 4096), "tail");
            blocks.copy_from_host(host.data(), host.size());
            // Activation: random N(0,1) BF16 [k, t_max], quantized per column count below.
            DeviceBuffer act_buf(gguf::vector_activation_bytes(k, 8));
            for (int t : t_list) {
                DeviceBuffer x(std::size_t(k) * t * 2);
                std::vector<__nv_bfloat16> hx(std::size_t(k) * t);
                std::normal_distribution<float> unit(0.0f, 1.0f);
                for (auto& v : hx) { v = __float2bfloat16(unit(brng)); }
                check(cudaMemcpy(x.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "act");
                gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.p), k, t,
                                                  nullptr, act_buf.p, 0);
                DeviceBuffer out(std::size_t(rows) * t * 2);
                const double ms = time_product(c, blocks.p, row_bytes, rows, k, act_buf.p, t, out,
                                               0);
                const double gb = std::size_t(rows) * row_bytes / (ms * 1e3);
                std::printf("%-8s %-12s %-10zu %-4d %12.3f %12.1f\n", c.name,
                            (std::to_string(rows) + "x" + std::to_string(k)).c_str(),
                            std::size_t(rows) * row_bytes, t, ms, gb);
            }
        }
    }
    return 0;
}
