// 本仓库 V100 增量（docs/v100/PLAN.md）：Q6_K 的加载时密集 6-bit 码字流 prepack 的数值 oracle。
//
// permute_q6k_to_dense 把同一行的超块序原地重排成逐行密集码字流（元素 e 在位 6e），载荷尺寸不
// 变（210 B/超块），从不存入 `.ninfer`。宽度覆盖按 27B 制品的实际消费路径选：
//   * t <= 8：两遍同一载荷各走自己的向量解码（密集版与超块版各自的内核，同形累加），要求逐位一致；
//   * t 9..64：原生走 DP4A 矩阵核、密集走 8 列分块向量（矩阵核不认密集布局）——两条路线累加
//     顺序不同，不能互相比位；两条路线都对照 CPU 的 FP64 参考在 BF16 存储误差的界内；
//   * t >= 65：sm70 按需反量化 + cuBLAS，两遍同形，要求逐位一致；
//   * dequantize_rows 的 FP32 输出逐位一致——这是重排变换本身的精确验证；
//   * embedding gather 是第四条消费路径，同形逐位一致。
#include "artifact/permute_q6k_dense.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;

// 两种布局解码的是同一批码字，所以要求逐位一致。
constexpr double kRelativeL2 = 0.0;
// t 9..64 的两条路线各与 FP64 参考对照：误差由 Q8_1 激活量化主导（仓库 GGUF 单产品 oracle
// 的预算，与 test_gguf 一致）。
constexpr double kToleranceL2 = 2.0e-2;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// 随机 Q6_K 超块；binary16 的 d（偏移 208）给有限值。
std::vector<std::uint8_t> random_blocks(std::int32_t rows, std::int32_t k, std::mt19937& rng) {
    const std::int64_t count = std::int64_t(rows) * (k / 256);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(count) * 210);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        const __half h = __float2half(scale(rng));
        std::memcpy(bytes.data() + static_cast<std::size_t>(i) * 210 + 208, &h, sizeof(h));
    }
    return bytes;
}

Weight weight_of(const void* data, QuantLayout layout, std::int32_t rows, std::int32_t k) {
    Weight w;
    w.qtype   = QType::GGUF_Q6_K;
    w.layout  = layout;
    w.qdata   = data;
    w.payload = data;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim       = 2;
    w.group_size = 256;
    w.group      = 256;
    return w;
}

int compare(const std::string& label, const std::vector<__nv_bfloat16>& native,
            const std::vector<__nv_bfloat16>& packed) {
    if (native.size() != packed.size() || native.empty()) {
        std::cerr << label << ": invalid comparison\n";
        return 1;
    }
    double squared           = 0.0;
    double reference_squared = 0.0;
    bool finite              = true;
    for (std::size_t i = 0; i < native.size(); ++i) {
        const float a = __bfloat162float(native[i]);
        const float b = __bfloat162float(packed[i]);
        const double d = double(b) - double(a);
        squared += d * d;
        reference_squared += double(a) * double(a);
        finite = finite && std::isfinite(a) && std::isfinite(b);
    }
    const double relative = std::sqrt(squared / static_cast<double>(native.size())) /
                           std::sqrt(reference_squared / static_cast<double>(native.size()));
    std::cout << label << ": relative_l2=" << relative << '\n';
    if (!finite) {
        std::cerr << label << ": non-finite output\n";
        return 1;
    }
    if (!(relative <= kRelativeL2)) {
        std::cerr << label << ": the dense layout differs from the artifact layout\n";
        return 1;
    }
    return 0;
}

int read_bf16(DeviceBuffer& device, std::size_t elements, std::vector<__nv_bfloat16>& values) {
    values.resize(elements);
    device.copy_to_host(values.data(), elements * sizeof(__nv_bfloat16));
    return 0;
}

// 普通 Linear：两遍同一载荷，比较 BF16 输出。exact 时要求逐位一致（两条路线同形累加）；否则
// 两遍都对照 CPU 的 FP64 参考在容差界内（t 9..64：DP4A 矩阵核 vs 分块向量，累加顺序不同）。
int run_product(const std::vector<std::uint8_t>& blocks, std::int32_t rows, std::int32_t k,
                std::int32_t t, bool exact) {
    DeviceBuffer native_device(blocks.size() + 4096);
    native_device.fill(0);
    native_device.copy_from_host(blocks.data(), blocks.size());
    DeviceBuffer dense_device(blocks.size() + 4096);
    dense_device.fill(0);
    dense_device.copy_from_host(blocks.data(), blocks.size());

    const std::int64_t row_bytes = std::int64_t(k / 256) * 210;
    Weight native = weight_of(native_device.p, QuantLayout::GgufBlocks, rows, k);
    Weight dense  = weight_of(dense_device.p, QuantLayout::VoltaQ6KDense, rows, k);
    {
        WeightGeometry geometry;
        geometry.format = QType::GGUF_Q6_K;
        geometry.layout = QuantLayout::GgufBlocks;
        geometry.shape  = {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(k)};
        permute_q6k_to_dense(geometry, static_cast<std::byte*>(dense_device.p), nullptr);
    }

    std::mt19937 rng(41U + static_cast<std::uint32_t>(t));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(k) * t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    const std::size_t bytes = ops::linear_workspace_capacity_bytes(QType::GGUF_Q6_K, rows, k,
                                                                  ops::LinearPolicy::A16Only, t, t);
    std::vector<__nv_bfloat16> out_native, out_dense;
    for (const Weight* w : {&native, &dense}) {
        DeviceBuffer out_device(std::size_t(rows) * t * sizeof(__nv_bfloat16));
        WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
        Tensor x(const_cast<void*>(x_device.p), DType::BF16, {k, t});
        Tensor out(const_cast<void*>(out_device.p), DType::BF16, {rows, t});
        ops::linear(x, *w, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "synchronize the linear op");
        auto& values = w == &native ? out_native : out_dense;
        read_bf16(out_device, std::size_t(rows) * t, values);
    }
    const std::string label = "Q6_K dense prepack linear t=" + std::to_string(t) + " k=" +
                             std::to_string(k);
    if (exact) {
        return compare(label, out_native, out_dense);
    }
    // 两条路线都对齐到独立的 FP64 参考：解码与累加都用双精度，x 的 BF16 值在双精度里精确。
    std::vector<double> reference(std::size_t(rows) * t);
    for (std::int32_t r = 0; r < rows; ++r) {
        const std::uint8_t* rowb = blocks.data() + std::size_t(r) * (k / 256) * 210;
        for (std::int32_t e = 0; e < k; ++e) {
            const std::uint8_t* blk = rowb + (e / 256) * 210;
            const int m  = (e % 256) / 32, il = e % 32;
            const std::uint8_t* ql = blk + 64 * (m / 4) + 32 * (m & 1) + il;
            const std::uint8_t* qh = blk + 128 + 32 * (m / 4) + il;
            const int lo           = ((m >> 1) & 1) ? (ql[0] >> 4) & 0xF : ql[0] & 0xF;
            const int hi           = (qh[0] >> (2 * (m % 4))) & 3;
            std::uint16_t h;
            std::memcpy(&h, blk + 208, sizeof(h));
            const std::uint32_t ebits = (h >> 10) & 0x1F, mants = h & 0x3FF;
            const double d = ebits == 0
                               ? double(mants) * 5.9604644775390625e-8
                               : (1.0 + double(mants) / 1024.0) * std::ldexp(1.0, int(ebits) - 15);
            const double value =
                d * double(reinterpret_cast<const std::int8_t*>(blk + 192)[(e % 256) / 16]) *
                double(((hi << 4) | lo) - 32);
            // Tensor 是列主序（core/tensor.h 的连续步长）：x[e, c] 在 c*k + e；输出的扁平序同是
            // r + c*rows，参考按同一序存放。
            for (std::int32_t c = 0; c < t; ++c) {
                reference[std::size_t(c) * rows + r] +=
                    value * double(__bfloat162float(x_host[std::size_t(c) * k + e]));
            }
        }
    }
    double worst = 0.0;
    for (const std::vector<__nv_bfloat16>& side : {out_native, out_dense}) {
        double squared = 0.0, reference_squared = 0.0;
        for (std::size_t i = 0; i < side.size(); ++i) {
            const double a = double(__bfloat162float(side[i]));
            const double b = reference[i];
            squared += (a - b) * (a - b);
            reference_squared += b * b;
        }
        worst = std::max(worst, std::sqrt(squared / static_cast<double>(side.size())) /
                                        std::sqrt(reference_squared / static_cast<double>(side.size())));
    }
    std::cout << label << ": max relative_l2 vs FP64 reference=" << worst << '\n';
    if (!(worst <= kToleranceL2)) {
        std::cerr << label << ": a dense or native route misses the FP64 reference bound\n";
        return 1;
    }
    return 0;
}

// dequantize_rows（sm70 与 embedding 的公共入口）：FP32 输出逐位比较。
int run_dequantize(const std::vector<std::uint8_t>& blocks, std::int32_t rows, std::int32_t k) {
    DeviceBuffer native_device(blocks.size() + 4096);
    native_device.fill(0);
    native_device.copy_from_host(blocks.data(), blocks.size());
    DeviceBuffer dense_device(blocks.size() + 4096);
    dense_device.fill(0);
    dense_device.copy_from_host(blocks.data(), blocks.size());

    const std::int64_t row_bytes = std::int64_t(k / 256) * 210;
    {
        WeightGeometry geometry;
        geometry.format = QType::GGUF_Q6_K;
        geometry.layout = QuantLayout::GgufBlocks;
        geometry.shape  = {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(k)};
        permute_q6k_to_dense(geometry, static_cast<std::byte*>(dense_device.p), nullptr);
    }

    std::vector<float> native, dense;
    for (const void* data : {native_device.p, dense_device.p}) {
        DeviceBuffer out(std::size_t(rows) * k * sizeof(float));
        ops::gguf::dequantize_rows(ops::detail::gguf_type(QType::GGUF_Q6_K), data, row_bytes, k,
                                   nullptr, rows, static_cast<float*>(out.p), k, nullptr,
                                   data == dense_device.p);
        check(cudaDeviceSynchronize(), "dequantize");
        auto& values = data == native_device.p ? native : dense;
        values.resize(std::size_t(rows) * k);
        out.copy_to_host(values.data(), values.size() * sizeof(float));
    }
    int failures = 0;
    for (std::size_t i = 0; i < native.size(); ++i) {
        if (native[i] != dense[i]) {
            std::cerr << "Q6_K dense dequantize: element " << i << " differs\n";
            return 1;
        }
        (void)failures;
    }
    std::cout << "Q6_K dense dequantize_rows k=" << k << ": exact\n";
    return 0;
}

// embedding gather：同一组随机行号，两遍比较。
int run_embedding(const std::vector<std::uint8_t>& blocks, std::int32_t rows, std::int32_t k) {
    DeviceBuffer native_device(blocks.size() + 4096);
    native_device.fill(0);
    native_device.copy_from_host(blocks.data(), blocks.size());
    DeviceBuffer dense_device(blocks.size() + 4096);
    dense_device.fill(0);
    dense_device.copy_from_host(blocks.data(), blocks.size());

    {
        WeightGeometry geometry;
        geometry.format = QType::GGUF_Q6_K;
        geometry.layout = QuantLayout::GgufBlocks;
        geometry.shape  = {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(k)};
        permute_q6k_to_dense(geometry, static_cast<std::byte*>(dense_device.p), nullptr);
    }

    constexpr std::int32_t kTokens = 37;
    std::mt19937 rng(73U);
    std::uniform_int_distribution<int> row(0, rows - 1);
    std::vector<std::int32_t> ids_host(kTokens);
    for (auto& r : ids_host) { r = row(rng); }
    DeviceBuffer ids_device(ids_host.size() * sizeof(std::int32_t));
    ids_device.copy_from_host(ids_host.data(), ids_host.size() * sizeof(std::int32_t));

    std::vector<__nv_bfloat16> native, dense;
    for (const void* data : {native_device.p, dense_device.p}) {
        const Weight table =
            weight_of(data, data == native_device.p ? QuantLayout::GgufBlocks :
                                                      QuantLayout::VoltaQ6KDense,
                      rows, k);
        DeviceBuffer out(std::size_t(k) * kTokens * sizeof(__nv_bfloat16));
        Tensor ids(const_cast<void*>(ids_device.p), DType::I32, {kTokens});
        Tensor result(const_cast<void*>(out.p), DType::BF16, {k, kTokens});
        ops::embedding(ids, table, result, nullptr);
        check(cudaDeviceSynchronize(), "embedding");
        auto& values = data == native_device.p ? native : dense;
        read_bf16(out, std::size_t(k) * kTokens, values);
    }
    return compare("Q6_K dense prepack embedding k=" + std::to_string(k), native, dense);
}

} // namespace

int main() {
    try {
        std::mt19937 blocks_rng(907U);
        int failures = 0;
        // 27B 制品的解码 K（5120 的 gate 形状与 6144 的 GDN 输出形状）。
        for (const std::int32_t k : {5120, 6144}) {
            const std::vector<std::uint8_t> blocks = random_blocks(512, k, blocks_rng);
            // t <= 8 向量解码（同形，逐位）；9..64 分块（对照 FP64 参考）；129 走 sm70 反量化 + cuBLAS。
            for (const std::int32_t t : {1, 4, 8, 9, 17, 64, 129}) {
                failures += run_product(blocks, 512, k, t, t <= 8 || t == 129);
            }
            failures += run_dequantize(blocks, 512, k);
            failures += run_embedding(blocks, 512, k);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q6_K dense prepack\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q6_K dense prepack: " << error.what() << '\n';
        return 1;
    }
}
