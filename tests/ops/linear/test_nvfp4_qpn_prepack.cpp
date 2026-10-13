// 本仓库 V100 增量（docs/v100/PLAN.md）：NVFP4 的 QPN 加载时置换（prepack）的数值 oracle。
//
// 置换把 artifact 的行主序码字/尺度原地重排成 QPN2 的片段序。它是尺寸不变的，所以验证方式是
// 把同一份权重跑两遍：一遍用 artifact 布局（native 解码路径），一遍先 prepack（片段序解码
// 路径），两次结果必须一致。这条断言覆盖本仓库唯一会改写权重字节的机制，也就是端到端模型
// 加载的前置条件（models/qwen3_5/load/storage_trades.cpp 对 MLP 那两个形状下 prepack 请求）。
#include <cuda_bf16.h>

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::quantized_weight;

// 两次运行都走 QPN2，只是码字/尺度的读取顺序不同，差异只来自 fp16 舍入。
constexpr double kRelativeL2 = 3.3e-3;

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t tokens) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            const std::uint32_t hash = static_cast<std::uint32_t>(column) * 2654435761U ^
                                       static_cast<std::uint32_t>(token) * 40503U;
            const float value = (static_cast<float>((hash >> 8) & 0xffU) / 256.0F - 0.5F) * 4.0F;
            bits[static_cast<std::size_t>(token) * k + column] =
                static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(value)));
        }
    }
    return bits;
}

std::vector<double> run_once(std::int32_t n, std::int32_t k, std::int32_t tokens, bool prepack) {
    PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    const PackedWeight host_weight = make_patterned_weight(QType::NVFP4, n, k, 704U, options);
    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    if (prepack) { ops::detail::nvfp4_prepack_qpn_sm70(weight, nullptr); }

    const std::vector<std::uint16_t> host_activation = make_activation(k, tokens);
    test::GuardedDeviceBuffer device_activation(host_activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(host_activation.data(),
                                     host_activation.size() * sizeof(std::uint16_t));

    const std::size_t workspace_bytes = ops::linear_workspace_capacity_bytes(
        QType::NVFP4, n, k, ops::LinearPolicy::A16Only, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
    test::GuardedDeviceBuffer device_output(static_cast<std::size_t>(n) * tokens *
                                            sizeof(std::uint16_t));
    Tensor x(device_activation.data(), DType::BF16, {k, tokens});
    Tensor out(device_output.data(), DType::BF16, {n, tokens});
    ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize the NVFP4 projection");

    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n) * tokens);
    device_output.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(__bfloat162float(__ushort_as_bfloat16(bits[i])));
    }
    return values;
}

int run_case(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    const std::string label = "NVFP4 QPN prepack " + std::to_string(n) + "x" + std::to_string(k) +
                              " T=" + std::to_string(tokens);
    const std::vector<double> native = run_once(n, k, tokens, /*prepack=*/false);
    const std::vector<double> packed = run_once(n, k, tokens, /*prepack=*/true);
    double squared = 0.0;
    double reference_squared = 0.0;
    double worst = 0.0;
    for (std::size_t i = 0; i < native.size(); ++i) {
        const double difference = packed[i] - native[i];
        squared += difference * difference;
        reference_squared += native[i] * native[i];
        worst = std::max(worst, std::abs(difference));
    }
    const double scale = static_cast<double>(native.size());
    const double relative =
        std::sqrt(squared / scale) / std::sqrt(reference_squared / scale);
    std::cout << label << ": relative_l2=" << relative << " max_abs=" << worst << '\n';
    if (!(relative <= kRelativeL2)) {
        std::cerr << label << ": the prepacked layout differs from the artifact layout\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (!test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (const std::int32_t tokens : {1, 5, 17, 32}) {
            failures += run_case(34816, 5120, tokens);
            failures += run_case(5120, 17408, tokens);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 QPN prepack\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 QPN prepack: " << error.what() << '\n';
        return 1;
    }
}
