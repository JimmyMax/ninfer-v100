// 本仓库 V100 增量（docs/v100/PLAN.md）：NVFP4 的 QPN 加载时置换（prepack）的数值 oracle。
//
// 置换把 artifact 的行主序码字/尺度原地重排成 QPN2 片段序。它是尺寸不变的，所以验证方式是把
// 每一份权重跑两遍——一遍 artifact 布局、一遍 prepack——两次结果必须一致。
//
// 覆盖必须按 **模型实际使用的算子与宽度** 来选：t <= 32 走 QPN（普通投射 / 拆分 / 残差），
// t > 32 走 CUTLASS 宽 T。两者各有一条独立的预打包解码路径，只测其中一条会漏掉另一条。
#include <cuda_bf16.h>

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
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

// 两次运行的差异只来自 fp16 舍入（同一份权重的两种解码顺序）。
constexpr double kRelativeL2 = 3.3e-3;

std::vector<std::uint16_t> make_pattern(std::int32_t rows, std::int32_t columns,
                                        std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t hash = static_cast<std::uint32_t>(row) * 2654435761U ^
                                       static_cast<std::uint32_t>(column) * 40503U ^ seed;
            const float value = (static_cast<float>((hash >> 9) & 0xffU) / 256.0F - 0.5F) * 4.0F;
            bits[static_cast<std::size_t>(column) * rows + row] =
                static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(value)));
        }
    }
    return bits;
}

// 用同一份权重载荷跑一次给定的算子；prepack 决定是否先做加载时置换。
template <class Launch>
std::vector<double> run_once(std::int32_t n, std::int32_t k, std::int32_t rows,
                             std::int32_t tokens, std::size_t workspace_bytes, bool prepack,
                             Launch&& launch) {
    PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    const PackedWeight host_weight = make_patterned_weight(QType::NVFP4, n, k, 704U, options);
    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    if (prepack) { ops::detail::nvfp4_prepack_qpn_sm70(weight, nullptr); }

    const std::vector<std::uint16_t> host_input = make_pattern(k, tokens, 11U);
    test::GuardedDeviceBuffer device_input(host_input.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(host_input.data(), host_input.size() * sizeof(std::uint16_t));
    const std::vector<std::uint16_t> host_output = make_pattern(rows, tokens, 23U);
    test::GuardedDeviceBuffer device_output(static_cast<std::size_t>(rows) * tokens *
                                            sizeof(std::uint16_t));
    device_output.copy_from_host(host_output.data(),
                                 host_output.size() * sizeof(std::uint16_t));

    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
    launch(weight, device_input, device_output, workspace);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize the NVFP4 op");

    std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * tokens);
    device_output.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(__bfloat162float(__ushort_as_bfloat16(bits[i])));
    }
    return values;
}

int compare(const std::string& label, const std::vector<double>& native,
            const std::vector<double>& packed) {
    if (native.size() != packed.size() || native.empty()) {
        std::cerr << label << ": invalid comparison\n";
        return 1;
    }
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
    const double relative = std::sqrt(squared / scale) / std::sqrt(reference_squared / scale);
    std::cout << label << ": relative_l2=" << relative << " max_abs=" << worst << '\n';
    if (!(relative <= kRelativeL2)) {
        std::cerr << label << ": the prepacked layout differs from the artifact layout\n";
        return 1;
    }
    return 0;
}

int run_projection(std::int32_t tokens) {
    constexpr std::int32_t kN = 34816;
    constexpr std::int32_t kK = 5120;
    const auto launch = [&](const Weight& weight, test::GuardedDeviceBuffer& input,
                            test::GuardedDeviceBuffer& output, WorkspaceArena& workspace) {
        Tensor x(const_cast<void*>(input.data()), DType::BF16, {kK, tokens});
        Tensor out(const_cast<void*>(output.data()), DType::BF16, {kN, tokens});
        ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, nullptr);
    };
    const std::size_t bytes = ops::linear_workspace_capacity_bytes(
        QType::NVFP4, kN, kK, ops::LinearPolicy::A16Only, tokens, tokens);
    const std::string label = "NVFP4 QPN prepack linear " + std::to_string(tokens);
    return compare(label, run_once(kN, kK, kN, tokens, bytes, false, launch),
                   run_once(kN, kK, kN, tokens, bytes, true, launch));
}

int run_swiglu(std::int32_t tokens) {
    constexpr std::int32_t kGateUp = 34816;
    constexpr std::int32_t kK      = 5120;
    constexpr std::int32_t kRows   = kGateUp / 2;
    const auto launch = [&](const Weight& weight, test::GuardedDeviceBuffer& input,
                            test::GuardedDeviceBuffer& output, WorkspaceArena& workspace) {
        Tensor x(const_cast<void*>(input.data()), DType::BF16, {kK, tokens});
        Tensor out(const_cast<void*>(output.data()), DType::BF16, {kRows, tokens});
        ops::linear_swiglu(x, weight, out, ops::LinearPolicy::A16Only, workspace, nullptr);
    };
    const std::size_t bytes = ops::linear_swiglu_workspace_capacity_bytes(
        QType::NVFP4, kGateUp, kK, ops::LinearPolicy::A16Only, tokens, tokens);
    const std::string label = "NVFP4 QPN prepack linear_swiglu " + std::to_string(tokens);
    return compare(label, run_once(kGateUp, kK, kRows, tokens, bytes, false, launch),
                   run_once(kGateUp, kK, kRows, tokens, bytes, true, launch));
}

int run_add(std::int32_t tokens) {
    constexpr std::int32_t kRows = 5120;
    constexpr std::int32_t kK    = 17408;
    const auto launch = [&](const Weight& weight, test::GuardedDeviceBuffer& input,
                            test::GuardedDeviceBuffer& output, WorkspaceArena& workspace) {
        Tensor x(const_cast<void*>(input.data()), DType::BF16, {kK, tokens});
        Tensor residual(const_cast<void*>(output.data()), DType::BF16, {kRows, tokens});
        ops::linear_add(x, weight, residual, ops::LinearPolicy::A16Only, workspace, nullptr);
    };
    const std::size_t bytes = ops::linear_add_workspace_capacity_bytes(
        QType::NVFP4, kRows, kK, ops::LinearPolicy::A16Only, tokens, tokens);
    const std::string label = "NVFP4 QPN prepack linear_add " + std::to_string(tokens);
    return compare(label, run_once(kRows, kK, kRows, tokens, bytes, false, launch),
                   run_once(kRows, kK, kRows, tokens, bytes, true, launch));
}

} // namespace

int main() {
    if (!test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        // 普通 linear 只有 QPN（t <= 32）；宽 T 的 prepacked 权重由 linear_swiglu/linear_add 的
        // CUTLASS 路径消费（该组合没有解码器时会明确报错，见 nvfp4_dispatch）。
        for (const std::int32_t tokens : {1, 5, 17, 32}) {
            failures += run_projection(tokens);
        }
        // 33/129 覆盖 CUTLASS 宽 T（按需解量化 + 片段序解码）——模型实际走的就是这两个算子。
        for (const std::int32_t tokens : {1, 5, 17, 32, 33, 129}) {
            failures += run_swiglu(tokens);
            failures += run_add(tokens);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 QPN prepack\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 QPN prepack: " << error.what() << '\n';
        return 1;
    }
}
