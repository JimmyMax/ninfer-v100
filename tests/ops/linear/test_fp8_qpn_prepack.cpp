// 本仓库 V100 增量（docs/v100/PLAN.md）：FP8 的 QPN 加载时置换（prepack）的数值 oracle。
//
// 置换把 artifact 的行主序码字原地重排成 QPN8 的片段序（尺度平面不动）。它是尺寸不变的，所以
// 验证方式是把每一份权重跑两遍——一遍 artifact 布局、一遍 prepack——两次结果必须逐位一致。
//
// 覆盖必须按 **模型实际使用的算子与宽度** 来选：t <= 32 走 QPN（普通投射 / 残差），t > 32 走
// CUTLASS 宽 T。两者各有一条独立的预打包解码路径，只测其中一条会漏掉另一条。
#include <cuda_bf16.h>

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::quantized_weight;

// 两种布局解码的是同一批码字，所以要求逐位一致。
constexpr double kRelativeL2 = 0.0;

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

// 用同一份权重载荷跑一次给定的算子；prepack 决定是否先做加载时置换。`plane_rows` 给出算子要
// 写的每个输出平面的行数，只有这些平面参与比较。
template <class Launch>
std::vector<double> run_once(const PackedWeight& host_weight, std::int32_t input_rows,
                             const std::vector<std::int32_t>& plane_rows, std::int32_t tokens,
                             std::size_t workspace_bytes, bool prepack, Launch&& launch) {
    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    if (prepack) { ops::detail::fp8_prepack_qpn_sm70(weight, nullptr); }

    const std::vector<std::uint16_t> host_input = make_pattern(input_rows, tokens, 11U);
    test::GuardedDeviceBuffer device_input(host_input.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(host_input.data(), host_input.size() * sizeof(std::uint16_t));

    std::vector<test::GuardedDeviceBuffer> outputs;
    for (std::size_t plane = 0; plane < plane_rows.size(); ++plane) {
        const std::vector<std::uint16_t> host =
            make_pattern(plane_rows[plane], tokens, 23U + static_cast<std::uint32_t>(plane));
        test::GuardedDeviceBuffer buffer(host.size() * sizeof(std::uint16_t));
        buffer.copy_from_host(host.data(), host.size() * sizeof(std::uint16_t));
        outputs.push_back(std::move(buffer));
    }

    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
    launch(weight, device_input, outputs, workspace);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize the FP8 op");

    std::vector<double> values;
    for (auto& output : outputs) {
        std::vector<std::uint16_t> bits(output.bytes() / sizeof(std::uint16_t));
        output.copy_to_host(bits.data(), output.bytes());
        for (const std::uint16_t raw : bits) {
            values.push_back(static_cast<double>(__bfloat162float(__ushort_as_bfloat16(raw))));
        }
    }
    return values;
}

int compare(const std::string& label, const std::vector<double>& native,
            const std::vector<double>& packed) {
    if (native.size() != packed.size() || native.empty()) {
        std::cerr << label << ": invalid comparison\n";
        return 1;
    }
    double squared           = 0.0;
    double reference_squared = 0.0;
    double worst             = 0.0;
    bool finite              = true;
    for (std::size_t i = 0; i < native.size(); ++i) {
        const double difference = packed[i] - native[i];
        squared += difference * difference;
        reference_squared += native[i] * native[i];
        worst = std::max(worst, std::abs(difference));
        finite = finite && std::isfinite(native[i]) && std::isfinite(packed[i]);
    }
    const double scale    = static_cast<double>(native.size());
    const double relative = std::sqrt(squared / scale) / std::sqrt(reference_squared / scale);
    std::cout << label << ": relative_l2=" << relative << " max_abs=" << worst << '\n';
    if (!finite) {
        std::cerr << label << ": non-finite output\n";
        return 1;
    }
    if (!(relative <= kRelativeL2)) {
        std::cerr << label << ": the prepacked layout differs from the artifact layout\n";
        return 1;
    }
    return 0;
}

// 普通 Linear：14336x5120。
int run_projection(const PackedWeight& weight, std::int32_t tokens) {
    constexpr std::int32_t kN = 14336;
    constexpr std::int32_t kK = 5120;
    const auto launch = [&](const Weight& w, test::GuardedDeviceBuffer& input,
                            std::vector<test::GuardedDeviceBuffer>& outputs,
                            WorkspaceArena& workspace) {
        Tensor x(const_cast<void*>(input.data()), DType::BF16, {kK, tokens});
        Tensor out(const_cast<void*>(outputs[0].data()), DType::BF16, {kN, tokens});
        ops::linear(x, w, out, ops::LinearPolicy::A16Only, workspace, nullptr);
    };
    const std::size_t bytes = ops::linear_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kN, kK, ops::LinearPolicy::A16Only, tokens, tokens);
    const std::string label = "FP8 QPN prepack linear " + std::to_string(tokens);
    return compare(label, run_once(weight, kK, {kN}, tokens, bytes, false, launch),
                   run_once(weight, kK, {kN}, tokens, bytes, true, launch));
}

// 注意力输入投影：融合的 QKV+gate 权重，四个输出平面。
int run_attn(const PackedWeight& weight, std::int32_t tokens) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const auto launch = [&](const Weight& w, test::GuardedDeviceBuffer& input,
                            std::vector<test::GuardedDeviceBuffer>& outputs,
                            WorkspaceArena& workspace) {
        Tensor x(const_cast<void*>(input.data()), DType::BF16, {kHidden, tokens});
        Tensor q(const_cast<void*>(outputs[0].data()), DType::BF16, {kQRows, tokens});
        Tensor gate(const_cast<void*>(outputs[1].data()), DType::BF16, {kQRows, tokens});
        Tensor k(const_cast<void*>(outputs[2].data()), DType::BF16, {kKvRows, tokens});
        Tensor v(const_cast<void*>(outputs[3].data()), DType::BF16, {kKvRows, tokens});
        ops::attn_input_proj(x, w, q, gate, k, v, ops::LinearPolicy::A16Only, workspace, nullptr);
    };
    const std::size_t bytes = ops::attn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kQRows * 2 + kKvRows * 2, kHidden, ops::LinearPolicy::A16Only,
        tokens, tokens);
    const std::string label = "FP8 QPN prepack attn_input_proj " + std::to_string(tokens);
    const std::vector<std::int32_t> planes{kQRows, kQRows, kKvRows, kKvRows};
    return compare(label, run_once(weight, kHidden, planes, tokens, bytes, false, launch),
                   run_once(weight, kHidden, planes, tokens, bytes, true, launch));
}

// 残差投射（down 形状）。
int run_add(const PackedWeight& weight, std::int32_t tokens) {
    constexpr std::int32_t kRows = 5120;
    constexpr std::int32_t kK    = 17408;
    const auto launch = [&](const Weight& w, test::GuardedDeviceBuffer& input,
                            std::vector<test::GuardedDeviceBuffer>& outputs,
                            WorkspaceArena& workspace) {
        Tensor x(const_cast<void*>(input.data()), DType::BF16, {kK, tokens});
        Tensor residual(const_cast<void*>(outputs[0].data()), DType::BF16, {kRows, tokens});
        ops::linear_add(x, w, residual, ops::LinearPolicy::A16Only, workspace, nullptr);
    };
    const std::size_t bytes = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kK, ops::LinearPolicy::A16Only, tokens, tokens);
    const std::string label = "FP8 QPN prepack linear_add " + std::to_string(tokens);
    return compare(label, run_once(weight, kK, {kRows}, tokens, bytes, false, launch),
                   run_once(weight, kK, {kRows}, tokens, bytes, true, launch));
}

} // namespace

int main() {
    if (!test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // FP8 的形状 fixture 由 linear 测试公共代码提供（NVFP4 的 divisor 选项不适用于它）。
        const PackedWeight projection = test::linear::make_fp8_weight(14336, 5120, 907U);
        const PackedWeight attention  = test::linear::make_fp8_weight(14336, 5120, 911U);
        const PackedWeight residual   = test::linear::make_fp8_weight(5120, 17408, 919U);

        int failures = 0;
        // 1..32 覆盖 QPN；33/129 覆盖 CUTLASS 宽 T（按需解量化 + 片段序解码）。
        for (const std::int32_t tokens : {1, 5, 17, 32, 33, 129}) {
            failures += run_projection(projection, tokens);
            failures += run_attn(attention, tokens);
            failures += run_add(residual, tokens);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 QPN prepack\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 QPN prepack: " << error.what() << '\n';
        return 1;
    }
}
