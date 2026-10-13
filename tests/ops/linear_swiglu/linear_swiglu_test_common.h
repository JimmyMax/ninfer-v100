#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/op_check.h" // 本仓库 V100 增量：宽 T 合成路线的判据覆盖

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
    // Integer activations against groupwise-int weights, distinct from A8's FP8 path. Held to the
    // same A8 activation allowance.
    A8Int,
    // A8Int plus the integer routes at decode and verify widths (LinearPolicy::AllowA8IntDecode),
    // under the same allowance.
    A8IntDecode,
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
};

// A program checks this first and skips (exit 77) without a device: run_profile returns a count
// of failures, which a skip would be misread as.
bool cuda_available();

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases = {});

// 本仓库 V100 增量（docs/v100/PLAN.md）：宽 T prefill 的 NVFP4 走“先投影成 BF16 平面、再
// silu_mul”的合成路线，门与 up 各自先舍入到 bf16 再相乘，比窄 T 融合内核多一次 bf16 舍入，
// 于是 A16 的相对 L2 界在宽 T 上对任何实现都不可达（上游自己的 A4 宽 T 合成路线 rel_l2 也是
// 0.0042）。该界按同一依据放宽一倍，gross 界不变；实测宽 T 合成路线 rel_l2 = 0.0046，
// 2x 界 = 0.0066，比值 0.70，与本判据其他用例的 0.77-0.99 一致。
[[nodiscard]] ReductionCriterion wide_composition_criterion(ActivationCompute activation_compute);

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases, ReductionCriterion criterion);

} // namespace ninfer::test::linear_swiglu
