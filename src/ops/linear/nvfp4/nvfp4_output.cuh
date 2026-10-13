#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// apply_scaled follows the contract in ops/linear/common/epilogue.cuh.
struct Nvfp4IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }

    __device__ __forceinline__ float apply_scaled(std::int32_t, std::int32_t, float value,
                                                  float scale) const {
        return value * scale;
    }
};

struct Nvfp4ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + parent_row;
        store_vec(destination, values);
    }
};

// 本仓库 V100 增量（docs/v100/PLAN.md）：residual += value with a single BF16 round
// (linear_add without a projected temporary).
struct Nvfp4ResidualOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + parent_row;
        data[index] = __float2bfloat16_rn(value + __bfloat162float(data[index]));
    }
};

// 本仓库 V100 增量（docs/v100/PLAN.md）：FP32 直出（不做 BF16 舍入），
// 供拆分式 SwiGLU 的 gate 道使用（up 道再从该 FP32 平面读取并做激活）。
struct Nvfp4Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

} // namespace ninfer::ops::detail
