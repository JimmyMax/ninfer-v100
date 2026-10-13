#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h" // 本仓库 V100 增量：上游 core/tensor.h 不带 Weight 定义

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                              std::int32_t cols);
void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
