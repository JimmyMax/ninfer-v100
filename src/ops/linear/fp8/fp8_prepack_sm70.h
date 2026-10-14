#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place load-time permutation from row-major FP8 codes to the coalesced Volta QPN8 stream.
// Row scales remain in their registered order and the payload size is unchanged.
void fp8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);

} // namespace ninfer::ops::detail
