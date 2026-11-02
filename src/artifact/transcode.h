#pragma once

#include "core/weight.h"
#include "core/weight_view.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::artifact {

// A device object may be materialized in a narrower grouped format than the artifact stores. The
// plan reserves the target encoding's size, the materializer writes target bytes, and every
// consumer (bindings, weight views, arena accounting) sees only the target format. This trades
// weight precision for device memory; model loading selects it per object from startup options.
//
// The only supported source is Q8_G32_FP16 in the row-split layout; targets are Q4_G64_FP16 and
// Q6_G64_FP16 in the same layout. Logical element addressing (row * columns + column) is the same
// before and after, so bindings over parts of a transcoded object stay valid.
[[nodiscard]] bool row_split_transcode_supported(QType source, QType target) noexcept;

// A device object may be transcoded at load to a format its tier runs faster. The supported pairs
// are the row-split Q8_G32_FP16 to Q4_G64_FP16/Q6_G64_FP16 device-memory trade, and NVFP4 in the
// block-scale layout to Q4_G64_FP16: a speed trade for tiers without FP4 tensor cores, where the
// native NVFP4 route is compute-bound and the row-split q4 GEMV and MoE routes are memory-bound.
[[nodiscard]] bool transcode_supported(QType source, QuantLayout layout, QType target) noexcept;

// Requantizes a row-split Q8_G32_FP16 payload to the target format's row-split encoding. Each
// 64-column group takes, of 25 clipping ratios of absmax/qmax in [0.70, 1.18], the fp16 scale whose
// round-to-nearest codes minimise the group's squared error against the Q8 values; plain
// absmax/qmax is one candidate. Groups are independent, so the result does not depend on the
// worker count.
void transcode_row_split(QType target, std::span<const std::uint64_t> shape,
                         std::span<const std::byte> source, std::span<std::byte> destination);

// Requantizes an NVFP4 block-scale payload (e2m1 codes, e4m3 block scales, a global divisor) to
// the row-split Q4_G64_FP16 encoding. The source geometry carries the stored layout and divisor
// count; the destination is the row-split q4 layout the Ampere GEMV and MoE routes read. Each
// 64-column group takes the single fp16 scale absmax/7: the dequantised NVFP4 values are already
// 4-bit, so the natural scale is within a fraction of a percent of the clipping-search optimum at
// a tenth of the load cost, and the result does not depend on the worker count. Used on tiers
// without FP4 tensor cores, where it trades the compute-bound native NVFP4 route for the
// memory-bound row-split q4 route at the same 4-bit weight precision.
void transcode_nvfp4_to_q4(const WeightGeometry& source_geometry,
                           std::span<const std::byte> source, std::span<std::byte> destination);

} // namespace ninfer::artifact
