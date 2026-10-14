#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

// Maps a logical row-major FP8 byte to its load-time Volta QPN stream position.
// K must be a multiple of 128 and N is tiled in groups of 32 rows.
__device__ __forceinline__ std::int64_t fp8_qpn_prepacked_offset(int row, int column, int k) {
    const int tile      = row / 32;
    const int row_inner = row % 32;
    const int qp        = row_inner / 8;
    const int r         = row_inner % 8;
    const int lane      = (qp << 2) | (r & 3) | ((r & 4) ? 16 : 0);
    const int block     = column / 128;
    const int slice     = (column % 128) / 16;
    const int offset    = column % 16;
    return (((static_cast<std::int64_t>(tile) * (k / 128) + block) * 8 + slice) * 32 + lane) *
               16 +
           offset;
}

} // namespace ninfer::ops::detail
