#pragma once

// Shim for the split-header layout: the shared enums live in main's monolithic types.h; the six
// that main does not carry are defined here.
#include "ninfer/types.h"

namespace ninfer {

enum class EmbeddingResidency : std::uint8_t {
    Device,
    Host,
};

enum class TensorAttention : std::uint8_t {
    Heads,
    Positions,
};

enum class TensorParallelTransport : std::uint8_t {
    Auto,
    Host,
    PeerRead,
    PeerWrite,
};

enum class TensorParallelPartials : std::uint8_t {
    Fp32,
    Bf16,
    Int8,
};

enum class PrefillTail : std::uint8_t {
    Full,
    Exact,
    Rows,
};

} // namespace ninfer
