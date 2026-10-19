#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

// A blob is copied without decoding or repacking. File extents concatenate in order.
struct ExpertExtent {
    std::size_t file = 0;
    std::uint64_t offset = 0, bytes = 0;
};
struct ExpertBlob {
    const void* pinned = nullptr;
    // The base of the host allocation `pinned` lies in. Neighbouring pinned blobs of one nonzero
    // region are copied together; a copy never spans two allocations.
    std::uintptr_t pinned_region = 0;
    std::vector<ExpertExtent> extents;
    std::uint64_t bytes = 0;
    // Zero bytes a kernel may read past the blob's end (whole K steps past a row's end decode the
    // following bytes as blocks of the same format, so they must be zeros or that format's data).
    std::uint64_t zero_tail = 0;
};
struct ExpertAlias {
    std::size_t blob = 0;
    std::uint64_t offset = 0;
};
struct ExpertDeliveryLayer {
    // [expert][blob]. Aliases describe the projection-major device pointer table.
    std::vector<std::vector<ExpertBlob>> experts;
    std::vector<ExpertAlias> aliases;
};
struct ExpertDeliveryOptions {
    std::uint32_t pair_capacity = 0;
    // Persistent cache only; the layer banks are separate. Both explicit and automatic cache keep
    // keep_free_bytes free after physical allocation rounding.
    std::optional<std::uint64_t> cache_bytes;
    std::uint64_t keep_free_bytes = 512ULL << 20;
    std::uint64_t staging_bytes = 128ULL << 20; // disk only, two pinned halves
    // Whole-layer device windows that hold the experts the cache does not. Two let the next layer
    // of a wide call (a prompt chunk, a wide verification) arrive while the current one computes;
    // one halves their memory and gives that up. Two fall back to one when memory is short.
    std::uint32_t banks = 2;
    bool suspendable = false; // existing VmmRegion allocations, fixed virtual addresses
};
struct ExpertDeliveryStats {
    std::uint64_t requests = 0, routes = 0, hits = 0, misses = 0, evictions = 0;
    std::uint64_t pairs = 0, pair_hits = 0, admissions = 0;
    // Misses found already present in a layer bank (a prefetched layer, a repeated wide call).
    std::uint64_t bank_reuses = 0, prefetched_layers = 0;
    // Host-to-device expert bytes (misses and prefetches); device copies from a bank into the cache.
    std::uint64_t read_bytes = 0, copied_bytes = 0, promoted_bytes = 0;
    double service_seconds = 0;
};
// Counters accumulated after `start` (startup warm-up work excluded from request statistics).
[[nodiscard]] inline ExpertDeliveryStats operator-(ExpertDeliveryStats now,
                                                   const ExpertDeliveryStats& start) noexcept {
    now.requests -= start.requests;
    now.routes -= start.routes;
    now.hits -= start.hits;
    now.misses -= start.misses;
    now.evictions -= start.evictions;
    now.pairs -= start.pairs;
    now.pair_hits -= start.pair_hits;
    now.admissions -= start.admissions;
    now.bank_reuses -= start.bank_reuses;
    now.prefetched_layers -= start.prefetched_layers;
    now.read_bytes -= start.read_bytes;
    now.copied_bytes -= start.copied_bytes;
    now.promoted_bytes -= start.promoted_bytes;
    now.service_seconds -= start.service_seconds;
    return now;
}
// Sum over ranks.
[[nodiscard]] inline ExpertDeliveryStats operator+(ExpertDeliveryStats a,
                                                   const ExpertDeliveryStats& b) noexcept {
    a.requests += b.requests;
    a.routes += b.routes;
    a.hits += b.hits;
    a.misses += b.misses;
    a.evictions += b.evictions;
    a.pairs += b.pairs;
    a.pair_hits += b.pair_hits;
    a.admissions += b.admissions;
    a.bank_reuses += b.bank_reuses;
    a.prefetched_layers += b.prefetched_layers;
    a.read_bytes += b.read_bytes;
    a.copied_bytes += b.copied_bytes;
    a.promoted_bytes += b.promoted_bytes;
    a.service_seconds += b.service_seconds;
    return a;
}
struct ExpertDeliveryMemory {
    std::uint64_t cache_bytes = 0, bank_bytes = 0, metadata_bytes = 0, pinned_bytes = 0;
    std::uint32_t cache_slots = 0, banks = 0;
};

// One Program, rank and serial compute stream. Sources outlive this object. All device addresses
// stay fixed until destruction. The caller drains graph users before destroying their graphs.
//
// Each call splits its routes into experts the persistent cache holds and the rest. A narrow call
// (decode, verification) admits its misses into the cache over the least recently used experts it
// does not route. A wide call (half the layer's experts or more: a prompt chunk) brings its misses
// into a layer bank, laid out like a resident bank (expert e of blob b at a fixed stride) with one
// copy per run of consecutive experts, starts the bank of the layer that followed it last time,
// and promotes its most routed experts from the bank into free cache slots by device copies, up
// to the layer's share; it evicts nothing, so a prompt does not push out the decode working set.
class ExpertDelivery {
public:
    ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                   std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options);
    ~ExpertDelivery();
    ExpertDelivery(const ExpertDelivery&) = delete;
    ExpertDelivery& operator=(const ExpertDelivery&) = delete;

    struct Call {
        const std::int32_t* cached_ids = nullptr;
        const std::int32_t* missing_ids = nullptr;
    };
    // Capturable. The next begin must follow every consumer of the preceding call on stream, and
    // every begin needs its await. Cached pairs can run immediately. Missing pairs run only after
    // await(). Neither set drops or renormalizes routes. -1 denotes a pair absent from that set.
    [[nodiscard]] Call begin(std::size_t layer, const std::int32_t* ids, std::uint32_t pairs,
                             cudaStream_t stream);
    void await(cudaStream_t stream);
    // Projection-major [aliases][experts], including aliases into the same blob.
    [[nodiscard]] const void* const* tables(std::size_t layer) const;
    void check() const;
    // Drain before suspend. Invalidate preserves allocations and graph addresses for restore.
    void drain();
    void invalidate();
    void release_device_backing(); // drain first, then release every transport device allocation
    void restore_device_backing(); // map at the same addresses and invalidate residency
    [[nodiscard]] std::uint64_t state_backing_bytes() const noexcept;
    [[nodiscard]] ExpertDeliveryStats stats() const;
    [[nodiscard]] ExpertDeliveryMemory memory() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
