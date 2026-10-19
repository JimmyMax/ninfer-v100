#pragma once

#include "models/qwen3_5/model.h"

namespace ninfer::models::qwen3_5::execution {

// Program-owned service. Immutable sources are borrowed from Model. No service or slot is shared
// between Programs, even when their Parameters refer to the same Model.
class NativeExpertRuntime {
public:
    NativeExpertRuntime(const Model& model, DeviceContext& device,
                         std::span<const std::uint64_t> keep_free_by_rank,
                         std::optional<std::uint64_t> cache_bytes = {});
    [[nodiscard]] ExpertDelivery& layer(std::size_t index) const;
    [[nodiscard]] std::size_t local_layer(std::size_t index) const;
    [[nodiscard]] ExpertDeliveryMemory memory(std::size_t rank) const noexcept;
    [[nodiscard]] ExpertDeliveryStats stats() const;
    void check() const;
    // Waits for every rank's delivery work, prefetches included.
    void drain();
    void release_device_backing();
    void restore_device_backing();
    [[nodiscard]] std::uint64_t device_backing_bytes() const noexcept;

private:
    struct Handle { std::size_t rank, local; };
    std::vector<Handle> handles_;
    std::vector<std::unique_ptr<ExpertDelivery>> ranks_;
};

} // namespace ninfer::models::qwen3_5::execution
