#include "models/qwen3_5/execution/expert_runtime.h"
#include "ops/sparse_moe/streamed.h"

namespace ninfer::models::qwen3_5::execution {

NativeExpertRuntime::NativeExpertRuntime(const Model& model, DeviceContext& device,
                                           std::span<const std::uint64_t> keep_free_by_rank,
                                           std::optional<std::uint64_t> cache_bytes) {
    const auto& source = model.expert_sources();
    if (keep_free_by_rank.size() != device.size()) {
        throw std::invalid_argument("native expert reservation must cover every device rank");
    }
    std::vector<std::vector<ExpertDeliveryLayer>> layers(device.size());
    for (const auto& layer : source.layers) {
        auto& rank = layers.at(layer.rank);
        handles_.push_back({layer.rank, rank.size()});
        rank.push_back(layer.delivery);
    }
    ranks_.resize(device.size());
    for (std::size_t rank = 0; rank < layers.size(); ++rank) {
        if (layers[rank].empty()) { continue; }
        ExpertDeliveryOptions options;
        options.pair_capacity = ops::detail::kSparseMoeStreamPairs;
        options.suspendable = model.weights_suspendable();
        if (cache_bytes) { options.cache_bytes = *cache_bytes / device.size(); }
        options.keep_free_bytes = expert_add(keep_free_by_rank[rank], 512ULL << 20);
        ranks_[rank] = std::make_unique<ExpertDelivery>(device.rank(rank), source.files,
                                                         std::move(layers[rank]), options);
    }
}

ExpertDelivery& NativeExpertRuntime::layer(std::size_t index) const {
    return *ranks_.at(handles_.at(index).rank);
}
std::size_t NativeExpertRuntime::local_layer(std::size_t index) const {
    return handles_.at(index).local;
}
ExpertDeliveryMemory NativeExpertRuntime::memory(std::size_t rank) const noexcept {
    return rank < ranks_.size() && ranks_[rank] ? ranks_[rank]->memory() : ExpertDeliveryMemory{};
}
void NativeExpertRuntime::check() const {
    for (const auto& rank : ranks_) { if (rank) { rank->check(); } }
}
void NativeExpertRuntime::drain() {
    for (const auto& rank : ranks_) { if (rank) { rank->drain(); } }
}
void NativeExpertRuntime::release_device_backing() {
    for (const auto& rank : ranks_) { if (rank) { rank->release_device_backing(); } }
}
void NativeExpertRuntime::restore_device_backing() {
    try {
        for (const auto& rank : ranks_) { if (rank) { rank->restore_device_backing(); } }
    } catch (...) {
        for (const auto& rank : ranks_) {
            if (rank) { try { rank->release_device_backing(); } catch (...) {} }
        }
        throw;
    }
}
std::uint64_t NativeExpertRuntime::device_backing_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const auto& rank : ranks_) { if (rank) { bytes += rank->state_backing_bytes(); } }
    return bytes;
}
ExpertDeliveryStats NativeExpertRuntime::stats() const {
    ExpertDeliveryStats out;
    for (const auto& rank : ranks_) {
        if (rank) { out = out + rank->stats(); }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::execution
