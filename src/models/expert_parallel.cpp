#include "models/expert_parallel.h"

#include "core/tensor_split.h"

#include <set>
#include <stdexcept>

namespace ninfer::models {

std::vector<double> default_expert_split(const artifact::Reader& reader,
                                         const std::function<bool(std::string_view)>& loaded,
                                         const std::function<bool(std::string_view)>& expert,
                                         std::span<const std::uint64_t> free_bytes,
                                         std::span<const double> speed) {
    if (free_bytes.size() < 2 || speed.size() != free_bytes.size()) {
        throw std::invalid_argument("an expert split needs the free bytes and speed of each rank");
    }
    // State, workspace and the CUDA context beside the weights: the sequences' state and the
    // activation planes on rank 0, a peer's planes, workspace and exchange buffers.
    constexpr std::uint64_t kFirstMargin = 3ULL << 30, kPeerMargin = 1ULL << 30;
    std::set<std::size_t> expert_objects, other_objects;
    for (const auto& [name, binding] : reader.directory().bindings) {
        if (!loaded(name)) { continue; }
        auto& objects = expert(name) ? expert_objects : other_objects;
        for (const auto& part : binding.parts) { objects.insert(part.object.index); }
    }
    const auto bytes = [&](const std::set<std::size_t>& objects) {
        std::uint64_t total = 0;
        for (const std::size_t index : objects) {
            total +=
                artifact::object_bytes(reader.directory().object(artifact::ObjectHandle{index}));
        }
        return total;
    };
    const std::uint64_t divided = bytes(expert_objects);
    const std::uint64_t first   = bytes(other_objects);
    if (divided == 0) { return std::vector<double>(speed.begin(), speed.end()); }
    std::vector<double> capacity(free_bytes.size());
    for (std::size_t rank = 0; rank < free_bytes.size(); ++rank) {
        const std::uint64_t whole = rank == 0 ? kFirstMargin + first : kPeerMargin;
        capacity[rank] =
            free_bytes[rank] > whole ? double(free_bytes[rank] - whole) / double(divided) : 0.0;
    }
    return fill_shares(speed, capacity);
}

} // namespace ninfer::models
