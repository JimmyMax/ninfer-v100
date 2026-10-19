#include "core/expert_delivery.h"

#include "core/arena.h"
#include "core/cuda_host_handshake.h"
#include "core/file_read_queue.h"
#include "core/vmm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninfer {
namespace {

// needed[e] of a call: its pairs routed to expert e, and whether the table entry the split read
// lies in the persistent cache.
constexpr std::uint32_t kCountMask = (1u << 30) - 1;
constexpr std::uint32_t kHit       = 1u << 30;
constexpr std::uint32_t kMiss      = 1u << 31;

std::uint64_t aligned(std::uint64_t n) {
    if (n > std::numeric_limits<std::uint64_t>::max() - 255) {
        throw std::invalid_argument("expert delivery: byte extent overflow");
    }
    return (n + 255) / 256 * 256;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a * b;
}

// Use the core's existing fixed-address allocation when the Program supports model suspend.
// The transport does not create a second virtual-memory or remapping mechanism.
struct Storage {
    DeviceBuffer owned;
    VmmRegion region;
    void* p = nullptr;
    std::size_t bytes = 0;
    Storage() = default;
    Storage(std::size_t n, int device, bool suspendable) : bytes(n) {
        if (suspendable) {
            region = VmmRegion(n, {.device = device});
            p = region.span().data;
        } else {
            owned = DeviceBuffer(n);
            p = owned.p;
        }
    }
    void fill() {
        CUDA_CHECK(cudaMemset(p, 0, bytes));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
        return region ? region.reserved_bytes() : bytes;
    }
};

// One block: a shared histogram of the call's experts, then each pair's cached/missing id by the
// table entry the expert has now. The mask is copied before publish(), on the same stream. Tables
// change only at missing entries or at experts absent from this call, so cached consumers can
// read them while the host prepares the rest.
__global__ void split_routes(const std::int32_t* ids, std::uint32_t pairs, int experts,
                             const void* const* tables, std::uintptr_t low, std::uintptr_t high,
                             std::int32_t* cached, std::int32_t* missing, std::uint32_t* needed) {
    extern __shared__ std::uint32_t counts[];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { counts[e] = 0; }
    __syncthreads();
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        bool valid  = e >= 0 && e < experts;
        bool hit    = false;
        if (valid) {
            const auto address = reinterpret_cast<std::uintptr_t>(tables[e]);
            hit = address >= low && address < high;
            atomicAdd(counts + e, 1u);
        }
        cached[p]  = valid && hit ? e : -1;
        missing[p] = valid && !hit ? e : -1;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const std::uint32_t count = counts[e];
        const auto address        = reinterpret_cast<std::uintptr_t>(tables[e]);
        const bool hit            = address >= low && address < high;
        needed[e] = count == 0 ? 0u : (count & kCountMask) | (hit ? kHit : kMiss);
    }
}

} // namespace

struct ExpertDelivery::Impl {
    struct Layer {
        ExpertDeliveryLayer source;
        Storage tables;
        std::unique_ptr<PinnedHostBuffer> mirror;
        // Whole-table uploads alternate between two pinned copies, so the mirror can change while
        // an earlier upload still reads its copy.
        std::unique_ptr<PinnedHostBuffer> upload[2];
        cudaEvent_t uploaded[2]{};
        int next_upload = 0;
        bool dirty      = false;
        std::vector<std::int32_t> slot_of;
        std::uint32_t cached = 0; // slots holding this layer's experts
        // Bank stride of each blob: the blob's bytes when every expert's are equal (the resident
        // layout, one copy per run of experts), else its aligned maximum.
        std::vector<std::uint64_t> stride;
        std::int64_t successor = -1;
        bool successor_wide    = false;
    };
    struct Bank {
        std::int64_t layer = -1;
        std::vector<unsigned char> valid;
    };
    struct Slot {
        std::int32_t layer = -1, expert = -1;
        bool pending       = false; // promoted; its table entry is patched at the next call
        // Recency list of occupied slots: decode and verification hits move a slot to the newest
        // end; misses take the oldest slot their call does not route.
        std::int32_t newer = -1, older = -1;
        std::vector<std::uint64_t> written;
    };
    struct Deferred {
        std::size_t layer;
        std::int32_t expert, slot;
    };
    // A host-to-device copy of contiguous bytes from one pinned range or a run of file extents.
    struct Piece {
        std::byte* target = nullptr;
        const std::byte* pinned = nullptr;
        std::uintptr_t region = 0;
        std::vector<ExpertExtent> extents;
        std::uint64_t bytes = 0;
    };

    RankContext rank;
    ExpertDeliveryOptions options;
    std::vector<Layer> layers;
    std::uint32_t max_experts = 0;
    // Per blob: the slot offset/size (aligned for the largest blob and tail of any layer), the
    // bank region offset/size and the zero tail kept after a bank run.
    std::vector<std::uint64_t> slot_offset, slot_size, bank_offset, bank_size, tail;
    std::uint64_t slot_bytes = 0, bank_bytes = 0;
    Storage cache, cached_ids, missing_ids, needed_device;
    std::vector<Storage> banks;
    std::vector<Bank> bank_state;
    int last_bank = 0;
    std::unique_ptr<PinnedHostBuffer> needed_host, staging;
    std::vector<Slot> slots;
    std::vector<std::int32_t> free_slots;
    std::vector<Deferred> deferred;
    std::int32_t newest = -1, oldest = -1;
    CudaHostHandshake handshake;
    cudaStream_t copies = nullptr;
    cudaEvent_t copied[2]{};
    std::uint64_t half_bytes = 0;
    int half = 0;
    std::uint64_t staging_cursor = 0;
    std::vector<QueuedFileRead> batch_reads;
    struct StagedCopy { std::byte* target; const std::uint8_t* source; std::size_t bytes; };
    std::vector<StagedCopy> batch_copies;
    std::vector<std::ifstream> files;
    std::unique_ptr<FileReadQueue> read_queue;
    std::atomic<bool> stopping{false};
    std::atomic<std::uint32_t> finished{0}; // the last request whose service, post work included, ended
    std::thread service;
    mutable std::mutex mutex;
    std::exception_ptr failure;
    ExpertDeliveryStats counters;
    ExpertDeliveryMemory allocation;
    bool released = false;
    // The call being served and its post work: layer, bank, whether wide, and its routed experts.
    // Between a call's ready signal and the next request the GPU may read any layer's table (the
    // next split, then that call's cached pairs), so post work writes only free slots. Cached
    // experts are replaced only while a request is served, never one it routes.
    std::int64_t previous_layer = -1;
    std::size_t call_layer      = 0;
    int call_bank               = 0;
    bool call_wide              = false;
    std::vector<unsigned char> call_routed;
    std::vector<std::uint32_t> call_count;
    std::vector<Piece> pieces;
    std::vector<std::pair<std::byte*, std::uint64_t>> zeroes;

    Impl(const RankContext& r, std::vector<std::filesystem::path> paths,
         std::vector<ExpertDeliveryLayer> specs, ExpertDeliveryOptions o)
        : rank(r), options(o), handshake(r) {
        DeviceBinding bind(rank.device);
        if (specs.empty() || options.pair_capacity == 0) {
            throw std::invalid_argument("expert delivery: layers and pair capacity are required");
        }
        if (options.banks < 1 || options.banks > 2) {
            throw std::invalid_argument("expert delivery: one or two layer banks");
        }
        std::uint64_t expert_count = 0;
        bool disk = false;
        std::vector<std::uint64_t> largest, largest_tail;
        for (const auto& layer : specs) {
            expert_count += layer.experts.size();
            if (layer.experts.empty() || layer.aliases.empty()) {
                throw std::invalid_argument("expert delivery: empty expert or alias table");
            }
            if (layer.experts.size() > 8192) {
                throw std::invalid_argument("expert delivery: at most 8192 experts per layer");
            }
            max_experts = std::max(max_experts, std::uint32_t(layer.experts.size()));
            const std::size_t blobs = layer.experts.front().size();
            for (const auto& expert : layer.experts) {
                if (expert.empty() || expert.size() != blobs) {
                    throw std::invalid_argument("expert delivery: every expert of a layer needs its blobs");
                }
                largest.resize(std::max(largest.size(), expert.size()), 0);
                largest_tail.resize(largest.size(), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    const auto& blob = expert[b];
                    if (!blob.bytes || (blob.pinned != nullptr) == !blob.extents.empty()) {
                        throw std::invalid_argument("expert delivery: choose pinned bytes or extents");
                    }
                    std::uint64_t extent_bytes = 0;
                    for (const auto& extent : blob.extents) {
                        if (extent.file >= paths.size() || !extent.bytes ||
                            extent.bytes > blob.bytes - std::min(blob.bytes, extent_bytes)) {
                            throw std::invalid_argument("expert delivery: invalid file extent");
                        }
                        extent_bytes += extent.bytes;
                    }
                    if (!blob.pinned && extent_bytes != blob.bytes) {
                        throw std::invalid_argument("expert delivery: extents do not cover blob");
                    }
                    if (blob.zero_tail > std::numeric_limits<std::uint64_t>::max() - blob.bytes) {
                        throw std::invalid_argument("expert delivery: blob tail overflow");
                    }
                    largest[b]      = std::max(largest[b], blob.bytes);
                    largest_tail[b] = std::max(largest_tail[b], blob.zero_tail);
                    disk |= !blob.extents.empty();
                }
                for (const auto& alias : layer.aliases) {
                    if (alias.blob >= expert.size() || alias.offset >= expert[alias.blob].bytes) {
                        throw std::invalid_argument("expert delivery: alias outside blob");
                    }
                }
            }
        }
        for (std::size_t b = 0; b < largest.size(); ++b) {
            slot_offset.push_back(slot_bytes);
            slot_size.push_back(aligned(checked_add(largest[b], largest_tail[b])));
            slot_bytes = checked_add(slot_bytes, slot_size.back());
        }
        tail = largest_tail;
        bank_size.assign(largest.size(), 0);
        for (auto& spec : specs) {
            Layer layer;
            const auto count = spec.experts.size();
            const auto blobs = spec.experts.front().size();
            for (std::size_t b = 0; b < blobs; ++b) {
                const auto first = spec.experts.front()[b].bytes;
                const bool equal = std::all_of(spec.experts.begin(), spec.experts.end(),
                    [&](const auto& expert) { return expert[b].bytes == first; });
                // Kernels read these planes as the resident bank: every expert at its index.
                const bool resident = equal && first % 4 == 0;
                layer.stride.push_back(resident ? first : aligned(largest[b]));
                bank_size[b] = std::max(bank_size[b],
                                        checked_add(checked_mul(count, layer.stride.back()), tail[b]));
            }
            layer.slot_of.assign(count, -1);
            const auto bytes = count * spec.aliases.size() * sizeof(void*);
            layer.mirror = std::make_unique<PinnedHostBuffer>(bytes);
            std::memset(layer.mirror->data(), 0, bytes);
            for (int u = 0; u < 2; ++u) {
                layer.upload[u] = std::make_unique<PinnedHostBuffer>(bytes);
                CUDA_CHECK(cudaEventCreateWithFlags(&layer.uploaded[u], cudaEventDisableTiming));
                allocation.pinned_bytes += bytes;
            }
            layer.source = std::move(spec);
            allocation.pinned_bytes += bytes;
            layers.push_back(std::move(layer));
        }
        for (const auto size : bank_size) {
            bank_offset.push_back(bank_bytes);
            bank_bytes = checked_add(bank_bytes, aligned(size));
        }
        cached_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        missing_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        needed_device = Storage(std::size_t(max_experts) * 4, rank.device, options.suspendable);
        needed_host = std::make_unique<PinnedHostBuffer>(std::size_t(max_experts) * 4);
        std::memset(needed_host->data(), 0, needed_host->size());
        allocation.metadata_bytes = cached_ids.backing_bytes() + missing_ids.backing_bytes() +
                                     needed_device.backing_bytes();
        allocation.pinned_bytes += needed_host->size() + handshake.pinned_bytes();
        for (auto& layer : layers) {
            const auto bytes = layer.mirror->size();
            layer.tables = Storage(bytes, rank.device, options.suspendable);
            layer.tables.fill();
            allocation.metadata_bytes += layer.tables.backing_bytes();
        }
        if (disk) {
            half_bytes = aligned(std::max<std::uint64_t>(options.staging_bytes / 2, 256));
            staging = std::make_unique<PinnedHostBuffer>(2 * half_bytes);
            allocation.pinned_bytes += staging->size();
            for (const auto& path : paths) {
                files.emplace_back(path, std::ios::binary);
                if (!files.back()) {
                    throw std::runtime_error("expert delivery: cannot open " + path.string());
                }
            }
            read_queue = FileReadQueue::open(paths, false, 16);
        }
        const std::uint64_t granule = options.suspendable ? vmm::granularity(rank.device) : 1;
        if (!granule) {
            throw std::invalid_argument("expert delivery: suspend needs VMM allocation granularity");
        }
        const auto physical = [&](std::uint64_t bytes) {
            return checked_add(bytes, granule - 1) / granule * granule;
        };
        const auto available = [&] {
            std::size_t free = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
            return free > options.keep_free_bytes ? free - options.keep_free_bytes : 0;
        };
        // Banks before the cache: every miss lands in one. A second bank is dropped before the
        // last one, and an explicit cache keeps its size by dropping the second bank first.
        std::uint32_t bank_count = options.banks;
        const std::uint64_t explicit_cache =
            options.cache_bytes ? physical((*options.cache_bytes / slot_bytes) * slot_bytes) : 0;
        while (bank_count > 1 && bank_count * physical(bank_bytes) + explicit_cache > available()) {
            --bank_count;
        }
        if (physical(bank_bytes) > available()) {
            throw std::invalid_argument(
                "expert delivery: a layer bank needs " + std::to_string(physical(bank_bytes)) +
                " device bytes, but only " + std::to_string(available()) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        banks.reserve(bank_count);
        for (std::uint32_t b = 0; b < bank_count; ++b) {
            banks.emplace_back(bank_bytes, rank.device, options.suspendable);
            banks.back().fill();
            bank_state.push_back({-1, std::vector<unsigned char>(max_experts, 0)});
            allocation.bank_bytes += banks.back().backing_bytes();
        }
        allocation.banks = bank_count;
        const std::uint64_t remaining = available();
        // Automatic slots may use only whole physical granules inside the remaining budget.
        // An explicit request must preserve the same reserve, including VMM rounding.
        const auto budget = options.cache_bytes.value_or(remaining / granule * granule);
        const auto count = std::min<std::uint64_t>(budget / slot_bytes, expert_count);
        const auto cache_bytes = count * slot_bytes;
        const auto backing_bytes = physical(cache_bytes);
        if (backing_bytes > remaining) {
            throw std::invalid_argument(
                "expert delivery: cache needs " + std::to_string(backing_bytes) +
                " device bytes, but only " + std::to_string(remaining) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        if (count) {
            cache = Storage(cache_bytes, rank.device, options.suspendable);
            cache.fill();
            slots.resize(count);
            for (auto& slot : slots) { slot.written.assign(slot_offset.size(), 0); }
            for (std::int64_t s = std::int64_t(count) - 1; s >= 0; --s) {
                free_slots.push_back(std::int32_t(s));
            }
        }
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = std::uint32_t(slots.size());
        call_routed.assign(max_experts, 0);
        call_count.assign(max_experts, 0);
        CUDA_CHECK(cudaFuncSetAttribute(split_routes, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(max_experts * 4)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
        try {
            for (auto& event : copied) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
            service = std::thread([this] { run(); });
        } catch (...) {
            for (auto event : copied) { if (event) { cudaEventDestroy(event); } }
            cudaStreamDestroy(copies);
            throw;
        }
    }

    ~Impl() {
        DeviceBinding bind(rank.device);
        // Keep the worker alive until the last graph's waits and consumers finish.
        cudaStreamSynchronize(rank.stream);
        stopping.store(true, std::memory_order_release);
        if (service.joinable()) { service.join(); }
        cudaStreamSynchronize(copies);
        for (auto event : copied) { cudaEventDestroy(event); }
        for (auto& layer : layers) {
            for (auto event : layer.uploaded) { if (event) { cudaEventDestroy(event); } }
        }
        cudaStreamDestroy(copies);
    }

    void check() const {
        std::lock_guard lock(mutex);
        if (failure) { std::rethrow_exception(failure); }
    }

    template <typename Function> void each_storage(Function&& function) {
        for (Storage* storage : {&cache, &cached_ids, &missing_ids, &needed_device}) {
            if (storage->bytes) { function(*storage); }
        }
        for (auto& bank : banks) { function(bank); }
        for (auto& layer : layers) { function(layer.tables); }
    }

    [[nodiscard]] std::byte* slot_base(std::int32_t slot) const {
        return static_cast<std::byte*>(cache.p) + std::uint64_t(slot) * slot_bytes;
    }
    [[nodiscard]] std::byte* bank_blob(int bank, std::size_t li, std::int32_t e, std::size_t b) const {
        return static_cast<std::byte*>(banks[std::size_t(bank)].p) + bank_offset[b] +
               std::uint64_t(e) * layers[li].stride[b];
    }

    // The table entries of (li, e) point at base(b) for each alias's blob, or at nothing.
    template <typename Base> void point(std::size_t li, std::int32_t e, Base&& base) {
        auto& layer = layers[li];
        const auto n = layer.source.experts.size();
        auto** mirror = static_cast<const void**>(layer.mirror->data());
        for (std::size_t a = 0; a < layer.source.aliases.size(); ++a) {
            const auto alias = layer.source.aliases[a];
            std::byte* blob = base(alias.blob);
            mirror[a * n + std::size_t(e)] = blob ? blob + alias.offset : nullptr;
        }
        layer.dirty = true;
    }
    void point_slot(std::size_t li, std::int32_t e, std::int32_t slot) {
        std::byte* base = slot_base(slot);
        point(li, e, [&](std::size_t b) { return base + slot_offset[b]; });
    }
    void point_bank(std::size_t li, std::int32_t e, int bank) {
        point(li, e, [&](std::size_t b) { return bank_blob(bank, li, e, b); });
    }
    void point_nowhere(std::size_t li, std::int32_t e) {
        point(li, e, [](std::size_t) { return static_cast<std::byte*>(nullptr); });
    }

    // One copy per changed table, from a pinned copy no earlier upload still reads.
    void upload_tables() {
        for (auto& layer : layers) {
            if (!layer.dirty) { continue; }
            const int u = layer.next_upload;
            layer.next_upload = 1 - u;
            CUDA_CHECK(cudaEventSynchronize(layer.uploaded[u]));
            std::memcpy(layer.upload[u]->data(), layer.mirror->data(), layer.mirror->size());
            CUDA_CHECK(cudaMemcpyAsync(layer.tables.p, layer.upload[u]->data(), layer.mirror->size(),
                                       cudaMemcpyHostToDevice, copies));
            CUDA_CHECK(cudaEventRecord(layer.uploaded[u], copies));
            layer.dirty = false;
        }
    }

    // Queue a host-to-device copy, extending the previous piece when both its source and its
    // target continue it.
    void add_piece(std::byte* target, const ExpertBlob& blob) {
        if (!pieces.empty()) {
            auto& last = pieces.back();
            const bool target_follows = last.target + last.bytes == target;
            if (target_follows && blob.pinned && last.pinned && blob.pinned_region != 0 &&
                blob.pinned_region == last.region &&
                last.pinned + last.bytes == static_cast<const std::byte*>(blob.pinned)) {
                last.bytes += blob.bytes;
                return;
            }
            if (target_follows && !blob.pinned && !last.pinned) {
                for (const auto& extent : blob.extents) {
                    auto& end = last.extents.back();
                    if (end.file == extent.file && end.offset + end.bytes == extent.offset) {
                        end.bytes += extent.bytes;
                    } else {
                        last.extents.push_back(extent);
                    }
                }
                last.bytes += blob.bytes;
                return;
            }
        }
        Piece piece;
        piece.target = target;
        piece.pinned = static_cast<const std::byte*>(blob.pinned);
        piece.region = blob.pinned_region;
        piece.extents = blob.extents;
        piece.bytes = blob.bytes;
        pieces.push_back(std::move(piece));
    }

    // Several reads share one bounded half. FileReadQueue can submit their extents together.
    // Its fallback reads the same ranges through the OS page cache.
    void flush_reads() {
        if (batch_reads.empty()) { return; }
        CUDA_CHECK(cudaEventSynchronize(copied[half]));
        if (read_queue) { read_queue->read(batch_reads); }
        else {
            for (auto& read : batch_reads) {
                auto& file = files.at(read.file);
                file.clear();
                file.seekg(std::streamoff(read.offset));
                file.read(reinterpret_cast<char*>(read.destination.data()),
                          std::streamsize(read.destination.size()));
                if (!file) { throw std::runtime_error("expert delivery: incomplete file read"); }
            }
        }
        for (const auto& copy : batch_copies) {
            CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                       cudaMemcpyHostToDevice, copies));
        }
        CUDA_CHECK(cudaEventRecord(copied[half], copies));
        batch_reads.clear();
        batch_copies.clear();
        staging_cursor = 0;
        half = 1 - half;
    }

    void stage(const Piece& piece, ExpertDeliveryStats& delta) {
        std::uint64_t destination = 0;
        for (const auto& extent : piece.extents) {
            std::uint64_t within = 0;
            while (within < extent.bytes) {
                if (staging_cursor == half_bytes) { flush_reads(); }
                auto* buffer = static_cast<std::uint8_t*>(staging->data()) +
                               half * half_bytes + staging_cursor;
                const auto bytes = std::min(extent.bytes - within, half_bytes - staging_cursor);
                batch_reads.push_back({extent.file, extent.offset + within,
                                       {buffer, std::size_t(bytes)}});
                batch_copies.push_back({piece.target + destination, buffer, std::size_t(bytes)});
                staging_cursor += bytes;
                destination += bytes;
                within += bytes;
                delta.read_bytes += bytes;
            }
        }
    }

    void flush_pieces(ExpertDeliveryStats& delta) {
        for (const auto& piece : pieces) {
            if (piece.pinned) {
                CUDA_CHECK(cudaMemcpyAsync(piece.target, piece.pinned, piece.bytes,
                                           cudaMemcpyHostToDevice, copies));
            } else {
                stage(piece, delta);
            }
            delta.copied_bytes += piece.bytes;
        }
        flush_reads();
        for (const auto& [target, bytes] : zeroes) {
            CUDA_CHECK(cudaMemsetAsync(target, 0, bytes, copies));
        }
        pieces.clear();
        zeroes.clear();
    }

    // Bytes of blob b written into slot s: zero what a larger earlier blob left past them, so the
    // slot's tail stays zero.
    void slot_written(std::int32_t s, std::size_t b, std::uint64_t bytes) {
        auto& written = slots[std::size_t(s)].written[b];
        if (written > bytes) { zeroes.push_back({slot_base(s) + slot_offset[b] + bytes, written - bytes}); }
        written = bytes;
    }

    void unlink(std::int32_t s) {
        auto& slot = slots[std::size_t(s)];
        (slot.newer >= 0 ? slots[std::size_t(slot.newer)].older : newest) = slot.older;
        (slot.older >= 0 ? slots[std::size_t(slot.older)].newer : oldest) = slot.newer;
        slot.newer = slot.older = -1;
    }
    void link_newest(std::int32_t s) {
        auto& slot = slots[std::size_t(s)];
        slot.older = newest;
        slot.newer = -1;
        (newest >= 0 ? slots[std::size_t(newest)].newer : oldest) = s;
        newest = s;
    }
    void link_oldest(std::int32_t s) {
        auto& slot = slots[std::size_t(s)];
        slot.newer = oldest;
        slot.older = -1;
        (oldest >= 0 ? slots[std::size_t(oldest)].older : newest) = s;
        oldest = s;
    }

    void evict(std::int32_t s, ExpertDeliveryStats& delta) {
        auto& slot = slots[std::size_t(s)];
        unlink(s);
        auto& layer = layers[std::size_t(slot.layer)];
        layer.slot_of[std::size_t(slot.expert)] = -1;
        --layer.cached;
        point_nowhere(std::size_t(slot.layer), slot.expert);
        slot.layer = slot.expert = -1;
        slot.pending = false;
        ++delta.evictions;
    }

    // A free slot, else the least recently used one the served call does not route. -1: none.
    std::int32_t choose_slot(ExpertDeliveryStats& delta) {
        if (!free_slots.empty()) {
            const auto s = free_slots.back();
            free_slots.pop_back();
            return s;
        }
        for (auto s = oldest; s >= 0; s = slots[std::size_t(s)].newer) {
            const auto& slot = slots[std::size_t(s)];
            if (slot.pending) { continue; }
            if (std::size_t(slot.layer) == call_layer && call_routed[std::size_t(slot.expert)]) {
                continue;
            }
            evict(s, delta);
            return s;
        }
        return -1;
    }

    void occupy(std::int32_t s, std::size_t li, std::int32_t e, bool pending) {
        auto& slot = slots[std::size_t(s)];
        slot.layer = std::int32_t(li);
        slot.expert = e;
        slot.pending = pending;
        layers[li].slot_of[std::size_t(e)] = s;
        ++layers[li].cached;
    }

    [[nodiscard]] int bank_holding(std::size_t li) const {
        for (std::size_t b = 0; b < bank_state.size(); ++b) {
            if (bank_state[b].layer == std::int64_t(li)) { return int(b); }
        }
        return -1;
    }

    // A served call may write any bank: the preceding call's consumers finished before it was
    // published. It takes the bank the last bank-using call did not use, which keeps that layer
    // for an immediate repeat (a second prefill slice, the next draft of the same layer).
    int claim_bank(std::size_t li, int avoid) {
        const int pick = bank_state.size() == 1 ? 0 : avoid >= 0 ? 1 - avoid : 1 - last_bank;
        auto& bank = bank_state[std::size_t(pick)];
        bank.layer = std::int64_t(li);
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
        return pick;
    }

    // Copy experts of layer li into bank k at their indices, one piece per contiguous run, and keep
    // zeros after the last expert of each run that the bank does not already continue.
    void fill_bank(int k, std::size_t li, const std::vector<std::int32_t>& experts) {
        auto& layer = layers[li];
        auto& bank = bank_state[std::size_t(k)];
        const auto blobs = layer.source.experts.front().size();
        for (std::size_t b = 0; b < blobs; ++b) {
            for (const auto e : experts) {
                add_piece(bank_blob(k, li, e, b), layer.source.experts[std::size_t(e)][b]);
            }
        }
        for (const auto e : experts) { bank.valid[std::size_t(e)] = 1; }
        for (std::size_t b = 0; b < blobs; ++b) {
            if (!tail[b]) { continue; }
            for (const auto e : experts) {
                const auto next = std::size_t(e) + 1;
                if (next < layer.source.experts.size() && bank.valid[next]) { continue; }
                const auto end = std::uint64_t(e) * layer.stride[b] +
                                 layer.source.experts[std::size_t(e)][b].bytes;
                const auto room = bank_size[b] - std::min(bank_size[b], end);
                if (room) {
                    zeroes.push_back({static_cast<std::byte*>(banks[std::size_t(k)].p) +
                                          bank_offset[b] + end,
                                      std::min(tail[b], room)});
                }
            }
        }
    }

    // Device copies of bank expert e into slot s.
    void promote(std::int32_t s, std::size_t li, std::int32_t e, int bank, ExpertDeliveryStats& delta) {
        for (std::size_t b = 0; b < layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto bytes = layers[li].source.experts[std::size_t(e)][b].bytes;
            CUDA_CHECK(cudaMemcpyAsync(slot_base(s) + slot_offset[b], bank_blob(bank, li, e, b),
                                       bytes, cudaMemcpyDeviceToDevice, copies));
            slot_written(s, b, bytes);
            delta.promoted_bytes += bytes;
        }
        ++delta.admissions;
    }

    void serve(const CudaHostHandshake::Request& request) {
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        // Promotions of an earlier wide call: its consumers are done, so the tables may move.
        for (const auto& d : deferred) {
            auto& slot = slots[std::size_t(d.slot)];
            if (slot.layer == std::int32_t(d.layer) && slot.expert == d.expert && slot.pending) {
                slot.pending = false;
                point_slot(d.layer, d.expert, d.slot);
            }
        }
        deferred.clear();
        const std::size_t li = request.tag;
        auto& layer = layers.at(li);
        call_layer = li;
        const auto* needed = static_cast<const std::uint32_t*>(needed_host->data());
        const int experts = int(layer.source.experts.size());
        delta.requests = 1;
        std::uint32_t routes = 0;
        std::fill(call_routed.begin(), call_routed.end(), 0);
        for (int e = 0; e < experts; ++e) {
            const std::uint32_t count = needed[e] & kCountMask;
            call_count[std::size_t(e)] = count;
            if (!count) { continue; }
            ++routes;
            call_routed[std::size_t(e)] = 1;
            delta.pairs += count;
            if (needed[e] & kHit) {
                ++delta.hits;
                delta.pair_hits += count;
            }
        }
        delta.routes = routes;
        const bool wide = 2 * routes >= std::uint32_t(experts);
        if (!wide) {
            // Decode and verification keep their experts recent; a prompt chunk, which routes to
            // nearly every expert, leaves the order alone.
            for (int e = 0; e < experts; ++e) {
                const auto s = layer.slot_of[std::size_t(e)];
                if (call_routed[std::size_t(e)] && s >= 0 && !slots[std::size_t(s)].pending) {
                    unlink(s);
                    link_newest(s);
                }
            }
        }
        if (previous_layer >= 0) {
            layers[std::size_t(previous_layer)].successor = std::int64_t(li);
            layers[std::size_t(previous_layer)].successor_wide = wide;
        }
        previous_layer = std::int64_t(li);
        const int held = bank_holding(li);
        int k = held;
        std::vector<std::int32_t> to_bank;
        for (int e = 0; e < experts; ++e) {
            if (!(needed[e] & kMiss)) { continue; }
            ++delta.misses;
            if (layer.slot_of[std::size_t(e)] >= 0) {
                // Promoted after the split read its entry.
                point_slot(li, e, layer.slot_of[std::size_t(e)]);
                continue;
            }
            if (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)]) {
                ++delta.bank_reuses;
                point_bank(li, e, held);
                continue;
            }
            if (!wide) {
                const auto s = choose_slot(delta);
                if (s >= 0) {
                    occupy(s, li, e, false);
                    link_newest(s);
                    for (std::size_t b = 0; b < layer.source.experts[std::size_t(e)].size(); ++b) {
                        const auto& blob = layer.source.experts[std::size_t(e)][b];
                        add_piece(slot_base(s) + slot_offset[b], blob);
                        slot_written(s, b, blob.bytes);
                    }
                    ++delta.admissions;
                    point_slot(li, e, s);
                    continue;
                }
            }
            to_bank.push_back(e);
        }
        if (!to_bank.empty() && k < 0) { k = claim_bank(li, -1); }
        for (const auto e : to_bank) { point_bank(li, e, k); }
        if (k >= 0) { last_bank = k; }
        call_bank = k;
        call_wide = wide;
        if (!to_bank.empty()) { fill_bank(k, li, to_bank); }
        flush_pieces(delta);
        upload_tables();
        CUDA_CHECK(cudaStreamSynchronize(copies));
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    // After the call's experts are ready: a wide call promotes its most routed bank experts into
    // free cache slots by device copies, up to the layer's share of the cache, so a first prompt
    // fills the cache across every layer and evicts nothing. It also starts the bank of the layer
    // that followed it last time.
    void post() {
        if (!call_wide) { return; }
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        const std::size_t li = call_layer;
        auto& layer = layers[li];
        if (!slots.empty() && call_bank >= 0) {
            const auto& bank = bank_state[std::size_t(call_bank)];
            std::vector<std::int32_t> candidates;
            for (std::size_t e = 0; e < layer.source.experts.size(); ++e) {
                if (call_routed[e] && bank.valid[e] && layer.slot_of[e] < 0) {
                    candidates.push_back(std::int32_t(e));
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](std::int32_t a, std::int32_t b) {
                const auto ca = call_count[std::size_t(a)], cb = call_count[std::size_t(b)];
                return ca != cb ? ca > cb : a < b;
            });
            const auto share = std::max<std::size_t>(1, slots.size() / layers.size());
            for (const auto e : candidates) {
                if (free_slots.empty() || layer.cached >= share) { break; }
                const auto s = free_slots.back();
                free_slots.pop_back();
                occupy(s, li, e, true);
                link_oldest(s);
                promote(s, li, e, call_bank, delta);
                deferred.push_back({li, e, s});
            }
        }
        if (bank_state.size() == 2 && layer.successor >= 0 && layer.successor_wide) {
            const auto next = std::size_t(layer.successor);
            const bool held = std::any_of(bank_state.begin(), bank_state.end(),
                                          [&](const Bank& b) { return b.layer == std::int64_t(next); });
            std::vector<std::int32_t> experts;
            for (std::size_t e = 0; e < layers[next].source.experts.size() && !held; ++e) {
                if (layers[next].slot_of[e] < 0) { experts.push_back(std::int32_t(e)); }
            }
            if (!experts.empty()) {
                // Never the bank this call's consumers are reading.
                const int k = claim_bank(next, call_bank);
                fill_bank(k, next, experts);
                ++delta.prefetched_layers;
            }
        }
        flush_pieces(delta); // tables are unchanged: promoted entries are patched at the next request
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    void add(const ExpertDeliveryStats& d) {
        std::lock_guard lock(mutex);
        counters = counters + d;
    }

    void fail(const CudaHostHandshake::Request* request) {
        {
            std::lock_guard lock(mutex);
            if (!failure) { failure = std::current_exception(); }
        }
        batch_reads.clear();
        batch_copies.clear();
        pieces.clear();
        zeroes.clear();
        staging_cursor = 0;
        if (request) {
            // Captured decode can finish safely; the caller checks before publishing output.
            cudaMemsetAsync(missing_ids.p, 0xff, std::size_t(request->width) * 4, copies);
        }
        cudaStreamSynchronize(copies);
    }

    void run() {
        DeviceBinding bind(rank.device);
        std::uint32_t served = 0;
        auto active_until = std::chrono::steady_clock::now();
        while (!stopping.load(std::memory_order_acquire)) {
            const auto request = handshake.pending(served);
            if (!request) {
                if (std::chrono::steady_clock::now() < active_until) { std::this_thread::yield(); }
                else { std::this_thread::sleep_for(std::chrono::microseconds(20)); }
                continue;
            }
            bool served_ok = false;
            try { check(); serve(*request); served_ok = true; }
            catch (...) { fail(&*request); }
            handshake.prepared(request->sequence);
            served = request->sequence;
            if (served_ok) {
                try { post(); }
                catch (...) { fail(nullptr); }
            }
            finished.store(served, std::memory_order_release);
            active_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        }
    }

    // After the compute stream is idle: the worker has finished every published request,
    // post work included, and its copies are done.
    void quiesce() {
        DeviceBinding bind(rank.device);
        CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        while (const auto request = handshake.pending(finished.load(std::memory_order_acquire))) {
            (void)request;
            std::this_thread::yield();
        }
        CUDA_CHECK(cudaStreamSynchronize(copies));
    }
};

ExpertDelivery::ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                               std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options)
    : impl_(std::make_unique<Impl>(rank, std::move(files), std::move(layers), options)) {}
ExpertDelivery::~ExpertDelivery() = default;

ExpertDelivery::Call ExpertDelivery::begin(std::size_t index, const std::int32_t* ids,
                                          std::uint32_t pairs, cudaStream_t stream) {
    auto& p = *impl_;
    p.check();
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    const auto& layer = p.layers.at(index);
    if (!ids || !pairs || pairs > p.options.pair_capacity || stream != p.rank.stream) {
        throw std::invalid_argument("expert delivery: invalid call or compute stream");
    }
    const auto low = reinterpret_cast<std::uintptr_t>(p.cache.p);
    const int experts = int(layer.source.experts.size());
    split_routes<<<1, 1024, std::size_t(experts) * 4, stream>>>(
        ids, pairs, experts, static_cast<const void* const*>(layer.tables.p), low,
        low + p.cache.bytes, static_cast<std::int32_t*>(p.cached_ids.p),
        static_cast<std::int32_t*>(p.missing_ids.p), static_cast<std::uint32_t*>(p.needed_device.p));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(p.needed_host->data(), p.needed_device.p, std::size_t(experts) * 4,
                               cudaMemcpyDeviceToHost, stream));
    p.handshake.publish(std::uint32_t(index), pairs, 0, stream);
    return {static_cast<const std::int32_t*>(p.cached_ids.p),
            static_cast<const std::int32_t*>(p.missing_ids.p)};
}
void ExpertDelivery::await(cudaStream_t stream) {
    impl_->handshake.wait_prepared(stream);
    impl_->check();
}
const void* const* ExpertDelivery::tables(std::size_t layer) const {
    return static_cast<const void* const*>(impl_->layers.at(layer).tables.p);
}
void ExpertDelivery::check() const { impl_->check(); }
void ExpertDelivery::drain() {
    impl_->quiesce();
    impl_->check();
}
void ExpertDelivery::invalidate() {
    auto& p = *impl_;
    if (p.released) { throw std::logic_error("expert delivery: restore backing before invalidation"); }
    drain();
    for (auto& layer : p.layers) {
        layer.tables.fill();
        std::memset(layer.mirror->data(), 0, layer.mirror->size());
        std::fill(layer.slot_of.begin(), layer.slot_of.end(), -1);
        layer.cached = 0;
        layer.dirty = false;
    }
    p.free_slots.clear();
    for (std::int64_t s = std::int64_t(p.slots.size()) - 1; s >= 0; --s) {
        auto& slot = p.slots[std::size_t(s)];
        slot.layer = slot.expert = -1;
        slot.pending = false;
        slot.newer = slot.older = -1;
        p.free_slots.push_back(std::int32_t(s));
    }
    p.newest = p.oldest = -1;
    for (auto& bank : p.bank_state) {
        bank.layer = -1;
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
    }
    p.deferred.clear();
    p.previous_layer = -1;
}
void ExpertDelivery::release_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (p.released) { return; }
    drain();
    DeviceBinding bind(p.rank.device);
    p.each_storage([](Storage& storage) { storage.region.release_backing(); });
    p.released = true;
}
void ExpertDelivery::restore_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (!p.released) { return; }
    DeviceBinding bind(p.rank.device);
    try {
        p.each_storage([](Storage& storage) { storage.region.restore_backing(); });
        p.each_storage([](Storage& storage) { storage.fill(); });
        for (auto& slot : p.slots) { std::fill(slot.written.begin(), slot.written.end(), 0); }
        p.released = false;
        invalidate();
    } catch (...) {
        p.each_storage([](Storage& storage) { storage.region.release_backing(); });
        p.released = true;
        throw;
    }
}
std::uint64_t ExpertDelivery::state_backing_bytes() const noexcept {
    const auto memory = impl_->allocation;
    return memory.cache_bytes + memory.bank_bytes + memory.metadata_bytes;
}
ExpertDeliveryStats ExpertDelivery::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->counters;
}
ExpertDeliveryMemory ExpertDelivery::memory() const noexcept { return impl_->allocation; }

} // namespace ninfer
