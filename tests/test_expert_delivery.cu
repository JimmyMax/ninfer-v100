#include "core/arena.h"
#include "core/device.h"
#include "core/expert_delivery.h"
#include "core/vmm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
namespace {
constexpr int kLayers = 4, kExperts = 16, kPairs = 2048, kTail = 512;
constexpr std::uint64_t kGu = 131072, kDown = 98304;

__host__ __device__ unsigned char signature(int layer, int expert, int blob, std::uint64_t at) {
    auto x = std::uint32_t(at) * 0x27d4eb2du ^ std::uint32_t(layer + 1) * 0x9e3779b1u ^
             std::uint32_t(expert + 1) * 0x85ebca77u ^ std::uint32_t(blob + 1) * 0xc2b2ae3du;
    x ^= x >> 15;
    return static_cast<unsigned char>(x);
}

__global__ void inspect(const void* const* table, const std::int32_t* ids, int pairs, int layer,
                        unsigned long long* failures, unsigned long long* visited) {
    const int p = blockIdx.x;
    if (p >= pairs || ids[p] < 0) { return; }
    const int expert = ids[p];
    if (threadIdx.x == 0) { atomicAdd(visited, 1ull); }
    for (int b = 0; b < 2; ++b) {
        const auto* data = static_cast<const unsigned char*>(table[(b ? 3 : 2) * kExperts + expert]);
        if (!data) { if (!threadIdx.x) { atomicAdd(failures, 1ull); } continue; }
        const auto bytes = b ? kDown - std::uint64_t(layer) * 256 : kGu;
        // Every represented byte, its last byte, and every tail byte.
        for (std::uint64_t at = threadIdx.x; at < bytes; at += blockDim.x) {
            if (data[at] != signature(layer, expert, b, at)) { atomicAdd(failures, 1ull); }
        }
        if (data[bytes - 1] != signature(layer, expert, b, bytes - 1)) { atomicAdd(failures, 1ull); }
        // A kernel may read kTail bytes past a blob: zeros, or in a layer bank the next expert of
        // the same layer and blob, laid out as in a resident bank.
        for (int at = threadIdx.x; at < kTail; at += blockDim.x) {
            const unsigned char value = data[bytes + at];
            if (value && (expert + 1 == kExperts || value != signature(layer, expert + 1, b, at))) {
                atomicAdd(failures, 1ull);
            }
        }
        if (!b && !threadIdx.x) {
            if (table[expert] != data || table[kExperts + expert] != data + kGu / 2) {
                atomicAdd(failures, 1ull);
            }
            if (reinterpret_cast<std::uintptr_t>(data) % 256) { atomicAdd(failures, 1ull); }
        }
    }
}

struct Fixture {
    std::vector<std::unique_ptr<PinnedHostBuffer>> pinned;
    std::vector<ExpertDeliveryLayer> layers;
    std::vector<std::filesystem::path> paths;
    explicit Fixture(bool disk, const std::filesystem::path& directory) {
        std::array<std::ofstream, 2> output;
        if (disk) {
            for (int i = 0; i < 2; ++i) {
                paths.push_back(directory / ("part-" + std::to_string(i)));
                output[i].open(paths[i], std::ios::binary);
                if (!output[i]) { throw std::runtime_error("fixture open failed"); }
            }
        }
        std::uint64_t file_offsets[2]{};
        for (int l = 0; l < kLayers; ++l) {
            ExpertDeliveryLayer layer;
            layer.aliases = {{0, 0}, {0, kGu / 2}, {0, 0}, {1, 0}};
            // Host blobs of one layer and blob index share an allocation, experts in order, as a
            // model's pinned parent does: runs of them copy as one.
            std::array<std::byte*, 2> host{};
            if (!disk) {
                for (int b = 0; b < 2; ++b) {
                    pinned.push_back(std::make_unique<PinnedHostBuffer>(
                        kExperts * (b ? kDown - std::uint64_t(l) * 256 : kGu)));
                    host[b] = static_cast<std::byte*>(pinned.back()->data());
                }
            }
            for (int e = 0; e < kExperts; ++e) {
                std::vector<ExpertBlob> expert;
                for (int b = 0; b < 2; ++b) {
                    ExpertBlob blob;
                    blob.bytes = b ? kDown - std::uint64_t(l) * 256 : kGu;
                    blob.zero_tail = kTail;
                    std::vector<unsigned char> data(blob.bytes);
                    for (std::uint64_t i = 0; i < blob.bytes; ++i) { data[i] = signature(l, e, b, i); }
                    if (disk) {
                        const auto split = blob.bytes / 3;
                        output[0].write(reinterpret_cast<const char*>(data.data()), split);
                        output[1].write(reinterpret_cast<const char*>(data.data() + split), blob.bytes - split);
                        blob.extents = {{0, file_offsets[0], split},
                                        {1, file_offsets[1], blob.bytes - split}};
                        file_offsets[0] += split;
                        file_offsets[1] += blob.bytes - split;
                    } else {
                        auto* target = host[b] + std::uint64_t(e) * blob.bytes;
                        std::memcpy(target, data.data(), data.size());
                        blob.pinned = target;
                        blob.pinned_region = reinterpret_cast<std::uintptr_t>(host[b]);
                    }
                    expert.push_back(std::move(blob));
                }
                layer.experts.push_back(std::move(expert));
            }
            layers.push_back(std::move(layer));
        }
    }
};

int run(DeviceContext& device, bool disk, std::uint64_t cache, const std::filesystem::path& path,
        bool shifting = false) {
    Fixture fixture(disk, path);
    ExpertDeliveryOptions options;
    options.pair_capacity = kPairs;
    options.cache_bytes = cache;
    options.staging_bytes = 4096; // exercise repeated DMA half reuse inside one blob
    options.suspendable = true;
    ExpertDelivery delivery(device.rank(0), fixture.paths, std::move(fixture.layers), options);
    DeviceBuffer ids(kPairs * 4), failures(8), visited(8);
    failures.fill();
    visited.fill();
    const auto pass = [&] {
        for (int l = 0; l < kLayers; ++l) {
            const auto call = delivery.begin(l, static_cast<const std::int32_t*>(ids.p), kPairs, device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.cached_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            delivery.await(device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.missing_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            CUDA_CHECK(cudaGetLastError());
        }
    };
    std::vector<std::int32_t> routes(kPairs);
    for (int i = 0; i < kPairs; ++i) { routes[i] = i % 5; }
    ids.copy_from_host(routes.data(), ids.bytes);
    pass(); // load all kernels before a captured GPU wait can run
    delivery.drain();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
    pass();
    CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    constexpr int repeats = 12;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        for (int p = 0; p < kPairs; ++p) {
            if (shifting) {
                // Narrow calls whose five experts move by three every replay, and a wide call every
                // fourth: once the cache is full, admissions evict the least recently used experts
                // of every layer while the GPU reads the next call's cached pairs.
                routes[p] = iteration % 4 == 3 ? p % kExperts : (3 * iteration + p % 5) % kExperts;
            } else {
                routes[p] = iteration % 3 == 0 ? p % 5 : (p + iteration) % kExperts;
            }
        }
        ids.copy_from_host(routes.data(), ids.bytes);
        CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
        delivery.drain();
        if (iteration == 5) {
            const auto* addresses = delivery.tables(0);
            const auto bytes = delivery.state_backing_bytes();
            if (!bytes) { throw std::runtime_error("suspend did not account device backing"); }
            delivery.release_device_backing();
            delivery.restore_device_backing();
            if (delivery.tables(0) != addresses) {
                throw std::runtime_error("suspend changed graph table address");
            }
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    unsigned long long bad = 0, count = 0;
    failures.copy_to_host(&bad, 8);
    visited.copy_to_host(&count, 8);
    const auto stats = delivery.stats();
    const auto memory = delivery.memory();
    // Two of three replays route all sixteen experts of every layer (wide calls): without a cache
    // the next layer's bank is prefetched and reused; with one, wide calls promote by device copies.
    const bool ok = !bad && count == std::uint64_t(repeats + 1) * kLayers * kPairs &&
                    stats.routes == stats.hits + stats.misses && stats.misses &&
                    (cache ? stats.hits != 0 : stats.hits == 0) &&
                    (disk ? stats.read_bytes == stats.copied_bytes : stats.read_bytes == 0) &&
                    (cache ? memory.cache_bytes != 0 : memory.cache_bytes == 0) &&
                    memory.banks == 2 &&
                    (cache ? shifting || stats.promoted_bytes != 0
                           : stats.prefetched_layers != 0 && stats.bank_reuses != 0) &&
                    (!shifting || stats.evictions != 0);
    std::printf("%s%s cache=%llu bad=%llu pairs=%llu hits=%llu misses=%llu evictions=%llu "
                "admissions=%llu bank_reuses=%llu prefetched=%llu read=%llu copied=%llu "
                "promoted=%llu banks=%u %s\n", disk ? "disk" : "host", shifting ? " shifting" : "",
                (unsigned long long)cache, bad, count, (unsigned long long)stats.hits,
                (unsigned long long)stats.misses, (unsigned long long)stats.evictions,
                (unsigned long long)stats.admissions, (unsigned long long)stats.bank_reuses,
                (unsigned long long)stats.prefetched_layers, (unsigned long long)stats.read_bytes,
                (unsigned long long)stats.copied_bytes, (unsigned long long)stats.promoted_bytes,
                memory.banks, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int failure_control(DeviceContext& device, const std::filesystem::path& path, bool omit_wait) {
    Fixture fixture(true, path);
    ExpertDeliveryOptions options;
    options.pair_capacity = kPairs;
    options.cache_bytes = 0;
    options.staging_bytes = 4096;
    ExpertDelivery delivery(device.rank(0), fixture.paths, std::move(fixture.layers), options);
    DeviceBuffer ids(kPairs * 4), failures(8), visited(8);
    failures.fill();
    visited.fill();
    std::vector<std::int32_t> routes(kPairs, -1);
    ids.copy_from_host(routes.data(), ids.bytes);
    // Load inspect before an asynchronous transfer can start.
    inspect<<<1, 32, 0, device.stream>>>(delivery.tables(0), static_cast<const std::int32_t*>(ids.p),
        1, 0, static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::fill(routes.begin(), routes.end(), 0);
    ids.copy_from_host(routes.data(), ids.bytes);
    if (!omit_wait) { std::filesystem::resize_file(fixture.paths.front(), 0); }
    const auto call = delivery.begin(0, static_cast<const std::int32_t*>(ids.p), kPairs, device.stream);
    if (omit_wait) {
        inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(0), call.missing_ids, kPairs, 0,
            static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
        delivery.await(device.stream);
        delivery.drain();
        unsigned long long bad = 0;
        failures.copy_to_host(&bad, 8);
        std::printf("positive control without ready dependency: %llu mismatches\n", bad);
        return bad ? 1 : 0; // this intentionally broken execution must fail
    }
    try { delivery.await(device.stream); }
    catch (const std::exception&) {
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        std::puts("truncated source: error propagated and GPU wait released PASS");
        return 0;
    }
    std::puts("FAIL truncated source was accepted");
    return 1;
}

int admission_boundaries(DeviceContext& device) {
    PinnedHostBuffer source(256);
    std::memset(source.data(), 0, source.size());
    ExpertDeliveryLayer layer;
    layer.experts = {{ExpertBlob{.pinned = source.data(), .bytes = 256}}};
    layer.aliases = {{0, 0}};
    for (const bool suspendable : {false, true}) {
        ExpertDeliveryOptions options;
        options.pair_capacity = 1;
        options.suspendable = suspendable;
        options.banks = 1;
        // Every miss needs a layer bank: with nothing available it is refused before allocation.
        options.keep_free_bytes = std::numeric_limits<std::uint64_t>::max();
        options.cache_bytes = 0;
        bool rejected = false;
        try { ExpertDelivery delivery(device.rank(0), {}, {layer}, options); }
        catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()).find("expert delivery: a layer bank needs ") == 0;
        }
        if (!rejected) { throw std::runtime_error("a layer bank consumed the reserved budget"); }
        // The exact zero-cache boundary remains valid, both explicitly and automatically.
        options.keep_free_bytes = 0;
        std::uint64_t mandatory_backing = 0;
        {
            ExpertDelivery delivery(device.rank(0), {}, {layer}, options);
            if (delivery.memory().cache_bytes || delivery.memory().banks != 1) {
                throw std::runtime_error("zero cache allocated backing");
            }
            mandatory_backing = delivery.state_backing_bytes();
        }
        // With available room, report the physical size of a sub-granule slot, not its 256-byte
        // logical size.
        options.cache_bytes = 256;
        {
            ExpertDelivery delivery(device.rank(0), {}, {layer}, options);
            const auto granule = suspendable ? vmm::granularity(device.device) : 1;
            const auto expected = (256 + granule - 1) / granule * granule;
            if (delivery.memory().cache_bytes != expected || delivery.memory().cache_slots != 1) {
                throw std::runtime_error("sub-granule cache did not report its physical backing");
            }
        }
#ifndef _WIN32
        // Linux pinned buffers do not consume device backing. Use the observed mandatory
        // footprint, without assuming its allocation layout, to leave 256 bytes for the cache.
        // The logical slot fits; its VMM granule must be refused, and auto must choose zero.
        if (suspendable) {
            const auto granule = vmm::granularity(device.device);
            if (granule > 256) {
                std::size_t free = 0, total = 0;
                CUDA_CHECK(cudaMemGetInfo(&free, &total));
                if (free <= mandatory_backing + granule) {
                    throw std::runtime_error("insufficient free memory for cache admission boundary test");
                }
                options.keep_free_bytes = free - mandatory_backing - 256;
                rejected = false;
                try { ExpertDelivery delivery(device.rank(0), {}, {layer}, options); }
                catch (const std::invalid_argument& error) {
                    rejected = std::string(error.what()).find("but only 256 remain") != std::string::npos;
                    if (!rejected) { throw std::runtime_error(error.what()); }
                }
                if (!rejected) { throw std::runtime_error("VMM cache rounding consumed the reserve"); }
                CUDA_CHECK(cudaMemGetInfo(&free, &total));
                options.keep_free_bytes = free - mandatory_backing - (granule - 1);
                options.cache_bytes.reset();
                ExpertDelivery delivery(device.rank(0), {}, {layer}, options);
                if (delivery.memory().cache_bytes) {
                    throw std::runtime_error("auto cache rounded past its physical remainder");
                }
            }
        }
#endif
        std::printf("cache admission suspendable=%d bank reject, zero/auto accept, "
                    "physical rounding PASS\n", int(suspendable));
    }
    return 0;
}

// Narrow calls on one layer with a four-slot cache, against a host model of the policy: a call's
// cached experts become the most recent in ascending order, then each miss in ascending order
// takes a free slot or evicts the least recently used expert the call does not route.
int recency_order(DeviceContext& device) {
    constexpr int experts = 12, slots = 4, calls = 400;
    std::vector<std::unique_ptr<PinnedHostBuffer>> sources;
    ExpertDeliveryLayer layer;
    layer.aliases = {{0, 0}};
    sources.push_back(std::make_unique<PinnedHostBuffer>(std::size_t(experts) * 256));
    std::memset(sources.back()->data(), 0, sources.back()->size());
    for (int e = 0; e < experts; ++e) {
        auto* data = static_cast<std::byte*>(sources.back()->data()) + std::size_t(e) * 256;
        layer.experts.push_back({ExpertBlob{.pinned = data,
            .pinned_region = reinterpret_cast<std::uintptr_t>(sources.back()->data()), .bytes = 256}});
    }
    ExpertDeliveryOptions options;
    options.pair_capacity = 8;
    options.cache_bytes = slots * 256;
    ExpertDelivery delivery(device.rank(0), {}, {layer}, options);
    if (delivery.memory().cache_slots != slots) { throw std::runtime_error("recency: slot count"); }
    DeviceBuffer ids(8 * 4);
    std::vector<int> order; // least recent first
    std::uint64_t state = 12345, failures = 0;
    for (int call = 0; call < calls; ++call) {
        std::vector<std::int32_t> routed;
        const int width = 1 + int((state = state * 6364136223846793005ULL + 1442695040888963407ULL) >> 62);
        while (int(routed.size()) < width) {
            const int e = int((state = state * 6364136223846793005ULL + 1442695040888963407ULL) >> 33) % 7 +
                          (call / 100) % 2 * 5; // the hot set moves halfway through
            if (std::find(routed.begin(), routed.end(), e) == routed.end()) { routed.push_back(e); }
        }
        std::sort(routed.begin(), routed.end());
        std::uint64_t expected_hits = 0;
        for (const int e : routed) {
            const auto at = std::find(order.begin(), order.end(), e);
            if (at != order.end()) { ++expected_hits; order.erase(at); order.push_back(e); }
        }
        for (const int e : routed) {
            if (std::find(order.begin(), order.end(), e) != order.end()) { continue; }
            if (int(order.size()) == slots) {
                const auto victim = std::find_if(order.begin(), order.end(), [&](int v) {
                    return std::find(routed.begin(), routed.end(), v) == routed.end();
                });
                order.erase(victim);
            }
            order.push_back(e);
        }
        ids.copy_from_host(routed.data(), routed.size() * 4);
        const auto before = delivery.stats();
        (void)delivery.begin(0, static_cast<const std::int32_t*>(ids.p), std::uint32_t(routed.size()),
                             device.stream);
        delivery.await(device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        delivery.drain();
        if (delivery.stats().hits - before.hits != expected_hits) { ++failures; }
    }
    const auto stats = delivery.stats();
    const bool ok = !failures && stats.evictions != 0;
    std::printf("recency order calls=%d hits=%llu evictions=%llu mismatched=%llu %s\n", calls,
                (unsigned long long)stats.hits, (unsigned long long)stats.evictions,
                (unsigned long long)failures, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        DeviceContext device(0);
        if (argc == 2 && std::string(argv[1]) == "--admission-only") {
            return admission_boundaries(device);
        }
        const auto path = std::filesystem::temp_directory_path() /
            ("ninfer-expert-delivery-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
        struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{path};
        if (argc == 2 && std::string(argv[1]) == "--no-ready-wait") {
            return failure_control(device, path, true);
        }
        int failures = admission_boundaries(device);
        failures += recency_order(device);
        for (bool disk : {false, true}) {
            for (auto cache : {0ULL, 6ULL << 20, 64ULL << 20}) {
                failures += run(device, disk, cache, path);
            }
            failures += run(device, disk, 6ULL << 20, path, true);
        }
        failures += failure_control(device, path, false);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
