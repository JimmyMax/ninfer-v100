#include "models/qwen4_exp/expert_misses.h"

#include "core/file_read_error.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cuda/atomic>
#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <type_traits>
#include <utility>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <fstream>
#    include <sched.h>
#endif

namespace ninfer::models::qwen4_exp {
namespace {

// Logical CPUs this process may keep busy: std::thread's count, bounded on Linux by the affinity
// mask and by a container's CPU quota (cgroup v2 cpu.max, v1 cpu.cfs_quota_us), so spinning
// workers do not exhaust the quota and stall the executor's thread.
unsigned usable_cpus() {
    unsigned count = std::max(1U, std::thread::hardware_concurrency());
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        count = std::min(count, unsigned(std::max(1, CPU_COUNT(&set))));
    }
    double quota = -1, period = 0;
    if (std::ifstream v2("/sys/fs/cgroup/cpu.max"); v2) {
        std::string limit;
        if (v2 >> limit >> period && limit != "max") { quota = std::strtod(limit.c_str(), nullptr); }
    } else if (std::ifstream q("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), p("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
               q && p) {
        if (!(q >> quota && p >> period)) { quota = -1; }
    }
    if (quota > 0 && period > 0) { count = std::min(count, std::max(1U, unsigned(quota / period))); }
#endif
    return count;
}

// One turn of a spin wait: the processor's pause hint where it has one, a yield elsewhere.
// YieldProcessor() is the Windows SDK's portable spelling; the GCC/Clang builtin is not known to
// nvcc's front end in MSVC mode.
inline void spin_hint() {
#if defined(_WIN32)
    YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

constexpr int kTop            = 10;
constexpr int kHidden         = 2560;
constexpr int kWidth          = 640;
constexpr int kMaxExperts     = 512;
constexpr int kSplitThreads   = kMaxExperts / 2; // two expert flags a thread
constexpr std::size_t kBehindLists = 16; // patch lists of admissions behind calls, reused in turn
constexpr std::uint64_t kTail = 256;
constexpr std::uint64_t kAlign = 256;

std::uint64_t aligned(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

using Atomic = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;

// t = a + b x, fitted by least squares with exponential forgetting and the intercept drawn
// toward its prior: how long a request's copies take for the bytes they move, or its CPU products
// for the work they do (x in megabytes, t in microseconds). The prior line seeds the fit. A sample
// counts at most four times the line's value (or a quarter of it), so a copy queued behind other
// transfers or a descheduled CPU request moves the line gradually; and a negative intercept is
// refitted as zero, the line through the origin, since samples of one size (one expert while a
// side only probes) leave the slope to the prior's point.
class TimeFit {
public:
    TimeFit(double a, double b) : prior_(a), a_(a), b_(b) { add(2.0, a + 2.0 * b, 4.0); }

    void add(double x, double t, double weight = 1.0) {
        constexpr double kDecay = 0.99, kPull = 1.0, kReach = 4.0;
        const double line = at(x);
        t   = std::clamp(t, line / kReach, line * kReach);
        s0_ = kDecay * s0_ + weight;
        s1_ = kDecay * s1_ + weight * x;
        s2_ = kDecay * s2_ + weight * x * x;
        t0_ = kDecay * t0_ + weight * t;
        t1_ = kDecay * t1_ + weight * x * t;
        const double m00 = s0_ + kPull, det = m00 * s2_ - s1_ * s1_;
        if (!(det > 1e-12)) { return; }
        double a = ((t0_ + kPull * prior_) * s2_ - s1_ * t1_) / det;
        double b = (m00 * t1_ - s1_ * (t0_ + kPull * prior_)) / det;
        if (a < 0) {
            a = 0;
            b = t1_ / s2_;
        }
        if (std::isfinite(a) && std::isfinite(b) && b > 0) {
            a_ = a;
            b_ = b;
        }
    }
    [[nodiscard]] double at(double x) const { return a_ + b_ * x; }

private:
    double prior_, a_, b_;
    double s0_ = 0, s1_ = 0, s2_ = 0, t0_ = 0, t1_ = 0;
};

// One device's doorbell in pinned host memory, read and written by both sides.
struct Mailbox {
    alignas(64) std::uint32_t requested = 0; // device: the last posted request
    std::uint32_t layer = 0, tokens = 0, missing = 0;
    alignas(64) std::uint32_t prepared = 0; // host: the last request ready
};

// [ranges[0], ranges[1]) and [ranges[2], ranges[3]): a layer's cached blocks.
__device__ __forceinline__ bool in_ranges(std::uintptr_t at, const std::uintptr_t* ranges) {
    return (at >= ranges[0] && at < ranges[1]) || (at >= ranges[2] && at < ranges[3]);
}

// Splits a call's pairs: a pair whose expert's gate entry lies in the layer's cached storage stays
// in `cached` (else -1); every other expert is listed once in `missing`. With `keep_ranks` above
// zero, an uncached expert that every token of the call ranks at or past that position of its top
// ten is left out instead: its pairs add nothing and the others keep their router weights. With
// missing experts the request is posted, after the pairs (and, for the CPU, the input) reach the
// host.
__global__ void __launch_bounds__(kSplitThreads)
    split_kernel(const std::int32_t* __restrict__ ids, int pairs, int experts,
                 const void* const* __restrict__ gate, const std::uintptr_t* __restrict__ ranges,
                 std::int32_t* __restrict__ cached, std::int32_t* __restrict__ host_ids,
                 std::int32_t* __restrict__ host_missing, std::int32_t* __restrict__ missing_ids,
                 std::int32_t* product_count, const __nv_bfloat16* __restrict__ m,
                 __nv_bfloat16* __restrict__ host_m, int m_values, Mailbox* box,
                 std::uint32_t layer, int tokens, int keep_ranks, std::uint32_t* need,
                 std::uint32_t* last_use, std::uint32_t* stamps) {
    __shared__ unsigned char missing[kMaxExperts];
    __shared__ int warp_total[kSplitThreads / 32];
    __shared__ int count;
    __shared__ std::uint32_t stamp;
    for (int e = threadIdx.x; e < kMaxExperts; e += blockDim.x) { missing[e] = 0; }
    if (threadIdx.x == 0 && last_use != nullptr) { stamp = ++*stamps; }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        bool resident = false;
        if (e >= 0 && e < experts) {
            resident = in_ranges(reinterpret_cast<std::uintptr_t>(gate[e]), ranges);
            // Pairs come in rank order, ten a token (the route's top ten, largest weight first).
            if (!resident && (keep_ranks == 0 || p % kTop < keep_ranks)) { missing[e] = 1; }
            // The recency the slot admission reads (a host copy follows each pass).
            if (last_use != nullptr) { last_use[e] = stamp; }
        }
        cached[p]      = resident ? e : -1;
        missing_ids[p] = -1; // the service rewrites the missing pairs' entries of a request
    }
    __syncthreads();
    // The missing experts in ascending order: thread t holds experts 2t and 2t + 1, a block scan
    // places them.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool first = missing[2 * threadIdx.x] != 0, second = missing[2 * threadIdx.x + 1] != 0;
    const int own    = int(first) + int(second);
    int scan         = own;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, scan, offset);
        if (lane >= offset) { scan += other; }
    }
    if (lane == 31) { warp_total[warp] = scan; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < kSplitThreads / 32; ++w) {
            const int here = warp_total[w];
            warp_total[w]  = total;
            total += here;
        }
        count = total;
    }
    __syncthreads();
    if (count == 0) {
        // No request: stage two adds nothing.
        if (threadIdx.x == 0) {
            *product_count = 0;
            *need          = 0;
        }
        return;
    }
    const int at = warp_total[warp] + scan - own;
    if (first) { host_missing[at] = 2 * threadIdx.x; }
    if (second) { host_missing[at + int(first)] = 2 * threadIdx.x + 1; }
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) { host_ids[p] = ids[p]; }
    if (host_m != nullptr) {
        for (int i = threadIdx.x; i < m_values; i += blockDim.x) { host_m[i] = m[i]; }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const std::uint32_t sequence = Atomic(box->requested).load(cuda::memory_order_relaxed) + 1;
        box->layer   = layer;
        box->tokens  = std::uint32_t(tokens);
        box->missing = std::uint32_t(count);
        *need        = sequence;
        Atomic(box->requested).store(sequence, cuda::memory_order_release);
    }
}

__global__ void wait_kernel(Mailbox* box, const std::uint32_t* need) {
    const std::uint32_t sequence = *need;
    if (sequence == 0) { return; }
    while (Atomic(box->prepared).load(cuda::memory_order_acquire) != sequence) { __nanosleep(128); }
}

// One thread per token: its pairs whose expert is cached keep their id, with the token's weights
// renormalized to its whole routed sum; the others become -1 with weight 0.
__global__ void drop_missing_kernel(const std::int32_t* __restrict__ ids,
                                    const float* __restrict__ weights, int tokens, int experts,
                                    const void* const* __restrict__ gate,
                                    const std::uintptr_t* __restrict__ ranges,
                                    std::int32_t* __restrict__ kept_ids,
                                    float* __restrict__ kept_weights) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) { return; }
    bool resident[kTop];
    float all = 0.0f, kept = 0.0f;
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
        const int e = ids[t * kTop + j];
        resident[j] = false;
        if (e >= 0 && e < experts) {
            resident[j] = in_ranges(reinterpret_cast<std::uintptr_t>(gate[e]), ranges);
        }
        all += weights[t * kTop + j];
        kept += resident[j] ? weights[t * kTop + j] : 0.0f;
    }
    const float scale = kept > 0.0f ? all / kept : 0.0f;
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
        kept_ids[t * kTop + j]     = resident[j] ? ids[t * kTop + j] : -1;
        kept_weights[t * kTop + j] = resident[j] ? weights[t * kTop + j] * scale : 0.0f;
    }
}

// A device table entry change of an admission: `expert` now reads `at`.
struct TablePatch {
    std::int32_t expert = -1;
    const void* at[3]   = {};
};

__global__ void patch_kernel(const void** gate, const void** up, const void** down,
                             const TablePatch* patches, int count) {
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const TablePatch patch = patches[i];
        gate[patch.expert]     = patch.at[0];
        up[patch.expert]       = patch.at[1];
        down[patch.expert]     = patch.at[2];
    }
}

bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

class File {
public:
    explicit File(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("expert misses: cannot open " + path.string());
        }
#else
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0) {
            throw std::runtime_error("expert misses: cannot open " + path.string() + ": " +
                                     std::strerror(errno));
        }
#endif
    }
    ~File() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) { CloseHandle(handle_); }
#else
        if (descriptor_ >= 0) { ::close(descriptor_); }
#endif
    }
    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    void read(std::uint64_t offset, std::byte* out, std::size_t bytes) const {
        while (bytes > 0) {
#ifdef _WIN32
            OVERLAPPED at{};
            at.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            at.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            DWORD done    = 0;
            const DWORD request =
                static_cast<DWORD>(std::min<std::size_t>(bytes, std::size_t(1) << 30));
            if (!ReadFile(handle_, out, request, &done, &at) || done == 0) {
                throw FileReadError("expert misses: read failed");
            }
#else
            const ssize_t done = ::pread(descriptor_, out, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) { continue; }
            if (done <= 0) {
                throw FileReadError(std::string("expert misses: read failed: ") +
                                    (done < 0 ? std::strerror(errno) : "end of file"));
            }
#endif
            out += done;
            offset += static_cast<std::uint64_t>(done);
            bytes -= static_cast<std::size_t>(done);
        }
    }

private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};

// Workers that spin while a batch is likely (between spin(true) and spin(false)) and sleep
// otherwise. run() executes job(i) for every i < count on the workers and the calling thread and
// returns when all are done, rethrowing the first failure. A claim is one CAS on a ticket holding
// the run's generation and next index, so a worker still leaving an earlier run can never claim an
// index of a later one.
class SpinPool {
public:
    explicit SpinPool(std::size_t threads) {
        for (std::size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] { work(); });
        }
    }
    ~SpinPool() {
        {
            std::lock_guard lock(mutex_);
            stop_.store(true);
        }
        wake_.notify_all();
        for (auto& worker : workers_) { worker.join(); }
    }
    SpinPool(const SpinPool&)            = delete;
    SpinPool& operator=(const SpinPool&) = delete;

    void spin(bool on) {
        {
            std::lock_guard lock(mutex_);
            spinning_.store(on);
        }
        wake_.notify_all();
    }

    template <class Job>
    void run(std::size_t count, Job&& job) {
        if (count == 0) { return; }
        context_ = &job;
        call_    = [](void* context, std::size_t i) { (*static_cast<std::remove_reference_t<Job>*>(context))(i); };
        const std::uint64_t generation = (ticket_.load(std::memory_order_relaxed) >> 32) + 1;
        done_.store(0, std::memory_order_relaxed);
        // A worker claims only when the ticket's generation matches this one.
        size_.store(generation << 32 | count, std::memory_order_release);
        {
            std::lock_guard lock(mutex_);
            ticket_.store(generation << 32, std::memory_order_release);
        }
        wake_.notify_all();
        drain();
        while (done_.load(std::memory_order_acquire) != count) { spin_hint(); }
        if (failure_) {
            auto failure = std::exchange(failure_, nullptr);
            std::rethrow_exception(failure);
        }
    }

private:
    void drain() {
        std::uint64_t ticket = ticket_.load(std::memory_order_acquire);
        for (;;) {
            const std::uint64_t size = size_.load(std::memory_order_acquire);
            const std::size_t i      = std::size_t(ticket & 0xffffffffULL);
            if ((size >> 32) != (ticket >> 32) || i >= (size & 0xffffffffULL)) { return; }
            if (!ticket_.compare_exchange_weak(ticket, ticket + 1, std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
                continue;
            }
            try {
                call_(context_, i);
            } catch (...) {
                std::lock_guard lock(failure_mutex_);
                if (!failure_) { failure_ = std::current_exception(); }
            }
            done_.fetch_add(1, std::memory_order_release);
            ticket = ticket_.load(std::memory_order_acquire);
        }
    }
    void work() {
        std::uint64_t seen = 0;
        for (;;) {
            if (stop_.load()) { return; }
            const std::uint64_t generation = ticket_.load(std::memory_order_acquire) >> 32;
            if (generation != seen) {
                seen = generation;
                drain();
                continue;
            }
            if (spinning_.load(std::memory_order_relaxed)) {
                spin_hint();
                continue;
            }
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] {
                return stop_.load() || spinning_.load() || (ticket_.load() >> 32) != seen;
            });
        }
    }

    std::vector<std::thread> workers_;
    void (*call_)(void*, std::size_t) = nullptr;
    void* context_                    = nullptr;
    std::atomic<std::uint64_t> ticket_{0}; // generation << 32 | next index
    std::atomic<std::uint64_t> size_{0};   // generation << 32 | count
    std::atomic<std::size_t> done_{0};
    std::atomic<bool> spinning_{false}, stop_{false};
    std::mutex mutex_, failure_mutex_;
    std::condition_variable wake_;
    std::exception_ptr failure_;
};

} // namespace

struct ExpertMisses::Impl {
    struct Rank {
        bool used = false;
        std::unique_ptr<PinnedHostBuffer> box_memory;
        Mailbox* box = nullptr;
        DeviceBuffer cached_ids, missing_device, staging;
        std::unique_ptr<PinnedHostBuffer> need; // the call's request, 0 without one
        // Host-written stage-two inputs, read by the GPU across the bus.
        std::unique_ptr<PinnedHostBuffer> tables;   // [3][512] expert pointers
        std::unique_ptr<PinnedHostBuffer> missing_ids, products, product_pairs, product_count;
        std::unique_ptr<PinnedHostBuffer> host_ids, host_missing, host_m;
        std::unique_ptr<PinnedHostBuffer> file_staging; // disk experts, two halves
        std::unique_ptr<PinnedHostBuffer> patches;      // one request's table changes
        // The CPU experts' admissions behind a call: their patch lists, and the event the next
        // pass waits for.
        std::unique_ptr<PinnedHostBuffer> behind_patches; // kBehindLists lists
        std::array<cudaEvent_t, kBehindLists> behind{};
        std::array<bool, kBehindLists> behind_pending{};
        std::uint32_t behind_next = 0;
        std::atomic<cudaEvent_t> behind_last{nullptr}; // the latest list's event
        DeviceBuffer stamps;                            // the calls' recency counter
        cudaStream_t copies    = nullptr;
        cudaEvent_t copying    = nullptr; // balance only: before a request's copies
        cudaEvent_t copied     = nullptr; // after them (timed for the balance)
        std::uint32_t served   = 0;
        // Balance: this device's copy times, and requests since one copied or computed.
        TimeFit copy_time{10.0, 50.0}; // about 20 GB/s
        std::uint32_t without_copies = 0, without_cpu = 0;
    };

    DeviceContext& device;
    std::vector<MissLayer> layers;
    ExpertCache* cache = nullptr;
    // Each expert's last call stamp of the admitting layers, by rank: on the device, and the host
    // copy of it as of the last pass; a layer's 512 stamps from stamp_offset on.
    std::vector<DeviceBuffer> stamp_device;
    std::vector<std::unique_ptr<PinnedHostBuffer>> stamp_host;
    std::vector<std::size_t> stamp_offset;

    std::span<const std::uint32_t> last_use_host(std::size_t index) const {
        const MissLayer& layer = layers.at(index);
        return {static_cast<const std::uint32_t*>(stamp_host.at(layer.rank)->data()) +
                    stamp_offset.at(index),
                std::size_t(kMaxExperts)};
    }

    std::uint32_t* last_use_device(std::size_t index) {
        const MissLayer& layer = layers.at(index);
        return static_cast<std::uint32_t*>(stamp_device.at(layer.rank).p) + stamp_offset.at(index);
    }

    void publish(cudaStream_t stream, std::size_t rank) {
        if (rank >= stamp_host.size() || !stamp_host[rank]) { return; }
        CUDA_CHECK(cudaMemcpyAsync(stamp_host[rank]->data(), stamp_device[rank].p,
                                   stamp_host[rank]->size(), cudaMemcpyDeviceToHost, stream));
    }
    // Per layer: its cached blocks as four device words, read by the kernels (captured graphs
    // see a change).
    std::vector<DeviceBuffer> ranges;
    std::vector<std::unique_ptr<File>> files;
    MissOptions options;
    std::vector<Rank> ranks;
    std::uint32_t pair_capacity = 0;
    std::uint64_t half_bytes = 0, file_half_bytes = 0;
    std::unique_ptr<SpinPool> pool;
    ops::GgufCpuBackend backend = ops::GgufCpuBackend::Automatic;
    std::uint32_t cpu_threads   = 0; // the pool's workers and the service thread
    bool disk = false;

    // CPU work of one request.
    struct CpuExpert {
        std::int32_t expert = -1;
        std::vector<std::int32_t> pairs, tokens; // pair index and its token, per column
        std::array<const std::uint8_t*, 3> bytes{};
        ops::GgufCpuActivation input, input_up, middle; // input_up: an up of another format
        std::vector<float> gate, up, down;
        std::vector<std::uint16_t> middle_bf16;
    };
    std::vector<CpuExpert> cpu_experts;
    TimeFit cpu_time{20.0, 100.0}; // about 10 GB/s, refitted by the first CPU requests

    mutable std::mutex stats_mutex;
    MissStats counters;
    std::atomic<std::int64_t> deadline{0};
    std::atomic<bool> sleeping{false}, stopping{false};
    std::mutex wake_mutex;
    std::condition_variable wake;
    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::thread service;

    Impl(DeviceContext& d, std::vector<MissLayer> l, std::vector<std::filesystem::path> paths,
         MissOptions o, ExpertCache* c)
        : device(d), layers(std::move(l)), cache(c), options(o), ranks(d.size()) {
        if (options.token_capacity == 0 || options.token_capacity > 64 || layers.empty() ||
            !std::isfinite(options.cpu_share) || options.cpu_share < 0 || options.cpu_share > 1 ||
            (options.cpu_balance && options.cpu_share != 0) ||
            options.tail_skip_rank >= std::uint32_t(kTop)) {
            throw std::invalid_argument("expert misses: invalid capacity, CPU share or tail skip");
        }
        pair_capacity = options.token_capacity * kTop;
        std::uint64_t largest = 0;
        for (const auto& layer : layers) {
            if (layer.rank >= ranks.size() || layer.experts < kTop || layer.experts > kMaxExperts ||
                !layer.gate_table) {
                throw std::invalid_argument("expert misses: invalid layer");
            }
            std::uint64_t bytes = 0;
            for (const auto& p : layer.projections) { bytes += aligned(std::uint64_t(p.expert_bytes)); }
            largest = std::max(largest, bytes + kTail);
            disk    = disk || !layer.projections[0].located.empty();
            ranks[layer.rank].used = true;
            if (layer.admit && (cache == nullptr || !layer.projections[0].located.empty())) {
                throw std::invalid_argument("expert misses: admission needs a cache of host experts");
            }
        }
        for (const auto& layer : layers) {
            RankBinding bind(device, layer.rank);
            ranges.emplace_back(4 * sizeof(std::uintptr_t));
            write_ranges(ranges.back(), layer.resident);
        }
        // Each admitting layer's stamps: written on the device by its calls, copied to the host
        // after each pass (publish), one block per rank.
        std::vector<std::size_t> admitting(device.size(), 0);
        for (const auto& layer : layers) {
            stamp_offset.push_back(layer.admit ? admitting[layer.rank]++ * kMaxExperts : 0);
        }
        stamp_device.resize(device.size());
        stamp_host.resize(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            if (admitting[r] == 0) { continue; }
            RankBinding bind(device, r);
            stamp_device[r] = DeviceBuffer(admitting[r] * kMaxExperts * 4);
            stamp_device[r].fill(0);
            stamp_host[r] = std::make_unique<PinnedHostBuffer>(admitting[r] * kMaxExperts * 4);
            std::memset(stamp_host[r]->data(), 0, stamp_host[r]->size());
        }
        for (const auto& path : paths) { files.push_back(std::make_unique<File>(path)); }
        half_bytes = std::max<std::uint64_t>(options.staging_bytes / 2, largest * 2);
        half_bytes = half_bytes / kAlign * kAlign;
        file_half_bytes = disk ? half_bytes : 0;
        const bool cpu  = options.cpu_share > 0 || options.cpu_balance;
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            Rank& rank = ranks[r];
            if (!rank.used) { continue; }
            RankBinding bind(device, r);
            rank.box_memory = std::make_unique<PinnedHostBuffer>(sizeof(Mailbox));
            rank.box        = ::new (rank.box_memory->data()) Mailbox();
            rank.cached_ids     = DeviceBuffer(std::size_t(pair_capacity) * 4);
            rank.missing_device = DeviceBuffer(std::size_t(pair_capacity) * 4);
            rank.staging    = DeviceBuffer(2 * half_bytes);
            rank.staging.fill(0);
            rank.need          = std::make_unique<PinnedHostBuffer>(64);
            *static_cast<std::uint32_t*>(rank.need->data()) = 0;
            rank.tables        = std::make_unique<PinnedHostBuffer>(3 * kMaxExperts * sizeof(void*));
            std::memset(rank.tables->data(), 0, rank.tables->size());
            rank.missing_ids   = std::make_unique<PinnedHostBuffer>(std::size_t(pair_capacity) * 4);
            rank.product_pairs = std::make_unique<PinnedHostBuffer>(std::size_t(pair_capacity) * 4);
            rank.product_count = std::make_unique<PinnedHostBuffer>(64);
            rank.products      = std::make_unique<PinnedHostBuffer>(
                cpu ? std::size_t(pair_capacity) * kHidden * 4 : 64);
            rank.host_ids      = std::make_unique<PinnedHostBuffer>(std::size_t(pair_capacity) * 4);
            rank.host_missing  = std::make_unique<PinnedHostBuffer>(kMaxExperts * 4);
            rank.host_m        = cpu ? std::make_unique<PinnedHostBuffer>(
                                    std::size_t(options.token_capacity) * kHidden * 2)
                                     : nullptr;
            if (disk) { rank.file_staging = std::make_unique<PinnedHostBuffer>(2 * file_half_bytes); }
            rank.patches = std::make_unique<PinnedHostBuffer>(2 * kMaxExperts * sizeof(TablePatch));
            rank.behind_patches = std::make_unique<PinnedHostBuffer>(
                kBehindLists * 2 * kMaxExperts * sizeof(TablePatch));
            rank.stamps  = DeviceBuffer(64);
            rank.stamps.fill(0);
            CUDA_CHECK(cudaStreamCreateWithFlags(&rank.copies, cudaStreamNonBlocking));
            // Timed only for the balance, which reads the copies' durations.
            const unsigned timing = options.cpu_balance ? cudaEventDefault : cudaEventDisableTiming;
            if (options.cpu_balance) { CUDA_CHECK(cudaEventCreate(&rank.copying)); }
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.copied, timing));
            for (auto& event : rank.behind) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
            // Load every kernel the misses launch now: a lazy load while a layer waits on the
            // device for this service could wait for that layer (see the executor's pass()).
            patch_kernel<<<1, 32, 0, rank.copies>>>(nullptr, nullptr, nullptr, nullptr, 0);
            wait_kernel<<<1, 1, 0, rank.copies>>>(rank.box, static_cast<const std::uint32_t*>(rank.need->data()));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(rank.copies));
        }
        if (cpu) {
            // By default the physical cores (assuming two hardware threads each) less two, for the
            // executor's thread and this service.
            const std::uint32_t threads =
                options.cpu_threads != 0 ? options.cpu_threads
                                         : std::clamp(usable_cpus() / 2, 3U, 18U) - 2U;
            pool        = std::make_unique<SpinPool>(threads > 1 ? threads - 1 : 0);
            backend     = ops::gguf_cpu_backend_selected(ops::GgufCpuBackend::Automatic);
            cpu_threads = threads;
        }
        service = std::thread([this] { serve(); });
    }

    ~Impl() {
        try {
            device.synchronize();
        } catch (...) {}
        {
            std::lock_guard lock(wake_mutex);
            stopping.store(true);
        }
        wake.notify_all();
        if (service.joinable()) { service.join(); }
        for (auto& rank : ranks) {
            if (rank.copies) { cudaStreamDestroy(rank.copies); }
            if (rank.copying) { cudaEventDestroy(rank.copying); }
            if (rank.copied) { cudaEventDestroy(rank.copied); }
            for (auto event : rank.behind) {
                if (event) { cudaEventDestroy(event); }
            }
        }
    }

    static void write_ranges(DeviceBuffer& buffer, const ExpertCache::Ranges& resident) {
        const std::uintptr_t words[4] = {resident[0].first, resident[0].second, resident[1].first,
                                         resident[1].second};
        CUDA_CHECK(cudaMemcpy(buffer.p, words, sizeof(words), cudaMemcpyHostToDevice));
    }

    void remember_failure() noexcept {
        std::lock_guard lock(failure_mutex);
        if (!failure) { failure = std::current_exception(); }
    }

    // Reads one projection of a disk expert into `host_out`.
    void fetch(const MissLayer::Projection& projection, std::int32_t expert, std::byte* host_out) {
        const ExpertLocation& location = projection.located[std::size_t(expert)];
        std::uint64_t at = 0;
        for (const auto& run : location.runs) {
            if (run.bytes == 0) { continue; }
            files.at(run.file)->read(run.offset, host_out + at, std::size_t(run.bytes));
            at += run.bytes;
        }
    }

    bool cpu_capable(const MissLayer& layer) const {
        for (const auto& p : layer.projections) {
            if (!ops::gguf_cpu_supported(p.format)) { return false; }
        }
        return pool != nullptr;
    }

    void admit_behind(Rank& rank, const MissLayer& layer, std::uint32_t layer_index,
                      const std::array<unsigned char, kMaxExperts>& in_use) {
        // A ring of patch lists: a list is rewritten once the copies stream has read it.
        const std::uint32_t list = rank.behind_next;
        if (rank.behind_pending[list]) {
            CUDA_CHECK(cudaEventSynchronize(rank.behind[list]));
            rank.behind_pending[list] = false;
        }
        auto* victims = static_cast<TablePatch*>(rank.behind_patches->data()) +
                        std::size_t(list) * 2 * kMaxExperts;
        auto* joiners  = victims + kMaxExperts;
        int leaving = 0, joining = 0;
        std::vector<std::pair<std::int32_t, ExpertCache::Admitted>> admitted;
        for (const CpuExpert& job : cpu_experts) {
            ExpertCache::Admitted slot;
            if (!cache->admit(layer_index, job.expert,
                              last_use_host(layer_index),
                              {in_use.data(), in_use.size()}, slot)) {
                continue;
            }
            if (slot.victim >= 0) {
                victims[leaving++] =
                    TablePatch{slot.victim, {slot.victim_host[0], slot.victim_host[1], slot.victim_host[2]}};
            }
            joiners[joining++] = TablePatch{job.expert, {slot.slot[0], slot.slot[1], slot.slot[2]}};
            admitted.emplace_back(job.expert, slot);
        }
        if (joining == 0) { return; }
        const auto tables_of = cache->tables(layer_index);
        const auto patch     = [&](const TablePatch* list, int count) {
            patch_kernel<<<1, 128, 0, rank.copies>>>(static_cast<const void**>(tables_of[0]),
                                                     static_cast<const void**>(tables_of[1]),
                                                     static_cast<const void**>(tables_of[2]), list,
                                                     count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (leaving > 0) { patch(victims, leaving); }
        std::uint64_t bytes = 0;
        for (const auto& [expert, slot] : admitted) {
            for (int k = 0; k < 3; ++k) {
                const auto& p = layer.projections[std::size_t(k)];
                CUDA_CHECK(cudaMemcpyAsync(slot.slot[std::size_t(k)], p.host[std::size_t(expert)],
                                           std::size_t(p.expert_bytes), cudaMemcpyHostToDevice,
                                           rank.copies));
                bytes += std::uint64_t(p.expert_bytes);
            }
        }
        patch(joiners, joining);
        CUDA_CHECK(cudaEventRecord(rank.behind[list], rank.copies));
        rank.behind_pending[list] = true;
        rank.behind_last.store(rank.behind[list], std::memory_order_release);
        rank.behind_next          = (list + 1) % kBehindLists;
        std::lock_guard lock(stats_mutex);
        counters.admitted_behind += std::uint64_t(joining);
        counters.staged_bytes += bytes;
    }

    void compute_cpu(const MissLayer& layer, Rank& rank) {
        const auto* m = static_cast<const std::uint16_t*>(rank.host_m->data());
        auto* products = static_cast<float*>(rank.products->data());
        const auto& gate_p = layer.projections[0];
        const auto& up_p   = layer.projections[1];
        const auto& down_p = layer.projections[2];
        // Inputs and outputs per expert; workers then split rows.
        std::vector<std::uint16_t> columns;
        for (auto& job : cpu_experts) {
            const auto n = static_cast<int>(job.tokens.size());
            columns.resize(std::size_t(n) * kHidden);
            for (int j = 0; j < n; ++j) {
                std::memcpy(columns.data() + std::size_t(j) * kHidden,
                            m + std::size_t(job.tokens[j]) * kHidden, kHidden * 2);
            }
            const int capacity = int(options.token_capacity);
            if (job.input.k() != kHidden) { job.input = ops::GgufCpuActivation(capacity, kHidden); }
            if (job.middle.k() != kWidth) { job.middle = ops::GgufCpuActivation(capacity, kWidth); }
            job.input.prepare_bf16(gate_p.format, columns.data(), kHidden, n);
            if (up_p.format != gate_p.format) {
                if (job.input_up.k() != kHidden) {
                    job.input_up = ops::GgufCpuActivation(capacity, kHidden);
                }
                job.input_up.prepare_bf16(up_p.format, columns.data(), kHidden, n);
            }
            job.gate.resize(std::size_t(n) * kWidth);
            job.up.resize(std::size_t(n) * kWidth);
            job.down.resize(std::size_t(n) * kHidden);
        }
        constexpr int kRowsPerTask = 64;
        const int gate_tasks = (kWidth + kRowsPerTask - 1) / kRowsPerTask;
        const std::size_t first_count = cpu_experts.size() * std::size_t(2 * gate_tasks);
        pool->run(first_count, [&](std::size_t i) {
            auto& job        = cpu_experts[i / (2 * gate_tasks)];
            const int part   = int(i % (2 * gate_tasks));
            const bool is_up = part >= gate_tasks;
            const int begin  = (part % gate_tasks) * kRowsPerTask;
            const int end    = std::min(begin + kRowsPerTask, kWidth);
            const auto& p    = is_up ? up_p : gate_p;
            const ops::GgufCpuMatrix w{p.format, job.bytes[is_up ? 1 : 0], p.row_bytes, p.rows, p.k};
            const auto& x = is_up && up_p.format != gate_p.format ? job.input_up : job.input;
            ops::gguf_cpu_rows(w, x, begin, end, (is_up ? job.up : job.gate).data(), kWidth, backend);
        });
        // The middle in BF16, as the GPU's up kernel stores it before quantizing it.
        for (auto& job : cpu_experts) {
            const auto n = static_cast<int>(job.tokens.size());
            job.middle_bf16.resize(job.gate.size());
            ops::gguf_cpu_swiglu_bf16(job.gate.data(), job.up.data(), job.gate.size(),
                                      job.middle_bf16.data(), backend);
            job.middle.prepare_bf16(down_p.format, job.middle_bf16.data(), kWidth, n);
        }
        const int down_tasks = kHidden / 128;
        pool->run(cpu_experts.size() * std::size_t(down_tasks), [&](std::size_t i) {
            auto& job       = cpu_experts[i / down_tasks];
            const int begin = int(i % down_tasks) * 128;
            const ops::GgufCpuMatrix w{down_p.format, job.bytes[2], down_p.row_bytes, down_p.rows,
                                       down_p.k};
            ops::gguf_cpu_rows(w, job.middle, begin, begin + 128, job.down.data(), kHidden, backend);
        });
        auto* pairs = static_cast<std::int32_t*>(rank.product_pairs->data());
        std::int32_t count = 0;
        for (auto& job : cpu_experts) {
            for (std::size_t j = 0; j < job.pairs.size(); ++j) {
                std::memcpy(products + std::size_t(count) * kHidden, job.down.data() + j * kHidden,
                            kHidden * 4);
                pairs[count++] = job.pairs[j];
            }
        }
        *static_cast<std::int32_t*>(rank.product_count->data()) = count;
    }

    // The CPU's work on one expert with `uses` pairs, in megabytes of its projections: the
    // decode of each stored block serves every token, a token's products add about a quarter.
    static double cpu_work(double megabytes, int uses) {
        constexpr double kTokenWork = 0.25;
        return megabytes * (1.0 + kTokenWork * double(std::max(uses, 1) - 1));
    }

    static double expert_megabytes(const MissLayer& layer) {
        double bytes = 0;
        for (const auto& p : layer.projections) { bytes += double(p.expert_bytes); }
        return bytes / 1e6;
    }

    // The CPU's count of a request's missing experts, the last of `order` (fewest pairs first to
    // the CPU): the split whose copies and CPU products end soonest by the fitted times. A side
    // left unused for kProbe requests takes one expert, so both fits follow the machine.
    std::size_t balance(Rank& rank, const MissLayer& layer,
                        const std::vector<std::pair<int, std::int32_t>>& order) {
        constexpr std::uint32_t kProbe = 8;
        const std::size_t n   = order.size();
        const double megabytes = expert_megabytes(layer);
        std::vector<double> work(n + 1, 0.0); // the CPU's work on experts [g, n)
        for (std::size_t i = n; i-- > 0;) { work[i] = work[i + 1] + cpu_work(megabytes, order[i].first); }
        std::size_t gpu = n;
        double soonest  = rank.copy_time.at(double(n) * megabytes);
        for (std::size_t g = n; g-- > 0;) {
            const double copies = g == 0 ? 0.0 : rank.copy_time.at(double(g) * megabytes);
            const double end    = std::max(copies, cpu_time.at(work[g]));
            if (end < soonest) {
                soonest = end;
                gpu     = g;
            }
        }
        if (gpu == n && rank.without_cpu >= kProbe) {
            gpu = n - 1;
        } else if (gpu == 0 && rank.without_copies >= kProbe) {
            gpu = 1;
        }
        rank.without_cpu    = gpu < n ? 0 : rank.without_cpu + 1;
        rank.without_copies = gpu > 0 ? 0 : rank.without_copies + 1;
        return n - gpu;
    }

    void serve_request(Rank& rank, std::uint32_t sequence) {
        const auto started = std::chrono::steady_clock::now();
        Mailbox& box       = *rank.box;
        const std::uint32_t layer_index = box.layer;
        const std::uint32_t tokens      = box.tokens;
        const std::uint32_t missing     = box.missing;
        const MissLayer& layer          = layers.at(layer_index);
        RankBinding bind(device, layer.rank);
        const auto* ids        = static_cast<const std::int32_t*>(rank.host_ids->data());
        const auto* list       = static_cast<const std::int32_t*>(rank.host_missing->data());
        auto* tables           = static_cast<const void**>(rank.tables->data());
        auto* missing_ids      = static_cast<std::int32_t*>(rank.missing_ids->data());
        const std::uint32_t pairs = tokens * kTop;
        // Requests of a rank come one at a time: a layer's request is posted only after the stage
        // two of the one before has run on the same stream, so both staging halves are free.
        const std::uint32_t half = sequence & 1U;
        auto* staging = static_cast<std::byte*>(rank.staging.p) + half * half_bytes;
        auto* file_staging =
            disk ? static_cast<std::byte*>(rank.file_staging->data()) + half * file_half_bytes
                 : nullptr;
        // Experts with the most pairs go to the GPU; the CPU takes its share from the rest.
        std::vector<std::pair<int, std::int32_t>> order;
        for (std::uint32_t i = 0; i < missing; ++i) {
            const std::int32_t e = list[i];
            int uses             = 0;
            for (std::uint32_t p = 0; p < pairs; ++p) { uses += ids[p] == e; }
            order.emplace_back(uses, e);
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::size_t cpu_count = 0;
        if (cpu_capable(layer)) {
            cpu_count = options.cpu_balance
                            ? balance(rank, layer, order)
                            : std::min(order.size(), std::size_t(std::lround(
                                                         double(order.size()) * options.cpu_share)));
        }
        const std::size_t gpu_count = order.size() - cpu_count;
        std::array<bool, kMaxExperts> on_gpu{};
        std::uint64_t used = 0, file_used = 0, staged_bytes = 0;
        std::uint64_t staged = 0, mapped = 0;
        bool copies = false;
        auto* patches       = static_cast<TablePatch*>(rank.patches->data());
        int patch_count     = 0;
        std::array<unsigned char, kMaxExperts> in_use{};
        for (std::uint32_t p = 0; p < pairs; ++p) {
            if (ids[p] >= 0 && ids[p] < kMaxExperts) { in_use[std::size_t(ids[p])] = 1; }
        }
        if (rank.copying && gpu_count > 0) { CUDA_CHECK(cudaEventRecord(rank.copying, rank.copies)); }
        for (std::size_t i = 0; i < gpu_count; ++i) {
            const std::int32_t e = order[i].second;
            on_gpu[std::size_t(e)] = true;
            ExpertCache::Admitted admitted;
            if (layer.admit &&
                cache->admit(layer_index, e, last_use_host(layer_index),
                             {in_use.data(), in_use.size()}, admitted)) {
                for (int k = 0; k < 3; ++k) {
                    const auto& p = layer.projections[std::size_t(k)];
                    CUDA_CHECK(cudaMemcpyAsync(admitted.slot[std::size_t(k)], p.host[std::size_t(e)],
                                               std::size_t(p.expert_bytes), cudaMemcpyHostToDevice,
                                               rank.copies));
                    tables[std::size_t(k) * kMaxExperts + std::size_t(e)] = admitted.slot[std::size_t(k)];
                    staged_bytes += std::uint64_t(p.expert_bytes);
                }
                patches[patch_count++] = TablePatch{e, {admitted.slot[0], admitted.slot[1], admitted.slot[2]}};
                if (admitted.victim >= 0) {
                    patches[patch_count++] =
                        TablePatch{admitted.victim, {admitted.victim_host[0], admitted.victim_host[1],
                                                     admitted.victim_host[2]}};
                }
                copies = true;
                ++staged;
                continue;
            }
            std::uint64_t need = kTail;
            for (const auto& p : layer.projections) { need += aligned(std::uint64_t(p.expert_bytes)); }
            if (used + need <= half_bytes) {
                std::uint64_t offset = 0;
                for (int k = 0; k < 3; ++k) {
                    const auto& p     = layer.projections[std::size_t(k)];
                    std::byte* target = staging + used + offset;
                    const void* source;
                    if (disk) {
                        if (file_used + std::uint64_t(p.expert_bytes) > file_half_bytes) {
                            throw std::runtime_error("expert misses: file staging exhausted");
                        }
                        fetch(p, e, file_staging + file_used);
                        source = file_staging + file_used;
                        file_used += aligned(std::uint64_t(p.expert_bytes));
                    } else {
                        source = p.host[std::size_t(e)];
                    }
                    CUDA_CHECK(cudaMemcpyAsync(target, source, std::size_t(p.expert_bytes),
                                               cudaMemcpyHostToDevice, rank.copies));
                    if (k == 2) {
                        CUDA_CHECK(cudaMemsetAsync(target + p.expert_bytes, 0, kTail, rank.copies));
                    }
                    tables[std::size_t(k) * kMaxExperts + std::size_t(e)] = target;
                    offset += aligned(std::uint64_t(p.expert_bytes));
                    staged_bytes += std::uint64_t(p.expert_bytes);
                }
                used += need;
                copies = true;
                ++staged;
            } else {
                // The staging half is full: the GPU reads this expert where it is.
                for (int k = 0; k < 3; ++k) {
                    const auto& p = layer.projections[std::size_t(k)];
                    const void* where;
                    if (disk) {
                        if (file_used + std::uint64_t(p.expert_bytes) > file_half_bytes) {
                            throw std::runtime_error("expert misses: file staging exhausted");
                        }
                        fetch(p, e, file_staging + file_used);
                        where = file_staging + file_used;
                        file_used += aligned(std::uint64_t(p.expert_bytes));
                    } else {
                        where = p.host[std::size_t(e)];
                    }
                    tables[std::size_t(k) * kMaxExperts + std::size_t(e)] = where;
                }
                ++mapped;
            }
        }
        if (patch_count > 0) {
            const auto tables_of = cache->tables(layer_index);
            patch_kernel<<<1, 128, 0, rank.copies>>>(static_cast<const void**>(tables_of[0]),
                                                     static_cast<const void**>(tables_of[1]),
                                                     static_cast<const void**>(tables_of[2]),
                                                     patches, patch_count);
            CUDA_CHECK(cudaGetLastError());
        }
        for (std::uint32_t p = 0; p < pairs; ++p) {
            const std::int32_t e = ids[p];
            missing_ids[p] = (e >= 0 && e < kMaxExperts && on_gpu[std::size_t(e)]) ? e : -1;
        }
        // The second stage reads the missing pairs from device memory, which the split kernel
        // filled with -1.
        if (gpu_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(rank.missing_device.p, missing_ids, std::size_t(pairs) * 4,
                                       cudaMemcpyHostToDevice, rank.copies));
            copies = true;
        }
        if (copies) { CUDA_CHECK(cudaEventRecord(rank.copied, rank.copies)); }
        // The CPU's share, while the copies run.
        std::uint64_t cpu_pairs = 0;
        const auto cpu_started  = std::chrono::steady_clock::now();
        double cpu_work_done    = 0;
        if (cpu_count > 0) {
            cpu_experts.resize(cpu_count);
            for (std::size_t i = 0; i < cpu_count; ++i) {
                CpuExpert& job = cpu_experts[i];
                job.expert     = order[gpu_count + i].second;
                job.pairs.clear();
                job.tokens.clear();
                for (std::uint32_t p = 0; p < pairs; ++p) {
                    if (ids[p] == job.expert) {
                        job.pairs.push_back(std::int32_t(p));
                        job.tokens.push_back(std::int32_t(p / kTop));
                    }
                }
                cpu_pairs += job.pairs.size();
                cpu_work_done += cpu_work(expert_megabytes(layer), int(job.pairs.size()));
                for (int k = 0; k < 3; ++k) {
                    const auto& proj = layer.projections[std::size_t(k)];
                    if (disk) {
                        if (file_used + std::uint64_t(proj.expert_bytes) > file_half_bytes) {
                            throw std::runtime_error("expert misses: file staging exhausted");
                        }
                        fetch(proj, job.expert, file_staging + file_used);
                        job.bytes[std::size_t(k)] =
                            reinterpret_cast<const std::uint8_t*>(file_staging + file_used);
                        file_used += aligned(std::uint64_t(proj.expert_bytes));
                    } else {
                        job.bytes[std::size_t(k)] =
                            static_cast<const std::uint8_t*>(proj.host[std::size_t(job.expert)]);
                    }
                }
            }
            // The CPU's experts also take slots, copied in behind this call: their victims leave
            // the tables before the copies, they join after, and the next pass waits for both.
            if (layer.admit) { admit_behind(rank, layer, layer_index, in_use); }
            compute_cpu(layer, rank);
            if (options.cpu_balance) {
                cpu_time.add(cpu_work_done,
                             1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                                 cpu_started)
                                       .count());
            }
        } else if (rank.product_count) {
            *static_cast<std::int32_t*>(rank.product_count->data()) = 0;
        }
        if (copies) { CUDA_CHECK(cudaEventSynchronize(rank.copied)); }
        if (options.cpu_balance && gpu_count > 0 && staged_bytes > 0 && mapped == 0) {
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, rank.copying, rank.copied));
            rank.copy_time.add(double(staged_bytes) / 1e6, 1e3 * double(milliseconds));
        }
        Atomic(box.prepared).store(sequence, cuda::memory_order_release);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::lock_guard lock(stats_mutex);
        ++counters.requests;
        counters.staged += staged;
        counters.mapped += mapped;
        counters.computed += cpu_count;
        counters.cpu_pairs += cpu_pairs;
        counters.staged_bytes += staged_bytes;
        counters.service_seconds += seconds;
    }

    static std::int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void serve() {
        bool awake = false;
        while (!stopping.load(std::memory_order_acquire)) {
            if (now_ns() > deadline.load(std::memory_order_acquire)) {
                if (awake && pool) { pool->spin(false); }
                awake = false;
                std::unique_lock lock(wake_mutex);
                sleeping.store(true, std::memory_order_release);
                wake.wait(lock, [this] {
                    return now_ns() <= deadline.load(std::memory_order_acquire) ||
                           stopping.load(std::memory_order_acquire);
                });
                sleeping.store(false, std::memory_order_release);
                continue;
            }
            if (!awake && pool) { pool->spin(true); }
            awake = true;
            bool worked = false;
            for (auto& rank : ranks) {
                if (!rank.used) { continue; }
                const std::uint32_t sequence =
                    Atomic(rank.box->requested).load(cuda::memory_order_acquire);
                if (sequence == rank.served) { continue; }
                try {
                    serve_request(rank, sequence);
                } catch (...) {
                    remember_failure();
                    // Release the GPU with an empty stage two; the failure surfaces at the
                    // executor's next check.
                    std::memset(rank.missing_ids->data(), 0xff, rank.missing_ids->size());
                    if (rank.product_count) {
                        *static_cast<std::int32_t*>(rank.product_count->data()) = 0;
                    }
                    Atomic(rank.box->prepared).store(sequence, cuda::memory_order_release);
                }
                rank.served = sequence;
                worked      = true;
                // Activity keeps the service awake: a pass is mid-flight.
                deadline.store(std::max(deadline.load(std::memory_order_relaxed),
                                        now_ns() + 500'000'000),
                               std::memory_order_release);
            }
            if (!worked) { spin_hint(); }
        }
    }

    void drop_missing(std::size_t index, const Tensor& ids, const Tensor& weights,
                      Tensor& kept_ids, Tensor& kept_weights, cudaStream_t stream) {
        const MissLayer& layer = layers.at(index);
        const int tokens       = ids.ne[1];
        drop_missing_kernel<<<(tokens + 127) / 128, 128, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), static_cast<const float*>(weights.data),
            tokens, int(layer.experts), layer.gate_table,
            static_cast<const std::uintptr_t*>(ranges.at(index).p),
            static_cast<std::int32_t*>(kept_ids.data), static_cast<float*>(kept_weights.data));
        CUDA_CHECK(cudaGetLastError());
    }

    Call begin(std::size_t index, const Tensor& ids, const Tensor& m, std::int32_t tokens,
               const ops::GgufMoeWeights& banks, cudaStream_t stream) {
        keep_alive();
        const MissLayer& layer = layers.at(index);
        Rank& rank             = ranks.at(layer.rank);
        if (tokens <= 0 || std::uint32_t(tokens) > options.token_capacity) {
            throw std::invalid_argument("expert misses: tokens exceed the call capacity");
        }
        const int pairs = tokens * kTop;
        split_kernel<<<1, kSplitThreads, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), pairs, int(layer.experts), layer.gate_table,
            static_cast<const std::uintptr_t*>(ranges.at(index).p),
            static_cast<std::int32_t*>(rank.cached_ids.p),
            static_cast<std::int32_t*>(rank.host_ids->data()),
            static_cast<std::int32_t*>(rank.host_missing->data()),
            static_cast<std::int32_t*>(rank.missing_device.p),
            static_cast<std::int32_t*>(rank.product_count->data()),
            static_cast<const __nv_bfloat16*>(m.data),
            rank.host_m ? static_cast<__nv_bfloat16*>(rank.host_m->data()) : nullptr,
            tokens * kHidden, rank.box, std::uint32_t(index), tokens, int(options.tail_skip_rank),
            static_cast<std::uint32_t*>(rank.need->data()),
            layer.admit ? last_use_device(index) : nullptr,
            static_cast<std::uint32_t*>(rank.stamps.p));
        CUDA_CHECK(cudaGetLastError());
        Call call;
        call.cached_ids  = Tensor(rank.cached_ids.p, DType::I32, {kTop, tokens});
        call.missing_ids = Tensor(rank.missing_device.p, DType::I32, {kTop, tokens});
        call.missing_banks = banks;
        auto* tables       = static_cast<const void* const*>(rank.tables->data());
        call.missing_banks.gate.experts = tables;
        call.missing_banks.up.experts   = tables + kMaxExperts;
        call.missing_banks.down.experts = tables + 2 * kMaxExperts;
        // Staged experts sit in device memory with zeros after each down, as the matrix kernel
        // wants; experts the GPU reads across the bus keep the call on the vector products.
        call.missing_banks.device_resident = false;
        call.products      = rank.host_m ? static_cast<const float*>(rank.products->data()) : nullptr;
        call.pairs         = static_cast<const std::int32_t*>(rank.product_pairs->data());
        call.count         = static_cast<const std::int32_t*>(rank.product_count->data());
        call.max_products  = rank.host_m ? pairs : 0;
        return call;
    }

    void await(std::size_t index, cudaStream_t stream) {
        Rank& rank = ranks.at(layers.at(index).rank);
        if (!capturing(stream)) {
            // Eager: wait on the host, so a lazily loaded kernel never waits behind a GPU spin.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::uint32_t sequence = *static_cast<volatile std::uint32_t*>(rank.need->data());
            if (sequence != 0) {
                while (Atomic(rank.box->prepared).load(cuda::memory_order_acquire) != sequence) {
                    std::this_thread::yield();
                    std::lock_guard lock(failure_mutex);
                    if (failure) { std::rethrow_exception(failure); }
                }
            }
            return;
        }
        wait_kernel<<<1, 1, 0, stream>>>(rank.box, static_cast<const std::uint32_t*>(rank.need->data()));
        CUDA_CHECK(cudaGetLastError());
    }

    void fence(cudaStream_t stream, std::size_t rank_index) {
        Rank& rank = ranks.at(rank_index);
        const cudaEvent_t last = rank.behind_last.load(std::memory_order_acquire);
        if (rank.used && last != nullptr) { CUDA_CHECK(cudaStreamWaitEvent(stream, last, 0)); }
    }

    void raise_failure() {
        std::lock_guard lock(failure_mutex);
        if (failure) { std::rethrow_exception(std::exchange(failure, nullptr)); }
    }

    void keep_alive() {
        raise_failure();
        deadline.store(now_ns() + 500'000'000, std::memory_order_release);
        if (sleeping.load(std::memory_order_acquire)) {
            std::lock_guard lock(wake_mutex);
            wake.notify_all();
        }
    }
};

ExpertMisses::ExpertMisses(DeviceContext& device, std::vector<MissLayer> layers,
                           std::vector<std::filesystem::path> files, MissOptions options,
                           ExpertCache* cache)
    : impl_(std::make_unique<Impl>(device, std::move(layers), std::move(files), options, cache)) {}
ExpertMisses::~ExpertMisses() = default;

ExpertMisses::Call ExpertMisses::begin(std::size_t layer, const Tensor& ids, const Tensor& m,
                                       std::int32_t tokens, const ops::GgufMoeWeights& banks,
                                       cudaStream_t stream) {
    return impl_->begin(layer, ids, m, tokens, banks, stream);
}
void ExpertMisses::drop_missing(std::size_t layer, const Tensor& ids, const Tensor& weights,
                                Tensor& kept_ids, Tensor& kept_weights, cudaStream_t stream) {
    impl_->drop_missing(layer, ids, weights, kept_ids, kept_weights, stream);
}
void ExpertMisses::await(std::size_t layer, cudaStream_t stream) { impl_->await(layer, stream); }
void ExpertMisses::keep_alive() { impl_->keep_alive(); }

void ExpertMisses::raise_failure() { impl_->raise_failure(); }
void ExpertMisses::fence(cudaStream_t stream, std::size_t rank) { impl_->fence(stream, rank); }
void ExpertMisses::publish(cudaStream_t stream, std::size_t rank) { impl_->publish(stream, rank); }
void ExpertMisses::set_resident(std::size_t layer, const ExpertCache::Ranges& resident) {
    auto& target = impl_->layers.at(layer);
    target.resident = resident;
    RankBinding bind(impl_->device, target.rank);
    Impl::write_ranges(impl_->ranges.at(layer), resident);
}
MissStats ExpertMisses::stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->counters;
}
bool ExpertMisses::cpu_enabled() const noexcept { return impl_->pool != nullptr; }
std::string ExpertMisses::cpu_description() const {
    if (!impl_->pool) { return {}; }
    return std::to_string(impl_->cpu_threads) + " threads, " +
           ops::gguf_cpu_backend_name(impl_->backend);
}

} // namespace ninfer::models::qwen4_exp
