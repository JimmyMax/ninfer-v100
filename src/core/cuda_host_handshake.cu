#include "core/cuda_host_handshake.h"

#include <cuda/atomic>

#include <new>
#include <thread>

namespace ninfer {
namespace {
struct Mailbox {
    alignas(64) std::uint32_t requested = 0;
    std::uint32_t tag = 0, width = 0, flags = 0;
    alignas(64) std::uint32_t prepared = 0;
    alignas(64) std::uint32_t completed = 0;
    alignas(64) std::uint32_t retired = 0;
    alignas(64) std::uint32_t released = 0;
};

// Naturally aligned 32-bit loads/stores to mapped host memory are system-atomic even on PCIe
// devices without hostNativeAtomicSupported. Do not replace these with RMW operations.
// https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html
using Atomic = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;

bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

void wait_host(Mailbox& box, bool complete) {
    const auto sequence = Atomic(box.requested).load(cuda::memory_order_acquire);
    auto& phase = complete ? box.completed : box.prepared;
    while (Atomic(phase).load(cuda::memory_order_acquire) != sequence) {
        std::this_thread::yield();
    }
}

__global__ void publish_request(Mailbox* box, std::uint32_t tag, std::uint32_t width,
                                std::uint32_t flags) {
    const auto sequence = Atomic(box->requested).load(cuda::memory_order_relaxed) + 1;
    box->tag = tag;
    box->width = width;
    box->flags = flags;
    Atomic(box->requested).store(sequence, cuda::memory_order_release);
}

__global__ void wait_phase(Mailbox* box, bool complete) {
    const auto sequence = Atomic(box->requested).load(cuda::memory_order_relaxed);
    auto& phase = complete ? box->completed : box->prepared;
    while (Atomic(phase).load(cuda::memory_order_acquire) != sequence) { __nanosleep(256); }
}

__global__ void retire_request(Mailbox* box) {
    const auto sequence = Atomic(box->requested).load(cuda::memory_order_relaxed);
    Atomic(box->retired).store(sequence, cuda::memory_order_release);
    while (Atomic(box->released).load(cuda::memory_order_acquire) != sequence) { __nanosleep(256); }
}
} // namespace

CudaHostHandshake::CudaHostHandshake(const RankContext& rank) {
    DeviceBinding bind(rank.device);
    CUDA_CHECK(cudaHostAlloc(&host_, sizeof(Mailbox), cudaHostAllocMapped | cudaHostAllocPortable));
    try {
        ::new (host_) Mailbox();
        CUDA_CHECK(cudaHostGetDevicePointer(&device_, host_, 0));
    } catch (...) {
        cudaFreeHost(host_);
        throw;
    }
}
CudaHostHandshake::~CudaHostHandshake() { if (host_) { cudaFreeHost(host_); } }

void CudaHostHandshake::publish(std::uint32_t tag, std::uint32_t width, std::uint32_t flags,
                               cudaStream_t stream) {
    publish_request<<<1, 1, 0, stream>>>(static_cast<Mailbox*>(device_), tag, width, flags);
    CUDA_CHECK(cudaGetLastError());
}
void CudaHostHandshake::wait_prepared(cudaStream_t stream) {
    if (!capturing(stream)) {
        // A later first-use kernel can make lazy loading synchronize the CUDA context while
        // holding its driver lock. A running GPU wait would then block both that load and the
        // worker's DMA calls. Eager execution waits on the host; graph capture loads its whole
        // kernel set before any wait runs.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        wait_host(*static_cast<Mailbox*>(host_), false);
        return;
    }
    wait_phase<<<1, 1, 0, stream>>>(static_cast<Mailbox*>(device_), false);
    CUDA_CHECK(cudaGetLastError());
}
void CudaHostHandshake::wait_completed(cudaStream_t stream) {
    if (!capturing(stream)) {
        wait_host(*static_cast<Mailbox*>(host_), true);
        return;
    }
    wait_phase<<<1, 1, 0, stream>>>(static_cast<Mailbox*>(device_), true);
    CUDA_CHECK(cudaGetLastError());
}
void CudaHostHandshake::retire(cudaStream_t stream) {
    const bool captured = capturing(stream);
    retire_request<<<1, 1, 0, stream>>>(static_cast<Mailbox*>(device_));
    CUDA_CHECK(cudaGetLastError());
    // No eager GPU wait may remain when the caller first launches another layer's kernels.
    if (!captured) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
}
std::optional<CudaHostHandshake::Request> CudaHostHandshake::pending(std::uint32_t last) const noexcept {
    auto& box = *static_cast<Mailbox*>(host_);
    const auto sequence = Atomic(box.requested).load(cuda::memory_order_acquire);
    if (sequence == last) { return std::nullopt; }
    return Request{sequence, box.tag, box.width, box.flags};
}
void CudaHostHandshake::prepared(std::uint32_t sequence) noexcept {
    Atomic(static_cast<Mailbox*>(host_)->prepared).store(sequence, cuda::memory_order_release);
}
void CudaHostHandshake::completed(std::uint32_t sequence) noexcept {
    Atomic(static_cast<Mailbox*>(host_)->completed).store(sequence, cuda::memory_order_release);
}
bool CudaHostHandshake::retired(std::uint32_t sequence) const noexcept {
    return Atomic(static_cast<Mailbox*>(host_)->retired).load(cuda::memory_order_acquire) == sequence;
}
void CudaHostHandshake::release(std::uint32_t sequence) noexcept {
    Atomic(static_cast<Mailbox*>(host_)->released).store(sequence, cuda::memory_order_release);
}
std::size_t CudaHostHandshake::pinned_bytes() const noexcept { return sizeof(Mailbox); }

} // namespace ninfer
