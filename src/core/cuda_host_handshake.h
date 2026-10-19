#pragma once

#include "core/device.h"

#include <cstdint>
#include <cstddef>
#include <optional>

namespace ninfer {

// One in-flight exchange between a CUDA stream and a host worker. Every launch is capturable;
// eager waits block the calling host thread to allow first-use CUDA kernel loading, while a
// captured graph waits on the GPU after its kernel set has loaded.
// the worker owns its scheduling and payload buffers. The stream publishes after its D2H copies,
// waits for preparation before consuming host operands, and retires after consuming the result.
// Retirement acknowledgement prevents a later graph node from reusing the shared host buffers.
class CudaHostHandshake {
public:
    struct Request {
        std::uint32_t sequence, tag, width, flags;
    };

    explicit CudaHostHandshake(const RankContext& rank);
    ~CudaHostHandshake();
    CudaHostHandshake(const CudaHostHandshake&) = delete;
    CudaHostHandshake& operator=(const CudaHostHandshake&) = delete;

    void publish(std::uint32_t tag, std::uint32_t width, std::uint32_t flags, cudaStream_t stream);
    void wait_prepared(cudaStream_t stream);
    void wait_completed(cudaStream_t stream);
    void retire(cudaStream_t stream);

    [[nodiscard]] std::optional<Request> pending(std::uint32_t last_sequence) const noexcept;
    void prepared(std::uint32_t sequence) noexcept;
    void completed(std::uint32_t sequence) noexcept;
    [[nodiscard]] bool retired(std::uint32_t sequence) const noexcept;
    void release(std::uint32_t sequence) noexcept;
    [[nodiscard]] std::size_t pinned_bytes() const noexcept;

private:
    void* host_ = nullptr;
    void* device_ = nullptr;
};

} // namespace ninfer
