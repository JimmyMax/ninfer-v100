#pragma once

// Host memory whose bytes read zero until written, in pages the OS maps on first touch: a large,
// sparsely written table costs only the pages it writes, where a filled allocation faults in every
// page up front (a request's 36 MiB n-gram index took 14-30 ms that way under WSL2).

#include <cstddef>
#include <type_traits>
#include <utility>

namespace ninfer {

// `bytes` of zero pages (null for none); throws std::bad_alloc when the OS refuses them.
[[nodiscard]] void* map_zero_pages(std::size_t bytes);
void unmap_zero_pages(void* pages, std::size_t bytes) noexcept;
// Asks the OS to back the pages first touched from now on with huge pages (Linux transparent huge
// pages; no effect elsewhere or when they are disabled). A table about to be written all over pays
// one fault per 2 MiB instead of one per 4 KiB: a 3.6K-token prompt's n-gram index took 23 ms of
// 4 KiB faults under WSL2 and 1.6 ms in huge pages, but a 24-token one 0.16 ms against 1.1 ms.
void advise_huge_pages(void* pages, std::size_t bytes) noexcept;

// `count` values of T over zero pages, each the all-zero value of T until written.
template <class T>
class ZeroPageArray {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                  "a zero-page array holds values whose all-zero bytes are a value");

public:
    ZeroPageArray() = default;
    explicit ZeroPageArray(std::size_t count)
        : data_(static_cast<T*>(map_zero_pages(count * sizeof(T)))), size_(count) {}
    ~ZeroPageArray() { unmap_zero_pages(data_, size_ * sizeof(T)); }
    ZeroPageArray(ZeroPageArray&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    ZeroPageArray& operator=(ZeroPageArray&& other) noexcept {
        if (this != &other) {
            unmap_zero_pages(data_, size_ * sizeof(T));
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    ZeroPageArray(const ZeroPageArray&)            = delete;
    ZeroPageArray& operator=(const ZeroPageArray&) = delete;

    void advise_huge_pages() noexcept { ninfer::advise_huge_pages(data_, size_ * sizeof(T)); }

    [[nodiscard]] T& operator[](std::size_t i) { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const { return data_[i]; }
    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }

private:
    T* data_          = nullptr;
    std::size_t size_ = 0;
};

} // namespace ninfer
