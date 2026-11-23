#include "core/zero_pages.h"

#include <new>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/mman.h>
#endif

namespace ninfer {

void* map_zero_pages(std::size_t bytes) {
    if (bytes == 0) { return nullptr; }
#ifdef _WIN32
    void* pages = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (pages == nullptr) { throw std::bad_alloc(); }
#else
    void* pages =
        ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) { throw std::bad_alloc(); }
#endif
    return pages;
}

void advise_huge_pages(void* pages, std::size_t bytes) noexcept {
#ifdef MADV_HUGEPAGE
    // Advisory: when transparent huge pages are disabled the pages stay 4 KiB.
    if (pages != nullptr) { (void)::madvise(pages, bytes, MADV_HUGEPAGE); }
#else
    (void)pages;
    (void)bytes;
#endif
}

void unmap_zero_pages(void* pages, std::size_t bytes) noexcept {
    if (pages == nullptr) { return; }
#ifdef _WIN32
    (void)bytes;
    ::VirtualFree(pages, 0, MEM_RELEASE);
#else
    ::munmap(pages, bytes);
#endif
}

} // namespace ninfer
