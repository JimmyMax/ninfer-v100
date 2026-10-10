// Windows/MSVC port shim: resolved instead of <sys/mman.h>, which MSVC does not ship.
// Covers the read-only private mapping NInfer's artifact loader performs via
// MappedFile (src/artifact/reader.cpp).
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <errno.h>
#include <io.h>

#define PROT_READ 1
#define MAP_PRIVATE 2
#define MAP_FAILED ((void*)-1)

inline void* mmap(void* addr, size_t length, int prot, int flags, int fd, long long offset) {
    (void)addr;
    (void)prot;
    (void)flags;
    if (length == 0) { return MAP_FAILED; }
    const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (file == INVALID_HANDLE_VALUE) { return MAP_FAILED; }
    const HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        const DWORD err = GetLastError();
        errno = (err == ERROR_NOT_ENOUGH_MEMORY || err == ERROR_COMMITMENT_LIMIT) ? ENOMEM : EACCES;
        return MAP_FAILED;
    }
    // The view keeps the mapping object alive, so the handle can be closed here.
    void* view = MapViewOfFile(mapping,
        FILE_MAP_READ,
        static_cast<DWORD>(static_cast<unsigned long long>(offset) >> 32),
        static_cast<DWORD>(static_cast<unsigned long long>(offset)),
        length);
    CloseHandle(mapping);
    if (view == nullptr) { return MAP_FAILED; }
    return view;
}

inline int munmap(void* addr, size_t length) {
    (void)length;
    return UnmapViewOfFile(addr) ? 0 : -1;
}
