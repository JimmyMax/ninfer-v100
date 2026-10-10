// Windows/MSVC port shim: resolved instead of <unistd.h>, which MSVC does not ship.
// Only the POSIX surface NInfer actually uses is provided here. Everything is a real
// inline function (not a macro) so unrelated member calls like fstream.close() and
// stream.write() are untouched.
#pragma once

#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <time.h>

#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

// MSVC fcntl.h has no O_CLOEXEC / O_DIRECT; stripping them is correct on Windows
// (no fd inheritance to avoid, and the page cache substitutes for direct I/O).
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#ifndef _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#define _SSIZE_T_DEFINED
#endif

// Force binary mode for all CRT file opens in this process: the CRT default is
// text mode, whose CRLF translation would corrupt model bytes read via pread().
namespace {
struct NinferPortFmodeBinary {
    NinferPortFmodeBinary() { _set_fmode(_O_BINARY); }
};
const NinferPortFmodeBinary ninfer_port_fmode_binary_init;
} // namespace

// MSVC io.h/process.h already declare isatty/getpid/open/close/read/write as
// deprecated POSIX aliases (warnings silenced via _CRT_NONSTDC_NO_WARNINGS).
// std::filesystem::path::c_str() yields wchar_t on Windows, so overload open()
// for wide paths; binary mode is mandatory for artifact bytes.
inline int open(const wchar_t* path, int flags, ...) {
    (void)flags;
    return _wopen(path, _O_RDONLY | _O_BINARY | _O_SEQUENTIAL);
}

inline struct tm* localtime_r(const time_t* timep, struct tm* result) {
    if (localtime_s(result, timep) == 0) { return result; }
    return nullptr;
}

inline struct tm* gmtime_r(const time_t* timep, struct tm* result) {
    if (gmtime_s(result, timep) == 0) { return result; }
    return nullptr;
}

// POSIX pread never moves the file position; emulate by saving and restoring it.
inline ssize_t pread(int fd, void* buf, size_t count, long long offset) {
    if (count == 0) { return 0; }
    const long long saved = _telli64(fd);
    if (_lseeki64(fd, offset, SEEK_SET) < 0) { return -1; }
    const ssize_t got = _read(fd, buf, static_cast<unsigned int>(count));
    _lseeki64(fd, saved, SEEK_SET);
    return got;
}
