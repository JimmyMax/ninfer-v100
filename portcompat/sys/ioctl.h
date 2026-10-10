// Windows/MSVC port shim: resolved instead of <sys/ioctl.h>, which MSVC does not ship.
// Only the TIOCGWINSZ terminal-size query NInfer's startup log performs is covered.
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
#include <stdarg.h>

#ifndef TIOCGWINSZ
#define TIOCGWINSZ 0x5413
#endif

struct winsize {
    unsigned short ws_row    = 0;
    unsigned short ws_col    = 0;
    unsigned short ws_xpixel = 0;
    unsigned short ws_ypixel = 0;
};

inline int ioctl(int fd, unsigned long request, ...) {
    if (request != TIOCGWINSZ) {
        errno = ENOTTY;
        return -1;
    }
    va_list args;
    va_start(args, request);
    auto* size = static_cast<winsize*>(va_arg(args, void*));
    va_end(args);
    if (size == nullptr) {
        errno = EINVAL;
        return -1;
    }
    const intptr_t os_handle = _get_osfhandle(fd);
    if (os_handle == -1) {
        errno = EBADF;
        return -1;
    }
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (!GetConsoleScreenBufferInfo(reinterpret_cast<HANDLE>(os_handle), &info)) {
        errno = ENOTTY;
        return -1;
    }
    size->ws_col = static_cast<unsigned short>(info.srWindow.Right - info.srWindow.Left + 1);
    size->ws_row = static_cast<unsigned short>(info.srWindow.Bottom - info.srWindow.Top + 1);
    return 0;
}
