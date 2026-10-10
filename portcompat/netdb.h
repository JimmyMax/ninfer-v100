#pragma once
// Windows shim: getaddrinfo/freeaddrinfo and addrinfo live in ws2tcpip.h on MSVC.
#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
