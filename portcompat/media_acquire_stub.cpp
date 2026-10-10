// Remote media acquisition disabled stub: satisfies ninfer_media_acquire's link
// interface without libcurl.
#include "product/media_acquire/acquire.h"

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw Error(ErrorKind::RemoteUnavailable,
        "remote media acquisition disabled: built without libcurl (NINFER_ENABLE_VISION=OFF)");
    return {};
}

} // namespace ninfer::product::media_acquire
