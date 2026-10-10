// Vision disabled stub: satisfies ninfer_media_decode's link interface without FFmpeg.
#include "media/decode/decode.h"

namespace ninfer::media::decode {

ImageInfo inspect_image(std::span<const std::uint8_t>, const Policy&) {
    throw Error(ErrorKind::InvalidInput,
        "image input disabled: built without FFmpeg (NINFER_ENABLE_VISION=OFF)");
    return {};
}

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw Error(ErrorKind::InvalidInput,
        "video input disabled: built without FFmpeg (NINFER_ENABLE_VISION=OFF)");
    return {};
}

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw Error(ErrorKind::InvalidInput,
        "image input disabled: built without FFmpeg (NINFER_ENABLE_VISION=OFF)");
    return {};
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw Error(ErrorKind::InvalidInput,
        "video input disabled: built without FFmpeg (NINFER_ENABLE_VISION=OFF)");
    return {};
}

} // namespace ninfer::media::decode
