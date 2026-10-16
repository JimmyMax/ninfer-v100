#include "artifact/reader.h"

#include "artifact/file_io.h"
#include "artifact/formats.h"
#include "artifact/layouts.h"
#include "artifact/v2_directory.h" // 本仓库 V100 增量：v2 直读

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

namespace ninfer::artifact {

namespace {
// 本仓库 V100 增量：v2 头没有 UUID。用目录文本 + 文件长度做一个稳定的 128 位 FNV-1a 指纹
// （同一次加载内恒定，不同制品不同；v2 也没有 continuation 文件，指纹不会跨文件比较）。
[[nodiscard]] ArtifactId derive_v2_artifact_id(std::string_view text,
                                               std::uint64_t file_bytes) noexcept {
    constexpr std::uint64_t kOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kPrime  = 0x100000001b3ULL;
    constexpr std::uint64_t kMix    = 0x9e3779b97f4a7c15ULL;
    std::uint64_t low               = kOffset;
    std::uint64_t high              = kOffset ^ kMix;
    const auto absorb               = [&](unsigned byte) {
        low  = (low ^ byte) * kPrime;
        high = (high ^ byte) * kPrime + kMix;
    };
    for (const char ch : text) { absorb(static_cast<unsigned char>(ch)); }
    for (unsigned i = 0; i < 8; ++i) { absorb((file_bytes >> (8 * i)) & 0xff); }
    ArtifactId id{};
    for (unsigned i = 0; i < 8; ++i) {
        id[i]     = std::byte{static_cast<unsigned char>((low >> (8 * i)) & 0xff)};
        id[8 + i] = std::byte{static_cast<unsigned char>((high >> (8 * i)) & 0xff)};
    }
    return id;
}
} // namespace

struct Reader::Impl {
    std::filesystem::path entry;
    Directory directory;
    ArtifactId id{};
    std::uint64_t entry_payload_start = 0;
    std::uint64_t file_bytes          = 0;
    mutable std::vector<std::unique_ptr<InputFile>> files;
    mutable std::vector<std::optional<WeightGeometry>> geometries;

    explicit Impl(const std::filesystem::path& path) : entry(path) {
        auto file = std::make_unique<InputFile>(entry);
        std::array<std::byte, kHeaderBytes> header{};
        file->read_exact(0, header);
        // 本仓库 V100 增量（docs/v100/v2-direct-read-2026-10-16.md）：v2 头的 JSON 长度在同一
        // 位置，但头只有 16 字节、没有 UUID。清单在内存里归一化成 v3 形态后交给同一个
        // parse_directory，权重字节原地读取（实测同一制品的 1184 个张量偏移升级前后 1:1 相同）。
        const bool is_v2 =
            std::equal(kEntryMagicV2.begin(), kEntryMagicV2.end(), header.begin());
        if (!is_v2 && !std::equal(kEntryMagic.begin(), kEntryMagic.end(), header.begin())) {
            throw ArtifactError(entry.string() + ": expected NInfer v2 or v3 entry magic");
        }
        const std::uint64_t header_bytes = is_v2 ? kV2HeaderBytes : kHeaderBytes;
        const auto json_bytes            = read_u64_le(header.data() + 8);
        if (!json_bytes || json_bytes > file->bytes() - header_bytes ||
            json_bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError(entry.string() + ": invalid directory length");
        }
        std::string text(static_cast<std::size_t>(json_bytes), '\0');
        file->read_exact(header_bytes,
                         std::as_writable_bytes(std::span(text.data(), text.size())));
        entry_payload_start = align_up(checked_add(header_bytes, json_bytes, "metadata end"),
                                       kPayloadAlignment, "entry payload start");
        if (is_v2) {
            // v2 没有 UUID；用目录文本 + 文件长度做一个稳定的 128 位指纹（同一次加载内恒定，
            // 不同制品不同；v2 也没有 continuation 文件，所以指纹不会跨文件比较）。
            id        = derive_v2_artifact_id(text, file->bytes());
            directory = parse_directory(
                normalize_v2_directory(parse_json(text, "artifact directory"), entry_payload_start,
                                       file->bytes(), entry.filename().string()),
                entry.filename().string());
        } else {
            std::copy_n(header.begin() + 16, id.size(), id.begin());
            directory =
                parse_directory(parse_json(text, "artifact directory"), entry.filename().string());
        }
        const auto expected =
            checked_add(entry_payload_start, directory.files[0].payload_bytes, "entry length");
        if (expected != file->bytes()) {
            throw ArtifactError(entry.string() + ": entry length differs from directory");
        }
        file_bytes = checked_add(entry_payload_start, directory.payload_bytes, "file bytes");
        file_bytes = checked_add(
            file_bytes,
            checked_mul(directory.files.size() - 1, kPayloadAlignment, "continuation headers"),
            "file bytes");
        files.resize(directory.files.size());
        files[0] = std::move(file);
        geometries.resize(directory.objects.size());
    }

    InputFile& file(std::size_t index) const {
        if (index >= files.size()) { throw ArtifactError("invalid continuation index"); }
        if (!files[index]) {
            const auto& record = directory.files[index];
            auto input         = std::make_unique<InputFile>(entry.parent_path() / *record.path);
            std::array<std::byte, kHeaderBytes> header{};
            input->read_exact(0, header);
            if (!std::equal(kPartMagic.begin(), kPartMagic.end(), header.begin()) ||
                read_u64_le(header.data() + 8) != index ||
                !std::equal(id.begin(), id.end(), header.begin() + 16)) {
                throw ArtifactError(*record.path +
                                    ": continuation header does not belong to this entry");
            }
            if (input->bytes() !=
                checked_add(kPayloadAlignment, record.payload_bytes, "part length")) {
                throw ArtifactError(*record.path + ": continuation length differs from directory");
            }
            files[index] = std::move(input);
        }
        return *files[index];
    }
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const Directory& Reader::directory() const noexcept { return impl_->directory; }

const ArtifactId& Reader::artifact_id() const noexcept { return impl_->id; }

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file_bytes; }

ObjectHandle Reader::find(std::string_view id) const {
    const auto found = directory().object_index.find(id);
    if (found == directory().object_index.end()) {
        throw ArtifactError("missing object " + std::string(id));
    }
    return found->second;
}

const WeightGeometry& Reader::geometry(ObjectHandle handle) const {
    const auto& tensor = directory().tensor(handle);
    auto& result       = impl_->geometries[handle.index];
    if (!result) { result = describe_tensor(tensor); }
    return *result;
}

void Reader::validate_object(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    if (std::holds_alternative<TensorObject>(object)) {
        (void)geometry(handle);
    } else if (std::get<ResourceObject>(object).encoding != kRawBytesEncoding) {
        throw ArtifactError(object_id(object) + ": unsupported resource encoding");
    }
}

std::vector<ReadSegment> Reader::segments(std::uint64_t offset, std::uint64_t bytes) const {
    const auto& data = directory();
    if (offset > data.payload_bytes || bytes > data.payload_bytes - offset) {
        throw ArtifactError("read range exceeds logical payload");
    }
    std::vector<ReadSegment> result;
    if (!bytes) { return result; }
    const auto found     = std::upper_bound(data.files.begin(), data.files.end(), offset,
                                            [](std::uint64_t position, const FileRecord& file) {
                                            return position < file.logical_begin;
                                        });
    std::size_t index    = static_cast<std::size_t>(found - data.files.begin() - 1);
    std::uint64_t copied = 0;
    while (copied < bytes) {
        const auto& file = data.files[index];
        const auto local = offset + copied - file.logical_begin;
        const auto count = std::min(bytes - copied, file.payload_bytes - local);
        const auto start = index == 0 ? impl_->entry_payload_start : kPayloadAlignment;
        result.push_back({index, checked_add(start, local, "file offset"), copied, count});
        copied += count;
        ++index;
    }
    return result;
}

void Reader::read_into(std::uint64_t offset, std::span<std::byte> destination) const {
    for (const auto& segment : segments(offset, destination.size())) {
        impl_->file(segment.file_index)
            .read_exact(segment.file_offset,
                        destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                            static_cast<std::size_t>(segment.bytes)));
    }
}

std::vector<std::byte> Reader::read_range(std::uint64_t offset, std::uint64_t bytes) const {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host read size exceeds size_t");
    }
    (void)segments(offset, bytes);
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    read_into(offset, result);
    return result;
}

std::vector<std::byte> Reader::read_object(ObjectHandle handle) const {
    validate_object(handle);
    const auto& object = directory().object(handle);
    return read_range(object_offset(object), object_bytes(object));
}

std::size_t Reader::read_direct(std::size_t file_index, std::uint64_t file_offset,
                                std::span<std::byte> destination) const {
    return impl_->file(file_index).read_direct(file_offset, destination);
}

} // namespace ninfer::artifact
