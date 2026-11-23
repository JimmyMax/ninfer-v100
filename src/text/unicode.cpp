#include "text/unicode.h"

#include <utf8proc/utf8proc.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::text::unicode_internal {
namespace {

bool is_ascii_whitespace(std::int32_t codepoint) noexcept {
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' ||
           codepoint == '\v' || codepoint == '\f';
}

// Whether NFC splits the text before this code point: a starter (combining class zero) without a
// canonical decomposition that no canonical composition takes as its second element (the Hangul
// vowel and trailing jamo compose algorithmically, outside utf8proc's table). Nothing reorders or
// composes across it, so the text on either side normalizes on its own, and a stretch of nothing
// but such code points is already in NFC.
bool nfc_boundary(std::int32_t codepoint) noexcept {
    if (codepoint < 0x80) { return true; }
    if ((codepoint >= 0x1161 && codepoint <= 0x1175) ||
        (codepoint >= 0x11a8 && codepoint <= 0x11c2)) {
        return false;
    }
    const utf8proc_property_t* property = utf8proc_get_property(codepoint);
    return property->combining_class == 0 && property->decomp_seqindex == UINT16_MAX &&
           !property->comb_issecond;
}

void append_utf8proc_nfc(std::string& out, std::string_view text) {
    utf8proc_uint8_t* mapped = nullptr;
    const utf8proc_ssize_t result =
        utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
                     static_cast<utf8proc_ssize_t>(text.size()), &mapped,
                     static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (result < 0) {
        throw std::invalid_argument(std::string("failed to normalize UTF-8 text as NFC: ") +
                                    utf8proc_errmsg(result));
    }
    out.append(reinterpret_cast<const char*>(mapped), static_cast<std::size_t>(result));
    std::free(mapped);
}

} // namespace

std::string normalize_nfc(std::string_view text) {
    // Only the stretches that hold a code point other than an NFC boundary go through utf8proc,
    // each from the boundary before it (a starter a following mark may compose with) to the next
    // one; the rest is copied. ASCII is all boundaries. Normalizing a 500K-byte paper with 36K
    // bytes of symbols in one utf8proc pass took 10.4 ms (2026-10-28).
    std::string normalized;
    std::size_t copied   = 0; // text before this is in `normalized` or, while that is empty, as is
    std::size_t boundary = 0; // the last NFC boundary at or before `offset`
    std::size_t offset   = 0;
    while (offset < text.size()) {
        if (static_cast<unsigned char>(text[offset]) < 0x80u) {
            boundary = offset++;
            continue;
        }
        const CodepointSpan codepoint = utf8_codepoint_at(text, offset, "NFC input");
        if (nfc_boundary(codepoint.value)) {
            boundary = offset;
            offset += codepoint.length;
            continue;
        }
        std::size_t end = offset + codepoint.length;
        while (end < text.size() && static_cast<unsigned char>(text[end]) >= 0x80u) {
            const CodepointSpan next = utf8_codepoint_at(text, end, "NFC input");
            if (nfc_boundary(next.value)) { break; }
            end += next.length;
        }
        if (normalized.empty()) { normalized.reserve(text.size() + 16); }
        normalized.append(text.substr(copied, boundary - copied));
        append_utf8proc_nfc(normalized, text.substr(boundary, end - boundary));
        copied = boundary = offset = end;
    }
    if (copied == 0) { return std::string(text); }
    normalized.append(text.substr(copied));
    return normalized;
}

CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset,
                                std::string_view context) {
    if (offset >= text.size()) {
        throw std::out_of_range("UTF-8 codepoint offset exceeds " + std::string(context));
    }
    utf8proc_int32_t codepoint = 0;
    const utf8proc_ssize_t length =
        utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
                         static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
    if (length < 0) {
        throw std::invalid_argument("invalid UTF-8 in " + std::string(context) + ": " +
                                    utf8proc_errmsg(length));
    }
    if (length == 0) {
        throw std::invalid_argument("truncated UTF-8 sequence in " + std::string(context));
    }
    return CodepointSpan{codepoint, offset, static_cast<std::size_t>(length)};
}

std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context) {
    std::vector<CodepointSpan> codepoints;
    for (std::size_t offset = 0; offset < text.size();) {
        const CodepointSpan codepoint = utf8_codepoint_at(text, offset, context);
        codepoints.push_back(codepoint);
        offset += codepoint.length;
    }
    return codepoints;
}

std::string codepoint_to_utf8(std::int32_t codepoint) {
    utf8proc_uint8_t buffer[4];
    const utf8proc_ssize_t length = utf8proc_encode_char(codepoint, buffer);
    if (length <= 0) { throw std::invalid_argument("invalid Unicode codepoint"); }
    return {reinterpret_cast<const char*>(buffer), static_cast<std::size_t>(length)};
}

bool is_letter(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
        return true;
    default:
        return false;
    }
}

bool is_upper_or_title_letter(std::int32_t codepoint) noexcept {
    const auto category = utf8proc_category(codepoint);
    return category == UTF8PROC_CATEGORY_LU || category == UTF8PROC_CATEGORY_LT;
}

bool is_lower_letter(std::int32_t codepoint) noexcept {
    return utf8proc_category(codepoint) == UTF8PROC_CATEGORY_LL;
}

bool is_modifier_or_other_letter(std::int32_t codepoint) noexcept {
    const auto category = utf8proc_category(codepoint);
    return category == UTF8PROC_CATEGORY_LM || category == UTF8PROC_CATEGORY_LO;
}

bool is_punctuation(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_PC:
    case UTF8PROC_CATEGORY_PD:
    case UTF8PROC_CATEGORY_PS:
    case UTF8PROC_CATEGORY_PE:
    case UTF8PROC_CATEGORY_PI:
    case UTF8PROC_CATEGORY_PF:
    case UTF8PROC_CATEGORY_PO:
        return true;
    default:
        return false;
    }
}

bool is_symbol(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_SM:
    case UTF8PROC_CATEGORY_SC:
    case UTF8PROC_CATEGORY_SK:
    case UTF8PROC_CATEGORY_SO:
        return true;
    default:
        return false;
    }
}

bool is_mark(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
        return true;
    default:
        return false;
    }
}

bool is_number(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
        return true;
    default:
        return false;
    }
}

bool is_whitespace(std::int32_t codepoint) noexcept {
    if (is_ascii_whitespace(codepoint)) { return true; }
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ZS:
    case UTF8PROC_CATEGORY_ZL:
    case UTF8PROC_CATEGORY_ZP:
        return true;
    default:
        return false;
    }
}

} // namespace ninfer::text::unicode_internal
