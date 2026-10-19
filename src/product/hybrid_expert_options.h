#pragma once

#include "ninfer/types.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

inline constexpr const char* kHybridExpertHelp =
    "  Flash-Next-only expert options:\n"
    "  --expert-dma-share F|auto     host or disk experts: GPU share of cache misses,\n"
    "                                0..1 (default 1); below 1 lets the CPU compute the rest\n"
    "                                (GGUF experts: the GPU's results bit for bit); auto: each\n"
    "                                call splits them by measured CPU and bus times (GGUF);\n"
    "                                calls of more than 8 tokens always use the GPU\n"
    "  --expert-misses staged|mapped GGUF experts: copy misses into device staging while the\n"
    "                                cached experts run (default), or read host experts across\n"
    "                                the bus inside the expert kernels\n"
    "  --expert-tail-skip N          GGUF host experts: leave out of a decode or verification\n"
    "                                call each uncached expert that all its tokens rank past\n"
    "                                place N (1..9) of their top ten; changes results\n"
    "  --expert-prefill-buffers 1|2  GGUF host experts: device buffers for a long prompt's\n"
    "                                uncached experts (default 1); 2 copies the next layer's\n"
    "                                while this one runs, with memory from the expert cache\n"
    "  --expert-cpu-threads N        CPU workers, 1..256 (default automatic)\n"
    "  --expert-cache-adaptive       replace cold cached experts between calls;\n"
    "                                changes the CPU/GPU arithmetic partition\n"
    "  --expert-profile FILE         fill the host expert cache from recorded counts\n"
    "  --expert-profile-out FILE     record expert counts for this artifact after requests\n";

template<class Value>
bool parse_hybrid_expert_option(std::string_view option, HybridExpertOptions& out, Value&& value) {
    if (option == "--expert-dma-share") {
        const std::string text(value());
        if (text == "auto") {
            out.dma_share      = 1;
            out.dma_share_auto = true;
            return true;
        }
        float share = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), share);
        if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(share) || share < 0 || share > 1) {
            throw std::invalid_argument("--expert-dma-share takes a finite value in 0..1 or auto");
        }
        out.dma_share      = share;
        out.dma_share_auto = false;
    } else if (option == "--expert-cpu-threads") {
        const std::string text(value());
        std::uint32_t threads = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), threads);
        if (error != std::errc{} || end != text.data() + text.size() || threads == 0 || threads > 256) {
            throw std::invalid_argument("--expert-cpu-threads takes an integer in 1..256");
        }
        out.cpu_threads = threads;
    } else if (option == "--expert-tail-skip") {
        const std::string text(value());
        std::uint32_t rank      = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), rank);
        if (error != std::errc{} || end != text.data() + text.size() || rank == 0 || rank > 9) {
            throw std::invalid_argument("--expert-tail-skip takes an integer in 1..9");
        }
        out.tail_skip_rank = rank;
    } else if (option == "--expert-prefill-buffers") {
        const std::string_view count(value());
        if (count != "1" && count != "2") {
            throw std::invalid_argument("--expert-prefill-buffers takes 1 or 2");
        }
        out.prefill_buffers = count == "2" ? 2 : 1;
    } else if (option == "--expert-misses") {
        const std::string_view mode(value());
        if (mode == "staged") {
            out.mapped_misses = false;
        } else if (mode == "mapped") {
            out.mapped_misses = true;
        } else {
            throw std::invalid_argument("--expert-misses takes staged or mapped");
        }
    } else if (option == "--expert-cache-adaptive") {
        out.adaptive_cache = true;
    } else if (option == "--expert-profile" || option == "--expert-profile-out") {
        const std::string path(value());
        if (path.empty()) { throw std::invalid_argument(std::string(option) + " takes a file path"); }
        (option == "--expert-profile" ? out.routing_profile : out.record_profile) = path;
    } else { return false; }
    return true;
}

} // namespace ninfer::product
