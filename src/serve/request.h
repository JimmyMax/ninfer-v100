#pragma once

#include "product/media_acquire/source.h"

#include <ninfer/types.h>

// Internal, wire-format-independent representation of a generation request.
//
// OpenAI and Anthropic schemas map into this wire-independent value.
// translate.cpp then produces the public PromptInput and RequestOptions consumed
// by Engine; media sources remain unresolved until the product service acquires
// owning bytes.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens = 8192;
    int max_context        = 8192; // --max-context, the upper bound of a "no limit" budget
    // --lenient-image-detail: an image detail outside auto, low and high is read as auto instead
    // of failing with image_detail_not_supported.
    bool lenient_image_detail = false;
};

enum class ContentKind {
    Text,
    Image,
    Video,
};

struct CacheBoundary {
    enum class Ttl : std::uint8_t {
        Default,
        FiveMinutes,
        OneHour,
    };

    ninfer::PromptCacheMarkerKind kind       = ninfer::PromptCacheMarkerKind::SharedStablePrefix;
    ninfer::SharedCandidateEvidence evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary;
    Ttl ttl                                  = Ttl::Default;

    [[nodiscard]] friend constexpr bool operator==(CacheBoundary, CacheBoundary) noexcept = default;
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original wire "type" string for diagnostics
    ninfer::product::media_acquire::Source source;
    ninfer::ImageResizePolicy image_resize_policy = ninfer::ImageResizePolicy::Downsize;
    ninfer::ImageDetail image_detail              = ninfer::ImageDetail::Auto;
    // A detail value outside auto, low and high, read as auto until settle_image_details accepts
    // or refuses it.
    std::string unknown_image_detail;
    std::optional<CacheBoundary> cache_boundary_after;
};

// OpenAI's image detail values; empty for any other (see settle_image_details).
[[nodiscard]] inline std::optional<ninfer::ImageDetail> parse_image_detail(std::string_view value) {
    if (value == "auto") { return ninfer::ImageDetail::Auto; }
    if (value == "low") { return ninfer::ImageDetail::Low; }
    if (value == "high") { return ninfer::ImageDetail::High; }
    return std::nullopt;
}

[[nodiscard]] inline const char* image_detail_name(ninfer::ImageDetail detail) noexcept {
    switch (detail) {
    case ninfer::ImageDetail::Auto:
        return "auto";
    case ninfer::ImageDetail::Low:
        return "low";
    case ninfer::ImageDetail::High:
        return "high";
    }
    return "auto";
}

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string input_schema_json;
    std::optional<std::string> input_examples_json;
    std::optional<CacheBoundary> cache_boundary_after;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ToolChoiceMode {
    Auto,
    None,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
};

struct ChatTurn {
    ChatRole role = ChatRole::User;
    std::vector<ContentPart> content; // ordered parts; may be empty when wire content is empty
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id; // populated for role=tool
    // Optional protocol assertion for a tool result. Call-graph normalization verifies it against
    // the function identified by tool_call_id before the Engine sees the history.
    std::optional<std::string> tool_result_name;
    bool tool_result_is_error = false;
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
    std::optional<CacheBoundary> cache_boundary_after;
};

// Sampling overrides that have an executable Engine meaning. Protocol-only
// fields are normalized or rejected before this value is constructed.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<double> min_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// The Engine's sampler resolves at most this many candidates. A larger request is clamped rather
// than refused: llama.cpp and Ollama default to 40 (the released WebUI's Top K field accepts the
// same range), and with top_p or min_p active over the full vocabulary the nucleus almost always
// closes inside 20 candidates, so the wider value selects the same token nearly always. The
// clamped value is what reaches the Engine and the request log. Negative values stay errors.
inline constexpr int kSamplerTopKCap = 20;

[[nodiscard]] constexpr int clamp_request_top_k(int top_k) noexcept {
    return top_k > kSamplerTopKCap ? kSamplerTopKCap : top_k;
}

// Protocol-level effort vocabulary. Each wire adapter accepts the values from
// its external contract; translation then resolves them against the capabilities
// advertised by the chat template embedded in the loaded artifact.
enum class RequestedReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::optional<RequestedReasoningEffort>
parse_requested_reasoning_effort(std::string_view value) noexcept {
    if (value == "none") { return RequestedReasoningEffort::None; }
    if (value == "minimal") { return RequestedReasoningEffort::Minimal; }
    if (value == "low") { return RequestedReasoningEffort::Low; }
    if (value == "medium") { return RequestedReasoningEffort::Medium; }
    if (value == "high") { return RequestedReasoningEffort::High; }
    if (value == "xhigh") { return RequestedReasoningEffort::XHigh; }
    if (value == "max") { return RequestedReasoningEffort::Max; }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view
requested_reasoning_effort_name(RequestedReasoningEffort effort) noexcept {
    switch (effort) {
    case RequestedReasoningEffort::None:
        return "none";
    case RequestedReasoningEffort::Minimal:
        return "minimal";
    case RequestedReasoningEffort::Low:
        return "low";
    case RequestedReasoningEffort::Medium:
        return "medium";
    case RequestedReasoningEffort::High:
        return "high";
    case RequestedReasoningEffort::XHigh:
        return "xhigh";
    case RequestedReasoningEffort::Max:
        return "max";
    }
    return {};
}

struct GenerationRequest {
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = 64;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    bool stop_strings_apply_to_reasoning = false;
    int max_tokens                       = 0; // resolved budget; zero means immediate output limit
    std::optional<bool> enable_thinking;      // unset => use the server default
    std::optional<std::uint32_t> thinking_budget;
    std::optional<RequestedReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    ninfer::PromptContinuationMode continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    bool allow_engine_automatic_shared_prefixes = true;
    SamplingParams sampling;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None;
    }

    [[nodiscard]] std::size_t media_item_count() const noexcept {
        std::size_t count = 0;
        for (const ChatTurn& message : messages) {
            for (const ContentPart& part : message.content) {
                if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
            }
        }
        return count;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == ChatRole::Tool) { return true; }
        }
        return false;
    }
};

} // namespace ninfer::serve
