#include "models/qwen3_5/frontend/deepseek_output.h"

#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

void skip_space(std::string_view text, std::size_t& at) {
    while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at])) != 0) { ++at; }
}

bool consume(std::string_view text, std::size_t& at, std::string_view marker) {
    if (!text.substr(at).starts_with(marker)) { return false; }
    at += marker.size();
    return true;
}

// The text from `at` to the next `close`, moving `at` past the close.
bool take_until(std::string_view text, std::size_t& at, std::string_view close,
                std::string_view& out) {
    const std::size_t end = text.find(close, at);
    if (end == std::string_view::npos) { return false; }
    out = text.substr(at, end - at);
    at  = end + close.size();
    return true;
}

// The length of the longest suffix of `text` that begins `marker` (shorter than the marker).
std::size_t partial_suffix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

const std::string kSeparatedOpen =
    std::string(kDeepSeekCallsSeparator) + std::string(kDeepSeekCallsOpen);

// One `<｜DSML｜ invoke ...>...</｜DSML｜ invoke>` at `at` as a call.
bool parse_invoke(std::string_view region, std::size_t& at, GeneratedToolCall& out,
                  ToolCallParseDiagnostics& diagnostics) {
    std::string_view name;
    if (!consume(region, at, kDeepSeekInvokeOpen) || !take_until(region, at, "\">", name) ||
        name.empty() || name.find_first_of("<\"\n") != std::string_view::npos) {
        return false;
    }
    nlohmann::ordered_json arguments = nlohmann::ordered_json::object();
    for (;;) {
        skip_space(region, at);
        if (consume(region, at, kDeepSeekInvokeClose)) { break; }
        std::string_view key, flag, value;
        if (!consume(region, at, kDeepSeekParameterOpen) || !take_until(region, at, "\"", key) ||
            key.empty() || !consume(region, at, " string=\"") ||
            !take_until(region, at, "\">", flag) || (flag != "true" && flag != "false") ||
            !take_until(region, at, kDeepSeekParameterClose, value)) {
            return false;
        }
        nlohmann::ordered_json parsed;
        if (flag == "true") {
            parsed = std::string(value);
        } else {
            parsed = nlohmann::ordered_json::parse(value, nullptr, false);
            if (parsed.is_discarded()) {
                // Not the JSON the flag promises: kept as the text it is.
                parsed = std::string(value);
                ++diagnostics.schema_mismatch_arguments;
            }
        }
        const std::string field(key);
        if (arguments.contains(field)) {
            arguments.erase(field);
            ++diagnostics.duplicate_parameters_repaired;
        }
        arguments[field] = std::move(parsed);
    }
    out = GeneratedToolCall{.name = std::string(name), .arguments_json = arguments.dump()};
    return true;
}

using Json = nlohmann::ordered_json;

Json whitespace_format() { return Json{{"type", "regex"}, {"pattern", "[ \\t\\r\\n]{0,8}"}}; }

Json literal(std::string text) { return Json{{"type", "const_string"}, {"value", std::move(text)}}; }

Json sequence(Json::array_t elements) {
    return Json{{"type", "sequence"}, {"elements", std::move(elements)}};
}

// One invoke of `tool` after its opener `<｜DSML｜ invoke name="NAME">`: its parameters, the
// invoke close and the whitespace after it.
Json::array_t invoke_after_opener(const ToolCallOutputContract::Tool& tool) {
    if (!tool.unambiguous) {
        throw std::invalid_argument("structured output requires unambiguous tool definitions");
    }
    Json::array_t elements = {whitespace_format()};
    if (tool.strict) {
        elements.push_back(Json{{"type", "json_schema"},
                                {"json_schema", Json::parse(strict_tool_parameters_schema(tool))},
                                {"style", "deepseek_v4_1_xml"}});
        elements.push_back(whitespace_format());
    } else {
        for (const auto& parameter : tool.parameters) {
            if (parameter.name.empty() || parameter.name.find_first_of("\"<\n") != std::string::npos) {
                throw std::invalid_argument("tool parameter cannot be represented in DSML");
            }
            const Json value{{"type", "any_text"},
                             {"excludes",
                              Json::array({kDeepSeekParameterClose, kDeepSeekParameterOpen,
                                           kDeepSeekInvokeClose, kDeepSeekCallsClose})}};
            Json::array_t flags;
            for (const std::string_view flag : {"true", "false"}) {
                flags.push_back(Json{{"type", "tag"},
                                     {"begin", std::string(kDeepSeekParameterOpen) + parameter.name +
                                                   "\" string=\"" + std::string(flag) + "\">"},
                                     {"content", value},
                                     {"end", kDeepSeekParameterClose}});
            }
            elements.push_back(Json{
                {"type", "optional"},
                {"content", sequence({Json{{"type", "or"}, {"elements", std::move(flags)}},
                                      whitespace_format()})}});
        }
    }
    elements.push_back(literal(std::string(kDeepSeekInvokeClose)));
    elements.push_back(whitespace_format());
    return elements;
}

// Any declared tool's invoke, from its opener on.
Json any_invoke(const ToolCallOutputContract& contract) {
    Json::array_t alternatives;
    for (const auto& tool : contract.tools) {
        Json::array_t elements = {
            literal(std::string(kDeepSeekInvokeOpen) + tool.name + "\">")};
        for (Json& element : invoke_after_opener(tool)) { elements.push_back(std::move(element)); }
        alternatives.push_back(sequence(std::move(elements)));
    }
    return Json{{"type", "or"}, {"elements", std::move(alternatives)}};
}

} // namespace

DeepSeekMarkers DeepSeekMarkers::from(const Tokenizer& tokenizer) {
    const std::vector<int> ids = tokenizer.encode("</think>");
    DeepSeekMarkers out;
    out.think_close = ids.size() == 1 ? ids.front() : -1;
    return out;
}

bool parse_deepseek_calls(std::string_view region, const ToolCallOutputContract& contract,
                          std::vector<GeneratedToolCall>& out,
                          ToolCallParseDiagnostics& diagnostics) {
    (void)contract; // DSML types its values itself
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics counted = diagnostics;
    std::size_t at                   = 0;
    for (;;) {
        skip_space(region, at);
        if (at == region.size()) { break; }
        GeneratedToolCall call;
        if (!parse_invoke(region, at, call, counted)) { return false; }
        calls.push_back(std::move(call));
    }
    if (calls.empty()) { return false; }
    diagnostics = counted;
    diagnostics.structured_call_count += static_cast<std::uint32_t>(calls.size());
    out.insert(out.end(), std::make_move_iterator(calls.begin()),
               std::make_move_iterator(calls.end()));
    return true;
}

DeepSeekToolCallDecoder::DeepSeekToolCallDecoder(
    std::shared_ptr<const ToolCallOutputContract> contract)
    : contract_(std::move(contract)) {
    if (contract_ == nullptr || contract_->forced_tool_name.empty()) { return; }
    // The generation prompt ends with the block's opener and the forced call's invoke tag, so
    // generation resumes in its parameters and the region the reader sees begins at that tag.
    region_       = std::string(kDeepSeekInvokeOpen) + contract_->forced_tool_name + "\">\n";
    seeded_bytes_ = region_.size();
    in_block_     = true;
    diagnostics_.marker_seen = true;
}

std::string DeepSeekToolCallDecoder::feed(std::string_view text) {
    std::string published;
    pending_.append(text);
    for (;;) {
        if (!in_block_) {
            const std::size_t open = pending_.find(kDeepSeekCallsOpen);
            if (open == std::string::npos) {
                const std::size_t hold = std::max(partial_suffix(pending_, kSeparatedOpen),
                                                  partial_suffix(pending_, kDeepSeekCallsOpen));
                published.append(pending_, 0, pending_.size() - hold);
                pending_.erase(0, pending_.size() - hold);
                return published;
            }
            // The separator before the opener belongs to the block.
            const bool separated =
                std::string_view(pending_).substr(0, open).ends_with(kDeepSeekCallsSeparator);
            published.append(pending_, 0, separated ? open - kDeepSeekCallsSeparator.size() : open);
            pending_.erase(0, open + kDeepSeekCallsOpen.size());
            in_block_                = true;
            separated_               = separated;
            diagnostics_.marker_seen = true;
            region_.clear();
            seeded_bytes_ = 0;
        }
        const std::size_t close = pending_.find(kDeepSeekCallsClose);
        if (close == std::string::npos) {
            const std::size_t hold = partial_suffix(pending_, kDeepSeekCallsClose);
            region_.append(pending_, 0, pending_.size() - hold);
            pending_.erase(0, pending_.size() - hold);
            return published;
        }
        region_.append(pending_, 0, close);
        pending_.erase(0, close + kDeepSeekCallsClose.size());
        in_block_ = false;
        if (!parse_deepseek_calls(region_, *contract_, calls_, diagnostics_)) {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            restore(published);
            published.append(kDeepSeekCallsClose);
        }
        region_.clear();
        seeded_bytes_ = 0;
        separated_    = false;
    }
}

void DeepSeekToolCallDecoder::restore(std::string& content) const {
    // Only what the model produced: a forced block's head came from the prompt.
    if (seeded_bytes_ == 0) {
        if (separated_) { content.append(kDeepSeekCallsSeparator); }
        content.append(kDeepSeekCallsOpen);
    }
    content.append(region_, seeded_bytes_, std::string::npos);
}

DeepSeekToolCallDecoder::Terminal DeepSeekToolCallDecoder::finish() {
    Terminal out;
    if (in_block_) {
        region_.append(pending_);
        if (parse_deepseek_calls(region_, *contract_, calls_, diagnostics_)) {
            // A turn that ends after its last invoke is a complete block missing only its closing
            // marker. Supplying that marker invents no argument byte; anything less complete does
            // not parse and stays text.
            if (seeded_bytes_ != 0) { diagnostics_.forced_call_closed = true; }
        } else {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            restore(out.content);
        }
    } else {
        out.content = pending_;
    }
    pending_.clear();
    region_.clear();
    seeded_bytes_   = 0;
    separated_      = false;
    in_block_       = false;
    out.tool_calls  = std::move(calls_);
    out.diagnostics = diagnostics_;
    return out;
}

std::string deepseek_structured_tool_call_format(const ToolCallOutputContract& contract) {
    if (contract.tools.empty()) { return {}; }
    return sequence({whitespace_format(), literal(std::string(kDeepSeekCallsOpen)),
                     whitespace_format(), Json{{"type", "plus"}, {"content", any_invoke(contract)}},
                     literal(std::string(kDeepSeekCallsClose)), whitespace_format()})
        .dump();
}

std::string deepseek_tool_call_text_format(const ToolCallOutputContract& contract) {
    // A block is one tag opened by its marker and holding one or more invokes; the text around
    // blocks is free.
    const Json invoke = any_invoke(contract);
    Json blocks{{"type", "triggered_tags"},
                {"triggers", Json::array({kDeepSeekCallsOpen})},
                {"tags", Json::array({Json{{"type", "tag"},
                                           {"begin", kDeepSeekCallsOpen},
                                           {"content", sequence({whitespace_format(),
                                                                 Json{{"type", "plus"},
                                                                      {"content", invoke}}})},
                                           {"end", kDeepSeekCallsClose}}})},
                {"at_least_one", false},
                {"stop_after_first", false}};
    if (contract.forced_tool_name.empty()) { return blocks.dump(); }
    // The prompt ends inside the forced call's block, after its invoke opener.
    const auto forced =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& t) { return t.name == contract.forced_tool_name; });
    if (forced == contract.tools.end()) {
        throw std::invalid_argument("the forced tool is not declared");
    }
    Json::array_t elements = invoke_after_opener(*forced);
    elements.push_back(Json{{"type", "star"}, {"content", invoke}});
    elements.push_back(literal(std::string(kDeepSeekCallsClose)));
    elements.push_back(std::move(blocks));
    return sequence(std::move(elements)).dump();
}

} // namespace ninfer::models::qwen3_5::frontend
