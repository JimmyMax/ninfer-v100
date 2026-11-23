#pragma once

// DeepSeek-V4.1's tool-call markup in the shared frontend: after the content, a DSML block
//
//   \n\n<｜DSML｜ calls>
//   <｜DSML｜ invoke name="NAME">
//   <｜DSML｜ parameter name="KEY" string="true|false">VALUE</｜DSML｜ parameter>
//   ...
//   </｜DSML｜ invoke>
//   ...
//   </｜DSML｜ calls>
//
// whose values are text: a string as it is (string="true"), anything else as its JSON. Its
// reasoning is the prompt's: the generation prompt opens <think> and the model closes it with
// </think>, as Qwen's does.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

class Tokenizer;
struct ToolCallOutputContract;

// The reasoning close, a single token of DeepSeek's vocabulary; -1 where the vocabulary lacks it.
// The model writes its content right after it, so the token alone closes the reasoning.
struct DeepSeekMarkers {
    std::int32_t think_close = -1; // </think>

    [[nodiscard]] static DeepSeekMarkers from(const Tokenizer& tokenizer);
};

// The content's separator before a calls block, which belongs to the block.
inline constexpr std::string_view kDeepSeekCallsSeparator = "\n\n";
inline constexpr std::string_view kDeepSeekCallsOpen      = "<｜DSML｜ calls>";
inline constexpr std::string_view kDeepSeekCallsClose     = "</｜DSML｜ calls>";
inline constexpr std::string_view kDeepSeekInvokeOpen     = "<｜DSML｜ invoke name=\"";
inline constexpr std::string_view kDeepSeekInvokeClose    = "</｜DSML｜ invoke>";
inline constexpr std::string_view kDeepSeekParameterOpen  = "<｜DSML｜ parameter name=\"";
inline constexpr std::string_view kDeepSeekParameterClose = "</｜DSML｜ parameter>";

// Pulls DSML tool calls out of the content stream: text outside calls blocks is published as it
// arrives (a possible opener's prefix is held back), each complete block becomes its structured
// calls, whose arguments the request's declared parameter types read (as Flint's text values are
// read). An unterminated or unreadable block returns to content at the end.
//
// A forced call is one the prompt already opened: it ends with the block's opener and
// `<｜DSML｜ invoke name="NAME">\n`, so generation resumes in that call's parameters and the
// decoder starts in its block. The model never emits those bytes, so they join the block but never
// the content a failed block returns.
class DeepSeekToolCallDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    explicit DeepSeekToolCallDecoder(std::shared_ptr<const ToolCallOutputContract> contract);

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    // Appends the open block as the model wrote it, for a block that does not parse.
    void restore(std::string& content) const;

    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string pending_;
    std::string region_;
    // Bytes at the head of region_ that came from the prompt rather than from the model; non-zero
    // only while the block being read is the forced call's.
    std::size_t seeded_bytes_ = 0;
    bool in_block_            = false;
    bool separated_           = false; // the open block's opener followed the separator
    std::vector<GeneratedToolCall> calls_;
    ToolCallParseDiagnostics diagnostics_;
};

// The invokes of a calls block (the text between its markers) as structured calls, appended to
// `out`; false, leaving `out` as it was, when any of them is unreadable.
[[nodiscard]] bool parse_deepseek_calls(std::string_view region,
                                        const ToolCallOutputContract& contract,
                                        std::vector<GeneratedToolCall>& out,
                                        ToolCallParseDiagnostics& diagnostics);

// The DSML forms of tool_call_text_format and structured_tool_call_format (tool_call_parser.h),
// XGrammar structural tags. The first is free text whose calls blocks follow the markup above,
// starting inside the forced call when the prompt opened one; the second is the calls block a
// structured final response may give instead. A strict tool's parameters follow its JSON Schema,
// marked string="true" exactly when a value is a string; another's are each declared parameter at
// most once, in declaration order, as text.
[[nodiscard]] std::string deepseek_tool_call_text_format(const ToolCallOutputContract& contract);
[[nodiscard]] std::string
deepseek_structured_tool_call_format(const ToolCallOutputContract& contract);

} // namespace ninfer::models::qwen3_5::frontend
