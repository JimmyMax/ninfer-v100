#pragma once

#include <string_view>

namespace ninfer::models {

// Qwen4Exp (Qwen3.8-Flash-Next) and Gemma 4 are their own families (models/qwen4_exp,
// models/gemma4); the Qwen3.5 config parser never resolves them.
enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp, Gemma4, Inkling2, Flint, DeepSeekV41 };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
