#include "artifact/v2_directory.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

// 本仓库 V100 增量（docs/v100/v2-direct-read-2026-10-16.md）：v2 清单 → v3 目录。
//
// 这是 tools/upgrade_ninfer_v2_to_v3.py 中 make_directory 的逐条移植（那个工具是规格与等价性
// oracle）。差别只有两点：这里只构造内存里的 JSON（权重字节原地保留，偏移不变），以及
// provenance 记为 read_as_v2 而不是 upgraded_from。
namespace ninfer::artifact {
namespace {

using Counts = std::vector<std::size_t>;

struct KnownInput {
    std::string_view model_id;
    std::string_view weights_id;
    Counts counts;
};

// 官方 v2 输入的白名单（与工具的 KNOWN_COUNTS 相同）。白名单之外的 v2 制品仍旧走
// "请用离线升级工具"的报错，避免对未知布局做猜测。
const std::array<KnownInput, 5> kKnownInputs{{
    {"qwen3.6-27b", "groupwise-int", {1124}},
    {"qwen3.6-27b", "nvfp4", {1307}},
    {"qwen3.8-27b", "groupwise-int", {1124, 1190}},
    {"qwen3.8-27b", "nvfp4", {1124, 1190}},
    {"qwen3.6-35b-a3b", "groupwise-int", {940}},
}};

const std::map<std::string, std::string, std::less<>> kFormats{
    {"BF16", "bf16"},
    {"FP32", "fp32"},
    {"I32", "int32"},
    {"Q4G64_F16S", "q4_g64_fp16"},
    {"Q5G64_F16S", "q5_g64_fp16"},
    {"Q6G64_F16S", "q6_g64_fp16"},
    {"W8G32_F16S", "q8_g32_fp16"},
    {"NVFP4", "nvfp4"},
    {"FP8_E4M3FN_ROW_BF16S", "fp8_e4m3fn_row_bf16"},
};

const std::map<std::string, std::string, std::less<>> kLayouts{
    {"contiguous-le-v1", "contiguous_le_v1"},
    {"row-split-k128-v1", "row_split_k128_v1"},
    {"blockscale-k16-m128x4-v1", "block_scale_k16_m128x4_v1"},
    {"row-scale-v1", "row_scale_v1"},
};

[[nodiscard]] double f32(double value) { return static_cast<double>(static_cast<float>(value)); }

[[nodiscard]] std::uint64_t align_to(std::uint64_t value, std::uint64_t amount) {
    return (value + amount - 1) / amount * amount;
}

[[nodiscard]] std::uint64_t product(const Shape& shape) {
    std::uint64_t total = 1;
    for (const auto value : shape) { total *= value; }
    return total;
}

[[nodiscard]] Json shape_json(std::initializer_list<std::uint64_t> values) {
    return Json(Shape(values.begin(), values.end()));
}

[[nodiscard]] bool has_prefix(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

[[nodiscard]] bool has_suffix(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// 与工具的 text_config 相同（未被用到的视觉/草稿分支在这里一起保留，便于对照）。
[[nodiscard]] Json text_config(bool moe) {
    const std::int32_t layers = moe ? 40 : 64;
    Json layer_types           = Json::array();
    for (std::int32_t i = 0; i < layers; ++i) {
        layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
    }
    Json result{
        {"architectures", Json::array({moe ? "Qwen3_5MoeForCausalLM" : "Qwen3_5ForCausalLM"})},
        {"model_type", moe ? "qwen3_5_moe_text" : "qwen3_5_text"},
        {"hidden_size", moe ? 2048 : 5120},
        {"vocab_size", 248320},
        {"tie_word_embeddings", false},
        {"num_hidden_layers", layers},
        {"layer_types", std::move(layer_types)},
        {"max_position_embeddings", 262144},
        {"rms_norm_eps", f32(1e-6)},
        {"num_attention_heads", moe ? 16 : 24},
        {"num_key_value_heads", moe ? 2 : 4},
        {"head_dim", 256},
        {"rope_parameters",
         {{"rope_theta", f32(10000000)},
          {"partial_rotary_factor", f32(0.25)},
          {"mrope_section", Json::array({11, 11, 10})}}},
        {"linear_num_key_heads", 16},
        {"linear_key_head_dim", 128},
        {"linear_num_value_heads", moe ? 32 : 48},
        {"linear_value_head_dim", 128},
        {"linear_conv_kernel_dim", 4},
    };
    if (moe) {
        result["num_experts"]                   = 256;
        result["num_experts_per_tok"]           = 8;
        result["moe_intermediate_size"]         = 512;
        result["shared_expert_intermediate_size"] = 512;
    } else {
        result["intermediate_size"] = 17408;
    }
    return result;
}

[[nodiscard]] Json vision_config(bool moe) {
    return Json{
        {"model_type", moe ? "qwen3_5_moe_vision" : "qwen3_5_vision"},
        {"depth", 27},
        {"hidden_size", 1152},
        {"intermediate_size", 4304},
        {"num_heads", 16},
        {"patch_size", 16},
        {"temporal_patch_size", 2},
        {"spatial_merge_size", 2},
        {"num_position_embeddings", 2304},
    };
}

[[nodiscard]] Json draft_config(bool second) {
    Json dflash{{"target_layer_ids", second ? Json::array({5, 19, 33, 47, 61})
                                            : Json::array({1, 6, 11, 16, 22, 27, 32, 37})},
                {"mask_token_id", second ? 248070 : 248077}};
    if (second) {
        dflash["conv_kernel_size"] = 2;
        dflash["conv_group_size"]  = 16;
        dflash["selector_rank"]    = 256;
        dflash["selector_top_k"]   = 16;
    }
    Json layer_types = Json::array();
    for (int i = 0; i < 5; ++i) { layer_types.push_back("sliding_attention"); }
    if (!second) { layer_types.push_back("full_attention"); }
    return Json{
        {"architectures", Json::array({second ? "DFlash2DraftModel" : "DFlashDraftModel"})},
        {"model_type", "qwen3"},
        {"intermediate_size", second ? 17408 : 6144},
        {"num_attention_heads", 32},
        {"num_key_value_heads", 8},
        {"head_dim", 128},
        {"num_hidden_layers", second ? 5 : 6},
        {"rms_norm_eps", f32(1e-6)},
        {"rope_parameters", {{"rope_theta", f32(10000000)}}},
        {"max_position_embeddings", 262144},
        {"layer_types", std::move(layer_types)},
        {"sliding_window", second ? 2048 : 4096},
        {"dflash_config", std::move(dflash)},
    };
}

// 与工具的 encoded_bytes 相同：既是长度校验，也是"这份清单我们认得"的判据。
[[nodiscard]] std::uint64_t encoded_bytes(const Json& object) {
    const Shape dims = object.at("shape").get<Shape>();
    const auto format = object.at("format").get<std::string>();
    const auto layout = object.at("layout").get<std::string>();
    for (const auto dim : dims) {
        if (dim == 0) {
            throw ArtifactError(object.at("id").get<std::string>() + ": invalid shape");
        }
    }
    if (layout == "contiguous_le_v1" &&
        (format == "bf16" || format == "fp32" || format == "int32")) {
        return product(dims) * (format == "bf16" ? 2U : 4U);
    }
    if (dims.size() != 2) {
        throw ArtifactError(object.at("id").get<std::string>() + ": expected matrix encoding");
    }
    const std::uint64_t n = dims[0];
    const std::uint64_t k = dims[1];
    const bool groupwise  = format == "q4_g64_fp16" || format == "q5_g64_fp16" ||
                           format == "q6_g64_fp16" || format == "q8_g32_fp16";
    if (layout == "row_split_k128_v1" && groupwise) {
        const unsigned bits   = static_cast<unsigned>(format[1] - '0');
        const std::uint64_t g = bits == 8 ? 32 : 64;
        const std::uint64_t groups = align_to(k, 128) / g;
        const std::uint64_t low =
            n * groups * (bits == 8 ? g : g / 2);
        const std::uint64_t high =
            n * groups * (bits == 4 || bits == 8 ? 0 : g * (bits - 4) / 8);
        return align_to(low, 256) + align_to(high, 256) + n * groups * 2;
    }
    if (layout == "block_scale_k16_m128x4_v1" && format == "nvfp4" && n % 128 == 0 &&
        k % 64 == 0) {
        return align_to(n * k / 2, 256) + n * k / 16 + 4;
    }
    if (layout == "row_scale_v1" && format == "fp8_e4m3fn_row_bf16") {
        return align_to(n * k, 256) + n * 2;
    }
    throw ArtifactError(object.at("id").get<std::string>() + ": unknown format/layout combination");
}

// 与工具的 _input_names 相同：参数名 → 它读哪块激活。
[[nodiscard]] std::vector<std::string> input_names(const std::string& name,
                                                   const Json& components) {
    std::vector<std::string> backends;
    for (const char* candidate : {"mtp", "dflash", "dflash2"}) {
        if (components.contains(candidate)) { backends.emplace_back(candidate); }
    }
    if (name == "text/output_head") {
        std::vector<std::string> result{"text/final_hidden"};
        for (const auto& backend : backends) { result.push_back(backend + "/final_hidden"); }
        return result;
    }
    if (name == "proposal/head") {
        std::vector<std::string> result;
        for (const auto& backend : backends) { result.push_back(backend + "/final_hidden"); }
        return result;
    }
    if (name == "mtp/input_projection") { return {"mtp/stem_input"}; }

    const auto layer_role = [](std::string_view text, std::string_view prefix)
        -> std::optional<std::pair<std::string, std::string>> {
        if (!has_prefix(text, prefix)) { return std::nullopt; }
        const std::string block(prefix);
        return std::pair{block, std::string(text.substr(prefix.size()))};
    };

    std::string block;
    std::string role;
    bool in_layer = false;
    if (has_prefix(name, "text/layers/")) {
        const auto slash = name.find('/', std::string("text/layers/").size());
        if (slash != std::string::npos && slash > std::string("text/layers/").size()) {
            const auto digits = name.substr(std::string("text/layers/").size(),
                                            slash - std::string("text/layers/").size());
            if (!digits.empty() &&
                std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) {
                block    = name.substr(0, slash + 1);
                role     = name.substr(slash + 1);
                in_layer = true;
            }
        }
    } else if (has_prefix(name, "mtp/layer/")) {
        block    = "mtp/layers/0/";
        role     = name.substr(std::string("mtp/layer/").size());
        in_layer = !role.empty();
    } else if (has_prefix(name, "mtp/layers/0/")) {
        // bindings 里的 mtp 块名（与工具的正则一致：text/layers/<n>/ 或 mtp/layers/0/）。
        block    = "mtp/layers/0/";
        role     = name.substr(std::string("mtp/layers/0/").size());
        in_layer = !role.empty();
    }
    if (in_layer) {
        static const std::set<std::string, std::less<>> mixer{
            "attention/query",  "attention/key",         "attention/gate",
            "attention/value",  "gdn/query",             "gdn/key",
            "gdn/value",        "gdn/z",                 "gdn/a_projection",
            "gdn/b_projection"};
        if (mixer.contains(role)) { return {block + "mixer_input"}; }
        if (role == "attention/output" || role == "gdn/output") {
            return {block + role.substr(0, role.rfind('/')) + "/gated_output"};
        }
        const bool mlp_input = role == "mlp/gate" || role == "mlp/up" || role == "moe/router" ||
                               role == "moe/shared_score";
        if (mlp_input) { return {block + "ffn_input"}; }
        if (role == "mlp/down") { return {block + "mlp/product"}; }
        // moe/(experts/N|shared)/(gate|up) 与 moe/(experts/N|shared)/down
        if (has_prefix(role, "moe/")) {
            const auto middle = role.find('/', 4);
            if (middle != std::string::npos) {
                const auto leaf  = role.substr(middle + 1);
                const auto group = role.substr(0, middle);
                if (group == "moe/shared" || has_prefix(group, "moe/experts/")) {
                    if (leaf == "gate" || leaf == "up") { return {block + "ffn_input"}; }
                    if (leaf == "down") { return {block + group + "/product"}; }
                }
            }
        }
        return {};
    }
    if (name == "vision/patch_embedding") { return {"vision/patch_input"}; }
    if (name == "vision/merger/fc1" || name == "vision/merger/fc2") {
        return {name == "vision/merger/fc1" ? "vision/merger/input" : "vision/merger/activation"};
    }
    if (has_prefix(name, "vision/layers/")) {
        const auto second = name.find('/', std::string("vision/layers/").size());
        if (second == std::string::npos) { return {}; }
        const std::string vision_block = name.substr(0, second + 1);
        const std::string vision_role  = name.substr(second + 1);
        const std::map<std::string, std::string, std::less<>> position{
            {"attention/query", "attention_input"},  {"attention/key", "attention_input"},
            {"attention/value", "attention_input"},  {"attention/output", "attention_output"},
            {"mlp/fc1", "mlp_input"},                {"mlp/fc2", "mlp_activation"}};
        const auto found = position.find(vision_role);
        if (found == position.end()) { return {}; }
        return {vision_block + found->second};
    }
    if (has_prefix(name, "dflash2/") || has_prefix(name, "dflash/")) {
        const auto slash  = name.find('/');
        const std::string backend = name.substr(0, slash);
        const std::string draft_role = name.substr(slash + 1);
        if (draft_role == "feature_projection") { return {backend + "/target_features"}; }
        if (draft_role == "candidate_selector/hidden_projection") {
            return {backend + "/final_hidden"};
        }
        if (has_prefix(draft_role, "layers/")) {
            const auto second = draft_role.find('/', std::string("layers/").size());
            if (second == std::string::npos) { return {}; }
            const std::string draft_block = backend + "/" + draft_role.substr(0, second + 1);
            const std::string leaf        = draft_role.substr(second + 1);
            if (leaf == "attention/context_key" || leaf == "attention/context_value") {
                return {backend + "/context_input"};
            }
            if (leaf == "attention/query" || leaf == "attention/key" ||
                leaf == "attention/value") {
                return {draft_block + "query_projection_input"};
            }
            if (leaf == "attention/output") { return {draft_block + "attention_output"}; }
            if (leaf == "mlp/gate" || leaf == "mlp/up") { return {draft_block + "mlp_input"}; }
            if (leaf == "mlp/down") { return {draft_block + "mlp_product"}; }
            if (leaf == "attention_conv/kernel_projection" ||
                leaf == "mlp_conv/kernel_projection") {
                return {draft_block + leaf.substr(0, leaf.find('/')) + "_input"};
            }
        }
        return {};
    }
    return {};
}

} // namespace

Json normalize_v2_directory(const Json& v2_root, std::uint64_t payload_start,
                            std::uint64_t file_bytes, std::string_view entry_name) {
    const Json& identity = v2_root.at("identity");
    const std::string model_id   = identity.at("model_id").get<std::string>();
    const std::string weights_id = identity.at("weights_id").get<std::string>();
    const Json& old_objects      = v2_root.at("objects");

    const auto known = std::find_if(
        kKnownInputs.begin(), kKnownInputs.end(), [&](const KnownInput& entry) {
            return entry.model_id == model_id && entry.weights_id == weights_id &&
                   std::find(entry.counts.begin(), entry.counts.end(), old_objects.size()) !=
                       entry.counts.end();
        });
    if (known == kKnownInputs.end()) {
        throw ArtifactError(
            std::string(entry_name) + ": NInfer v2 artifact " + model_id + "/" + weights_id +
            " with " + std::to_string(old_objects.size()) +
            " objects is not one of the known official inputs. Upgrade it offline with: "
            "python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer");
    }

    const bool moe = model_id == "qwen3.6-35b-a3b";
    bool has_dflash2 = false;
    for (const auto& object : old_objects) {
        if (has_prefix(object.at("name").get<std::string>(), "dflash2/")) { has_dflash2 = true; }
    }
    const Json text  = text_config(moe);

    Json components = Json::object();
    components["text"] = Json{{"config", text},
                              {"resources", Json::object()},
                              {"proposal", Json{{"domain", "indexed"}, {"rows", 131072}}}};
    components["vision"] = Json{{"config", vision_config(moe)},
                                {"target", "text"},
                                {"resources", Json::object()}};
    components["mtp"] = Json{{"config", Json{{"architectures", Json::array({moe ? "Qwen3_5MoeMTP"
                                                                                  : "Qwen3_5MTP"})}}},
                             {"target", "text"}};
    if (moe) { components["dflash"] = Json{{"config", draft_config(false)}, {"target", "text"}}; }
    if (has_dflash2) {
        components["dflash2"] = Json{{"config", draft_config(true)}, {"target", "text"}};
    }

    // 对象：名字、偏移、字节数与权重字节全部沿用 v2 的，只换词汇。
    Json objects = Json::array();
    for (const auto& value : old_objects) {
        const std::string id     = value.at("name").get<std::string>();
        const std::string kind   = value.at("kind").get<std::string>();
        Json object{{"id", id}, {"kind", kind}, {"offset", value.at("offset")},
                    {"bytes", value.at("bytes")}};
        if (kind == "tensor") {
            const auto format = kFormats.find(value.at("format").get<std::string>());
            const auto layout = kLayouts.find(value.at("layout").get<std::string>());
            if (format == kFormats.end() || layout == kLayouts.end()) {
                throw ArtifactError(id + ": unknown v2 format/layout vocabulary");
            }
            object["shape"]  = value.at("shape");
            object["format"] = format->second;
            object["layout"] = layout->second;
            if (object.at("offset").get<std::uint64_t>() % 256 != 0 ||
                object.at("bytes").get<std::uint64_t>() != encoded_bytes(object)) {
                throw ArtifactError(id + ": invalid tensor geometry/encoding length");
            }
        } else if (kind == "resource") {
            if (value.at("encoding").get<std::string>() != "raw-bytes-v1") {
                throw ArtifactError(id + ": unsupported resource encoding");
            }
            object["encoding"] = "raw_bytes_v1";
        } else {
            throw ArtifactError(id + ": unsupported object kind");
        }
        objects.push_back(std::move(object));
    }

    std::map<std::string, Json, std::less<>> by_id;
    for (const auto& object : objects) {
        by_id.emplace(object.at("id").get<std::string>(), object);
    }
    if (by_id.size() != objects.size()) { throw ArtifactError("duplicate v2 object names"); }

    Json bindings              = Json::object();
    Json scalar_uses           = Json::object();
    std::map<std::string, std::string, std::less<>> parameter_formats;

    const auto put = [&](const std::string& name, const Json& object, const Shape& shape,
                         const std::optional<std::pair<std::uint64_t, std::uint64_t>>& bounds) {
        if (bindings.contains(name)) { throw ArtifactError("duplicate logical mapping " + name); }
        const std::string id = object.at("id").get<std::string>();
        if (!bounds.has_value()) {
            if (object.at("shape").get<Shape>() != shape) {
                throw ArtifactError(id + ": unexpected logical shape");
            }
            bindings[name] = Json{{"object", id}};
        } else {
            const auto [begin, end] = *bounds;
            if (!(begin < end) || end > product(object.at("shape").get<Shape>()) ||
                end - begin != product(shape)) {
                throw ArtifactError(name + ": invalid known row mapping");
            }
            bindings[name] =
                Json{{"parts", Json::array({Json{{"object", id}, {"range", Json::array({begin, end})}}})}};
        }
        parameter_formats[name] = object.at("format").get<std::string>();
    };

    const auto split = [&](const Json& object,
                           const std::vector<std::pair<std::string, std::uint64_t>>& entries,
                           std::uint64_t width) {
        std::uint64_t cursor = 0;
        for (const auto& [name, rows] : entries) {
            put(name, object, {rows, width}, {{cursor * width, (cursor + rows) * width}});
            cursor += rows;
        }
        if (object.at("shape").get<Shape>() != Shape{cursor, width}) {
            throw ArtifactError(object.at("id").get<std::string>() + ": unexpected fused parent shape");
        }
    };

    const std::uint64_t h      = text.at("hidden_size").get<std::uint64_t>();
    const std::uint64_t r      = text.at("vocab_size").get<std::uint64_t>();
    const std::uint64_t q      = text.at("num_attention_heads").get<std::uint64_t>() * 256;
    const std::uint64_t k      = text.at("num_key_value_heads").get<std::uint64_t>() * 256;
    const std::uint64_t nv     = text.at("linear_num_value_heads").get<std::uint64_t>();
    const std::uint64_t kg     = 2048;
    const std::uint64_t vg     = nv * 128;
    const std::uint64_t channels = 4096 + nv * 128;

    for (const auto& object : objects) {
        const std::string name = object.at("id").get<std::string>();
        if (object.at("kind").get<std::string>() == "resource") {
            if (!has_prefix(name, "frontend/")) {
                throw ArtifactError("unknown official resource " + name);
            }
            const std::string role = name.substr(std::string("frontend/").size());
            const std::string component =
                (role == "preprocessor_config.json" || role == "video_preprocessor_config.json")
                    ? "vision"
                    : "text";
            components[component]["resources"][role] = name;
            continue;
        }
        if (has_suffix(name, "/input_scale_divisor")) {
            if (object.at("format").get<std::string>() != "fp32" || !object.at("shape").empty()) {
                throw ArtifactError(name + ": invalid activation scalar");
            }
            const std::string prefix = name.substr(0, name.size() - std::string("/input_scale_divisor").size());
            const auto slash         = prefix.rfind('/');
            if (slash == std::string::npos) { throw ArtifactError("unknown activation scalar " + name); }
            const std::string group     = prefix.substr(0, slash);
            const std::string operation = prefix.substr(slash + 1);
            std::vector<std::string> roles;
            if (operation == "input_projection") {
                roles = has_suffix(group, "/attention")
                            ? std::vector<std::string>{"query", "key", "gate", "value"}
                            : std::vector<std::string>{"query", "key", "value", "z"};
            } else if (operation == "output_projection") {
                roles = {"output"};
            } else if (operation == "gate_up_projection") {
                roles = {"gate", "up"};
            } else if (operation == "down_projection") {
                roles = {"down"};
            } else {
                throw ArtifactError("unknown activation scalar " + name);
            }
            for (const auto& role : roles) {
                scalar_uses[group + "/" + role] = Json{{"object", name}};
            }
            continue;
        }
        if (name == "text/token_embedding" || name == "text/output_head" ||
            name == "text/final_norm") {
            put(name, object, has_suffix(name, "final_norm") ? Shape{h} : Shape{r, h}, std::nullopt);
            continue;
        }
        if (name == "text/draft_head" || name == "text/draft_head_token_ids") {
            const bool head = has_suffix(name, "draft_head");
            put(head ? "proposal/head" : "proposal/token_ids", object,
                head ? Shape{131072, h} : Shape{131072}, std::nullopt);
            continue;
        }
        if (name == "mtp/input_projection" || name == "mtp/embedding_norm" ||
            name == "mtp/hidden_norm" || name == "mtp/final_norm") {
            put(name, object,
                has_suffix(name, "input_projection") ? Shape{h, 2 * h} : Shape{h}, std::nullopt);
            continue;
        }
        {
            // (text/layers/<n>/ | mtp/layer/) <role>
            std::string block;
            std::string role;
            std::string layer_text;
            bool matched = false;
            if (has_prefix(name, "text/layers/")) {
                const auto slash = name.find('/', std::string("text/layers/").size());
                if (slash != std::string::npos) {
                    layer_text = name.substr(std::string("text/layers/").size(),
                                             slash - std::string("text/layers/").size());
                    block = name.substr(0, slash + 1);
                    role  = name.substr(slash + 1);
                    matched = !layer_text.empty() &&
                              std::all_of(layer_text.begin(), layer_text.end(),
                                          [](char c) { return c >= '0' && c <= '9'; });
                }
            } else if (has_prefix(name, "mtp/layer/")) {
                block   = "mtp/layers/0/";
                role    = name.substr(std::string("mtp/layer/").size());
                matched = !role.empty();
            }
            if (matched) {
                if (block == "mtp/layers/0/" || layer_text.empty()) {
                    // mtp/layer/ 只有一个块
                } else {
                    const auto layer = static_cast<std::uint64_t>(std::stoull(layer_text));
                    if (layer >= text.at("num_hidden_layers").get<std::uint64_t>()) {
                        throw ArtifactError(name + ": layer outside known model");
                    }
                }
                if (role == "input_norm" || role == "post_attention_norm") {
                    put(block + role, object, Shape{h}, std::nullopt);
                } else if (role == "attention/query_norm" || role == "attention/key_norm") {
                    put(block + role, object, Shape{256}, std::nullopt);
                } else if (role == "attention/output") {
                    put(block + role, object, Shape{h, q}, std::nullopt);
                } else if (role == "attention/query_key" || role == "attention/gate_value" ||
                           role == "attention/query_key_gate_value") {
                    const bool query_key = has_suffix(role, "/query_key");
                    const bool gate_value = has_suffix(role, "/gate_value");
                    const std::vector<std::string> roles =
                        query_key ? std::vector<std::string>{"query", "key"}
                                  : (gate_value ? std::vector<std::string>{"gate", "value"}
                                                : std::vector<std::string>{"query", "key", "gate",
                                                                           "value"});
                    std::vector<std::pair<std::string, std::uint64_t>> entries;
                    for (const auto& value : roles) {
                        entries.emplace_back(
                            block + "attention/" + value,
                            (value == "query" || value == "gate") ? q : k);
                    }
                    split(object, entries, h);
                } else if (role == "gdn/query_key" || role == "gdn/value_z" ||
                           role == "gdn/query_key_value_z") {
                    const bool query_key = has_suffix(role, "/query_key");
                    const bool value_z   = has_suffix(role, "/value_z");
                    const std::vector<std::string> roles =
                        query_key ? std::vector<std::string>{"query", "key"}
                                  : (value_z ? std::vector<std::string>{"value", "z"}
                                             : std::vector<std::string>{"query", "key", "value", "z"});
                    std::vector<std::pair<std::string, std::uint64_t>> entries;
                    for (const auto& value : roles) {
                        entries.emplace_back(block + "gdn/" + value,
                                             (value == "query" || value == "key") ? kg : vg);
                    }
                    split(object, entries, h);
                } else if (role == "gdn/a_b_projection") {
                    split(object, {{block + "gdn/a_projection", nv}, {block + "gdn/b_projection", nv}},
                          h);
                } else if (role == "gdn/a_projection" || role == "gdn/b_projection") {
                    put(block + role, object, Shape{nv, h}, std::nullopt);
                } else if (role == "gdn/a_log" || role == "gdn/dt_bias" || role == "gdn/norm" ||
                           role == "gdn/convolution" || role == "gdn/output") {
                    Shape shape;
                    if (role == "gdn/a_log" || role == "gdn/dt_bias") {
                        shape = {nv};
                    } else if (role == "gdn/norm") {
                        shape = {128};
                    } else if (role == "gdn/convolution") {
                        shape = {4, channels};
                    } else {
                        shape = {h, vg};
                    }
                    put(block + role, object, shape, std::nullopt);
                } else if (role == "mlp/gate_up" && !moe) {
                    split(object, {{block + "mlp/gate", 17408}, {block + "mlp/up", 17408}}, h);
                } else if (role == "mlp/down" && !moe) {
                    put(block + role, object, Shape{h, 17408}, std::nullopt);
                } else if (has_prefix(role, "moe/") && moe) {
                    const std::string prefix = block + "moe/";
                    if (role == "moe/router_shared_gate") {
                        split(object, {{prefix + "router", 256}, {prefix + "shared_score", 1}}, h);
                    } else if (role == "moe/routed_gate_up") {
                        std::vector<std::pair<std::string, std::uint64_t>> entries;
                        for (std::uint64_t expert = 0; expert < 256; ++expert) {
                            for (const char* part : {"gate", "up"}) {
                                entries.emplace_back(prefix + "experts/" +
                                                         std::to_string(expert) + "/" + part,
                                                     512);
                            }
                        }
                        split(object, entries, h);
                    } else if (role == "moe/routed_down") {
                        std::vector<std::pair<std::string, std::uint64_t>> entries;
                        for (std::uint64_t expert = 0; expert < 256; ++expert) {
                            entries.emplace_back(prefix + "experts/" + std::to_string(expert) + "/down",
                                                 h);
                        }
                        split(object, entries, 512);
                    } else if (role == "moe/shared_gate_up") {
                        split(object, {{prefix + "shared/gate", 512}, {prefix + "shared/up", 512}},
                              h);
                    } else if (role == "moe/shared_down") {
                        put(prefix + "shared/down", object, Shape{h, 512}, std::nullopt);
                    } else {
                        throw ArtifactError("unknown MoE object " + name);
                    }
                } else {
                    throw ArtifactError("unknown decoder object " + name);
                }
                continue;
            }
        }
        if (has_prefix(name, "vision/")) {
            if (name == "vision/patch_embedding" || name == "vision/patch_embedding_bias" ||
                name == "vision/position_embedding") {
                const Shape shape = name == "vision/patch_embedding"
                                        ? Shape{1152, 1536}
                                        : (name == "vision/patch_embedding_bias"
                                               ? Shape{1152}
                                               : Shape{2304, 1152});
                put(name, object, shape, std::nullopt);
                continue;
            }
            if (has_prefix(name, "vision/layers/")) {
                const auto slash = name.find('/', std::string("vision/layers/").size());
                if (slash != std::string::npos) {
                    const auto digits = name.substr(std::string("vision/layers/").size(),
                                                    slash - std::string("vision/layers/").size());
                    const std::string vision_block = name.substr(0, slash + 1);
                    const std::string vision_role  = name.substr(slash + 1);
                    if (!digits.empty() &&
                        std::all_of(digits.begin(), digits.end(),
                                    [](char c) { return c >= '0' && c <= '9'; })) {
                        if (std::stoull(digits) >= 27) {
                            throw ArtifactError("Vision layer outside known model");
                        }
                        if (vision_role == "attention/qkv") {
                            split(object,
                                  {{vision_block + "attention/query", 1152},
                                   {vision_block + "attention/key", 1152},
                                   {vision_block + "attention/value", 1152}},
                                  1152);
                        } else if (vision_role == "attention/qkv_bias") {
                            if (object.at("shape").get<Shape>() != Shape{3456}) {
                                throw ArtifactError("unexpected Vision qkv bias shape");
                            }
                            std::uint64_t index = 0;
                            for (const char* value : {"query", "key", "value"}) {
                                put(vision_block + "attention/" + value + "_bias", object, Shape{1152},
                                    {{index * 1152, (index + 1) * 1152}});
                                ++index;
                            }
                        } else {
                            const std::map<std::string, Shape, std::less<>> sizes{
                                {"attention/output", {1152, 1152}},
                                {"attention/output_bias", {1152}},
                                {"mlp/fc1", {4304, 1152}},
                                {"mlp/fc1_bias", {4304}},
                                {"mlp/fc2", {1152, 4304}},
                                {"mlp/fc2_bias", {1152}},
                                {"norm1/weight", {1152}},
                                {"norm1/bias", {1152}},
                                {"norm2/weight", {1152}},
                                {"norm2/bias", {1152}},
                            };
                            const auto found = sizes.find(vision_role);
                            if (found == sizes.end()) {
                                throw ArtifactError("unknown Vision object " + name);
                            }
                            std::string role_name = vision_role;
                            for (const char* from : {"norm1/", "norm2/"}) {
                                const std::string stem(from);
                                if (has_prefix(role_name, stem)) {
                                    role_name = stem.substr(0, stem.size() - 1) + "_" +
                                                role_name.substr(stem.size());
                                }
                            }
                            put(vision_block + role_name, object, found->second, std::nullopt);
                        }
                        continue;
                    }
                }
            }
            const std::string merger_role =
                has_prefix(name, "vision/merger/")
                    ? name.substr(std::string("vision/merger/").size())
                    : std::string();
            const std::map<std::string, Shape, std::less<>> merger_sizes{
                {"fc1", {4608, 4608}}, {"fc1_bias", {4608}}, {"fc2", {h, 4608}},
                {"fc2_bias", {h}},     {"norm/weight", {1152}}, {"norm/bias", {1152}},
            };
            const auto found = merger_sizes.find(merger_role);
            if (found == merger_sizes.end()) {
                throw ArtifactError("unknown Vision object " + name);
            }
            std::string role_name = merger_role;
            if (has_prefix(role_name, "norm/")) {
                role_name = "norm_" + role_name.substr(std::string("norm/").size());
            }
            put("vision/merger/" + role_name, object, found->second, std::nullopt);
            continue;
        }
        if (has_prefix(name, "dflash2/") || has_prefix(name, "dflash/")) {
            const auto slash             = name.find('/');
            const std::string backend    = name.substr(0, slash);
            const std::string draft_role = name.substr(slash + 1);
            if (!components.contains(backend)) {
                throw ArtifactError("unexpected backend " + backend);
            }
            const Json config = components[backend]["config"];
            const std::uint64_t inter =
                config.at("intermediate_size").get<std::uint64_t>();
            if (draft_role == "feature_projection" || draft_role == "context_norm" ||
                draft_role == "final_norm") {
                const Shape shape =
                    draft_role == "feature_projection"
                        ? Shape{h, config.at("dflash_config").at("target_layer_ids").size() * h}
                        : Shape{h};
                put(name, object, shape, std::nullopt);
                continue;
            }
            if (has_prefix(draft_role, "candidate_selector/") && backend == "dflash2") {
                const Shape shape = has_suffix(draft_role, "hidden_projection") ? Shape{256, h}
                                                                               : Shape{r, 256};
                put(name, object, shape, std::nullopt);
                continue;
            }
            if (!has_prefix(draft_role, "layers/")) {
                throw ArtifactError("unknown draft object " + name);
            }
            const auto second_slash = draft_role.find('/', std::string("layers/").size());
            if (second_slash == std::string::npos) {
                throw ArtifactError("unknown draft object " + name);
            }
            const auto digits = draft_role.substr(std::string("layers/").size(),
                                                 second_slash - std::string("layers/").size());
            const std::string block = backend + "/" + draft_role.substr(0, second_slash + 1);
            const std::string leaf  = draft_role.substr(second_slash + 1);
            if (digits.empty() ||
                !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
                std::stoull(digits) >= config.at("num_hidden_layers").get<std::uint64_t>()) {
                throw ArtifactError(name + ": layer outside known draft");
            }
            if (leaf == "attention/query_key_value") {
                split(object,
                      {{block + "attention/query", 4096},
                       {block + "attention/key", 1024},
                       {block + "attention/value", 1024}},
                      h);
                put(block + "attention/context_key", object, Shape{1024, h},
                    {{4096 * h, 5120 * h}});
                put(block + "attention/context_value", object, Shape{1024, h},
                    {{5120 * h, 6144 * h}});
            } else if (leaf == "mlp/gate_up") {
                split(object, {{block + "mlp/gate", inter}, {block + "mlp/up", inter}}, h);
            } else {
                std::map<std::string, Shape, std::less<>> sizes{
                    {"input_norm", {h}},
                    {"post_attention_norm", {h}},
                    {"attention/query_norm", {128}},
                    {"attention/key_norm", {128}},
                    {"attention/output", {h, 4096}},
                    {"mlp/down", {h, inter}},
                };
                if (backend == "dflash2") {
                    for (const char* branch : {"attention_conv", "mlp_conv"}) {
                        sizes[std::string(branch) + "/base_kernel"]       = {2, 2, h};
                        sizes[std::string(branch) + "/kernel_projection"] = {h / 4, h};
                    }
                }
                const auto found = sizes.find(leaf);
                if (found == sizes.end()) { throw ArtifactError("unknown draft object " + name); }
                put(block + leaf, object, found->second, std::nullopt);
            }
            continue;
        }
        throw ArtifactError("unknown stored tensor " + name);
    }

    Json uses = Json::array();
    for (const auto& [parameter, format] : parameter_formats) {
        const std::string policy = format == "nvfp4"                    ? "AllowA4"
                                   : format == "fp8_e4m3fn_row_bf16"    ? "AllowA8"
                                                                        : "A16Only";
        for (const auto& input : input_names(parameter, components)) {
            Json use{{"parameter", parameter}, {"input", input}, {"activation_policy", policy}};
            if (scalar_uses.contains(parameter)) {
                use["auxiliaries"] = Json{{"activation_input_divisor", scalar_uses[parameter]}};
            }
            uses.push_back(std::move(use));
        }
    }
    for (const auto& [name, unused] : scalar_uses.items()) {
        if (!bindings.contains(name)) {
            throw ArtifactError("activation scalar refers to an absent logical projection");
        }
    }
    const std::size_t expected = moe ? 32713 : (has_dflash2 ? 1513 : 1422);
    if (bindings.size() != expected) {
        throw ArtifactError(std::string(entry_name) +
                            ": the known logical parameter mapping is incomplete (" +
                            std::to_string(bindings.size()) + " vs " + std::to_string(expected) +
                            " bindings)");
    }
    for (const auto& [component, required] :
         std::vector<std::pair<std::string, std::vector<std::string>>>{
             {"text",
              {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json"}},
             {"vision", {"preprocessor_config.json", "video_preprocessor_config.json"}}}) {
        std::set<std::string> actual;
        for (const auto& [key, unused] : components[component]["resources"].items()) {
            actual.insert(key);
        }
        std::set<std::string> wanted(required.begin(), required.end());
        if (actual != wanted) {
            throw ArtifactError(std::string(entry_name) +
                                ": the known frontend resource set is incomplete");
        }
    }

    return Json{{"components", components},
                {"objects", objects},
                {"bindings", bindings},
                {"uses", uses},
                // v2 永远是单文件（没有 continuation）：payload_bytes 必须正好是头与目录之后
                // 剩下的字节，否则 Reader 的长度校验会拒绝。
                {"files", Json::array({Json{{"path", nullptr},
                                            {"payload_bytes", file_bytes - payload_start}}})},
                {"metadata", Json{{"name", model_id}}},
                {"provenance",
                 Json{{"read_as_v2",
                       Json{{"model_id", model_id}, {"weights_id", weights_id}}}}}};
}

} // namespace ninfer::artifact
