#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/typed_binding.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

bool is_early_attention_input(std::size_t layer) {
    return layer == 3 || layer == 7 || layer == 11 || layer == 15 || layer == 19 || layer == 23;
}

bool is_bf16_attention_output(std::size_t layer) { return layer == 3 || layer == 7; }

bool is_bf16_gdn_output(std::size_t layer) { return layer == 4; }

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        return NumericFormat::Q6G64_F16S;
    case WeightsProfile::Qwen38GroupwiseInt:
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38Nvfp4W4a4:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::Qwen38Nvfp4:
    case WeightsProfile::Qwen38Nvfp4Mixed:
        return NumericFormat::FP8_E4M3FN_ROW_BF16S;
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - static_cast<std::size_t>(offset) < 4) {
        throw artifact::ArtifactError(std::string(label) + ": FP32 word is outside payload");
    }
    const std::byte* value = bytes.data() + static_cast<std::size_t>(offset);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8U) |
           (std::to_integer<std::uint32_t>(value[2]) << 16U) |
           (std::to_integer<std::uint32_t>(value[3]) << 24U);
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw artifact::ArtifactError(std::string(label) + ": divisor must be finite and positive");
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape,
                       artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("NVFP4 weight requires a paired input divisor");
    }
    return WeightPlan{.object = artifact::bind_tensor(binder, name, format, shape, placement),
                      .format = format};
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent      = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);

    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, input_divisor_name, NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, input_divisor_name);
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, input_divisor_name);
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns,
                           bool prepack_for_qpn = true) {
    if (plan.format != NumericFormat::NVFP4) {
        Weight out = artifact::materialized_weight(materialized, plan.object, plan.format, rows,
                                                   columns);
#ifdef NINFER_VOLTA_BUILD
        // The load-time QPN permutation is only legal for weights consumed by the QPN linear
        // kernels. text/token_embedding is read row-major by the embedding gather, so it must
        // keep its checkpoint layout.
        if (prepack_for_qpn && out.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
            ::ninfer::ops::detail::fp8_prepack_qpn_sm70(out);
        }
#else
        (void)prepack_for_qpn;
#endif
        if (plan.input_columns) {
            out.input_columns = static_cast<const std::int32_t*>(
                materialized.device_data(*plan.input_columns));
        }
        return out;
    }
    (void)prepack_for_qpn;

    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(plan.object));

    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = rows;
    out.k                    = columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = rows;
    out.shape[1]             = columns;
    out.padded_shape[0]      = rows;
    out.padded_shape[1]      = columns;
    out.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(plan.input_scale_divisor_bits);
    return out;
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n) {
        throw std::logic_error("invalid target row view");
    }
    if (block.layout == QuantLayout::GgufBlocks) {
        // GGML blocks are row-aligned: a row view only advances qdata by whole rows of blocks.
        const auto gblock = gguf_block_shape(block.qtype);
        if (gblock.elements == 0 || block.k % gblock.elements != 0) {
            throw std::logic_error("GGUF row view needs whole-block columns");
        }
        const std::int64_t row_bytes =
            static_cast<std::int64_t>(block.k / gblock.elements) * gblock.bytes;
        Weight out                = block;
        out.qdata                 = static_cast<const std::byte*>(block.qdata) +
                                    static_cast<std::int64_t>(row_begin) * row_bytes;
        out.payload_bytes         = static_cast<std::int64_t>(row_count) * row_bytes;
        out.n                     = row_count;
        out.shape[0]              = row_count;
        out.padded_shape[0]       = row_count;
        return out;
    }
    if (block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up,
                                      plan.gate_up.rows != 0 ? plan.gate_up.rows : 34816, 5120);
    if (plan.up) {
        out.up = materialized_weight(materialized, *plan.up,
                                     plan.up->rows != 0 ? plan.up->rows : 17408, 5120);
    }
    out.down = materialized_weight(materialized, plan.down,
                                   plan.down.rows != 0 ? plan.down.rows : 5120, 17408);
#ifdef NINFER_VOLTA_BUILD
    // The mixed profile allows gate_up and down formats to differ; prepack
    // each NVFP4 weight on its own qtype instead of keying off gate_up.
    if (out.gate_up.qtype == QType::NVFP4) {
        ::ninfer::ops::detail::nvfp4_prepack_qpn_sm70(out.gate_up);
    }
    if (out.down.qtype == QType::NVFP4) {
        ::ninfer::ops::detail::nvfp4_prepack_qpn_sm70(out.down);
    }
#endif
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        return SplitAttentionProjectionPayload{
            .query_key  = materialized_weight(materialized, split->query_key, 7168, 5120),
            .gate_value = materialized_weight(materialized, split->gate_value, 7168, 5120),
        };
    }
    if (const auto* gguf = std::get_if<GgufAttentionProjectionPlan>(&plan.projection)) {
        // Materialize each distinct physical parent once; components sharing a parent take row
        // views of the same weight.
        std::vector<std::pair<std::size_t, Weight>> parents;
        auto parent_weight = [&](const WeightPlan& wp) -> const Weight& {
            for (const auto& [index, weight] : parents) {
                if (index == wp.object.index) { return weight; }
            }
            parents.emplace_back(wp.object.index,
                                 materialized_weight(materialized, wp, wp.rows, 5120));
            return parents.back().second;
        };
        GgufAttentionProjectionPayload out;
        for (const auto& entry : gguf->entries) {
            out.weights.parts.push_back(ops::GgufProjectionPart{
                .weight = row_view(parent_weight(entry.parent), entry.row, entry.rows),
                .output = entry.output,
                .row    = 0});
        }
        return out;
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, 14336, 5120),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key, 4096, 5120),
            .value_z   = materialized_weight(materialized, split->value_z, 12288, 5120),
        };
    }
    if (const auto* gguf = std::get_if<GgufGdnInputProjectionPlan>(&plan.input_projection)) {
        // q/k/v at rows 0/2048/4096 of the qkv plane; z at row 0 of its own plane (or at 10240
        // of the shared parent when the artifact fuses all four into one [16384,5120] matrix).
        const Weight qkv_parent =
            materialized_weight(materialized, gguf->query_key_value, gguf->query_key_value.rows,
                                5120);
        Weight z_parent = qkv_parent;
        std::int32_t z_row = 10240;
        if (gguf->z) {
            z_parent = materialized_weight(materialized, *gguf->z, gguf->z->rows, 5120);
            z_row    = 0;
        }
        GgufGdnInputProjectionPayload out;
        out.weights.parts.push_back(
            ops::GgufProjectionPart{.weight = row_view(qkv_parent, 0, 2048), .output = 0,
                                    .row    = 0});
        out.weights.parts.push_back(
            ops::GgufProjectionPart{.weight = row_view(qkv_parent, 2048, 2048), .output = 0,
                                    .row    = 2048});
        out.weights.parts.push_back(
            ops::GgufProjectionPart{.weight = row_view(qkv_parent, 4096, 6144), .output = 0,
                                    .row    = 4096});
        out.weights.parts.push_back(
            ops::GgufProjectionPart{.weight = row_view(z_parent, z_row, 6144), .output = 1,
                                    .row    = 0});
        return out;
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, 16384, 5120),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnControlProjectionPlan>(&plan.control_projection)) {
        return SplitGdnControlProjectionPayload{
            .a_projection = materialized_weight(materialized, split->a_projection, 48, 5120),
            .b_projection = materialized_weight(materialized, split->b_projection, 48, 5120),
        };
    }
    const auto& fused = std::get<FusedGdnControlProjectionPlan>(plan.control_projection);
    return FusedGdnControlProjectionPayload{
        .a_b_projection = materialized_weight(materialized, fused.a_b_projection, 96, 5120),
    };
}

void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {7168, 5120}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {7168, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {48, 5120}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {48, 5120}),
            };
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {4096, 5120}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {12288, 5120}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {34816, 5120});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {5120, 17408});
    }
}

// Returns a plan that binds the object with the format the artifact actually stores.
WeightPlan bind_adaptive(artifact::Binder& binder, std::string_view name,
                         std::initializer_list<std::uint64_t> /*expected_shape*/ = {},
                         artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    const auto* descriptor = binder.peek(name);
    if (descriptor == nullptr) {
        throw artifact::ArtifactError(std::string("missing artifact object ") + std::string(name));
    }
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(descriptor);
    if (tensor == nullptr) {
        throw artifact::ArtifactError(std::string("artifact object is not a tensor: ") +
                                      std::string(name));
    }
    WeightPlan plan{.object  = binder.require_tensor(name, tensor->format, tensor->layout,
                                                     std::span<const std::uint64_t>(
                                                         tensor->shape)),
                    .format  = tensor->format,
                    .rows    = static_cast<std::int32_t>(tensor->shape.at(0))};
    if (placement == artifact::TensorPlacement::Device) {
        binder.materialize_on_device(plan.object);
    } else {
        binder.validate_only(plan.object);
    }
    return plan;
}

void bind_gguf_text_layers(artifact::Binder& binder, BindingPlan& out) {
    // The input-gather columns table is one shared INT32 object referenced by every GDN output
    // binding; bind it on first sight and reuse the handle (an object can only be bound once).
    std::optional<artifact::ObjectHandle> columns_handle;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            // The four components may share GGUF parents; group them by physical object and
            // slice the rows in the canonical [query, key, gate, value] order.
            const std::array<const char*, 4> names = {"attention/query", "attention/key",
                                                      "attention/gate", "attention/value"};
            const std::array<std::int32_t, 4> rows = {6144, 1024, 6144, 1024};
            const std::array<std::int32_t, 4> out  = {0, 2, 1, 3};
            std::array<const artifact::TensorDescriptor*, 4> desc{};
            for (std::size_t i = 0; i < 4; ++i) {
                const auto* d = binder.peek(prefix + names[i]);
                if (d == nullptr || !std::holds_alternative<artifact::TensorDescriptor>(*d)) {
                    throw artifact::ArtifactError("missing attention component " +
                                                  std::string(names[i]));
                }
                desc[i] = &std::get<artifact::TensorDescriptor>(*d);
            }
            GgufAttentionProjectionPlan plan;
            std::array<std::uint64_t, 4> bound_offsets{};
            bool bound[4] = {false, false, false, false};
            // row cursor per distinct physical object, advanced in canonical order
            std::array<std::uint64_t, 4> cursors{};
            for (std::size_t i = 0; i < 4; ++i) {
                std::size_t group = i;
                for (std::size_t j = 0; j < i; ++j) {
                    if (desc[j]->offset == desc[i]->offset) { group = j; break; }
                }
                if (!bound[group]) {
                    plan.entries[group].parent = bind_adaptive(binder, prefix + names[i]);
                    bound[group]               = true;
                }
                plan.entries[i].parent = plan.entries[group].parent;
                plan.entries[i].row    = static_cast<std::int32_t>(cursors[group]);
                plan.entries[i].rows   = rows[i];
                plan.entries[i].output = out[i];
                cursors[group] += static_cast<std::uint64_t>(rows[i]);
            }
            target.attention.projection = std::move(plan);
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_adaptive(binder, prefix + "attention/output",
                                                    {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {48, 5120}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {48, 5120}),
            };
            // q/k/v/z are contiguous GGML block rows of one fused [16384,5120] parent, or a
            // [10240,5120] qkv parent plus a separate [6144,5120] z parent. The aliases point at
            // the whole physical object, so bind each parent exactly once.
            {
                GgufGdnInputProjectionPlan gdn_input;
                gdn_input.query_key_value = bind_adaptive(binder, prefix + "gdn/query");
                if (gdn_input.query_key_value.rows != 16384) {
                    gdn_input.z = bind_adaptive(binder, prefix + "gdn/z");
                }
                target.gdn.input_projection = std::move(gdn_input);
            }
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output       = bind_adaptive(binder, prefix + "gdn/output", {5120, 6144});
            if (!columns_handle && binder.contains(prefix + "gdn/output/input_columns")) {
                columns_handle = bind_adaptive(binder, prefix + "gdn/output/input_columns",
                                               {6144}).object;
            }
            target.gdn.output.input_columns = columns_handle;
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        // gate and up are either one fused [gate;up] parent or two independent parents. The
        // fused parent exposes the whole [34816,5120] matrix through either alias; a 17408-row
        // parent is one half only, so bind up separately.
        target.mlp.gate_up = bind_adaptive(binder, prefix + "mlp/gate");
        if (target.mlp.gate_up.rows != 34816) {
            target.mlp.up = bind_adaptive(binder, prefix + "mlp/up");
        }
        target.mlp.down = bind_adaptive(binder, prefix + "mlp/down", {5120, 17408});
    }
}

void bind_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            WeightPlan input;
            if (is_early_attention_input(layer)) {
                input = bind_weight(binder, prefix + "attention/query_key_gate_value",
                                    NumericFormat::BF16, {14336, 5120});
            } else {
                input = bind_nvfp4_weight(
                    binder, prefix + "attention/query_key_gate_value", 14336, 5120,
                    prefix + "attention/input_projection/input_scale_divisor");
            }
            target.attention.projection =
                FusedAttentionProjectionPlan{.query_key_gate_value = input};
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            if (is_bf16_attention_output(layer)) {
                target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                      NumericFormat::BF16, {5120, 6144});
            } else {
                target.attention.output =
                    bind_nvfp4_weight(binder, prefix + "attention/output", 5120, 6144,
                                      prefix + "attention/output_projection/input_scale_divisor");
            }
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {48, 5120}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {48, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_nvfp4_weight(binder, prefix + "gdn/query_key_value_z", 16384, 5120,
                                      prefix + "gdn/input_projection/input_scale_divisor"),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            if (is_bf16_gdn_output(layer)) {
                target.gdn.output =
                    bind_weight(binder, prefix + "gdn/output", NumericFormat::BF16, {5120, 6144});
            } else {
                target.gdn.output =
                    bind_nvfp4_weight(binder, prefix + "gdn/output", 5120, 6144,
                                      prefix + "gdn/output_projection/input_scale_divisor");
            }
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                              prefix + "mlp/gate_up_projection/input_scale_divisor");
        target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                            prefix + "mlp/down_projection/input_scale_divisor");
    }
}

void bind_qwen38_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    constexpr NumericFormat kFp8 = NumericFormat::FP8_E4M3FN_ROW_BF16S;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value = bind_weight(
                    binder, prefix + "attention/query_key_gate_value", kFp8, {14336, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", kFp8, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = FusedGdnControlProjectionPlan{
                .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                              NumericFormat::BF16, {96, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_weight(binder, prefix + "gdn/query_key_value_z", kFp8, {16384, 5120}),
            };
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {128});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", kFp8, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        // nvfp4full artifacts (e.g. kvnxiao swift-1.5) store MLP as NVFP4 in ALL
        // layers; official nvfp4 artifacts keep layers >= 56 in FP8. Probe the
        // descriptor and bind accordingly so both variants work.
        const auto* gate_up_desc  = binder.peek(prefix + "mlp/gate_up");
        const auto* gate_up_tensor =
            gate_up_desc != nullptr ? std::get_if<artifact::TensorDescriptor>(gate_up_desc) : nullptr;
        const bool mlp_is_nvfp4 =
            gate_up_tensor != nullptr && gate_up_tensor->format == NumericFormat::NVFP4;
        if (mlp_is_nvfp4) {
            target.mlp.gate_up =
                bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                                  prefix + "mlp/gate_up_projection/input_scale_divisor");
            target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                                prefix + "mlp/down_projection/input_scale_divisor");
        } else {
            target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", kFp8, {34816, 5120});
            target.mlp.down    = bind_weight(binder, prefix + "mlp/down", kFp8, {5120, 17408});
        }
    }
}

void bind_qwen38_nvfp4_w4a4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value =
                    bind_nvfp4_weight(
                        binder, prefix + "attention/query_key_gate_value", 14336, 5120,
                        prefix + "attention/input_projection/input_scale_divisor"),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output =
                bind_nvfp4_weight(binder, prefix + "attention/output", 5120, 6144,
                                  prefix + "attention/output_projection/input_scale_divisor");
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = FusedGdnControlProjectionPlan{
                .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                              NumericFormat::BF16, {96, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_nvfp4_weight(binder, prefix + "gdn/query_key_value_z", 16384, 5120,
                                      prefix + "gdn/input_projection/input_scale_divisor"),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_nvfp4_weight(binder, prefix + "gdn/output", 5120, 6144,
                                  prefix + "gdn/output_projection/input_scale_divisor");
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                              prefix + "mlp/gate_up_projection/input_scale_divisor");
        target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                            prefix + "mlp/down_projection/input_scale_divisor");
    }
}

// Merkyor W4A4+W8A8 allocation: FP8 projections everywhere; MLP modules are
// NVFP4 or FP8 per the fixed per-layer sets from the source quantization
// config (group_0 / group_1).
constexpr std::array<std::size_t, 15> kMixedFp8GateUpLayers = {
    0, 1, 2, 3, 50, 52, 54, 56, 57, 58, 59, 60, 61, 62, 63,
};
constexpr std::array<std::size_t, 22> kMixedFp8DownLayers = {
    0, 1, 2, 3, 21, 42, 44, 46, 48, 49, 50, 52, 53, 54,
    56, 57, 58, 59, 60, 61, 62, 63,
};

template <std::size_t N>
bool mixed_layer_is_fp8(std::size_t layer, const std::array<std::size_t, N>& layers) {
    return std::find(layers.begin(), layers.end(), layer) != layers.end();
}

void bind_qwen38_nvfp4_mixed_text_layers(artifact::Binder& binder, BindingPlan& out) {
    constexpr NumericFormat kFp8 = NumericFormat::FP8_E4M3FN_ROW_BF16S;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value = bind_weight(
                    binder, prefix + "attention/query_key_gate_value", kFp8, {14336, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", kFp8, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = FusedGdnControlProjectionPlan{
                .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                              NumericFormat::BF16, {96, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_weight(binder, prefix + "gdn/query_key_value_z", kFp8, {16384, 5120}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", kFp8, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        if (mixed_layer_is_fp8(layer, kMixedFp8GateUpLayers)) {
            target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", kFp8, {34816, 5120});
        } else {
            target.mlp.gate_up =
                bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                                  prefix + "mlp/gate_up_projection/input_scale_divisor");
        }
        if (mixed_layer_is_fp8(layer, kMixedFp8DownLayers)) {
            target.mlp.down = bind_weight(binder, prefix + "mlp/down", kFp8, {5120, 17408});
        } else {
            target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                                prefix + "mlp/down_projection/input_scale_divisor");
        }
    }
}

DFlash2Plan bind_dflash2(artifact::Binder& binder, artifact::TensorPlacement placement) {
    const auto bind_tensor = [&](std::string_view name, NumericFormat format,
                                 std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    DFlash2Plan out;
    out.feature_projection = bind_weight(binder, "dflash2/feature_projection",
                                         NumericFormat::W8G32_F16S, {5120, 25600}, placement);
    out.context_norm       = bind_tensor("dflash2/context_norm", NumericFormat::BF16, {5120});
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        DFlash2LayerPlan& target = out.layers[layer];
        const std::string prefix = "dflash2/layers/" + std::to_string(layer) + "/";
        target.input_norm        = bind_tensor(prefix + "input_norm", NumericFormat::BF16, {5120});
        target.attention_conv.base_kernel =
            bind_tensor(prefix + "attention_conv/base_kernel", NumericFormat::BF16, {2, 2, 5120});
        target.attention_conv.kernel_projection =
            bind_weight(binder, prefix + "attention_conv/kernel_projection", NumericFormat::BF16,
                        {1280, 5120}, placement);
        target.query_key_value = bind_weight(binder, prefix + "attention/query_key_value",
                                             NumericFormat::W8G32_F16S, {6144, 5120}, placement);
        target.query_norm =
            bind_tensor(prefix + "attention/query_norm", NumericFormat::BF16, {128});
        target.key_norm = bind_tensor(prefix + "attention/key_norm", NumericFormat::BF16, {128});
        target.attention_output = bind_weight(binder, prefix + "attention/output",
                                              NumericFormat::W8G32_F16S, {5120, 4096}, placement);
        target.post_attention_norm =
            bind_tensor(prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp_conv.base_kernel =
            bind_tensor(prefix + "mlp_conv/base_kernel", NumericFormat::BF16, {2, 2, 5120});
        target.mlp_conv.kernel_projection =
            bind_weight(binder, prefix + "mlp_conv/kernel_projection", NumericFormat::BF16,
                        {1280, 5120}, placement);
        target.gate_up = bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::W8G32_F16S,
                                     {34816, 5120}, placement);
        target.down    = bind_weight(binder, prefix + "mlp/down", NumericFormat::W8G32_F16S,
                                     {5120, 17408}, placement);
    }
    out.final_norm = bind_tensor("dflash2/final_norm", NumericFormat::BF16, {5120});
    out.candidate_selector.hidden_projection =
        bind_weight(binder, "dflash2/candidate_selector/hidden_projection", NumericFormat::BF16,
                    {256, 5120}, placement);
    out.candidate_selector.predecessor_codebook = bind_tensor(
        "dflash2/candidate_selector/predecessor_codebook", NumericFormat::BF16, {248320, 256});
    out.candidate_selector.successor_codebook = bind_tensor(
        "dflash2/candidate_selector/successor_codebook", NumericFormat::BF16, {248320, 256});
    return out;
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    const bool gguf_profile = weights_profile == WeightsProfile::Qwen38Gguf;
    if (gguf_profile) {
        // The GGUF bundle stores the vocabulary endpoints in their own block formats; the
        // endpoint_format profiles (W8/FP8) do not apply.
        out.token_embedding = bind_adaptive(binder, "text/token_embedding", {248320, 5120});
    } else {
        const NumericFormat vocabulary_format = endpoint_format(weights_profile);
        out.token_embedding =
            bind_weight(binder, "text/token_embedding", vocabulary_format, {248320, 5120});
    }
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        bind_groupwise_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Gguf:
        bind_gguf_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen36Nvfp4:
        bind_nvfp4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4:
        bind_qwen38_nvfp4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4W4a4:
        bind_qwen38_nvfp4_w4a4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4Mixed:
        bind_qwen38_nvfp4_mixed_text_layers(binder, out);
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {5120});
    out.output_head =
        gguf_profile ? bind_adaptive(binder, "text/output_head", {248320, 5120})
                     : bind_weight(binder, "text/output_head", endpoint_format(weights_profile),
                                   {248320, 5120});
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    // The GGUF bundle names the indexed proposal head "proposal/head" / "proposal/token_ids".
    const std::string_view draft_head_name =
        gguf_profile ? std::string_view("proposal/head") : std::string_view("text/draft_head");
    const std::string_view draft_ids_name =
        gguf_profile ? std::string_view("proposal/token_ids")
                     : std::string_view("text/draft_head_token_ids");
    if (gguf_profile) {
        out.draft_head           = bind_adaptive(binder, draft_head_name, {131072, 5120},
                                                 proposal_placement);
        out.draft_head_token_ids = bind_adaptive(binder, draft_ids_name, {131072},
                                                 proposal_placement);
    } else {
        out.draft_head = WeightPlan{
            .object = artifact::bind_tensor(binder, draft_head_name, NumericFormat::Q4G64_F16S,
                                             {131072, 5120}, proposal_placement),
            .format = NumericFormat::Q4G64_F16S};
        out.draft_head_token_ids = WeightPlan{
            .object = artifact::bind_tensor(binder, draft_ids_name, NumericFormat::I32, {131072},
                                            proposal_placement),
            .format = NumericFormat::I32};
    }
    validate_draft_ids(binder, out.draft_head_token_ids.object);


    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp_handle                    = [&](std::string_view name, NumericFormat format,
                                        std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    // The GGUF bundle keys its single MTP layer as "mtp/layers/0/...".
    const std::string mtp_layer = gguf_profile ? "mtp/layers/0/" : "mtp/layer/";
    const auto bind_mtp_weight                    = [&](std::string_view name,
                                       NumericFormat native_format,
                                       std::initializer_list<std::uint64_t> shape) {
        return gguf_profile ? bind_adaptive(binder, name, shape, mtp_placement)
                            : WeightPlan{
                                  .object = bind_mtp_handle(name, native_format, shape),
                                  .format = native_format};
    };
    out.mtp.input_projection =
        bind_mtp_weight("mtp/input_projection", NumericFormat::W8G32_F16S, {5120, 10240});
    out.mtp.embedding_norm = bind_mtp_handle("mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm    = bind_mtp_handle("mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm = bind_mtp_handle(mtp_layer + "input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value =
        bind_mtp_weight(mtp_layer + "attention/query_key_gate_value", NumericFormat::W8G32_F16S,
                        {14336, 5120});
    out.mtp.query_norm =
        bind_mtp_handle(mtp_layer + "attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm =
        bind_mtp_handle(mtp_layer + "attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output = bind_mtp_weight(mtp_layer + "attention/output", NumericFormat::W8G32_F16S,
                                     {5120, 6144});
    out.mtp.post_attention_norm =
        bind_mtp_handle(mtp_layer + "post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up = bind_mtp_weight(mtp_layer + "mlp/gate_up", NumericFormat::W8G32_F16S,
                                          {34816, 5120});
    out.mtp.mlp.down = bind_mtp_weight(mtp_layer + "mlp/down", NumericFormat::W8G32_F16S,
                                        {5120, 17408});
    out.mtp.final_norm = bind_mtp_handle("mtp/final_norm", NumericFormat::BF16, {5120});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {5120, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    const bool has_dflash2 = binder.contains("dflash2/feature_projection");
    if (features.dflash2() && !has_dflash2) {
        throw artifact::ArtifactError(
            "DFlash2 was selected but the artifact has no DFlash2 weight bundle");
    }
    if (has_dflash2) {
        const artifact::TensorPlacement placement = features.dflash2()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
        out.dflash2                               = bind_dflash2(binder, placement);
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding        = materialized_weight(backing, plan.token_embedding, 248320, 5120,
                                                /*prepack_for_qpn=*/false);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {5120});
            target.projection            = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = materialized_weight(backing, source.attention.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {5120});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {48});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {48});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {10240, 4});
            target.projection.control_projection = load_gdn_control_projection(source.gdn, backing);
            target.projection.input_projection   = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output = materialized_weight(backing, source.gdn.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {5120});
    output_head = materialized_weight(backing, plan.output_head, 248320, 5120);
    if (plan.features.optimized_proposal()) {
        auto& proposal    = runtime.optimized_proposal.emplace();
        proposal.head     = materialized_weight(backing, plan.draft_head, 131072, 5120);
        proposal.token_ids = artifact::materialized_tensor(backing,
                                                           plan.draft_head_token_ids.object,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = materialized_weight(backing, plan.mtp.input_projection, 5120, 10240);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.attention.packed =
            materialized_weight(backing, plan.mtp.query_key_gate_value, 14336, 5120);
        mtp.attention.query       = row_view(mtp.attention.packed, 0, 6144);
        mtp.attention.key         = row_view(mtp.attention.packed, 6144, 1024);
        mtp.attention.output_gate = row_view(mtp.attention.packed, 7168, 6144);
        mtp.attention.value       = row_view(mtp.attention.packed, 13312, 1024);
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {256});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {256});
        mtp.output              = materialized_weight(backing, plan.mtp.output, 5120, 6144);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {5120});
    }

    if (plan.features.dflash2()) {
        if (!plan.dflash2) {
            throw std::logic_error("selected DFlash2 weights are absent from the binding plan");
        }
        const DFlash2Plan& source = *plan.dflash2;
        auto& dflash2             = runtime.dflash.emplace();
        dflash2.feature_projection =
            materialized_weight(backing, source.feature_projection, 5120, 25600);
        dflash2.context_norm = artifact::materialized_tensor(backing, source.context_norm,
                                                             NumericFormat::BF16, {5120});
        for (std::size_t layer = 0; layer < dflash2.layers.size(); ++layer) {
            const DFlash2LayerPlan& layer_source = source.layers[layer];
            qwen3_6::DFlash2LayerWeights& target = dflash2.layers[layer];
            target.input_norm = artifact::materialized_tensor(backing, layer_source.input_norm,
                                                              NumericFormat::BF16, {5120});
            target.attention_conv.base_kernel =
                artifact::materialized_tensor(backing, layer_source.attention_conv.base_kernel,
                                              NumericFormat::BF16, {5120, 2, 2});
            target.attention_conv.kernel_projection = materialized_weight(
                backing, layer_source.attention_conv.kernel_projection, 1280, 5120);
            target.query_key_value =
                materialized_weight(backing, layer_source.query_key_value, 6144, 5120);
            target.context_key   = row_view(target.query_key_value, 4096, 1024);
            target.context_value = row_view(target.query_key_value, 5120, 1024);
            target.query_norm    = artifact::materialized_tensor(backing, layer_source.query_norm,
                                                                 NumericFormat::BF16, {128});
            target.key_norm      = artifact::materialized_tensor(backing, layer_source.key_norm,
                                                                 NumericFormat::BF16, {128});
            target.attention_output =
                materialized_weight(backing, layer_source.attention_output, 5120, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, layer_source.post_attention_norm, NumericFormat::BF16, {5120});
            target.mlp_conv.base_kernel = artifact::materialized_tensor(
                backing, layer_source.mlp_conv.base_kernel, NumericFormat::BF16, {5120, 2, 2});
            target.mlp_conv.kernel_projection =
                materialized_weight(backing, layer_source.mlp_conv.kernel_projection, 1280, 5120);
            target.gate_up = materialized_weight(backing, layer_source.gate_up, 34816, 5120);
            target.down    = materialized_weight(backing, layer_source.down, 5120, 17408);
        }
        dflash2.final_norm =
            artifact::materialized_tensor(backing, source.final_norm, NumericFormat::BF16, {5120});
        dflash2.candidate_selector.hidden_projection =
            materialized_weight(backing, source.candidate_selector.hidden_projection, 256, 5120);
        dflash2.candidate_selector.predecessor_codebook =
            artifact::materialized_tensor(backing, source.candidate_selector.predecessor_codebook,
                                          NumericFormat::BF16, {256, 248320});
        dflash2.candidate_selector.successor_codebook =
            artifact::materialized_tensor(backing, source.candidate_selector.successor_codebook,
                                          NumericFormat::BF16, {256, 248320});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 5120, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {5120});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
