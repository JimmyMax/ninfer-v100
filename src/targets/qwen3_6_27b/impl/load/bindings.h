#pragma once

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include <ninfer/ops/gguf_projection.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kTextLayers          = 64;
inline constexpr std::size_t kFullAttentionLayers = 16;
inline constexpr std::size_t kGdnLayers           = 48;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format          = artifact::NumericFormat::BF16;
    std::int32_t rows                       = 0;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
    // INT32 [K] input-gather columns object bound alongside a GGUF matrix whose stored columns
    // are a permutation of its input's. Empty when absent.
    std::optional<artifact::ObjectHandle> input_columns;
};

struct MlpPlan {
    WeightPlan gate_up;
    // Present for GGUF profiles whose gate and up halves are separate parents of possibly
    // different block types; gate_up then holds the gate and up the up.
    std::optional<WeightPlan> up;
    WeightPlan down;
};

struct SplitAttentionProjectionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
};

struct FusedAttentionProjectionPlan {
    WeightPlan query_key_gate_value;
};

// One GGUF attention component: a row range [row, row + rows) of a possibly shared parent.
struct GgufAttentionProjectionEntry {
    WeightPlan parent;
    std::int32_t row    = 0;
    std::int32_t rows   = 0;
    std::int32_t output = 0; // 0 = q, 1 = gate, 2 = k, 3 = v
};

// The four components may share physical parents (e.g. a fused [query; key] matrix); entries
// referencing the same object are bound once and sliced by row at materialization time.
struct GgufAttentionProjectionPlan {
    std::array<GgufAttentionProjectionEntry, 4> entries;
};

struct FullAttentionPlan {
    std::variant<SplitAttentionProjectionPlan, FusedAttentionProjectionPlan,
                 GgufAttentionProjectionPlan>
        projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct SplitGdnInputProjectionPlan {
    WeightPlan query_key;
    WeightPlan value_z;
};

struct FusedGdnInputProjectionPlan {
    WeightPlan query_key_value_z;
};

// A GDN input projection stored as GGUF block matrices. The qkv parent holds q/k/v/z (fused) or
// just q/k/v; the optional z parent covers the split layout (qkv [10240,5120] plus z [6144,5120]).
struct GgufGdnInputProjectionPlan {
    WeightPlan query_key_value;
    std::optional<WeightPlan> z;
};

struct SplitGdnControlProjectionPlan {
    WeightPlan a_projection;
    WeightPlan b_projection;
};

struct FusedGdnControlProjectionPlan {
    WeightPlan a_b_projection;
};

using GdnControlProjectionPlan =
    std::variant<SplitGdnControlProjectionPlan, FusedGdnControlProjectionPlan>;

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    GdnControlProjectionPlan control_projection;
    std::variant<SplitGdnInputProjectionPlan, FusedGdnInputProjectionPlan,
                 GgufGdnInputProjectionPlan>
        input_projection;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    WeightPlan input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    WeightPlan query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct DFlash2DynamicConvPlan {
    artifact::ObjectHandle base_kernel;
    WeightPlan kernel_projection;
};

struct DFlash2LayerPlan {
    artifact::ObjectHandle input_norm;
    DFlash2DynamicConvPlan attention_conv;
    WeightPlan query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan attention_output;
    artifact::ObjectHandle post_attention_norm;
    DFlash2DynamicConvPlan mlp_conv;
    WeightPlan gate_up;
    WeightPlan down;
};

struct DFlash2CandidateSelectorPlan {
    WeightPlan hidden_projection;
    artifact::ObjectHandle predecessor_codebook;
    artifact::ObjectHandle successor_codebook;
};

struct DFlash2Plan {
    WeightPlan feature_projection;
    artifact::ObjectHandle context_norm;
    std::array<DFlash2LayerPlan, qwen3_6::DFlash2Weights::layer_count> layers;
    artifact::ObjectHandle final_norm;
    DFlash2CandidateSelectorPlan candidate_selector;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    WeightPlan draft_head;
    WeightPlan draft_head_token_ids;
    MtpPlan mtp;
    std::optional<DFlash2Plan> dflash2;

    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate_up;
    // Set when the GGUF profile stores gate and up as separate parents; gate_up then is the gate.
    std::optional<Weight> up;
    Weight down;
};

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_gate_value;
};

// Four independent GGUF block parents (q/gate/k/v), each written whole into its output plane.
struct GgufAttentionProjectionPayload {
    ops::GgufProjectionWeights weights;
};

using FullAttentionProjectionPayload =
    std::variant<SplitAttentionProjectionPayload, FusedAttentionProjectionPayload,
                 GgufAttentionProjectionPayload>;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

// Materialized GGUF GDN parts: q/k/v write the qkv plane at rows 0/2048/4096 and z its own plane.
struct GgufGdnInputProjectionPayload {
    ops::GgufProjectionWeights weights;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload,
                 GgufGdnInputProjectionPayload>;

struct SplitGdnControlProjectionPayload {
    Weight a_projection;
    Weight b_projection;
};

struct FusedGdnControlProjectionPayload {
    Weight a_b_projection;
};

using GdnControlProjectionPayload =
    std::variant<SplitGdnControlProjectionPayload, FusedGdnControlProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    GdnControlProjectionPayload control_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlash2Weights,
                       kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
