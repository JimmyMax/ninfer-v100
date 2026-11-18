#include "models/qwen3_5/program/structured_round.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/execution/linear.h"
#include <cmath>
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dspine.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

using detail::DFlashPersistentState;

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.parameters.draft.has_value()) {
        throw std::logic_error("masked draft schedule requires its weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    {
        require_dflash_state(state);
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const std::uint32_t>(config.target_layer_ids),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    {
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const std::uint32_t>(config.target_layer_ids),
        };
    }
}

template <class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    {
        const auto& target         = state.execution.parameters.model.config().text;
        const auto& config         = *state.execution.parameters.model.config().draft;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        nvtx::ScopedRange append_range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                       static_cast<std::uint64_t>(columns));
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] !=
                dimension(target.hidden_size * std::uint64_t(config.target_layer_ids.size())) ||
            features.ne[3] != 1 || positions.dtype != DType::I32 || positions.ne[0] != width ||
            positions.ne[1] != batch || commit_counts.dtype != DType::I32 ||
            commit_counts.ne[0] != batch || lanes.dtype != DType::I32 || lanes.ne[0] != batch ||
            table_rows.dtype != DType::I32 || table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window =
            batch == 1 && width > dimension(config.sliding_window.value_or(0));
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset =
            replace_local_window ? width - dimension(config.sliding_window.value_or(0)) : 0;
        const int local_width =
            replace_local_window ? dimension(config.sliding_window.value_or(0)) : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window
                ? static_cast<std::uint32_t>(dimension(config.sliding_window.value_or(0)))
                : envelope.min_count,
            replace_local_window
                ? static_cast<std::uint32_t>(dimension(config.sliding_window.value_or(0)))
                : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, dimension(config.sliding_window.value_or(0)),
                                state.execution.device.stream);
        }

        const int projected_width = config.dflash2.has_value() ? local_width : width;
        const Tensor input        = config.dflash2.has_value() && replace_local_window
                                        ? features.slice(1, local_offset, local_width)
                                        : features;
        const auto roots          = workspace::dflash_context(state.execution.work, target, config,
                                                              projected_width * batch);
        Tensor projected          = roots.projected;
        Tensor context            = roots.normalized;
        project(input.view(
                    {dimension(target.hidden_size * std::uint64_t(config.target_layer_ids.size())),
                     projected_width * batch}),
                state.execution.parameters.draft->feature_projection, projected,
                state.execution.work, state.execution.device.stream);
        ops::rmsnorm(projected, state.execution.parameters.draft->context_norm, config.rms_norm_eps,
                     false, context, state.execution.device.stream);

        if (config.dflash2) {
            std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
            if (config.num_hidden_layers != layers.size() ||
                config.local_layer_count() != layers.size() || config.rms_norm_eps != 1e-6F ||
                config.rope_theta != 1e7F) {
                throw std::invalid_argument(
                    "masked draft context: config is outside the fused materialization contract");
            }
            for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
                const auto& weights = state.execution.parameters.draft->layers[layer];
                layers[layer]       = {weights.context_key.weight, weights.context_value.weight,
                                       weights.key_norm, dflash_state(state).local_layer(layer)};
            }
            const Tensor local_positions =
                replace_local_window ? positions.slice(0, local_offset, local_width) : positions;
            ops::context_kv_materialize(
                context.view({dimension(target.hidden_size), local_width, batch}), local_positions,
                local_counts, lanes, layers, {local_envelope.min_count, local_envelope.max_count},
                state.execution.work, state.execution.device.stream);
        } else {
            for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
                auto layer_scope = state.execution.work.scope();
                const auto& weight =
                    state.execution.parameters.draft->layers.at(static_cast<std::size_t>(layer));
                const bool local_layer =
                    config.layer_types[layer] == DraftAttentionKind::SlidingAttention;
                const int layer_width   = local_layer ? local_width : width;
                const int layer_columns = layer_width * batch;
                Tensor layer_context    = local_layer && replace_local_window
                                              ? context.slice(1, local_offset, local_width)
                                              : context;
                Tensor layer_positions  = local_layer && replace_local_window
                                              ? positions.slice(0, local_offset, local_width)
                                              : positions;
                auto layer_roots = workspace::dflash_context_layer(state.execution.work, target,
                                                                   config, layer_columns);
                Tensor key_raw   = layer_roots.key_raw.view(
                    {dimension(config.attention.head_dim),
                       dimension(config.attention.num_key_value_heads), layer_columns});
                Tensor value = layer_roots.value.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_key_value_heads), layer_columns});
                Tensor key_flat =
                    key_raw.view({dimension(config.attention.key_width()), layer_columns});
                Tensor value_flat =
                    value.view({dimension(config.attention.key_width()), layer_columns});
                ops::linear_pair(layer_context, weight.context_key.weight,
                                 weight.context_value.weight, key_flat, value_flat,
                                 state.execution.device.stream);
                Tensor key = layer_roots.key.view({dimension(config.attention.head_dim),
                                                   dimension(config.attention.num_key_value_heads),
                                                   layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, config.rms_norm_eps, false, key,
                             state.execution.device.stream);
                ops::rope(layer_positions.view({layer_columns}),
                          dimension(config.attention.head_dim), config.rope_theta, key,
                          state.execution.device.stream);
                Tensor key_batch =
                    key.view({dimension(config.attention.head_dim),
                              dimension(config.attention.num_key_value_heads), layer_width, batch});
                Tensor value_batch    = value.view({dimension(config.attention.head_dim),
                                                    dimension(config.attention.num_key_value_heads),
                                                    layer_width, batch});
                Tensor position_batch = layer_positions.view({layer_width, batch});
                if (local_layer) {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                        dflash_state(state).local_layer(config.compact_layer_index(layer)),
                        state.execution.device.stream);
                } else {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                        dflash_state(state).full_batch_layer(config.compact_layer_index(layer)),
                        state.execution.device.stream);
                }
            }
        }
    }
}

void prepare_dynamic_branch(ExecutionCore& execution, const Tensor& residual, const Tensor& norm,
                            float eps, const DynamicConvParameters& weights,
                            workspace::DFlash2BranchRoots& branch) {
    auto scope      = execution.work.scope();
    const int width = residual.ne[1], batch = residual.ne[2];
    WorkspaceArena scratch(execution.work.alloc_bytes(
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(width, width, batch,
                                                                           batch)));
    ops::rmsnorm_dynamic_grouped_conv_prepare(
        residual, norm, eps, weights.base_kernel, weights.kernel_projection.weight, branch.prepared,
        branch.finish_delta, scratch, execution.device.stream);
}

void finish_dynamic_branch(ExecutionCore& execution, const Tensor& input,
                           const LinearParameters& projection, const DynamicConvParameters& weights,
                           const Tensor& finish_delta, Tensor& residual) {
    auto scope      = execution.work.scope();
    const int width = input.ne[1], batch = input.ne[2];
    WorkspaceArena scratch(
        execution.work.alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            input.ne[0], width, width, batch, batch)));
    ops::linear_dynamic_grouped_conv_add(input, projection.weight, weights.base_kernel,
                                         finish_delta, residual, scratch, execution.device.stream);
}

void propose_dflash2_batch(DFlashBatchContext& state, qwen3_5::DFlashDecodeState& frame, int batch,
                           int k, DFlashEnvelopes envelopes);

// DSpine (Tencent): a DFlash2 drafter with the code bridge injected between layers. After each
// layer it writes the previous position's predicted feature into the next position through a
// gated message (gate = sigmoid(query + sender + bias) * value), then refines the last layer's
// write with a serial table over the head's top candidates. The convolutions and selector are
// reused unchanged.
// DSpine (Tencent): a plain-DFlash drafter with the code bridge injected between layers, ported
// op-for-op from the upstream proven implementation (ninfer-wavecut src/models/dspine/drafter.cpp).
// The anchor column is the anchor TOKEN's embedding like every other column; the target's stacked
// features reach the drafter only through the context K/V ring (append_context_impl).
void prepare_dspine_head_write(DFlashBatchContext& state) {
    auto& persistent = dflash_state(state);
    if (persistent.dspine_head_prepared) { return; }
    const auto& target       = state.execution.parameters.model.config().text;
    const DraftParameters& weights = *state.execution.parameters.draft;
    const auto& bridge       = *weights.dspine_bridge;
    const std::int32_t h     = dimension(target.hidden_size);
    const std::int32_t rank  = dimension(bridge.rank);
    const std::int32_t vocab = dimension(target.vocab_size);
    auto& work               = state.execution.work;
    const cudaStream_t stream = state.execution.device.stream;
    // The head-write table's columns, one write row each: scaled by the final norm's weight, then
    // through the target's head, stored transposed. A one-time startup cost; small chunks keep the
    // logits plane inside the decode arena.
    work.reset();
    constexpr std::int32_t chunk = 32;
    for (std::int32_t first = 0; first < rank; first += chunk) {
        auto scope              = work.scope();
        const std::int32_t n    = std::min(chunk, rank - first);
        Tensor columns          = bridge.head_write.slice(1, first, n);
        Tensor scaled           = work.alloc(DType::BF16, {h, n});
        ops::dspine_scale_rows(columns, weights.final_norm, scaled, stream);
        Tensor logits           = work.alloc(DType::BF16, {vocab, n});
        project(scaled, weights.output_head, logits, work, stream);
        ops::dspine_store_head_write(logits, first, persistent.dspine_head_write, stream);
    }
    work.reset();
    persistent.dspine_head_prepared = true;
}

void propose_dspine_batch(DFlashBatchContext& state, qwen3_5::DFlashDecodeState& frame, int batch,
                          int k, DFlashEnvelopes envelopes) {
    if (!state.execution.parameters.model.config().draft->dspine) {
        return;
    }
    const auto& target = state.execution.parameters.model.config().text;
    const auto& config = *state.execution.parameters.model.config().draft;
    if (config.rms_norm_eps != 1e-6F || config.rope_theta != 1e7F) {
        throw std::invalid_argument(
            "DSpine: fused norm/RoPE requires epsilon=1e-6 and theta=1e7");
    }
    const auto& dspine = *config.dspine;

    const int width        = k + 1;
    const int columns      = width * batch;
    const int mask_columns = k * batch;
    auto& work             = state.execution.work;
    const cudaStream_t stream = state.execution.device.stream;
    const DraftParameters& weights = *state.execution.parameters.draft;
    const auto& bridge   = *weights.dspine_bridge;
    const std::int32_t h     = dimension(target.hidden_size);
    const std::int32_t rank  = dimension(bridge.rank);
    const std::int32_t code  = dimension(bridge.code_size);
    const std::int32_t d     = dimension(config.attention.head_dim);
    const std::int32_t hq    = dimension(config.attention.num_attention_heads);
    const std::int32_t kv    = dimension(config.attention.num_key_value_heads);
    const std::int32_t inner = dimension(config.intermediate_size);
    const float eps          = config.rms_norm_eps;

    Tensor anchors            = frame.anchors.slice(0, 0, batch);
    Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch);
    Tensor valid_columns      = frame.proposal_valid_columns.slice(0, 0, batch);
    Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch);
    Tensor ids                = frame.proposal_ids.slice(1, 0, batch);
    Tensor positions          = frame.proposal_positions.slice(1, 0, batch);
    Tensor drafts             = frame.draft_tokens.slice(1, 0, batch);
    work.reset();
    ops::prepare_masked_block(anchors, frontiers, valid_columns,
                              dimension(config.mask_token_id), ids, positions, stream);

    // The residual stream and its normed form; the bridge's refinement state (rms, packed,
    // message) and the last write's denominator the serial table reads.
    Tensor residual    = work.alloc(DType::BF16, {h, width, batch});
    Tensor normed      = work.alloc(DType::BF16, {h, width, batch});
    Tensor rms         = work.alloc(DType::FP32, {width, batch});
    Tensor denominator = work.alloc(DType::FP32, {width, batch});
    Tensor packed      = work.alloc(DType::BF16, {3 * rank, width, batch});
    Tensor message     = work.alloc(DType::BF16, {rank, width, batch});
    Tensor residual_flat = residual.view({h, columns});
    Tensor normed_flat   = normed.view({h, columns});

    embed_tokens(ids.view({columns}), state.execution.parameters.text.token_embedding,
                 state.execution.parameters.text.token_embedding_signs, residual_flat, stream);
    ops::rmsnorm(residual_flat, weights.layers.front().input_norm, eps, false, normed_flat,
                 stream);

    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));
    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
        const auto& layer = weights.layers[l];
        const bool last   = l + 1 == weights.layers.size();
        {
            // Attention over the normalized input, causal inside the block. The fused norm and
            // rotation work in place on the projections' outputs.
            auto scope   = work.scope();
            Tensor q     = work.alloc(DType::BF16, {d, hq, width, batch});
            Tensor key   = work.alloc(DType::BF16, {d, kv, width, batch});
            Tensor value = work.alloc(DType::BF16, {d, kv, width, batch});
            Tensor q_flat = q.view({d * hq, columns});
            Tensor k_flat = key.view({d * kv, columns});
            Tensor v_flat = value.view({d * kv, columns});
            ops::attn_input_proj(normed_flat, layer.query_key_value.weight, q_flat, k_flat,
                                 v_flat, stream);
            ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, q, key, stream);
            Tensor attended = work.alloc(DType::BF16, {d, hq, width, batch});
            ops::sliding_window_attention(
                q, key, value, positions, valid_columns, state_destinations, {d, hq, kv},
                dimension(config.sliding_window.value_or(0)), scale,
                dflash_state(state).local_layer(config.compact_layer_index(
                    static_cast<std::uint32_t>(l))),
                envelopes.local, work, attended, stream);
            // The attention output's [5120, 4096] Q8 geometry is not one the fused linear_add
            // admits on sm70; run the upstream's separate projection + residual add.
            Tensor output = work.alloc(DType::BF16, {h, columns});
            project(attended.view({d * hq, columns}), layer.output, output, work, stream);
            ops::residual_add(output, residual_flat, stream);
        }
        {
            // The MLP over the post-attention norm; the residual then holds the bridge's raw row.
            auto scope     = work.scope();
            Tensor x       = work.alloc(DType::BF16, {h, columns});
            Tensor product = work.alloc(DType::BF16, {inner, columns});
            ops::rmsnorm(residual_flat, layer.post_attention_norm, eps, false, x, stream);
            project_swiglu(x, layer.mlp.gate_up, product, work, stream);
            project_add(product, layer.mlp.down, residual_flat, work, stream);
        }
        {
            // The bridge's write, and the next layer's (or the final) norm of its output.
            auto scope        = work.scope();
            Tensor normalized = work.alloc(DType::BF16, {h, columns});
            Tensor rms_flat   = rms.view({columns});
            ops::dspine_bridge_read(residual_flat, dspine.eps, normalized, rms_flat, stream);
            Tensor codes = work.alloc(DType::BF16, {code, columns});
            project(normalized, bridge.read_weights[l], codes, work, stream);
            ops::dspine_bridge_codes(codes, dspine.eps, codes, stream);
            Tensor packed_flat = packed.view({3 * rank, columns});
            project(codes, bridge.projection, packed_flat, work, stream);
            ops::dspine_bridge_message(packed, anchors, bridge.sender_table, bridge.gate_bias,
                                       message, stream);
            Tensor delta      = work.alloc(DType::BF16, {h, width, batch});
            Tensor delta_flat = delta.view({h, columns});
            project(message.view({rank, columns}), bridge.write_weights[l], delta_flat, work,
                    stream);
            const Tensor& next = last ? weights.final_norm : weights.layers[l + 1].input_norm;
            ops::dspine_bridge_finish(residual, rms, delta, next, eps, residual, normed,
                                      last ? &denominator : nullptr, stream);
        }
    }
    // The mask columns' final hidden, packed row after row: the head's candidates, then the
    // serial table's walk over them.
    Tensor hidden = work.alloc(DType::BF16, {h, mask_columns});
    const std::size_t row_bytes = std::size_t(h) * dtype_size(DType::BF16);
    CUDA_CHECK(cudaMemcpy2DAsync(hidden.data, std::size_t(k) * row_bytes,
                                 static_cast<const std::byte*>(normed.data) + row_bytes,
                                 std::size_t(width) * row_bytes, std::size_t(k) * row_bytes,
                                 std::size_t(batch), cudaMemcpyDeviceToDevice, stream));
    Tensor candidates = work.alloc(DType::I32, {ops::kDSpineCandidates, k, batch});
    Tensor scores     = work.alloc(DType::FP32, {ops::kDSpineCandidates, k, batch});
    {
        auto head_scope = work.scope();
        Tensor logits   = work.alloc(DType::BF16, {dimension(target.vocab_size), mask_columns});
        project(hidden, weights.output_head, logits, work, stream);
        ops::candidate_topk(
            logits,
            dimension(state.execution.parameters.model.resources().public_token_count),
            candidates, scores, work, stream);
    }
    // Greedy draft sampling configs, one per row, uploaded with the block.
    static const std::vector<ops::SamplingConfig> greedy(kMaximumConcurrency);
    DeviceSpan configs =
        work.alloc_bytes(batch * sizeof(ops::SamplingConfig), alignof(ops::SamplingConfig));
    CUDA_CHECK(cudaMemcpyAsync(configs.data, greedy.data(), batch * sizeof(ops::SamplingConfig),
                               cudaMemcpyHostToDevice, stream));
    Tensor proposal_q = work.alloc(DType::FP32, {ops::kDSpineCandidates, k, batch});
    ops::dspine_path(candidates, scores, denominator, rms, packed, message,
                     dflash_state(state).dspine_head_write, bridge.sender_table, bridge.gate_bias,
                     anchors, frontiers,
                     static_cast<const ops::SamplingConfig*>(configs.data), drafts, proposal_q,
                     nullptr, work, stream);
    work.reset();
}

void propose_dflash2_batch(DFlashBatchContext& state, qwen3_5::DFlashDecodeState& frame, int batch,
                           int k, DFlashEnvelopes envelopes) {
    if (state.execution.parameters.model.config().draft->dflash2) {
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        if (config.rms_norm_eps != 1e-6F || config.rope_theta != 1e7F) {
            throw std::invalid_argument(
                "DFlash2: fused norm/RoPE requires epsilon=1e-6 and theta=1e7");
        }

        const int width                = k + 1;
        const int columns              = width * batch;
        const int mask_columns         = k * batch;
        auto& work                     = state.execution.work;
        const cudaStream_t stream      = state.execution.device.stream;
        const DraftParameters& weights = *state.execution.parameters.draft;
        Tensor anchors                 = frame.anchors.slice(0, 0, batch);
        Tensor frontiers               = frame.execution_frontiers.slice(0, 0, batch);
        Tensor valid_columns           = frame.proposal_valid_columns.slice(0, 0, batch);
        Tensor state_slots             = frame.state_destination_slots.slice(0, 0, batch);
        Tensor ids                     = frame.proposal_ids.slice(1, 0, batch);
        Tensor positions               = frame.proposal_positions.slice(1, 0, batch);
        work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns,
                                  dimension(config.mask_token_id), ids, positions, stream);
        Tensor residual = work.alloc(DType::BF16, {dimension(target.hidden_size), width, batch});
        Tensor flat_residual = residual.view({dimension(target.hidden_size), columns});
        embed_tokens(ids.view({columns}), state.execution.parameters.text.token_embedding,
                     state.execution.parameters.text.token_embedding_signs, flat_residual, stream);
        for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
            const auto& layer = weights.layers[layer_index];
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          layer_index);
            {
                auto scope  = work.scope();
                auto branch = workspace::dflash2_branch(work, target, config, width, batch);
                prepare_dynamic_branch(state.execution, residual, layer.input_norm,
                                       config.rms_norm_eps, *layer.attention_conv, branch);
                Tensor query = work.alloc(
                    DType::BF16, {dimension(config.attention.head_dim),
                                  dimension(config.attention.num_attention_heads), width, batch});
                Tensor key = work.alloc(
                    DType::BF16, {dimension(config.attention.head_dim),
                                  dimension(config.attention.num_key_value_heads), width, batch});
                Tensor value = work.alloc(
                    DType::BF16, {dimension(config.attention.head_dim),
                                  dimension(config.attention.num_key_value_heads), width, batch});
                Tensor query_flat =
                    query.view({dimension(config.attention.query_width()), columns});
                Tensor key_flat   = key.view({dimension(config.attention.key_width()), columns});
                Tensor value_flat = value.view({dimension(config.attention.key_width()), columns});
                ops::attn_input_proj(branch.prepared.view({dimension(target.hidden_size), columns}),
                                     layer.query_key_value.weight, query_flat, key_flat, value_flat,
                                     stream);
                ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, query, key, stream);
                Tensor attention = work.alloc(
                    DType::BF16, {dimension(config.attention.head_dim),
                                  dimension(config.attention.num_attention_heads), width, batch});
                ops::sliding_window_attention(
                    query, key, value, positions, valid_columns, state_slots,
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_attention_heads),
                     dimension(config.attention.num_key_value_heads)},
                    dimension(config.sliding_window.value_or(0)),
                    static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                    state.dflash.local_layer(static_cast<std::uint32_t>(layer_index)),
                    envelopes.local, work, attention, stream);
                finish_dynamic_branch(
                    state.execution,
                    attention.view({dimension(config.attention.query_width()), width, batch}),
                    layer.output, *layer.attention_conv, branch.finish_delta, residual);
            }
            {
                auto scope  = work.scope();
                auto branch = workspace::dflash2_branch(work, target, config, width, batch);
                prepare_dynamic_branch(state.execution, residual, layer.post_attention_norm,
                                       config.rms_norm_eps, *layer.mlp_conv, branch);
                Tensor intermediate =
                    work.alloc(DType::BF16, {dimension(config.intermediate_size), width, batch});
                Tensor intermediate_flat =
                    intermediate.view({dimension(config.intermediate_size), columns});
                project_swiglu(branch.prepared.view({dimension(target.hidden_size), columns}),
                               layer.mlp.gate_up, intermediate_flat, work, stream);
                finish_dynamic_branch(state.execution, intermediate, layer.mlp.down,
                                      *layer.mlp_conv, branch.finish_delta, residual);
            }
        }
        Tensor hidden = work.alloc(DType::BF16, {dimension(target.hidden_size), mask_columns});
        ops::rmsnorm_pack_tail(residual, weights.final_norm, hidden, stream);
        Tensor candidates = frame.candidate_ids.slice(2, 0, batch);
        Tensor ids_flat =
            candidates.view({dimension(config.dflash2->selector_top_k), mask_columns});
        Tensor scores =
            work.alloc(DType::FP32, {dimension(config.dflash2->selector_top_k), mask_columns});
        // A rotated head reads its own copy: the selector projection below takes the primal hidden.
        if (state.execution.proposal_head == ProposalHead::Full) {
            const auto& head        = state.execution.parameters.draft->output_head;
            const Tensor head_input = rotated_input(hidden, head.hadamard_signs, work, stream);
            ops::linear_topk(
                head_input, head.weight,
                dimension(state.execution.parameters.model.resources().public_token_count),
                ids_flat, scores, work, stream);
        } else {
            const auto& head        = *state.execution.parameters.proposal;
            const Tensor head_input = rotated_input(hidden, head.head.hadamard_signs, work, stream);
            if (head.token_ids) {
                ops::linear_topk(head_input, head.head.weight, *head.token_ids, ids_flat, scores,
                                 work, stream);
            } else {
                ops::linear_topk(
                    head_input, head.head.weight,
                    dimension(state.execution.parameters.model.resources().public_token_count),
                    ids_flat, scores, work, stream);
            }
        }
        Tensor projected =
            work.alloc(DType::BF16, {dimension(config.dflash2->selector_rank), mask_columns});
        project(hidden, weights.selector->hidden_projection, projected, work, stream);
        Tensor drafts     = frame.draft_tokens.slice(1, 0, batch);
        Tensor proposal_q = frame.proposal_q.slice(2, 0, batch);
        ops::candidate_selector_path(
            candidates, scores.view({dimension(config.dflash2->selector_top_k), k, batch}),
            projected.view({dimension(config.dflash2->selector_rank), k, batch}), anchors,
            weights.selector->predecessor_codebook, weights.selector->successor_codebook, frontiers,
            frame.sampling, drafts, proposal_q, work, stream);
        work.reset();
    }
}

void propose_batch_impl(DFlashBatchContext& state, qwen3_5::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    const auto& draft_cfg = state.execution.parameters.model.config().draft;
    if (draft_cfg->dspine) {
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(k + 1U) * batch_size);
        propose_dspine_batch(state, frame, batch_size, k, envelopes);
    } else if (draft_cfg->dflash2) {
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(k + 1U) * batch_size);
        propose_dflash2_batch(state, frame, batch_size, k, envelopes);
    } else {
        const auto& target         = state.execution.parameters.model.config().text;
        const auto& config         = *state.execution.parameters.model.config().draft;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(columns));
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor full_rows          = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions          = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns,
                                  dimension(config.mask_token_id), ids, positions,
                                  state.execution.device.stream);
        Tensor residual =
            state.execution.work.alloc(DType::BF16, {dimension(target.hidden_size), columns});
        embed_tokens(ids.view({columns}), state.execution.parameters.text.token_embedding,
                     state.execution.parameters.text.token_embedding_signs, residual,
                     state.execution.device.stream);

        for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          static_cast<std::uint64_t>(layer));
            const auto& weight =
                state.execution.parameters.draft->layers.at(static_cast<std::size_t>(layer));
            {
                nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                                  nvtx::Category::Attention,
                                                  static_cast<std::uint64_t>(layer));
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace::dflash_attention(state.execution.work, target, config, columns);
                ops::rmsnorm(residual, weight.input_norm, config.rms_norm_eps, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw = roots.query_raw.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_attention_heads), columns});
                Tensor key_raw =
                    roots.key_raw.view({dimension(config.attention.head_dim),
                                        dimension(config.attention.num_key_value_heads), columns});
                Tensor value =
                    roots.value.view({dimension(config.attention.head_dim),
                                      dimension(config.attention.num_key_value_heads), columns});
                Tensor query_flat =
                    query_raw.view({dimension(config.attention.query_width()), columns});
                Tensor key_flat = key_raw.view({dimension(config.attention.key_width()), columns});
                Tensor value_flat = value.view({dimension(config.attention.key_width()), columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value.weight, query_flat,
                                     key_flat, value_flat, state.execution.device.stream);
                Tensor query =
                    roots.query.view({dimension(config.attention.head_dim),
                                      dimension(config.attention.num_attention_heads), columns});
                Tensor key =
                    roots.key.view({dimension(config.attention.head_dim),
                                    dimension(config.attention.num_key_value_heads), columns});
                ops::rmsnorm(query_raw, weight.query_norm, config.rms_norm_eps, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, config.rms_norm_eps, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), dimension(config.attention.head_dim),
                          config.rope_theta, query, key, state.execution.device.stream);
                Tensor query_batch = query.view({dimension(config.attention.head_dim),
                                                 dimension(config.attention.num_attention_heads),
                                                 width, batch_size});
                Tensor key_batch =
                    key.view({dimension(config.attention.head_dim),
                              dimension(config.attention.num_key_value_heads), width, batch_size});
                Tensor value_batch     = value.view({dimension(config.attention.head_dim),
                                                     dimension(config.attention.num_key_value_heads),
                                                     width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_attention_heads), width, batch_size});
                if (config.layer_types[layer] == DraftAttentionKind::SlidingAttention) {
                    ops::sliding_window_attention(
                        query_batch, key_batch, value_batch, positions, valid_columns,
                        state_destinations,
                        {dimension(config.attention.head_dim),
                         dimension(config.attention.num_attention_heads),
                         dimension(config.attention.num_key_value_heads)},
                        dimension(config.sliding_window.value_or(0)),
                        static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                        dflash_state(state).local_layer(config.compact_layer_index(layer)),
                        envelopes.local, state.execution.work, attention_batch,
                        state.execution.device.stream);
                } else {
                    ops::context_softmax_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        {dimension(config.attention.head_dim),
                         dimension(config.attention.num_attention_heads),
                         dimension(config.attention.num_key_value_heads)},
                        static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                        dflash_state(state).full_batch_layer(config.compact_layer_index(layer)),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.stream);
                }
                project_add(
                    roots.attention.view({dimension(config.attention.query_width()), columns}),
                    weight.output, residual, state.execution.work, state.execution.device.stream);
            }
            {
                nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                            static_cast<std::uint64_t>(layer));
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace::dflash_mlp(state.execution.work, target, config, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, config.rms_norm_eps, false,
                             roots.hidden, state.execution.device.stream);
                project_swiglu(roots.hidden, weight.mlp.gate_up, roots.intermediate,
                               state.execution.work, state.execution.device.stream);
                project_add(roots.intermediate, weight.mlp.down, residual, state.execution.work,
                            state.execution.device.stream);
            }
        }

        Tensor packed =
            state.execution.work.alloc(DType::BF16, {dimension(target.hidden_size),
                                                     static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes     = static_cast<std::size_t>(dimension(target.hidden_size)) *
                                      static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(dimension(target.hidden_size)) * width * element_bytes;
        const auto* source =
            static_cast<const std::byte*>(residual.data) +
            static_cast<std::size_t>(dimension(target.hidden_size)) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden =
            state.execution.work.alloc(DType::BF16, {dimension(target.hidden_size),
                                                     static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.parameters.draft->final_norm, config.rms_norm_eps,
                     false, proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16,
                {dimension(target.vocab_size), static_cast<std::int32_t>(k) * batch_size});
            project(proposal_hidden, state.execution.parameters.draft->output_head, logits,
                    state.execution.work, state.execution.device.stream);
            ops::argmax(logits, flat_drafts,
                        dimension(state.execution.parameters.model.resources().public_token_count),
                        state.execution.device.stream);
        } else {
            if (!state.execution.parameters.proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.parameters.proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {dimension(proposal.rows), static_cast<std::int32_t>(k) * batch_size});
            project(proposal_hidden, proposal.head, logits, state.execution.work,
                    state.execution.device.stream);
            ops::argmax(
                logits, flat_drafts,
                proposal.token_ids
                    ? dimension(proposal.rows)
                    : dimension(state.execution.parameters.model.resources().public_token_count),
                state.execution.device.stream);
            if (proposal.token_ids)
                ops::proposal_remap_token_ids(
                    flat_drafts, static_cast<const std::int32_t*>(proposal.token_ids->data),
                    dimension(proposal.rows), state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::CausalAttentionExecutionEnvelope target_envelope) {
    target_envelope.wide_verification = state.ngram && k > 15;
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashVerifyMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        auto frame               = state.frame.narrowed(k);
        const std::int32_t width = static_cast<std::int32_t>(k) + 1;
        const std::size_t ingress_bytes =
            state.ngram ? sizeof(qwen3_5::DFlashDecodeIngress)
                        : offsetof(qwen3_5::DFlashDecodeIngress, ngram_tokens);
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress, ingress_bytes,
                                   cudaMemcpyHostToDevice, state.execution.device.stream));

        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts     = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents            = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows        = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor active_lanes       = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions   = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts      = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.verify_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {dimension(state.execution.parameters.draft->feature_projection.weight.k),
                          frame.append_positions.ne[0], batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, active_lanes,
                                   context_starts, frontiers, compact_features, append_positions,
                                   append_counts, state.execution.device.stream);
        append_context_impl(state, compact_features, append_positions, append_counts,
                            state_destinations, dflash_rows,
                            {0, static_cast<std::uint32_t>(frame.append_positions.ne[0])});

        const auto proposal_k = state.neural_proposal_drafts;
        if (proposal_k == 0 || proposal_k > kDFlashDecodeMaximumDrafts || proposal_k > k ||
            (!state.ngram && proposal_k != k)) {
            throw std::logic_error("neural proposal is outside its supported frame");
        }
        // A neural round verifies at the drafter's own width. A batch>1 ngram round also runs the
        // drafter, at the round's wider width: its leading proposal_k drafts are unchanged under a
        // wider causal proposal, and rows without a copy verify only those (their extent).
        if (!state.ngram || batch_size > 1) {
            propose_batch_impl(state, frame, batch_size, k, envelopes);
        }
        if (state.ngram) {
            auto* ingress = static_cast<std::byte*>(frame.ingress.data);
            // Per-row copy payload, column-major per row at the round's draft width (k), matching
            // the frame's [k, batch] stride. Only rows flagged in copy_rows take it.
            Tensor copy_rows(ingress + offsetof(qwen3_5::DFlashDecodeIngress, copy_rows),
                             DType::I32, {batch_size});
            Tensor copy_drafts(ingress + offsetof(qwen3_5::DFlashDecodeIngress, ngram_tokens),
                               DType::I32, {static_cast<std::int32_t>(k), batch_size});
            Tensor copy_candidates;
            Tensor copy_q;
            Tensor candidates;
            Tensor proposal_q;
            // DFlash keeps its deterministic-draft verifier and count publication contract.
            // DFlash2 represents the same deterministic proposal as a one-hot sparse law.
            if (state.execution.parameters.model.config().draft->dflash2.has_value()) {
                if (!frame.candidate_ids.data || !frame.proposal_q.data) {
                    throw std::logic_error("DFlash2 ngram requires a sparse acceptance frame");
                }
                const std::initializer_list<std::int32_t> sparse_shape{
                    ops::kSparseSpeculativeCandidates, static_cast<std::int32_t>(k), batch_size};
                copy_candidates =
                    Tensor(ingress + offsetof(qwen3_5::DFlashDecodeIngress, ngram_candidates),
                           DType::I32, sparse_shape);
                copy_q     = Tensor(ingress + offsetof(qwen3_5::DFlashDecodeIngress, ngram_q),
                                    DType::FP32, sparse_shape);
                candidates = frame.candidate_ids.slice(2, 0, batch_size);
                proposal_q = frame.proposal_q.slice(2, 0, batch_size);
            }
            ops::speculative_overlay_copy_proposals(copy_rows, copy_drafts, copy_candidates, copy_q,
                                                    drafts, candidates, proposal_q,
                                                    state.execution.device.stream);
        }
        if (state.execution.io.structured) {
            state.execution.io.structured->enqueue_dflash(drafts, state.execution.device.stream);
        }
        ops::speculative_prepare_verify_inputs(anchors, drafts, frontiers, extents, verify_ids,
                                               target_positions, state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        card.set_stage_runtime(state.execution.stages);
        card.set_expert_runtime(state.execution.experts);
        card.set_rope_yarn(state.execution.rope_yarn);
        DFlashFeatureSink sink =
            batch_feature_sink_impl(state, active_lanes, valid_columns, width, batch_size);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeDFlashTarget, nvtx::Category::DFlash,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(
                state.execution, state.continuation_hidden_store, card,
                TargetVerifyFrameView{
                    .ids                     = verify_ids,
                    .cache_positions         = target_positions,
                    .rope_positions          = target_rope,
                    .valid_columns           = valid_columns,
                    .kv_table_rows           = text_rows,
                    .state_source_slots      = state_sources,
                    .state_destination_slots = state_destinations,
                    .target_hidden           = target_hidden,
                    .target_logits           = target_logits,
                    .target_tokens           = target_tokens,
                    .drafts                  = drafts,
                    .current_extents         = extents,
                    .candidate_ids           = frame.candidate_ids.data
                                                   ? frame.candidate_ids.slice(2, 0, batch_size)
                                                   : Tensor{},
                    .proposal_q =
                        frame.proposal_q.data ? frame.proposal_q.slice(2, 0, batch_size) : Tensor{},
                    .frontiers       = frontiers,
                    .anchors         = anchors,
                    .licensed_tokens = licensed_tokens,
                    .licensed_counts = licensed_counts,
                    .accepted_drafts = accepted,
                    .selected_hidden = selected_hidden,
                    .replay_records  = state.execution.replay_records,
                    .sampling        = frame.sampling,
                    .feature_sink    = &sink,
                },
                target_envelope);
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_5::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl(state, features, positions, commit_counts, lanes, table_rows, envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl(state, features, positions, commit_counts, lanes, table_rows, envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::CausalAttentionExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    if (state.execution.parameters.model.config().draft->dspine) {
        // A one-time table build, on the stream but outside the captured body.
        prepare_dspine_head_write(state);
    }
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::CausalAttentionExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    if (state.execution.parameters.model.config().draft->dspine) {
        prepare_dspine_head_write(state);
    }
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution
