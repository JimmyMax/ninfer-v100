#include "models/qwen3_5/load/bindings.h"

#include "artifact/permute_fp8_qpn.h"    // 本仓库 V100 增量：FP8 QPN 加载时重排
#include "artifact/permute_nvfp4_qpn.h" // 本仓库 V100 增量：NVFP4 QPN 加载时重排
#include "artifact/permute_row_split.h"

#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace ninfer::models::qwen3_5::loading {

// Device-memory trades selected at startup. Each stores a Q8 matrix at a narrower group-64 width
// while the weights load; the artifact is unchanged and the measured quality cost of each is in
// docs/maintainer/quality-trade-experiments.md.
void apply_storage_trades(Bindings& bindings, const Config& config, const ModelWeights& weights,
                          const LoadOptions& options) {
    if (options.lm_head_q4 && options.lm_head_q6) {
        throw std::invalid_argument("--lm-head-q4 and --lm-head-q6 are mutually exclusive");
    }
    if (options.embedding_q4 && options.embedding_q6) {
        throw std::invalid_argument("--embedding-q4 and --embedding-q6 are mutually exclusive");
    }
    if (options.lm_head_q4 || options.lm_head_q6) {
        if (options.masked_draft()) {
            // DFlash's candidate top-k reads the Q8 output head through its own kernel.
            throw std::invalid_argument(
                "--lm-head-q4/--lm-head-q6 are not supported with DFlash or DFlash2");
        }
        // A head that already stores Q6 needs no work for --lm-head-q6; anything else must be
        // row-split Q8 (checked by Bindings::transcode).
        (void)bindings.transcode(weights.text.output_head,
                                 options.lm_head_q4 ? QType::Q4_G64_FP16 : QType::Q6_G64_FP16,
                                 options.lm_head_q4 ? "--lm-head-q4" : "--lm-head-q6");
    }
    if (options.embedding_q4 || options.embedding_q6) {
        (void)bindings.transcode(weights.text.token_embedding,
                                 options.embedding_q4 ? QType::Q4_G64_FP16 : QType::Q6_G64_FP16,
                                 options.embedding_q4 ? "--embedding-q4" : "--embedding-q6");
    }
    if (options.mtp_experts_q4) {
        if (!std::holds_alternative<MoeConfig>(config.text.ffn)) {
            throw std::invalid_argument(
                "--mtp-experts-q4 applies to a mixture-of-experts MTP layer (Qwen3_5MoeForCausalLM)");
        }
        // The MTP layer is bound only when MTP is selected; without it there is nothing to store.
        if (weights.mtp) {
            // Stored like the Text layers of the official 35B-A3B recipe: Q4 gate/up, Q6 down.
            // Drafts are verified exactly, so this can change acceptance but never the output
            // distribution.
            const auto& moe = std::get<MoeWeights>(weights.mtp->layer.ffn);
            for (const auto& expert : moe.experts) {
                (void)bindings.transcode(expert.gate, QType::Q4_G64_FP16, "--mtp-experts-q4");
                (void)bindings.transcode(expert.up, QType::Q4_G64_FP16, "--mtp-experts-q4");
                (void)bindings.transcode(expert.down, QType::Q6_G64_FP16, "--mtp-experts-q4");
            }
        }
    }
}

void apply_device_layouts(Bindings& bindings, const ModelWeights& weights) {
#if defined(NINFER_SM70)
    // sm_70's projection routes (ops/linear/sm70) read row-split codes in the panel layout, where
    // one 128-byte line holds a k-group's records of four rows: row-major records cost the decode
    // reads 10 to 40% of the card's bandwidth. The text layers' attention, GDN and dense MLP
    // projections and the MTP input projection take it in the formats those routes read in every
    // Op that consumes them (T2, Q2, Q4, Q5, Q6; Q8 also reaches an Op without them). Experts,
    // heads, embeddings, the MTP attention and drafter weights keep the stored layout, and so does
    // an object another parameter also binds.
    std::vector<bool> selected(bindings.weights.size(), false);
    std::vector<WeightId> projections;
    const auto take = [&](WeightId id) {
        if (id.index < selected.size() && !selected[id.index]) {
            selected[id.index] = true;
            projections.push_back(id);
        }
    };
    const auto dense = [&](const DenseWeights& d) {
        take(d.gate);
        take(d.up);
        take(d.down);
    };
    const auto ffn = [&](const BlockWeights& block) {
        if (const auto* d = std::get_if<DenseWeights>(&block.ffn)) {
            dense(*d);
        } else {
            dense(std::get<MoeWeights>(block.ffn).shared);
        }
    };
    for (const auto& block : weights.text.layers) {
        if (const auto* a = std::get_if<AttentionWeights>(&block.mixer)) {
            for (const WeightId id : {a->query, a->key, a->gate, a->value, a->output}) { take(id); }
        } else {
            const auto& g = std::get<GdnWeights>(block.mixer);
            for (const WeightId id : {g.query, g.key, g.value, g.z, g.output}) { take(id); }
        }
        ffn(block);
    }
    if (weights.mtp) {
        take(weights.mtp->input_projection);
        ffn(weights.mtp->layer);
    }
    std::unordered_map<std::size_t, bool> only_projections;
    for (std::size_t i = 0; i < bindings.weights.size(); ++i) {
        for (const auto& part : bindings.weights[i].reference.binding.parts) {
            const auto [entry, inserted] = only_projections.emplace(part.object.index, selected[i]);
            if (!inserted) { entry->second = entry->second && selected[i]; }
        }
    }
    for (const WeightId id : projections) {
        const auto& parameter = bindings.at(id);
        if (parameter.reference.residency != artifact::Residency::Device) { continue; }
        for (const auto& part : parameter.reference.binding.parts) {
            const auto& geometry = bindings.binder.reader().geometry(part.object);
            const bool format =
                geometry.format == QType::T2_G128_FP16 || geometry.format == QType::Q2_G64_FP16 ||
                geometry.format == QType::Q4_G64_FP16 || geometry.format == QType::Q5_G64_FP16 ||
                geometry.format == QType::Q6_G64_FP16;
            if (format && only_projections.at(part.object.index) &&
                geometry.layout == QuantLayout::RowSplit &&
                artifact::row_split_panel_supported(geometry)) {
                bindings.binder.panel_device(part.object);
            }
            // 本仓库 V100 增量（docs/v100/PLAN.md）：NVFP4 的 MLP 投影（gate_up 34816x5120 与
            // down 5120x{6144,17408}）在加载时重排成 Volta QPN 的片段序。sm_70 上这两个形状
            // 的全部消费路径都认这个布局：t <= 32 走 QPN（拆分/残差），更宽走 CUTLASS sm_70。
            // 形状过滤是必要的：其余 NVFP4（如注意力投影）仍走 sm_70 的 SIMT 解码路线。
            const bool mlp_shape =
                geometry.shape.size() == 2 &&
                ((geometry.shape[0] == 34816 && geometry.shape[1] == 5120) ||
                 (geometry.shape[0] == 5120 &&
                  (geometry.shape[1] == 6144 || geometry.shape[1] == 17408)));
            if (geometry.format == QType::NVFP4 && mlp_shape &&
                artifact::nvfp4_qpn_prepack_supported(geometry)) {
                bindings.binder.qpn_prepack_device(part.object);
            }
            // 本仓库 V100 增量：FP8 的投影同理在 sm_70 上由 QPN（t <= 32）与 CUTLASS（更宽）
            // 拥有全部宽度：注意力的融合输入投影（14336x5120）、输出投影（5120x6144）、GDN
            // 输入投影（16384x5120）与残差形状（5120x17408，走 linear_add）。
            //
            // 不在名单里的：FP8 的 SwiGLU gate_up（34816x5120）—— 它的宽 T 交回上游阶梯（那条
            // 路线要求激活之前是 FP32，且按行主序读），词汇表/嵌入同理（嵌入是按行 gather，而
            // 输出头虽然安全却与嵌入同形状，无法用形状区分，收益也只有一轮 2.3 ms 里的一点）。
            // GDN 的 16384x5120 不在名单里：这个模型里它的消费者是 fused 的
            // fp8_gdn_input_proj_conv_record（按行主序读，会明确拒绝预打包布局）。
            const bool fp8_projection =
                geometry.shape.size() == 2 &&
                ((geometry.shape[0] == 14336 && geometry.shape[1] == 5120) ||
                 (geometry.shape[0] == 5120 &&
                  (geometry.shape[1] == 6144 || geometry.shape[1] == 17408)));
            if (geometry.format == QType::FP8_E4M3FN_ROW_BF16 && fp8_projection &&
                only_projections.at(part.object.index) &&
                artifact::fp8_qpn_prepack_supported(geometry)) {
                bindings.binder.qpn_prepack_device(part.object);
            }
        }
    }
#else
    (void)bindings;
    (void)weights;
#endif
}

} // namespace ninfer::models::qwen3_5::loading
