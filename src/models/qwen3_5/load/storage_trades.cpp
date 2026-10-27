#include "models/qwen3_5/load/bindings.h"

#include "artifact/permute_fp8_qpn.h"    // 本仓库 V100 增量：FP8 QPN 加载时重排
#include "artifact/permute_nvfp4_qpn.h" // 本仓库 V100 增量：NVFP4 QPN 加载时重排
#include "artifact/permute_q6k_dense.h" // 本仓库 V100 增量：Q6_K 密集 6-bit 码字流加载时重排
#include "artifact/permute_row_split.h"
#if defined(NINFER_TIER_VOLTA)
// 本仓库 V100 增量：加载时预留 GDN 输入投影的 FP16 激活暂存区（图捕获之前）。
#    include "ops/linear/fp8/fp8_launch.h"
#endif

#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
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

void apply_device_layouts(Bindings& bindings, const ModelWeights& weights,
                          std::span<const ops::DeviceTier> rank_tiers) {
    if constexpr (!ops::kBuildVolta) {
        (void)bindings;
        (void)weights;
        (void)rank_tiers;
        return;
    }
    // 本仓库 V100 增量：GDN 输入投影的 QPN 路线需要一块固定 FP16 激活暂存区（320 KiB）。
    // 它不向 arena 记账，所以在**这里**（模型加载、任何 CUDA graph 捕获之前）预留一次，
    // 运行期只读指针；捕获中绝不 cudaMalloc（否则捕获被 cudaErrorStreamCaptureInvalidated
    // 打断）。实测同形状同布局下 bf16 激活 548 GB/s vs 暂存 fp16 768 GB/s。
#if defined(NINFER_TIER_VOLTA)
    (void)ops::detail::fp8_gdn_qpn_stage_reserve();
#endif
    // sm_70's projection routes (ops/linear/sm70) read row-split codes in the panel layout, where
    // one 128-byte line holds a k-group's records of four rows: row-major records cost the decode
    // reads 10 to 40% of the card's bandwidth. The text layers' attention, GDN and dense MLP
    // projections and the MTP input projection take it in the formats those routes read in every
    // Op that consumes them (T2, Q2, Q4, Q5, Q6; Q8 also reaches an Op without them), on a rank
    // whose device runs Volta's code. Experts, heads, embeddings, the MTP attention and drafter
    // weights keep the stored layout, and so does an object another parameter also binds.
    const auto volta_rank = [&](std::size_t rank) {
        if (rank >= rank_tiers.size()) {
            throw std::logic_error("device layouts: no device tier for pipeline rank " +
                                   std::to_string(rank));
        }
        return rank_tiers[rank] == ops::DeviceTier::Volta;
    };
    std::vector<bool> selected(bindings.weights.size(), false);
    std::vector<WeightId> projections;
    const auto take = [&](WeightId id) {
        if (id.index < selected.size() && !selected[id.index]) {
            selected[id.index] = true;
            projections.push_back(id);
        }
    };
    const auto dense = [&](const DenseWeights& d) {
        // A fused [gate; up] parent (gate and up stored row-contiguously in one object, as the
        // fused-swiglu recipes do) keeps the stored layout: the dense permute is per-object and
        // the fused swiglu kernel family reads standard GGUF blocks only, so prepacking the
        // parent would break that path (gguf swiglu guards it).
        const auto& gate_parts = bindings.at(d.gate).reference.binding.parts;
        const auto& up_parts   = bindings.at(d.up).reference.binding.parts;
        const bool fused_parent =
            !gate_parts.empty() && !up_parts.empty() &&
            gate_parts.back().object == up_parts.front().object &&
            gate_parts.back().end == up_parts.front().begin &&
            bindings.binder.reader().geometry(gate_parts.back().object).format == QType::GGUF_Q6_K;
        if (!fused_parent) {
            take(d.gate);
            take(d.up);
        }
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
    // 本仓库 V100 增量：GGUF 制品的 proposal 头（MTP/DFlash 的输出头）是 Q6_K（131072x5120），
    // 每轮一次，是 MTP 解码里最大的一笔 Q6_K 读取——把它纳入 projections 名单让它走密集重排。
    // 只在格式确实是 GGUF Q6_K 时纳入：NVFP4 制品的 proposal 头是 Q4_G64，若进名单会被 panel
    // 重排，而草稿头的消费路径要求 RowSplit，重排后草稿静默失效（接受率归零）。
    if (weights.proposal) {
        bool q6k_head = false;
        for (const auto& part : bindings.at(weights.proposal->head).reference.binding.parts) {
            q6k_head |= bindings.binder.reader().geometry(part.object).format == QType::GGUF_Q6_K;
        }
        if (q6k_head) { take(weights.proposal->head); }
    }
    // 本仓库 V100 增量（docs/v100/PLAN.md）：输出头（248320x5120 FP8，每轮一次）也纳入——
    // sm_70 上 QPN 读预打包片段序比行主序快 ~1.5x（实测 1.61 ms vs 2.5 ms 一次），master
    // 也是这么做的。词表嵌入虽然同形状，但不在 projections 名单里（它按行 gather），所以
    // 不会被重排；这也是“同形状不可区分”的正确解法：按权重角色而不是按形状过滤。
    take(weights.text.output_head);
    std::unordered_map<std::size_t, bool> only_projections;
    for (std::size_t i = 0; i < bindings.weights.size(); ++i) {
        for (const auto& part : bindings.weights[i].reference.binding.parts) {
            const auto [entry, inserted] = only_projections.emplace(part.object.index, selected[i]);
            if (!inserted) { entry->second = entry->second && selected[i]; }
        }
    }
    for (const WeightId id : projections) {
        const auto& parameter = bindings.at(id);
        if (parameter.reference.residency != artifact::Residency::Device ||
            !volta_rank(bindings.weights.at(id.index).rank)) {
            continue;
        }
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
            // 路线要求激活之前是 FP32，且按行主序读）。词表嵌入同理（按行 gather）；输出头则
            // 通过 projections 名单加入（它是 FP8、每轮一次、QPN 预打包序比行主序快 1.5x）。
            // GDN 的 16384x5120 与注意力并列为两大投影之一（每层 1 个、共 48 个 GDN 层），
            // 也在名单里：它的解码宽度（W=5/6 等）此前落在只有 ~10% 带宽的 ksplit 小 MMA 上，
            // 是每轮最大的一笔开销。纳入 prepack 后由布局感知的路线独占：t<=32 走 QPN，
            // t>=33 走 CUTLASS，行主序的阶梯（SIMT/ksplit/gemm）不再被选中。
            // FP8 的 SwiGLU gate_up（34816x5120）也在名单里：内核级剖析显示，未预打包时
            // 解码走 fp8_volta_qpn_gemm_kernel<...,false,...>（294 us/次），而预打包的
            // <...,true,...> 只要 111 us/次（2.65x）——这是 1.5 线在同一张卡上更快的主要来源
            // 之一。宽 T 的 CUTLASS 路线按 w.layout 选择预打包偏移，所以 prefill 不受影响。
            const bool fp8_projection =
                geometry.shape.size() == 2 &&
                ((geometry.shape[0] == 14336 && geometry.shape[1] == 5120) ||
                 (geometry.shape[0] == 16384 && geometry.shape[1] == 5120) ||
                 (geometry.shape[0] == 248320 && geometry.shape[1] == 5120) ||
                 (geometry.shape[0] == 34816 && geometry.shape[1] == 5120) ||
                 (geometry.shape[0] == 5120 &&
                  (geometry.shape[1] == 6144 || geometry.shape[1] == 17408)));
            if (geometry.format == QType::FP8_E4M3FN_ROW_BF16 && fp8_projection &&
                only_projections.at(part.object.index) &&
                artifact::fp8_qpn_prepack_supported(geometry)) {
                bindings.binder.qpn_prepack_device(part.object);
            }
            // 本仓库 V100 增量（docs/v100/PLAN.md）：GGUF Q6_K 的投影在加载时重排成逐行密集 6-bit
            // 码字流：sm_70 的向量解码内核每个 32 值切片只读 3 个 64-bit 字而不是 16 个 2-byte
            // 字，27B 形状实测快 17–24%（~550 → ~720 GB/s）。该布局的全部消费路径都支持它：
            // t ≤ 8 走密集向量解码，t 9–64 交回 8 列分块向量路线（矩阵内核不认该布局），
            // sm70 反量化与 embedding gather 各有密集分支。MoE 专家与别的角色绑定的对象保持超块序。
            if (geometry.format == QType::GGUF_Q6_K && only_projections.at(part.object.index) &&
                artifact::q6k_dense_prepack_supported(geometry)) {
                bindings.binder.q6k_dense_device(part.object);
            }
        }
    }
}

} // namespace ninfer::models::qwen3_5::loading
