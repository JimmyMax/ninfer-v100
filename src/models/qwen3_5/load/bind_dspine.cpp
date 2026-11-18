#include "models/qwen3_5/load/bindings.h"

#include "core/dtype.h"
#include "core/weight.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::loading {

void bind_dspine_code_bridge(Bindings& b, DSpineCodeBridgeWeights& weights, const DraftConfig& config,
                             const TextConfig& target) {
    const auto& dspine = config.dspine.value();
    const auto h       = target.hidden_size;
    const auto c       = dspine.code_size;
    const auto r       = dspine.rank;

    // The DSpine bridge parameters use the "dspine/bridge/" prefix.
    weights.gate_bias = b.direct("dspine/bridge/gate_bias", {r});
    weights.projection = b.parameter(
        "dspine/bridge/projection", {3 * r, c}, {"dspine/bridge/code_input"});
    weights.sender_table = b.direct("dspine/bridge/sender_table", {target.vocab_size, c});
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const auto prefix = "dspine/bridge/layers/" + std::to_string(layer) + "/";
        weights.read_weights.push_back(
            b.parameter(prefix + "read", {c, h}, {prefix + "read_input"}));
        weights.write_weights.push_back(
            b.parameter(prefix + "write", {h, r}, {prefix + "message"}));
    }
    weights.head_write = b.direct(
        "dspine/bridge/head_write", {r, h});
    weights.code_size = c;
    weights.rank      = r;
    weights.kappa     = dspine.kappa;
    weights.eps       = dspine.eps;
}

} // namespace ninfer::models::qwen3_5::loading
