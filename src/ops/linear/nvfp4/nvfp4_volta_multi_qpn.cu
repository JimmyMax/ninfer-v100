// 本仓库 V100 增量（docs/v100/PLAN.md）：NVFP4 注意力输入（14336x5120）与 GDN 输入
// （16384x5120）在 sm_70 上的解码投影走 Volta QPN，与 MLP 投影同一个内核，只是输出分成
// 多个目的地（q/gate/k/v 与 qkv/z）。此前这两个形状走 sm_70 的 SIMT 解码路线，实测
// 纯解码比 v1.5 线慢 12%（docs/v100/v1.6-vs-v1.5-2026-10-20.md）。
//
// QPN 内核自身接受存放布局（BlockScaleK16M128x4）与加载期重排后的片段序，所以这里不需要
// 改动加载期：宽 T 的 prefill 路线继续读存放布局，互不影响。

#include "ops/linear/nvfp4/nvfp4_launch.h"

#if defined(NINFER_TIER_VOLTA)
#    include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#    include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#    include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

namespace ninfer::ops::detail {

void launch_nvfp4_attn_input_volta_qpn(const Tensor& x, const Weight& w, Tensor& q, Tensor& gate,
                                       Tensor& k, Tensor& v, cudaStream_t stream) {
    const unified::Nvfp4AttentionInputOutput output{static_cast<__nv_bfloat16*>(q.data),
                                                    static_cast<__nv_bfloat16*>(k.data),
                                                    static_cast<__nv_bfloat16*>(gate.data),
                                                    static_cast<__nv_bfloat16*>(v.data)};
    launch_nvfp4_volta_qpn_with_output(x, w, output, w.n, 1.0F / w.weight_scale_divisor, stream);
}

void launch_nvfp4_gdn_input_volta_qpn(const Tensor& x, const Weight& w, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    const Nvfp4GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                     static_cast<__nv_bfloat16*>(z.data)};
    launch_nvfp4_volta_qpn_with_output(x, w, output, w.n, 1.0F / w.weight_scale_divisor, stream);
}

} // namespace ninfer::ops::detail
#endif // NINFER_TIER_VOLTA
