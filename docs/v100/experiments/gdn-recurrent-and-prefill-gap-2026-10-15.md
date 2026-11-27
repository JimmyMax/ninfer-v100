# GDN prefill route on sm_70 + the remaining prefill gap (2026-10-15)

本仓库 V100 增量。本轮目标：**找出与上游 master 的性能差距并尽量追平**。用 kernel 级剖析（nsys）
定位到差距的确切位置，修掉其中的 GDN 一项，并给出剩余差距的量化证据。

## 1. 方法（可复现）

- 逐算子对账：同一 bench 源码在两棵树上分别构建，跑同一形状/宽度。
  - master 参考构建：`/mnt/data/ninfer-build-master`（容器 `ninfer-master-bench`），源
    `/home/max/code/ninfer-v3-v100`。
  - 本线：`/mnt/data/ninfer-build-gemma4`（容器 `ninfer-sm70-build`），`ninfer_benches <name>`。
- kernel 级剖析：`nsys profile --trace=cuda --cuda-graph-trace=node`。
  **图执行的子内核必须带 `--cuda-graph-trace=node`**，否则只会看到一条 `GRAPH_TRACE` 事件。
  剖析结束时用 `pkill -x <进程名>`（精确名）终止目标——用 `pgrep -f ninfer-serve` 会**先匹配到
  nsys 自身**（它的命令行含目标路径），把 nsys 杀掉会导致报告写不出来，并留下占显存的孤儿进程。
  这正是此前数次“采不到 kernel 事件”的原因。
- e2e：同请求（27,217 token）、同 CLI、GPU 空载，背靠背跑两个二进制。

## 2. 已修：sm_70 的 GDN prefill 应走 recurrent 内核

上游 master 的 sm_70 构建在 GDN 里**强制 `T_full = 0`**（`gated_delta_net.cpp`：chunked 管道需要
Ampere+ 的 `mma.sync`），即整段 prompt 交给单内核 `launch_recurrent_inout`。本线此前把
chunked/two-stage 管道移植到了 sm_70，但在这块卡上是负优化：

`ninfer_gated_delta_net_bench`（cold L2、同 geometry、T=1024..8192）：

| T | 本线 chunked/two-stage | recurrent | 提速 |
|---|---|---|---|
| 1024 | 3885 µs | 1338 µs | 2.90× |
| 2048 | 7750 µs | 2436 µs | 3.18× |
| 4096 | 15872 µs | 4774 µs | 3.32× |
| 8192 | 31516 µs | 9604 µs | 3.28× |

修复（`src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp`）：`NINFER_SM70` 下把整个宽度交给
`launch_recurrent_inout`（就是 decode 用的那个内核，状态留在寄存器、q/k 在核内归一化、零 workspace），
`NINFER_GDN_RECURRENT=0` 可回退旧管道做 A/B。

验收：`ctest -R "gated_delta_net|gdn"` **22/22 通过**；e2e 27K prefill **873.9 → 903.9 tok/s（+3.4%）**；
27K prefill 的 GPU busy 里 GDN 从 **3.75s → 2.18s**。

## 3. 剩余差距（唯一的主要项）：prefill 注意力

27K prefill 的 kernel 预算（GPU busy，含请求尾部）：

| 内核 | master | 本线 |
|---|---|---|
| CUTLASS GEMM（nvfp4/fp8 宽 T） | 16.14s（3840 次） | 16.32s（4096 次） | 
| **prefill 注意力** | **3.93s（432 次）** | **7.28s（290 次）** |
| GDN | 2.82s（912 次，recurrent） | **2.18s（912 次）** |
| dequant nvfp4 / fp8 | 0.91s / 0.77s | 1.23s / 0.93s |
| **合计** | **28.60s** | **31.93s** |

- 注意力：master 用 llama.cpp 的 **`flash_attn_ext_f16<256,256,32,2>`**（vendored
  `third_party/llama_cpp_fattn`，由 `src/ops/launcher/gqa_attention_volta_flash.cu` 驱动：paged int8 KV
  → gather 成连续 FP16 → 逐 Q-block 跑 flash 内核）。本线这条线**没有**该路线，prefill 走自研
  `prompt_kernel<PagedKVBatchMetadata<false>,...>`：每 token **1.85× 慢**，是全部残差的来源。
- GDN：本线的 recurrent 内核比 master 的还快 23%（master 的 `recurrent_bf16_direct_kernel<true>` 无
  FP32 状态特化）。
- CUTLASS：两边持平（±1%），本线的精确解码（见 `PLAN.md` 的数值契约）在宽 T 下不构成代价。
- 因此 e2e 27K prefill：master 1010 tok/s vs 本线 904 tok/s（−10.5%），TTFT 27.1s vs 30.2s。
- 27K 的 decode：75.1 vs 80.8 tok/s。差异来自投机接受率（本线 mixed speculation 50.6%/62.5% 对齐的
  MTP 接受率低于 master 的纯 MTP），不是算子效率（逐算子 bench 本线全面更快：注意力 W=8/16/32 快
  1.8×/1.8×/8.6×，plain linear 宽 T 快 1.6–14×）。

## 4. 结论与下一步

- 短上下文（生产配置，MTP4）：本线 **+38%**（87.0 vs 62.9 tok/s）。
- 27K prefill：本线仍落后 10.5%，**唯一原因是缺少 `volta_flash` prefill 注意力路线**（3.35s GPU busy）。
- 下一步（按预期收益）：把 master 的 `gqa_attention_volta_flash.cu` + 路由/workspace 接进本线的
  `causal_softmax_attention.cpp`（本线缺少的依赖只有 `round_up_keys` 等小 helper），预计把 prefill
  拉回 ~1010 tok/s 一线。
