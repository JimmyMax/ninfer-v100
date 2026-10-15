# 追平 master 的差距收敛（2026-10-15）

目标：找出本线（上游 sm_70 线 + 本仓库 V100 增量）与 master（`ninfer:v1.5.0`，即仓库
`master` 分支的构建）的性能差距，并尽力追平。本文记录差距定位、四项落地优化、最终对照，
以及仍未消除的 prefill 残差及其证据。

## 1. 差距定位（先量，再改）

**结论先行**：本线落后的两块是 **(a) 解码验证遍里的 FP8 GDN 输入投影** 与
**(b) 所有 FP8 QPN 投影的激活处理路径**；前者是单点灾难，后者是均匀的 1.4×。长上下文
prefill 仍差 14%，根因与 master 的 CUTLASS 覆盖面/解码策略有关（见 §4）。

定位手段：
* `nsys profile --trace=cuda --cuda-graph-trace=node`（**必须加 node**，否则图内子 kernel
  不落表，会误判成"GPU 空闲 77 ms"）。ncu 在本机因驱动限制不可用
  （`ERR_NVGPUCTRPERM`），nsys 不受限。
* 算子隔离 bench：`ninfer_linear_bench --qtype FP8 --prepack [--fp16-act]`（后者为本轮
  新增的对照开关，直接调用 QPN 的 fp16 激活变体，绕开 op 分派）。

关键观测：
1. `fp8_a16_ksplit_mma_kernel` 在解码窗口里 48 次/轮、**42.8 ms/轮**（占 77 ms 轮的 56%），
   每次 0.89 ms 读 84 MiB 权重 ≈ 10% 带宽；同一形状融合核 0.40 ms、QPN 0.30 ms。
   上游把 `b1_a16_plan` 的 W=4..6 交给物化路线，5..6 正好落到这个核上。
2. FP8 QPN 在同形状、同 grid/block、同预打包布局下比 master 慢 **1.40–1.58×**
   （中位数：14336×5120 137.6 vs 97.2 µs；16384×5120、248320×5120 同比例；分布 ±2%，
   非噪声）。bench 隔离出唯一变量：**激活类型**。

   | 形状 | bf16 激活 | 先暂存 fp16 |
   |---|---:|---:|
   | 14336×5120 T=5 | 544.7 GB/s | **741.2** |
   | 16384×5120 T=5 | 547.7 | **767.9** |
   | 5120×6144 T=5 | 522.7 | **685.3** |
   | 5120×17408 T=5 | 501.6 | **758.9** |
   | 14336×5120 T=1 | 574.0 | **763.3** |
   | 16384×5120 T=1 | 635.6 | **788.4** |
   | 248320×5120 T=1（词表） | 716.6 | 711.7（无收益，跳过） |

   master 的构建每轮有约 190 次 `fp8_stage_bf16_activation_kernel` ≈ 3 次/层，正是这三类
   投影各暂存一次；我们此前传 bf16，由内核逐元素转换。

## 2. 落地优化（各自可验收）

### 2.1 FP8 GDN 输入投影走上 QPN（单点最大收益）

* `fp8_gdn_input_a16_dispatch`：预打包权重且 `t<=32` → `launch_fp8_gdn_input_volta_qpn`；
  更宽仍交旧阶梯。
* `b1_a16_plan`（主与 unified 两处）：sm_70 上一律物化——预打包布局只有 QPN（t≤32）与
  CUTLASS（t≥33）读得对，融合核按行主序读会读到错数据。
* 加载时把 16384×5120 纳入 FP8 QPN prepack（`storage_trades.cpp`）。
* 收益：T=5 每层 909 → 295 µs（3.1×）；固定 d=4 端到端 **28.5 → 44.1 tok/s（+55%）**。

### 2.2 输出头（248320×5120）纳入 prepack

* 头此前一直走行主序（2.5 ms/次），master 是预打包（1.61 ms/次）。按**权重角色**而非形状
  过滤（词表嵌入同形状但不在 `projections` 名单里）把它加入 `take(...)`，收益约 1 ms/轮。

### 2.3 FP8 QPN 的 FP16 激活暂存（三类投影）

* attn 输入投影、plain linear：从各自 arena 借暂存区，容量函数按逐宽度精确建模
  （t=1 为 0；2..32 为 k×t 的 fp16；≥33 为 CUTLASS 自带暂存）。
* GDN 输入投影：**不占 arena**。容量函数无法得知权重布局，所以用一块 320 KiB 的进程级固定
  缓冲，**在模型加载时预留**（`apply_device_layouts`），运行期只读指针——图捕获中绝不
  `cudaMalloc`（否则 `cudaErrorStreamCaptureInvalidated`，本轮踩过并据此改掉）。
* 词表形状（n≥100000）无收益，不暂存。

### 2.4 小 T 分裂折叠的越界修复（正确性，非性能）

移植"长上下文 560 splits + 256 折叠"时，三处 reduce 的 `weights` 共享缓冲仍是 `[256]`，
而折叠路径会写到 `splits-1`（`d256-h16-kv2` 在 65536 keys 下 273 > 256）→ 越界写、数值
崩坏（oracle 实测 actual=28.875 vs reference=-0.0019）。三处都改为按几何上限
`Geometry::SmallTMaximumSplits` 分配后，`ninfer_softmax_attention*_test` 全绿。

## 3. 最终对照（干净、单实例、同一 harness）

| 场景 | 本线 `ninfer:v100-parity2` | master `ninfer:v1.5.0` | 差 |
|---|---:|---:|---:|
| 短·无投机 | 34.1 tok/s（29.33 ms/轮） | 34.8（28.74 ms） | +2.1% |
| 短·MTP4 | **87.0**（38.02 ms，接受 61.5%） | 62.9（37.98 ms，接受 38.9%） | **+38%** |
| 27K·无投机 | 31.6（31.65 ms） | 32.3（30.96 ms） | +2.2% |
| 27K·MTP4 | 75.6（41.30 ms，接受 52.0%） | 78.3（41.21 ms，接受 54.6%） | 每轮持平，tok/s −3.5% |
| 27K prefill | 876 tok/s（TTFT 31.1 s） | 1020（26.8 s） | −14% |

**解码每遍成本已在四个配置上与 master 持平（0.2–2.2%）**；tok/s 的差异来自 MTP 接受率
（短上下文我们高得多，长上下文略低）。生产（`docker-compose.nvfp.yml`）已切到
`ninfer:v100-parity2` 并实测正常。

## 4. 未消除的残差：27K prefill（−14%）

本线 27K prefill 的 GPU 忙时 31.7 s 构成：CUTLASS 宽 T 15.6 s（49%，NVFP4 MLP 约 12.3 s +
FP8 attn/GDN 输入 3.3 s）、本线 sm70 prompt 注意力 6.92 s（22%）、GDN two_stage 3.74 s、
反量化 1.87 s、其余 ~3.5 s。master 为 27.6 s。

已排除：
* **chunk 粒度**：1024/2048/4096/8192 → 850/878/855/590 tok/s，2048 已是最优。
* **prompt 开关**：`NINFER_PROMPT_PACK_GQA`、`NINFER_PROMPT_PV_F16`、`NINFER_PROMPT_FAST`
  五个组合全部 869–872 tok/s（无差异）。
* **prompt 内核选择**：把 master 的 `prompt_volta.cuh` 移植进来 A/B，其 quadratic 形态
  （一 CTA 一行、全 key 扫描）在 27K 下只剩 87.8 tok/s；本线 sm70 prompt 内核 872 tok/s，
  通用阶梯 473 tok/s —— 本线内核才是对的。

仍在的两条线索（需要成规模的移植/调优，非本次预算内可闭环）：
1. **宽 T CUTLASS 覆盖面**：master 的构建引用了 8 个我们树里没有的 `*_cutlass_sm70` 实现
   （`linear/bf16`、`attn_input/{bf16,q4_q5}`、`gdn_input/{q4_q5,w8}`、`linear_add/q5`、
   `linear_swiglu/{fp8,q4}`）；宽度不占优的那些形状在我们这边走行主序阶梯。
   （注：`fp8_linear_swiglu_cutlass_sm70.cu` 我们树里有、也参与编译，但**没有调用者**。）
2. **反量化路径的取舍**：master 的 NVFP4 反量化用 fp16 移位技巧（快），本线为满足"判据只
   属于激活档位、与实现路径无关"的契约改成 fp32 精确解码（慢、但必要，见
   `nvfp4_cutlass_sm70.cu` 头部注释）；两者在同一 tile 配置（128×128×32，2 stages）下跑。

## 5. 方法学备注

* 图执行的 kernel 必须 `--cuda-graph-trace=node` 才会出现在
  `CUPTI_ACTIVITY_KIND_KERNEL` 里；否则只能看到 `GRAPH_TRACE`。
* 端到端请求要从容器内或容器 IP 发：服务绑 `0.0.0.0` 时宿主 curl 要 `--noproxy '*'`，
  否则被宿主 HTTP 代理劫持成 502。
* 两个 harness 并行会互相抢卡（轮次从 38 ms 掉到 53 ms）——测性能前必须确认只有单实例在跑；
  另外 `pkill -f ab_perround` 会匹配到自己的命令行，用 `pgrep` 取 PID 再杀。
* fork 构建的二进制在 nsys 下采集 CUPTI 事件不稳定（多次只收到 1 个事件），master 的
  prefill 侧 kernel 级归因因此未能取到。
