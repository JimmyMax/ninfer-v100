# 实验 6：FP16-累积 GEMM（prefill 提速研究）

**状态**：进行中。**原则**：任何可测的质量损失 → 不启用（默认关闭的实验 flag）。

## 假设

Turing 张量核的 FP16-累积速率是 FP32-累积的 2×（~107 vs ~53.8 TFLOPS）。
`sm70_project` 的 cuBLAS GEMM 当前用 `CUBLAS_COMPUTE_32F`；改 `CUBLAS_COMPUTE_16F`
理论上 prefill GEMM 部分近 2×，总 prefill +25~40%。

## 风险（为什么默认必须关闭）

FP16 累积器：eps ≈ 4.9e-4，max 65504。K 最长 17408 的 dot product：
- 溢出风险低（RMSNorm 后激活 ~N(0,1)，权重量级 ~0.03，|sum| ~ O(10)）
- **精度损失是主要风险**：相对误差 ~1e-3 级，逐层传递后可能改变 argmax（greedy 下）
  或漂移采样分布。这正是 gdn-state-fp16 走过的路（官方实验背书 ≤0.34% 漂移）,
  但 GEMM 累积的影响面更大，必须实测。

## 实现

`src/ops/linear/sm70/sm70_gemm.cu::gemm()`：
- 运行时开关：环境变量 `NINFER_GEMM_F16ACC=1`（默认 0 = 现状）
- 仅 Turing 生效（`current_device_turing()`），Volta 行为不变
- `CUBLAS_COMPUTE_32F -> CUBLAS_COMPUTE_16F`，alpha/beta 相应改 half

## 验证计划（质量一票否决）

1. **贪心一致性**（快速门禁）：≥5 组 prompt（物理/代码/数学/创作/长文续写），
   400+ tokens，FP16-accum vs FP32-accum 输出必须逐字一致；任何一组不一致 → 判负。
2. **困惑度对比**（量化指标）：同一片语料（≥50K tokens），两种模式各算一次
   perplexity，相对变化 >0.5% → 判负。
3. 性能：bench pp2048/pp27000。

判定：1 或 2 不过 → flag 保持默认关闭，仅作为分支实验保留；全过 → 考虑默认开启
（仍需保守：prefill-only 路由，decode 路径不经过此 GEMM）。

## decode 提速研究（不改数值的前提下）

现状：decode 99% GPU busy，ggml vec kernel L1/TEX 91.6% 是硬墙，DRAM 49.9% 有余量。
候选（按收益/工程量排序）：

1. **LM head 转换期量化**（converter recipe override：text/output_head bf16 → Q4_K）：
   每 token 省 ~1.9GB 流量（2.5GB→0.6GB），decode 理论 +15~20%。
   数值变化 ≠ 无损，需与实验 6 同级的验证；若做，应同样走"实验 flag + 验证"流程。
   注意：GGUF 源里 head 已是 q4_k（转换日志 `proposal/head: gguf_q4_k`）——
   需先确认主 output_head 的实际存储格式与每 token 读取量（profile 证实后再立项）。
2. **vec kernel 访存模式重写**：IQ 格式的 scale/sign 分散读取是 L1 热点；
   预打包（load 时重排成 dequant 友好的布局）可在不改算术的前提下降 L1 压力。
   工程量大（每格式一套），收益 decode +10~20%。
3. **小试 nvidia-smi -lgc 锁频**（零代码）：decode 是延迟敏感型负载，
   锁 SM/显存最高频可能降波动；收益预期 <5%。
4. **draft/verify 批处理形状**：spec 轮的 verify GEMM 形状（M=drafts+1）在
   Turing tile 上是否最优，可用 ncu 看 verify 轮的内核占用，调 draft-tokens。

## 工作记录

- 2026-11-07 计划建立。
- 2026-11-07 **负结果，实验关闭**：Turing 上 FP16 输入 + FP32 累积的张量核本来就是
  全速率（~107 TFLOPS，2× 惩罚是 Ampere 消费卡才引入的），没有可榨的差值。
  - `CUBLAS_COMPUTE_16F` + FP32 输出被 cuBLAS 12.8 拒绝（NOT_SUPPORTED）。
  - `CUBLAS_COMPUTE_32F_FAST_16F` 可运行但 **704.6 vs 708.2 tok/s，无差异**。
  - 代码已回滚（flag 删除），本文档保留负结果。
  - 推论：prefill GEMM 实测 ~37 TFLOP/s ≈ Turing 峰值的 35%，瓶颈不在指令速率，
    后续 prefill 研究应转向 GEMM 形状/调度（ncu turing_s1688gemm 实例的 wave 占用）。

## 实验 7：LM head 量化 —— 前提不成立（2026-11-07）

设想：bf16 head（~2.4GB/token）→ Q4_K 省 24% 流量。
实际（artifact 目录查证）：Swift-1.5 artifact 的 `text/output_head` **已是 gguf_q4_k**
（248320×5120，715MB，GGUF 源即量化）；token_embedding 是 iq2_s（407MB）；
proposal/head 也是 q4_k（377MB）。`--lm-head-q4` 报 cannot transcode 正是因为它已是 q4_k。
head 流量占每 token 读量仅 ~6%，量化 head 无收益空间。实验关闭。

## Decode 研究结论（2026-11-07，ncu 实测）

| 指标 | vec kernel 实测 | 解读 |
|---|---|---|
| DRAM 吞吐 | **297 GB/s（48% 峰值）** | 余量 ~1.6× |
| L1/TEX 吞吐 | **91.6%** | 真正的墙 |
| L1 命中率 | 87% | 反复重读 scale/sign，事务数过多 |
| Occupancy | 理论 100% / 实测 85% | 不是瓶颈 |
| No Eligible | 65-70% | warp 全部在等 L1 |

**结论**：decode 的唯一大成因是 ggml vec kernel 的反量化访存模式（逐字节 gather
scale/sign → L1 事务爆炸）。修复 = 重写 kernel：uint4 向量化加载 + smem 预存 scale，
把每字节权重的 L1 事务数降下来。预期 decode 48%→75% DRAM ≈ **29.8 → 42-45 tok/s（+50%）**。
工程量：IQ3_S/IQ3_XXS/IQ4_XS/Q4_K 四个格式各一套（第三周级）。

**次因（已排除）：LM head**。artifact 中 head 已是 q4_k（715MB，~6% 流量），无收益空间。
decode 理论上限复核：10.9GiB/token ÷ 616GB/s ≈ 19ms → 52.6 tok/s；
实测 29.8（55%），缺口全部来自 vec kernel 的 L1 事务效率（实测 DRAM 仅 48%）。
