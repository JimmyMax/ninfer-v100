# EXPERIMENT-12: GDN prefill 路径 A/B（two_stage vs chunked vs recurrent）

日期：2026-11-07，RTX 2080 Ti @ 250 W，swift15_iq3s_mtp（11 GB），INT8 KV，gdn-state-fp16

## 背景
EXPERIMENT-11 review 显示 `chunk_recurrence` 占 prefill 10%（288 ms/prefill，grid 仅 96 blocks、
128 chunk 串行）。本实验测试两条替代路径能否更快。

## 三条路径 bench（2048/8192/32768-token prefill，tok/s）

| 路径 | pp2048 | pp8192 | pp32768 |
|---|---|---|---|
| two_stage（默认） | 708.1 | 654.7 | 576.0 |
| chunked（NINFER_GDN_TWO_STAGE=0） | 635.8 | — | — |
| recurrent（NINFER_GDN_RECURRENT=1） | 705.7 | 657.9 | 577.4 |

**结论：recurrent ≈ two_stage（打平），chunked 明显更慢。**

## 为什么 recurrent 没有更快

虽然 chunk_recurrence 单独占 288 ms/prefill，但它是大 kernel（96 blocks × 6 ms），
在 68 SM 上能与 GEMM（73.7% 时间）重叠执行。recurrent 路径（width-wide kernel，无
chunk 串行）虽然避免了串行延迟，但它的 kernel 结构与 two_stage 的总工作量相同，
因此 wall-clock 打平。chunk_recurrence 的"串行"并没有成为 wall-clock 瓶颈——
它是 latency-bound 但足够宽，能 hiding 在 GEMM 后面。

## 质量验证

greedy 200-token 生成，4 个不同 prompt（量子计算/快速排序/诗歌/TCP-UDP），
two_stage vs recurrent **逐字一致**。NINFER_GDN_RECURRENT=1 保留为可选开关。

## 教训

1. "单 kernel 占比高"不等于"它是 wall-clock 瓶颈"——大 kernel 能与其他 kernel 重叠。
2. 判断瓶颈要用关键路径分析（nsys 时间线），不能只看 kernel 时间汇总。
3. chunk_recurrence 的真正问题不是串行，而是它必须存在（GDN 的固有串行状态传递）。
