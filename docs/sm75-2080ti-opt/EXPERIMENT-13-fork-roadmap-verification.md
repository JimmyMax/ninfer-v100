# EXPERIMENT-13：对照 2080ti-22g fork 路线图逐项验证（#2 #3 #4）

日期：2026-11-07，RTX 2080 Ti @ 250 W

## 背景
EXPERIMENT-12 review 后按 fork 路线图剩余项逐项验证。每项先用 profile 确认是否适用，
不适用则记录原因，避免重复无效工作。

## #4 GDN T=1 staged GEMV → direct SIMT（fork P2，+3.3% decode）
**不适用。** fork 的 P2 针对其 Q5 `launch_q5_t1` staged shared-memory GEMV。
我们的模型是 GGUF IQ3_S，decode 的 GDN input 走 `gguf::vec::kernel` 系列。
Decode profile（256 tok）显示：
- **~85% kernel 时间是 IQ3_S/Q4_K/IQ3_S vec GEMV 投影**（已知 LSU-bound，30 tok/s 天花板）
- GDN 相关 kernel（recurrent_batch_update 7.4µs、projected_conv 3.3µs）合计 <2%
- `quantize_vector_kernel` 519 次/step × 2.4µs ≈ 4%（fork P7 测过类似融合上限 1-2%，不划算）

## #2 INT8-KV attention 64×32 tile（fork P3，+9% @64k）
**已等价存在。** fork 的改动是 query rows/CTA 32→64。我们的
`CausalPromptI8FastShape` 在 Turing 上（8-warp 放不下 64KB）已选 4-warp = **64 rows/CTA**
（nsys 确认 grid block=128=4 warps）。64k prefill profile：attention 占 27.2%
（73.5ms/层/chunk），是 KV 带宽固有成本。

进一步减半 KV 读放大的唯一途径是 8-warp=128 rows/CTA，需要 smem ≤64KB：
- int8 K+V 8-warp 需要 66,560 B → 放不下（static_assert 已验证）
- 唯一放法：`PackedValues`（V 用 int4，`--kv-dtype rk8v4`）→ **有质量损失，违反零损失原则，不做**

## #3 store_plane 消除（预估 +3-4% prefill）
**不值得。** 实测 store_plane 每次 ~39MB @ ~500 GB/s，已接近 2080 Ti roofline。
可省的只有 FP32 plane 的写读往返（GEMM 必须输出 FP32，cuBLAS 不支持 FP16→BF16 out，
实测 `CUBLAS_STATUS_NOT_SUPPORTED`）。可回收 ~1%，改动风险大于收益。

## 最终结论

| 路线图项 | 结果 |
|---|---|
| chunk_recurrence 三段式/recurrent | 打平，greedy 一致，保留开关（EXPERIMENT-12） |
| #4 GDN T=1 SIMT | 不适用（GGUF 路径无对应 kernel） |
| #2 attention 64×32 | 已等价存在；64k attention 27% 为无损格式下的带宽下限 |
| #3 store_plane | 近 roofline，可回收 ~1%，不做 |

**2080 Ti 上本栈（GGUF IQ3_S + cuBLAS + spec decode）的优化空间已收敛：**
- prefill：GEMM 79% 效率，非 GEMM 26% 经逐项验证均接近各自 roofline 或为固有串行
- decode：LSU-bound 30 tok/s 天花板，spec decode（已调优 128.9 bench）是唯一杠杆
- 长 context：attention KV 带宽下限（无损格式），rk8v4 可破但牺牲质量

剩余可选项均涉及质量权衡（rk8v4 KV、Q6 head、keep-list head）或收益 <2%
（小 kernel 融合），不符合零损失原则或投入产出比。
