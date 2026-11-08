# EXPERIMENT-11 复盘：prefill 瓶颈的实测修正

日期：2026-11-07
方法：`nsys profile` 全 kernel 时间分布（2048-token prefill，RTX 2080 Ti @ 250 W）

## 结论先行

**第 10 号实验建立的"prefill 瓶颈 = cuBLAS GEMM occupancy 25%"这一前提是错的。**
实测 GEMM 已跑到约 79% 算力效率，不是瓶颈。真正的 26% 时间花在 GEMM 之外，
其中 `chunk_recurrence` 一项就占 10%，而其算力效率仅约 0.8%（纯延迟受限）。

因此 EXPERIMENT-11 的 CUTLASS 改写、手动 split-K 两条路线**从方向选择上就不可能赢**，
实测也确实都输了。这不是实现问题，是前提问题。

## 实测时间分布（`-pg 2048,16 -r 1`，含 1 次 warmup + 1 次测量，故绝对值约为单次 prefill 的 2 倍）

| 占比 | 总量(ms) | 调用数 | 单次(us) | kernel |
|---|---|---|---|---|
| 43.3% | 2491.6 | 1856 | 1342 | `turing_s1688gemm_fp16_128x256_ldg8_tn` |
| 21.1% | 1215.0 | 864 | 1406 | `turing_s1688gemm_fp16_256x128_ldg8_tn` |
| **10.0%** | **576.2** | **96** | **6002** | **`gated_delta_net::two_stage::chunk_recurrence_kernel<32>`** |
| 9.3% | 533.1 | 608 | 877 | `turing_s1688gemm_fp16_128x128_ldg8_tn` |
| 3.5% | 201.1 | 2560 | 79 | `gguf::detail::store_plane_kernel` |
| 1.8% | 104.6 | 1158 | 90 | `vec::slice_dequantize_kernel<ggml_type 21,__half>` (IQ3_S) |
| 1.5% | 86.4 | 32 | 2700 | `causal_attention_prompt_i8_fast` |
| 1.4% | 79.4 | 778 | 102 | `vec::slice_dequantize_kernel<23,__half>` |
| 1.2% | 69.8 | 512 | 136 | `sm70::convert_activation_kernel` |
| 1.2% | 67.6 | 740 | 91 | `vec::slice_dequantize_kernel<18,__half>` |
| 1.0% | 59.4 | 96 | 618 | `bf16_gdn_gating_proj_gemm_mma_kernel` |
| 0.7% | 38.8 | 96 | 404 | `gated_delta_net::two_stage::prepare_kernel` |
| 0.5% | 30.0 | 96 | 312 | `causal_conv1d_prefill_pairs_kernel` |

聚合：**GEMM 73.7%，非 GEMM 26.3%**。

## 为什么 GEMM 已经不是瓶颈

2080 Ti FP16 Tensor Core 峰值 65.7 TFLOPS。27B 模型 2048-token prefill：

```
FLOPs = 2 × 27e9 × 2048 = 1.106e14
理论时间 = 1.106e14 / 65.7e12 = 1683 ms
实测 GEMM = (2491.6 + 1215.0 + 533.1) / 2 = 2120 ms   ← 除以 2 去掉 warmup
效率 = 1683 / 2120 = 79%
```

反推总时间 2120 / 0.737 = 2876 ms → 2048 / 2.876 = 712 tok/s，与 bench 实测 691 tok/s 吻合。

cuBLAS 的实际选型也说明它没有"只用一个大 tile 卡在 25% occupancy"：

| grid (X×Y×Z) | 调用 | ms | 含义 |
|---|---|---|---|
| 11×16×4 | 256 | 173 | rows=1408，split-K=4 |
| 19×16×2 | 128 | 158 | rows=2432，split-K=2 |
| 21×16×1 | 128 | 173 | rows=2688，split-K=1 |
| 5×16×4 | 96 | 29 | rows=640，split-K=4 |

Y=16 恒为 `tokens/128=16`，Z 即 split-K。**cuBLAS 本来就在做 split-K**，
而且会按 `rows` 大小在 128×128 / 128×256 / 256×128 之间切换。

于是三条"改进 GEMM"的路线全部撞墙：
- CUTLASS 128×128 无 split-K → grid 只有 136–400 blocks，慢到挂起；
- CUTLASS split-K → sm_75 上 `initialize` 返回 `kErrorWorkspaceNull`；
- 手动 split-K（分段 cuBLAS + reduce）→ 每段 grid 反而更小，且 partial 缓冲
  `S × rows × tokens × 4B` 带来 S 倍 DRAM 写读，实测 691→650→610→539 tok/s 单调变差。

## 真正的大头：chunk_recurrence（287 ms / 单次 prefill）

```cpp
// launch.h
inline constexpr int kChunkSize = 16;      // 2048 tokens ⇒ 128 个 chunk
// recurrence.cu
grid = args.value_heads * (kStateDim / DV);   // = 96 blocks
// kernel 内部：for (chunk = 0; chunk < chunks; ++chunk)  ← 完全串行
```

- grid 仅 96 blocks，68 SM × 2 blocks = 136 容量 ⇒ 0.7 波，且尾部只有 48 个 SM 有活。
- 每个 block 串行推进 128 个 chunk，chunk 间通过 `state` 依赖。
- 单次 6.0 ms / 128 chunk = 47 µs 每 chunk。

而每 chunk 的算术量（kStateDim=128、DV=32 量级）约 2.6e5 FLOP，
在 2080 Ti 上应 <1 µs。即 **47× 的串行延迟开销**。

按整模型估算：96 blocks × 128 chunks × 2.6e5 = 3.2 GFLOP，用时 576 ms（两次 prefill 的合计）
⇒ **约 0.8% 算力效率**，而 GEMM 是 79%。这是全流程里效率最低的一段。

### 可行的重构（经典 chunked linear attention 三段式）

当前是"单 kernel 串行推进"，把三件本可分离的事耦合在同一条依赖链上：

1. **chunk 内局部量**（`solve`、`mqk`、`delta`）：只依赖本 chunk 的 Q/K/V/g/beta，
   **128 个 chunk 彼此独立** ⇒ 可合并成一个 128×96 的大 grid，几乎线性加速。
2. **state prefix scan**：`S_{c+1} = gamma_c S_c + K_c^T diag(suffix) Delta_c`。
   这步才真正串行，但每 chunk 只做一次 MMA（无输出、无 solve），critical path 短一个量级。
3. **输出**：`O_c = prefix_c · (Q_c S_c^T) + Q_c Delta_c · mqk_c`，
   用的 `S_c` 由第 2 步一次性给出，**128 个 chunk 同样彼此独立并行**。

即把"128 步串行、每步都很重"改成"1 段并行 + 128 步极轻串行 + 1 段并行"。
按 GEMM 段的效率类比，287 ms 有望压到 60–100 ms 量级。

## 次要但便宜的两项

### store_plane（100 ms / 3.5%）

`gguf_linear.cpp` 的 `sm70_project`：`direct` 路径（`p.f32 && epilogue==Store`）直接写目标；
其余走 `plane`（FP32，`count × tokens`）再 `store_plane` 搬到目标。2560 次调用 / 78 µs。

这是当初引入"dequant→FP16→cuBLAS"路线时带来的额外搬运。可选做法：

- 让 cuBLAS 直接把结果输出成 BF16（`cublasGemmEx` 的 `C_type = CUDA_R_16BF`）到目标，
  省掉 FP32 plane 往返；带 epilogue 的少数 product 才保留 plane 路径。
- 或把 epilogue 融进一个轻量 post kernel，避免 `store_plane` 的独立 launch。

### convert_activation + gather_columns（34 + 9 ms）

`BF16 → FP16` 的整块转换按 `input_columns` 去重后仍要 512 次 / 136 µs。
可考虑按 K-tile 在 GEMM 的 A 装载阶段就地转换，省掉 21 MB × 多次的 DRAM 往返。

## 预期收益

| 项 | 现状 | 目标 | 省 |
|---|---|---|---|
| chunk_recurrence | 287 ms | 70–100 ms | ~200 ms |
| store_plane | 100 ms | ~20 ms | ~80 ms |
| convert/gather | 43 ms | ~15 ms | ~28 ms |
| **合计** | 2876 ms | ~2570 ms | **~11%** |

⇒ prefill 691 → **~775 tok/s**（若 chunk_recurrence 能达到上限则 ~820 tok/s）。
GEMM 段不动。

## 教训

1. **优化前必须先 profile，不能从"理论 occupancy"推瓶颈。** 本仓库第 10 号实验的
   "238 regs / 65 KB smem / 25% occupancy"推断了方向，但没有任何 ncu 实测支撑，
   导致第 11 号实验整体跑偏。
2. **cuBLAS 是自适应选型的**（tile 切换 + split-K），在 M×N tiles 只有几百个的中等
   矩阵上已经接近峰值；想用"更小的 tile 提高 occupancy"去赢它是方向性错误。
3. **占比 10% 的一项，效率只有 0.8%，比占比 74% 的一项，效率 79%，更值得动。**
