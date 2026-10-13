# 定位测量：4.3× 差距在 **NVFP4 投影路线**（2026-10-13）

目标是找出上游 sm_70 线在 `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4` 上比 fork 慢 4.3× 的
具体位置。方法：用上游自带的**单算子基准**（`ninfer_linear_bench`，本仓库 `bench/ops/linear_bench.cu`）
逐格式测同一形状的投影，再与 artifact 的格式分布相乘。

## 1. 方法（可复现）

```bash
# 构建（需打开基准开关）
docker exec ninfer-sm70-build bash -c 'cd /build && cmake -S /src -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON'
docker exec ninfer-sm70-build bash -c 'cd /build && ninja -j 16 ninfer_benches'   # 76 个目标

# 逐格式测同一形状（n=17408、k=5120 是本模型 MLP 的 gate/up 宽度）
docker exec ninfer-sm70-build bash -c 'cd /build && ./bench/ninfer_benches ninfer_linear_bench \
  --qtype nvfp4 --n 17408 --k 5120 --sweep 1:2:1 --repeat 10'
```

基准自报 `actual_gpu=Tesla V100-PCIE-32GB sm=70`、`dram_spec_gbs=898 sustained_read_gbs=835`、
冷缓存（`--flush-mib 256`），因此 `eff_GB/s` 与 `DRAM_%` 可直接与硬件上限比较。

## 2. 结果：只有 NVFP4 是异常值

同一形状 `n=17408, k=5120`，`--policy` 默认（a16）：

| qtype | T=1 median | 有效带宽 | DRAM 占用 | T=2 median | 判定 |
|---|---:|---:|---:|---:|---|
| **NVFP4** | **589.8 µs** | **85.1 GB/s** | **9.5 %** | 598.0 µs | **异常：远低于带宽** |
| fp8 | 143.4 µs | 622.3 GB/s | 69.3 % | 152.6 µs | 接近带宽 |
| q8 | 157.7 µs | 600.8 GB/s | 66.9 % | 170.0 µs | 接近带宽 |
| q5 | 121.9 µs | 480.2 GB/s | 53.5 % | 131.1 µs | 合理 |
| q4 | 94.2 µs | 502.9 GB/s | 56.0 % | 103.4 µs | 合理 |
| **NVFP4 @T=2048** | **6207 µs** | **22.9 GB/s** | **2.6 %** | — | **prefill 更糟** |

对照 fork 的记录（`docs/V100-PERF-PLAN.md`，`nvfp4_volta_qpn_gemm.cuh` 的 `2b3f16cd` 条目）：
**同一矩阵 T=4：96.0 → 88.4 µs**。也就是说上游的 NVFP4 路线比 fork 的 QPN 慢 **≈6.8×**。

## 3. 这个格式占多少权重（artifact 解析）

| 格式 | 张量数 | 参数量 |
|---|---:|---:|
| **nvfp4** | 128 | **17.11 G** |
| fp8_e4m3fn_row_bf16 | 130 | 9.76 G |
| q8_g32_fp16 | 28 | 2.20 G |
| q4_g64_fp16 | 55 | 0.91 G |
| bf16 | 579 | 0.22 G |
| q5_g64_fp16 | 54 | 0.17 G |

**过半权重（17.11 G / ≈30 G）是 NVFP4**，全部走那条 85 GB/s 的路。

## 4. 结论与量化预测

1. **瓶颈精确定位**：不是"整个投影层"，而是**NVFP4 这一条路线**。同形状下 fp8/q4/q5/q8 都已跑到
   480–620 GB/s（离 835 GB/s 的可持续读带宽不远），只有 NVFP4 掉到 85 GB/s（decode）
   与 22.9 GB/s（prefill）。这与上游文档自述一致：*FP8 A8 与 NVFP4 W4A4 在 V100 上不可用，
   那些权重走 A16 路线*——而它的 NVFP4 A16 路线实现很差（DRAM 占用只有 9.5%/2.6%，说明是
   计算/延迟受限，而非带宽受限）。
2. **量级对齐**：每轮解码要过 17.11 G × ≈0.5 B ≈ **8.6 GB 的 NVFP4 权重**；
   85 GB/s → **≈100 ms/轮**，而我们实测每轮约 150 ms（16 tok/s ÷ 2.4 token/轮）。即
   **NVFP4 一条路线就占了解码轮的三分之二**。把它提到 fork 的 ~500 GB/s ⇒ 17 ms/轮，
   轮耗时降到 ~65 ms ⇒ **≈37 tok/s**（其余部分不动）；再叠加 fp8/注意力的后续优化才可能到 fork 的 62。
3. **移植范围因此收窄**：先只做 **NVFP4**（fork 的
   `nvfp4_prepack_sm70.{cu,h}` + `nvfp4_volta_qpn_gemm.cuh`(503) + `nvfp4_gemv.cuh`(250)
   + `nvfp4_volta_mma_gemm.cuh`(224) + `nvfp4_small_t.cuh`(374) + dispatch/plan 接线），
   约 **8–12 个文件**，而不是原先估计的 20+；fp8/q4/q5/q8 暂时不用碰。
   prefill 侧同样受益（NVFP4 @T=2048 现在只有 22.9 GB/s）。
4. **注意**：fp8 那一族（9.76 G）在 V100 上已经是 622 GB/s，**没有首先优化的价值**；
   若 NVFP4 做完后仍与 fork 有差距，再按 fp8 → q4/q5/q8 顺序看待。

## 5. 下一步

1. 移植 NVFP4 QPN（范围见 §4.3），用已建好的 oracle（`ctest -R ninfer_softmax_attention_test`
   及 linear 相关单测）逐个验收。
2. 每步用 `ninfer_linear_bench --qtype nvfp4 --n 17408 --k 5120 --sweep 1:2:1` 做算子级对照，
   目标：589.8 µs → ~90 µs（T=1/T=2），并让 `DRAM_%` 从 9.5% 抬到 50%+。
3. 端到端验收用 `bench/v100/compare_lines.py`（同一 artifact，fork 线 62.1 tok/s 作为参照）。
