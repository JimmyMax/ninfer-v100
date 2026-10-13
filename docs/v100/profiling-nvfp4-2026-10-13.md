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

## 6. 两线完整对照与"能否追上"的量化预测（2026-10-13 补测）

同一 artifact、同一 GPU、同一参数（int8 KV、MTP 4 drafts、prefill-chunk 2048、`temperature=0`），
两线各测三个上下文。prompt 字符数相同（130k / 570k 字符），因两侧分词/文档拼接略有差异，
服务端计数分别为 42,518 与 38,084（42K 档）。

| 上下文 | 引擎 | prefill | decode | TTFT | MTP 接受率 |
|---|---|---:|---:|---:|---:|
| 短（76 tok） | 上游 gemma 线 | — | 15.9 tok/s | — | 39.4% |
| 短（76 tok） | **fork（master 线）** | — | **62.1 tok/s** | — | 32.8% |
| 42K | 上游 gemma 线（V2=1） | 360.5 tok/s | 17.2 tok/s | 1m 58s | 49.4% |
| 42K | **fork** | **960.3 tok/s** | **72.0 tok/s** | **39.7s** | 50.6% |
| 152K | 上游 gemma 线（V2=1） | 268.9 tok/s | 17.1 tok/s | 9m 28s | 51.2% |
| 152K | **fork** | **622.1 tok/s** | **52.4 tok/s** | **4m 06s** | 47.7% |

差距：decode 短上下文 3.9×、42K 4.2×、152K 3.1×；prefill 42K 2.7×、152K 2.3×；TTFT 3.0× / 2.3×。

### 按"每轮耗时"拆解（tokens/轮 = 1 + drafts × 接受率）

| | 短上下文 | 42K |
|---|---:|---:|
| 上游 轮耗时 | 162 ms | 173 ms |
| fork 轮耗时 | 37 ms | 42 ms |

上游的 NVFP4 权重读取（8.6 GB @ 85 GB/s）≈ **101 ms**，与上下文无关 → 它占了上游每轮的约 2/3。
把它提到 fork 水平（同一矩阵 88.4 µs @T=4 ⇒ ~17 ms）：

* **只做 NVFP4**：上游短上下文轮耗时 162 → ~61 ms ⇒ **≈42 tok/s**（15.9 的 2.6×，
  约为 fork 62.1 的 **68%**）；42K 约 33–35 tok/s（fork 的 ~47%）。
* prefill 侧收益更大：NVFP4 在 T=2048 只有 22.9 GB/s（6207 µs/矩阵），修好后 prefill
  有机会从 360 提到 700–900 tok/s，接近 fork 的 960。
* **要完全追平**还差 ~24 ms/轮，来源是 fp8 路线（上游 143.4 µs vs fork 的 QPN）与 fork
  在 v1.4.0 记的那批 decode 侧改动（fp16 激活暂存、NVFP4 residual 融入 QPN epilogue、
  更宽的 small-T split reduce，合计 −7~−11% 轮耗时）。也就是需要把**量化投影那一整套**
  （~30 文件）搬过来，而不只是 NVFP4 的 8–12 个。

### 结论

* NVFP4 QPN 是**收益最大、范围最小**的第一步，值得先做；算子级门禁：
  `589.8 µs → ~90 µs`、`DRAM_% 9.5% → 50%+`（decode），以及 T=2048 从 6207 µs 显著下降。
* 它能把这条线从"慢 3.9×"拉到"慢 1.5×"（decode），prefill 基本追平。
* 完全追平需要第二步：fp8 QPN + v1.4.0 那批 decode 侧端口。

## 7. NVFP4 QPN 移植：第 1 阶段（解码宽度）进行中

按 §4.3 的收窄范围动手，进程如下（2026-10-13）：

| 步骤 | 状态 |
|---|---|
| 拷入 fork 的 6 个文件并统一宏（`NINFER_VOLTA_BUILD` → `NINFER_SM70`）：`nvfp4_volta_qpn_gemm.{cuh,cu}`(503+61)、`nvfp4_volta_mma_gemm.{cuh,cu}`(224+88)、`nvfp4_prepack_sm70.{cu,h}`(75+13) | ✅ |
| 接口补齐：`QuantLayout::VoltaQpnPrepacked`（`core/weight.h`）、`Nvfp4ResidualOutput`（`nvfp4_output.cuh`）、`nvfp4_prepack_sm70.h` 补 `core/weight.h`、`nvfp4_launch.h` 增加 `NINFER_SM70` 声明块 | ✅ |
| 单文件编译探针：3 个 `.cu` 全部零错误 | ✅ |
| 新增 `nvfp4_volta_sm70.{h,cpp}`（`nvfp4_volta_takes` / `nvfp4_volta_project`），在 `linear.cpp` 的 `QType::NVFP4` 分支前挂接：解码宽度（tokens ≤ 32）且形状受 QPN 支持时走 QPN | ✅ 编译通过 |
| `sources.cmake` 加入 4 个新文件 | ✅ |
| 完整重建 + 链接 | ◐ 进行中（434 个目标） |
| 算子级门禁：`ninfer_linear_bench --qtype nvfp4 --n 17408 --k 5120 --sweep 1:2:1` 期望 589.8 µs → ~90 µs、DRAM 9.5% → 50%+ | ☐ |
| 端到端验收（短/42K/152K，`bench/v100/compare_lines.py`） | ☐ |
| 第 2 阶段：prefill 宽 T（`launch_nvfp4_volta_mma`）+ 加载时 prepack（`VoltaQpnPrepacked`，需按 `panel` 的先例走 binder demand → materializer） | ☐ |

关键设计选择：**第 1 阶段不依赖 prepack** —— fork 的 QPN 内核自带 `prepacked=false` 分支，
可直接消费 artifact 原样的 `BlockScaleK16M128x4` 布局，因此不必先改 artifact 层。
`VoltaQpnPrepacked` 与 `nvfp4_prepack_sm70.cu` 已就位，留作第 2 阶段（prepack 还能再省一截索引开销）。

## 8. 第 1 阶段结果：钩子生效，但挂错了入口（2026-10-13）

### 算子级：3.8×

| `ninfer_linear_bench --qtype nvfp4 --n 17408 --k 5120` | median | 有效带宽 | DRAM |
|---|---:|---:|---:|
| 移植前 | 589.8 µs | 85.1 GB/s | 9.5 % |
| **移植后（QPN，未 prepack）** | **154.6 µs** | **324.5 GB/s** | **36.1 %** |

即 **3.8×**、DRAM 占用从 9.5% 抬到 36.1%。门上到 50%+/90 µs 还差一截，那一段正是 fork 记录的
prepack（"single card 5120x17408 T=4: 96.0 → 88.4 µs"），留到下一步。

**踩到的坑**：第一版钩子挂在 `dispatch_linear` 的 `switch (w.qtype)` 里，但 `linear()` 在 switch
**之前**先试 `sm70_route()`，NVFP4 在 small-T 上被它截走（用 `Nvfp4Decoder`）。
修法是让 `sm70_route()` 对 QPN 权重返回 false。这个修正只改了 `linear.cpp`，重链后算子级提升立刻出现。

### 端到端：0%（入口挂错）

移植后跑真实模型（同 artifact、同参数）：短上下文 17.3 tok/s、42K 16.5 tok/s —— 与移植前
（15.9 / 17.2）持平。原因已定位，且是结构性的：

| NVFP4 张量形状 | 数量 | 参数量 | 走的分发器 |
|---|---:|---:|---|
| `(34816, 5120)` | 64 | **11.41 G** | `linear_swiglu`（gate+up 融合） |
| `(5120, 17408)` | 64 | **5.70 G** | `linear_add`（down + residual） |

**模型的 17.11 G NVFP4 参数 100% 落在两个融合投影上，没有一个走纯 `linear`。**

### 第 1b 阶段（下一步，范围已确定）

1. `linear_swiglu/nvfp4`：移植 fork 的 `nvfp4_linear_swiglu_volta_qpn.cuh`（239 行）并挂接
   （对应 11.41 G / 64 层）；
2. `linear_add/nvfp4`：挂接已移植的 `launch_nvfp4_volta_qpn_fp16_residual`
   （对应 5.70 G / 64 层）；
3. 之后再上加载时 prepack（`VoltaQpnPrepacked`，按上游 `panel` 的先例走 binder demand → materializer），
   把 154.6 µs 推向 fork 的 ~90 µs。
