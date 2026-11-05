# RTX 2080 Ti (sm_75/Turing) 优化计划

分支：`sm75-2080ti-opt`。目标：缩小与 V100 (sm_70) 官方标定路线的 prefill 差距
（当前 ~655 vs ~906 tok/s，2K prompt），并保持 decode 优势（MTP 65.5 vs 58.7 tok/s）。

## 硬件

- RTX 2080 Ti 22GB 改版（Turing TU102，68 SM，616 GB/s HBM，64KiB smem/SM）
- 驱动 595.71.05，CUDA 12.8，GTX 250W 档位
- 已有内建设备 profile（nvidia-geforce-rtx-2080-ti-sm75，70 routed keys，2026-10-16 标定）

## 基线（`ninfer_bench`，Qwen3.8-27B GSQ-RCO IQ3_S artifact，

`Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`，--kv-dtype int8）

| 配置 | prefill tok/s | decode tok/s | 备注 |
|---|---|---|---|
| chunk 2048, 2K prompt | 658.5 ± 3.1 | 29.3 ± 0.1 | 基准 |
| chunk 2048, 27K prompt | 579.4 ± 1.6 | 27.0 ± 0.1 | 长文衰减 ~12% |
| chunk 2048 + spec mtp×4 + lm-head-draft | 655.0 | 65.5 (acc 75.9%) | decode 最优 |
| + --prefill-cublas | 654.7 | 29.1 | **无效**（兼容路径不接管） |
| + --prefill-cublas, chunk 4096 | 656.9 | 29.2 | 无效 |
| + --fast-prefill-kernel | 654.5 | 29.2 | 无效 |
| + --gdn-state-fp16 | 652.1 | 29.2 | 无效 |
| + --no-prefill-a8 | 654.2 | 29.1 | 无效 |

复现：
```bash
build/bench/ninfer_bench --weights <artifact> -pg 2048,512 -r 3 \
  --kv-dtype int8 --prefill-chunk 2048 [--spec mtp --draft-tokens 4 --lm-head-draft]
```
注意：跑 bench 前必须停掉占显存的其它进程（如 ComfyUI / ninfer-serve 容器）。

## Profile（nsys, 2048-token prefill，占 GPU kernel 总时间比例）

| 占比 | kernel | ggml_type | 说明 |
|---|---|---|---|
| 27.2% | mul_mat_q<21,128> | IQ2_S | GGUF MMQ 反量化 GEMM |
| 17.4% | mul_mat_q<18,128> | IQ1_S | 同上 |
| 15.2% | mul_mat_q<23,128> | IQ1_M | 同上 |
| 9.4% | gdn chunk_recurrence_kernel<32> | — | GDN 线性注意力 prefill |
| 5.1% | mul_mat_q<12,128> | Q4_K | MMQ |
| 5.0% | mul_mat_q<22,128> | IQ4_XS | MMQ |
| 4.0% | mul_mat_q<10,64> | Q2_K | MMQ |
| 3.1% | store_plane_kernel | — | MMQ stream-k fixup plane（float） |
| 3.1% | mul_mat_q<17,128> | IQ3_XXS | MMQ |
| 1.7% | magma_sgemmEx (cuBLAS) | — | bf16 投影 GEMM |
| 1.4% | causal_attention_prompt_i8_fast | — | 全注意力 prefill |
| 其余 | quantize_matrix/vec/rmsnorm/conv1d 等 | — | <2% 每项 |

**结论：GGUF MMQ kernel（mul_mat_q 系列）合计 ~78%**，是 prefill 的绝对热点。
这些是从 ggml 移植的 SIMT dp4a kernel，Turing 上没有张量核心路径，
tile 配置沿用 ggml 上游常数（非本卡标定）。GDN 9.4% 次之。

## 优化方向（按预期收益排序）

### 0. 代码事实核查（2026-11-05，排除的错误假设）
- ~~“mul_mat_q 在 Turing 上纯 SIMT”~~ **不成立**：`common.cuh` 在
  `__CUDA_ARCH__ >= GGML_CUDA_CC_TURING(750)` 定义 `TURING_MMA_AVAILABLE`，
  `mmq.cuh: use_mma_data_layout()` 对它返回 true，kernel 走 mma.sync 数据布局。
- ~~“Ampere 配置表配 SIMT 回退”~~ **不成立**：launch config 按 cc 查
  `ggml_cuda_mmq_get_config`（`>= VOLTA` 用 ampere 表），该表的 tile/threads
  与 Turing mma 布局兼容。
- 结论：MMQ 在 2080 Ti 上已按设计运行张量核心路径。与 V100 的 prefill 差距
  （655 vs 906 tok/s）与硬件规模吻合（68 SM @1.65GHz vs 80 SM @1.4GHz +
  kernel 常数非本卡标定），非功能缺陷。

### 1. GGUF prefill 大-T 路由：反量化平面 + 张量核心 GEMM（预期 +30~60% prefill）
- 现状：README 记载 native profile 走 "CUTLASS wide-T over a dequantized staging plane"，
  GGUF 固定走 MMQ（激活量化 Q8_1 + mma dp4a）。
- 方案：prefill T≥512 时，把 IQ/Q 格式权重整块反量化为 FP16 plane，
  走已有的 bf16 mma GEMM（sm75_mma.cuh）或 cuBLAS（magma_sgemmEx 已出现在 profile）。
  反量化带宽开销（3.5bit→fp16 ≈ 4.6×）在 M=2048 的计算密度下可忽略。
- 代码位置：`src/ops/linear/gguf/ggml_bridge.cu`（路由），`ggml_bridge_mmq.cuh`，
  参考 `src/ops/linear/bf16/` 的 wide-T 调度与 `store_plane_kernel`。
- 风险：数值契约（MMQ 的 Q8_1 激活量化 vs 平面 A16），需对照 native profile 的实现。

### 2. mul_mat_q tile 配置 Turing 重标定（预期 +5~15% prefill）
- 虽然 mma 路径已启用，`mmq-config-ampere.cuh` 的 tile 表对 sm_75 未单独标定
  （68 SM / 64KiB smem 与 Ampere 不同）。用 ncu 对热点实例
  （IQ2_S/IQ1_S/IQ1_M，各 ~5-9ms/次）测 occupancy 与 stall 后，
  在 `third_party/ggml-quants/src/ggml-cuda/mmq-config-ampere.cuh`
  增加 cc==750 分支或新增 mmq-config-turing.cuh。
- 便宜，适合第一步落地。

### 3. GDN chunk_recurrence_kernel<32> 调优（预期 +3~5% 总 prefill）
- `src/ops/linear/gated_delta_net`（two_stage chunk_recurrence_kernel），
  48 次调用 × 5.9ms；查 32-block 配置在 68-SM 卡上的 wave 利用率。

### 4. store_plane float→bf16（预期 +1~2%）
- plane 目前 float；改 bf16 减半带宽。注意 MMQ fixup 的数值契约。

## 工作记录

- 2026-11-05 分支建立；基线与 profile 固化（本文档）。
- 2026-11-05 实验 3（正结果，运行时开关）：decode 叠加 n-gram 草稿。
  - `--spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 8`
    （验证式复制草稿，数值无损）：2048 prompt decode 59.9 -> **71.8 tok/s（+20%）**；
    7051 代码 prompt 79.5 -> **104.4 tok/s（+31%）**，超 V100 同负载（108.4 的 96%）。
  - ngram 16 在通用文本上回退（66.7），8 是平衡点；重复性高的负载可用更大值。
  - 注意：该路由增大 workspace 预留（sm70_project_bytes 按激活数计），
    serve 配置需相应下调 --max-context（200000 -> 65536）和 --max-concurrency（2 -> 1）
    才能在 22GB 卡上启动。

- 2026-11-05 实验 2（正结果，已启用）：GGUF 宽产品走 dequant→FP16→cuBLAS 路由。
  - 实现：`gguf_linear.cpp` 的 `turing_cublas_route()`——Turing 卡上 t≥65 的 GGUF 产品
    复用 sm_70 的 `sm70_project`（按 slab 反量化权重到 FP16 + cublasGemmEx 到 FP32 plane
    + epilogue），窄宽度仍走 vec kernel。workspace 计算同步放开。
  - profile 变化：mul_mat_q 系列（~78%）→ `turing_s1688gemm_fp16_*`（~73%）+
    slice_dequantize ~4.4% + store_plane 3.4% + convert_activation 1.2%。
  - 结果（bench, int8 KV, chunk 2048）：
    - 2K prompt：658.5 → **707.8 tok/s（+7.5%）**
    - 27K prompt：579.4 → **590.2 tok/s（+2.0%）**
    - decode 不变（29.8），MTP 组合 2048：688 prefill（spec 接受率 67.1%，
      采样路径不同导致与基线 75.9% 不可直接比）
  - 数值验证：3 组贪心生成（物理/代码/数学，150-200 tokens）新旧路由输出逐字一致。
  - 分析：cuBLAS FP16+FP32-accum GEMM 实测 ~37 TFLOP/s ≈ Turing 该精度的峰值
    （~53 TFLOPS）的 70%；MMQ 原本已接近。进一步提升需 FP16-accum（数值风险）
    或减小 dequant/epilogue 开销。
  - 代码：`gguf_linear.cpp`（gate），sm70 命名空间下的 kernel 均为通用 CUDA/cuBLAS。

- 2026-11-05 实验 1（负结果）：MMQ 缩小 tile 提升 occupancy。
  - 动机：ncu 显示 mul_mat_q 255 regs/thread、~58KiB smem/block、occupancy 25%。
  - 实现：`mmq-config-turing.cuh`，热门口径 I=64 + nthreads=128 + J≤64
    （mma vec-dot 的 warp 行映射不变量 `I==(nwarps/ntx)*rows_per_warp` 要求
    I=64 配 4 warps；J=128 无法满足 footprint 减半，禁用）。
  - 结果：prefill 2048 token **658 -> 549 tok/s（-17%）**，decode 不变。
  - 结论：该 kernel L1/TEX 带宽绑定（baseline 已达 70%），缩小 tile 使激活
    重读次数翻倍，occupancy 收益被带宽开销吞没。代码保留在
    `third_party/ggml-quants/src/ggml-cuda/mmq-config-turing.cuh`，
    dispatch 已用 `if (false && ...)` 关闭（mmq.cuh 两处）。
  - 教训：MMQ 在 Turing 上继续调 tile 空间不大；应转向方向 #1（plane 路由）。
