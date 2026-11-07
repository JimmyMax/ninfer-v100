# 实验 10：Prefill GEMM 调度优化（Turing 张量核利用率）

**目标**：prefill 777 → 1000+ tok/s（cuBLAS GEMM 45% SM 利用率 → 70%+）

## 现状（ncu 实测）

| 指标 | 实测 | 解读 |
|---|---|---|
| turing_s1688gemm 占用 | **72.6%** GPU 时间 | 绝对热点 |
| **SM 利用率** | **45.8%** | 一半算力闲置 |
| **Occupancy** | **25.0%** | 238 regs/thread + 65KB smem/block |
| Grid | 64 blocks = 0.94 waves | **严重欠载**（68 SM 只跑 64 block） |
| DRAM | 9.8% | 完全非瓶颈 |

## 根因分析

GEMM 形状：**M=2048（tokens）× N=17408（输出特征）× K=5120（隐藏维）**

- cuBLAS 选择 `turing_s1688gemm_fp16_256x128_ldg8_tn`（256×128 tile）
- grid = (2048/256) × (17408/128) = 8 × 136 = **1088 blocks**
- 但 ncu 显示 grid=64 → **cuBLAS 用了 split-K**（K 方向拆分 17 份），64 blocks = 1088/17
- 238 regs/thread + 65KB smem → 每 SM 只能跑 1 block → **0.94 waves = 单波次欠载**

## 优化方向（按收益/工程量排序）

### 1. 强制更大的 split-K（预期 +30~50% prefill）
- cuBLAS heuristic 选择了 split-K=17，但 Turing 68 SM 需要 68×N（N≥2）才能满载
- 手动设置 `CUBLAS_SPLIT_K_MODE` 或 `cublasGemmEx` 的 `CUBLAS_GEMM_ALGO_TENSOR_OP`
- **关键**：`cublasGemmEx` 不支持显式 split-K，需要 `cublasLtMatmul` + algo tuning

### 2. 减小 tile 尺寸（预期 +20~40%）
- 256×128 tile 在 Turing 上寄存器/smem 溢出（238 regs > 255 阈值边缘）
- 128×128 tile（grid 2176 blocks，3.2 waves）或 128×64 tile（grid 4352，6.4 waves）
- 需要 cuBLASLt algo 搜索，或手写 CUTLASS Turing kernel

### 3. 分块 N 维度（预期 +15~30%，工作量小）
- 把 N=17408 拆成 4×4352 的 4 个独立 GEMM，流式执行
- 每个 GEMM 的 grid 变小，但多个 stream 并行可重叠
- **风险**：CUDA graph capture 的复杂度，多 stream 同步开销

### 4. 手写 CUTLASS Turing GEMM（预期 +50~80%，工作量大）
- 完全控制 tile/调度/寄存器分配
- 参考 `src/ops/linear/bf16/bf16_gemm_mma.cuh` 的 sm75_mma 实现
- 需要处理 IQ 反量化 + GEMM 的融合（消掉 dequant 中间平面）

## 实验计划

### Phase 1：cuBLASLt algo tuning（1-2 天，低风险）
```cpp
// sm70_gemm.cu 的 gemm() 改用 cublasLtMatmul + heuristic search
cublasLtMatmulAlgo_t algo;
cublasLtMatmulAlgoGetHeuristic(..., &algo, ...);  // 搜索所有 algo
// 或手动指定 split-K
algo.algoId = CUBLASLT_MATMUL_ALGO_DEFAULT;
algo.ctaSwizzling = ...;
algo.customOption = ...;  // split-K factor
```

### Phase 2：N 维度分块（0.5 天，验证用）
```cpp
// 简单分块：把 N=17408 拆成 4 个 4352 的 GEMM
for (int i = 0; i < 4; ++i) {
    gemm(w + i*4352*K, 4352, K, x, tokens, out + i*4352, ldc, stream);
}
```

### Phase 3：CUTLASS Turing kernel（1 周，高风险高收益）
- 参考 `third_party/cutlass` 的 sm75 示例
- 融合 IQ 反量化（消掉 dequant 的 3.4% + store_plane 的 3.5%）

## 质量门禁

所有优化必须满足：
1. 贪心生成 5 组逐字一致（FP16 GEMM 本身已是 bit-exact，调度不改数值）
2. bench prefill 提升 ≥15% 才启用

## 工作记录

- 2026-11-07 预研完成：ncu 定位 45% SM 利用率 + 25% occupancy + 0.94 waves 欠载
- 2026-11-07 **实验 10 负结果（N 维度分块）**：
  - 实现：把 N=17408 拆成 4 个 4352 的 GEMM
  - 结果：prefill 777.2 -> 705.8 tok/s（**-9.2%**）
  - **根因分析**：
    1. ncu 的 grid=64 是**单个 kernel 实例**的 grid，不是整个 GEMM 的并发度
       cuBLAS 的 split-K 会把 K 方向拆成 17 份，每份一个 kernel
       总 block 数 = 64 × 17 = 1088 blocks，不是 64
    2. N 维度分块后，每个 GEMM 的 grid 更小（16 blocks），但 4 个 GEMM 串行执行
       总 block 数不变（16×17×4 = 1088），且多了 3 次 kernel launch 开销
    3. 每个 4352-wide GEMM 的 K=5120 不变，split-K 因子不变，wave 利用率不变
  - **结论**：N 分块无效，因为 cuBLAS 的 split-K 已经充分利用了并行度。
    真正的瓶颈是 **238 regs/thread + 65KB smem/block** 导致的 25% occupancy，
    这需要 CUTLASS 重写（更小 tile/更少寄存器），不是 cuBLAS 参数能解决的。

## 工作记录

- 2026-11-07 预研完成：ncu 定位 45% SM 利用率 + 25% occupancy + 0.94 waves 欠载
