# 实验 11：CUTLASS Turing GEMM（prefill 重写）

**目标**：prefill 777 → 1000+ tok/s（+30~50%）

## 根因（实验 10 确认）

cuBLAS `turing_s1688gemm_fp16_256x128`：
- 238 regs/thread + 65KB smem → 1 block/SM → **25% occupancy**
- split-K=17 已饱和 block 并行，但单 block 的 tile 太大
- 需要 128×128 或 128×64 tile，<128 regs，<32KB smem → 2+ block/SM

## 设计

基于 `src/ops/common/sm75_mma.cuh` 的 `mma_f16_k8`（FP16 m16n8k8）：

```
Block tile:  128×128×64  (2 warps × 2 warps = 4 warps)
Warp tile:   64×64×64
MMA:         m16n8k8 × 4 (k64 = 8×k8)
Shared mem:  A: 128×64×2B = 16KB, B: 64×128×2B = 16KB → 32KB
Registers:   每 warp 64×64 accum = 8×float4 = 32 regs + 输入 ~64 regs ≈ 96 regs
Occupancy:   32KB smem → 2 block/SM, 96 regs → 2 warp/scheduler → 50% occupancy
```

## 关键：IQ 反量化融合

当前流程：
1. `slice_dequantize_kernel` 把 IQ 权重反量化到 FP16 平面（全局内存）
2. cuBLAS 读 FP16 平面做 GEMM

融合后：
1. GEMM kernel 的 load A 阶段直接读 IQ 格式
2. 在 shared memory 或寄存器中反量化
3. 立即做 mma

消掉：dequant kernel（3.4%）+ 平面读写带宽（2×10.9GiB/2048tokens ≈ 10MB/层）

## 风险

1. **数值**：IQ 反量化是确定性的（查表+位运算），与 FP16 mma 的顺序无关 → bit-exact
2. **寄存器压力**：IQ 反量化需要 table lookup，可能增加寄存器使用
3. **边界处理**：K=5120 是 64 的倍数，无尾块；M=2048 是 128 的倍数，无尾块

## 实现计划

Phase 1：纯 FP16 GEMM（不含反量化融合）
- 验证 128×128 tile 的正确性和性能
- 对比 cuBLAS：同样的输入，输出必须逐 bit 一致

Phase 2：融合 IQ3_S 反量化
- 在 load A 时直接读 IQ3_S 格式
- 用 `sm75_mma.cuh` 的 `bf16_fragments_as_f16` 验证范围

Phase 3：推广到 IQ3_XXS/IQ4_XS/Q4_K
- 其他格式的反量化逻辑复用现有 Decoder

## 质量门禁

1. 贪心生成 5 组逐字一致（vs cuBLAS 路径）
2. bench prefill ≥ 1000 tok/s 才启用

## 工作记录

- 2026-11-07 预研完成：确认 sm75_mma.cuh 的 mma_f16_k8 可用
