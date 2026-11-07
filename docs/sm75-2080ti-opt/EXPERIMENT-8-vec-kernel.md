# 实验 8：ggml vec kernel 访存模式重写（decode 提速研究）

**目标**：decode 29.8 → 42-45 tok/s（DRAM 48% → 75%）
**硬约束**：数值逐 bit 一致（同一反量化公式、同一 fp32 累加顺序）→ 零质量风险。

## 现状解剖（`ggml_bridge_vec.cuh::kernel`）

- 几何：4 warps/block（128 线程），**每 warp 负责 2 行**（kRows=2），lane l 处理行内
  32 值切片 l, l+32, ...
- 每切片：D::decode 把 32 个权重值解成 8×int32（值序）+ 因子，再与激活做 dp4a
- 激活是 planar q8_1：int8 [columns][K] + 每 32 值一个 (d, sum) half2
- 分发：`launch()` 用 occupancy 常驻 block 数 × SM 数封顶 grid
- ncu（IQ3_XXS 212us 实例）：DRAM 297GB/s(48%)、L1/TEX 91.6%、L1 命中 87%、
  occupancy 85%、No Eligible 65-70%

## 瓶颈理论

L1/TEX 91.6% 且 L1 命中 87% → 大量小事务命中 L1（IQ 格式的 sign/scale 分散读取），
L1 管道吞吐是墙，DRAM 有一半余量闲置。方向：
1. **减少每字节的 L1 事务数**：uint4(16B) 对齐向量加载替代多路 u32 读取；
   scale/sign 一次读入寄存器复用（当前每次切片读一次）。
2. **改几何让 DRAM 流水更满**：kRows / kVecWarps / grid 上限扫描（不改算术，
   bit-exact 保证）——先做，便宜。
3. **smem 预排**：每 warp 先把一行的一段反量化进 smem，再与 T 列点积——
   decode T=1 时收益有限（权重读一次即用），优先级低。

## 验证（每个变体必过）

1. `ninfer_bench -pg 2048,512` decode t/s（对比 29.8 基线）
2. 贪心生成 3 组逐字一致（几何变体可跳过——算术不变；内层重写必做）

## 工作记录

- 2026-11-07 研究启动：kernel 解剖 + 瓶颈理论。
- 2026-11-07 几何扫描（负结果）：kVecWarps 4->8 decode 无变化（30.18 vs 29.8）；
  kRows 2->4 崩溃（Fused 变体对 kRows=2 有隐含假设），放弃。
- 2026-11-07 smem staging 重写（负结果，-12%，已回滚）：
  - 实现：32 切片 chunk 经 uint2 合并加载进双缓冲 smem，decode 读相同字节
    （bit-exact），IQ3_S/IQ3_XXS/IQ4_XS/Q4_K/IQ2_S/IQ1_S/IQ1_M 七个格式。
  - 修了一个偏移 bug（chunk 内偏移必须从 0 重启）后功能正确。
  - 结果：decode 29.8 -> 26.1 tok/s（-12%）。
  - **机理**：Turing 的 L1 与 smem 共用 LSU 管道，staging 把全局 L1 事务换成
    等量 smem 事务，没有减少管道压力，反而增加了拷贝指令 + 2 次 syncwarp/chunk。
  - 结论：**任何不减少 decode 加载指令数量的重写都不会赢**。真正的优化必须重构
    decode 数据流本身（如每 lane 用 uint4 加载整个 256 值块再 shuffle 分发），
    是更深层的重写，收益不确定。
