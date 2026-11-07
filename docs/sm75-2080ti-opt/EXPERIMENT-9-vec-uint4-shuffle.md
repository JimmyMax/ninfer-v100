# 2080 Ti 绝对性能对标（替代 V100 相对值）

## Prefill：计算 vs 带宽的完整账

2080 Ti 理论峰值（实测 SM 1845MHz @ 250W，标称 1545MHz）：

| 项目 | 理论 | 实测 | 缺口 |
|---|---|---|---|
| **FP16+FP32-acc 张量核** | 68 SM × 2 × 256 ops/cyc × 1.845GHz = **507 TFLOP/s** | cuBLAS s1688gemm ~37 TFLOP/s | **13.7×** ← 大 |
| 激活值反量化后 DRAM 读 | 616 GB/s | 混合后 prefill 占时 <3% | 非瓶颈 |
| QKV 激活 L1 带宽 | 68 × 128B/cyc × 1.845 = 16.1 TB/s | 张量核流片读取 | — |

**结论：prefill 离硬件极限还很远（37/507 = 7%），不是"实用天花板"。**
cuBLAS s1688gemm 单 kernel 本身可能达到 ~200 TFLOP/s（Turing 张量核算 GEMM 的经验上限），
我们的 37 被三个因素压低：
1. **小 M**：prefill 2048 token = M=2048，N=每行 17408 个输出 —— GEMM 形状离 cuBLAS 最优远
2. **PLE（线形注意力）部分不走张量核**：GDN 的 state/conv/recurrence 是带宽/延迟绑定
3. **量化反量化开销**：IQ 权重反量化到 FP16 平面的指令成本

## Decode：DRAM vs LSU 的硬墙

| 项目 | 理论 | 实测 | 结论 |
|---|---|---|---|
| DRAM 带宽 | 616 GB/s | 297 GB/s（48%） | 有余量但被挡 |
| **LSU 事务**（Turing） | 128B/cyc/SM × 68 × 1.845GHz = 16 TB/s | 91.6% 饱和 | **真正的墙** |
| IQ 反量化指令数/字节 | ~5-8 次 1-4B 读 / 32 值 | 实测满 | LSU 事务主导 |

裸 decode 天花板 = 10.9GiB ÷ 297GB/s ≈ **38.7 tok/s**（LSU 满时的 DRAM 上限）。
现在 30.4 = 天花板的 79%。剩余空间在 LSU 效率（vec kernel 事务数/字节），非 DRAM。

## Vec kernel 深度优化的真实空间

LSU 墙下，decode 的唯一出路是**减少每次反量化的 LSU 事务数**：

| 方案 | 原理 | 预期 | 质量 |
|---|---|---|---|
| A. uint4 整块加载 + shuffle 分发 | 每 lane 一次 16B 加载，shuffle 拼出 256 值块 | 事务数 /4 → decode +30~50% | bit-exact（只是重排读取顺序） |
| B. 反量化表查表重构 | IQ1_S 的 2048 项 table 改为寄存器内位运算 | 消 table 读，-20% 事务 | 数值完全一致 |
| C. 反量化到 smem 常驻（跨 token） | 每行只反量化一次，后续 token 直接读 | 理论 +N×（N=token 数） | bit-exact，但 smem 不足（10.9GiB 权重 >> 64KB） |

**A 是唯一可行的**：把"每 32 值 lane 读 ~5 次小事务"改成"每 lane 一次 16B 加载 + 3 次 shuffle"，
LSU 事务数降为 1/4，DRAM 带宽可接近 600 GB/s → decode 42-48 tok/s。

**风险**：IQ3_S 的 sign/scale 字段不规则（1 字节 + 4 字节对齐错位），
shuffle 分发的位操作复杂，需要仔细处理边界。

## 下一步

先做 A 的原型：选 IQ3_S（27% 热点）写 uint4 加载 + shuffle 分发的 decode 变体，
bench 验证 bit-exact + 提速。如果 IQ3_S 成功，再推广到 IQ1_S/IQ4_XS/Q4_K。

### IQ3_S 块布局（110B，256 值）

```
offset  size  content
0       2     half d (block scale)
2       64    qs[8]   : 8 × uint32，每 8 值一个索引（4B/值组）
66      8     qh[8]   : 1B 高位索引（1B/值组，第 8 位）
74      32    sg[8]   : 8 × uint32 符号（4B/值组）
106     4     sc[4]   : 4bit/值组 scale
```

现有 decode（lane u 管 32 值，u = lane%8）：
- `u32_a2(b+2+8u)` + `u32_a2(b+6+8u)`：2 次 4B 读（64B 段内，不同 lane 分散）
- `b[66+u]`：1B 读
- `u32_a2(b+74+4u)`：4B 读
- `b[106+u/2]`：1B 读
- 每 lane 每块共 **5 次独立事务**，跨 104B 跨度

### uint4 + shuffle 重构方案

每 lane 加载 `b + 16*lane` 的 16B（110B 需要 7 个完整 uint4 + 6B 尾部，用 8 个 lane 覆盖）。
然后按字段分发：
- `q0/q1`（qs 段）：lane u 需要 b[2+8u..9+8u] = lane (8u+2)/16 的 uint4 里的 2 个 uint32
  → `__shfl_sync` 从持有该 uint4 的 lane 取
- `qh`：lane u 需要 b[66+u] = lane 4 的 uint4 里的 1 字节
  → `__shfl_sync(lane4_uint4, u&3, ...) >> (8*(u>>2))` 提取
- `sg`：lane u 需要 b[74+4u..77+4u] = lane 4/5 的 uint4
- `sc`：lane u 需要 b[106+u/2] 的 nibble

### 关键问题：shuffle 的位重排不等于"bit-exact"

**重点**：IQ3_S 的 `idx = (qs_byte | qh_bit) -> grid[idx]` 的 grid 查表是**非线性映射**，
`negate_bytes` 是逐字节取反。这意味着：
- 如果 shuffle 后得到的 qs/qh/sg/sc 与直接读取**完全一致**，则输出逐 bit 相同
- 但如果 shuffle 引入任何位序/符号/字节序错误，输出就完全不同

**风险**：`u32_a2` 是两个 uint16 的拼接（小端），shuffle 提取必须保持完全相同的字节序。
IQ3_S 的字段偏移不规则（66, 74, 106 不是 4 的倍数），需要精确的位运算。

### 替代方案：B（表查表重构）

IQ1_S 的 `iq1s_grid_gpu` 是 2048 项的 smem 表，每项是 8 个 4-bit 值。
可以预计算为寄存器位运算：输入 12bit 索引 → 输出 8×4bit 的 32bit 字，
用查表逻辑本身的开销换取 LSU 事务。但 2048 项的表压缩成算术表达式不现实。

### 实际可行的变体：IQ3_S 的 qs/qh 合并加载

不改 shuffle，只把 qs[8]（64B）和 qh[8]（8B）合并为一次 uint4 加载：
- lane 0-3：加载 qs[0..63]（4×uint4 = 64B）
- lane 4：加载 qh[0..7] + sg[0..3]（1×uint4 = 8+4=12B，但 74 不对齐 16B）
- 然后 shuffle 分发到各 lane

这要求 8 个 lane 共享一个块（kPerBlock=8），与现有架构一致。
但 IQ3_S 的 110B 块不是 16B 对齐的倍数（110 % 16 = 14），
跨块的 lane 会读到错位的数据。需要按块基址对齐到 16B 边界加载，
然后位偏移修正——这正是复杂度所在。

### 更深入的方案：预转置（Pretranspose）

在权重加载时（或离线转换时）把 IQ3_S 的 110B 块**重新排列**为：
```
[qs 64B | qh 8B | sg 32B | sc 4B | d 2B | pad 2B]  ->  112B
```
并进一步把 qs 拆成 8 个独立的 8B 段，使每个 lane 的 8B 读取完全对齐。

但这改变了 artifact 格式，需要：
1. 转换器新增布局（`gguf_blocks_v2_iq3s`）
2. 运行时识别新布局并选择新 kernel
3. 旧 artifact 的兼容性处理

这是"转换期预处理"路线，与 MMQ 的"激活预量化"思路一致，但工程量大。

### 结论：实验 9 的评估

uint4+shuffle 方案在 IQ3_S 上**理论可行但实现风险高**：
- 位序/字节序/符号的精确处理需要逐位验证
- 110B 的非对齐块布局使 shuffle 分发的位操作极其复杂
- 即使正确，LSU 事务减少 5→2 的收益被 shuffle 指令开销部分抵消
  （Turing shuffle 是 32 周期/32 lane，比 L1 命中 ~30 周期略慢）

**建议**：先做 IQ4_XS（136B，对齐更规整：136 % 8 = 0，字段偏移 0/2/4/8/16 都是 4 的倍数）
作为原型验证 shuffle 路径的正确性，再评估是否推广到 IQ3_S。
