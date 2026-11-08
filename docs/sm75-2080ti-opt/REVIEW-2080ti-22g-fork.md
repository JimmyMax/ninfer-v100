# 对照 mr-september/ninfer-2080ti-22g 的 Review

日期：2026-11-07

## 两个分支的根本差异

| 维度 | 我们的 ninfer-v100 | mr-september/ninfer-2080ti-22g |
|---|---|---|
| **权重格式** | GGUF（IQ3_S 量化，约 11 GB） | `groupwise-int`（Q4/Q5/Q6/Q8，18.2 GB） |
| **prefill GEMM 路线** | dequant → FP16 → **cuBLAS** | **内核内置 dequant**（Q5/Q4/Q8 fused MMA） |
| **prefill 速度（8k tok）** | **~700 tok/s** | **~612 tok/s** |
| **prefill 速度（64k tok）** | 未测 | **~458 tok/s** |
| **decode 速度** | 30.4 tok/s（无spec） | 27.0 tok/s（无spec） |

关键洞察：他用了**完全不同**的 prefill GEMM 路线（内核内置 dequant，不走 cuBLAS），
而我们走的是 dequant→FP16→cuBLAS。两者各有优劣。

## 他的 prefill GEMM 路线（内核内置 dequant）

```
Fused Q5/Q4/Q8 MMA kernel:
  - 每个 MMA kernel 直接从量化权重读码本 + 缩放，算到 BF16→FP16 tensor core
  - 无单独的 dequantize kernel，无 FP16 weight plane 往返
  - Turing 无 cp.async → 用 register prefetch 手动预取下一个 tile
```

他的 prefill 优化（按顺序做了 P1–P7 共 7 项）：

### P1：Tile 尺寸调优（+50%）

上游 5090 tile 在 Turing 上：
- `~46 KB smem` → 每 SM 只塞 1 个 CTA
- 9.8% tensor pipe util，stall 在 global load（无 cp.async 隐藏延迟）

调小到 `~27 KB smem` 后：
- 每 SM 塞 2 个 CTA
- 一个 tile 的 MMA 与另一个 tile 的 decode/load 重叠

| 操作（1024 tok）| 之前 | 之后 |
|---|---|---|
| Q5 MLP down 5120×17408 | 19.4 ms | **8.9 ms** |
| Q5 down + residual | 18.4 ms | 9.8 ms |
| Q4 gate/up 34816×5120 | 32.6 ms | **16.2 ms** |
| Q4 gate/up SwiGLU folded | 21.7 ms | 13.2 ms |
| GDN grouped input proj | 12.1 ms | 6.8 ms |

### P5：Register Prefetch 预取（+17% prefill）

核心改动。Turing 没有 cp.async，stage/load 是同步的 load+store。
他把下一个 activation tile 的 load 提到 MMA 之前用寄存器预取，
MMA 执行完后再 store 到 smem：

| Shape, T | 之前 | 仅 batch load | + prefetch across MMA（保留）|
|---|---|---|---|
| Q5 n5120_k17408, T=512 | 5026 µs | 3463 µs (-31%) | **3166 µs (-37%)** |
| Q5 n5120_k17408, T=1024 | 8413 µs | 5816 µs (-31%) | **5542 µs (-34%)** |
| Q5 n5120_k6144, T=1024 | 2900 µs | 2065 µs | **1903 µs (-33%)** |

### P2：GDN T=1 Q5 launch（+3.3% decode）
### P3：INT8-KV attention 64×32 tile（+9% at 64k）
### P4：FP16 activation 预转换——**回退**（-4.4%，不保留）
### P6：Q6 output head——不推荐（-0.4% decode）
### P7：小 kernel 融合——**不做**（上限 1-2%，不值得）

## 我们的路线 vs 他的路线

### 为什么 cuBLAS 路线在 prefill 上其实更快

**我们的实测（GGUF IQ3_S，2048-token prefill）**：
- GEMM 占 73.7%，效率 **79%**（接近天花板）
- cuBLAS 已经做了 split-K（gridZ=2/4/17），自适应切 tile
- 不需要做 register prefetch（cuBLAS 内部已处理）

**他的路线（Q5/Q4 18.2GB）**：
- GEMM 也占 ~95%，但原始效率更低（需要 fused dequant）
- 靠 P1 tile 调优 + P5 register prefetch 补上来
- 最终 prefill 612 tok/s@8k vs 我们的 700 tok/s@8k

**结论：我们的 cuBLAS 路线本身已经更快**（79% vs 他的优化后效率），
不需要额外做 register prefetch。

### 他做了但我们没做的

1. **Tile 尺寸适配 Turing smem**（我们的 cuBLAS 不需要）
2. **GDN T=1 SIMT kernel 切换**（+3.3% decode，我们可借鉴）
3. **INT8-KV attention 64×32 tile**（我们已测，可借鉴）
4. **FP16 范围探针**（他的 C1，防止 BF16→FP16 饱和，我们可借鉴）

### 我们做了但他没做的

1. **GGUF dequant→FP16→cuBLAS**（更快的 prefill GEMM）
2. **spec decoding 全套调优**（我们的 decode 30.4→128.9，他没做）
3. **--gdn-state-fp16**（我们没有，他也没做）
4. **chunk_recurrence 三段式重构**（他也没做，我们正要做）

## 我们的下一步（按收益排序）

基于他的经验 + 我们的 profile：

| 优先级 | 改动 | 预期收益 | 他的经验 |
|---|---|---|---|
| **1** | `chunk_recurrence` 三段式重构 | +12~19% prefill | 没做，但他的 GDN 也是瓶颈之一 |
| 2 | INT8-KV attention tile 64×32 | +1~9%（8k~64k） | P3，+9% at 64k |
| 3 | `store_plane` 消除（epilogue 融合） | +3~4% prefill | 没做，他直接用 bf16 out |
| 4 | GDN T=1 direct SIMT kernel | +3% decode | P2，+3.3% decode |
| 5 | FP16 范围探针 | 保险 | C1 |

## 一个重要警示：他的 P4 教训

他尝试"把 BF16→FP16 转换移到 stage 阶段做一次"，结果 **decode 慢 4.4%**。
原因是：Turing 的 cp.async 模拟是同步 load+store，
中间插入转换会延长依赖链，而移除的 per-MMA 转换反而不多。

**教训：在 Turing 上，"少做一次转换"不一定快——同步 load+store 的依赖链更关键。**
这对我们的 `convert_activation`（21MB BF16→FP16 往返）是一个警示：
消除它可能反而更慢，需要先 profile 再改。
