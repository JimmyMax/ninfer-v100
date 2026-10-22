# V100 GGUF 解码优化:vec 内核指令缓存瓶颈(2026-10-21)

目标:评估并把 `Qwen3.8-27B-GSQ-RCO-IQ3_S`（GGUF v3 混合量化,11.9 GiB)在 V100 上的解码
性能向 NVFP4 路线对齐或超越;若可,给出内核改动。本文是 1.6.2 基线上的优化记录。

## 1. 基线结论(优化前)

同机(device 1,V100-32GB)、同 27K prompt、96 输出:

| 路线 | plain decode | MTP d4 |
|---|---|---|
| NVFP4(约 20 GiB 权重) | 31.5 tok/s | 60.9 tok/s |
| GGUF IQ3_S(11.9 GiB 权重) | 38.6 tok/s | 57.0 tok/s |

GGUF 的 plain 解码已超过 NVFP4(权重小 40%,访存主导),MTP 差 6%。两者都在 30 tok/s
量级,瓶颈都不是算力而是解码期的权重读取。因此"GGUF 追上 NVFP4"的问题应细化为:
**vec 解码内核的有效带宽还能提多少。**

## 2. 解码期 kernel 构成(生产 trace,nsys)

MTP 一步(约 60 ms)中 `vec::kernel`(GGUF vector GEMV)占 70.2%,约 42.3 ms,
有效带宽约 281 GB/s(V100-32GB ECC 峰值约 803 GB/s,即 35%)。按类型:

| 类型 | ms/step | 占比 |
|---|---|---|
| IQ3_S | 14.5 | 34% |
| IQ4_XS | 7.8 | 18% |
| IQ3_XXS | 7.4 | 18% |
| Q4_K | 5.9 | 14% |
| Q6_K | 3.1 | 7% |
| 其余(IQ2_*、IQ1_M 等) | 3.6 | 9% |

T=5(校验 5 列 × 48 层全模型)是主体;T=1 只出现在 MTP 草稿层(约 4 层)。
prefill(wide-T mmq 张量核心路线)不在本文范围,其速度已与 NVFP4 相当。

## 3. 内核剖析(nsys/ncu,合成权重 harness)

harness:同块布局、同访问模式、合成随机权重,`bench_gguf_gemv`(sm_70,V100,4 个
真实 shape × T=1/5,见 §5 表)。关键 ncu 数据(IQ3_S 17408×5120,T=5,KS=5120,
基线):

| 指标 | 值 |
|---|---|
| 时长 | 125 µs(harness 中位) |
| DRAM 吞吐 | 30% |
| L1/TEX 吞吐 | 68.6% |
| SM 计算吞吐 | 34.5% |
| 每 warp 指令间隔 | 7.5 cycles |
| 其中指令取指/L1I miss 停顿 | **31.9%** |
| 理论/实测占用率 | 18.75% / 18.0%(130 寄存器,3 CTA/SM) |

即:**不是 DRAM 瓶颈,而是指令缓存(L1I)瓶颈** —— T≥2 时 slice 体(8 查表 + 8 negate
+ T 列激活读取 + T×2×8 dp4a)被 KS 循环完全展开(KS=5120 → ×5;KS=17408 → ×17),
代码体积冲出 Volta L1I,取指停顿占 1/3 的周期。T=1 代码小(56 寄存器、32 warps/SM),
无此问题 —— 与"只有校验步慢"一致。

## 4. 实验矩阵(合成 harness,V100,µs/GB/s 见 §5)

| 变体 | 结论 |
|---|---|
| V0 基线 | T=5 比 T=1 慢 40-50% |
| kRows 2→1(T≥2) | **更慢**:激活读取翻倍(每行对不再共享 `a[T][8]`),L1 饱和到 84.8%,T=5 反而 +17%。放弃 |
| KS 循环全部 unroll 1(T=1 也) | T=5 改善但 T=1 个别 shape +5-8%(循环分支开销)。放弃 |
| **KS 循环 T≥2 unroll 1,T=1 保持全展开** | **T=5 全面改善 8-19%,T=1 持平**。采纳 |
| KS 循环 T≥2 unroll 2 | 比 unroll 1 差(代码又变大)。放弃 |
| sm70_small 张量核心路线 | 1-lane/row 2B 全局读,估算 50-110 GB/s,远差于 vec 的 281。放弃 |
| cp.async 预取 | sm_70 无 cp.async;且 110 B 块(110 mod 8 = 6)只有偶数块 4B 对齐,收益有限。放弃 |
| 预解码权重 | 属于运行时权重重打包,超出本范围。放弃 |

## 5. 采纳变体的 harness 测量(V100,µs;括号内 GB/s)

T=5 是校验步主体,改进 8-19%(主 shape 13-19%):

| shape | 类型 | T | 基线 | 采纳变体 |
|---|---|---|---|---|
| 17408×5120 | iq3_s | 1 | 72 | 73 |
| 17408×5120 | iq3_s | 5 | 125 (307) | **109 (353)** |
| 5120×17408 | iq3_s | 1 | 71 | 72 |
| 5120×17408 | iq3_s | 5 | 124 | **101 (379)** |
| 6144×5120 | iq3_s | 5 | 51 | 47 |
| 2048×10240 | iq3_s | 5 | 42 | 38 |
| 17408×5120 | iq4_xs | 1 | 69 | 69 |
| 17408×5120 | iq4_xs | 5 | 95 (500) | **85 (557)** |
| 5120×17408 | iq4_xs | 5 | 92 | **80 (593)** |
| 6144×5120 | iq4_xs | 5 | 45 | 38 |
| 2048×10240 | iq4_xs | 5 | 35 | 29 |
| 17408×5120 | iq3_xxs | 5 | 117 | 102 |
| 5120×17408 | iq3_xxs | 5 | 122 | 95 |
| 17408×5120 | q4_k | 5 | 91 | 85 |
| 5120×17408 | q4_k | 5 | 90 | 93 |
| 6144×5120 | q4_k | 5 | 40 | 39 |

(harness 数字为单跑中位,±5% 噪声;T=1 行全部持平。)

## 6. 端到端 A/B

(同 27K prompt、96 输出、device 1;old = ninfer:1.6.2 基线二进制,new = 本变体)

| 配置 | old tok/s | new tok/s | Δ |
|---|---|---|---|
| plain | 38.6 | 38.6 | 0% |
| MTP d4 | 57.0 | **67.8** | **+18.9%** |

MTP 接受率不变(61/129,47.3%,temp=0 确定性);prefill 不变(916-927 tok/s,未触及 mmq 路线)。
对照 NVFP4(31.5 plain / 60.9 MTP,同日同机):优化后 GGUF 两种模式均超过 NVFP4
(plain +22.5%,MTP +11.3%)。

## 7. 正确性

- `ninfer_linear_gguf_test`(GGUF 线性全部类型/shape,T=1..8 向量与矩阵):OK
- `ninfer_moe_experts_gguf_test`:OK
- 改动只影响代码生成(循环展开),整数点积路径与舍入语义不变;
  MoE CPU/GPU 位级一致契约不涉及 dense vec 路径。

## 8. 剩余差距与后续方向

- vec 内核当前受 L1/LSU 与占用率双重约束(§3:DRAM 仅 30%)。后续可量化方向:
  1. ds(d/sum)半字合并:每 slice 5 次 half2 读 → 2 次,省 3 条 LG 指令/行/slice。
  2. IQ3_S 查表 8 LDS/slice 是最大 L1 消费者;表 2 KB 无法进寄存器,常量缓存会
     序列化 32 个不同索引,均不可行;若未来块尺寸对齐允许 8B 向量读可重排表布局。
  3. `negate_bytes` 每 slice 32 条 ALU,可用查表合并的符号表(16 变体)替换,
     代价是 32 KB 共享表、占用率减半,需实测权衡。
- 对齐限制:块 110 B(110 mod 8 = 6),字段只有 2B/4B 对齐,8B/16B 向量化全局读
  在现有块布局下不可得;要拿到大块向量读需改量化格式本身(超出本范围)。

## 9. 数据与复现

- 数据目录(入仓):`bench/v100/data/gguf_vec_opt_2026-10-21/`
  - `bench_gguf_baseline.txt` / `bench_gguf_v4.txt`:harness 原始输出(基线/采纳变体)
  - `gguf_optab.log`:端到端 A/B 日志
  - `gguf_gemv_bench.cpp` + `gguf_harness_stubs.cpp` + `build_bench.sh`:合成 harness
    与构建脚本(nvcc -O3 -arch=sm_70,链接 `ggml_bridge.cu` 与各 `ggml_vec_*.cu`)
- 生产 trace:`/home/max/.cache/ninfer-prof/gguf/gguf_mtp.sqlite`、
  `/home/max/.cache/ninfer-prof/nvfp4/nvfp4_mtp.sqlite`。
- 端到端脚本:`/tmp/gguf_optab.py`(方法:同 27K prompt、96 输出、device 1,读 serve 日志 done 行)。
