# Swift-1.5 v3 A/B(NVFP4 vs GGUF Q4_K_M,同引擎)+ 调参矩阵

日期:2026-10-22。设备:GPU 1 Tesla V100-PCIE-32GB(sm_70,ECC on,实测带宽 803 GB/s)。
引擎:`ninfer:1.6.2-ggufopt`(1.6.1 基线 + vec T≥2 `#pragma unroll 1` 优化)。

两个工件,**同一 Swift-1.5 v3 权重**:
- NVFP4:`swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3.ninfer`(设备权重 18.4 GiB)
- GGUF:`Swift-1.5-Qwen3.8-27B-Q4_K_M-ninfer-v3.ninfer`(源 GGUF 16.23 GiB;设备权重 16.2 GiB)
  - 类型构成:Q4_K 10.66 GiB / Q6_K 3.84 GiB / Q8_0 1.30 GiB / Q4_0 0.22 GiB / Q5_K 0.11 GiB / F32 0.10 GiB
- 两边都有 MTP(nextn)头;greedy,96 输出(除注明),int8 KV,1 lane,`--ngram-draft-tokens` 未开。
- 提示词:与 A/B 一致的重复 filler("第N条：关于季节…" ×N)+ 散文任务;N=295(7,518 tok)/1050(27,250 tok)/3267(89,326 tok)。

## 1. 调参矩阵(part1:matrix.log,part2:matrix2.log,part3:matrix3.log)

### Prefill(tok/s)

| ctx | chunk | NVFP4 | GGUF |
|---|---|---|---|
| 7.6K | 2048 | 1125 | 1014 |
| 7.6K | 4096 | 1103 | **1099** |
| 27K | 1024 | 977 | 869 |
| 27K | 2048 | **1006** | 920 |
| 27K | 4096 | 1010 | **988** |
| 82K | 2048 | 756 | 713 |
| 82K | 4096 | 754 | **753** |

- NVFP4:chunk 2048/4096 基本等价(±1%)。
- GGUF:chunk 4096 全面最优(27K +7.5%,7.6K +8.8%,82K +5.6%);82K 时追平 NVFP4(753 vs 754)。
- 原因(prefill nsys 构成):prefill GEMM 走 cutlass s884 tensor-core(51% 时间),大 chunk = 更大 GEMM M 维 = 更高 TC 利用率;2048→4096 在 GGUF 上收益更大。

### Decode(tok/s,greedy,96 输出)

| ctx | 模式 | NVFP4 | GGUF |
|---|---|---|---|
| 7.6K | plain | 34.2 | 33.4 |
| 7.6K | MTP d2 | — | — |
| 7.6K | MTP d4 (ch2048) | 73.7(接受 47.3%) | 65.2(接受 45.2%) |
| 7.6K | MTP d4 (ch4096) | 71.7(接受 47.7%) | **71.6**(接受 54.2%) |
| 27K | plain | **32.7** | 31.8 |
| 27K | MTP d2 | 68.8(71.8%) | 59.9(65.4%) |
| 27K | MTP d4 (ch2048) | **75.5**(52.0%) | 66.6(52.9%) |
| 27K | MTP d4 (ch4096) | 75.5(52.0%) | **73.8**(58.9%) |
| 27K | MTP d6 | 69.4(39.1%) | 55.1(37.6%) |
| 27K | MTP d8 | 60.0(30.9%) | 39.5(33.0%) |
| 27K | MTP d4 + c2 | 75.5 / 75.1 | 66.4 / 66.9 |
| 27K | MTP d4 + ngram15 | 75.5 | 66.8 |
| 82K | plain | **28.7** | 28.1 |
| 82K | MTP d4 (ch2048) | **67.0**(52.0%) | 63.8(55.1%) |
| 82K | MTP d4 (ch4096) | — | 57.0(50.0%) |

- MTP:d4 两边都是最优点;d2 接受率高但每步草稿少;d6/d8 接受率断崖,得不偿失。
- `--max-concurrency 2`:单请求 decode 无提升(75.5/66.9 与 c1 相同);两条相同提示词的第二条因上下文缓存只 prefill 2,290 tok(缓存效应,生产不同用户无此收益)。
- `--ngram-draft-tokens 15`:无增益(该 filler 上 copy draft 未被接受)。
- 接受率随提示词/分块波动 ±5pp(冻结 draft 头 + 分块改变 hidden state 的浮点求和)。
- ch4096 对 GGUF decode 也有 +10.8%(27K d4:66.6→73.8),主要因接受率漂移(52.9%→58.9%)+ 每步略快。
- 82K MTP d4:ch4096 57.0(接受 50.0%)vs ch2048 63.8(接受 55.1%)——差距全在接受率漂移(63.8 × 50.0/55.1 ≈ 57.8),chunk 对 MTP decode 无系统性影响;82K 两侧最快配置仍是 ch2048。

### 调优后每侧最优配置

- NVFP4:`--prefill-chunk 2048 --spec mtp --draft-tokens 4 --lm-head-draft`(27K 75.5 / 82K 67.0 / plain 32.7)
- GGUF:`--prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft`(27K 73.8 / 82K 63.8(ch2048)/ plain 31.8)

## 2. Q4_K_M decode 内核构成(nsys,profile/q4k_plain.nsys-rep,27K plain)

decode 段(GPU 忙碌率 97.6%,96 tok / 3.29 s):

| 内核 | 时间 | 占比 | 每 token 权重 | 纯带宽时间(803 GB/s) | 效率 |
|---|---|---|---|---|---|
| vec::kernel Q4_K(type 12,4 个实例化) | 2270 ms | 70.6% | 10.66 GiB | 14.2 ms | ~60% |
| vec::kernel Q6_K(type 14) | 797 ms | 24.8% | 3.84 GiB | 5.1 ms | ~62% |
| vec::kernel Q8_0(type 8) | 216 ms | 6.7% | 1.30 GiB | 1.7 ms | ~77% |
| vec::kernel Q5_K(type 13) | 18 ms | 0.6% | 0.11 GiB | 0.14 ms | — |
| attention small_t TC (int8 KV) | 251+24 ms | 8.6% | KV ~0.87 GB/tok | ~1.1 ms | ~37%(含计算) |
| quantize/rmsnorm/GDN/其余 | ~210 ms | ~6.5% | — | — | — |

(占比合计 >100% 因为多 stream 重叠。)

## 3. GGUF 侧剩余空间分析(ncu + op 级基准)

### 3.1 op 级隔离测量(ninfer_linear_bench 同款 harness,真实时钟,cold L2)

| 类型 | 17408×5120 t=1 | 5120×17408 t=1 | 占比 |
|---|---|---|---|
| q4_k | 644 GB/s | 644 GB/s | 80% 峰值 |
| q6_k | 506 GB/s | 430 GB/s | 54–63% |
| iq3_s(参考) | 527 GB/s | 534 GB/s | 66% |

(q4_k t=5:353–378 GB/s——T≥2 已在上个 session 用 `#pragma unroll 1` 修过 L1I,本数字为修复后。)q8_0 约 600–700 GB/s(75–87%)。

### 3.2 ncu 隔离 replay(60 个 decode launch,profile/q4k_ncu.ncu-rep)

- 注意:ncu 默认锁 base clock,绝对时间偏慢,不能与 nsys 生产时间直接比;但 dram__bytes 是 DRAM 层计数,带宽比值仍有参考意义。
- 隔离下 Q4_K 各实例化 83–94% 峰值、Q8_0 75%、Q6_K 49%(base clock)——与 op 级基准一致:Q4_K 隔离接近峰值,慢的是 Q6_K。

### 3.3 缺口分解(27K plain,32.0 tok/s = 31.25 ms/tok;理论带宽上限 21.8 GB/tok ÷ 803 = 27.2 ms/tok = 36.9 tok/s)

1. **Q6_K 内核低效(~2–3 ms/tok,~7–9%)**:op 级仅 54–63% 峰值。根因:210 B 行只有 2 B 对齐,解码每 32 元素 slice 用 16 次 2 B 加载(ql/qh 各 8),LSU 指令翻倍且无法向量化。修复需要转换端重新打包布局(行对齐 16 B,如 256 B 填充块,+22% Q6_K 流量,预计净收益 ~2–3%),是转换配方 + 新 decoder + 测试的大工程。
2. **多 stream 争用(Q4_K ~6 ms/tok,~19%)**:nsys 并发下 Q4_K 有效 484 GB/s vs 隔离 644 GB/s。decode 图里 965 launch/token,vec 大核与 rmsnorm/quantize/GDN/attention 小核跨 stream 共享 SM。修复 = GEMV+后处理融合(现有 F1 swiglu 融合的同方向)或调度重构,高工程量高风险。
3. **attention/KV(27K ~5 ms/tok)**:int8 KV 读 3.6 GB/tok,已近带宽;82K 时占比升至 ~20%,无生产选项可再降。
4. **小核(rmsnorm/quantize/gdn/conv ~2–3 ms/tok)**:各已很小,融合收益被 #2 吞没。

### 3.4 结论

- GGUF 侧现实可兑现的剩余空间:**~3–6%**(小核 + Q8_0/Q5_K 边角),全部在低收益区。
- 大空间项(Q6_K 布局重打包 ~2–3%、Q4_K 争用融合)是周级工程,且与 NVFP4 的差距(2.3–5.0%)同量级——**即使做完,GGUF decode 也只会接近而非超过 NVFP4**;差距本质是 NVFP4 的 4 bit 值 + E4M3 标量解码比 Q4_K 的 superblock 解码便宜(每元素整数 ALU 更少、行 16 B 对齐)。

## 4. 1.7.0 / 1.5.0 A/B + 短 prompt(2026-10-22,同会话背靠背)

短 prompt 982 tok / 27.2K(27,222 tok),greedy,96 输出,int8 KV,各侧最快 chunk(NVFP4 2048 / GGUF 4096)。
原始日志:<code>v17_ab_runs.txt</code>(12 case + 1 复测)。

| prompt | 引擎 | 工件 | plain | MTP d4 | prefill |
|---|---|---|---:|---:|---:|
| 短 | 1.7.0 | NVFP4 v3 | 34.3 | **84.7**(58.9%) | **1,000** |
| 短 | 1.7.0 | GGUF v3 | 33.3 | 68.4(51.6%) | 786 |
| 短 | 1.5.0 | NVFP4 v3 | 33.9 | 77.6(52.5%) | 960 |
| 短 | 1.5.0 | GGUF v3 | n/a | n/a | n/a |
| 27.2K | 1.7.0 | NVFP4 v3 | — | 68.3(50.0%)×2 复测 | 975 |
| 27.2K | 1.7.0 | GGUF v3 | — | 59.5(45.5%) | 969 |
| 27.2K | 1.5.0 | NVFP4 v3 | — | **75.8**(54.6%) | 977 |

- **1.5.0 无法加载 v3 GGUF 工件**:1.5.0 loader 不支持 v3 帧的 F32 GDN 控制,启动即
  `ERROR startup failed | inspecting artifact`——1.5.0 从未可运行此 GGUF,表中 n/a。
- **1.7.0 vs 1.5.0(同 NVFP4 工件)方向随 prompt 反转**:短 prompt 1.7.0 +9.1%(84.7 vs 77.6),
  27.2K 1.5.0 +11.0%(75.8 vs 68.3);两个方向的差距都主要由接受率驱动(同一工件同一 prompt,
  两版引擎 draft/target 数值不同,greedy 下分叉),单轮时间在两版间差 ~5% 以内。
- 1.7.0 测得最高吞吐:**短 prompt MTP 84.7 tok/s**(58.9%)。
- **接受率差距调查**(2026-10-22,同 NVFP4 工件):greedy 256-token 文本流对比 + KV dtype × 引擎 2×2 +
  bf16 KV 复测(数据:<code>txt_ab.txt</code>、<code>kv22.txt</code>、<code>kv22b.txt</code>、<code>txt_*.json</code>):
  ① 同引擎内 MTP 流与 plain 流逐字符一致;② 跨引擎 plain 流早期分叉(10K 第 12 字符、1K 第 49、
  27.2K 第 115);③ bf16 KV(无量化)下两版仍分叉(10K 第 66 字符)→ merge 后计算路径数值差异是主因;
  ④ 27.2K MTP d4: int8 1.7.0 50.0% vs 1.5.0 54.6%(差 4.6pp),bf16 48.8% vs 52.0%(差 3.2pp)——
  计算路径差 ~3.2pp 为主,int8 量化再放大 ~1.4pp;⑤ 短 prompt 方向反转(1.7.0 58.9% > 1.5.0 52.5%),
  非「越长越差」的系统性退化。

## 5. 备注 / 未完成

- IQ3_S 复测已取消:`Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` 是 Qwen3.8 基座,不是 Swift-1.5 v3 家族,不属于本 A/B;生产 `llama-cpp-text` 服务跑的是 Q4_K_M 工件,本矩阵已给出其最快配置(ch4096 + MTP d4)。
- 生产 compose 若要把 27K prefill 从 920 提到 988 tok/s,可把 `--prefill-chunk 2048` 改为 4096(本 session 未动生产配置)。
