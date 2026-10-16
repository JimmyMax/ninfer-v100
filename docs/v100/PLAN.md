# V100 (sm_70) 加速与 Gemma 支持计划

> 本目录（`docs/v100/`）是**本仓库对上游 NInfer-all 的 V100 增量工作**的唯一文档入口。
> 上游文档（`docs/*.md`）保持原样不改动，以减少跟进上游时的合并冲突。

## 1. 目标

一台 **Tesla V100-PCIE-32GB**（`sm_70`），要同时拿到两件事：

1. **模型覆盖**：能跑 Gemma 4 26B-A4B（含 MTP assistant 与 vision），以及上游支持的 gguf 家族
   （Qwen3.8 GSQ-RCO、Ternary Bonsai 2、Qwen3.8-Flash-Next）。
2. **Volta 加速**：把 `JimmyMax/ninfer-v100`（下称 *fork*）在 Qwen 家族上积累的 Volta 特化加速
   技术，迁移到这条线上，让 Gemma 也能拿到同等量级的收益。

## 2. 为什么是"正向"迁移，而不是把 Gemma 搬进 fork

两条线的血缘（2026-10-12 实测）：

| | fork（`ninfer-v3-v100` 的 `master`） | 上游（`iamwavecut/ninfer-all` `master`） |
|---|---|---|
| 定位 | **V100 专精线**，只服务 V100 | 多卡主线：3090/4090/5090/PRO 6000 |
| sm70/volta 专用文件 | 72 个 | 18 个（≈4,985 行，来自 PR #58 一次性移植） |
| V100 相关提交 | 44 个 | 8 个（全在 PR #58 里） |
| Volta 加速资产 | 83 文件 / 11,574 行（`volta`/`qpn`/`sm70`/`dflash2`） | 兼容路径（模拟 `cp.async`/`ldmatrix`/`m16n8`，宽 T 走 cuBLAS 反量化） |
| Gemma 4 / `gguf_*` 格式 | **不支持** | 支持（22 文件 / 6,700 行的 `src/models/gemma4`，`ops/linear/gguf` 45 文件 / 4,479 行） |

两条线自 2026-09-09（`b88c0f6f`）分叉，此后 fork 独有 72 个提交、上游独有 1,438 个提交，
双方都改过的文件 198 个（fork 改过 341、上游改过 2,494）。

**反向搬运（Gemma → fork）的代价**：需要搬 `src/models/gemma4`（22 文件 / 6.7k 行）+
它缺失的框架层（`ops/linear/gguf` 4.5k、`moe_experts` 2.5k、`kv_cache` 3.0k、
`models/qwen3_5/frontend` 9.8k、`runtime/engine/gemma4_core` 2.4k）+ artifact 的 gguf 格式层，
合计 **22,000+ 行**；而且 `core/weight_view.h`、`artifact/schema.h`、`ops/kv_cache`、
`qwen3_5/frontend` 在上游都已演进，API 不一致 → 属于"照着重写"而非复制。
做完之后 fork 会永久持有两套 `frontend`/`ops`，上游每加一个模型家族都要再搬一次。

**正向搬运（fork 的 Volta 加速 → 上游）的代价**：只动加速内核本身，
即 fork 那 83 个文件里真正可迁移的部分，适配量级 **2–4k 行**（见 §5 的 S2–S4）。

> 结论：本仓库（`gemma` 分支）以**上游 master 为基座**，在此之上做 V100 加速增量。
> fork 的 `master` 继续作为 Qwen 生产线保留，两条线并存、按需切换。

## 3. 已测基线（2026-10-12，V100-32GB）

同一张卡、同一 artifact、同一参数（`--max-context 200000 --kv-capacity auto --kv-dtype int8
--max-concurrency 2 --spec mtp --draft-tokens 4 --lm-head-draft --preserve-thinking --vision`）、
预热后测量：

| 模型 / 负载 | 上游 sm70 线（本分支，`ninfer:v100-gemma4`） | fork（`ninfer:v1.5.0`） | 倍数 |
|---|---:|---:|---:|
| `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4`，decode | **14.2–14.8 tok/s** | **62.1 tok/s** | **≈4.3×** |
| 同上，prefill / TTFT | 81.6 tok/s / 975 ms | 191.5 tok/s / 510 ms | 2.3× |
| MTP 接受率 | 32–35% | 32.8% | 持平 |
| `Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP-nvfp4`（gguf 混合格式），decode | **157.5 tok/s**（`--draft-tokens 2`，接受率 56%） | 不支持该 artifact | — |
| 同上，prefill / TTFT | 487 tok/s / 82 ms | — | — |
| 同上，`--draft-tokens 4` | 119.0 tok/s（接受率 39%） | — | — |

**结论**：上游 sm70 是"能跑"的兼容路径（Qwen 上比 fork 慢 4.3×），fork 是 V100 专精实现；
Gemma 只有上游能跑。加速增量应当往上游线上做。

详细环境与原始日志见 [`baseline-2026-10-12.md`](baseline-2026-10-12.md)。

## 4. 分支与上游跟进策略

| 分支 | 基座 | 角色 |
|---|---|---|
| `master`（原仓库） | geoffwatts/ninfer-v100 + 本地 72 提交 | **Qwen 生产线**，V100 专精实现，继续调优 |
| **`gemma`（本分支）** | `iamwavecut/master`（`fcc4928a`） | **Gemma + gguf 家族线**，在其上做 V100 加速增量 |

跟进上游：

```bash
git fetch iamwavecut master
git switch gemma && git merge iamwavecut/master     # 本分支有自己的提交后，用普通 merge
```

推送（首次需要指定目标远端，因为本地分支默认跟踪 `iamwavecut/master`）：

```bash
git push -u origin gemma
```

* 不修改上游已存在的文件（除非该文件正是加速补丁的目标）；新增能力优先落在**新文件**里，
  把合并冲突压到最小。
* 每阶段的性能结论写回本目录，不改上游的 `docs/performance.md`。

## 5. 路线图

状态：`☐ 未开始` / `◐ 进行中` / `☑ 完成`

### S0 基座与工具链 ☑（2026-10-12）

* [x] 建立 `gemma` 分支（基座 `iamwavecut/master`）+ 独立 git worktree
* [x] V100 构建配方：`Dockerfile.sm70.local`（CUDA 12.8 + `-DCMAKE_CUDA_ARCHITECTURES=70`）
* [x] 快速重打包配方：`Dockerfile.sm70-runtime.local`（复用 `ninfer:v100-stable` 运行时基底）
* [x] 两线对照验收脚本：`bench/v100/compare_lines.py`
* [x] 基线数据落档：`docs/v100/baseline-2026-10-12.md`
* 验收：`ninfer:v100-gemma4` 镜像可从本仓库源码重建；`ninfer-serve --version` 正常；
  Gemma 4 在 V100 上加载并出词（见 §3）。

### S0.5 用 Qwen 家族验证"正向搬运"可行 ☑（v2 已移植并通过 oracle；长上下文只 +3~6%，瓶颈不在注意力——见 [结果](S0.5-result-2026-10-13.md) 与 [oracle/长上下文](S0.5-oracle-and-longctx-2026-10-13.md)）

> S0.5 的结论改写了后续优先级：Qwen 的 4.3× 差距经[定位测量](profiling-nvfp4-2026-10-13.md)
> 精确落在 **NVFP4 投影路线**（同一形状：NVFP4 85 GB/s / DRAM 9.5%，而 fp8 622、q4 503、
> q5 480、q8 601 GB/s）——不是注意力，也不是 fp8/q4/q5/q8；Gemma 4 的杠杆则在
> **MoE 专家 + GGUF 投影**（S3/S4）。S2 的注意力工作降级为"顺带"，NVFP4 移植升为下一步的首要项。

以**同源代码**的注意力内核作为样板，证明 fork 的 Volta 加速能落到上游线上。

* 目标文件（两侧同源，可 patch 而非重写）：
  `src/ops/softmax_attention/dense/causal_cache/{prompt_i8.cuh,small_t_i8.cuh,prompt_volta.cuh,
  small_t_*_volta.cuh,causal_softmax_attention.cpp,launch.h}`
* 来源提交（fork）：`9e2a9bff`（Volta prefill retune + INT8 small-T decode attention v2）、
  `db947f95`（small-T PV 累加链改 fp16）、`6856d323`、`c380*` 系列（`perf(sm70)` 端口）
* 验收：上游线上 Qwen3.8-27B nvfp4 的 decode 从 ~15 tok/s 提升到接近 fork 的量级；
  与 fork 的输出在贪心解码下逐 token 一致；用 `bench/v100/compare_lines.py` 记录前后对照。
* 预估：3–7 天。**这一步是整条路线的前提，验证通过才继续投 S2/S3。**

### S0.6 FP8 投影的 Volta QPN + 加载时 prepack ☑（2026-10-14）

NVFP4 阶段（见 [profiling-nvfp4-2026-10-13.md](profiling-nvfp4-2026-10-13.md)）做完后，账本里剩下的最大
单项是 fp8（130 张量 / 9.76 GiB，含每层的注意力输入投影与输出投影）。把 fork 的 FP8 QPN + CUTLASS
路径按"QPN 拥有全部宽度"的结构搬过来，并让权重在加载时就重排成 QPN8 的 fragment 序。

* [x] QPN 直读 BF16 激活、零 workspace（保 workspace 的"峰值 == 容量"契约）
* [x] 宽 T 走 CUTLASS（attn_input / linear_add / gdn_input / plain linear；swiglu 回上游阶梯）
* [x] hook 放在 route 查表之前，只截 A16 政策族；容量绕过 route table
* [x] 加载时 prepack（`artifact/permute_fp8_qpn.cu`）+ 256 MiB 宽 T 上限
* 验收：43/43 oracle 通过；端到端 42K prefill **407.8 → 951.2 tok/s**、TTFT **40.2 → 17.4 s**，
  decode 31.0 → 31.1（短）/ 32.9（42K）；42K 输出逐字相同。详见
  [fp8-qpn-2026-10-14.md](fp8-qpn-2026-10-14.md)。
* 生产：`docker-compose.nvfp.yml` 已切到 `ninfer:v100-fp8`（制品同时从 v2 换成同目录的 v3 文件，
  上游代码只支持 v3 制式）。

### S0.7 解码注意力线侦察 + 长上下文 split 修复 ◐（2026-10-14）

* [x] 定位：只看 t=1，本线 vs fork 只差 1.26×；t=5 投机验证遍差 2.06×（42K prefill 本线已反超 951 vs 622）
* [x] 发现 v100 的 `attn_i8_small/*` 路由键为空是**设计如此**（sm_70 走自己的解码注意力路由，
      `route_catalog.cpp` 明写只标定 GDN prefill），不是缺配置
* [x] 移植 fork `c58a92ba`：Volta `SmallTMaximumSplits` 85 → 560 + `small_t.cuh` 的 256 折叠
      （62K/113K/140K token 上 +0.6%/+1.1%/+1.4%，接受数一致 → 数值无变化）
* [x] 可用绕行落地：`--adaptive-mtp` 避开定宽 t=5 异常，短 **28.5 → 40.9**、42K **33.1 → 48.0** tok/s
* ☑ 真因已定位（S0.8）：**不在注意力**，是 FP8 GDN 输入投影（16384×5120）在 W=5/6 落到
  sm_70 上只有 10% 带宽的 `fp8_a16_ksplit_mma` → 48 层 × 0.89 ms = 42.8 ms/轮（占 77 ms 轮的 56%）
* ☐ 下一步：t=1 还余 1.26×（需要新的 kernel 级归因，见 S0.8 的方法学）
* 详见 [attention-2026-10-14.md](attention-2026-10-14.md)。

### S0.8 nsys 归因 + FP8 GDN 投影 W=5/6 改走融合核 ☑（2026-10-14）

用 nsys（侧车容器 `cuda-nsight-systems-12-8` + `--cuda-graph-trace=node`）对一轮 decode 做
kernel 级归因，把持已久的 t=5 异常钉死在单个算子上。

* [x] nsys 侧车剖析跑通（ncu 因 `ERR_NVGPUCTRPERM` 不可用；图内子 kernel 必须 `--cuda-graph-trace=node`）
* [x] 归因：`fp8_a16_ksplit_mma_kernel` 48 次/轮、**42.8 ms/轮**（0.89 ms/次 vs 融合核 0.40 ms）
* [x] 修复：`b1_a16_plan` 在 `NINFER_SM70` 下把 W=5..6 并入融合区间（W=4 仍走 SIMT）
* 验收：oracle `ctest -R gdn_input` 12/12；短上下文固定 d=4 **28.5 → 44.1** tok/s（+55%）、
  `--adaptive-mtp` 40.9 → 44.4、27.2K 上下文 49.2 → 52.8；prefill 不变；宽度曲线变平
* 生产：`ninfer:v100-fusedgdn`
* 详见 [gdn-fused-5-6-2026-10-14.md](gdn-fused-5-6-2026-10-14.md)。

### S0.9 与 master 的差距收敛：解码追平 + prefill 残差定位 ☑/◐（2026-10-15）

以 `ninfer:v1.5.0`（仓库 `master` 分支的构建）为基准，逐算子定位并消除差距。

* [x] **FP8 GDN 输入投影走上 QPN**：`fp8_gdn_input_a16_dispatch` 对预打包权重且 `t<=32` 走 QPN，
      `b1_a16_plan` 在 sm_70 上一律物化，加载时把 16384×5120 纳入 prepack。
      收益：T=5 每层 909 → 295 µs，固定 d=4 端到端 **28.5 → 44.1 tok/s（+55%）**。
* [x] **输出头（248320×5120）纳入 prepack**（按权重角色而非形状过滤）：约 1 ms/轮。
* [x] **FP8 QPN 的 FP16 激活暂存**（attn 输入、plain linear 用 arena；GDN 输入用加载期预留的
      320 KiB 固定缓冲，图捕获中绝不 malloc）：同形状同布局 548 → 768 GB/s。
* [x] **小 T 分裂折叠越界修复**（正确性）：三处 reduce 的 `weights` 缓冲按几何上限分配，
      修掉 `d256-h16-kv2` 在 65536 keys（273 分裂）下的数值崩坏。
* 验收（干净单实例对照）：解码每遍成本四个配置全部持平（短/27K × 无投机/MTP4，差 0.2–2.2%）；
  生产配置（MTP4）短上下文 **87.0 vs 62.9 tok/s（+38%）**，27K 每轮持平（41.30 vs 41.21 ms）。
* ☐ **prefill 残差 5–14%**：本线 876 vs master 1020 tok/s（27K）。已排除 chunk 粒度
  （1024/2048/4096/8192）、prompt 开关（pack_gqa/pv_f16/prompt_fast）、prompt 内核选择
  （master 的 `prompt_volta` 是 quadratic，移植后 87.8 tok/s）、反量化向量化（±2% 噪声内）、
  `--prefill-cublas`（+3%）、`--fast-prefill-kernel`（+2.5%）。已构建 `master` 分支参考二进制
  以便做 prefill 逐算子对账。
* 生产：`ninfer:v100-deq`
* 详见 [gap-closure-2026-10-15.md](gap-closure-2026-10-15.md)。

### S0.10 prefill 残差归因到具体内核 + GDN 路由修复 ☑/◐（2026-10-15）

首次拿到 **master 参考二进制与本线在同一 27K 请求下的 kernel 级剖析**（此前 nsys 一直采不到事件，
原因是收尾时用 `pgrep -f ninfer-serve` 会先匹配到 nsys 自身，把 nsys 杀掉既不写报告又留下占显存的
孤儿进程；正确做法是 `pkill -x <精确进程名>`）。

* [x] **sm_70 的 GDN prefill 改走 recurrent 单内核**（上游 sm_70 就是 `T_full = 0`）：
      `ninfer_gated_delta_net_bench` T=1024..8192 上 **2.90–3.32×**（T=2048 7750 → 2436 µs）；
      oracle `ctest -R "gated_delta_net|gdn"` **22/22 通过**；27K prefill GPU busy 里 GDN
      **3.75 → 2.18 s**，e2e prefill **873.9 → 903.9 tok/s（+3.4%）**。`NINFER_GDN_RECURRENT=0` 可回退。
* [x] **残差定位到单一内核**：27K prefill GPU busy master 28.60s vs 本线 31.93s，其中 CUTLASS 持平
      （16.14 vs 16.32s）、GDN 本线更快（2.82 vs 2.18s）、**prefill 注意力 master 3.93s vs 本线
      7.28s（每 token 1.85×）**。master 用 vendored llama.cpp 的 `flash_attn_ext_f16<256,256,32,2>`
      经 `volta_flash` 路线（paged int8 KV → gather 成 FP16 → 逐 Q-block flash），本线**没有**该路线，
      走自研 `prompt_kernel`。
* ☐ **移植 `volta_flash` prefill 注意力路线**（唯一剩下的主要项，预计 3.3s GPU busy → 追平 1010 tok/s）。
      依赖在本线齐全（`CausalD256H24Kv4/16Kv2`、`PagedKVBatchLayerView`、`Int8Group64`、paged KV 地址
      helper；缺的只是 `round_up_keys` 之类小 helper）。
* 生产：`ninfer:v100-gdnrec`
* 详见 [gdn-recurrent-and-prefill-gap-2026-10-15.md](gdn-recurrent-and-prefill-gap-2026-10-15.md)。

### S0.11 Volta flash-attention prefill 路线移植 ☑（2026-10-15）

把上游参考构建的 prefill 注意力（vendored pinned llama.cpp `flash_attn_ext_f16`，tensor cores）接到本线。

* [x] **数值正确**：`--int8-prompt-only`（默认即走该路线）**PASS**；主注意力测试（全存储）**PASS**；
      bf16 + flash 端到端输出正确。
* [x] 用 oracle harness 修掉三个真 bug：容量门控（`min_width<=128` → `max_width>=128`）、
      分配未按 256 取整（统一到 `causal_attention_volta_flash_key_extent()`）、
      **每 Q-block 的 key 上界漏掉前缀**（主因）。
* [x] **决定性发现**：慢的不是 staging（gather+mask+convert 仅 0.6 ms/次 = 1%），而是**桥后面的内核版本**
      ——`gguf::flash_attention`（较新 ggml fattn）1.8 ns/query-key 对，pinned vendored 内核 0.75 ns。
      移植 `third_party/llama_cpp_fattn` + `ops/launcher/volta_flash_llama.cu`（直接 grid，无 ggml 桥）后：
      bench 41.1 → **25.1 ms**；27K e2e TTFT 30.1 → **27.3 s**（上游参考 27.0 s）⇒ **prefill 追平上游**。
* [x] 路线默认开启（int8/bf16），`NINFER_ATTN_VOLTA_FLASH[_INT8]=0` 可关。
* 剩余：反量化 +0.49s（本线刻意选的精确解码，不动）；27K decode −7%（MTP 接受率差异，与 prefill 无关）。
* 详见 [volta-flash-prefill-2026-10-15.md](volta-flash-prefill-2026-10-15.md)。

### S1 Gemma 4 的 V100 route/device profile 调优 ☐

纯配置与标定，不动 kernel。

* `--draft-tokens`（1/2/3）、`--prefill-chunk`（512/1024/2048）、`--kv-dtype` 全 9 种
  （Gemma 4 支持 int8/fp8/nvfp4/rk4v4/rk2v4-e8/…，其滑窗 ring 与全局 V 平面各一张表）、
  `--vision-residency`（resident/overlay/cpu）、`--host-kv-mib`、CUDA graph allowance。
* 参考上游自己的 16GB 卡数据（`docs/performance/tesla-v100-2026-10.md`：Gemma 4 Q4_0 短对话
  164 tok/s 无投机 / 226 tok/s MTP-2），本机 32GB 应更高；当前 157.5 tok/s 有提升空间。
* 产出一张 `--kv-dtype × max-context × drafts` 的 V100-32G 参数表。
* 预估：2–5 天。

### S2 Gemma 4 的 Volta 解码注意力（attention v2 技术落地） ☐

**Gemma 4 不使用共享的 `causal_cache` 家族**，它有自己的
`src/models/gemma4/{kernels.cu,kv_kernels.cu,kv_codec.cuh}`（25 层滑窗 window=1024 / head 256 /
16q·8kv，5 层全局 head 512 / 16q·2kv，且全局层只缓存 V、K 由 V 派生）。

* 要移植的是**技术**而非文件：key-split 8-warp `Bc=64` 解码、`byte_perm` int8→fp16 反量化、
  fp16 PV 累加链、prefill 阈值重调；两个几何分别实现。
* 验收：Gemma 4 decode 提升百分比 + 数值等价（上游自带 FP64 oracle 测试：
  `tests/models/gemma4/*`，`ninfer_gemma4_kv_test` 等）。
* 预估：1–2 周。

### S3 Volta GGUF 投影加速 ☐

Gemma 4 的权重是逐张量 GGUF block 格式（Q4_K / Q5_0 / Q6_K / Q8_0 / Q4_0（草稿头）/ BF16（视觉塔）/ FP32）。
上游在 Volta 上的现状：≤8 列走 NInfer 的 DP4A 向量核，>8 列走 ggml 的 DP4A 矩阵核，
≥65 列反量化到 FP16 走 cuBLAS。V100 没有整数张量核但有 DP4A，这里有明确空间。

* 可借鉴 fork 的手法：权重的 Volta 布局 prepack、双缓冲分组预取
  （`docs/V100-PERF-PLAN.md` 的 `2b3f16cd` / `9913bc22` 条目）、epilogue 融合、
  激活分块暂存。
* 验收：`pp2048` 与 decode 段耗时对照表（`ninfer_bench`）。
* 预估：1–2 周。

### S4 MoE 专家投影的 Volta 路径 ☐

Gemma 4 是 128 专家 × 8/token、专家宽度 704，A4B 的算力几乎全在专家上。

* 借鉴 fork：`437e9f98`（按 route 选流水线深度、按工作量定 grid）、
  `7f14d963`（MoE down 尾部预热下一层投影的 L2）、`ce954918`（共享专家 down 权重预取）。
* 验收：MoE 段耗时 + 端到端 decode。
* 预估：约 1 周。

### S5 视觉塔在 Volta 上的驻留策略 ☐（按需）

27 层、宽度 1152、BF16 矩阵，上游有 `--vision-residency resident|overlay|cpu`。
fork 侧有 `perf(sm70): accelerate vision encoding` 可对照。
验收：图片（768×512，260 soft token）与 32 帧视频的编码延迟。

## 6. 构建与部署

### 完整构建（约 1 小时，中间产物约 10 GB）

```bash
# 建议把编译输出放到容量大的盘上，避免根分区吃紧
docker run -d --name ninfer-sm70-build \
  -v "$PWD:/src" -v /mnt/data/ninfer-build-sm70:/build \
  nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 sleep infinity
docker exec ninfer-sm70-build bash -c '
  sed -i "s|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g" \
    /etc/apt/sources.list.d/ubuntu.sources
  apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake ninja-build pkg-config libavformat-dev libavcodec-dev \
    libavutil-dev libswscale-dev libcurl4-openssl-dev zlib1g-dev'
docker exec ninfer-sm70-build bash -c '
  cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF &&
  cmake --build /build --parallel 16 --target ninfer-serve'
```

> * 必须 CUDA 12.8/12.9：CUDA 13 起 nvcc 不再编译 `sm_70`（`CMakeLists.txt` 会直接报错）。
> * `third_party/` 全部 vendored，配置期不下载任何依赖；外部依赖只有
>   `libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7 libcurl>=7.85`。
> * 本机直连可用，**不要走 `proxy.n.jimy.top:1080`**（对本项目 apt/下载不稳，且直连更快）。

### 打包镜像

```bash
docker build -f Dockerfile.sm70.local         -t ninfer:v100-gemma4 .        # 全量
# 或复用已编好的二进制（秒级）：
cp /mnt/data/ninfer-build-sm70/apps/ninfer-serve docker-runtime/
docker build -f Dockerfile.sm70-runtime.local -t ninfer:v100-gemma4 docker-runtime/
```

### 内核开发的增量循环

首次全量构建会建立 ninja 的依赖数据库；之后改动任何源文件只需在**同一个构建目录**里重跑：

```bash
docker exec ninfer-sm70-build bash -c 'cd /build && ninja -j 16 ninfer-serve'
cp /mnt/data/ninfer-build-sm70/apps/ninfer-serve docker-runtime/
docker build -f Dockerfile.sm70-runtime.local -t ninfer:v100-gemma4 docker-runtime/
```

只重编受影响的对象再链接（约 1–3 分钟），重打包 1 分钟内完成。

> ⚠️ 不要用 `git worktree add` / 重新 clone 得到的源码目录去复用旧构建目录：checkout 会刷新所有
> 文件的 mtime，ninja 的 deps 记录随之失效，会退化成 **全量重建（约 1 小时）**。
> 需要新工作树时，让构建目录从新源码树重新做一次全量构建，然后再进入增量循环。

### 运行

生产编排在本机 Docker 仓库（不在本仓库）：
`/home/max/code/docker/ai/ninfer-v100/docker-compose.yml`（Gemma 线）与
`docker-compose.swift.yml`（fork 的 Qwen 线），30–60 秒切换。示例见
[`deploy/compose.gemma-v100.example.yml`](../../deploy/compose.gemma-v100.example.yml)。

## 7. 验收工具

```bash
# 两线对照：同一模型、同一参数、预热后各跑 N 次，输出 markdown 表
python3 bench/v100/compare_lines.py \
  --image ninfer:v100-gemma4 --image ninfer:v1.5.0 \
  --model /workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer \
  --extra-args "--spec mtp --draft-tokens 4" --runs 2
```

每阶段的性能结论请以该脚本输出为准，避免"单次冷启动"造成的误判
（冷启动首请求 decode 可低至稳态的 1/4）。

## 8. 风险与回退

| 风险 | 处理 |
|---|---|
| 上游继续演进，加速补丁与上游 ops 层冲突 | 只改必要的加速文件；每阶段一次小 merge，不积压 |
| 加速补丁造成数值不等价 | 每步跑上游自带的 FP64 oracle 测试 + 贪心逐 token 对照 |
| 单卡显存被 Gemma 4 占满（16.9 GiB 权重 + 3.3 GiB KV@200K） | 降低 `--max-context` 或 `--max-concurrency`；`--vision-residency overlay` |
| 构建机根分区吃紧（<16 GB 可用） | 编译输出放 `/mnt/data`；`docker builder prune` |
| V100 ECC 打开，损失约 6% 容量与 12% 带宽 | 追求极限时可 `nvidia-smi -e 0` + 重启（牺牲显存校验） |
| 官方镜像不可用于 V100（只有 sm86/sm120a） | 只能自建；本目录的两个 Dockerfile 就是为此 |
