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
| **`main`（本分支）** | `origin/main`（`JimmyMax/ninfer-v100`，v1.6.1，`12a74bd`）+ merge `upstream/master`，本地发布为 **v1.6.2** | **Gemma + gguf 家族线**，在其上做 V100 加速增量 |
| `backup/imported-gemma` | 2026-10-20 重建前的导入快照 | 仅作回退参照，不再推进 |

`main` 的真实血缘：`origin/main` 是上游 `fcc4928a`（2026-10-12，上游 PR #58）的后代，此后
**没有再次 merge 上游**，因此相对 `iamwavecut/master` 落后 148 个提交（其中 39 个触及 V100 关键
路径）。`git merge-base origin/main upstream/master` = `fcc4928a`，所以可以直接 merge。

跟进上游：

```bash
git fetch origin main && git switch main                 # origin = JimmyMax/ninfer-v100
git fetch https://github.com/iamwavecut/ninfer-all.git master:refs/remotes/upstream/master
git merge --no-ff upstream/master                       # 2026-10-20 试合并：12 个冲突 / 459 自动
```

* 不修改上游已存在的文件（除非该文件正是加速补丁的目标）；新增能力优先落在**新文件**里，
  把合并冲突压到最小。
* 每阶段的性能结论写回本目录，不改上游的 `docs/performance.md`。
* 2026-10-20 的冲突清单与解决取舍见
  [v1.6-vs-v1.5-2026-10-20.md](v1.6-vs-v1.5-2026-10-20.md) §3–§4。

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

### S0.12 解码投机配置：关掉 n-gram 混合草稿 ☑（2026-10-16）

27K prefill 追平后，decode 仍落后。逐项 A/B（同一 v3 制品、27K prompt、256 输出）：

| 配置 | decode | 接受率 |
|---|---|---|
| 原默认（`--ngram-draft-tokens` 默认 15 + `--adaptive-mtp`） | 64.2 tok/s | 混合稀释 |
| 纯 MTP + adaptive + d4 | 75.2 | 53.8% |
| **纯 MTP、无 adaptive、d4（新默认）** | **78.3** | 54.3% |
| 纯 MTP + adaptive + d6 | 60.0 ✗ | 48.9% |
| 纯 MTP + `--full-draft-head` | 71.8 ✗ | — |
| 上游参考构建 | 81.7 | ~58% |

* 根因：`--spec` 开启时 `--ngram-draft-tokens` **默认为 15**，引擎因此把 n-gram 拷贝草稿混进 MTP
  （日志 "mixed speculation"），稀释了接受率。
* **质量零影响**：贪心输出逐字相同（76 字符完全一致），短上下文 60.9 → 61.9 tok/s 无回退。
* 生产已改为 `--ngram-draft-tokens 0`、去掉 `--adaptive-mtp`：27K decode **64.2 → 78.3（+22%）**，
  与上游参考（81.7）差 4%（接受率 54.3% vs ~58%）。

### S0.13 反量化：精确解码没有更快的空间 ☑（2026-10-16）

上游的"移位技巧"反量化比本线的精确解码快（27K prefill 里 1.674s vs 2.166s），于是检查能否既精确又更快。

* 先按非 QPN 路径的结构把 `decode_qpn_e2m1_word_exact` 向量化（每字节一次 fp4x2 转换、两个值一次
  `__float22half2_rn`，逐位等价）：nvfp4 四个 oracle 测试全过、贪心输出逐字相同 ✓，
  **但 kernel 时间不变**（1.2335s vs 1.230s）✗。
* 原因（更正我先前的估计）：该内核**已经是带宽受限**——每次调用实际搬 ~468MB（packed 读 + FP16 写），
  0.6ms ⇒ **~780 GB/s**，就是 V100 的实用上限；不是我先前算的 330 GB/s/指令受限。
* 结论：**精确解码没有指令级空间**。真要比上游省时间，只能少搬字节（Volta 没有 fp4 张量核，
  必须物化 FP16）或少反量化张量——而多出来的那些张量（例如我们额外纳入 prepack 的输出头）
  在别处换来了收益，所以维持现状。该改动已回退。

### S0.14 长上下文 decode：旋钮已穷尽，剩余 4% 在引擎内部 ☑（2026-10-16）

27K decode 从 64.2 提到 78.3 tok/s 后，把接受率相关旋钮逐个测完（同 v3 制品、27K、256 输出）：

| 旋钮 | decode | 接受率 |
|---|---|---|
| **d4 + 无 adaptive + ngram0（现生产）** | **78.3** | 53.8–54.3% |
| d3 / d5 / d6 | 74.5 / 74.3 / 60.0 ✗ | 62.7 / 48.0 / 48.9% |
| `--mtp-attention-window N` | — | 默认 **0 = 全历史**（草稿精度最高），调小只会更差 |
| `--full-draft-head` | 71.8 ✗ | — |
| 上游参考构建 | 81.7 | 58.0% |

* 关键事实：上游二进制**没有** `--adaptive-mtp` / `--mtp-attention-window` / `--ngram-draft-tokens`
  这些开关，配置与我们现在完全一致（`--spec mtp --draft-tokens 4 --lm-head-draft`）⇒ 4pp 的接受率差
  不是开关差异，而在**引擎内部的 MTP 草稿链实现**。
* 每轮成本两边几乎相同（按 4 草稿/轮折算 ≈41.2ms vs 40.6ms）⇒ decode 差距由接受率解释，不是轮开销。
* 继续追需要改引擎解码环路（影响所有模型，风险更高），已超出"配置/单算子"的范围。

### S0.15 启动首请求 503：不是引擎问题，是测试脚本的竞态 ☑（2026-10-16）

收尾时查到：引擎在加载期间对请求返回 `503 {"code":"model_loading","message":"The model is still
loading. Retry shortly."}`，这是**正确行为**；就绪后第一条请求 200（实测：`/health` 在 t≤18s 返 503、
t=21s 返 200，紧接着的请求即 200）。

* 之前观察到的 503 全部出在 **`docker compose up -d` / `docker restart` 之后**：这些操作在复用已有容器时
  保留历史日志，脚本 grep "listening on" 匹配到了**上一次启动**的那行，于是在模型仍在加载时就发了请求。
* 结论：**引擎无需改动**；修的是工具。新增 `bench/v100/wait_ready.py`（轮询 `/health`，无需鉴权，
  不会陈旧），并在 8 个 A/B 脚本里于解析容器 IP 后插入一行探针；`compare_lines.py` 用其端口映射做同样的事。
* 记录备用探针语义：`GET /health` 加载中 503 / 就绪 200；`GET /v1/models` 加载中 503、就绪后需鉴权（否则 401）。

### S0.16 GSQ（IQ3_S 混合）制品的 prepack 可行性评估 ☑ 只评估、不改代码（2026-10-16）

同二进制/同请求实测：GSQ 比 nvfp4 制品 **prefill 慢 12%、decode 慢 10%**（且其 MTP 接受率更高 70.8% vs
62.5%）。kernel 级归因：差在**权重消费路径**——GSQ 是 18 种块量化格式的混合（平均 3.95 bit/weight，
IQ3_S 只占 3.39 GiB），运行期由约两万个 `gguf::dequantize_rows_kernel`/`VecStore` 小内核现场反量化，
CUTLASS 被打碎成 5.5 倍的碎 GEMM；注意力/GDN 两边完全一致。

* **方案 A（FP16 常驻 prepack）不可行**：全量 FP16 需 81.5 GiB（现 11.9 GiB），单个格式 +12.4 GiB、
  单个角色 +11.5 GiB，而运行期空闲只有 3.08 GiB。
* **方案 B（保持位宽、只重排布局，QPN 思路）可行**：显存 +0.3–0.7 GiB，需为 IQ3_S/IQ4_XS/IQ3_XXS/Q4_K
  等 4–5 种布局写"一次干净反量化"内核（覆盖 71% 张量字节），预期可追平 nvfp4 制品（约 +10%）。
* **方案 C（不改）建议**：生产用 nvfp4 制品；GSQ 作为长上下文选项（+150K 上下文 ≈ 那 7 GiB 权重之差）。
* 详见 [gsq-iq3s-prepack-assessment-2026-10-16.md](gsq-iq3s-prepack-assessment-2026-10-16.md)。

### S0.17 FP8 SwiGLU 宽 T：混合量化制品 prefill 慢 2× 的真因 ☑（2026-10-16）

用 v2 制品（升级成 v3 后）实测：我们 **TTFT 53.6s / prefill 508 tok/s**，master **27.1s / 1.00k**。
nsys 逐内核对比后，55.05s vs 28.78s 的 GPU busy 差**只来自一个内核**：
`fp8_simt_kernel<Fp8Geometry<34816,5120>, …, Fp8SwiGluOutput>` **27.67s × 54520 次、grid=(2176,1,1)**
（一维网格，T 在核内串行）——FP8 的 MLP gate_up 落到了 SIMT 后备。

* **根因**：`fp8_linear_swiglu_plan.cpp` 的 sm_70 钩子只接管 **t ≤ 32**（QPN 拆分），宽 T 交回上游
  A16 阶梯（每 4 个 token 一块的 small-t SIMT）；而 `34816×5120` 的 FP16 暂存 **356 MB > 256 MiB**
  的 CUTLASS 宽 T 上限，于是既不 CUTLASS 也不 QPN。其余 FP8 形状（≤178MB）都在限内，所以只有
  这一个中招。
* **不是缺 q4/q5/w8 内核的问题**：两个二进制里那族内核总计只占 0.2–4.5%，移植收益≈0。
* **修复**：新增 **FP32 出口**的宽 T CUTLASS 路线（`fp8_linear_swiglu_cutlass_sm70_launch_f32`，
  同一个 sm_70 mma，输出 `float` 平面、行缩放在 FP32 做、SwiGLU 直接读平面），t ≥ 33 走它；
  并修掉容量函数里 `min_tokens <= 32` 的前置判断（单 T 查询 T≥33 时会算 0）。满足上游 A16
  "激活前保持 FP32"的判据，不改测试、不动 v3/nvfp4 热路径。
* **验收**：oracle 7/7 通过（含 A16 的 T=1…1024、非线性精度判据、单 T 容量精确性）；v2 制品
  **53.6s → 27.3s / 997.6 tok/s**（master 27.1s / 1.00k），decode 82.7、MTP 69.2%（master 75.7 /
  63.9%）；生产 SWIFT 与 GSQ 逐项不变（1.00k / 78.3、884.7 / 66.6）。
* 详见 [fp8-swiglu-wide-t-2026-10-16.md](fp8-swiglu-wide-t-2026-10-16.md)。

### S0.18 v2 制品直读（不再需要离线升级）☑（2026-10-16）

引擎现在直接读 v2 原件：**27.3s / 998.5 tok/s**、decode 82.6、MTP 69.2%，与"升级成 v3 后再读"
（27.3s / 997.6 / 82.7 / 69.2%）**输出逐字一致**，与 master（27.1s / 1.00k / 75.7 / 63.9%）持平；
生产 v3 制品 SWIFT 同一二进制 **997.1 / 78.3 / 54.3%** 逐项不变，`ctest -R artifact` **4/4**。

* **前提**：v2 与 v3 的差别只在清单——实测同一制品 1184 个张量偏移在升级前后 **1184/1184 相同**，
  升级只是词汇改名 + 由角色名合成 1513 条 bindings + 由 identity 合成 components/uses/files。
* **实现**：`reader.cpp` 接受 v2 magic（16 字节头、无 UUID → 用目录文本+文件长度的 128 位 FNV-1a
  合成 ArtifactId）；新增 `src/artifact/v2_directory.{h,cpp}`，把 `{identity, objects}` 归一化成
  v3 形态的目录 JSON 再交给同一个 `parse_directory`。规则是 `tools/upgrade_ninfer_v2_to_v3.py`
  中 `make_directory` 的逐条移植（那个工具即规格与等价性 oracle）；白名单与它相同（5 个官方
  v2 输入），名单之外仍旧报错并指向离线工具。
* **纯加载期**：v3 的 `parse_directory`/binder/materializer/所有算子逐字未动，**不影响性能**。
* 详见 [v2-direct-read-2026-10-16.md](v2-direct-read-2026-10-16.md)。

### S0.19 制品家族：原生配方 vs GGUF 转制（性能差异的真正来源）☑ 只调查、不改代码（2026-10-16）

用同一份清单解析器把所有在用 `.ninfer` 过了一遍，并做了判决实验（拿存档参考线的二进制去读
GGUF 转制制品）。

* **两族制品**：
  * **原生 NInfer 配方**（`weights_id: nvfp4`）：`nvfp4`（块缩放 k16×m128×4）、
    `fp8_e4m3fn_row_bf16`（每权重一个 FP8 码 + 每输出行一个 BF16 scale）、`q4_g64_fp16` /
    `q5_g64_fp16` / `q6_g64_fp16` / `q8_g32_fp16`（分组 + FP16 scale）、`bf16` / `fp32` / `int32`。
    来源是 HF safetensors / compressed-tensors / block-FP8，由 NInfer 自己的 converter 量化。
  * **GGUF 转制**（`gguf_blocks_v1` 布局 + 19 种 `gguf_*` 格式）：Gemma4-26B-A4B（`gguf_q4_k` /
    `q8_0` / `q5_0` / `q6_k` / `q4_0`）与 GSQ IQ3_S（`gguf_iq3_s` / `iq4_xs` / `iq3_xxs` / `q4_k` /
    `iq2_s`…，共 17 种）。它们走 ggml bridge（逐张量块反量化 + 对应 MMQ/Vec 内核）。
* **容器版本不影响性能**：同一份权重的 v2 原件测 998 tok/s、升级后的 v3 测 997 tok/s（同为 27K
  请求的 prefill），本分支两者都原地直读。
* **性能差来自配方**：原生 nvfp4/fp8 走 QPN/CUTLASS 快路（prefill 1.00k / decode 76–82）；GGUF
  混合（18 种格式，IQ3_S 仅 3.39 GiB）在现场反量化，打碎成约两万个小内核 → 885 / 66–74（比原生
  低约 12%）。
* **存档参考线只支持原生族**：`archive/fork-master`（`0d24f83e`）的 `src/ops/linear/` 没有 gguf
  算子、格式注册表里 0 个 `gguf_*`；实测让它读 GSQ 制品直接
  `FATAL server failed during startup | unknown tensor format: gguf_iq4_xs`（退出码 1）。本分支
  两族、两个容器版本都支持。
* 结论已写进默认分支 README 的 "Two artifact families" 一节。

### S0.20 Gemma 4 宽行内核：dense 31B（hidden 5376）跑通 ☑（2026-10-16）

用户提供了 `Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced` 的 GGUF + MTP + mmproj 三件套，按
`--recipe gemma4_gguf --components text,mtp,vision` 转成 v3 制品（18.74 GB）后引擎拒绝加载：
`gemma4 row kernels: hidden must be whole float4s, at most 4096` —— 三个行内核每线程一个 float4，
一行要装进一个 block，dense 31B 的 5376（1344 个 float4）超过 1024 线程上限 ✗。

* **改动**：新增三个宽行内核（512 线程 + 按 `blockDim.x` 步进 ✓，只在 `hidden/4 > 1024` 分派），
  既有内核与窄行路径**逐字未改** ✓ → 对其他制品零影响 ✓。
* **踩到的真 bug**：宽行版本要靠"重算中间值"跨趟传递，第一版把 `a.h` 的写回放在提前的一趟 ✗，
  后续重算读到已写回的 h → **残差加两次** → 复读乱码 ✗。差分测试（把 26B 强制走宽行）复现 ✗，
  按位探针（`NINFER_PROBE_WIDE`）缩到 `attn_residual` + `ffn_combine` ✓；修法是把写回推到最后一趟 ✓。
* **制品已验证**：F32 norms + Q4_K/Q6_K 矩阵（含融合 gate_up 的两半）+ 注意力 q/k/v/o **9/9 逐字节等于
  GGUF** ✓（乱码的责任在引擎侧，不在转换 ✓）。
* **验收**：新增 oracle `ninfer_gemma4_wide_rows_test`（2816 窄行 + 5376 宽行 vs FP64 ✓）；`ctest -R
  gemma4` **8/8** ✓；26B 在探针强制宽行下输出与窄行逐字一致 ✓；31B 端到端 **decode 29.3 tok/s
  （无 MTP）/ 41.8 tok/s（assistant，4 drafts，接受率 24.1%）** ✓，回答正确 ✓。
* 详见 [gemma4-wide-rows-2026-10-16.md](gemma4-wide-rows-2026-10-16.md)。

### S0.21 v1.5 → v1.6 差距的归因与收敛 ☑(2026-10-20)

外部报告([#2](https://github.com/JimmyMax/ninfer-v100/issues/2)):同制品同参数下 1.5 线 MTP 接受率
~75%、1.6 线 60–68%。拆成三条独立原因:度量口径(已修)、验证端 KV 切分(上游 `ddacb96`)、
sm_70 的 FP16 范围饱和(上游 `67778be`+`2d69547`)。

* [x] 拷贝草稿改为显式启用(默认 0),`speculative.*` 只统计神经行(`f0ee4fe`、`edcb69d`)
* [x] 工作树重建到 `origin/main`(真实历史、与上游 `fcc4928a` 有 merge-base);核实 `main` 落后上游 148 个提交,其中 39 个触及 V100 关键路径
* [x] 试合并 `upstream/master`:冲突 12 个文件 / 459 自动合并 → 策略定为**一次 merge**而非串行 cherry-pick
* [x] 解决 12 个冲突完成 merge(5 处 sm_70 钩子 vs 上游 unified 路由):`6988552`
* [x] decode 侧收敛:`1fcfa17`、`e563078`、`a639388`(FP8 预打包/暂存、Q8/Q4 窄宽度)
* [x] 长上下文 prefill:`89be923`(flash 快路线放宽到 131K 可见键)、`b735b91`(宽 T 反量化)、
      `80e9ae4`(Q 块 1024)
* [x] 终验:两个制品上 MTP decode 全深度领先(+2.4% ~ +11.1%),纯解码持平,prefill 27K 持平 /
      66K −1.0%(归属:精确 NVFP4 解码 645 ms + flash 每次调用 21.2 对 21.6 ms 但多 80 次)
* 详见 [v1.6-vs-v1.5-2026-10-20.md](v1.6-vs-v1.5-2026-10-20.md)。

### S1 Gemma 4 的 V100 route/device profile 调优 ☐

纯配置与标定，不动 kernel。

* `--draft-tokens`（1/2/3）、`--prefill-chunk`（512/1024/2048）、`--kv-dtype` 全 9 种
  （Gemma 4 支持 int8/fp8/nvfp4/rk4v4/rk2v4-e8/…，其滑窗 ring 与全局 V 平面各一张表）、
  `--vision-residency`（resident/overlay/cpu）、`--host-kv-mib`、CUDA graph allowance。
* 参考上游自己的 16GB 卡数据（`docs/performance/tesla-v100-2026-10.md`：Gemma 4 Q4_0 短对话
  164 tok/s 无投机 / 226 tok/s MTP-2），本机 32GB 应更高；当前 157.5 tok/s 有提升空间。
* 产出一张 `--kv-dtype × max-context × drafts` 的 V100-32G 参数表。
* 预估：2–5 天。

### S2 Gemma 4 的 Volta 解码注意力（attention v2 技术落地） ☐（滑窗几何已落地 2026-10-23）

**Gemma 4 不使用共享的 `causal_cache` 家族**，它有自己的
`src/models/gemma4/{kernels.cu,kv_kernels.cu,kv_codec.cuh}`（25 层滑窗 window=1024 / head 256 /
16q·8kv，5 层全局 head 512 / 16q·2kv，且全局层只缓存 V、K 由 V 派生）。

* 要移植的是**技术**而非文件：key-split 8-warp `Bc=64` 解码、`byte_perm` int8→fp16 反量化、
  fp16 PV 累加链、prefill 阈值重调；两个几何分别实现。
* 验收：Gemma 4 decode 提升百分比 + 数值等价（上游自带 FP64 oracle 测试：
  `tests/models/gemma4/*`，`ninfer_gemma4_kv_test` 等）。
* 预估：1–2 周。

#### S2 进展（滑窗几何，2026-10-23）

重写 `decode_attention_kernel`（D256 滑窗）：128 线程/4 warp 改为 256 线程/8 warp，
warp w 对 head w/4 取 tile 内 8 个 key，K **直接从全局读进寄存器**（不再过 shared
memory），V 保留 smem 暂存 → 每块 smem 从 33KB 降到 16KB；`key_reduce8` 蝴蝶归约
（3 轮二分交换 + 2 轮自由位混合，主机模拟验证）；fetch 移到打分之后，下一 tile 的
加载与 softmax/PV 重叠。`__launch_bounds__(256, 2)`，80 寄存器零 spill。

* oracle：`ninfer_gemma4_kernels_test` 全绿（D256 全形状，rms ≈ 1.65e-3，与基线相同）；
  `ninfer_gemma4_{kv,output,wide_rows}_test` 全绿。
* ncu 固定时钟 A/B（同 87 个注意力 launch 窗口）：滑窗 kernel 577→507 µs（**−12.1%**，
  平均 15.6→13.7 µs/launch），wide 与 combine 不变，注意力合计 1014→948 µs（−6.5%）。
* E2E V100-32G（Gemma4-26B-A4B MTP 制品，prompt 1813 tokens）：无投机 126.7→129.4 tok/s
  （**+1.6%**，交替 A/B ×3 可复现；轨迹分叉使 v3 早 1 token 到 EOS）；MTP-2
  159.2→170.0 tok/s，per-token **+6.7%**（每轮 9.98→9.45 ms；接受率相当：58% vs 59%）。
* 试过并否决：512 线程/16 warp 版（64 寄存器上限下热循环 spill 到 local memory，
  local 流量 ~28MB/launch，47 µs 反而更慢）；splits 32/64（无收益）；`launch_bounds(256,3)`
  （无差异）；wide kernel `kTile` 16→8（E2E −7%，barrier 密度加倍吞掉 smem 收益）。
* 剩余：wide（D512）几何的同款重构、`byte_perm` int8→fp16 反量化路径、fp16 PV 累加链
  （目前 PV 为 fp32 累加）。

### S3 Volta GGUF 投影加速 ☐（decode 投影已落地 2026-10-24）

Gemma 4 的权重是逐张量 GGUF block 格式（Q4_K / Q5_0 / Q6_K / Q8_0 / Q4_0（草稿头）/ BF16（视觉塔）/ FP32）。
上游在 Volta 上的现状：≤8 列走 NInfer 的 DP4A 向量核，>8 列走 ggml 的 DP4A 矩阵核，
≥65 列反量化到 FP16 走 cuBLAS。V100 没有整数张量核但有 DP4A，这里有明确空间。

* 可借鉴 fork 的手法：权重的 Volta 布局 prepack、双缓冲分组预取
  （`docs/V100-PERF-PLAN.md` 的 `2b3f16cd` / `9913bc22` 条目）、epilogue 融合、
  激活分块暂存。
* 验收：`pp2048` 与 decode 段耗时对照表（`ninfer_bench`）。
* 预估：1–2 周。

#### S3 进展（vec 核 constexpr K，2026-10-24）

31B decode 的 kernel 分解（nvprof，1.8K prompt，100 token，MTP-4）：GGUF 反量化 GEMM 族
占 decode kernel 时间 ~85%，其中 vec matvec（Q4_K 73% / Q6_K 22% / Q4_0 5%）为主。
微基准（纯 DRAM，N=20480 单 buffer > L2，K=5376，T=5）基线 Q4_K 125.6 µs/product
（493 GB/s，V100-32GB 实测 DRAM 上限 875 GB/s 的 56%）。

根因：kernel 是延迟受限而非带宽受限（ncu：long_scoreboard 54%，L1 35% / L2 23% /
DRAM 50%，均不饱和）；编译器不做预取。constexpr K 让 (j, it) 激活基址在编译期预计算
（去掉每 it 的 IMAD 地址链），单独带来 14%。另外原静态 T>1 路径的 `#pragma unroll 1`
kFull/tail 分裂循环（132 µs）比简单循环慢；T=1 全展开（96.3 µs）又略快于运行时形式
（97.1 µs）。

落地改动（`src/ops/linear/gguf/ggml_bridge_vec.cuh`）：

* 静态 K 循环统一为单一 slice 循环；`T==1 && KS!=0` 用全展开形式（含尾界检查）。
* 静态列表加入 K=5376（31B hidden，尾 256 元素）；Q4_0 加入 `kVecStaticK`
  （微基准 139.5→114.1 µs/product）。
* Q6_K 明确排除出 `kVecStaticK`：constexpr K 使其变慢（265 vs 224 µs/product），
  其散 2B 读取是主导，留待 block 解码重构。
* 移除死代码：失败的 smem 暂存核（staged kernel，1.8× 慢）与 j-chunking 实验。

验收（V100-32GB，2026-10-24）：

| 项 | 基线 | 落地后 | 变化 |
|---|---|---|---|
| Q4_K vec 微基准（K=5376 T=5，纯 DRAM） | 125.6 µs / 493 GB/s | 109.5 µs / 565 GB/s | +13% |
| Q4_K vec 微基准（K=5376 T=1） | 97.1 µs | 96.3 µs | +1% |
| 31B bench `tg128`（MTP-4，int8 KV，chunk 2048） | 96.5 tok/s | 98.44 tok/s | +2.0% |
| 31B bench `pp2048` | 759.0 tok/s | 759.2 tok/s | 不变 |
| 31B bench `tg128` no-spec | — | 29.09 tok/s | 无回归 |
| oracle（`ninfer_linear_gguf_test`） | — | 全绿 | — |
| Gemma 4 kernels / KV tests | — | 全绿 | — |

### S3 进展（26B MoE 覆盖，2026-10-25）

26B decode 的 kernel 分解（nvprof，2048 prompt，200 token，MTP-2）：GGUF 反量化 GEMM 族
占 ~74%，其中 `vec::moe_kernel`（路由专家）37% 与 `vec::kernel`（稠密 + 共享专家）37%。
26B 的 K 集合 {2816, 4096, 2112, 704} 此前全在运行时路径（5376 静态列表只覆盖 31B/27B）。

改动（同文件 + `src/models/gemma4/executor.cpp`）：

* 静态 dispatch 加入 K=704/2112/2816/4096；Q5_0、Q8_0 加入 `kVecStaticK`
  （微基准：Q8_0 K=2816 T=3 +5.7%，Q4_K K=2816 T=4 +7.5%）。
* `moe_kernel` 加同一 constexpr K 机制（K=704/2816，Q4_K/Q5_0/Q8_0）。
* MoE chunk 规则：原按 t 取 1/2/4/8；改为按 pairs/active-experts 期望值取最小 2 的幂。
  26B decode（24 pairs / 128 专家）从 T=4 降为 T=1——原 T=4 对每专家 ~1.9 对的一半列
  是纯填充浪费。单项贡献 +4.7%（隔离测量：constexpr K +1.6%，chunk +4.7%）。
* T=1 的静态 K 只对 K=5376 生效：全展开形式在 26B 形状实测变慢（Q4_K K=4096 +5.6%、
  Q5_0 K=2816 +8.7%），其余 T=1 保持运行时形式。

验收（V100-32GB，2026-10-25，int8 KV，chunk 2048，`tg128`）：

| 配置 | 基线 | 落地后 | 变化 |
|---|---|---|---|
| 26B no-spec | 128.17 tok/s | 127.79 tok/s | −0.3%（MoE T=1 静态的代价） |
| 26B MTP-1 | 190.67 tok/s | 192.78 tok/s | +1.1% |
| 26B MTP-2（主配置） | 216.24 tok/s | 229.91 tok/s | **+6.3%** |
| 26B `pp2048` | 2502 tok/s | 2500 tok/s | 不变 |
| 31B `tg128`（MTP-4） | 98.44 tok/s | 98.31 tok/s | 不变（31B 无 {704,2112,2816,4096} 形状，结构不受影响） |
| 31B `tg128` no-spec | 29.09 tok/s | 29.06 tok/s | 不变 |
| oracle（linear GGUF + MoE GeGLU） | — | 全绿 | — |

serve 二进制已重链；`ninfer-gemma4-31b` 容器结构上等价（31B 路径不变）。

剩余：Q6_K 的散 2B 块解码（占 31B vec 时间 22%）尚无可行加速——smem 暂存、j-chunk、
权重/激活预取全部实测为负（寄存器墙：per-it 权重 512 B = 128 寄存器，96 寄存器基线
已在 5 block/SM 边界）。Q4_K 距 DRAM 上限 65%，剩余差距为 DRAM 延迟；进一步需要
布局级 prepack（31B 在本卡 VRAM 预算下不可行）或 block 解码重构。

### S4 MoE 专家投影的 Volta 路径 ☐（部分：constexpr K 与 chunk 规则已并入 S3）

Gemma 4 是 128 专家 × 8/token、专家宽度 704，A4B 的算力几乎全在专家上。S3 已把路由
专家核（`vec::moe_kernel`）纳入 constexpr K 并把 chunk 从按 t 改为按 pairs/专家取最小
2 的幂（26B decode T=4→T=1，单项 +4.7%）。剩余的是专家权重本身的块解码结构与布局。

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
