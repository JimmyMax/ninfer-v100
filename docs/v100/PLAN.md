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

> S0.5 的结论改写了后续优先级：Qwen 的 4.3× 差距来自 **NVFP4/FP8 投影的 Volta 原生路径**
> （上游在 V100 上没有 NVFP4 W4A4，走 A16 反量化），不是注意力；Gemma 4 的杠杆则在
> **MoE 专家 + GGUF 投影**（S3/S4）。S2 的注意力工作降级为"顺带"。

以**同源代码**的注意力内核作为样板，证明 fork 的 Volta 加速能落到上游线上。

* 目标文件（两侧同源，可 patch 而非重写）：
  `src/ops/softmax_attention/dense/causal_cache/{prompt_i8.cuh,small_t_i8.cuh,prompt_volta.cuh,
  small_t_*_volta.cuh,causal_softmax_attention.cpp,launch.h}`
* 来源提交（fork）：`9e2a9bff`（Volta prefill retune + INT8 small-T decode attention v2）、
  `db947f95`（small-T PV 累加链改 fp16）、`6856d323`、`c380*` 系列（`perf(sm70)` 端口）
* 验收：上游线上 Qwen3.8-27B nvfp4 的 decode 从 ~15 tok/s 提升到接近 fork 的量级；
  与 fork 的输出在贪心解码下逐 token 一致；用 `bench/v100/compare_lines.py` 记录前后对照。
* 预估：3–7 天。**这一步是整条路线的前提，验证通过才继续投 S2/S3。**

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
