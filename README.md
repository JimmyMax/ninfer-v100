# NInfer-all

A C++/CUDA inference engine for the Qwen3.5 model family (Qwen3.6 and Qwen3.8 27B, Qwen3.6-35B-A3B,
their fine-tunes and Ternary Bonsai 2), Qwen3.8-Flash-Next and Gemma 4 26B-A4B, on the RTX 3090,
RTX 4090, RTX 5090 and RTX PRO 6000 Blackwell, and on the Tesla V100 from a
[source build](docs/build-linux.md#tesla-v100-sm_70). It serves the OpenAI, Anthropic and llama.cpp
APIs from one GPU, or from up to eight as pipeline stages on Linux.

NInfer-all continues [NInfer](https://github.com/Neroued/ninfer) through the NInfer-3090 line and
carries the work of several forks; [CONTRIBUTORS.md](CONTRIBUTORS.md) credits each of them.
The [documentation map](docs/README.md) lists the guides and references.

## Tesla V100-32GB 单卡（sm_70）优化线

本仓库的主线是**在单张 Tesla V100-PCIE-32GB（sm_70）上把 NInfer 跑快、跑全**：改动集中在算子
路由与预打包（加载期或首次调用时完成），不改变模型的数值契约；v3 制品路径逐字未动，v2 制品
在原地直读。下面所有数字都是同一张卡上的 27K 上下文请求（int8 KV、MTP draft 4、贪心）。

*This branch targets single-GPU Tesla V100 32GB (sm_70). The generic documentation for the
other cards follows below.*

### 支持的制品

| 制品 | 权重 | 上下文 | 27K 请求实测 |
|---|---:|---:|---|
| `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3.ninfer`（nvfp4，生产默认） | 19.1 GiB | 254,208 | TTFT **27.2s**、prefill **1.00k tok/s**、decode **76–78 tok/s** |
| `Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2.ninfer`（**v2 原件，直读**） | 22.1 GiB | 200,000 | TTFT **27.3s**、prefill **998 tok/s**、decode **82.7 tok/s** |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（IQ3_S，18 种格式混合） | 11.9 GiB | 400,000 | TTFT **30.8s**、prefill **885 tok/s**、decode **66–74 tok/s** |

短上下文（1K prompt）decode **82.3 tok/s**；27K decode 的 MTP 接受率 69%。

**v2 与 v3 都原生支持**，v2 不需要任何离线转换。格式覆盖 v3 的 `nvfp4`、
`fp8_e4m3fn_row_bf16`、`q4_g64_fp16` / `q5_g64_fp16` / `q6_g64_fp16` / `q8_g32_fp16`、`bf16`、
`fp32`，以及 v2 官方词汇的全部组合（`NVFP4`、`Q4G64_F16S`…，包括混合 MLP 的 nvfp4full 风格
制品）；同目录下的 `Huihui-…-nvfp4`、`orcarouter-…-nvfp4`、`Gemma4-26B-A4B-…` 等 v3 制品同样
可直接加载。

### 主要原理改动

| 改动 | 收益（实测） |
|---|---|
| **Volta QPN**：nvfp4 / fp8 的 quadpair-split-N 预打包 GEMM（`t≤32` 走 QPN，宽 T 走 CUTLASS） | 投影路线 85 → 480–620 GB/s |
| **GDN prefill 用 recurrent 单内核**（sm_70 上 chunked / two-stage 是负优化） | GDN T=1024…8192 **2.9–3.3×**，27K prefill 中 3.75 → 2.18s |
| **Flash attention prefill 换核**：paged→连续 staging + sm_70 `mma.sync` 内核 | 同形状 41.1 → 25.1 ms，27K TTFT 30.1 → 27.2s |
| **Decode 投机配置**：`--ngram-draft-tokens 0`、去掉 `--adaptive-mtp`（默认 15 个 n-gram 草稿会稀释 MTP 接受率） | 27K decode **64.2 → 78.3 tok/s**，贪心输出逐字不变 |
| **FP8 SwiGLU 宽 T 走 FP32 出口的 CUTLASS**（原先落在每 4 个 token 一块的 SIMT 阶梯） | 混合量化制品 prefill **508 → 998 tok/s** |
| **v2 制品直读**：清单在内存里归一化成 v3 形态，权重原地读 | v2 原件与升级后逐字一致，省掉离线转换与 24 GB 副本 |

### 快速开始

```bash
# 单卡（host GPU 1 = V100-32GB），生产 compose 示例在 Docker 仓库的 ai/ninfer-v100/
docker run -d --gpus device=1 -v /path/to/models:/workspace/models \
  --entrypoint ninfer-serve ninfer:v100 \
  /workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3.ninfer \
  --host 0.0.0.0 --port 8000 --api-key sk-xxx --model-id qwen --max-context 200000 \
  --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 --max-concurrency 2 \
  --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0
```

构建：`Dockerfile.sm70.local`（CUDA 12.8 + `-DCMAKE_CUDA_ARCHITECTURES=70` 全量）或
`Dockerfile.sm70-runtime.local`（复用已编好的 `ninfer-serve` 秒级重打包）。

### 验收与文档

每个改动都要同时过上游自带的 FP64 oracle（`ctest`）与 27K 端到端 A/B；两线对照用
`bench/v100/compare_lines.py`（预热后连测，输出 markdown 表），健康检查用 `GET /health`
（加载中 503、就绪 200）。设计与实测记录全部在 [`docs/v100/`](docs/v100/)：
路线图 [PLAN.md](docs/v100/PLAN.md) 加每阶段一篇实测文档。

## Highlights

### Ternary Bonsai 2 27B and Qwen3.8-27B

Measured in September 2026, one card each, greedy, one request unless the row says otherwise.
Setups and full tables: [reference measurements](docs/performance/reference-2026-09.md).

| | RTX 3090 | RTX 4090 | RTX 5090 | RTX PRO 6000 |
|---|---:|---:|---:|---:|
| **Ternary Bonsai 2 27B**, short chat (DFlash2, 7 drafts) | 202 tok/s | 256 tok/s | 397 tok/s | 381 tok/s |
| decode after a 261K-token document (fastest drafter) | 90 tok/s | 123 tok/s | 218 tok/s | 218 tok/s |
| time to first token for a 261K-token prompt | 215 s | 102 s | 82 s | 78 s |
| largest context, filled and all three needles found | 970,752 | 958,464 | 978,944 | 1,048,576\* |
| eight requests at once (MTP, 3 drafts), total | 551 tok/s | 824 tok/s | 1,063 tok/s | 1,155 tok/s |
| **Qwen3.8-27B**, short chat (DFlash2, 7 drafts) | 118 tok/s | 149 tok/s | 236 tok/s | 237 tok/s |
| largest context, filled and all three needles found | 417,792 | 405,504 | 872,448 | 1,048,576\* |
| eight requests at once (MTP, 3 drafts), total | 329 tok/s | 442 tok/s | 690 tok/s | 739 tok/s |

\* The engine's ceiling. The RTX PRO 6000 (96 GB) starts there with every KV format and drafter;
filled to it, both models find two of the three needles.

- **RTX PRO 6000 against RTX 5090.** Re-measured in October beside an RTX 5090, both at 600 W, the
  PRO 6000 came back within 2% of September wherever the drafts accepted the same share. One
  request's decode is bound by memory bandwidth, and both cards have 1.79 TB/s of GDDR7 (Bonsai 2
  without speculation: 167.7 against 170.4 tok/s). The PRO 6000's 188 SMs against 170 show where
  compute decides: the 261K prompt is 2% faster and eight requests at once 6 to 7% faster
  ([re-check](docs/performance/reference-2026-09.md#re-check-october-2026)).
- **Draft length.** DFlash2 with seven drafts is fastest on short answers; after long documents the
  best count is three to seven. MTP runs up to fifteen drafts and is fastest at three to five.
- **Past the native window.** Filled to about 880K tokens, Bonsai 2 returned all three planted codes
  on every card. At 1,048,576 tokens (RTX 5090 and PRO 6000 only) it misses the one at 943K.

### Qwen3.8-Flash-Next

One request, greedy, October 2026. Each cell is **decode of a short answer · prefill of a
4,463-token prompt**, in tok/s. Decode after the long prompt, memory, host links and power limits:
[Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md#measurements).

**Q2_0** (35.9 GiB model + 26.8 GiB n-gram table):

| Experts | RTX PRO 6000 | RTX 5090 | RTX 4090 | RTX 3090 |
|---|---:|---:|---:|---:|
| on the GPU (two cards, except the PRO 6000) | 138 · 2,939 | 136 · 3,809 | 97 · 3,608 | 90 · 1,504 † |
| in pinned host memory | 55 · 1,297 | 74 · 1,677 | 53 · 1,138 | 49 · 842 |
| on disk, the files in the page cache | 76 · 2,024 | 68 · 1,739 | 42 · 723 | 47 · 630 |
| on disk, cold (pages evicted every second) | 35 · 1,000 | 25 · 694 | 18 · 160 | 17 · 195 |

† Two RTX 3090 Ti.

The host and disk rows were measured before the October 9 decode work. Host experts on an RTX 3090
now decode 87-94 tok/s in a different workload, 98-100 with a repeated prompt
([host experts on one RTX 3090](docs/qwen3-8-flash-next.md#host-experts-on-one-rtx-3090)).

**IQ3_S** (51.9 GiB model + the same table):

| Experts | RTX PRO 6000 | RTX 5090 | RTX 3090 |
|---|---:|---:|---:|
| on the GPU (two cards for the RTX 5090) | 125 · 2,541 | 122 · 3,326 | — |
| in pinned host memory | 29 · 803 | 41 · 1,100 | 34 · 533 |
| on disk, the files in the page cache | 66 · 1,661 | 55 · 439 | 19 · 209 ‡ |
| on disk, cold | 29 · 773 | 19 · 171 | 11 · 47 |

— Not measured: every expert needs three 24 GB cards. ‡ The host's 62 GB of RAM cached only part
of the file.

**Coder IQ1_M** (28.4 GiB model + the same table), one NVIDIA L40S (48 GB):

| Experts | NVIDIA L40S |
|---|---:|
| in pinned host memory | 35 · 1,648 |
| on disk, the files in the page cache | 42 · 1,938 |

- **The host matters.** Host and disk rows depend on the host as much as on the card. The RTX 5090
  host had PCIe 5.0 x16; the others had PCIe 4.0 x16.
- **The PRO 6000 caches almost everything.** Its 96 GB device expert cache ends up holding nearly
  every expert.
- **RTX 4090 host pinning.** The RTX 4090 rows ran pinned to the GPUs' NUMA node in a two-socket VM.
  Unpinned, host experts decode there at 40 tok/s and disk experts at 32 to 33.
- **Many requests at once.** Six reasoning requests at once on two RTX 5090s produced 193 tok/s of
  output with Q2_0.

## Features

**Models and weight formats**

- **Qwen3.5 family.** Dense and MoE (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) with MTP,
  DFlash (35B-A3B) and DFlash2 (Qwen3.8-27B) drafters, Vision for images and video, and
  `groupwise-int`, FP8, NVFP4 and ternary weights. NVFP4 expert banks prefill through W4A4 and need
  a `120a` build.
- **GGUF block formats.** Releases that pick a ggml type per tensor, such as ISTA-DASLab's GSQ-RCO,
  convert without requantization and multiply their blocks in place. The 3.5-bit Qwen3.8-27B IQ3_S
  release holds 10.95 GiB of weights instead of 15.9, scores WikiText-2 perplexity 7.071 (the
  official artifact: 7.286) and decodes at 59.9 tok/s against the official artifact's 40.3 on an
  RTX 3090 ([GGUF block formats](docs/gguf.md)).
- **Ternary Bonsai 2 27B.** PrismML's [ternary Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
  runs from `t2_g128_fp16` weights: 2.125 bits per weight, imported without rounding, with the
  Hadamard rotations fused into the norms and gates and integer activations for its projections.
  The `bonsai2_27b_ternary` recipe adds ProCreations' MTP head and DFlash2 adapter and an exact
  proposal head.
- **Qwen3.8-Flash-Next.** The 125B-parameter MoE (512 experts, about 6B active) from the GSQ-RCO
  GGUF releases, with its experts on the GPUs, in pinned host memory with a GPU cache
  (`--expert-residency host`) or in the artifact's files read into a GPU cache
  (`--expert-residency disk`, under 1 GB of RAM). It serves up to eight requests with prefix reuse,
  structured output, images and video, and MTP drafting. On two RTX 5090s the Q2_0 release scores
  93.3% on AIME 2025 and 86.4% on GPQA-Diamond ([Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md)).

- **Gemma 4 26B-A4B.** Google's QAT Q4_0 GGUF release, imported block for block (13.4 GiB), or a
  mixed-format GGUF of it: the sliding-window and full-attention blocks (the latter caching values
  only), the dense MLP beside 128 routed experts, every `--kv-dtype` format, up to eight concurrent
  requests with prefix reuse through host memory and shared system prompts, tool calls and
  structured output, images and video through the release's vision tower (on the
  GPU, streamed from pinned memory or on CPU threads), and Google's multi-token-prediction assistant
  for speculative decoding. On an RTX 3090 it prefills a 10,000-token prompt at 6,045 tok/s and decodes at 215
  tok/s, 190 after 10,000 tokens, against llama.cpp's 4,697, 169 and 152 on the same GGUF; with two
  drafts a round chat answers decode at 291 tok/s instead of 209 ([Gemma 4](docs/gemma4.md)).

**Speculative decoding**

- MTP, DFlash and DFlash2 with 1 to 15 drafts; the artifact's proposal head drafts by default when it
  stores one. Under MTP and DFlash a sampled request keeps a draft only while it equals the model's
  own keyed draw, so every emitted token is the model's sample; DFlash2 verifies its proposal
  distribution by rejection sampling.
- Gemma 4 drafts with Google's multi-token-prediction assistant (`--spec mtp`, 1 to 7 drafts), four
  blocks that read the model's own KV cache; its verification rows are bit for bit plain decode rows,
  so the output is the one plain decoding gives.
- `--adaptive-mtp` picks each round's width from measured draft survival and round cost.
- N-gram copy drafting beside the neural drafter (on with `--spec`) verifies up to 15 tokens copied
  from earlier prompt, tool-result or output text ([ngram copy proposals](docs/ngram.md)).

**Context and memory**

- Nine KV formats, from `bf16` to `rk2v4-e8` at 216 bytes per token and KV head; `rk8v4` is the
  all-round choice and `rk4v4` holds the native 262,144-token window of Qwen3.8-27B on a 24 GB card
  ([context and memory](docs/cli.md#context-and-memory)).
- Up to 1,048,576 tokens of context, four times the native window, with plain RoPE or YaRN
  (`--rope-yarn`).
- A context cache of checkpoints on the device, in pinned host memory and in an optional disk tier
  that survives restarts (`--disk-kv-path`, `--disk-kv-restore`), with endpoint and branch anchors
  that let an edited or regenerated turn resume where it left the conversation. The hybrid prefix
  cache (`--use-alt-prefix-caching`) shares content-addressed KV blocks across requests instead.
- Precision trades that buy context: `--embedding-q4`, `--lm-head-q6`, `--gdn-state-fp16`
  ([quality trades](docs/maintainer/quality-trade-experiments.md)).

**Serving**

- OpenAI Responses and Chat Completions, Anthropic Messages, llama.cpp's `/completion`,
  `/tokenize`, `/detokenize`, `/apply-template`, `/v1/rerank`, `/metrics`, `/slots` and `/props`,
  streaming, tool calls, structured output through xgrammar (`--structured-output`), token log
  probabilities and prompt grafts ([HTTP serving](docs/serving.md)).
- **Model suspend.** `--model-suspend` lets an idle server give its device memory back without
  exiting and take it again on the next request, retained conversations included. Ternary Bonsai 2
  27B on an RTX 3090: 7.9 GiB down to 0.3 GiB in 0.33 s, back in 1.1 s
  ([model suspend](docs/serving.md#model-suspend)).
- **Several models behind one server.** With `--models-dir` or a llama.cpp-style
  `--models-preset`, `ninfer-serve` is a router with llama.cpp's model API; with `--model-suspend` a
  model sleeps instead of unloading: two 27B models on one 24 GB RTX 3090 swap in 1.8 s instead of
  an 11.6 s cold load ([several models](docs/serving.md#several-models-router)).
- `GET /stats` and a [monitoring dashboard](tools/monitor/README.md), request logs with rotation,
  and an optional compiled-in WebUI.

**Kernels and devices**

- **Device route profiles.** Each card's measured profile picks the kernel route for every
  operation and width. Profiles are built in for the RTX 3090, 4090, 5090, the three RTX PRO
  6000 editions and the Tesla V100-SXM2-16GB; any other GPU is measured once at first start, and
  `ninfer-calibrate` re-measures ([device profiles](docs/device-profiles.md)).
- **Pipeline stages.** `--devices 0,1,...` splits the layers over up to eight GPUs on Linux, each
  stage owning its weights, KV and state ([pipeline stages](docs/maintainer/pipeline-stages.md)).
- FP8 A8 and NVFP4 W4A4 tensor-core routes on every `120a` build, an FP4 tensor-core prompt kernel
  for an NVFP4 KV cache, and programmatic dependent launches on Blackwell.

## Running

Download an artifact from the [table below](#artifacts) and serve it from the [Docker image](#docker)
or from a [build](#building). The server speaks the OpenAI and Anthropic APIs, and each card picks
up its device profile on its own.

```bash
hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local-dir models

# The image: `serve`, the artifact under /models, the flags. Listens on http://localhost:8080/v1.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 \
  --gdn-state-fp16 --spec dflash2 --draft-tokens 5

# A build: the same flags. Listens on 127.0.0.1:8080 unless --host and --port say otherwise.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

The recipes below are the configurations the [reference tables](docs/performance/reference-2026-09.md)
use. They are written for a build; in the container, replace `ninfer-serve models/` with the
`docker run ... serve /models/` line above.

<details>
<summary><b>Ternary Bonsai 2 27B</b>: DFlash2, MTP, the largest context, a disk tier</summary>

```bash
# Fastest single stream: DFlash2 with five drafts over the full 262,144-token window.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# MTP drafting through the proposal head.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3

# The largest context a 24 GB card holds: 958,464 tokens of rk4v4.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 958464 --kv-capacity 958464 --kv-dtype rk4v4 --gdn-state-fp16 --rope-yarn

# Adaptive MTP and a disk tier that keeps evicted conversations.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 5 --adaptive-mtp \
  --disk-kv-path /var/cache/ninfer --disk-kv-gib 64 --disk-kv-restore
```

- **Draft count.** Five drafts are the all-round choice. Seven are faster on short answers; after
  long documents three to seven win ([draft length](docs/performance/reference-2026-09.md#draft-length)).
- **Images.** `--vision --vision-residency overlay --vision-max-merged 12288` adds them. The encode
  borrows the drafter's memory, so the whole window still fits a 24 GB card.
- **Past 958,464 tokens.** The RTX 5090 and the PRO 6000 hold the 1,048,576-token maximum with
  `rk4v4`, DFlash2 or MTP included; the RTX 5090 holds 978,944 with `rk8v4`. Filled to 1,048,576
  tokens, the model misses the code at 90% (about 943K); up to about 880K it found every code on
  every card.

</details>

<details>
<summary><b>Qwen3.8-27B</b>: the official artifact and the GSQ-RCO IQ3_S release</summary>

```bash
# The official artifact on a 24 GB card: DFlash2 with five drafts over 245,760 tokens of rk4v4.
ninfer-serve models/qwen3_8_27b.ninfer --model-id qwen3.8-27b \
  --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# GSQ-RCO IQ3_S: MTP over 176,128 tokens of rk8v4.
ninfer-serve models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 176128 --kv-capacity 176128 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3
```

- **The official artifact with `rk8v4`.** The same speculation fits 167,936 tokens on an RTX 4090
  and 176,128 on an RTX 3090. An RTX 5090 takes the full 262,144 with either KV format.
- **IQ3_S.** It drafts with `--spec dflash2 --draft-tokens 5` as well, and `--vision` adds images.

</details>

<details>
<summary><b>Qwen3.8-Flash-Next</b>: experts on the GPUs, in host memory or on disk</summary>

```bash
hf download WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3 \
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer --local-dir models
hf download WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3 \
  Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer --local-dir models
M=models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer
T=models/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer

# Every expert on one RTX PRO 6000 (96 GB).
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768

# Every expert on two GPUs, one pipeline stage each (Linux): 24 GB cards for Q2_0, 32 GB for IQ3_S.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --devices 0,1

# One 24 GB GPU: the experts in pinned host memory, the most used of them cached on the GPU.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --expert-residency host

# One 24 GB GPU and little RAM: the experts stay in the file and stream into a GPU cache.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --expert-residency disk

# The container, experts in host memory.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all serve /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer \
  --ngram-table /models/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer \
  --model-id qwen3.8-flash-next --max-context 32768 --expert-residency host
```

- **Other releases.** IQ3_S and the Coder IQ1_M build take the same flags with their own file and
  the same table. The [tables above](#qwen38-flash-next) show the placements measured for each.
- **Memory.** Host experts pin 34 GB of RAM for Q2_0, 50 GB for IQ3_S and 25 GB for the Coder
  build. Disk experts need under 1 GB of RAM; the page cache does the rest. The GPU expert cache
  takes what is free after startup, or `--expert-cache-mib`.
- **The n-gram table.** Its rows are read from the file, 16 per token, unless
  `--ngram-residency ram` loads all 28.8 GB or `ram-hot` keeps the rows a profile ranks first in a
  RAM budget ([the n-gram rows](docs/qwen3-8-flash-next.md#the-n-gram-rows)). A model started
  without its table is refused; `--no-ngram-table` overrides that, an experimental mode with no
  practical use.
- **Drafting.** `--spec mtp --draft-tokens 3` drafts with the MTP block the published artifacts
  carry.
- **Images and video.** `--vision` adds the Vision tower (0.9 GB on the GPU).

</details>

<details>
<summary><b>Gemma 4 26B-A4B</b>: converted from Google's QAT GGUF</summary>

```bash
# The release's GGUF and its Hugging Face text files; one conversion writes the artifact.
mkdir -p models/gemma4/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
         processor_config.json; do
  curl -fsSL -o models/gemma4/hf/$f https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/$f
done
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf gemma-4-26B-it-mmproj.gguf \
  --local-dir models/gemma4
# With the release's QAT assistant (multi-token prediction) as the artifact's `mtp` component and
# its vision tower (the mmproj) as the `vision` component.
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF mtp-gemma-4-26B-A4B-it.gguf --local-dir models/gemma4
python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf --components text,mtp,vision \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf \
  --source mtp=models/gemma4/mtp-gemma-4-26B-A4B-it.gguf \
  --source vision=models/gemma4/gemma-4-26B-it-mmproj.gguf --name Gemma-4-26B-A4B-it-QAT-Q4_0-MTP \
  --out models/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --device cpu

# Four concurrent requests over 40K-token contexts, two drafts a round, images and video; thinking
# stays off unless a request asks.
ninfer-serve models/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4 --spec mtp --draft-tokens 2 --vision
```

</details>

<details>
<summary><b>Qwen3.6-35B-A3B NVFP4</b>: RTX 50 series and RTX PRO 6000 only</summary>

```bash
ninfer-serve models/Qwen3.6-35B-A3B-NVFP4-ninfer-v3.ninfer --model-id qwen3.6-35b-a3b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --vision
```

The NVFP4 experts prefill through W4A4 on Blackwell's FP4 tensor cores, so this needs a `120a`
build (the image has one); `sm_86` and `sm_89` builds refuse the file. On an RTX 5090 it starts in
17 s and leaves 7.4 GiB of the card free.

</details>

<details>
<summary><b>Qwen3.8-27B fine-tunes</b>: Huihui abliterated, HauhauCS Aggressive, MXFP8-CRACK</summary>

All three have the official Qwen3.8-27B identity and size, so the Qwen3.8-27B recipes above serve
them too; CRACK has no DFlash2 adapter. Their cards' configurations:

```bash
# Huihui abliterated on one RTX 3090: 198,400 tokens with MTP and Vision.
ninfer-serve models/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 \
  --vision --vision-residency overlay --vision-max-merged 12288

# HauhauCS Aggressive: DFlash2 with seven drafts.
ninfer-serve models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-ninfer-v3.ninfer \
  --model-id qwen3.8-27b --max-context 32768 --kv-capacity auto \
  --spec dflash2 --draft-tokens 7

# MXFP8-CRACK: MTP.
ninfer-serve models/Qwen3.8-27B-MXFP8-CRACK-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 32768 --kv-capacity auto --spec mtp --draft-tokens 3
```

</details>

<details>
<summary><b>A card without a built-in profile</b></summary>

```bash
ninfer-calibrate --print > my-gpu.json
```

The engine measures it by itself at first start. Running it by hand refreshes the profile after a
driver or clock change. See [device profiles](docs/device-profiles.md).

</details>

## Artifacts

The Bonsai, GSQ-RCO, Flash-Next and Gemma 4 artifacts use formats that only this engine reads.

| model | artifact | size | contents |
|---|---|---:|---|
| Ternary Bonsai 2 27B | [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 GiB | ternary weights, Vision, Bonsai-trained MTP head and DFlash2 adapter, exact proposal head |
| Qwen3.8-27B | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.03 GiB | the official `groupwise-int` (Q4/Q5) with MTP and DFlash2; the reference tables' artifact |
| Qwen3.8-27B GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 GiB | ISTA-DASLab's 3.5-bit GGUF blocks byte for byte, Q6_K MTP head, Vision, DFlash2, proposal head |
| Qwen3.8-27B, abliterated | [WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 GiB | the official `qwen3_8_27b` recipe with MTP, DFlash2 and a proposal head |
| Qwen3.8-27B Uncensored, HauhauCS Aggressive | [WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3) | 19.03 GiB | `groupwise-int` with the tune's MTP head, DFlash2 and a proposal head |
| Qwen3.8-27B MXFP8-CRACK | [WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3) | 16.96 GiB | `groupwise-int` with MTP and a proposal head; no DFlash2 |
| Qwen3.8-Flash-Next GSQ-RCO Q2_0 | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.49 GiB | ISTA-DASLab's 2.4-bit GGUF blocks byte for byte, Vision, shared-Q8_0 MTP; reads the n-gram table |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.50 GiB | the 3.5-bit release with Vision and shared-Q8_0 MTP |
| Qwen3.8-Flash-Next Coder GSQ-RCO IQ1_M | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 31.02 GiB | the expert-pruned coding build (256 experts per layer), Vision, shared-Q8_0 MTP |
| Qwen3.8-Flash-Next n-gram table | [WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.92 GiB | unchanged IQ4_NL rows plus an optional broad hot-row profile (`--ngram-table`) |
| Qwen3.6-35B-A3B NVFP4 | [WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 20.39 GiB | RedHatAI's NVFP4 experts code for code, Q8 projections, Vision, MTP, proposal head; `sm_120a` GPUs only |
| Gemma 4 26B-A4B it QAT | [WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3) | 14.75 GiB | Google's QAT Q4_0 GGUF blocks byte for byte with the QAT MTP assistant and the vision tower |

The official NInfer artifacts (Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B, `groupwise-int` and
`nvfp4`) load too; the [documentation map](docs/README.md#model-artifacts) links them. Weight
conversion shows how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Docker

`ghcr.io/iamwavecut/ninfer-all:latest` is built from the newest `master` commit that passes CI, on
CUDA 13.4 and Ubuntu 26.04; a commit that changes only documentation or tooling the image does not
carry keeps the image before it. It carries two builds and starts the one that matches the GPU: `sm_86` for
the RTX 30 and RTX 40 series, `sm_120a` for the RTX 50 series and the RTX PRO 6000 Blackwell.

- **Host.** An NVIDIA driver of the CUDA 13 branch (580 or newer) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Tags.** `latest`, `sha-<commit>` and the `VERSION`; the image is 1.6 GB compressed.
- **Volumes.** `/models` holds artifacts, `/cache` the device profile measured on first start, and
  `/grafts/<model>/` optional [prompt grafts](docs/serving.md#prompt-grafts).
- **Checked** on an RTX 3090 and an RTX 5090 with driver 580.159.03.

The container's command chooses what runs:

| command | runs |
|---|---|
| `serve`, `ninfer`, `perplexity`, `calibrate` `[args]` | that binary with any artifact and flags; `serve` listens on `0.0.0.0:8080` unless given `--host` or `--port` |
| `run <model> [profile]` (the default: `run qwen38-27b`) | the [launcher profiles](docs/maintainer/launcher-profiles.md) of `scripts/run.sh` for the official `qwen38-27b` and `qwen36-35b-a3b`, with its `NINFER_*` overrides (`-e NINFER_SPEC=mtp`, `-e NINFER_CONTEXT=131072`, ...) |
| `download <model>` | `scripts/download-model.sh` into `/models`: `qwen38-27b`, `qwen36-27b` or `qwen36-35b-a3b` |

The official Qwen3.8-27B with the measured `tuned` profile, on http://localhost:8080/v1:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all run qwen38-27b
```

[compose.yaml](compose.yaml) wires the GPU, the port and the volumes for the same:
`docker compose run --rm ninfer download qwen38-27b`, then `docker compose up -d`.
`NINFER_IMAGE_ARCH=sm86|sm120a` overrides the GPU detection, and `docker build -t ninfer .` builds
the image from source (`--build-arg ARCHS=86` for one architecture).

## Building

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve ninfer-calibrate
```

`CMAKE_CUDA_ARCHITECTURES` is `86` for the RTX 30 series (that build also runs the RTX 40 series),
`89` for the RTX 40 series and `120a` for the RTX 50 series and the RTX PRO 6000 Blackwell (on the
`mma.sync` compatibility path, which the ternary route needs). CUDA 12.8 or newer builds `86` and
`89`; a `120a` build needs CUDA 13.1 or newer, since 12.8 and 12.9 miscompile sm_120a kernels and
configure refuses them. The [Linux](docs/build-linux.md) and [Windows](docs/build-windows.md)
guides cover dependencies, the opt-in build options and the launcher scripts;
[tests](tests/README.md) and [benchmarks](bench/README.md) have their own guides.

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices. The Qwen3.8-Flash-Next artifacts carry the Qwen
Community License 1.0 of their model.
