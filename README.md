# NInfer-all

A C++/CUDA inference engine for the Qwen3.5 model family (Qwen3.6 and Qwen3.8 27B, Qwen3.6-35B-A3B,
their fine-tunes and Ternary Bonsai 2), Qwen3.8-Flash-Next and Gemma 4 26B-A4B, on NVIDIA GPUs from
Volta to consumer Blackwell, from 8 GB of memory. It serves the OpenAI, Anthropic and llama.cpp APIs
from one GPU, or from up to eight as pipeline stages on Linux.

[Quick start](#quick-start) · [Supported GPUs](#supported-gpus) · [Models](#artifacts) ·
[Highlights](#highlights) · [Features](#features) · [Docker](#docker) · [Building](#building) ·
[Documentation](docs/README.md)

NInfer-all continues [NInfer](https://github.com/Neroued/ninfer) through the NInfer-3090 line and
carries the work of several forks; [CONTRIBUTORS.md](CONTRIBUTORS.md) credits each of them.

## Quick start

Download an artifact ([all models](#artifacts)) and serve it from the [Docker image](#docker). The server
speaks the OpenAI and Anthropic APIs on http://localhost:8080/v1, and each card picks up its device
profile on its own.

```bash
hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local-dir models

docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 \
  --gdn-state-fp16 --spec dflash2 --draft-tokens 5

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-27b","messages":[{"role":"user","content":"Hello"}]}'
```

Flags for every model and memory size, from an 8 GB card to the largest context: [launch recipes](docs/recipes.md).

<details>
<summary><b>Without Docker</b> — the same flags on a build</summary>

```bash
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

A build listens on 127.0.0.1:8080 unless `--host` and `--port` say otherwise. [Building](#building) shows
how to make one.

</details>

## Supported GPUs

NVIDIA GPUs from Volta (compute capability 7.0) to consumer Blackwell (12.0), from 8 GB of memory:
Ternary Bonsai 2 27B runs on an 8 GB RTX 3060 Ti, and Qwen3.8-Flash-Next streams its experts from
host memory or disk through 8 GB. The [Docker image](#docker) carries the `75`, `86` and `120a` builds
and starts the one that matches the GPU. Every card by name, with what has run on it:
[Supported GPUs](docs/gpus.md).

*Click a section to expand it.*

<details>
<summary><b>Architectures and builds</b> — six architectures, one build each</summary>

| Architecture | Compute capability | Build | CUDA | In the image | Status |
|---|---|---|---|---|---|
| Blackwell: GeForce RTX 50, RTX PRO Blackwell | 12.0 | `120a` | 13.1 or newer | yes | measured; built-in device profiles |
| Ada Lovelace | 8.9 | `89`, or the `86` build | 12.8 or newer | the `86` build | measured; built-in device profiles |
| Ampere | 8.6 | `86` (the default) | 12.8 or newer | yes | measured (the reference card); built-in device profiles |
| Ampere, A100 class | 8.0 | `80` | 12.8 or newer | no | built-in device profiles; not benchmarked |
| Turing | 7.5 | `75` | 12.8 or newer | yes | measured; built-in device profiles |
| Volta | 7.0 | `70` | 12.8 or 12.9 | no: [source build](docs/build-linux.md#tesla-v100-sm_70) | measured; built-in device profiles |
| Hopper; data-center Blackwell; GB10; Pascal and older | | none | | | no build |

</details>

<details>
<summary><b>What fits in your GPU memory</b> — measured, from 8 GB to 48 GB</summary>

| GPU memory | Runs | Measured on |
|---|---|---|
| 8 GB | Ternary Bonsai 2 27B (24,576 tokens with `rk4v4`, no MTP); Qwen3.8-Flash-Next with its experts in host memory or on disk | RTX 3060 Ti |
| 11 GB | Ternary Bonsai 2 27B (200,704 tokens with `rk4v4`; 143,360 with MTP) | RTX 2080 Ti |
| 12 GB | Ternary Bonsai 2 27B (266,240 tokens); Qwen3.8-27B GSQ-RCO IQ3_S (8,192 tokens, no MTP) | RTX 3060 |
| 16 GB | Gemma 4 26B-A4B Q4_0 (61,440 tokens); Qwen3.8-27B IQ3_S (241,664 tokens); the official Qwen3.8-27B with `--embedding-q4 --lm-head-q4` | Tesla V100-SXM2-16GB |
| 24 GB | the official Qwen3.8-27B and its MXFP8 tune; Qwen3.6-35B-A3B `groupwise-int` on an RTX 3090; Flash-Next with every expert on the GPU over two cards | RTX 3090, RTX 4090, TITAN RTX |
| 32 GB | Qwen3.6-35B-A3B NVFP4 (Blackwell only) | RTX 5090 |
| 48 GB | Flash-Next Q2_0 with every expert on one card | L40S, RTX PRO 6000 |

</details>

<details>
<summary><b>Device profiles and limits</b> — how a card gets its kernel routes</summary>

- **Device profiles.** A profile picks the kernel route of every operation and width. The binary
  carries profiles measured on 47 GPU models, from the Tesla V100 and RTX 2060 Super to the RTX PRO
  6000 (the Profile column of [supported GPUs](docs/gpus.md#cards)); a built-in profile applies to
  the same GPU name, compute capability and SM count. Any other card is calibrated once at first
  start, which takes about half a minute to three and a half minutes, and the result is stored for
  later starts; `ninfer-calibrate` measures again after a driver or clock change
  ([device profiles](docs/device-profiles.md)). The `70` build runs its own
  routes, so the V100 profiles hold only the GDN prefill switch and, for the 32 GB card, the Q8 MoE
  prefill frontier.
- **Architecture limits.** NVFP4 expert weights need a `120a` build. The Turing and Volta builds
  replace instructions their GPUs lack and reshape kernels to their shared memory
  ([Turing](docs/build-linux.md#turing-sm_75), [Tesla V100](docs/build-linux.md#tesla-v100-sm_70)).
  Configure accepts no other target, so Pascal and older, Hopper and data-center Blackwell have no
  build.
- **Speed per card.** [Performance](docs/performance.md) lists the measured rates of every card.

</details>

## Artifacts

Each name links to its Hugging Face card; `hf download REPO FILE.ninfer --local-dir models` fetches the
file. The Bonsai, GSQ-RCO, Flash-Next and Gemma 4 artifacts use formats that only this engine reads.
✓ means the artifact carries the component, — that it does not.

| Model | Size, GiB | Weights | Vision | MTP | DFlash2 | Proposal head | Notes |
|---|---:|---|:-:|:-:|:-:|:-:|---|
| [Ternary Bonsai 2 27B](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 | ternary, 2.125 bits | ✓ | ✓ | ✓ | ✓ | Bonsai-trained MTP head and DFlash2 adapter |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.03 | groupwise-int (Q4/Q5) | ✓ | ✓ | ✓ | ✓ | the official artifact; the reference tables use it |
| [Qwen3.8-27B GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 | GGUF blocks, 3.5 bits | ✓ | ✓ | ✓ | ✓ | ISTA-DASLab's GGUF blocks byte for byte, Q6_K MTP head |
| [Qwen3.8-27B, abliterated](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 | groupwise-int (Q4/Q5) | ✓ | ✓ | ✓ | ✓ | the official recipe |
| [Qwen3.8-27B Uncensored, HauhauCS Aggressive](https://huggingface.co/WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3) | 19.03 | groupwise-int (Q4/Q5) | ✓ | ✓ | ✓ | ✓ | the tune's MTP head |
| [Qwen3.8-27B MXFP8-CRACK](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3) | 16.96 | groupwise-int | ✓ | ✓ | — | ✓ |  |
| [Qwen3.6-35B-A3B NVFP4](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 20.39 | NVFP4 experts, Q8 projections | ✓ | ✓ | — | ✓ | RedHatAI's NVFP4 experts code for code; `sm_120a` GPUs only |
| [Qwen3.8-Flash-Next GSQ-RCO Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.49 | GGUF blocks, 2.4 bits | ✓ | ✓ | — | — | shared-Q8_0 MTP; reads the n-gram table |
| [Qwen3.8-Flash-Next GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.50 | GGUF blocks, 3.5 bits | ✓ | ✓ | — | — | shared-Q8_0 MTP; reads the n-gram table |
| [Qwen3.8-Flash-Next Coder GSQ-RCO IQ1_M](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 31.02 | GGUF blocks | ✓ | ✓ | — | — | expert-pruned coding build, 256 experts per layer; shared-Q8_0 MTP; reads the n-gram table |
| [Gemma 4 26B-A4B it QAT](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3) | 14.75 | GGUF Q4_0 blocks | ✓ | ✓ | — | — | Google's QAT GGUF byte for byte; the MTP column is Google's QAT assistant |

The three Flash-Next releases read one shared n-gram table, which the model names by digest: the
[n-gram table](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) (26.92 GiB,
unchanged IQ4_NL rows plus an optional broad hot-row profile), passed with `--ngram-table`.

The official NInfer artifacts (Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B, `groupwise-int` and
`nvfp4`) load too; the [documentation map](docs/README.md#model-artifacts) links them. Weight
conversion shows how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Tesla V100-32GB (sm_70), single card

This branch is the **single-GPU Tesla V100-PCIE-32GB (sm_70)** line. The optimizations live in
operator routing and weight prepacking, done at load time or on first use; no route changes its
numeric contract, the v3 path is byte-identical to before, and v2 artifacts are read where they
are. Every number below is one request with a 27K-token prompt on that card (int8 KV, MTP with four
drafts, greedy).

### Supported artifacts and measured performance

| Artifact | Container | Weights | Context | 27K-token request |
|---|---|---:|---:|---|
| `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3.ninfer` (native `nvfp4` profile, production default) | v3 | 19.1 GiB | 254,208 | TTFT **27.2 s**, prefill **1.00k tok/s**, decode **76-78 tok/s** |
| `Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2.ninfer` (same family, **v2, read in place**) | v2 | 22.1 GiB | 200,000 | TTFT **27.3 s**, prefill **998 tok/s**, decode **82.7 tok/s** |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` (GGUF-derived, IQ3_S among 18 block formats) | v3 | 12.9 GiB | 400,000 | TTFT **30.8 s**, prefill **885 tok/s**, decode **66-74 tok/s** |

Short-context decode (1K prompt) is **82.3 tok/s**, and MTP accepts 69% of its drafts on the 27K
prompt. `Huihui-...-nvfp4` and `orcarouter-...-nvfp4` (v2, the same native family) load as well, so do both Gemma 4 artifacts below.

#### Gemma 4

Both Gemma 4 artifacts are GGUF-derived and run through the ggml bridge, but their shapes differ:
the 26B-A4B routes 8 of 128 experts per token, while the 31B is **dense** — 60 layers at hidden
5376, no experts. Both were measured on this card, one request each, a 5,513-token prompt over int8
KV with 500 output tokens ([Gemma 4 on the V100-32GB](docs/performance/gemma4-v100-32gb-2026-10.md)):

| Artifact | Container | Weights | Measured |
|---|---|---:|---|
| `Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP-ninfer-v3.ninfer` (dense, hidden 5376) | v3 | 17.6 GiB | prefill **715 tok/s**; decode **54.1 tok/s** (MTP, 4 drafts, 54.5% acceptance) or **27.4 tok/s** without speculation. 32,768 tokens of int8 KV leave 10.5 GiB free |
| `Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP-ninfer-v3.ninfer` (128 experts, 8 per token) | v3 | 16.9 GiB | prefill **2.39k tok/s**; decode **157.2 tok/s** (MTP, 2 drafts, 71.7% acceptance) or **120.3 tok/s** without speculation. 14.1 GiB free at 28,672 tokens |

The dense 31B reads all 17.6 GiB per token, so unspeculated decode is bandwidth: 27.4 tok/s is
about 517 GB/s of the ~790 GB/s this card's HBM2 moves with ECC on. Speculation is what lifts it —
four drafts verified per weight read take the same traffic to 54.1 tok/s. The A4B's file is only 4%
smaller, but a token touches 8 of 128 experts, so it decodes 4.4× faster and gains less from
drafting (1.3× against 2.0×).

### Two artifact families, and why their numbers differ

The container version does **not** affect speed: the same weights measure 998 tok/s as a v2 file and
997 tok/s as the upgraded v3 file, and this branch reads both in place. What changes the numbers is
the **weight recipe**:

* **Native NInfer profiles**, quantized by the NInfer converter from HF safetensors,
  compressed-tensors or block-FP8 sources. Formats: `nvfp4` (block scale, k16 x m128 x 4),
  `fp8_e4m3fn_row_bf16` (one FP8 code per weight plus one BF16 scale per output row),
  `q4_g64_fp16` / `q5_g64_fp16` / `q6_g64_fp16` / `q8_g32_fp16` (group-wise, FP16 scale), plus
  `bf16`, `fp32` and `int32`. These get the fast routes: QPN prepacking for `t <= 32`, CUTLASS
  wide-T over a dequantized staging plane, and a row-scale FP8 dequant kernel.
* **GGUF-derived artifacts**, converted from a `.gguf` file: 19 registered formats (`gguf_q4_k`,
  `gguf_q8_0`, `gguf_iq3_s`, `gguf_iq4_xs`, `gguf_iq2_s`, ...) in the `gguf_blocks_v1` layout.
  They run through the ggml bridge, meaning per-tensor block dequantization plus the matching
  MMQ/Vec kernels. That is the honest price of a 2-4 bit block format: the GSQ artifact mixes 18
  formats with IQ3_S holding only 3.39 GiB of its 12.9 GiB, so its prefill fragments into roughly
  twenty thousand small dequantize and GEMM kernels and lands 12% below the native profile.

The archived reference build (`archive/fork-master`) serves the native profiles only: asked to load
a GGUF-derived artifact it stops with `unknown tensor format: gguf_iq4_xs`. This branch serves both
families and both container versions.

### Principal changes

| Change | Measured effect |
|---|---|
| **Volta QPN**: quadpair-split-N prepacked GEMMs for nvfp4 and fp8 (`t <= 32` QPN, wide T CUTLASS) | projection routes 85 to 480-620 GB/s |
| **GDN prefill on the recurrent kernel** (chunked and two-stage are a regression on sm_70) | GDN T=1024...8192 **2.9-3.3x**; 3.75 to 2.18 s of the 27K prefill |
| **Flash-attention prefill kernel**: paged-to-contiguous staging plus an sm_70 `mma.sync` kernel | same shape 41.1 to 25.1 ms; 27K TTFT 30.1 to 27.2 s |
| **Speculative-decode configuration**: `--ngram-draft-tokens 0` and no `--adaptive-mtp` (the default 15 n-gram drafts dilute the MTP acceptance rate) | 27K decode **64.2 to 78.3 tok/s**, greedy output byte-identical |
| **FP8 SwiGLU wide T on an FP32-output CUTLASS path** (it used to fall into a four-token-chunk SIMT ladder) | mixed-quantization prefill **508 to 998 tok/s** |
| **v2 artifacts read in place**: the directory is normalized to the v3 shape in memory, the weights are read where they are | same numbers as the upgraded copy, no offline conversion |


### Quick start

```bash
# One card (host GPU 1 = V100-32GB). The production compose file lives in the Docker repo.
docker run -d --gpus device=1 -v /path/to/models:/workspace/models \
  --entrypoint ninfer-serve ninfer:v100 \
  /workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3.ninfer \
  --host 0.0.0.0 --port 8000 --api-key sk-xxx --model-id qwen --max-context 200000 \
  --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 --max-concurrency 2 \
  --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0
```

Build with `Dockerfile.sm70.local` (CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=70`, full build) or
`Dockerfile.sm70-runtime.local` (repack an already built `ninfer-serve` in seconds).

### Validation and documentation

Every change has to pass both the upstream FP64 oracles (`ctest`) and a 27K-token end-to-end A/B.
Cross-line comparisons use `bench/v100/compare_lines.py` (warm runs, markdown table), and readiness
is `GET /health` (503 while loading, 200 when ready). Designs and measurements live in
[`docs/v100/`](docs/v100/): the roadmap is [PLAN.md](docs/v100/PLAN.md), with one measurement
document per stage.

## Highlights

Ternary Bonsai 2 27B decodes 397 tok/s on an RTX 5090 with DFlash2 and holds a 978,944-token context with
every planted code found. Qwen3.8-Flash-Next decodes 138 tok/s with every expert on one RTX PRO 6000, and
74 tok/s with its experts in host memory on one RTX 5090. Every table of every card is on the
[performance page](docs/performance.md).

*Click a section to expand it.*

<details>
<summary><b>Ternary Bonsai 2 27B</b> — up to 397 tok/s, a million tokens of context</summary>

Measured in September 2026, one card each, greedy, one request unless the column says otherwise.

| Card | Short chat, DFlash2 7 drafts (tok/s) | After a 261K-token document (tok/s) | 261K-token prompt, to the first token | Largest context, all three codes found | Eight requests at once, MTP 3 (tok/s) |
|---|---:|---:|---:|---:|---:|
| RTX 3090 | 202 | 90 | 215 s | 970,752 | 551 |
| RTX 4090 | 256 | 123 | 102 s | 958,464 | 824 |
| RTX 5090 | 397 | 218 | 82 s | 978,944 | 1,063 |
| RTX PRO 6000 | 381 | 218 | 78 s | 1,048,576, two of three codes | 1,155 |

The RTX PRO 6000 (96 GB) starts at the engine's ceiling of 1,048,576 tokens with every KV format and drafter.
Filled to it, the model misses the code at 90% (about 943K); up to about 880K it found every code on every card.

</details>

<details>
<summary><b>Qwen3.8-27B</b> — up to 237 tok/s, 872,448 tokens of context filled</summary>

The official `groupwise-int` artifact, measured in September 2026, one card each, greedy, one request unless the column says otherwise.

| Card | Short chat, DFlash2 7 drafts (tok/s) | Largest context, all three codes found | Eight requests at once, MTP 3 (tok/s) |
|---|---:|---:|---:|
| RTX 3090 | 118 | 417,792 | 329 |
| RTX 4090 | 149 | 405,504 | 442 |
| RTX 5090 | 236 | 872,448 | 690 |
| RTX PRO 6000 | 237 | 1,048,576, two of three codes | 739 |

- **RTX PRO 6000 against RTX 5090.** Re-measured in October beside an RTX 5090, both at 600 W, the
  PRO 6000 came back within 2% of September wherever the drafts accepted the same share. One
  request's decode is bound by memory bandwidth, and both cards have 1.79 TB/s of GDDR7 (Bonsai 2
  without speculation: 167.7 against 170.4 tok/s). The PRO 6000's 188 SMs against 170 show where
  compute decides: the 261K prompt is 2% faster and eight requests at once 6 to 7% faster
  ([re-check](docs/performance.md#rtx-pro-6000-against-rtx-5090-october-2026)).
- **Draft length.** DFlash2 with seven drafts is fastest on short answers; after long documents the
  best count is three to seven. MTP runs up to fifteen drafts and is fastest at three to five.

</details>

<details>
<summary><b>Qwen3.8-Flash-Next</b> — 138 tok/s with the experts on a GPU, 74 tok/s from host memory</summary>

The Q2_0 release (35.9 GiB model and 26.8 GiB n-gram table), one request, greedy, October 2026. Decode of
a short answer, in tok/s:

| Card | Every expert on the GPUs | Experts in host memory | Experts on disk, in the page cache | Experts on disk, cold |
|---|---:|---:|---:|---:|
| RTX PRO 6000 | 138 (one card) | 55 | 76 | 35 |
| RTX 5090 | 136 (two cards) | 74 | 68 | 25 |
| RTX 4090 | 97 (two cards) | 53 | 42 | 18 |
| RTX 3090 | 90 (two RTX 3090 Ti) | 49 | 47 | 17 |

Prefill of a 4,463-token prompt, in tok/s:

| Card | Every expert on the GPUs | Experts in host memory | Experts on disk, in the page cache | Experts on disk, cold |
|---|---:|---:|---:|---:|
| RTX PRO 6000 | 2,939 | 1,297 | 2,024 | 1,000 |
| RTX 5090 | 3,809 | 1,677 | 1,739 | 694 |
| RTX 4090 | 3,608 | 1,138 | 723 | 160 |
| RTX 3090 | 1,504 | 842 | 630 | 195 |

- **The host matters.** Host and disk rows depend on the host as much as on the card. The RTX 5090
  host had PCIe 5.0 x16; the others had PCIe 4.0 x16. The RTX 4090 rows ran pinned to the GPUs' NUMA
  node in a two-socket VM; unpinned, host experts decode there at 40 tok/s and disk experts at 32 to 33.
- **Older host and disk rows.** They were measured before the October 9 decode work. Host experts on
  an RTX 3090 now decode 87-94 tok/s in a different workload, 98-100 with a repeated prompt
  ([host experts on one RTX 3090](docs/qwen3-8-flash-next.md#host-experts-on-one-rtx-3090)).
- **Other releases and cards.** IQ3_S, the Coder IQ1_M build and the 8 and 12 GB cards are in the
  [expert placements](docs/performance.md#expert-placements). Six reasoning requests at once on two
  RTX 5090s produced 193 tok/s of output with Q2_0.

</details>

## Features

*Click a section to expand it.*

<details>
<summary><b>Models and weight formats</b> — Qwen3.5 family, GGUF block formats, Ternary Bonsai 2, Flash-Next, Gemma 4</summary>

- **Qwen3.5 family.** Dense and MoE (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) with MTP,
  DFlash (35B-A3B) and DFlash2 (Qwen3.8-27B) drafters, Vision for images and video, and
  `groupwise-int`, FP8, NVFP4 and ternary weights. Native MoE routed experts can stay in pinned
  host memory or the artifact's files, brought into whole-layer GPU banks with an optional device
  cache ([expert residency](docs/maintainer/qwen3_5-model.md#routed-expert-residency)). NVFP4
  expert banks prefill through W4A4 and need a `120a` build.
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


- **Gemma 4 26B-A4B and 31B.** Google's QAT GGUF blocks, imported block for block (13.4 GiB for the
  26B-A4B, 17.6 GiB for the dense 31B), or a mixed-format GGUF of either: the sliding-window and
  full-attention blocks (the latter caching values only), the dense MLP beside 128 routed experts,
  every `--kv-dtype` format, up to eight concurrent requests with prefix reuse through host memory
  and shared system prompts, tool calls and structured output, images and video through the
  release's vision tower (on the GPU, streamed from pinned memory or on CPU threads), and Google's
  multi-token-prediction assistant for speculative decoding. The dense 31B is wide: its row kernels
  cover `hidden` 5376 as well as 2816/3584, so a 60-layer dense artifact runs unchanged. On an RTX
  3090 the 26B-A4B prefills a 10,000-token prompt at 6,045 tok/s and decodes at 215 tok/s, 190 after
  10,000 tokens, against llama.cpp's 4,697, 169 and 152 on the same GGUF; with two drafts a round
  chat answers decode at 291 tok/s instead of 209 ([Gemma 4](docs/gemma4.md)).er

</details>

<details>
<summary><b>Speculative decoding</b> — MTP, DFlash, DFlash2 and n-gram copying</summary>

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

</details>

<details>
<summary><b>Context and memory</b> — nine KV formats, up to 1,048,576 tokens, a context cache with a disk tier</summary>

- Nine KV formats, from `bf16` to `rk2v4-e8` at 216 bytes per token and KV head; `rk8v4` is the
  all-round choice and `rk4v4` holds the native 262,144-token window of Qwen3.8-27B on a 24 GB card
  ([context and memory](docs/cli.md#context-and-memory)).
- Up to 1,048,576 tokens of context, four times the native window, with plain RoPE or YaRN
  (`--rope-yarn`).
- A hybrid prefix cache by default, which shares content-addressed KV blocks and sparse state
  snapshots across requests on the device and in pinned host memory; or a checkpoint catalog
  (`--use-original-prefix-caching`) with an optional disk tier that survives restarts
  (`--disk-kv-path`, `--disk-kv-restore`) and endpoint and branch anchors that let an edited or
  regenerated turn resume where it left the conversation.
- Precision trades that buy context: `--embedding-q4`, `--lm-head-q6`, `--gdn-state-fp16`
  ([quality trades](docs/maintainer/quality-trade-experiments.md)).

</details>

<details>
<summary><b>Serving</b> — OpenAI, Anthropic and llama.cpp APIs, model suspend, several models behind one server</summary>

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

</details>

<details>
<summary><b>Kernels and devices</b> — route profiles per card, pipeline stages, FP8 and NVFP4 tensor-core routes</summary>

- **Device route profiles.** Each card's measured profile picks the kernel route for every
  operation and width: built in for the cards in [supported GPUs](#supported-gpus), measured once
  at first start on any other ([device profiles](docs/device-profiles.md)).
- **Pipeline stages.** `--devices 0,1,...` splits the layers over up to eight GPUs on Linux, each
  stage owning its weights, KV and state; the cards may differ where the build runs on each, such as
  RTX 30 with RTX 40 cards on the `86` build ([pipeline stages](docs/maintainer/pipeline-stages.md)).
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
  ghcr.io/jimmymax/ninfer-v100 serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
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
  ghcr.io/jimmymax/ninfer-v100 serve /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer \
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
| Gemma 4 26B-A4B, HauhauCS uncensored | [HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP](https://huggingface.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP) via `tools.convert` | 16.94 GiB | the tune's mixed GGUF blocks byte for byte with its MTP assistant and vision tower; the 26B-A4B the [Gemma 4 measurements](docs/performance/gemma4-v100-32gb-2026-10.md) use |
| Gemma 4 31B it QAT, dense | [HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP](https://huggingface.co/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP) via `tools.convert` | 17.45 GiB | the tune's Q4_K_M, Q6_K and Q4_0 GGUF blocks byte for byte, its MTP assistant and the vision tower; dense, 60 layers, hidden 5376 ([wide rows](docs/v100/gemma4-wide-rows-2026-10-16.md)) |

The official NInfer artifacts (Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B, `groupwise-int` and
`nvfp4`) load too; the [documentation map](docs/README.md#model-artifacts) links them. Weight
conversion shows how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Docker

`ghcr.io/jimmymax/ninfer-v100:latest` **is the Tesla V100 build**: sm_70, CUDA 12.8 on Ubuntu 24.04,
packed by the release pipeline from [Dockerfile.sm70](Dockerfile.sm70). `:<version>` pins a release
(`1.6.1`), and `sm70` is kept as an alias. Like the other ninfer images it sets no entrypoint, so the
command names `ninfer-serve` first:

```bash
docker run --rm --gpus all -p 8000:8000 --ulimit memlock=-1 \
  -v "$PWD/models:/models" ghcr.io/jimmymax/ninfer-v100:latest \
  ninfer-serve /models/model.ninfer --host 0.0.0.0 --port 8000 --model-id model \
  --max-context 32768 --kv-dtype int8
```

- **Host.** An NVIDIA driver of the CUDA 12 branch or newer and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Tags.** `latest`, `<version>` and `sm70`; `sha-<commit>` for a single commit. About 3 GB compressed.
- **Volumes.** `/models` holds the artifacts.

[deploy/compose.gemma-v100.example.yml](deploy/compose.gemma-v100.example.yml) is the same service in
a compose file; its `command:` list names `ninfer-serve`, which is what the V100 image expects.

### The CUDA 13 image (RTX 30, 40 and 50)

`ghcr.io/jimmymax/ninfer-v100:sm86-sm120a` is the other build, from the upstream [Dockerfile](Dockerfile):
`sm_86` for the RTX 30 and RTX 40 series, `sm_120a` for the RTX 50 series and the RTX PRO 6000
Blackwell, CUDA 13.4 on Ubuntu 26.04. **No V100 can run it** — on this card use `latest`.

- **Tags.** `sm86-sm120a` and `sha-<commit>`; 1.6 GB compressed.er
- **Volumes.** `/models` holds artifacts, `/cache` the device profile measured on first start, and
  `/grafts/<model>/` optional [prompt grafts](docs/serving.md#prompt-grafts).
- **Checked** on an RTX 3090 and an RTX 5090 with driver 580.159.03.

Its command chooses what runs:

| command | runs |
|---|---|
| `serve`, `ninfer`, `perplexity`, `calibrate` `[args]` | that binary with any artifact and flags; `serve` listens on `0.0.0.0:8080` unless given `--host` or `--port` |
| `run <model> [profile]` (the default: `run qwen38-27b`) | the [launcher profiles](docs/maintainer/launcher-profiles.md) of `scripts/run.sh` for the official `qwen38-27b` and `qwen36-35b-a3b`, with its `NINFER_*` overrides (`-e NINFER_SPEC=mtp`, `-e NINFER_CONTEXT=131072`, ...) |
| `download <model>` | `scripts/download-model.sh` into `/models`: `qwen38-27b`, `qwen36-27b` or `qwen36-35b-a3b` |

The official Qwen3.8-27B with the measured `tuned` profile, on http://localhost:8080/v1:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/jimmymax/ninfer-v100:sm86-sm120a download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/jimmymax/ninfer-v100:sm86-sm120a run qwen38-27b
```

[compose.yaml](compose.yaml) wires that image's GPU, port and volumes:
`docker compose run --rm ninfer download qwen38-27b`, then `docker compose up -d`.

`NINFER_IMAGE_ARCH=sm86|sm120a` overrides the GPU detection, and `docker build -t ninfer .` builds it
from source (`--build-arg ARCHS=86` for one architecture).er

## Building

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve ninfer-calibrate
```

`CMAKE_CUDA_ARCHITECTURES` takes one build from [supported GPUs](#supported-gpus), or a list of them
for a pipeline across cards of different architectures
([several architectures](docs/build-linux.md#cards-of-several-architectures-in-one-pipeline)).

<details>
<summary><b>Toolchains and options</b> — CUDA versions per target, guides</summary>

`120a` runs the `mma.sync` compatibility path, which the ternary route needs. CUDA 12.8 or newer builds
`75`, `80`, `86` and `89`, and `70` needs CUDA 12.8 or 12.9. A `120a` build needs CUDA 13.1 or newer, since
12.8 and 12.9 miscompile sm_120a kernels and configure refuses them. The [Linux](docs/build-linux.md)
and [Windows](docs/build-windows.md) guides cover dependencies, the opt-in build options and the
launcher scripts; [tests](tests/README.md) and [benchmarks](bench/README.md) have their own guides.

</details>

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices. The Qwen3.8-Flash-Next artifacts carry the Qwen
Community License 1.0 of their model.
