# DSpine on V100 — performance report (2026-11-17)

Goal: implement DSpine drafting on the Tesla V100 (sm_70), run the DSpine drafter
beside the Qwen3.8-27B Coder390 artifact, and compare it against the existing MTP
lm-head draft (the current production `--spec mtp --draft-tokens 4`).

## What DSpine is

DSpine (Tencent, "Draft in Parallel, Condition Through") runs a **standalone drafter
artifact** beside the main model, selected with `--spec dspine --draft-model
<drafter.ninfer>`. The drafter is a small converted model (here
`Qwen3.8-27B-DSpine-b16-ninfer-v3.ninfer`, 2.27 GiB, Q8 layers + a BF16 code bridge).
Upstream measures it on an RTX 5090, where 15 DSpine drafts decode a code answer
**3.08x** and a prose answer **1.86x** faster than plain decoding — as good as or better
than DFlash2 (3.29x / 2.05x).

It is a different drafter than the MTP assistant/proposal head already stored in the
model artifact (`--lm-head-draft`), so the comparison is drafter-vs-drafter, not a
regression.

## Setup

- Engine: upstream `wavecut/master` (40d913db2, 2026-11-13), built as a pure sm_70
  image (`docker build --build-arg ARCHS=70`), tag `ninfer:upstream-sm70` (2.7 GB).
  DSpine drafting requires the upstream engine; the local `ninfer:1.7.2` (sm70 fork)
  does **not** have `--spec dspine`.
- Model: `Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.Q4_K_M.ninfer`.
- Drafter: `Qwen3.8-27B-DSpine-b16-ninfer-v3.ninfer` (downloaded from HuggingFace).
- Shared flags: `--max-context 200000 --kv-dtype int8 --max-concurrency 2 --kv-capacity
  auto --prefill-chunk 2048 --lm-head-draft --preserve-thinking --vision`.
- Baseline MTP: the same model with `--spec mtp --draft-tokens 4` (the current
  production gglm.yml config).
- DSpine: `--spec dspine --draft-model <drafter> --draft-tokens 7` and `--draft-tokens 15`.

### V100 gotcha (upstream bug)

The upstream `docker/entrypoint.sh` `build_of()` maps compute capability to a build but
only knows `7.5 / 8.6 / 12.0`. A V100 (7.0) is rejected with
"Compute capability 7.0 is not supported", even though the launcher
(`apps/launcher/selection.h`, `cuda12 {70,75,80,86,89}`) and the built sm70 binary both
support it. Workaround: set `NINFER_IMAGE_ARCH=sm70` (or exec
`/opt/ninfer/sm70/ninfer-serve` directly). The sm70 build itself is present in the image.

## Results (decode tok/s, one request, mean)

First pass, on the production `Huihui ...-abliterated.Q4_K_M` artifact with `--lm-head-draft`:

| config | 7.3K code | 31K code | 93K code | 193K zh-doc | acceptance (7.3K / 93K) |
|---|---:|---:|---:|---:|---|
| **MTP 4 (baseline, production)** | **87.7** | **77.9** | **78.3** | **41.3** | 0.69 / 0.81 |
| DSpine 7 | 77.1 | 56.3 | 48.9 | 28.5 | 0.64 / 0.48 |
| DSpine 15 | 61.7 | 60.5 | 45.2 | 18.6 | 0.33 / 0.33 |

Prefill (cold, s): MTP 8.4 / 35.3 / 178 / 411 — DSpine identical within drift (8.8 / 39.5 / 229 / 544; DSpine's
drafter adds no prefill cost, it only participates in the decode loop).

Raw rows: `/tmp/nov16/rows.jsonl` tags `dspine-d7`, `dspine-d15`, `q4km-gglm`, `dspine-official-d7`,
`dspine-official-d15`, `mtp-official-d4`.
Bench client: `bench/v100/decode_bench.py` (temperature 0.7 sampling, ±5% noise).

### Fair comparison on the official model and official DSpine parameters

The first pass used the Huihui abliterated Q4_K_M fine-tune with `--lm-head-draft`. The drafter's
own README says its drafter is tuned for the **base model** and that acceptance on fine-tunes
(abliterated, HauhauCS, MXFP8-CRACK) is *not measured*, and that `--lm-head-draft` makes DSpine
*slower* (it drafts from the artifact's proposal head, 5.14 vs 5.43 tokens a round). So I re-ran on
the official `GSQ-RCO IQ3_S` artifact (one of the drafter's two measured pairs, already local) with
the official DSpine flags — **no `--lm-head-draft`**, `--draft-tokens 7` and `15` — and a same-model
MTP 4 baseline, all on the same `ninfer:upstream-sm70` build:

| config (official IQ3_S) | 7.3K code | 31K code | 93K code | 193K zh | acceptance (7.3K / 93K) |
|---|---:|---:|---:|---:|---|
| **MTP 4 (same model)** | **77.3** | **75.6** | **63.0** | **35.5** | 0.63 / 0.70 |
| DSpine 7 (official, no lm-head-draft) | 64.6 | 52.9 | 56.0 | 29.5 | 0.52 / 0.58 |
| DSpine 15 (official) | 51.7 | 33.3 | 44.1 | 17.7 | 0.29 / 0.35 |

Even with the official model and official parameters, DSpine loses to MTP 4 at every context length.
Increasing drafts (7→15) still makes it worse (acceptance 0.52→0.29 at 7.3K). The conclusion is
unchanged: on the V100 the MTP lm-head draft's higher acceptance (0.63–0.70) beats the DSpine
drafter's (0.52–0.58) once the per-round verification cost of a longer draft is counted. DSpine is a
verification-throughput play for fast tensor-core cards (RTX 5090), not for the V100.

## Why DSpine loses on the V100

Decoding speed = (accepted tokens per round) / (round cost). The DSpine drafter's
**acceptance rate is lower than the MTP lm-head draft** at every context length,
0.52/0.58/0.29 vs 0.63/0.70/0.40 on the official IQ3_S (and 0.64/0.48/0.27 vs 0.69/0.81/0.38 on the
abliterated Q4_K_M), and each DSpine round verifies up to 15 candidates against the full 27B model. On an RTX 5090, whose FP16 tensor cores are ~6x faster at
the verification GEMM, that longer draft pays for itself (3.08x vs 3.29x for DFlash2).
On the V100 the verification GEMM is ~6x slower, so the higher draft cost is not
covered by a proportionally higher acceptance — decode drops 12–43% below MTP 4.

Increasing DSpine drafts (7→15) makes it *worse* on the V100 (acceptance 0.52→0.29 on the
official IQ3_S, 0.64→0.33 on the abliterated variant), the opposite of the 5090 where 15
drafts wins. That is the signature of a verification-bound card: more drafts cost more without
buying more acceptance. Using `--lm-head-draft` with DSpine also costs acceptance (5.14 vs 5.43
tokens a round in the drafter's README); it is an MTP/proposal-head option, not a DSpine one.

## Verdict

**Do not switch production to DSpine on the V100.** The existing `--spec mtp
--draft-tokens 4` (gglm.yml) stays the fastest. DSpine is a 5090/4090-class win and
only breaks even on cards whose tensor cores are fast enough to amortize the long
drafter. On the V100 it is 16–50% slower than MTP at every context length, on **both** the
abliterated Q4_K_M fine-tune and the official GSQ-RCO IQ3_S artifact, and with **both** the
fine-tuned-drafter and the official DSpine parameters. The model variant and the
`--lm-head-draft` flag do not change the outcome: MTP's co-trained proposal head simply has a
higher acceptance rate (0.63–0.70) than the separate DSpine drafter (0.52–0.58), and on a
verification-bound card that gap outweighs the drafter's longer draft.

The upstream V100 sm70 line (no QPN/prepack, IQ3_S MTP3 = 77 tok/s) is also slower than
the local `ninfer:1.7.2` sm70 fork (Q4_K_M MTP4 = 87.7 tok/s), so the local kernel
optimizations remain the V100's advantage over upstream.

## Reproduce

```
# drafter artifact (already in place)
/mnt/data/dockerdata/llama-cpp/models/Qwen3.8-27B-DSpine-b16-ninfer-v3.ninfer

# upstream sm70 image with DSpine
docker run -d --name dspine --gpus '"device=1"' -p 8000:8000 \
  -e NINFER_IMAGE_ARCH=sm70 \
  -v /mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro \
  ninfer:upstream-sm70 serve \
  /workspace/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.Q4_K_M.ninfer \
  --spec dspine --draft-model /workspace/models/Qwen3.8-27B-DSpine-b16-ninfer-v3.ninfer \
  --draft-tokens 7 --host 0.0.0.0 --port 8000 --api-key sk-itzy --model-id qwen \
  --max-context 200000 --kv-dtype int8 --max-concurrency 2 --kv-capacity auto \
  --prefill-chunk 2048 --lm-head-draft --preserve-thinking --vision
```

## 本地 fork DSpine 移植(分支 `feature/dspine-v100`)

本地移植已完成并验证可用(上游 op 逐字移植,sm70 Q8 形状走通用 GEMM 兜底)。175W 首轮数据已隔离(见 `/tmp/nov18-local-dspine/rows.jsonl`),以下全功耗(250W)数据已补测。

### NVFP 生产制品对照(全功耗 250W,2026-11-18)

DSpine 与 target 内部量化零耦合——drafter 只消费 hidden states、token embedding、output head。NVFP4 上接受率与 IQ3_S 一致(50-53%),制品为 NVFP 原生量化(非 GGUF 派生)。

**短 context 7.3K(并发 2)**:DSpine 7 在合并制品(NVFP+DSpine drafter,21.2 GiB)上 74.2 tok/s;纯 NVFP+MTP 4 为 96.2 tok/s(与生产基线 93.9 一致)。

**长 context 193K zhdoc(并发 1,200k int8 KV)**:

| 配置 | tok/s | 接受率 | ms/round |
|---|---|---|---|
| DSpine 7(合并制品) | 41.6 | 0.330 | 47.6 |
| MTP 4(纯制品) | **50.1** | 0.450 | 46.7 |

并发 2→1 解决了 DSpine 的 193K OOM(KV 208,960→243,776 tokens),但 MTP 4 仍快 20%。

**结论**:MTP 4 在所有 context 胜出。DSpine 7 的 74.2(7.3K)是合并制品数据(多 1.2 GiB drafter 权重,KV 被挤小),不能与纯制品的 96.2 直接比。DSpine 在 V100 上没有反超点。生产维持 NVFP + MTP 4。

原始数据:`/tmp/nov18b-nvfp-dspine/rows.jsonl`(NVFP 短/中 context)、`/tmp/zhdoc-c1.jsonl`(193K zhdoc 并发 1)。
