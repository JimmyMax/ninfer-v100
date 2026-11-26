# NInfer on the Tesla V100-32GB

A C++/CUDA inference engine tuned for a **single Tesla V100-PCIE-32GB (compute capability 7.0,
`sm_70`)**. It runs the Qwen3.8-27B family (dense and MoE), Qwen3.8-Flash-Next, and Gemma 4, on
weights from native NInfer profiles (NVFP4, FP8, groupwise-int) or GGUF block formats converted from
a `.gguf`. The optimizations are operator routing and weight prepacking, applied at load time or on
first use: no route changes a numeric contract, and the numbers below are that card, not another.

[Quick start](#quick-start) · [Artifacts and formats](#artifacts-and-formats) ·
[Measured performance](#measured-performance) · [Technical highlights](#technical-highlights) ·
[Flash-Next](#flash-next) · [Gemma 4](#gemma-4) · [Inkling 2 Mini](#inkling-2-mini) · [Docker](#docker) · [Building](#building) ·
[Documentation](docs/README.md)

The `sm_89` (RTX 4080 SUPER) build is carried in the same image and serves the same artifacts; this
document is the V100 line. Every measurement is one request on a V100-PCIE-32GB unless stated.

This is the `inkling2` line: it carries the upstream perf port (twelve perf commits, merged
`a738ddd6c`) with the Volta gate that keeps the shared/vector-jobs launches and the dense spread
form off `sm_70` (`22908d4cd`). The 2026-11-25/26 full-artifact sweep on both cards found no
regression; current per-artifact numbers are in
[docs/v100/regression-sweep-2026-11-25.md](docs/v100/regression-sweep-2026-11-25.md).

## Quick start

```bash
# Host GPU 1 = V100-32GB.
docker run -d --gpus device=1 --ulimit memlock=-1 \
  -v /path/to/models:/workspace/models -p 8000:8000 \
  ninfer:1.7.2 \
  ninfer-serve /workspace/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.Q4_K_M.ninfer \
  --host 0.0.0.0 --port 8000 --model-id qwen \
  --max-context 200000 --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 2 --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}]}'
```

`--api-key` is omitted above so the server answers unauthenticated, which is what you want while
debugging; add `--api-key sk-...` and send `-H 'Authorization: Bearer sk-...'` to require one.

The server speaks the OpenAI and Anthropic APIs on `:8000/v1` (`/v1/messages` accepts
`"thinking": {"type": "disabled"}`). `GET /health` is 503 while loading, 200 when ready. Build with
[`Dockerfile.1.7.2.local`](Dockerfile.1.7.2.local); the production compose files live in the Docker
repo.

## Artifacts and formats

Two families load, in place, with no offline conversion:

* **Native NInfer profiles** — quantized by the converter from HF safetensors, compressed-tensors
  or block-FP8 sources: `nvfp4` (block scale), `fp8_e4m3fn_row_bf16`, `q4_g64_fp16` / `q5_g64_fp16`
  / `q6_g64_fp16` / `q8_g32_fp16` (groupwise, FP16 scale), plus `bf16`, `fp32`, `int32`. These take
  the fast routes: QPN prepacking for `t <= 32`, CUTLASS wide-T over a dequantized staging plane,
  and a row-scale FP8 dequant kernel.
* **GGUF-derived artifacts** — converted from a `.gguf`: 19 registered block formats
  (`gguf_q4_k`, `gguf_q8_0`, `gguf_iq3_s`, `gguf_iq4_xs`, `gguf_iq2_s`, …) in the `gguf_blocks_v1`
  layout, run through the ggml bridge (per-tensor block dequant plus the matching MMQ/Vec kernels).
  That is the honest price of a 2–4 bit block format: a mixed-precision artifact fragments prefill
  into many small dequantize-and-GEMM kernels and lands ~12% below the native profile.

| Artifact | File, GiB | Format | Notes |
|---|---:|---|---|
| `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3.ninfer` | 21.2 | native `nvfp4` | production default; 19.1 GiB on device; DFlash2 adapter + MTP head |
| `Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer` | 22.1 | native `nvfp4`/`fp8`/groupwise | Coder390 family; DFlash2 + MTP head |
| `Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer` | 21.2 | native `nvfp4`/`fp8`/groupwise | Coder390 W8A8 activation variant; current production artifact |
| `orcarouter-Qwen3.8-27B-Uncensored-nvfp4-NInfer.ninfer` | 20.0 | native `nvfp4` | orca's uncensored variant |
| `Swift-1.5-Qwen3.8-27B-Q4_K_M-ninfer-v3.ninfer` | 17.0 | GGUF `q4_k` | the same 27B base as orca, converted from the GGUF |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | 14.0 | GGUF blocks, 3.5 bits | 18 block formats, IQ3_S holding 3.39 GiB |
| `Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.Q4_K_M.ninfer` | 16.5 | GGUF `q4_k` | abliterated dense |
| `Gemma4-31B-...-MTP-ninfer-v3.ninfer` | 18.7 | GGUF (dense) | 60 layers, hidden 5376, no experts |
| `Gemma4-26B-A4B-...-MTP-ninfer-v3.ninfer` | 17.0 | GGUF Q4_0 | 128 experts, 8 routed per token |
| `Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0.ninfer` | 6.9 | GGUF PTQ1_0 (ternary) | linear-attention (GDN) layers, no experts |
| `Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.ninfer` | 7.1 | PTQ1_0 + MTP (Q4_0) | the same base plus the official MTP head; 82.6 tok/s decode, MTP 4 drafts |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | 54.0 | GGUF blocks, 3.5 bits | MoE, host/disk experts, n-gram table |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer` | 37.9 | GGUF blocks, 2.4 bits | MoE; reads the shared n-gram table |

The container version does not affect speed — a v2 file is read in place and measured within 1% of
the upgraded v3 (998 vs 997 tok/s of prefill on the native profile). What changes the numbers is
the weight recipe (native vs GGUF), not the container. The three Flash-Next releases read one shared
n-gram table the model names by digest, passed with `--ngram-table` (IQ4_NL rows, 26.92 GiB).

## Measured performance

`ninfer_bench`, the 1.7.2 sm_70 binary, int8 KV, prefill chunk 2048, greedy, three measured
repetitions after one warmup, on the V100-32GB. Two prompts: a short 1,024-token one and a long
27,000-token one (repeated 2,048-token corpus). Prefill is the whole prompt; decode is 512
generated tokens from a one-token seed, so it is prompt-length-independent here.

### Current line (inkling2, 2026-11-25/26)

Decode tok/s from the full-artifact sweep on the current `inkling2` binary (the upstream perf port
plus the Volta gate), 8,192 context, int8 KV, one request per artifact —
[docs/v100/regression-sweep-2026-11-25.md](docs/v100/regression-sweep-2026-11-25.md):

| Artifact | Decode, tok/s | Drafter |
|---|---:|---|
| Gemma 4 31B (dense) | 47.8 | MTP |
| Gemma 4 26B-A4B | 161.9 | MTP |
| GSQ-RCO IQ3_S | 42.4 | MTP |
| Huihui Q4_K_M | 54.3 | MTP |
| Inkling 2 Mini | 70.0 | DSpark |
| Coder390 W4A4 DFlash2 | 65.0 | MTP |
| swift NVFP4 DFlash2 | 67.1 | MTP |
| Ternary-Bonsai PTQ1_0 | 69.8 | none |
| Ternary-Bonsai PTQ1_0 + MTP | **82.6** | MTP 4 (29% accepted) |
| Flash-Next IQ3_S (host experts) | 33.6 | none |

All nine artifacts start, probe and decode at or above their pre-merge rates. The sweep is the
regression gate for this line — mandatory before any engine change is finalized, all artifact
types, both cards (method and current numbers in
[docs/v100/regression-sweep-2026-11-25.md](docs/v100/regression-sweep-2026-11-25.md)).

### 1.7.2 baseline (`ninfer_bench`)

| Artifact | File, GiB | Prompt | TTFT | Prefill | Decode, MTP 4 drafts | Decode, no drafter |
|---|---|---:|---:|---:|---:|---:|
| orcarouter `nvfp4` v3 | 21.2 | 1,024 | 0.95 s | 1,079 tok/s | **78.1 tok/s** (50.3% accepted) | 34.4 tok/s |
| orcarouter `nvfp4` v3 | 21.2 | 27,000 | 27.6 s | 980 tok/s | 78.1 tok/s (50.3%) | |
| Swift-1.5 Q4_K_M | 17.0 | 1,024 | 1.13 s | 907 tok/s | 64.8 tok/s (41.9%) | 38.1 tok/s |
| Swift-1.5 Q4_K_M | 17.0 | 27,000 | 30.3 s | 891 tok/s | 64.8 tok/s (41.9%) | |
| GSQ-RCO IQ3_S | 14.0 | 1,024 | 1.13 s | 906 tok/s | 58.7 tok/s (34.9%) | 43.0 tok/s |
| GSQ-RCO IQ3_S | 14.0 | 27,000 | 30.3 s | 890 tok/s | 58.7 tok/s (34.9%) | |

**Fastest decode the card sustains** — a 7,051-token code-prompt continuation, the
highest-acceptance workload (`ninfer_bench -pg 7051,512`), same settings otherwise:

| Artifact | MTP 4 decode | No drafter | Speedup | Acceptance | Prefill 7,051 |
|---|---:|---:|---:|---:|---:|
| orcarouter `nvfp4` v3 | **108.4 tok/s** | 33.2 tok/s | 3.26× | 84.9% | **1,089 tok/s** |
| Swift-1.5 Q4_K_M | 104.8 tok/s | 36.8 tok/s | 2.85× | 87.9% | 980 tok/s |
| GSQ-RCO IQ3_S | 104.5 tok/s | 41.2 tok/s | 2.54× | 86.7% | 980 tok/s |

At 85–88% acceptance the three artifacts are within 3.4–3.7%; the spread is decided by per-round
format speed (NVFP4's QPN route reads 4-bit blocks directly, the mixed GGUF blocks pay a
dequantize pass). So the table above's 25% drafter gap is a one-token-seed trajectory, not the
artifacts' genuine edge. Full record in [docs/performance.md](docs/performance.md).

### Coder390 serving matrix (ninfer-serve, temp 0.7)

A separate bench through `ninfer-serve` on the Coder390 production line (V100 at 250W, int8 KV,
chunk 2048, lm-head-draft, concurrency 2, one seed per point; harness lives in the deployment
repo, `bench/run_mtp_dflash2_matrix.sh`). Decode tok/s, MTP 4 drafts unless noted:

| Arm | 7.3K code | 28K code | 93K code | 193K zh |
|---|---:|---:|---:|---:|
| NVFP (W8A8) MTP 4 | 96.2 | 96.5 | 77.9 | 44.9 |
| NVFP MTP 7 | 102.3 | 102.9 | 78.5 | 32.3 |
| NVFP DFlash2 4 | 79.4 | 78.7 | 69.4 | OOM |
| IQ3_S MTP 4 | 92.3 | 89.2 | 81.6 | 48.9 |
| Q4_K_M (Huihui) MTP 4 | 97.1 | 80.5 | 82.6 | 44.0 |

Three conclusions this matrix settled (full data `/tmp/mtpdf2-matrix/rows.jsonl` at measurement
time; rerun the harness before each release):

* **MTP draft 4 is the right width on every artifact line.** Wider drafts win short context
  (draft 7: 102–103 tok/s at ≤28K on NVFP) but collapse at 193K (32.3, acceptance 0.23); draft 3
  wins only at 193K. Draft 4 is the only width that holds all four context points.
* **DFlash2 loses to MTP everywhere on the V100** (11–30% behind at equal acceptance; also +2.7 GiB
  runtime that caps NVFP at 185K context). DFlash2 is the right drafter on Ada (4080: 66.9 vs 34.9
  tok/s nospec) where its selector/conv ops ride tensor cores — not on sm_70.
* **GGUF Q4_K_M/IQ3_S match NVFP in decode** (±6%, each winning somewhere) and beat it beyond
  93K (Q4_K_M 82.6 vs 77.9 at 93K; IQ3_S 48.9 vs 44.9 at 193K) on smaller weight footprints.
  Prefill is 4–8% slower on GGUF blocks (the per-tensor dequantize-and-GEMM fragmentation above).

The native profile prefills ~16% faster than the GGUF block formats (the mixed-precision
fragmentation above), and decodes faster **with** the drafter because the draft's acceptance
favors it (50.3% versus 34.9–41.9%). Without a drafter the order inverts: the 14.0-GiB IQ3_S
artifact reads 43.0 tok/s against the native's 34.4 (+25%), Q4_K_M 38.1. The drafter's acceptance
is generation-dependent — the same engine at a 128-token generation takes 34.4% and 61.6 tok/s on
the native profile — and drifts a few points across engine revisions (the frozen draft head reads
a hidden state that changes with the engine), so these figures are the honest rate of this sitting
at a 512-token generation, not a best case. Both Gemma 4 artifacts load and run (below).

Where the drafter gap comes from: a verify round costs the same on all three (12.9 / 13.9 / 13.6 ms,
only 44–52% above its no-drafter t=1 step), so the whole difference is the acceptance chain,
per draft position: NVFP4 78.2 → 52.9 → 39.4 → 30.6%, Q4_K_M 71.7 → 47.6 → 30.9 → 17.3%, IQ3_S
66.8 → 38.8 → 21.5 → 12.1%. The MTP draft module is frozen — trained to predict the
**unquantized** model's next tokens — and it is quantized to the artifact's own format, so its
noise compounds over the draft's four autoregressive steps; the 4.5-bpw native profile keeps both
the base model and the draft module closest to the reference, and its chain decays least. At equal
chains all three would decode ~73–78 tok/s. That chain is a property of the *generation*, not the
artifact: the two artifacts branch at their first divergent token, so each scores its own trajectory
here — which is why the best-condition continuation recorded above narrows the gap to a few percent.

## Technical highlights

The gap to a modern card is closed in operator routing and weight prepacking, measured per change:

| Change | Measured effect |
|---|---|
| **Volta QPN** — quadpair-split-N prepacked GEMMs for `nvfp4` and FP8 (`t <= 32` QPN, wide-T CUTLASS) | projection routes 85 → 480–620 GB/s |
| **GDN prefill on the recurrent kernel** (chunked and two-stage regress on `sm_70`) | GDN T=1024…8192 **2.9–3.3×**; 3.75 → 2.18 s of a 27K prefill |
| **Flash-attention prefill kernel** (paged-to-contiguous staging plus an `sm_70` `mma.sync` kernel) | same shape 41.1 → 25.1 ms; 27K TTFT 30.1 → 27.2 s |
| **Speculative-decode configuration** — `--ngram-draft-tokens 0`, no `--adaptive-mtp` (15 n-gram drafts dilute the MTP acceptance) | 27K decode 64.2 → 78.3 tok/s, greedy output byte-identical |
| **FP8 SwiGLU wide T on an FP32-output CUTLASS path** (previously a four-token-chunk SIMT ladder) | mixed-quantization prefill 508 → 998 tok/s |
| **v2 artifacts read in place** (the directory is normalized to the v3 shape in memory, weights read where they are) | same numbers as the upgraded copy, no offline conversion |
| **GGUF decode projection** (S2 sliding-window attention geometry, S3 `constexpr-K` vector kernels, Q6_K dense prepack) | dense decode +8.7%; 26B MoE covered |
| **Upstream perf port, Volta-gated** (merge `a738ddd6c`, gate `22908d4cd`) — the shared/vector-jobs launches and dense spread form are `sm_80`-only on this line; the MoE rows kernel runs on every tier | no regression: 9/9 artifacts at or above pre-merge rates on the 2026-11-25 sweep |

Every change passes the upstream FP64 oracles (`ctest`) and a 27K-token end-to-end A/B. Cross-line
comparisons use `bench/v100/compare_lines.py`; designs and per-stage measurements live in
[`docs/v100/`](docs/v100/) (roadmap [PLAN.md](docs/v100/PLAN.md)).

## Flash-Next

Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) is a 48-layer MoE that routes 512 experts and reads the
shared n-gram table. Its routed experts do not fit the 32 GB card in device memory (IQ3_S needs
53.7 GiB), so they live in **pinned host memory** (`--expert-residency host`) and the engine stages
only the experts a token routes into a bounded device cache beside an optional persistent one. The
router and shared experts stay resident; all arithmetic is on the GPU.

The decode rate is set by the miss traffic — the experts a round routes that the cache lacks — not
by compute. IQ3_S's routed experts are 53.7 GiB, so on a 32 GB card they cross PCIe 3.0 (~13 GB/s
effective) instead of being read at ~875 GB/s from HBM. That is the structure's floor, not the
card's ceiling.

Tesla V100-PCIE-32GB, the IQ3_S release, host experts, every miss on the GPU, MTP with three
drafts, INT8 KV, an 8,192-position context, and a 128-token answer over a 7,051-token code corpus,
two repetitions; the expert cache holds 24,118 MiB (7,372 of 24,576 experts, 30.0%):

| Drafter | Decode, tok/s | Drafts accepted |
|---|---:|---:|
| none | 33.6 | — |
| MTP 3 | **43.5–44.0** | 62.5% |

Prefill is staging-bound too, and it scales with the prompt, because every prompt pays a nearly
fixed ~20–25 GB of routed-expert staging over PCIe 3.0:

| Prompt, tokens | 512 | 2,048 | 4,096 |
|---|---:|---:|---:|
| Prefill, tok/s | 158 | 403 | 695 |
| Staging per prompt | ~20 GB | 21 GB | 25 GB |

A 512-token prompt therefore looks slow (158 tok/s) only because that fixed staging is spread over
few tokens. The live 200,000-token service fits **TTFT ≈ 3.3 s + 0.47 ms per prompt token**
(marginal ≈ 2,100 tok/s), measured at 19 / 524 / 1,280 / 2,330 / 4,220 / 8,820 / 30,820-token
prompts: 23.9 / 146.0 / 350.3 / 606.4 / 824.8 / 905.8 / 951.1 tok/s. `--expert-prefill-buffers 2`
double-buffers the long-prompt span path (+18–27% at 4,096). The same A/B that set the 200,000-token
service's KV to `rk4v4` also set its decode, unseeded: **36.1 tok/s without a drafter, 45.2 with
MTP 3** (62.5% accepted), a little above the 8,192-context bench numbers (33.6 / 43.5–44.0) because
that bench gives the expert cache 24,118 MiB against the service's 18,753. The production service
additionally seeds the cache from a route histogram (`--expert-profile`), a further +25% on a cold
start ([docs/performance.md](docs/performance.md), "IQ3_S on one V100: the final configuration"). The [Q2_0 release](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3)
(37.9 GiB) halves the expert traffic — its experts are 1.32 MiB against IQ3_S's 3.27 MiB — and
reaches **69.6 tok/s** decode with MTP 3 (70.7% accepted) at a perplexity cost (2.6509 vs 2.1317).
The levers measured on this card (CPU share, the two-stage staging, prompt-pollution fix, profile
seeding) and why 70+ at IQ3_S quality needs Q2_0 or a second GPU are in
[docs/performance.md](docs/performance.md) (the "IQ3_S on one V100: the final configuration" section).

## Gemma 4

Both Gemma 4 artifacts are GGUF-derived and run through the ggml bridge, but their shapes differ:
the 26B-A4B routes 8 of 128 experts per token, while the 31B is **dense** — 60 layers at hidden
5376, no experts. One request each, a 5,513-token prompt over int8 KV, 500 output tokens
([Gemma 4 on the V100-32GB](docs/performance/gemma4-v100-32gb-2026-10.md)):

| Artifact | Weights | Prefill | Decode |
|---|---:|---:|---|
| Gemma 4 31B (dense, hidden 5376) | 17.6 GiB | 715 tok/s | **54.1** tok/s (MTP 4 drafts, 54.5%) or 27.4 without speculation |
| Gemma 4 26B-A4B (128 experts, 8 per token) | 16.9 GiB | 2.39k tok/s | **157.2** tok/s (MTP 2 drafts, 71.7%) or 120.3 without speculation |

The dense 31B reads all 17.6 GiB per token, so unspeculated decode is bandwidth: 27.4 tok/s is
about 517 GB/s of the ~790 GB/s this card's HBM2 moves with ECC on. Speculation lifts it — four
drafts verified per weight read take the same traffic to 54.1 tok/s. The A4B's file is only 4%
smaller, but a token touches 8 of 128 experts, so it decodes 4.4× faster and gains less from
drafting.

## Inkling 2 Mini

The [Inkling 2 Mini](docs/inkling2.md) looped MoE (26.4B params, 3.4B active) runs from any
Inkling GGUF release converted to `.ninfer` by `tools/convert/inkling2.py`, with vision and the
DSpark drafter. On the V100-32GB it loads UD-Q4_K_XL (18.7 GiB) with `gguf_q8_0` dense layers
and `gguf_nvfp4` routed experts, or any other quantization (IQ4_XS, Q2_K_XL, Q3_K_XL, Q8_K_XL).
The engine also loads upstream NInfer-v3 artifacts directly ([WaveCut's
pack](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3)). Measured on GPU 1,
int8 KV, `--expert-residency device`, 600K-token context:

| Decode | Prefill | Context | Notes |
|---:|---:|---:|---|
| **~70-94 tok/s** (DSpark 3 drafts, 42% accepted) | **~1000-1300 tok/s** | 600K tokens | 24 layers, 48 visits; NVFP4 experts software-emulated on sm_70 |

DSpark is the artifact's drafter (3 layers over a block of 8); it starts automatically with the
artifact. `--spec none` disables it for pure decode (~66 tok/s). The NVFP4 experts are the decode
bottleneck on this card — V100 has no FP4 tensor core, so they run through a sm_70-specific
software-emulation path. On cards with FP4 hardware (H100/B200) the same artifact runs much
faster.

```bash
# From a GGUF release
python -m tools.convert --model models/inkling2/hf --recipe inkling2_gguf \
  --source gguf=models/inkling2/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --source vision=models/inkling2/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --source dspark=models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf \
  --components text,vision,dspark --out models/inkling2/Inkling-2-Mini-UD-Q4_K_XL.ninfer \
  --device cpu

# Serve
ninfer-serve models/inkling2/Inkling-2-Mini-UD-Q4_K_XL.ninfer --model-id inkling2 \
  --max-context 600000 --kv-dtype int8 --expert-residency device --vision
```

## Docker

Two images carry the current line:

- **V100 (`sm_70`)** — `ninfer:1.7.2`, whose `ninfer-serve` is this branch's sm_70 build. The
  image has no `ENTRYPOINT`, so `command[0]` is the executable.
- **4080 SUPER (`sm_89`)** — `ninfer:1.8.0-inkling2-sm89`, which serves
  `/opt/ninfer/ninfer-serve.sm89`, this branch's sm_86;89 build.

A freshly built binary is mounted in and run under the matching image (the regression-test
containers do exactly this):

```bash
docker run -d --gpus device=1 -e NVIDIA_VISIBLE_DEVICES=1 \
  -v /path/to/models:/models -v /tmp/ninfer-serve-70:/opt/ninfer/ninfer-serve.sm70:ro \
  -p 8080:8080 \
  ninfer:1.8.0-inkling2 /opt/ninfer/ninfer-serve.sm70 /models/model.ninfer \
  --host 0.0.0.0 --port 8080
```

`--ulimit memlock=-1` is required for the pinned host-expert buffers (Flash-Next host experts).

## Building

```bash
# sm_70 (V100): CUDA 12.8, full build.
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release -DNINFER_TIER_VOLTA=ON
cmake --build build -j

# sm_89 (4080 SUPER): same tree, Ampere tier.
cmake -B build89 -DCMAKE_CUDA_ARCHITECTURES=86;89 -DCMAKE_BUILD_TYPE=Release -DNINFER_TIER_AMPERE=ON
cmake --build build89 -j

# The apps are build/apps/ninfer and build/apps/ninfer-serve; the test bundle is
# build/tests/ninfer_tests (a dispatcher over the per-op and contract tests).
```

Volta does not support `--expt-relaxed-constexpr`; the tree guards every Volta-specific kernel
route behind `NINFER_TIER_VOLTA`. The V100's kernel routes come from its measured device profile,
calibrated on the card — the upstream tuning constants are not authoritative here.

## License

See [LICENSE](LICENSE). This line continues the NInfer work; [CONTRIBUTORS.md](CONTRIBUTORS.md)
credits the upstream and fork authors.
