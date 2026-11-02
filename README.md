# NInfer on the Tesla V100-32GB

A C++/CUDA inference engine tuned for a **single Tesla V100-PCIE-32GB (compute capability 7.0,
`sm_70`)**. It runs the Qwen3.8-27B family (dense and MoE), Qwen3.8-Flash-Next, and Gemma 4, on
weights from native NInfer profiles (NVFP4, FP8, groupwise-int) or GGUF block formats converted from
a `.gguf`. The optimizations are operator routing and weight prepacking, applied at load time or on
first use: no route changes a numeric contract, and the numbers below are that card, not another.

[Quick start](#quick-start) · [Artifacts and formats](#artifacts-and-formats) ·
[Measured performance](#measured-performance) · [Technical highlights](#technical-highlights) ·
[Flash-Next](#flash-next) · [Gemma 4](#gemma-4) · [Docker](#docker) · [Building](#building) ·
[Documentation](docs/README.md)

The `sm_89` (RTX 4080 SUPER) build is carried in the same image and serves the same artifacts; this
document is the V100 line. Every measurement is one request on a V100-PCIE-32GB unless stated.

## Quick start

```bash
# Host GPU 1 = V100-32GB. The image carries both the sm_70 and sm_89 builds and starts the one
# that matches the GPU.
docker run -d --gpus device=1 --ulimit memlock=-1 \
  -v /path/to/models:/workspace/models -p 8000:8000 \
  --entrypoint ninfer-serve ninfer:1.7.2 \
  /workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3.ninfer \
  --host 0.0.0.0 --port 8000 --model-id qwen \
  --max-context 200000 --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0

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
| `orcarouter-Qwen3.8-27B-Uncensored-nvfp4-NInfer.ninfer` | 20.0 | native `nvfp4` | orca's uncensored variant |
| `Swift-1.5-Qwen3.8-27B-Q4_K_M-ninfer-v3.ninfer` | 17.0 | GGUF `q4_k` | the same 27B base as orca, converted from the GGUF |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | 14.0 | GGUF blocks, 3.5 bits | 18 block formats, IQ3_S holding 3.39 GiB |
| `Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.Q4_K_M.ninfer` | 16.5 | GGUF `q4_k` | abliterated dense |
| `Gemma4-31B-...-MTP-ninfer-v3.ninfer` | 18.7 | GGUF (dense) | 60 layers, hidden 5376, no experts |
| `Gemma4-26B-A4B-...-MTP-ninfer-v3.ninfer` | 17.0 | GGUF Q4_0 | 128 experts, 8 routed per token |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | 54.0 | GGUF blocks, 3.5 bits | MoE, host/disk experts, n-gram table |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer` | 37.9 | GGUF blocks, 2.4 bits | MoE; reads the shared n-gram table |

The container version does not affect speed — a v2 file is read in place and measured within 1% of
the upgraded v3 (998 vs 997 tok/s of prefill on the native profile). What changes the numbers is
the weight recipe (native vs GGUF), not the container. The three Flash-Next releases read one shared
n-gram table the model names by digest, passed with `--ngram-table` (IQ4_NL rows, 26.92 GiB).

## Measured performance

`ninfer_bench`, the 1.7.2 sm_70 binary, int8 KV, prefill chunk 2048, greedy, three measured
repetitions after one warmup, on the V100-32GB. Two prompts: a short 1,024-token one and a long
27,000-token one (repeated 2,048-token corpus). Decode is 128 generated tokens from a one-token
seed, so it is prompt-length-independent here; TTFT is the short prompt's.

| Artifact | File, GiB | Prompt | TTFT | Prefill | Decode, MTP 4 drafts | Decode, no drafter |
|---|---|---:|---:|---:|---:|---:|
| orcarouter `nvfp4` v3 | 21.2 | 1,024 | 0.95 s | 1,079 tok/s | **61.6 tok/s** (34.4% accepted) | 34.4 tok/s |
| orcarouter `nvfp4` v3 | 21.2 | 27,000 | 27.6 s | 980 tok/s | 61.6 tok/s (34.4%) | |
| Swift-1.5 Q4_K_M | 17.0 | 1,024 | 1.13 s | 907 tok/s | 51.9 tok/s (28.7%) | 38.1 tok/s |
| Swift-1.5 Q4_K_M | 17.0 | 27,000 | 30.3 s | 891 tok/s | 52.0 tok/s (28.7%) | |
| GSQ-RCO IQ3_S | 14.0 | 1,024 | 1.13 s | 906 tok/s | 51.5 tok/s (27.7%) | 43.0 tok/s |
| GSQ-RCO IQ3_S | 14.0 | 27,000 | 30.3 s | 890 tok/s | 51.5 tok/s (27.7%) | |

The native profile prefills ~16% faster than the GGUF block formats (the mixed-precision
fragmentation above), and decodes faster **with** the drafter because the draft's acceptance
favors it (34.4% versus 27.7–28.7%). Without a drafter the order inverts: the 14.0-GiB IQ3_S
artifact reads 43.0 tok/s against the native's 34.4 (+25%), Q4_K_M 38.1. The drafter's acceptance
is generation-dependent — a different 128-token generation gives a different rate — and drifts a
few points across engine revisions (the frozen draft head reads a hidden state that changes with
the engine), so these decode figures are the honest rate of this sitting, not a best case.
Both Gemma 4 artifacts load and run (below).

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

## Docker

`ninfer:1.7.2` carries both builds in one image: the `sm_70` binary for the V100 and the `sm_89`
binary for the RTX 4080 SUPER, with a dispatcher that starts the one matching the GPU. The image
has no `ENTRYPOINT`, so `command[0]` is the executable — use `--entrypoint ninfer-serve` to serve.
`--ulimit memlock=-1` is required for the pinned host-expert buffers. Build it from
[`Dockerfile.1.7.2.local`](Dockerfile.1.7.2.local) after copying the two binaries into
`docker-runtime/1.7.2/`. The `sm_70` build needs CUDA 12.8 with `-DCMAKE_CUDA_ARCHITECTURES=70`;
the `sm_89` build is a full build of the same tree.

## Building

```bash
# sm_70 (V100): CUDA 12.8, full build.
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# The apps are build/apps/ninfer and build/apps/ninfer-serve; the test bundle is
# build/tests/ninfer_tests (a dispatcher over the per-op and contract tests).
```

Volta does not support `--expt-relaxed-constexpr`; the tree guards every Volta-specific kernel
route behind `NINFER_TIER_VOLTA`. The V100's kernel routes come from its measured device profile,
calibrated on the card — the upstream tuning constants are not authoritative here.

## License

See [LICENSE](LICENSE). This line continues the NInfer work; [CONTRIBUTORS.md](CONTRIBUTORS.md)
credits the upstream and fork authors.
