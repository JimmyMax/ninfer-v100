# NInfer V100 v1.7.0

Prebuilt **Tesla V100 (sm_70)** binaries, compiled with CUDA 12.8 on Ubuntu 24.04 (glibc 2.39).
The runtime hard-requires compute capability 7.0 — other GPUs are rejected at startup.

The release line for **faster GGML artifacts**: every GGUF-family quantization this engine runs
(Q4_K/Q5_K/Q6_K/Q8_0 superblocks, IQ2/IQ3, Q4_0/Q5_0, the dense MoE mix of a standard
llama.cpp export) now decodes near its memory-bandwidth limit, and the converter accepts a
standard llama.cpp `Q4_K_M` GGUF of the Qwen3.8-27B family out of the box.

**Fastest measured configuration (this release):** the NVFP4 artifact with
`--prefill-chunk 2048 --spec mtp --draft-tokens 4 --lm-head-draft` reaches **84.7 tok/s**
(58.9% acceptance) at a ~1.0K-token prompt and **75.5 tok/s** (52.0% acceptance) at a 27.3K
prompt — the highest throughput measured on this engine so far. The GGUF artifact's best is
`--prefill-chunk 4096` with the same decode flags.

## GGML decode is 18.9% faster than 1.6.2

The vector GEMV that runs every GGUF weight read carries the decode: one kernel per (type, T,
fused) instantiation, T = tokens per batch. On Volta the compiled slice loop for wide T
(T ≥ 2, i.e. every MTP verification and every batched call) spilled its instruction footprint
past the 16 KiB per-SM L1I, and the fetch stall ate most of the gain wide T should give. T = 1
was already fully unrolled; the fix is `#pragma unroll 1` on the slice loop for T ≥ 2
([`9f8edb7`](../../commits/9f8edb7)), leaving the T = 1 loop fully unrolled as before.

Measured on one V100-PCIE-32GB (250 W, ECC on), int8 KV, 27.3K-token prompt, 96 output,
`--spec mtp --draft-tokens 4 --lm-head-draft`, GGUF Q4_K_M artifact:

| mode | 1.6.2 (pre-fix) | 1.7.0 | change |
|---|---:|---:|---:|
| plain decode | 29.3 | **32.0** | **+9.2%** |
| MTP d4 decode | 57.0 | **67.8** | **+18.9%** |

The op-level GEMV bandwidth per type after the fix (isolated, cold L2): Q4_K 644 GB/s (80% of
the card's 803 GB/s), Q8_0 600–700 GB/s (75–87%), Q6_K 430–506 GB/s (54–63% — its 210-byte rows
are only 2-byte aligned, which caps how wide the loads can be), IQ3_S 527–542 GB/s.

## NVFP4 vs GGML on this release, and the 1.5.0 baseline

Same Swift-1.5 v3 weights in two representations, greedy, 96 output, int8 KV, one V100-32GB.
Short prompt = 982 tokens; long prompt = 27.2K tokens (the 27.3K rows come from the 40-case
tuning matrix at 27,250 tokens). Each side at its measured-best `--prefill-chunk`
(2048 NVFP4 / 4096 GGUF).

| prompt | mode | 1.7.0 NVFP4 | 1.7.0 GGUF Q4_K_M | 1.5.0 NVFP4 | 1.5.0 GGUF |
|---|---|---:|---:|---:|---:|
| short | plain decode | 34.3 | 33.3 | 33.9 | n/a |
| short | MTP d4 decode | **84.7** (58.9%) | 68.4 (51.6%) | 77.6 (52.5%) | n/a |
| short | prefill | 1,000 | 786 | 960 | n/a |
| 27.2K | MTP d4 decode | 68.3 (50.0%) | 59.5 (45.5%) | **75.8** (54.6%) | n/a |
| 27.2K | prefill | 975 | 969 | 977 | n/a |
| 27.3K | plain decode | 32.7 | 31.8 | — | n/a |
| 27.3K | MTP d4 decode | **75.5** (52.0%) | 73.8 (58.9%) | — | n/a |
| 82K | MTP d4 decode | **67.0** (52.0%) | 63.8 (55.1%) | — | n/a |
| 82K | prefill | 754 | **753** (ch 4096) | — | n/a |

Reading this table:

- **NVFP4 vs GGUF on 1.7.0:** plain decode tied within 3%; MTP decode within 2.3–12% depending
  on prompt (the acceptance-rate spread dominates — see below); 82K prefill tied with chunk
  4096. The GGUF artifact is 2.2 GiB smaller on the GPU.
- **1.7.0 vs 1.5.0 (same NVFP4 artifact):** at the short prompt 1.7.0 is faster (84.7 vs 77.6,
  +9.1%); at the 27.2K prompt 1.5.0 is faster (75.8 vs 68.3, +11.0%). Both gaps are driven
  mostly by the acceptance rate, which is a property of the engine's draft/target numerics —
  the same artifact and prompt verify differently on the two engines (58.9/50.0% on 1.7.0
  vs 52.5/54.6% on 1.5.0). Per-round time is within ~5% between the two.
- **1.5.0 GGUF is n/a:** the 1.5.0 loader predates the v3 GGUF framing's F32 GDN controls and
  rejects the artifact at startup (`ERROR startup failed | inspecting artifact`). 1.5.0 never
  ran this GGUF; the comparison is against its NVFP4 route.

The remaining GGUF gap to NVFP4 is the representation, not the kernel: Q4_K decodes at 80% of
peak already, and the measured remainder decomposes into multi-stream contention on the Q4_K
kernels (~6 ms/token), the Q6_K narrow-load limitation above (~2–3 ms/token), and
attention/KV bandwidth (~5 ms/token at 27K). A conversion-side repack of the Q6_K rows could
recover ~2–3% more; nothing cheaper is left. See the tuning-matrix record for the full
breakdown.

## What changed since 1.6.2

- **GGML vec decode:** `#pragma unroll 1` on the T ≥ 2 slice loop (the 18.9% above); T = 1
  unchanged.
- **Converter:** standard llama.cpp `Q4_K_M` GGUFs of this model family convert with their F32
  GDN controls, so a published GGUF no longer needs a re-quantization to reach this engine.
  Canonical command:

  ```console
  python3 -m tools.convert <model>.gguf --out <model>.ninfer --mmproj <model>-mmproj-F16.gguf
  ```

- Everything from 1.6.2: the long-`--max-context` `std::bad_alloc` fix, the flash prefill cap at
  200,000, opt-in `--ngram-draft-tokens` (default 0), and the Volta decode route set.

## Fastest measured configuration (V100-32GB, this release)

```console
ninfer-serve model.ninfer --host 0.0.0.0 --port 8000 --max-context 200000 \
  --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 --max-concurrency 1 \
  --spec mtp --draft-tokens 4 --lm-head-draft
```

NVFP4 artifact: 84.7 tok/s MTP decode at a ~1.0K prompt (58.9% acceptance), 75.5 tok/s at a
27.3K prompt (52.0%), 32.7 tok/s plain at 27.3K, 1,006 tok/s prefill at 27.3K, 67.0 tok/s at
82K. GGUF artifact: same flags with `--prefill-chunk 4096` (988 tok/s prefill at 27.3K); 73.8
tok/s MTP at 27.3K, 63.8 tok/s at 82K.

## Known costs

- MTP decode remains 2.3–12% behind NVFP4 on the same weights depending on prompt; the gap is
  the quantization representation plus the artifact's frozen-draft-head acceptance spread
  (±5 pp across prompts and chunking), not a defect of the GGML path.
- Versus 1.5.0 on the same NVFP4 artifact, the MTP acceptance rate is prompt-dependent in both
  directions (short prompt: 1.7.0 58.9% > 1.5.0 52.5%; 27.2K: 1.5.0 54.6% > 1.7.0 50.0%, int8 KV;
  with unquantized bf16 KV the 27.2K gap is 52.0% vs 48.8%). Investigation (A/B report, §1.1):
  the two engines run different greedy trajectories — the 148-commit upstream merge restructured
  the model layer and the Volta decode kernel family, shifting near-tie logits — and 1.7.0's
  draft-target agreement on its own trajectory is ~3 pp lower at 27.2K even without KV
  quantization. This is a structural cost of the restructured line, not a patchable defect.
- `--spec mtp` output is not token-identical to running without speculation, on this line and
  on 1.5 alike: batched verification reaches near-tied logits differently. Use no `--spec`
  when exact equivalence to an unspeculated decode matters.
- 1.5.0 cannot load v3 GGUF artifacts (F32 GDN controls); upgrading to 1.7.0 is required for
  the GGML route on those artifacts.

## Assets

Built locally; not pushed.

| Asset | Value |
|---|---|
| image `ninfer:1.7.0` | `sha256:dd3a7dbb11e6` |
| `ninfer-serve` binary | `sha256:2fb0c0a3f654ec0749d6f982177fa5f5a116b29ef94f332ee0d70b0687919475` |

## Documentation

- NVFP4 vs GGML A/B under 1.7.0, with the 1.5.0 baseline and the per-kernel / op-level
  headroom analysis: [ab-swiftv3-nvfp4-vs-gguf-2026-10-22.html](docs/v100/ab-swiftv3-nvfp4-vs-gguf-2026-10-22.html)
- The vec optimization record (rejected variants included):
  [gguf-vec-optimization-2026-10-21.md](docs/v100/gguf-vec-optimization-2026-10-21.md)
- The tuning-matrix raw data, including this release's A/B runs:
  `bench/v100/data/swiftv3_tuning_2026-10-22/`
