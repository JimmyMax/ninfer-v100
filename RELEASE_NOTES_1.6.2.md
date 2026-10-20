# NInfer V100 v1.6.2

Prebuilt **Tesla V100 (sm_70)** binaries, compiled with CUDA 12.8 on Ubuntu 24.04 (glibc 2.39).
The runtime hard-requires compute capability 7.0 — other GPUs are rejected at startup.

A performance release for the report in [#2](https://github.com/JimmyMax/ninfer-v100/issues/2): the
1.6 line is now faster than 1.5 everywhere the reporter measured the loss. Nine commits merge
upstream's 148 post-1.6.1 commits and add the V100 route work; measured on one V100-PCIE-32GB
(250 W, ECC on), int8 KV, `--prefill-chunk 2048`, alternating A/B against `1.6.1`.

## Speculative decode is faster than 1.5 at every depth

`--spec mtp --draft-tokens 4 --lm-head-draft`, Swift v2 NVFP4 artifact:

| prompt | 1.5 | 1.6.2 | change | acceptance 1.5 → 1.6.2 |
|---|---:|---:|---:|---|
| 502 | 65.7 | **70.7** | **+7.6%** | 40.5% → 43.2% |
| 2,942 | 62.8 | **65.4** | **+4.1%** | 39.2% → 40.1% |
| 27,222 | 53.3 | **59.2** | **+11.1%** | 32.7% → 36.9% |

A second, native-nvfp4 artifact (`orcarouter-Qwen3.8-27B-Uncensored-nvfp4`) gains +2.8%, +2.4% and
+4.7% at the same depths, so the win is not specific to a GGUF-derived artifact. Plain decode is at
parity, and prefill is at parity at 27K and 1.0% behind at 66K.

## What changed

- **Copy drafting is opt-in.** `--ngram-draft-tokens` defaults to 0 and `speculative.*` counts only
  neural rows; copy rounds are reported through `ngram_*`. The reported acceptance is now comparable
  across configurations, which is what made the [#2](https://github.com/JimmyMax/ninfer-v100/issues/2)
  numbers look like a 1.5 → 1.6 regression. This is the behaviour change in the release: a service
  that relied on the old default loses the 15-token copy drafting it never named.
- **Upstream merge.** 148 commits, including the verify pass that partitions KV by the live draft
  window and the sm_70 FP16 saturation fix. `NINFER_SM70` became `NINFER_TIER_VOLTA` (the
  `NINFER_SM70_*` environment variables keep their names).
- **Volta decode routes.** FP8 SwiGLU `gate_up` is prepacked at load (294 → 111 µs a call), FP8
  residual QPN activations stage through FP16 (62.9 → 47.4 µs), NVFP4 attention and GDN inputs join
  the QPN path, and narrow Q8/Q4 calls use their own kernels (Q8 282 → 150 µs, Q4 1072 → 567 µs).
- **Long-context prefill.** The vendored flash route kept a 32K visible-key staging cap, so every
  chunk past 32K fell back to the sm_70 prompt kernel at 105 ms a call; at 66K that alone cost 32 s.
  The cap is now 131,072 keys and the route runs 1024-token Q blocks. 66K prefill 710.8 → 838 tok/s,
  27K 1000 → 1010. The staging planes come out of Auto KV: capacity at `--max-context 131072` drops
  from 286,400 to 281,152 tokens, and free VRAM after startup is unchanged.

## Known costs

- Prefill at 66K is 1.0% behind 1.5. The attribution is measured, not guessed: the exact NVFP4
  decode costs 645 ms more than 1.5's two-rounding FP16 shift trick over an 80 s prefill (that trick
  fails this line's A16 criterion, relative L2 3.3e-3), and the prompt is split into 1920-token
  attention steps plus a 128-token tail, which issues 8% more flash launches than 1.5's 2048-token
  steps. Per request the decode gain dominates: a 27K prompt with 500 output tokens finishes 2.5%
  sooner, a 66K prompt 0.3% sooner.
- `--spec mtp` output is not token-identical to running without speculation, on this line and on
  1.5 alike: batched verification reaches near-tied logits differently. Use no `--spec` when exact
  equivalence to an unspeculated decode matters.

## Assets

| File | Contents |
|---|---|
| `ninfer-v100-v1.6.2-linux-x86_64-sm70.tar.gz` | `ninfer` CLI + `ninfer-serve` + `ninfer-perplexity` |
| `ninfer-serve-sm70.tar.gz` | server binary only |
| `SHA256SUMS.txt` | checksums for both |

## Documentation

The measured record, including the rejected approaches and the per-kernel attribution, is in
[v1.6-vs-v1.5-2026-10-20.md](docs/v100/v1.6-vs-v1.5-2026-10-20.md).
