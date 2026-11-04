# NInfer V100 v1.7.2

Prebuilt **Tesla V100 (sm_70)** binary, compiled with CUDA 12.8 on Ubuntu 24.04 (glibc 2.39).
The V100 runtime hard-requires compute capability 7.0 — other GPUs are rejected at startup.
The published image packages the sm_70 build. A local two-architecture image (sm_70 + sm_89,
`Dockerfile.1.7.2.local`) covers hosts that also run an RTX 40 series card.

## Ampere: NVFP4 decode is memory-bound now (e2m1 shared-memory LUT)

Ada has no FP4 tensor cores, so the NVFP4 SIMT/GEMV kernels paid the `__nv_fp4x2_e2m1`
intrinsic's 4–6-instruction ALU sequence for every 4-bit code. The decode is now a 16-entry
`__shared__` LUT (bit-exact with the e2m1 spec) in all four NVFP4 decode kernels
([`b1a1a1a`](../../commits/b1a1a1afa)). Swift1.5 27B NVFP4, RTX 4080 SUPER (sm_89),
`--nvfp4-native`:

| mode | 1.7.1 | 1.7.2 | change |
|---|---:|---:|---:|
| plain decode (tg512) | 19.97 | **34.86** | **+75%** |
| DFlash2 K5 (tg512) | 41.89 | **66.92** | **+60%** |

Post-change profiling shows the NVFP4 kernels at 68–70% of DRAM throughput — the arithmetic
is no longer the limit. Quality is unchanged (bit-exact; perplexity identical).

## Ampere: load-time NVFP4 → row-split Q4 transcode, and `--nvfp4-native`

Ampere-tier loads now transcode NVFP4 tensors to `Q4_G64_FP16` row-split at load time
([`95e1f37`](../../commits/95e1f3773)), which suits mixed-format W4A4 artifacts
(+12.3% decode on the Coder390 W4A4). Pure-NVFP4 artifacts (Swift1.5) are faster **native**,
so `--nvfp4-native` (CLI, serve, bench, perplexity) keeps the stored NVFP4 path — and with
the LUT above, native is the right choice there (+119% over transcode). The flag is ignored
on Volta, which always runs its native QPN route.

## Flash-Next (V100): CPU experts skip the bus, and the n-gram table is safe

- Hybrid execution: CPU threads now compute their expert share straight from the host copy
  instead of staging it over PCIe first ([`ee2ef85`](../../commits/ee2ef8504)), with the
  expert-miss staging path attributed in the bench. Measured final configuration and lever
  space are in `docs/performance.md`.
- The n-gram table no longer garbles output: untrained/uninitialised rows are zeroed and
  non-finite rows rejected ([`0aa1761`](../../commits/0aa17618a)).
- Converter: the qwen4_exp GGUF recipe accepts ggml block-stored GDN controls and older MTP
  block encodings ([`64f6c5c`](../../commits/64f6c5c7d), [`229fe30`](../../commits/229fe3037)).
- 200,000-token context with `--kv-dtype rk4v4` without losing expert cache
  (+0.040% perplexity at that setting).
- Optional ggml-cpu fast path for host experts (`20aa77a`), off by default.

## Build and diagnostics

- Multi-architecture builds link again, and Volta-only units compile for sm_70 alone
  ([`5062dcc`](../../commits/5062dccef)) — a fat `70;89` build was broken before.
- `NINFER_ROUND_TIMING=1` logs per-decode-round device/finish/commit wall time
  ([`ab24fd1`](../../commits/ab24fd17f)); default off, zero overhead.

## Documentation

The README is now the V100-32GB single-card document, and `docs/performance.md` carries only
this line's own measurements (V100-32GB and RTX 4080 SUPER), including the 512-token decode
protocol — the 128-token figures understated the rate because the draft head's acceptance
rises over the sequence.

## Install

Docker (V100):

```
docker pull ghcr.io/jimmymax/ninfer-v100:1.7.2
```

Or unpack `ninfer-v100-v1.7.2-linux-x86_64-sm70.tar.gz` and run `bin/ninfer-serve` on a host
with CUDA 12.8+ driver libraries. See `README.md` for the measured artifact table and the
serving commands.
