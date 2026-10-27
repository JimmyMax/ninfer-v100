# NInfer V100 v1.7.1

Prebuilt **Tesla V100 (sm_70)** binary, compiled with CUDA 12.8 on Ubuntu 24.04 (glibc 2.39).
The V100 runtime hard-requires compute capability 7.0 — other GPUs are rejected at startup.
The same image also carries an **sm_89** binary (RTX 40 series); a dispatch script in
`/usr/local/bin/ninfer-serve` selects the right one from the visible GPU's compute
capability, so one image serves both cards. `NINFER_IMAGE_ARCH=sm70|sm89` overrides the
choice.

## NVFP4 MTP acceptance restored

The Q6_K dense-prepack change in this release pulled the proposal head into the Volta
device-layout list unconditionally. On NVFP4 artifacts the proposal head is Q4_G64, so it
got panel-prepacked, the draft scores broke, MTP acceptance fell to ~0 (decode dropped to
~28 tok/s) and DFlash2 failed at startup. The layout pass now takes the proposal head only
when it is actually Q6_K ([`0311d39`](../../commits/0311d39)).

NVFP4 27B, V100, int8 KV, `--spec mtp --draft-tokens 3`: **65.5 tok/s** (44.2% acceptance)
after the fix, vs ~28 tok/s before. `--full-draft-head` is no longer needed as a workaround.

## GGUF Q6_K projections prepack into a dense 6-bit stream

Of a Q4_K_M 27B artifact's weights, Q6_K is ~26% and its vec kernel reads 210-byte rows
that are only 2-byte aligned, which caps the load width at 54–63% of the card's bandwidth.
The projections (dense and MTP, ~4.35 GiB of a 16.95 GiB artifact) are now repacked at load
time, device-side and in place, into a same-size dense 6-bit stream the vec kernel reads
with 64-bit loads ([`bd052b2`](../../commits/bd052b2)). MoE experts keep the standard
layout; the permute is a no-op for artifacts without Q6_K projections, so NVFP4 and IQ
artifacts are unaffected.

Q4_K_M 27B, V100, int8 KV:

| mode | 1.7.0 | 1.7.1 | change |
|---|---:|---:|---:|
| plain decode (tg128) | 35.1 | **38.1** | **+8.7%** |
| MTP d3 decode (tg128) | 54.1 | **57.2** | **+5.7%** |

## vs v1.5.0 (NVFP4 27B artifact, greedy, int8 KV, one V100-32GB)

| scenario | v1.5.0 | v1.7.1 |
|---|---:|---:|
| plain decode, short prompt | 33.9 | **34.3** |
| plain decode, 7K prompt | 32.7 | **33.0** |
| MTP d3, 7K prompt | 76.0 | **78.8** |
| MTP d3, 27K prompt | 77.4 | **78.5** |
| MTP d4, three-prompt | 70.8 | **78.5** |
| prefill, 27K, chunk 4096 | 965.6 | **982.0** |

Decode wins at every prompt length, with acceptance at parity. Prefill trails by 0.8–2.7%
at chunk 2048 and leads at chunk 4096, which v1.5.0 does not benefit from.

## Build

Dual-arch (`70;89`) compilation of this tree is pathologically slow on a few int8/splitk
units (30–174× on the sm_89 ptxas pass, ~10 h total), so the image is built from two
single-arch trees and packaged with the dispatcher above. Runtime behavior is identical to
a dual-arch build. The Volta kernel guards gained an explicit `__CUDA_ARCH__ == 700`
clause so the tree also compiles cleanly as one unit for several architectures
([`ec1d53a`](../../commits/ec1d53a)).

## Upgrade

- V100 services: `image: ninfer:1.7.1`, no other changes.
- NVFP4 artifact MTP: drop `--full-draft-head` if you added it as a workaround.
- Rollback: `ninfer:1.7.0`.
