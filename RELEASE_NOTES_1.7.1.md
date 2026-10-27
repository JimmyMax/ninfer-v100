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
layout; the permute is a no-op for artifacts without Q6_K projections, so NVFP4 artifacts
are unaffected. Artifacts whose recipes store a fused `[gate; up]` Q6_K parent (for example
the IQ3_S 27B, whose MTP layer stores gate and up row-contiguously in one object) keep that
parent in the stored layout: the fused swiglu kernel family reads standard GGUF blocks only,
and the dense permute is per-object (`fused parents stay standard` fix below).

A follow-up fix keeps exactly those fused Q6_K parents out of the prepack. Before it, such an
artifact failed to start on sm_70 with `gguf swiglu: dense Q6_K parents must be non-fused`,
because the prepack selected the shared parent object and permuted it while the fused swiglu
route could not read the dense layout.

Q4_K_M 27B, V100, int8 KV:

| mode | 1.7.0 | 1.7.1 | change |
|---|---:|---:|---:|
| plain decode (tg128) | 35.1 | **38.1** | **+8.7%** |
| MTP d3 decode (tg128) | 54.1 | **57.2** | **+5.7%** |

## vs v1.5.0 (NVFP4 27B artifact, greedy, INT8 group-64 KV, `--max-context 200000`, one V100-32GB)

Matched to the v1.5.0 release protocol: `--spec mtp --draft-tokens 4`, 1024 generated
tokens per run, four prompts of the same content on both builds:

| prompt | tokens | v1.5.0 decode | v1.5.0 round | v1.7.1 decode | v1.7.1 round |
|---|---:|---:|---:|---:|---:|
| code | 7,093 | 81.8 | 43.7 ms | 81.7 | **40.4 ms** |
| code | 30,837 | 75.3 | 46.6 ms | **82.3** | **42.8 ms** |
| Chinese | 6,365 | 77.7 | 43.9 ms | **86.4** | **40.5 ms** |
| Chinese | 25,485 | 73.7 | 45.6 ms | **81.0** | **42.4 ms** |

The round time is the drift-free quantity: v1.7.1 is 7–8% faster per round at every
length, so at equal draft acceptance it decodes faster. On three of the four prompts it
also takes the acceptance (MTP d4: 57.8–62.2% vs 58.8–63.3%, prompt-dependent trajectory
drift, both directions) and wins decode by 9–11%; on the 7K code prompt the acceptance
runs 6.5 pp lower and exactly offsets the round gain (81.7 vs 81.8).

The v1.5.0 release's headline numbers (101.1 @ 7.4K code, 94.8 @ 31.5K code) are
acceptance-driven and prompt-specific: this same v1.5.0 build gives 81.8 and 75.3 on
the code prompts above, so the gap to those figures is prompt predictability, not a
version regression. Prefill: 1.07k tok/s at 7K on both builds; at 27K, 957.7 (v1.7.1)
vs 965.5 (v1.5.0) at chunk 2048 and 982.0 (v1.7.1) vs 965.6 (v1.5.0) at chunk 4096.

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
