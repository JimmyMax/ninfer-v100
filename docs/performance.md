# Performance

Measured results for the **single-GPU Tesla V100-PCIE-32GB (`sm_70`)** line and, where it was the
point of the measurement, an **RTX 4080 SUPER (`sm_89`)** on the same host. Every number here was
measured in this repository, on these two cards, with the artifacts in the table below; results
from other cards, other forks or other artifact families are not carried into this document.

| Card | Memory | Build | Host link | Measured here |
|---|---:|---|---|---|
| Tesla V100-PCIE-32GB | 32 GB | `70` | PCIe 3.0 x16 | everything below |
| RTX 4080 SUPER | 32 GB | `89` | PCIe 3.0 x16 (Gen4 part, Gen3 platform) | the cross-card section |

Artifacts measured: `Swift-1.5-Qwen3.8-27B` Q4_K_M (GGUF `q4_k`), `Qwen3.8-27B-GSQ-RCO-IQ3_S`
(GGUF blocks), `Huihui-Qwen3.8-27B-abliterated` Q4_K_M, `orcarouter-Qwen3.8-27B` NVFP4 (native),
`orcarouter-Qwen3.8-27B-Uncensored` NVFP4, `Qwen3.8-27B-Coder390-W4A4-MTP-DFlash2`,
`Gemma4-31B-QAT` (dense GGUF), `Gemma4-26B-A4B-QAT` (GGUF Q4_0), `Qwen3.8-Flash-Next-GSQ-RCO`
IQ3_S and Q2_0 (GGUF blocks, host experts, shared n-gram table).

## How the numbers were taken

- **Server**: `ninfer-serve`, one request at a time, greedy, thinking off. *Short chat* is the mean
  decode rate of five 512-token answers; a depth is a cold document of that many tokens followed by
  an answer of about 400 words; prompt time is the server's time to the first token.
- **Engine**: `ninfer_bench`, the Engine without HTTP. `pp512` is a 512-token prompt, `tg128` 128
  decoded tokens. Bench rows name their corpus and repetitions.
- **CLI**: one `ninfer` request; its summary gives the prompt and decode rates.

Numbers from different methods, hosts or builds are not read against each other as a benchmark.
Drafters are written as MTP or DFlash2 with their draft count; acceptance is accepted / drafted
drafts with the tokens committed per verify round in brackets.

## Largest context

The largest `--max-context` each artifact was started and served at on this line's 32 GB cards,
INT8 KV unless a row says otherwise. The expert residency is the one the artifact needs on a 32 GB
card: weights that fit stay on the device, Flash-Next's routed experts go to pinned host memory.

| Artifact | Card | Residency | Window, tokens | Notes |
|---|---|---|---:|---|
| `orcarouter-Qwen3.8-27B` NVFP4 (19.1 GiB) | V100 | device | 254,208 | DFlash2 adapter + MTP head |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S` (11.6 GiB on device) | V100 | device | 400,000 | GGUF blocks |
| `Gemma4-31B-QAT` (dense, 17.6 GiB) | V100 | device | 32,768 | 10.5 GiB still free |
| `Gemma4-26B-A4B-QAT` (16.9 GiB, 8 of 128 experts) | V100 | device | 28,672 | 14.1 GiB still free |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` | V100 | pinned host memory | 200,000 | rk4v4, with the vision tower; expert cache 18,753 MiB |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` | V100 | pinned host memory | 262,144 | rk4v4; expert cache 18,285 MiB |
| `Qwen3.8-27B-Coder390-W4A4-MTP-DFlash2` | 4080 SUPER | device | 200,000 | the serving configuration |

Flash-Next's window trades against its decode rate through the expert cache. On this card the
context's cost is the GDN recurrent state, which is stored in the KV format and grows linearly:
at 200,000 tokens the state is 3,046 MiB with `int8`, 2,451 with `rk8v4` and 1,816 with `rk4v4`,
against 2,019 at 130,000 with `int8`. Measured on the service (concurrency 1, vision, MTP 3,
double-buffered prefill), the expert cache at 200,000 tokens is:

| `--kv-dtype` | GDN state | Workspace | Expert cache | Perplexity, against `int8` |
|---|---:|---:|---:|---:|
| `bf16` | 5,506 MiB | 5,884 | 15,113 MiB | lossless reference |
| `int8` | 3,046 MiB | 5,884 | 17,656 MiB | — |
| `rk8v4` | 2,451 MiB | 5,884 | 18,129 MiB | +0.81% (4,094 tokens) |
| `rk4v4` | 1,816 MiB | 5,884 | 18,753 MiB | **+0.040% (20,470 tokens)** |

So 200,000 tokens does not have to cost cache: `rk4v4` at 200,000 holds 130 MiB more than the old
130,000/`int8` configuration did. The format costs something: on 4,094 scored tokens of prose
`rk4v4` read +1.28% over `int8` and `int8` read −1.16% against `bf16`, differences inside that
sample's ±2.7% standard error; over 20,470 tokens on the same text the `rk4v4`-versus-`int8`
difference settles at **+0.040%** (mean nll 2.704615 against 2.705015, perplexity 14.9486 against
14.9545). `--gdn-state-fp16` halves only the host state image and does not change the device state.
A decode request on the 200,000/`rk4v4` service reads 37.3 and 37.2 tok/s over a 256-token answer,
against 35.0 and 37.5 on the 130,000/`int8` one (same prompt, same sitting, MTP 3), so the wider
context does not cost decode.

## Qwen3.8-27B GSQ-RCO IQ3_S, one V100-PCIE-32GB

### Against the NVFP4 OrcaRouter artifact

`ninfer_bench`, INT8 group-64 KV, `--max-context 200000`, prefill chunk 2048, greedy, 3 measured
repetitions plus one warmup; the same four prompts on both artifacts, tokenized with the
artifact's own tokenizer and chat template. The NVFP4 artifact is the OrcaRouter fine-tune
(18.9 GiB of weights); IQ3_S is 11.6 GiB on the device. The bases differ, so this is an
end-to-end artifact comparison. MTP 4 decodes 1,024 tokens; the no-drafter row decodes 256.
Prefill is the measured prompt rate of the MTP run:

| prompt | tokens | MTP 4, IQ3_S | MTP 4, NVFP4 | no drafter, IQ3_S | no drafter, NVFP4 | prefill, IQ3_S / NVFP4 |
|---|---:|---:|---:|---:|---:|---:|
| code | 7,051 | 102.1 (0.84) | **107.3** (0.84) | **41.1** | 33.2 | 975 / **1,089** |
| code | 30,795 | 97.7 (0.85) | **104.2** (0.86) | **38.0** | 31.1 | 870 / **960** |
| Chinese | 6,323 | 75.3 (0.55) | **78.2** (0.54) | **41.2** | 33.3 | 959 / **1,070** |
| Chinese | 25,443 | 72.7 (0.56) | **75.8** (0.55) | **38.7** | 31.6 | 894 / **986** |

NVFP4 decodes 4–7% faster under MTP 4 with acceptance within 1.5 points on every prompt
(13.4–14.2 ms per verify round against 14.2–15.0), so the gap is per-round format speed,
not trajectory. Without a drafter IQ3_S decodes 22–24% faster — the 11.6 GiB against 18.9 GiB
of device weight bytes in the bandwidth-bound round — and NVFP4 prompts 10–12% faster. IQ3_S keeps
7.2 GiB of VRAM free relative to the NVFP4 artifact.

IQ3_S stores five Q6_K objects, all in its MTP layer; the fused `[gate; up]` parent of the MTP
MLP keeps its stored layout (the dense prepack's fused-parent exclusion) while the other Q6_K
projections take the dense layout.

### Fastest decode record, all three artifacts (1.7.2)

The same continuation workload on all three 27B artifacts: `ninfer_bench -pg 7051,512`, a true
long-prompt decode (a 7,051-token code prompt, then a timed 512-token continuation), MTP 4,
INT8 KV, chunk 2048, greedy, three repetitions. This is the highest-acceptance workload the
model produces (the continuation is near-deterministic), so it is the fastest decode the card
can sustain. The standard `-p`/`-n` flags are *not* this: they run a prefill plus a one-token-seed
free generation, whose acceptance is a trajectory artifact, not the prompt's.

| artifact | MTP 4 decode | no drafter | speedup | acceptance | prefill 7,051 |
|---|---:|---:|---:|---:|---:|
| NVFP4 OrcaRouter | **108.4** | 33.2 | 3.26× | 84.9% | **1,089** |
| Q4_K_M | 104.8 | 36.8 | 2.85× | 87.9% | 980 |
| IQ3_S GSQ-RCO | 104.5 | 41.2 | 2.54× | 86.7% | 980 |

Fastest single number: **108.4 tok/s** (NVFP4). Under these best conditions the native profile's
decode edge over both GGUF artifacts is only **3.4–3.7%** — not the ~25% that the one-token-seed
free-generation table in the README shows. On a near-deterministic continuation all three artifacts
accept 85–88% of drafts (a 2–3-point spread), so the per-round format speed decides: NVFP4's
QPN route reads 4-bit blocks directly (13.4–14.2 ms per round) while the mixed GGUF blocks pay a
dequantize pass (14.2–15.0 ms), and that outweighs NVFP4's slightly lower acceptance. Prefill stays
+11% for NVFP4, and without a drafter the order inverts (IQ3_S 41.2, +24% over NVFP4's 33.2).

## Qwen3.8-Flash-Next

A 48-layer MoE that routes 512 experts and reads a shared n-gram table. Its routed experts do not
fit the 32 GB card in device memory (53.7 GiB for IQ3_S), so they live in pinned host memory
(`--expert-residency host`) and each round stages only the experts a token routes.

**The decode rate is set by miss traffic, not compute.** IQ3_S's routed experts are 53.7 GiB, so
on a 32 GB card they cross PCIe 3.0 (~13 GB/s effective) instead of being read at ~875 GB/s from
HBM. That is the structure's floor, not the card's ceiling.

### CPU arithmetic for the host experts, one V100

The host experts the CPU computes (the `--expert-dma-share` share of a call's missing experts)
run on the exact q8_1 products (the default, bit for bit with the GPU) or on ggml-cpu's AVX2
vecdot kernels (`--expert-cpu-arithmetic ggml`), which quantize the activation to q8_K for the
i-quant gate/up and to q8_0 for the IQ4_NL and Q8_0 down; a projection ggml-cpu does not cover
(GSQ-RCO's Q2_0) keeps the exact products, so both modes run the same artifact.

Tesla V100-PCIE-32GB, the IQ3_S release, host experts, an 8,192-position context, INT8 KV, MTP
with three drafts, a 512-token prompt and 128 generated tokens of a 7,051-token code corpus,
two repetitions; the expert cache holds 24,118 MiB. Decode tok/s, the drafts accepted:

| CPU share of the misses | Arithmetic | Decode, tok/s | Drafts accepted |
|---|---|---:|---:|
| 0% (all on the GPU) | exact | 30.0 | 62.5% |
| 0% (all on the GPU) | ggml | 30.1 | 62.5% |
| 50% | exact | 26.0 | 62.5% |
| 50% | ggml | 26.5 | 63.6% |
| balanced | ggml | 25.4 | 60.1% |

The host CPU is a Xeon E5-2696 v3 (18 cores, AVX2). At a 50% share the CPU computes a round's
share of the missing experts in about 20 ms, more than the round's slack at ~30 tok/s, so the
second stage waits on it and the hybrid loses to the all-GPU miss path (30.0-30.1 tok/s); the
vecdot kernels are +1.8% over the exact q8_1 products for the share they compute.

The same sweep on a clean sitting (this host, the exact q8_1 arithmetic):

| DMA share | 1.0 | 0.8 | 0.7 | 0.9 | 0.6 |
|---|---:|---:|---:|---:|---:|
| Decode, tok/s | 43.5 | 43.5 | 28.1 | 28.6 | 27.0 |

A 20% CPU share costs nothing (it fits in the round's staging time); every other share tested
costs about 35%, because the CPU's i-quant vecdot products run at a few GB/s of expert bytes
(compute-bound on this Xeon, ~0.6-2 GB/s effective), far slower than the 13 GB/s bus they replace,
so the round ends up waiting on the CPU. **The all-GPU miss path (`--expert-dma-share 1`) is the
fastest configuration on this host.**

Making the CPU skip the device cache (a CPU-computed expert is no longer re-copied into a slot)
removes its bytes from the bus — a 50% share cuts the round's host-to-device expert traffic
104 → 66 GB — but decode still falls (same sitting, MTP 3: 27.7 all-GPU → 23.9 at a 50% share,
service time 9.1 → 12.5 s), because the CPU is the slower resource here. The change is kept as the
default for a faster host, with no effect on the `--expert-dma-share 1` path.

### Where a decode round's expert bytes go

The bench reports a run's expert bytes by category (`expert_seed_bytes`,
`expert_rebalance_bytes`, `expert_admit_bytes`, plus the two-stage staging-bank copies), so a
104 GB run breaks into the one-time profile seed (23.1 GB, the warm start), the pass-end budget
(4.9 GB), the steady-state miss admissions into slots (38.0 GB) and the two-stage staging-bank
copies (38.0 GB). The steady-state traffic is split evenly between slot admissions and
staging-bank copies: a miss is admitted into a slot and also staged for the round that needs it
before the admission lands, and the long tail that never fits a slot is staged every round it is
routed. 76 GB of steady-state traffic per run is set by the miss rate and the expert size, not by
a removable copy.

### Prefill of host experts, one V100

Prefill is staging-bound the same way decode is. A wide pass (≤2,048 tokens, one pass) runs each
layer serially: the router needs the layer's input, so the experts the next layer will route to
are only known after the current layer's MoE; the staged copy of a layer's routed experts and its
MoE cannot overlap across layers. A long prompt (>2,048 tokens) goes through the span path, which
stages each layer's uncached experts once per span (up to 8 chunks) instead of once per pass. The
measured per-prompt expert staging is nearly fixed:

| Prompt | Staging transferred | Prefill, tok/s | Per token |
|---|---:|---:|---:|
| 512 | 19.4-21.5 GB | 157.5 (clean) | ~38-42 MiB |
| 2,048 | 21.3 GB | 402.9 | 10.4 MiB |
| 4,096 | 25.4 GB | 695.2 | 6.2 MiB |

The 512-token 157.5 tok/s is a small-prompt artifact: the per-prompt staging floor is ~19-25 GB of
routed experts plus the decode-bound admissions, ~1.5 s of PCIe 3.0 time plus ~1.6 s of compute,
and the measured 3.25 s sits on that floor. A cold first prompt (no warm-up overlap) measures the
same staging (19.4 GB, 10.1% cache hits) because the pool stages the routed subset, not the whole
bank. Disk residency costs −60% of prefill (page-cache reads instead of pinned DMA plus a second
staging).

The live service reproduces the same shape, because its reported `prefill = prompt / TTFT` includes
that fixed staging. Five prompts on the 130,000-token, vision-enabled service (expert cache
18,623 MiB), one sitting, `thinking` off:

| Prompt, tokens | 19 | 524 | 1,280 | 2,330 | 4,220 |
|---|---:|---:|---:|---:|---:|
| TTFT, s | 0.80 | 3.6 | 3.8 | 4.0 | 5.3 |
| Reported prefill, tok/s | 23.9 | 146.0 | 350.3 | 606.4 | 824.8 |

The five points fit a fixed cost plus a marginal rate:

    TTFT ≈ 3.3 s + 0.47 ms per prompt token      (marginal ≈ 2,100 tok/s)

The 3.3 s is the per-prompt expert staging; the 0.47 ms is the token streaming. So the number the
log prints is `N / (3.3 s + 0.47 ms·N)`, which collapses for a short prompt and approaches ~2,100
tok/s as the prompt grows: 6 tok/s at 19 tokens on that line, 148 at 524, 799 at 4,220, ~1,470 at
16,000. A short prompt's reported rate is therefore not a measurement of the model's prefill
throughput. (The 19-token row also carries the first request's warm-up, so it sits below the line.)

`--expert-prefill-buffers 2` double-buffers the span path so the next layer's bank is copied while
the current layer computes. Interleaved A/B at a 4,096-token prompt, same sitting: 363.7 / 338.7
(one buffer) against 428.1 / 430.8 (two) — +17.7% / +27.0%. At a 512-token prompt it is neutral to
negative, because a single pass cannot prefetch across layers at all. The second buffer holds
1.1 GiB, taken from the expert cache; the Flash-Next service enables it for its long prompts.

The nvprof decomposition of a 4,096-token span prefill (two buffers, one pass per layer):

| Activity | Share of profiled GPU time |
|---|---:|
| host-to-device expert staging (25.4 GB, 120,649 copies) | 48.5% |
| MoE wide calls (i-quant, all layers) | ~14% |
| GDN recurrent attention | 9.1% |
| cuBLAS GEMMs (embed, router, shared expert) | 7.7% |
| full-attention layers (12, O(T²)) | 3.9% |
| host-side waits | 3.1% |
| hyper-connections, routing, KV, quantize | ~10% |

Staging at the bus peak (~12 GB/s wall) is ~2.0 s and compute (~55 TFLOP of the 4,096 tokens) is
~1.2-2.2 s; with the two buffers they overlap, and the measured 4.3 s is close to
max(staging, compute) plus the wait tail. The kernel-side levers were checked and all came up
empty: the Volta prefill already runs GDN as one full-width recurrent kernel per layer-pass (op
bench, cold L2, T=2048: 3.54 ms against 8.40 ms for the chunked pipeline); the wide i-quant calls
already use the widest available column tile (128, the vendored ggml config's limit); and the wait
tail is the GPU spinning during the staging itself, so it falls with the hit rate and has no
independent lever. A same-sitting 2,048-versus-4,096 A/B (both arms double-buffered) measured
436.0 against 496.9 tok/s (+14%): a fixed per-prompt staging cost of ~1.15 s on top of ~1.73 ms
token streaming.

### IQ3_S on one V100: the final configuration

Experts in pinned host memory, every miss on the GPU, INT8 KV, an 8,192-position context, exact
CPU arithmetic, CUDA graphs on. A 512-token prompt and 128 generated tokens of a 7,051-token code
corpus, two repetitions; the expert cache holds 24,118 MiB (7,372 of 24,576 experts, 30.0%):

| Drafter | Prefill, tok/s | Decode, tok/s | Drafts accepted | Per draft position |
|---|---:|---:|---:|---|
| none | 157.5 | 33.6 | | |
| MTP 2 | 155.4 | 43.5 | 71.9% | 84 / 67% |
| MTP 3 (final) | 156.8-157.4 | 43.5-44.0 | 62.5% | 67 / 58 / 42% |
| MTP 4 | 157.4 | 39.8 | 54.0% | 62 / 49 / 34 / 30% |

What a decode round spends on (nvprof over the measured region, MTP 3, all misses on the GPU):

| Activity | Share of GPU time |
|---|---:|
| host-to-device expert staging (145,603 copies, 80.4 µs each) | 49.0% |
| waiting on the CPU side of the round | 13.5% |
| GDN recurrent attention | 6.8% |
| MoE second stage (down projections, Q4_NL) | 3.4% |
| cuBLAS GEMMs (embed, router, shared expert) | 3.1% |
| MoE first stage, vector ops, routing, PLE, KV | ~24% |

Levers measured on this card, clean sittings:

| Lever | Result |
|---|---|
| `--draft-min-p 10` | 43.8 tok/s; no draft falls below the 10% floor (the MTP head is over-confident), so the filter removes nothing |
| `--expert-dma-share 0.8` | 43.5 tok/s (equal; a 20% CPU share fits in the staging time) |
| `--expert-dma-share` 0.7 / 0.9 / 0.6 | 28.1 / 28.6 / 27.0 tok/s (−35%) |
| `--expert-dma-share auto` | 24.9 tok/s; the balance pushes more share to the slow CPU |
| ggml-cpu arithmetic at `--expert-dma-share 1` | no effect (no miss is computed on the CPU) |
| CUDA graphs off | equal |
| n-gram copy drafting (15 tokens) | the mixed speculation dilutes the acceptance and adds distinct experts to the round's staging |
| n-gram lookup drafting (one more verify token) | +1 verify token ≈ +10% of the round's staging; the extra token's own staging outweighs the draft's gain |
| `--max-context 8192` instead of 200,000 | 8192 frees 8.5 GiB for the expert cache (24.1 vs 15.6 GiB); decode scales with the cache — 24.9 / 27.9 / 30.5 tok/s at 200,000 / 32,768 / 8,192 context |
| disk experts (page cache) | decode equal to host, prefill −60% |
| two-stage staging (gate+up, then down) | neutral at t=1; kept, harmless |
| prompt routes kept out of the cache's frequency tiers | nospec 27.7 → 34.0 tok/s (+23%), hit rate 66.9% → 92.7%; MTP 3 with a seeded cache 31.5 → 33.3 (+5.7%) |
| seeding from a route histogram (`--expert-profile`) | decode 31.1 → 38.9 tok/s (+25%) on a cold cache, staging service time 2.91 → 1.67 s |

The 60-80 tok/s target is out of reach for IQ3_S on this card **in the current shape of the
round**: the round's staging and compute are serialized, the bus is ~13 GB/s, the CPU's i-quant
products are far slower than the bus, and the cache cannot grow much further without taking VRAM
from the context. Two measurements show the remaining structure:

* An 8-token prompt with 512 decode tokens reaches **56.3 tok/s** (acceptance 71.8%), because the
  small prompt leaves the cache to decode routing.
* The recorded per-layer route histogram of a 512-token decode shows that the 248 experts per
  layer that 24.1 GiB can hold cover **95.1% of all routes** if residency were chosen by frequency
  (top-100: 75.1%, top-150: 85.9%, top-300: 98.5%). The adaptive policy reaches ~86% in steady
  state, so ~9 points of hit rate — and with them ~40% of the staging — are recoverable without
  more VRAM.

Device residency (53.7 GiB for IQ3_S) and the Q2_0 release remain the other two ways past the
44 tok/s line.

### Q2_0 on one V100

The Q2_0 release (40.7 GiB; the n-gram table is the shared one), experts in pinned host memory,
every miss on the GPU, an 8,192-position context, INT8 KV, a 512-token prompt and 128 generated
tokens of a 7,051-token code corpus, two repetitions; the expert cache holds 25,759 MiB (nospec) /
25,638 MiB (MTP). Greedy, one request:

| Drafter | Prefill, tok/s | Decode, tok/s | Drafts accepted | Misses served by |
|---|---:|---:|---:|---|
| none | 224.8 | 44.0 | | GPU, 67.0% cache hits |
| MTP 3 | 241.3 | 69.6 | 70.7% | GPU, 73.1% cache hits |

The Q2_0 experts are 1.32 MiB against IQ3_S's 3.27 MiB, so a round's miss traffic falls by the
same factor; the cache holds 19,539 of the 24,576 experts (79.5%) instead of 7,372 (30.0%).
Prefill rises from 157 to 225 tok/s on the same protocol.

### IQ3_S on an RTX 4080 SUPER, against one V100

Same artifact, same flags, both cards in the same sitting: host experts, every miss on the GPU,
INT8 KV, an 8,192-position context, a 512-token prompt and 128 generated tokens of a 7,051-token
code corpus, a warmup and two repetitions, double-buffered prefill, no profile seed. Each card runs
its own build (`sm_70` and `sm_89`):

| Card | Drafter | Prefill, tok/s | Decode, tok/s | Drafts accepted | Expert cache | Cache hits |
|---|---|---:|---:|---:|---:|---:|
| Tesla V100-PCIE-32GB | none | 175.8 | 36.1 | — | 22,983 MiB | 92.1% |
| Tesla V100-PCIE-32GB | MTP 3 | 174.9 | 45.2 | 58.4% | 22,816 MiB | 90.0% |
| RTX 4080 SUPER | none | 180.3 | **75.0** | — | 22,773 MiB | 92.6% |
| RTX 4080 SUPER | MTP 3 | 182.1 | **87.2** | 62.5% | 22,605 MiB | 90.9% |

The 4080 SUPER decodes **2.08× (no drafter) / 1.93× (MTP 3)** the V100's rate at the same cache
size and hit rate, and prefills at the same speed (1.03-1.04×). The split has one cause: both cards
sit on the host's **PCIe 3.0 x16** link (the 4080 SUPER is a Gen4 part whose link negotiates down to
Gen3 on this X99 platform; `nvidia-smi` reports a Gen3 maximum), and the two phases lean on different
resources.

* Prefill is expert-staging-bound: every prompt moves its routed experts across that same Gen3
  link, so the extra compute of the 4080 SUPER buys nothing. At the same cache size and hit rate,
  the bus sets the number and it lands within 4% on both cards.
* Decode at ~90% cache hits is not bus-bound: the miss traffic is nearly the same per token
  (494 against 465 MB of counted expert bytes per output token), so what separates 45 from 87 tok/s
  is the per-expert arithmetic. `sm_89` runs the GGUF i-quant GEMV and MMA routes natively (and
  clocks at 3.1 GHz against the V100's 1.38), where `sm_70` reaches them through its Volta routes;
  the 4080 SUPER keeps its advantage with a drafter too, because the round's compute is not the
  part that speculation removes.

The practical reading: on this host a 4080 SUPER is the better decode card for Flash-Next and no
better at all for prefill, so a long-prompt workload gains little from it, while a
chat/instruct workload gains about 2×.

### Quality

The first 72 KB of the WikiText sample in `perplexity-1m`, 2,560-token windows advancing by 1,024,
BF16 KV, the fourteen windows llama.cpp evaluates identically: Q2_0 2.6509 on the Tesla V100
(llama.cpp 2.6502); IQ3_S 2.1317 (llama.cpp 2.1278). Host and disk experts give the same perplexity
on this card (2.6057 over the first 40 KB with INT8 KV). On the same smoke prompt the MTP answer is
identical to the plain decode's.

## Gemma 4, one V100-PCIE-32GB

Both artifacts are GGUF-derived and run through the ggml bridge, but their shapes differ: the
26B-A4B routes 8 of 128 experts per token, while the 31B is **dense** — 60 layers at hidden 5376,
no experts. One request each, a 5,513-token prompt over int8 KV, 500 output tokens
([Gemma 4 on the V100-32GB](performance/gemma4-v100-32gb-2026-10.md)):

| Artifact | Weights | Prefill, tok/s | Decode, tok/s | Without speculation |
|---|---:|---:|---:|---:|
| Gemma 4 31B (dense, hidden 5376) | 17.6 GiB | 715 | **54.1** (MTP 4, 54.5%) | 27.4 |
| Gemma 4 26B-A4B (128 experts, 8 per token) | 16.9 GiB | 2.39k | **157.2** (MTP 2, 71.7%) | 120.3 |

The dense 31B reads all 17.6 GiB per token, so unspeculated decode is bandwidth: 27.4 tok/s is
about 517 GB/s of the ~790 GB/s this card's HBM2 moves with ECC on. Speculation is what lifts it —
four drafts verified per weight read take the same traffic to 54.1 tok/s. The A4B's file is only 4%
smaller, but a token touches 8 of 128 experts, so it decodes 4.4× faster and gains less from
drafting (1.3× against 2.0×).

Context that fits beside them: 32,768 tokens of int8 KV leave 10.5 GiB free on the 31B; the A4B
leaves 14.1 GiB free at 28,672 tokens.

## Why a host-expert MoE is slower to prefill than a dense model that fits

The question comes up because a MoE activates fewer parameters per token. On a 32 GB card the
condition does not hold: a MoE only wins when its activated experts are cheap to *read*, and
Flash-Next's 53.7 GiB of routed experts are in host memory.

| Model | Where the weights are | Prompt rate |
|---|---|---:|
| Qwen3.8-27B (native, 19.1 GiB) | device | 1.00k tok/s at 27K tokens |
| Gemma 4 26B-A4B (16.9 GiB, 8 of 128 experts) | device | 2.39k tok/s at 5.5K tokens |
| Qwen3.8-Flash-Next IQ3_S (53.7 GiB routed) | pinned host memory | 403 tok/s at 2,048; 695 at 4,096 |
| Qwen3.8-Flash-Next Q2_0 (34 GiB routed) | pinned host memory | 225 tok/s at 512; 241 with MTP 3 |

A model whose weights fit is read at HBM speed; Flash-Next pays ~20-25 GB of PCIe 3.0 staging per
prompt instead. That is a memory-placement difference, not a density difference, and it is why the
same engine on the same card reaches 2.39k tok/s on the Gemma MoE and 400-700 tok/s on Flash-Next.

## NVFP4 on an RTX 4080 SUPER: the e2m1 LUT decode and the load-time Q4 transcode

Ada (`sm_86`/`sm_89`) has no FP4 Tensor Cores, so a stored NVFP4 weight runs on the SIMT/MMA
route. The original route was instruction-bound: the e2m1 hardware intrinsic (`__nv_fp4x2_e2m1`)
expands to a 4–6 instruction ALU sequence per 4-bit code, and the kernels sat at 83% SM active
with only ~1 TFLOP/s of useful throughput.

A 16-entry shared-memory LUT now replaces the intrinsic in all four NVFP4 decode kernels (GEMV,
SIMT, Swiglu decode, Swiglu small-t). The LUT values are bit-exact with the e2m1 spec, so the
numerics and perplexity are unchanged. The kernels are now memory-bound (SM 56%, DRAM 68–70%),
and the decode round runs 1.6–1.75× faster:

| Swift1.5 NVFP4 27B, 4080 SUPER, 512-token | Before LUT | After LUT |
|---|---:|---:|
| nospec decode | 19.97 tok/s | **34.86 tok/s** (1.75×) |
| DFlash2 K5 decode | 41.89 tok/s | **66.92 tok/s** (1.60×), acc 0.386 |
| swiglu T=1 microbench | 327.7 µs | **148.5 µs** (2.2×) |

An Ampere build can additionally transcode each device-resident NVFP4 weight to row-split
`Q4_G64_FP16` at load, so the linear layers take the memory-bound row-split Q4 GEMV.
`--nvfp4-native` keeps the native route. The transcode is a single-scale requantization
(`absmax / 7`, within a fraction of a percent of the clipping-search optimum at a tenth of the
load cost). It costs load time (9 s to 119 s for the 8–9 GiB of NVFP4) and one consequence: a
DFlash2 drafter that **shares** the main model's MLP through a binding range now sees the
transcoded weights, so its acceptance collapses toward zero. The net is artifact-dependent,
measured on the 4080 SUPER with a 512-token generation, DFlash2 K5, three repetitions, **with
the LUT decode**:

| Artifact | NVFP4 | Native, tok/s | Native acc | Transcode, tok/s | Transcode acc | Faster |
|---|---:|---:|---:|---:|---:|---|
| `Qwen3.8-27B-Coder390-W4A4-MTP-DFlash2` | 8.4 GiB | 25.00 | 0.150 | **28.32** | 0.004 | transcode +13% |
| `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-v3` | 9.0 GiB | **66.92** | 0.386 | 30.52 | 0.000 | native +119% |

The LUT decode makes the native route much faster, so the native route now wins decisively for
the Swift1.5 artifact (+119%, up from +37% before the LUT). The transcode still wins for W4A4
(where the drafter acceptance is low either way). The default is the transcode because the 4080
serving configuration is W4A4; use `--nvfp4-native` for an artifact whose native profile is the
stronger one (now the clear choice for Swift1.5). Volta (`sm_70`) is unaffected: its NVFP4 QPN
route is the fast path there, so the transcode is a compile-time no-op (the LUT also applies to
the sm_70 build, where it is a harmless shared-memory load).

## Determinism

Every configuration reproduces its own output exactly. Configurations do not reproduce each other:
a verification evaluates K+1 columns in one pass where plain decode evaluates one, a batch
evaluates several requests together, and the different reduction orders can flip a near-tied
argmax. MTP and DFlash2 outputs differ from each other and from plain greedy decoding, and DFlash2's
output varies with the draft count, without a measurable perplexity cost
([batch composition](device-profiles.md#batch-composition)).
