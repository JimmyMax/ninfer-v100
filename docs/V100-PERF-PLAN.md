# V100 performance plan

Working plan for continued single-GPU performance work on the V100 build. Update it as items land
or are rejected; it is the entry point for "what next" alongside
[the V100 port notes](v100.md) and [the build recipe](V100-BUILD.md).

## Current state

Released as **v1.3.0** (commit `b0ad4799`, image `ninfer:v1.3.0`, published on origin with a GitHub
release), running in the local deployment. It contains:

| Change | Effect |
|---|---|
| Volta D256 flash prompt retune (64-column Q tile) | first token −26% to −29% at 186K–193K |
| INT8 small-T decode attention v2 (key-split 8-warp `Bc=64`, `byte_perm` int8→fp16 dequant) | decode +37% to +47% at 186K–193K, +4% to +17% at 8K–32K |
| Earlier upstream ports: DFlash prefill binding + cleanup synchronization, spin sync, materialization planner | correctness fixes; no measurable throughput effect on V100 |

Rejected after measurement: the upstream `perf(engine): reduce preparation and context management
overhead` port (a8e212ac). It showed no effect on prepare time or TTFT when measured with
server-side `timings_seconds.prepare`; the earlier apparent win was session-to-session drift.
Kept as the local tag `wip-a8e212ac` for reference.

## Measurement harness

| Piece | Location |
|---|---|
| Protocol, measurement rules, example commands | `bench/v100/README.md` |
| Prompt generator (code / Chinese documents, exact token counts) | `bench/v100/ctx_prompt.py` |
| Decode / first-token client (reads server `timings`) | `bench/v100/decode_bench.py` |
| Orchestration that swaps images and starts the server | deployment directory: `bench/run_ctx_matrix.sh`, `bench/run_longctx.sh`, `bench/docker-compose.bench.yml` |

Read `bench/v100/README.md` before running anything: it records the in-image A/B switch, the
drift-control rule, and the one-tenant-per-host rule. A concurrent build inflated host-side
preparation measurements by 2x in one session, and an uncontrolled comparison once produced a
spurious 36% "win" that a controlled rerun reduced to nothing.

Build and verify recipe:

```bash
# runtime image straight from the working tree (small export, ~35 min cold)
docker build -f Dockerfile.build-runtime.local --target runtime -t ninfer:<tag> .

# iteration image: apps + the Op benchmarks + selected tests, ccache-backed (Dockerfile.opt.local)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.opt.local --target opt -t ninfer:<tag> .
DOCKER_BUILDKIT=1 docker build -f Dockerfile.opt.local --target runtime -t ninfer:<tag> .
DOCKER_BUILDKIT=1 docker build -f Dockerfile.opt.local --target testrun -t ninfer:<tag>-tests .

# the decode attention Op in isolation: 1.2 ms per layer at 186K, W=4, int8 KV
docker run --rm --gpus '"device=1"' ninfer:<tag> ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h24-kv4 --kv-dtype int8 --batch 1 --tokens 1,4 \
  --context 7353,31538,186432 --cache warm --execution eager --repeat 20

# numeric + regression checks (the dflash2 batch is the fourth positional argument; the
# default of 8 exhausts memory on a 32 GB card — use 2)
docker run --rm --gpus '"device=1"' ninfer:<tag>-tests /tests/ninfer_softmax_attention_test
docker run --rm --gpus '"device=1"' -e NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS=<artifact> \
  -v <models-dir>:/workspace/models:ro ninfer:<tag>-tests \
  /tests/ninfer_qwen3_6_27b_dflash2_real_test 7 1 1 2
```

## Where the round time goes (measured)

Decode throughput is `(1 + draft_tokens × accept) / round`. Round time depends only on context
length, not on the content, while `accept` ranges from 0.36 (Chinese at 193K) to 0.79 (code at 8K)
at the same round time. `ms/round` — not tok/s — is the invariant to compare across runs, images,
or machines. The server's own `engine_timing` puts 95% of a round in device wait and 5% in host
exposure (2.9 ms of a 62.6 ms round at 186K), so every remaining lever is device-side.

| round time (ms) | 7.4K | 31.5K | 186K |
|---|---:|---:|---:|
| pre-v1.3.0 kernel, draft 3 | 41.0 | 49.3 | 89.9 |
| v1.3.0, draft 3 | 39.5 | 43.0 | 63.0 |
| v1.4.0, draft 4 | 38.6 | 41.3 | 62.5 |
| v1.4.0, draft 4, `NINFER_MTP_ATTN_WINDOW=8192` | 38.5 | 41.2 | 59.4 |

Pools at 186K, draft 3 (62.6 ms round):

| pool | ms/round | basis |
|---|---:|---|
| weight streaming | 24.1 | 19.1 GiB at ~850 GB/s, unavoidable |
| decode attention | 23.8 | Op microbench: 1.19 ms per layer × 20 attention layers |
| draft path | 3 × 2.0 fixed + 0.012 µs/token | draft sweep slope |
| other device | ~6 | projections, GDN, LM head, launches |
| host | 2.9 | `engine_timing` |

KV bytes per token are measured, not assumed: two `--max-context` allocations (100,032 vs 200,000
tokens) differ by 3.93 GiB of runtime, i.e. 42.2 KB per token, which is exactly 20 attention layers
× 2112 bytes (4 kv heads × 256 head dim × K,V int8 + group-64 scales). This model is a hybrid: most
blocks are GDN/recurrent and keep no KV. At ~850 GB/s the KV floor for one full pass is 9.3 ms at
186K, so **the decode attention runs at ~2.6x its bandwidth floor**: the Op microbench reads
2112 B/token and lands at 294 GB/s before the v1.4.0 port and 325 GB/s after, against ~850 GB/s
achievable. Per-CTA accounting at 186K says why: 389 splits × 4 kv heads = 1556 CTAs, one resident
CTA per SM (93 KB of shared memory), each reading 253 KB in ~55 µs ≈ 4.6 GB/s. The per-tile
pipeline is serialized — dequantize into shared memory, three barriers, tensor-core work — and the
register double buffer only hides the load latency, not the dequant/staging cost. A double-buffered
shared-memory staging layer (e.g. `Bc=32` staged twice) is the obvious next experiment.

## Probes and what they settled

| Probe | Result |
|---|---|
| Draft sweep 1/3/4/5 at 8K/32K/186K | Draft 4 wins everywhere (+9% / +17% / +14% tok/s against draft 3); each draft costs ~2.0 ms fixed + 0.012 µs/token; draft 5 collapses above 32K (118.8 ms/round at 186K) |
| Attention Op microbench (`bench/ops/causal_softmax_attention_bench`) | 1.34 ms -> 1.21 ms per layer at 186K W=4 (−10%), 325 GB/s vs 850 achievable |
| 8K attribution | Host is 5% of the round; `--no-cuda-graph` is net-neutral (it moves 2.6 ms from host to device); weight streaming is 24.1 ms of the 39.5 ms round |
| Concurrency 1 vs 2 at 8K | +21% aggregate throughput, +35% per-request latency (round time is shared, so production's `--max-concurrency 2` is justified) |
| `--kv-dtype k8v4` / `nvfp4` | Engine refuses on Volta: `NVFP4 KV-cache storage is unavailable on Volta`. Closed, not a speed question |
| `--spec dflash2 --draft-tokens 3` | 55.8 ms/round at 8K against MTP's 40.2 — 40% slower, no acceptance advantage |
| `NINFER_MTP_ATTN_WINDOW` isolation | With speculation disabled, window 0 and 8192 produce byte-identical output, so the target model is untouched |

## Landed in v1.4.0

| Change | Effect |
|---|---|
| Port `53f65504` (fp16 activation staging for QPN residual/down projections) | part of −10% Op time |
| Port `2c9e20c9` (NVFP4 residual fused into the QPN epilogue; wider small-T split reduce; draft key window) | part of −10% Op time |
| `--draft-tokens 4` in the deployment | +9% to +17% tok/s at 8K–32K, +14% at 186K |

Drift control: the baseline image was re-run after the candidate and matched its first run within
0.3% (40.2/43.5/62.6 ms vs 40.3/43.5/62.6 ms), so the 7–11% round-time gains are not session drift.

## Next items

### 1. Decode attention at 2.6x its bandwidth floor (up to ~14 ms/round at 186K, +22%)

The measured gap and the per-CTA arithmetic above are the strongest remaining signal. Candidate
changes, in order of expected value: double-buffered shared-memory staging (`Bc=32` staged twice
keeps the 93 KB footprint) so dequant overlaps the previous tile's MMA; a second resident CTA per SM
by cutting shared memory (`K` kept as int8 in shared, converted in registers); a split-policy retune
(more splits only help the tail, not the per-CTA rate). Verify each with the Op microbench first —
it isolates the kernel from everything else and runs in seconds.

### 2. Draft key window as an opt-in

`NINFER_MTP_ATTN_WINDOW=8192` is worth +11% at 186K code and no harm at 8K, but −7% at 32K and −24%
acceptance on Chinese documents at 193K. Keep it off by default; enable it for deployments whose
contexts are consistently beyond ~100K and whose content is code-like.

### 3. Prefill (first token)

186K cold prefill is 341 s (547 tok/s) after the v1.3.0 retune — still the largest user-visible cost
at long context. Nothing has been profiled here; the flash prompt kernel and the GDN/attention
projections are the places to look.

### 4. Product-side backlog

Prometheus metrics, constrained decoding, custom Jinja template support — all exist upstream and
have not been ported.

### 5. Deferred

Tensor parallel across two V100s (the source port reaches 101.9 tok/s at 186K with TP2). Needs a
second V100 in the same host; out of scope until then.

## Publishing

`v1.3.0` is published (master, tag, GitHub release, ghcr.io image). `v1.4.0` is local only: it is
tagged in this tree, built as `ninfer:v1.4.0`, and running in the local deployment; pushing it would
trigger the release workflow. `v1.2.0` stays a local tag because its content is included in v1.3.0;
`wip-a8e212ac` and the `ninfer:v1.3.0b` image are local-only reference material and must not be
pushed.
