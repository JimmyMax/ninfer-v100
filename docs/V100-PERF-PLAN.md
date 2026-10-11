# V100 performance plan

Working plan for continued single-GPU performance work on the V100 build. Update it as items land
or are rejected; it is the entry point for "what next" alongside
[the V100 port notes](v100.md) and [the build recipe](V100-BUILD.md).

## Current state

Local release **v1.5.0** (tag `v1.5.0`, image `ninfer:v1.5.0`) is what the deployment runs. It stacks
the v1.3.0 attention rewrite, the v1.4.0 decode-side ports and the v1.5.0 PV epilogue change on top
of the earlier correctness ports:

| Change | Effect |
|---|---|
| Volta D256 flash prompt retune (64-column Q tile) | first token −26% to −29% at 186K–193K |
| INT8 small-T decode attention v2 (key-split 8-warp `Bc=64`, `byte_perm` int8→fp16 dequant) | decode +37% to +47% at 186K–193K, +4% to +17% at 8K–32K |
| v1.4.0 ports: fp16 activation staging for QPN residual/down projections, NVFP4 residual fused into the QPN epilogue, wider small-T split reduce | decode round −7% to −11% |
| v1.5.0: chained fp16 PV accumulator in the decode attention kernel | decode round −1.0% (8K) to −4.6% (186K) |
| Deployment switch to `--draft-tokens 4` | tok/s +9% to +17% (8K–32K), +14% (186K) |
| Earlier upstream ports: DFlash prefill binding + cleanup synchronization, spin sync, materialization planner | correctness fixes; no measurable throughput effect on V100 |

v1.3.0 is published on origin (tag, GitHub release, ghcr.io image). v1.4.0 and v1.5.0 are local tags
and local images only; pushing them would trigger the release workflow.

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
| Prefill decomposition | 186K prefill is 188 s device + 149 s host-exposed; instrumented counters put ~100% of the prefill path inside the model's per-chunk prefill execution (engine helpers ~0), with one device wait at the end of the chunk |

## Landed in v1.4.0

| Change | Effect |
|---|---|
| Port `53f65504` (fp16 activation staging for QPN residual/down projections) | part of −10% Op time |
| Port `2c9e20c9` (NVFP4 residual fused into the QPN epilogue; wider small-T split reduce; draft key window) | part of −10% Op time |
| `--draft-tokens 4` in the deployment | +9% to +17% tok/s at 8K–32K, +14% at 186K |

Drift control: the baseline image was re-run after the candidate and matched its first run within
0.3% (40.2/43.5/62.6 ms vs 40.3/43.5/62.6 ms), so the 7–11% round-time gains are not session drift.

## Next items

### 1. Port the remaining single-GPU changes from the sibling project

An independent V100 port of this codebase carries two unported changes that matter on one card,
from two commits, the second depending on the first:

| Source | Change | Their measurement |
|---|---|---|
| `2b3f16cd`, `nvfp4_volta_qpn_gemm.cuh` | batch eight group loads before decoding any of them | single card 5120x17408 T=4: 96.0 -> 88.4 us |
| `9913bc22`, same file | double-buffered four-group prefetch on top of that; unroll count behind `NINFER_NVFP4_QPN_UNROLL` | 186K round 31.5 -> 30.9 ms, output bit-identical; the 8-group buffered variant spills and is unusable |

Both are decode-side kernel changes with claimed bit-identical output, so the verification order below
is short: the kernel shape measured directly, then the drift-controlled ladder.

**The attention hunk from the same commit was attempted and abandoned.** It switches the PV epilogue to
`mma.m8n8k4...f32.f16.f16.f32` accumulating straight into the fp32 register block. Ported behind a
three-way `NINFER_SM70_ATTN_V3` switch (0 = original per-group fp16, 1 = v1.5.0 chained fp16,
2 = fp32 mma) it measured 186K W=4 at 1092 us against 1115 us for v1.5.0 - about 2% of the kernel,
roughly 1% of a decode round - and it did not survive qualification:

- with the existing per-lane `alpha` rescale, mode 2 fails the public-contract oracle: the fp32 mma
  writes 8 elements per thread whose rows are selected by `i & 2`, so the online-softmax rescale has
  to be per element;
- with that per-element rescale applied to every mode, mode 0 fails the same oracle, so the rescale
  semantics would have to be per mode and each state re-qualified;
- the fast interior-tile load path from the same commit is what broke the oracle first, while the
  rest of the port had already been reverted. Its page-relative arithmetic is equivalent on paper to
  the generic path, but it needs its own qualification before it can be used.

Verdict: ~1% end-to-end is not worth three extra verified states. v1.5.0 stays as shipped and the
QPN batching below is the next thing to port.

#### Next session: start here

The remaining change is a software pipeline over our chunk loop rather than a mechanical port,
because this kernel already batches its four groups per chunk: per `st` it issues 32 code bytes (two
`uint4`) and one 32-bit scale word, then decodes the four groups out of registers. What the source
commit shows on the equivalent loop is that those loads still stall (Nsight: 70% DRAM throughput with
long scoreboard as the top stall) and that keeping the *next* chunk in flight while the current one
is decoded is what removes them.

1. `src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh`, the `st` loop at the top of the kernel body:
   prologue-load chunk `st0`, then per iteration issue the loads for `st + 1` before the four-group
   decode and mma of `st`, and rotate the buffered pair (two `uint4` + one `std::uint32_t`). Two live
   chunks is all it needs; the source's 8-group buffered variant spills and is unusable.
2. Leave the activation loads (`raw0`/`raw1`) where they are: they are already known per chunk and
   consumed immediately by the mma.
3. Gate it with `NINFER_NVFP4_QPN_PIPE=0` restoring the current loop, so the A/B ladder and any later
   bisect remain possible.
4. Kernel numbers first: `ninfer_linear_bench --qtype nvfp4` already carries the affected shape
   (`27b.mtp_down`, 5120x17408, continuous T). Add `ninfer_linear_bench` to the target list in
   `Dockerfile.opt.local` so the iteration image can run it. Expect ~-8% on the kernel (source:
   96.0 -> 88.4 us at T=4) and ~-2% on a decode round at 186K.
5. Then the usual evidence: the ladder (baseline -> candidate -> baseline, draft 4, 8K / 32K / 186K),
   a bit-identical check (this change must not alter accumulation order, and the source reports
   identical output), and `ninfer_qwen3_6_runtime_mechanisms_test` plus the DFlash2 real-model test
   (`7 1 1 2`, skip code 77) on the candidate image.

### 2. Decode attention: what is left after v1.5.0

After the v1.5.0 epilogue change the remaining kernel time is QK^T, four barriers per tile, the
softmax, the P-tile stores and the address math. The two candidates worth measuring are a second
resident CTA per SM (needs the shared-memory footprint cut from 93 KB, e.g. by keeping K as int8 in
shared and converting in registers) and a barrier reduction per tile. Both are rewrites of the same
delicate kernel, and the end-to-end ceiling is modest: at 8K the attention Op is only ~6% of a decode
round.

### 2b. Decode attention ablation (v1.5.0), for reference

An ablation build settled what limits the INT8 small-T decode attention. Template variants that
skip one part each (`NINFER_SM70_ATTN_ABLATE=1..4`, diagnostic only because every non-zero value
produces wrong numbers) give, at 31.5K with W=4:

| variant | median | delta |
|---|---:|---:|
| exact | 305 us | - |
| skip the PV mma loop | 226 us | -26% |
| skip the QK^T mma loop | 248 us | -19% |
| skip the tile loads | 280 us | -8% |
| skip the int8 dequant | 287 us | -6% |

So the kernel is **instruction-issue bound, dominated by the PV epilogue**, and the earlier framing
of "3.9x off the KV bandwidth floor" (or "2.6x") was misleading: the tile loads are only 8% of the
kernel, so a bandwidth-roofline comparison does not describe what a fix could recover.

Landed in v1.5.0: the PV loop now chains one fp16 accumulator per output chunk across the whole
64-key tile instead of starting a fresh accumulator per 8-key group and converting each result
(4 conversions and 8 float adds per 4 mma). That is the fp16-accumulate form of `mma.m8n8k4`, which
issues at twice the fp32 rate on Volta, and it cuts the epilogue instruction count 8x. The 64-key
partial sum now rounds once in fp16; the public-contract oracle test passes on both paths
(`NINFER_SM70_ATTN_V3=0` restores the old epilogue for in-image A/B).

| measurement | before | after |
|---|---:|---:|
| Op microbench 8K W=4 | 108.5 us | 94.2 us (-13%) |
| Op microbench 31.5K W=4 | 304.1 us | 251.9 us (-17%) |
| Op microbench 186K W=4 | 1403.9 us | 1137.7 us (-19%) |
| decode round 8K (draft 4) | 38.7 ms | 38.3 ms (-1.0%) |
| decode round 32K | 41.5 ms | 41.0 ms (-1.2%) |
| decode round 186K | 62.9 ms | 60.0 ms (-4.6%) |

Drift control: the baseline ladder was repeated after the change and matched its first run exactly
(8K 38.7 ms, 32K 41.5 ms, 186K 62.9 ms).

What is left is spread thinly: after this change the remaining kernel time is QK^T (~19% before,
already f32-accumulate with two loads per mma), four barriers per tile, the softmax, the P-tile
stores and the address math. The two candidates worth measuring if this line is reopened are a
second resident CTA per SM (needs the shared-memory footprint cut from 93 KB, e.g. by keeping K as
int8 in shared and converting in registers) and a barrier reduction per tile. Both are rewrites of
the same delicate kernel, and the end-to-end return above shows why the ceiling is modest: at 8K the
attention Op is only ~6% of a decode round.

### 3. Draft key window as an opt-in

`NINFER_MTP_ATTN_WINDOW=8192` is worth +11% at 186K code and no harm at 8K, but −7% at 32K and −24%
acceptance on Chinese documents at 193K. Keep it off by default; enable it for deployments whose
contexts are consistently beyond ~100K and whose content is code-like.

### 4. Prefill: host-exposed serial time is 44% of the first token

Measured on v1.4.0 (186K code prompt, `--prefill-chunk 2048`): `prompt_ms` 337 s, and the request
log splits it into **device wait 188 s + host submit 149 s** (the remainder is post and boundary
bookkeeping). The host phase is real CPU work, not idle blocking: reading `/proc/<pid>/stat` across a
running prefill shows **100.4% of one core** with two threads carrying ~72 s and ~19 s of that time.
The two phases are attributed serially by the engine — the sum lands on the prefill total in every
configuration measured — but they overlap in reality, because the host launches each layer while the
device still executes the previous ones; the device wait is the residual after the last launch.

| configuration | prefill | units | device | host submit | host per token |
|---|---:|---:|---:|---:|---:|
| 186K, chunk 2048 | 337.1 s | 94 | 188.0 s | 149.2 s | 0.80 ms |
| 186K, chunk 4096 | 335.7 s | 48 | 168.2 s | 167.6 s | 0.90 ms |
| 186K, chunk 8192 | 434.2 s | 25 | 182.9 s | 251.4 s | 1.35 ms |
| 32K, chunk 2048 | 32.1 s | 18 | 17.8 s | 14.3 s | 0.45 ms |

Larger chunks make the host phase worse (10.1 s per 8192-token chunk against 1.6 s per 2048-token
chunk), so the per-chunk cost grows with chunk size *and* with the accumulated prefix — an O(n²)
shaped host operation at fixed chunk size. Ruled out by A/B at 32K (all 31.9–32.1 s): sync mode
(`NINFER_CUDA_SYNC=blocking`), prefix reuse (`--no-prefix-reuse`), CUDA graphs (`--no-cuda-graph`),
host KV mirror size (`--host-kv-mib 128`), checkpoint slots (`--host-state-slots 1
--device-state-slots 1`), the draft path (server started without `--spec`), and vision (`--vision`
removed). The single-request `ninfer` CLI reproduces it exactly (31.8 s for the same 32K prompt), so
it lives in the shared Program prefill path, not in the server's HTTP or context-cache layers.

Next step: the host cost sits in the model's per-chunk prefill execution, not in the engine's
bookkeeping. Instrumented counters (temporary probe, not merged) over a 32K + 186K pair give
`card_prefill_chunk` ~142 s against `advance_prefill` ~138 s, while `state_selectors`,
`settle_state_fork`, `ensure_kv_mapped`, `commit_sequence_kv` and `copy_tail` are all ~0.000 s;
`TextContext::prefill_impl` contains a single `begin_wait()` at its very end, so the whole per-layer
launch sequence is attributed to the host phase. In other words the host time is CPU-side kernel
submission plus per-layer host setup inside the prefill execution, and it grows with the number of
attention splits (which follows the prefix) and with the chunk size. The natural fix is to reduce
that submission work per chunk — graph-captured prefill, or fewer and larger launches — which is an
architectural change in the prefill path, not a local tweak. `--no-cuda-graph` changes nothing
today, which is consistent with prefill running eagerly.

Correction to an earlier revision of this document: a perf profile whose addresses were resolved
naively pointed at `ProgramImplCore::prepare_materialization`. Instrumentation disproves it — that
function runs twice per request, walks 0 pages, and takes ~0 s. The perf `--symfs` route is the one
to use if this is revisited.

### 5. Product-side backlog

Prometheus metrics, constrained decoding, custom Jinja template support — all exist upstream and
have not been ported.

### 6. Deferred

Tensor parallel across two V100s (the source port reaches 101.9 tok/s at 186K with TP2). Needs a
second V100 in the same host; out of scope until then.

## Publishing

`v1.3.0` is published (master, tag, GitHub release, ghcr.io image). `v1.4.0` is local only: it is
tagged in this tree, built as `ninfer:v1.4.0`, and running in the local deployment; pushing it would
trigger the release workflow. `v1.2.0` stays a local tag because its content is included in v1.3.0;
`wip-a8e212ac` and the `ninfer:v1.3.0b` image are local-only reference material and must not be
pushed.
