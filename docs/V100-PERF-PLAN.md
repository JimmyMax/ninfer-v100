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

# test binaries only (small local export, reuses the build stage)
docker build -f Dockerfile.build-runtime.local --target testexport --output type=local,dest=/tmp/tests .

# numeric + regression checks
docker run --rm --gpus '"device=1"' -v /tmp/tests/tests:/tests ninfer:<tag> /tests/ninfer_softmax_attention_test
docker run --rm --gpus '"device=1"' -v /tmp/tests/tests:/tests ninfer:<tag> /tests/ninfer_qwen3_6_runtime_mechanisms_test
```

## Where the round time goes (v1.3.0)

Decode throughput is `(1 + 3 × accept) / round`. Round time depends only on context length, not on
the content, while `accept` ranges from 0.45 (Chinese documents) to 0.80 (code) **at the same round
time**: at 7.4K, `ms/round` is 39.5 for both kinds, and tok/s is 85 for code against 60 for Chinese.
Acceptance and round time are therefore two independent levers, and `ms/round` — not tok/s — is the
invariant to compare across runs, images, or machines.

| | 7.4K | 31.5K | 186K |
|---|---:|---:|---:|
| round (ms), v1.3.0 kernel | 39.5 | 43.0 | 63.0 |
| round (ms), previous kernel (`NINFER_SM70_ATTN_V2=0`) | 41.0 | 49.3 | 89.9 |
| accept, code / Chinese | 0.79 / 0.51 | 0.77 / 0.49 | 0.68–0.80 / 0.45–0.50 |

Context-proportional cost is `(63.0 − 39.5) / 179K = 0.131 µs/token`, down from 0.267 before the v1.3.0 kernel. Two pools are unexplained by that measurement:

1. **~15 ms/round is context-independent** (38% of the 7.4K round, 24% of the 186K round). Weight
   streaming accounts for ~24 ms of the 39.5 ms round (19.1 GiB at ~850 GB/s), KV for ~0.2 ms at
   7.4K; the rest is launch, host, and fixed draft work.
2. **The context-proportional cost is 3.9x the KV bandwidth floor.** This is a hybrid model: only the
   full-attention blocks keep a KV cache, and the pool fits inside the logged 10.9 GiB runtime,
   which bounds those blocks at ≤14 (4 kv heads × 256 head dim × K,V × 1 byte int8 = ≤28 KB per
   token). At ~850 GB/s that floor is **0.034 µs/token**, a quarter of the measured 0.131. So the
   growth is not KV bandwidth, and it is either a latency/page-overhead-bound decode attention
   kernel (up to ~18 ms/round recoverable at 186K, +29%) or every MTP draft step streaming the
   context KV again (3 drafts ≈ 3x the traffic), which would put the same cost in the draft path.

Two cheap measurements separate those cases and should run before any kernel work:

- `--draft-tokens 1` vs `3` vs `5` at 186K in one image: the round-time slope per extra draft token
  is the draft path's context cost. A slope near 0.034 µs/token puts the cost in attention.
- `bench/ops/causal_softmax_attention_bench` (needs `-DNINFER_BUILD_BENCHMARKS=ON`) at 8K/32K/186K:
  the kernel's own time against the same floor, with no draft path in the way.

## Next items

### 1. Attribute the two pools above (up to +29% at 186K, up to +38% at 8K)

Run the two measurements, then decide. An nsys trace of one 8K decode round attributes the ~15 ms
fixed pool directly (8K needs no long prefill, so the loop is fast). Likely suspects: per-round
host/launch work, CUDA-graph coverage of the draft steps, sampling and logits handoff, KV append.
This is where the remaining tpx commits (`53f65504` fp16 staging, `2c9e20c9` fused residual epilogue)
would act, so measure first and then pick from them.

### 2. Acceptance is half of tok/s (config-only first)

Chinese documents accept at ~0.47 against code's ~0.75, which is a 40% tok/s difference at identical
round time. Sweep `--draft-tokens` (2–6) and the MTP key-window knob from `2c9e20c9`; treat `accept`
and `ms/round` as separate reported quantities in every benchmark.

### 3. KV dtype: real bytes, wrong kernels today

`--kv-dtype k8v4` (−25% KV bytes) and `nvfp4` (−50%) are supported, but the Volta small-T v2 kernel
only handles INT8 group-64; the other dtypes fall back to the older paths (~+27 ms/round at 186K),
so they lose more than they save today. Extending the v2 kernel to k8v4 keeps the fast path and
removes a quarter of the KV traffic — worth it at long context, but it is kernel work, not a flag.

### 4. Concurrency

Every number here is single-request, while the deployment runs `--max-concurrency 2`. The
bandwidth-bound part will not double, but the ~15 ms fixed pool may overlap. Nothing is known yet.

### 5. Product-side backlog

Prometheus metrics, constrained decoding, custom Jinja template support — all exist upstream and
have not been ported.

### 6. Deferred

Tensor parallel across two V100s (the source port reaches 101.9 tok/s at 186K with TP2). Needs a
second V100 in the same host; out of scope until then.

## Publishing

`v1.3.0` is published: master, the tag, a GitHub release with the long-context numbers, and the
ghcr.io image. `v1.2.0` stays a local tag because its content is included in v1.3.0; `wip-a8e212ac`
and the `ninfer:v1.3.0b` image are local-only reference material and must not be pushed.
