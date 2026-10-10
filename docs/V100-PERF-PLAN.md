# V100 performance plan

Working plan for continued single-GPU performance work on the V100 build. Update it as items land
or are rejected; it is the entry point for "what next" alongside
[the V100 port notes](v100.md) and [the build recipe](V100-BUILD.md).

## Current state

Released locally as **v1.3.0** (commit `9e2a9bff`, image `ninfer:v1.3.0`), running in the local
deployment. It contains:

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

Everything needed to reproduce or extend the numbers:

| Piece | Location |
|---|---|
| Prompt generator (code / Chinese documents, exact token counts) | `bench/v100/ctx_prompt.py` |
| Decode / first-token client (reads server `timings`) | `bench/v100/decode_bench.py` |
| Multi-context matrix runner (image × 8K/32K/186K/193K) | `bench/run_ctx_matrix.sh` |
| Long-context runner | `bench/run_longctx.sh` |
| Local bench compose (image override, GPU 1) | `bench/docker-compose.bench.yml` |

Rules that made results trustworthy, and that any new perf item must follow:

1. **Prefer an in-image switch for A/B.** The attention kernel has `NINFER_SM70_ATTN_V2=0`; use it
   rather than two images so both variants see the same session state.
2. **Always include a drift control.** Re-run the baseline image after the candidate, or interleave.
   Long-context decode drifts by ~1%, first token by ~5%, and short-request latency by far more.
3. **Never benchmark while something else uses the host.** A concurrent `nvcc` build inflated
   tokenize measurements by 2×; a background image export did the same.
4. Long-context numbers need the GPU to itself: stop the deployment container first
   (`docker compose -f docker-compose.yml stop llama-cpp-text`).

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

## Next items

### 1. Close the rest of the gap to the tpx port (est. +9% to +13% at long context)

We measure 51.3 / 38.1 decode tok/s at 186K/193K; the source port reports 56 / 43 with its full set
of commits. Remaining candidates, in the order they should be attempted:

- `53f65504` — stage QPN residual/down-projection activations as fp16 instead of bf16.
- `2c9e20c9` — fused NVFP4 residual epilogue, faster split reduce, MTP key-window knob.
- Linear dispatch retunes for nvfp4/fp8/w8/q4 configs. Pure tuning; A/B each one.

Caveat: the two `nvfp4_*` files involved have diverged from that baseline (our swiglu epilogue
fusion), so these are manual ports, not clean applications.

### 2. Profile what is left at 186K (no estimate yet)

Decode is now ~63 ms/round at 186K. Attention is no longer the dominant cost; find the next one
with kernel/nsys profiling before writing code. Candidate suspects: NVFP4 weight streaming, MTP
draft rounds, LM head, per-round host work.

### 3. Short-context path

8K only improved 3.7% (code) and 7.2% (Chinese), well below the long-context gains. Something
other than attention dominates there — likely fixed per-round overhead, sampling, or scheduling.
Profile before changing code.

### 4. Concurrency

Every number above is single-request. The deployment runs `--max-concurrency 2`; nothing is known
about how attention, scheduling, or the context cache behave with two active requests.

### 5. Product-side backlog (independent of performance)

Prometheus metrics, constrained decoding, custom Jinja template support. All exist upstream and
have not been ported.

### 6. Deferred

Tensor parallel across two V100s (the source port reaches 101.9 tok/s at 186K with TP2). Needs a
second V100 in the same host; out of scope until then.

## Publishing

Local commits and tags are intentionally unpushed: `v1.2.0` (correctness fixes, CI, materialization
planner) and `v1.3.0` (attention). Push once the deployment has run for a while without issues.
`wip-a8e212ac` and the `ninfer:v1.3.0b` image are local-only reference material and must not be
pushed.
