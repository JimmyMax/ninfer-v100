# NInfer V100 v1.6.1

Prebuilt **Tesla V100 (sm_70)** binaries, compiled with CUDA 12.8 on Ubuntu 24.04 (glibc 2.39).
The runtime hard-requires compute capability 7.0 — other GPUs are rejected at startup.

A patch release on [v1.6.0](https://github.com/JimmyMax/ninfer-v100/releases/tag/v1.6.0): one build
fix, the dense Gemma 4 31B, and the container image the instructions name.

## Fixed: building from source other than Release

Building the tree in any configuration that keeps the vendored ggml CUDA host helpers —
`CMAKE_BUILD_TYPE=Debug`, `-O0`, `-lineinfo` — failed at the link step with

```text
undefined reference to `ggml_cuda_get_device()'
undefined reference to `ggml_cuda_error(...)'
```

`third_party/ggml-quants` ships the ggml CUDA headers but not the `common.cu` that defines those two
functions, and only `ggml_abort` had a local definition. Release builds linked anyway because `-O3`
drops the unused inline callers that would reference them, so CI and the published binaries never
saw it — report: [#2](https://github.com/JimmyMax/ninfer-v100/issues/2). Both are now defined beside
`ggml_abort` in `src/ops/linear/gguf/ggml_bridge.cu`, and the new
`tests/ops/test_vendor_host_helpers.cpp` links them from a static initializer, so deleting either
definition fails the test build at every optimisation level instead of failing a user's.

**The v1.6.0 release binaries are not affected.** They contain neither the call nor the reference;
only a build that keeps the unused callers needed the definitions.

## Added: the dense Gemma 4 31B

The Gemma 4 row kernels capped `hidden` at 4096 (1024 float4 lanes per block), which the 31B's 5376
does not fit, so a dense 31B artifact stopped with *"gemma4 row kernels: hidden must be whole
float4s, at most 4096"*. Wide-hidden variants of `rms_norm`, `attn_residual` and `ffn_combine` now
cover it, dispatched by `hidden/4 > 1024` — the existing narrow kernels are untouched, so every
artifact that ran before runs exactly as before. The new FP64 oracle
(`tests/models/gemma4/test_wide_rows.cpp`) checks both widths against the same reference, and the
26B forced through the wide path reproduces its narrow-path output token for token.

`Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP` (Q4_K_M blocks, 17.6 GiB, dense, 60 layers,
hidden 5376), one V100-32GB, 4K context, greedy:

| | prefill | decode | MTP acceptance |
|---|---:|---:|---:|
| `--spec mtp --draft-tokens 4` | 674 tok/s | 48.8 tok/s | 43.0% |
| `--spec mtp --draft-tokens 3` | 674 tok/s | 49.7 tok/s | 56.4% |
| no speculation | 674 tok/s | 27.8 tok/s | — |

Convert it with `python3 -m tools.convert --recipe gemma4_gguf --components text,mtp,vision`; draft
three pays on this model because a wider verification batch costs more weight traffic per accepted
token than it saves.

## Changed: the V100 container image

The V100 build is published as **`ghcr.io/jimmymax/ninfer-v100:sm70`** (`:<version>-sm70` pins a
release), built from the new `Dockerfile.sm70`: CUDA 12.8 on Ubuntu 24.04, about 3 GB.

```bash
docker run --rm --gpus all -p 8000:8000 --ulimit memlock=-1 \
  -v "$PWD/models:/models" ghcr.io/jimmymax/ninfer-v100:sm70 \
  ninfer-serve /models/model.ninfer --host 0.0.0.0 --port 8000 --model-id model \
  --max-context 200000 --prefill-chunk 2048 --kv-dtype int8 \
  --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0
```

`latest` stays the `sm_86`/`sm_120a` image, which no V100 can run; on this card use `:sm70`.

## Assets

| File | Contents |
|---|---|
| `ninfer-v100-v1.6.1-linux-x86_64-sm70.tar.gz` | `ninfer` CLI + `ninfer-serve` + `ninfer-perplexity` |
| `ninfer-serve-sm70.tar.gz` | server binary only |
| `SHA256SUMS.txt` | checksums for both |

Everything in the [v1.6.0 highlights](https://github.com/JimmyMax/ninfer-v100/releases/tag/v1.6.0)
holds: 27K-token prefill 1.00k tok/s, decode 78 tok/s, v2 artifacts read in place.

## Documentation

`docs/v100/gemma4-wide-rows-2026-10-16.md` records the wide-row work, including the residual
added twice by writing the normalized activations back too early. `docs/v100/PLAN.md` is the
roadmap with one measurement document per stage.

Apache-2.0, as upstream; model weights keep their own licenses (see `LICENSE` and the README).
