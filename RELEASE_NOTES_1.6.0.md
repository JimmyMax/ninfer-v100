# NInfer V100 v1.6.0

Prebuilt **Tesla V100 (sm_70)** binaries, compiled with CUDA 12.8 on Ubuntu 24.04 (glibc 2.39).
The runtime hard-requires compute capability 7.0 — other GPUs are rejected at startup.

## Read this first: this release is a different line from v1.0.0 – v1.5.0

v1.6.0 is the **first release cut from `main`**, which follows the NInfer-all upstream and adds the
single-card Tesla V100-32GB work. The earlier tags v1.0.0 … v1.5.0 were cut from the previous V100
line; that line now lives on as the `archive/fork-master` branch. It is **not** an ancestor of this
commit, so v1.6.0 is not a continuation of v1.5.0 and the two are not comparable line for line.

| | v1.0.0 … v1.5.0 (`archive/fork-master`) | v1.6.0 (`main`) |
|---|---|---|
| Base | previous V100 line | NInfer-all upstream + the V100-32GB work below |
| Containers | v2 (native vocabulary) | **v2 and v3** |
| v2 artifacts | read natively | read **in place**, no offline conversion and no 24 GB scratch copy |
| GGUF-derived artifacts | not supported — stops with `unknown tensor format: gguf_iq4_xs` | supported through the built-in ggml bridge |
| Default branch, README, docs | — | `main`, with the V100-32GB line stated up front |

## Highlights

One Tesla V100-PCIE-32GB, a 27K-token prompt, int8 KV, MTP with four drafts, greedy, one request.

| Artifact | Container | Weights | Context | 27K-token request |
|---|---|---:|---:|---|
| Qwen3.8-27B `nvfp4` (native profile, production default) | v3 | 19.1 GiB | 254,208 | TTFT **27.2 s**, prefill **1.00k tok/s**, decode **76–78 tok/s** |
| The same family as a **v2** file, read in place | v2 | 22.1 GiB | 200,000 | TTFT **27.3 s**, prefill **998 tok/s**, decode **82.7 tok/s** |
| Qwen3.8-27B `IQ3_S` (GGUF-derived, 18 block formats) | v3 | 12.9 GiB | 400,000 | TTFT **30.8 s**, prefill **885 tok/s**, decode **66–74 tok/s** |

Short-context decode (1K prompt) is **82.3 tok/s**, and MTP accepts 69% of its drafts on the 27K
prompt. `Huihui-…-nvfp4` and `orcarouter-…-nvfp4` (v2, the native family) and
`Gemma4-26B-A4B-…-MTP` (v3, GGUF-derived) load as well.

The container version does not affect speed — the same weights measure 998 tok/s as a v2 file and
997 tok/s as the upgraded v3 file. What the numbers track is the weight recipe: the native
NVFP4 / FP8 / group-wise formats get the prepacked and CUTLASS routes, while GGUF-derived artifacts
run through the ggml bridge with per-tensor block dequantization.

## What changed

Six changes carry the numbers above. All of them are load-time or operator routing, none changes a
numeric contract, the v3 path is byte-identical to before, and each landed with the upstream FP64
oracles green (`ctest`) plus a 27K end-to-end A/B.

- **Volta QPN** — quadpair-split-N prepacked GEMMs for nvfp4 and fp8, with wide T on CUTLASS over a
  dequantized staging plane: the projection routes go from 85 GB/s to 480–620 GB/s.
- **GDN prefill on the recurrent kernel** — chunked and two-stage are a regression on sm_70: 2.9–3.3×
  for T = 1024…8192, and 3.75 s → 2.18 s out of the 27K prefill.
- **Flash-attention prefill kernel** — paged-to-contiguous staging plus an sm_70 `mma.sync` kernel:
  41.1 ms → 25.1 ms for the same shape, and 27K TTFT 30.1 s → 27.2 s.
- **Speculative-decode configuration** — `--ngram-draft-tokens 0` and no `--adaptive-mtp`: the
  default fifteen n-gram drafts dilute the MTP acceptance rate. 27K decode 64.2 → 78.3 tok/s with
  byte-identical greedy output.
- **FP8 SwiGLU wide T on an FP32-output CUTLASS path** — it used to fall into a four-token-chunk
  SIMT ladder: mixed-quantization prefill 508 → 998 tok/s.
- **v2 artifacts read in place** — the directory is normalized to the v3 shape in memory and the
  weights are read where they are, so a v2 file and its upgraded copy give the same output and the
  same numbers.

## Assets

| File | Contents |
|---|---|
| `ninfer-v100-v1.6.0-linux-x86_64-sm70.tar.gz` | `ninfer` CLI + `ninfer-serve` + `ninfer-perplexity` |
| `ninfer-serve-sm70.tar.gz` | server binary only |
| `SHA256SUMS.txt` | checksums for both |

Run the server on the V100 with:

```bash
./ninfer-serve /path/to/model.ninfer --host 0.0.0.0 --port 8000 --api-key sk-xxx \
  --max-context 200000 --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 \
  --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0
```

## Documentation

`docs/v100/PLAN.md` is the roadmap with one measurement document per stage. The design records
behind this release: `docs/v100/v2-direct-read-2026-10-16.md`,
`docs/v100/fp8-swiglu-wide-t-2026-10-16.md`, `docs/v100/volta-flash-prefill-2026-10-15.md`,
`docs/v100/gap-closure-2026-10-15.md` and `docs/v100/gdn-recurrent-and-prefill-gap-2026-10-15.md`.

Apache-2.0, as upstream; model weights keep their own licenses (see `LICENSE` and the README).
