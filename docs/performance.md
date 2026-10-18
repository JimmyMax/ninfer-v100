# Performance

Where the measurements live, and the measured comparisons that choose a configuration. Every table
names its card, artifact and settings; numbers from different pages were taken on different hosts
and builds and are not read against each other.

| Page | What it measures | Hardware |
|---|---|---|
| [Reference measurements, September 2026](performance/reference-2026-09.md) | Ternary Bonsai 2 27B and Qwen3.8-27B at full context: decode and prompt time by depth, drafters, largest context per card, needles, several requests at once, what the device profiles change, and the October re-check | RTX 3090, 4090, 5090, PRO 6000 |
| [Tesla V100, October 2026](performance/tesla-v100-2026-10.md) | a `70` build on 16 GB: what fits, engine and server rates of Ternary Bonsai 2, the Qwen3.8-27B releases, Flash-Next and Gemma 4, perplexity against other cards, llama.cpp on the same card | Tesla V100-SXM2-16GB |
| [Gemma 4 on a Tesla V100-32GB](performance/gemma4-v100-32gb-2026-10.md) | the dense 31B and the 26B-A4B on one 32 GB card: prefill, decode and MTP acceptance with and without speculation, what fits the card | Tesla V100-PCIE-32GB |
| [Qwen3.8-Flash-Next](qwen3-8-flash-next.md#measurements) | every expert placement of the GSQ-RCO releases, MTP, perplexity against llama.cpp, AIME and GPQA | RTX 3090, 4090, 5090, PRO 6000, L40S |
| [GGUF block formats](gguf.md#measurements) | the Qwen3.8-27B GSQ-RCO IQ3_S release against the official artifact: perplexity, reasoning benchmarks, speed | RTX 3090, 4090, 5090 |
| [Quality trades](maintainer/quality-trade-experiments.md) | what each precision option costs in perplexity and buys in memory or speed | RTX 3090 |
| [Configuration calculator](config-calculator.html) | which context, KV format and speculation fit an RTX 3090, from measured sizes | RTX 3090 |

The tools behind them: [`tools/bench/refbench.py`](../tools/bench/README.md#reference-measurements-refbenchpy)
for the reference tables, the [serving corpus runner](../tools/bench/README.md#serving-corpus-benchmark),
`ninfer_bench` for the Engine without HTTP and the Op benchmarks for single operations
([benchmarks](../bench/README.md)), and `ninfer-perplexity` for quality ([perplexity](perplexity.md)).
An Op benchmark establishes an Op result, not an end-to-end one; a whole request does.

## KV formats on Qwen3.8-27B

One RTX 3090, the official `groupwise-int` artifact, September 2026. Perplexity is
`ninfer-perplexity --quick` on `ninfer-ppl-1m-v1` (context 4,096, stride 2,048, 261,167 scored
tokens); decode is 128 steps after a 32,768-token prefill, no speculation, the mean of two runs.

| KV format | Bytes per token | Perplexity | Change against `bf16` | Decode at 32K, tok/s |
|---|---:|---:|---:|---:|
| `bf16` | 65,536 | 4.342517 | — | 31.93 |
| `int8` | 33,792 | 4.342425 | −0.002% | 33.63 |
| `fp8` | 33,024 | 4.344724 | +0.051% | 30.34 |
| `rk8v4` | 26,112 | 4.346413 | +0.090% | 33.17 |
| `k8v4` | 25,728 | 4.347258 | +0.109% | 28.90 |
| `nvfp4` | 18,432 | 4.352201 | +0.223% | 29.86 |
| `rk4v4` | 17,920 | +0.214% against `int8` | | within 1% of `rk8v4` |

The `rk4v4` row was measured in a later session (`int8` 4.343155 there) and is stated against that
session's `int8`. On Ternary Bonsai 2, `rk4v4-e8` moves quick-corpus perplexity from 5.631 to 5.650
and `rk2v4-e8` to 5.820 against `rk8v4`; a simulation of `rk2v4-e8`'s keys on Qwen3.8-27B cost
+3.08%. [Context and memory](cli.md#context-and-memory) turns these into recommendations.

## Speculative backend by concurrency

One RTX 3090, Qwen3.8-27B, INT8 KV, thinking off, greedy; aggregate decode tok/s through the server,
decode time from the request log:

| Requests at once | MTP, 3 drafts | DFlash2, 7 drafts | Tokens per round, MTP / DFlash2 |
|---:|---:|---:|---|
| 1 | 135.0 | 187.1 | 3.51 / 5.57 |
| 2 | 238.2 | 313.5 | 3.51 / 5.82 |
| 4 | 387.4 | 406.2 | 3.54 / 5.63 |
| 8 | 522.8 | does not fit | 3.53 / — |

Speculation pays because a multi-column round costs about one sweep of the weights; batching
amortizes that sweep already, so the lead shrinks as requests are added while acceptance holds.
DFlash2's draft weights and its refusal of `--lm-head-q6` are what stop it at eight lanes and cost
it about 65K tokens of context against MTP on a 24 GB card. The best DFlash2 draft count depends on
the work: over a reasoning, a code and a summarization prompt seven was best on average, but the
reasoning prompt kept improving to twelve while summarization fell off there.

## MTP draft count by content

One RTX 3090, Qwen3.8-27B, `rk4v4` KV with `--lm-head-q6 --embedding-q4 --gdn-state-fp16`,
greedy, 512 generated tokens, one stream; decode tok/s with tokens per round in brackets:

| `--draft-tokens` | edit code (copies the prompt) | write new code | explain a concept | short story |
|---:|---:|---:|---:|---:|
| none | 47.3 | 47.5 | 47.5 | 47.6 |
| 3 | 141.9 (3.9) | 118.2 (3.3) | 102.1 (2.8) | 77.1 (2.1) |
| 5 | 177.7 (5.7) | 124.8 (4.0) | 95.6 (3.1) | 75.1 (2.4) |
| 7 | 226.0 (7.4) | 130.6 (4.3) | 103.9 (3.4) | 67.7 (2.2) |
| 9 | 226.4 (8.8) | 129.9 (5.0) | 91.3 (3.5) | 56.3 (2.2) |
| 15 | 262.6 (12.5) | 114.7 (5.4) | 75.1 (3.5) | 47.6 (2.2) |

Each round verifies K+1 columns and runs K draft steps whether or not the drafts survive, so a
larger K pays only where the head keeps guessing right ([choosing a draft
count](cli.md#choosing-a-draft-count)).

## Determinism

Every configuration reproduces its own output exactly. Configurations do not reproduce each other:
a verification evaluates K+1 columns in one pass where plain decode evaluates one, a batch
evaluates several requests together, and the different reduction orders can flip a near-tied
argmax. MTP and DFlash2 outputs differ from each other and from plain greedy decoding, and DFlash2's
output varies with the draft count, without a measurable perplexity cost
([batch composition](device-profiles.md#batch-composition)).
