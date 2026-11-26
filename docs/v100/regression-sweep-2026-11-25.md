# Regression sweep after the upstream perf port (2026-11-25)

Scope: the merge of 12 upstream perf commits + sm_86/89 build fixes into `inkling2`/`main`
(`a738ddd6c`). Goal: prove no regression across every artifact format family, on both cards.

Method: each artifact loaded fresh (`--max-context 8192 --kv-dtype int8`), correctness probe
(`The capital of France is`, greedy, 8-16 tokens), then 3 timed 400-token generations
(temperature 0.7, "Write a detailed essay about the history of mathematics."), decode tok/s from
the engine's request log. OLD = the `ninfer:1.7.2` release binary; PRE = `inkling2` before the
perf port (sm_89 build); NEW = merged code (sm_70 on V100, sm_86;89 on the 4080S).

## V100 (sm_70), NEW binary

| # | Artifact | Format family | Correct | Decode |
|---|---|---|---|---|
| 1 | Inkling-2-Mini-CHiPS-UD-Q4_K_XL (production) | gguf_nvfp4 experts + gguf_q8_0 dense | Paris ✓ | 62.9 tok/s |
| 2 | Inkling-2-Mini-UD-Q3_K_XL | IQ2_XS/IQ2_S/IQ3_XXS/Q3_K/IQ3_S + 12 NVFP4 layers | Paris ✓ | 63.3 tok/s |
| 3 | WaveCut-Inkling-2-Mini-UD-IQ4_XS (upstream artifact) | gguf IQ4_XS/Q8_0/Q6_K/Q5_K + NVFP4 | Paris ✓ | 62.8 tok/s |
| 4 | Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3 | gguf IQ3_S dense | Paris ✓ | 42.4 tok/s |

Baseline: pre-port plain decode measured 15.3 ms/token (65 tok/s) via phase timing; the three
Inkling artifacts land at 62.9-63.3 tok/s including prefill and first-token cost — within noise.
No regression on V100.

### V100 full artifact sweep (2026-11-26, NEW sm_70 binary)

Every artifact family on the card, same harness as the 4080S sweep
(`/tmp/ninfer-serve-70-FINAL`, 8k ctx, concurrency 1, W4A4/swift with `--nvfp4-native`,
Flash-Next with `--expert-residency host`):

| Artifact | Type | Spec | Probe | Decode |
|---|---|---|---|---|
| Gemma4-31B | dense | MTP4 | Paris ✓ | 47.8 |
| Gemma4-26B-A4B | MoE | MTP2 | Paris ✓ | 161.9 |
| GSQ IQ3_S | Qwen MoE | plain | Paris ✓ | 42.4 |
| Huihui Q4_K_M | Qwen MoE | MTP4 | Paris ✓ | 54.3 |
| Inkling-2 Mini | Inkling MoE NVFP4 | DSpark | ✓ | 70.0 |
| W4A4 DFlash2 | Qwen W4A4 | MTP4 | Paris ✓ | 65.0 |
| swift nvfp4 DFlash2 | Qwen NVFP4 | MTP4 | Paris ✓ | 67.1 |
| Ternary-Bonsai (plain, no MTP in artifact) | Qwen PTQ1_0 | plain | Paris ✓ | 69.8 |
| Ternary-Bonsai **+ MTP** (re-converted, 2026-11-26) | Qwen PTQ1_0 | MTP4 | Paris ✓ | **82.6** (29% accepted) |
| Flash-Next IQ3_S | Flash-Next | MTP4 + host RAM | Paris ✓ | 33.6 |

9/9 pass, no regression on sm_70. The MTP companion was converted from the official
`-mtp.gguf` (MTP block as `blk.64.*`, Q4_0; the PTQ1_0 text layers come from the plain GGUF) —
see the note under Findings for the converter fix that made `--source mtp=` work from GGUF.

## RTX 4080 SUPER (sm_89)

| # | Artifact | Binary | Correct | Decode |
|---|---|---|---|---|
| 5 | Huihui-Qwen3.8-27B-Q4_K_M | OLD 1.7.2 | Paris ✓ | 39.2 tok/s |
| 5 | same | PRE (no perf) | Paris ✓ | 37.4 tok/s |
| 5 | same | NEW merged | Paris ✓ | 37.4 tok/s |
| 6 | orcarouter-Qwen3.8-27B-nvfp4-NInfer (**v2**) | NEW, `--nvfp4-native` | Paris ✓ | 33.7 tok/s |
| 6 | same | without the flag | garbage — same root cause as #8 | — |
| 7 | Gemma4-26B-A4B-QAT-MTP | OLD 1.7.2 | degenerate repeat on both | 197 tok/s |
| 7 | same | NEW merged | degenerate repeat on both | 186 tok/s |
| 8 | Qwen3.8-27B-W4A4-MTP-DFlash2 | OLD 1.7.2, correct flags | Paris ✓ | 71.8/63.8/70.6 tok/s (acc 40.5/33.5/39.3%) |
| 8 | same | NEW merged, correct flags | Paris ✓ | 69.3/65.0/65.6 tok/s (acc 37.8/34.7/36.0%) |

## Findings

1. **The perf port introduces no regression.** Every OLD↔NEW gap is reproduced exactly by
   PRE↔NEW (no perf): Q4_K_M 37.4 = 37.4, so the port is neutral on Qwen dense GGUF.
2. **A pre-existing fork↔1.7.2 gap of about -5% on Qwen dense** (39.2 → 37.4 tok/s on the 4080S)
   and Gemma4 (197 → 186). It exists in PRE, i.e. it comes from the wavecut-master base the fork
   tracks, not from this port. Worth a separate look if the 4080S Qwen production speed matters.
3. **Artifacts 6 and 8 needed `--nvfp4-native`.** Both are NVFP4-weight / A4-activation Qwen
   artifacts (`v2_directory.cpp` binds nvfp4 with `AllowA4`); the A4 route needs the e2m1
   LUT decode that `--nvfp4-native` selects (see `docker-compose.nvfp-4080.yml`). Without it the
   activation path misfires and output is garbage on every binary — my earlier sweeps used no
   such flag (#6) and the wrong spec backend (#8). With the flag: #6 (a genuine **v2** file,
   read through the fork's v2-direct-read path) gives clean text at 33.7 tok/s, and #8 matches
   the 1.7.2 release within dflash2's natural spread. The port does not regress either path.
   Remaining gap: Qwen dense GGUF and Gemma4 sit ~5% below the 1.7.2 release on the 4080S in
   PRE too — a wavecut-base delta worth a separate look, not this port's.
4. Gemma4's repetition appears on both binaries with a raw completion prompt (no chat template);
   behavioral parity, not a regression.
5. **`--source mtp=` from a GGUF needed two fixes.** The standard `gguf` package does not know
   PTQ1_0 (ggml type 143) — `GGUFReader` crashed on the official `-mtp.gguf`, which mixes the
   PTQ1_0 text layers with the Q4_0 MTP block (`blk.64.*`). Added `PQ2_0 = 142` / `PTQ1_0 = 143`
   to the local gguf install's `GGMLQuantizationType` and `GGML_QUANT_SIZES` (block 128, type 28 /
   34 bytes), and taught `_open_named_source` to wrap the `mtp` GGUF source in `HFAliasSource`
   with `qwen35_mtp_name_map(64)` so the HF-style MTP names resolve against the `blk.64.*`
   tensors. The re-converted `Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.ninfer` (7.1 GiB,
   790 objects) loads and decodes at 82.6 tok/s with MTP 4 drafts on the V100, up from 69.8
   without the MTP component.

### 4080S production A/B (2026-11-26, post-gate-fix)

Same production container and flags (`docker-compose.gglm-4080.yml`: 200k ctx,
`--lm-head-draft`, concurrency 2, MTP4), only the binary swapped:

| Binary | Decode | MTP accept | TTFT | Prefill |
|---|---|---|---|---|
| OLD 1.7.2 (shipped) | 70.5 tok/s | 33.8% | 168 ms | 1.23k tok/s |
| NEW inkling2 (sm_89) | **78.7 tok/s** | **40.0%** | 159 ms | 1.29k tok/s |

Four consecutive NEW runs: 78.7 tok/s each, 40.0% accept — stable. The perf port
gains **+11.6% decode** on the 4080S under production settings (the earlier
37.4 = 37.4 "neutral" reading was a 8k-ctx / no-`--lm-head-draft` test container;
the production configuration exposes the port's gain). Both pass the Paris probe.
Deployed: `ninfer:1.8.0-inkling2-sm89`, `/opt/ninfer/ninfer-serve.sm89`.

### 4080S full artifact sweep (2026-11-26, NEW sm_89 binary)

Same 8k-ctx / concurrency-1 test harness as the V100 sweep (`/tmp/ninfer-serve-89-FINAL`
from the inkling2 branch, W4A4/swift with `--nvfp4-native`, Flash-Next with
`--expert-residency host`):

| Artifact | Type | Spec | Probe | Decode |
|---|---|---|---|---|
| Gemma4-31B | dense | MTP4 | Paris ✓ | 71.7 |
| Gemma4-26B-A4B | MoE | MTP2 | Paris ✓ | 262.1 |
| GSQ IQ3_S | Qwen MoE | plain | Paris ✓ | 52.3 |
| Huihui Q4_K_M (production A/B) | Qwen MoE | MTP4 | Paris ✓ | 78.7 |
| Inkling-2 Mini | Inkling MoE NVFP4 | DSpark | ✓ | 125.5 |
| W4A4 DFlash2 | Qwen W4A4 | MTP4 | Paris ✓ | 60.3 |
| swift nvfp4 DFlash2 | Qwen NVFP4 | MTP4 | Paris ✓ | 63.3 |
| Ternary-Bonsai (plain, no MTP in artifact) | Qwen PTQ | plain | Paris ✓ | 82.3 |
| Flash-Next IQ3_S | Flash-Next | MTP4 + host RAM | Paris ✓ | 53.3 |

9/9 pass (Ternary's MTP companion is still being downloaded for a re-convert; the
plain artifact is verified here). No regressions on sm_89 — every artifact starts,
passes the probe, and decodes.

## Environment notes

- Both production services (`ninfer-inkling2` on the V100, `ninfer-4080` + ComfyUI on the 4080S)
  were stopped for the sweep and restored afterwards; health checks pass.
- Test binaries: `/tmp/ninfer-serve-70-NEW` (sm_70), `/tmp/ninfer-serve-89-{PRE,POST}` (sm_86;89).

## V100 spec-mode regression: attribution and fix

The full 96-point sweep (16 artifacts x V100/4080S x OLD/PRE/NEW x plain/spec) found a
V100-only regression the 4-artifact sweep above had not covered: MTP/DSpark spec mode on
sm_70 lost 15-20% versus the 1.7.2 release (Huihui Q4_K_M 67.6 -> 54.1, GSQ IQ3_S
61.5 -> 52.2, Gemma4-31B 48.5 -> 40.7, Gemma4-26B 164.8 -> 148.6 tok/s), while plain
decode (t=1) lost only 2-6% and the 4080S gained +8.5% on the same artifacts.

### Attribution

Built PRE = the fork base without the perf port (6734d5016, sm_70) and BISECT = PRE plus
only the two gguf vector-product commits of the port (1494f934b shared launches,
7e37de5b3 spread slices):

| V100 case | OLD 1.7.2 | PRE (no port) | BISECT (PRE+2) | NEW (HEAD) |
|---|---|---|---|---|
| Gemma4-31B spec | 48.5 | 47.9 | 41.2 | 40.7 |
| Gemma4-26B spec | 164.8 | 161.9 | 156.5 | 148.6 |
| GSQ IQ3_S plain | 45.2 | 42.6 | 42.4 | 42.4 |

- PRE matches OLD on the Gemma4 artifacts (the port's clean control).
- Adding only the two vector commits reproduces most of the spec drop (Gemma4-31B:
  47.9 -> 41.2 of the 47.9 -> 40.7; Gemma4-26B: 161.9 -> 156.5 of the 161.9 -> 148.6).
  They are the cause of the port-induced spec regression.
- The GSQ plain gap (45.2 -> 42.6) is already present in PRE: it is the wavecut base,
  not the port (the two commits move it only 42.6 -> 42.4, within noise). Same class as
  the ~-5% base gap on the 4080S noted above.
- The two commits target sm_80+ (upstream measured 3090/4090/5090; the 4080S gains
  +8.5%), so the fix gates their V100-hostile behavior to sm_80+ rather than reverting it.

Separate finding: the wavecut base changed V100 t=1 numerics (greedy text drifts versus
1.7.2; PRE and NEW produce identical text, both differing from OLD). That explains the
acceptance-rate shifts in spec mode (e.g. Huihui 42.8% -> 34.8%) - the engines generate
different content on the same prompt, not worse drafting. Present in PRE, not caused by
the port. It also accounts for most of Huihui's spec gap: 67.6 -> 54.3 (-19.7%) matches
the accept drop 42.8% -> 34.8% (-18.7%) to within the residual step-time difference.
Huihui cannot be measured on PRE (the pre-port base cannot load its Q6_K dense
projections), so its base-vs-port split rests on the accept arithmetic plus the
Gemma4/GSQ controls.

### Fix

Compile-time gates restore the release's dense kernels on sm_70/sm_75 (the SIMT-only
tiers) while sm_80+ keeps the ported kernels. Two earlier attempts gate-gated the
whole spread form (`kSpreadForm` false below sm_80), which also swapped the MoE
`moe_rows_kernel` (spread form) for `moe_kernel` (non-spread). Those binaries failed
production startup (`gguf moe product launch: invalid resource handle` during CUDA-graph
preparation); `NINFER_MOE_DBG` traced the failure to the first non-spread `moe_kernel`
launch (chunk 4) - the spread `moe_rows_kernel` is capture-safe on V100 but the
non-spread `moe_kernel` is not. The final gate therefore gates only the dense path and
keeps the MoE spread form on every tier:

- `ggml_bridge_vec.cuh`:
  - `kVecSpread` (sm_80+ only) gates the spread tail in `vec_rows` (the dense product).
  - the MoE `moe_rows1_kernel` (chunk 1) and `moe_rows_kernel` (chunks 2-8) stay on
    every tier - both are capture-safe, the non-spread `moe_kernel` is not.
  - `kSpreadForm` itself is arch-independent (format-level eligibility).
- `gguf_linear.cpp`: `gguf_vector_project` flushes one product per launch in builds with
  no sm_80+ tier (`NINFER_TIER_AMPERE`/`NINFER_TIER_BLACKWELL` undefined) - `vec_launch`
  with count=1 takes the unchanged one-product kernel, i.e. the pre-port behavior.

Verification (GATE = HEAD + the two gates, sm_70):

| V100 case | NEW (HEAD) | GATE | verdict |
|---|---|---|---|
| Gemma4-31B spec | 40.7 | 47.8 | recovered to PRE (47.9) |
| Gemma4-26B spec | 148.6 | 161.9 | recovered to PRE (161.9) |
| GSQ IQ3_S plain | 42.4 | 42.4 | unchanged - its gap is the base's, not the port's |
| Huihui Q4_K_M spec | 54.1 | 54.3 | unchanged - its gap is the accept-rate drift (base numerics) |
| Inkling-2 Q4_K_XL spec (production) | 66.5 | 70.0 | improved +5% (accept 41.4% -> 47.1%) |

All GATE runs pass the correctness probe (Paris). GATE text equals NEW's (the gated
sm_70 path is the pre-port kernel, measured bit-identical to PRE/NEW).
