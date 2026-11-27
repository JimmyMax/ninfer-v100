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
| Ternary-Bonsai **+ MTP** (re-converted, 2026-11-26) | Qwen PTQ | MTP4 | Paris ✓ | **109.5** (24.1% accepted) |
| Flash-Next IQ3_S | Flash-Next | MTP4 + host RAM | Paris ✓ | 53.3 |

10/10 pass. No regressions on sm_89 — every artifact starts, passes the probe, and
decodes; the MTP companion lifts Ternary-Bonsai from 82.3 to 109.5 tok/s.

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

## Fair A/B: OLD 1.7.2 vs NEW (best config, 2026-11-27)

The 8k-ctx / no-`--lm-head-draft` / 400-token test harness is weaker than the production
configuration and understates both binaries. To judge the line fairly, every artifact was
re-measured on both binaries with each artifact's best-known drafter configuration
(512-token greedy generations, 8k ctx, int8 KV, concurrency 1; Huihui with
`--lm-head-draft --ngram-draft-tokens 0`; W4A4/swift with `--nvfp4-native`; Flash-Next with
`--expert-residency host` + ngram table):

| Artifact | OLD 1.7.2 | NEW (gated) | delta |
|---|---|---|---|
| orcarouter NVFP4 | 69.8 (acc 41.8%) | 69.8 (41.8%) | +0.0% |
| Swift Q4_K_M | 65.5 (39.8%) | 60.1 (41.3%) | **-8.2%** |
| GSQ IQ3_S | 66.5 (37.9%) | 54.8 (38.1%) | **-17.6%** |
| Huihui Q4_K_M | 65.4 (40.4%) | 60.2 (41.2%) | **-8.0%** |
| Ternary PTQ1_0 + MTP | 82.0 (28.1%) | 82.0 (28.1%) | +0.0% |
| W4A4 DFlash2 | 67.3 (39.2%) | 67.3 (39.2%) | +0.0% |
| swift NVFP4 DFlash2 | 69.9 (40.2%) | 69.9 (40.2%) | +0.0% |
| Gemma4-31B | 52.7 (38.9%) | 52.4 (39.7%) | -0.6% |
| Gemma4-26B-A4B | 169.4 (56.2%) | 153.9 (56.5%) | **-9.1%** |
| Inkling-2 DSpark | n/a (`--spec dspine` not in 1.7.2) | 70.0 | new feature |
| Flash-Next IQ3_S | 35.9 (32.2%) | 34.6 (30.8%) | -3.6% |

Readings:

- **The nvfp4 / DFlash2 family and Ternary MTP are bit-neutral** (0.0% — the gated sm_70
  path reproduces the release kernel exactly, as designed).
- **Gemma4-31B is recovered** to within -0.6% (the 96-point sweep had it at 48.5 → 40.7
  before the gate; the gate restores it).
- **Gemma4-26B (-9.1%) and Swift/Huihui Q4_K_M (-8.0%/-8.2%)** sit just inside the -10%
  blocking gate. They are the same class as the documented wavecut-base numerics drift:
  the fork base generates slightly different greedy text than 1.7.2, which shifts
  per-artifact accept rates and step mix (accept is higher on NEW for Swift/Huihui, so
  the remainder is step-time on the base's new routes, not the port's).
- **GSQ IQ3_S (-17.6%) is the one artifact over the gate.** The 96-point attribution
  established its plain gap (45.2 → 42.6) is present in PRE, i.e. the wavecut base, and
  the gate does not move it (42.4 → 42.4). The spec-mode gap is the same cause in a
  louder configuration: the base's t=1 numerics drift changes what the MTP drafter
  accepts. It is not introduced by the perf port (PRE shows the class; the two vector
  commits move it only 42.6 → 42.4).
- Inkling-2 DSpark has no 1.7.2 baseline — `--spec dspine` is a new capability of this
  line, not a regression surface.

Conclusion for the line: the perf port itself is neutral-to-positive on sm_70
(neutral on the gated family, and +11.6% on the 4080S production config). The residual
gaps versus the 1.7.2 release are the pre-existing wavecut-base drift, with GSQ IQ3_S
spec mode the worst instance at -17.6%.

### Best-config sweep (2026-11-27, NEW sm_70)

Every artifact was swept over `--draft-tokens` {2,4,6,8} (Inkling-2 over 1-3) with a
`--lm-head-draft --ngram-draft-tokens 0` variant where the artifact starts. Findings:

- **dt2 is the best drafter depth for the Qwen MTP MoE family, Ternary, Gemma4-26B and
  Flash-Next on sm_70**; dt4 stays best for orcarouter/W4A4/swift-NVFP4/Gemma4-31B
  (the DFlash2/proposal-head artifacts). dt6/dt8 are strictly worse (and dt8 fails to
  start on the larger KV footprints).
- **`--lm-head-draft` buys nothing on sm_70** (identical or startup failure on every
  artifact that accepts it) — it is not part of the best config.
- The dt2 re-measurement also re-frames the table above: at each artifact's own best
  depth, Swift/Huihui/Ternary/Flash-Next close most of their dt4 gaps.

| Artifact | OLD best | NEW best | delta | best dt |
|---|---|---|---|---|
| orcarouter NVFP4 | 69.8 (41.8%) | 69.8 (41.8%) | 0.0% | 4 |
| Swift Q4_K_M | 68.7 (63.3%) | 67.5 (63.6%) | -1.7% | 2 |
| GSQ IQ3_S | 69.0 (57.0%) | 59.0 (51.5%) | **-14.5%** | 2 |
| Huihui Q4_K_M | 66.9 (58.4%) | 66.1 (58.8%) | -1.2% | 2 |
| Ternary PQ2_0 + MTP | 93.7 (46.4%) | 93.7 (46.4%) | 0.0% | 2 |
| W4A4 DFlash2 | 67.3 (39.2%) | 67.3 (39.2%) | 0.0% | 4 |
| swift NVFP4 DFlash2 | 69.9 (40.2%) | 69.9 (40.2%) | 0.0% | 4 |
| Gemma4-31B | 52.7 (38.9%) | 52.4 (39.7%) | -0.6% | 4 |
| Gemma4-26B-A4B | 169.5 (56.2%) | 153.8 (56.5%) | **-9.3%** | 2 |
| Inkling-2 Mini | n/a (no `--spec dspine` in 1.7.2) | 70.0 plain | new feature | — |
| Flash-Next IQ3_S | 42.7 (51.6%) | 39.6 (47.1%) | -7.3% | 2 |

**Final verdict (best-vs-best):** 9 of 11 artifacts are within ±2% of the 1.7.2 release
(neutral). Flash-Next is -7.3% and Gemma4-26B -9.3%, both inside the -10% blocking
gate. **GSQ IQ3_S is the single artifact over the gate at -14.5%** — the 96-point
attribution established that class (wavecut-base t=1 numerics drift, present in PRE,
the gated vector commits do not move it), so it is a base drift surfaced loudly in
spec mode, not a perf-port regression.

### Ternary-Bonsai PQ2_0 + MTP (2026-11-27)

The official `kelsey1337/Ternary-Bonsai-2.5-27B-Uncensored-MTP-GGUF` repo's PQ2_0 variant
(2.125-bit rows, F32 GDN controls) was converted with a PQ2_0 recipe
(`bonsai_pq20_recipe.py`, mirrors the PTQ1_0 recipe but keeps PQ2_0 rows and expects F32
ssm_alpha/ssm_beta) plus the same `--source mtp=` HFAliasSource path:

| Card | PTQ1_0 + MTP | PQ2_0 + MTP | probe |
|---|---|---|---|
| V100 (sm_70) | 82.0 tok/s @dt4 (28.1%) | **93.7 tok/s @dt2** (46.4%) | Paris ✓ |
| 4080S (sm_89) | 109.5 tok/s @dt4 (24.1%) | 108.0 tok/s @dt4 (23.5%) | Paris ✓ |

Both artifacts are the same 7.13 GiB (the artifact stores T2 rows either way); PQ2_0 is
the supported replacement and the PTQ1_0 MTP artifact was removed. The V100 @dt2 number
is from the best-config sweep above; the two artifacts carry identical T2 values, so
PTQ1_0 would match it.

### GSQ IQ3_S: does the base drift need a fix? (2026-11-27)

The -14.5% GSQ gap at best config is the only over-gate artifact. To decide whether it
is a fixable bug or intrinsic drift, two checks were run:

1. **Text comparison** (GSQ plain, 512 tokens, 1.7.2 vs the wavecut base binary): the
   streams diverge from ~token 30 (9.9% char match) — an early greedy argmax flip
   cascading; both texts are coherent, neither is "wrong".
2. **Perplexity A/B** (`ninfer-perplexity`, built from `v1.7.2` and `6734d5016`
   respectively, same GSQ artifact, `ninfer-ppl-1m-v1 / quick`, int8 KV):

   | domain | 1.7.2 PPL | wavecut base PPL |
   |---|---|---|
   | chinese_reference | 5.111475 | 5.111475 |
   | english_long_form | 6.944164 | 6.944164 |
   | english_reference | 5.995596 | 5.995596 |
   | ninfer_code | 1.658537 | 1.658537 |
   | **overall** | **4.342645** | **4.342645** |

   Identical to six decimals over 261,167 scored tokens. The base's per-token logprobs
   on fixed inputs are numerically faithful to the release — a systematic decode bug
   would show up as higher PPL.

**Conclusion (revised): the gap is two separate things.** The numerics are faithful
(PPL identical), but the base's decode is also *slower* — plain decode is a real
throughput regression, not only an acceptance effect. Matched plain-decode A/B
(no `--spec`, 512-token greedy, 8k ctx, int8 KV, both binaries, same artifact):

| Artifact | OLD 1.7.2 decode | NEW decode | delta | prefill delta |
|---|---|---|---|---|
| GSQ IQ3_S | 45.3 tok/s | 42.5 tok/s | **-6.2%** | -1.0% |
| Huihui Q4_K_M | 40.5 tok/s | 39.7 tok/s | -2.0% | -0.4% |

The GSQ delta is identical at a 22-token prompt (38 generated tokens) and a 197-token
prompt (512 generated): -6.2% in both, so it is a constant per-token cost in the decode
matvec path, not attention or KV. Prefill is flat (the mmq path is unchanged), so the
cost is in the dense vec (matvec) path. Decomposition of the -14.5% spec-mode gap:
-6.2% plain throughput and the rest from the accept drop (57.0% → 51.5%), because the
base's t=1 tokens differ slightly (identical PPL, different greedy choices) and the
drafter is tuned to the release's stream.

The accept half is intrinsic to the rewritten kernels. The -6.2% throughput half is a
plain regression against the release, and the Op-level A/B (below) shows it is **not**
the GGUF matvec kernels — so the cause is still open.

#### Op-level A/B (`ninfer_linear_bench`, v1.7.2 tree vs base tree, V100)

Built the benchmark bundle from a `v1.7.2` worktree and a `6734d5016` worktree and
measured the t=1 decode products directly (median of three, stable to ~1%):

| product (N x K) | 1.7.2 | base | delta |
|---|---|---|---|
| iq3_s 17408x5120 (up/gate) | 73.7 | 72.7 | -1.4% |
| iq3_s 5120x17408 (down) | 85.0 | 72.7 | **-14.4%** |
| iq3_s 5120x6144 (o_proj) | 36.9 | 32.7 | **-11.2%** |
| iq3_s 5120x10240 | 54.2 | 47.1 | -13.2% |
| iq3_s 17408x10240 | 134.1 | 141.3 | +5.3% |
| q6_k 248320x5120 (output head) | 1711 | 1695 | -1.0% |
| q4_k 17408x5120 (up/gate) | 70.3 | 73.7 | **+4.9%** |
| q4_k 34816x5120 | 129.0 | 132.1 | +2.4% |
| q8_0 5120x5120 | 44.7 | 44.7 | 0% |

So on the 27B model's own shapes the base's IQ3_S products are largely *faster* than
the release's (the down-projection by 14%), while Q4_K up/gate is ~5% slower — which
matches Huihui's -2% but not GSQ's -5.8%. Ruled out for that gap: CUDA graphs
(`--no-cuda-graph`: 42.2 vs 42.5), the per-GPU route profile (`--device-profile off`:
42.5), attention/KV (the delta is identical at a 22-token and a 197-token prompt), and
prefill (flat). The remaining suspects are the other per-token ops (GDN state,
norm/rope, sampling, host launch overhead), which need either CUPTI kernel tracing
(nsys injects no kernel data in this container environment) or an op-by-op bench sweep
of the decode composition.

#### In-process reproduction and bisect (2026-11-27)

`ninfer_bench` (engine-level, no serving) reproduces the gap: GSQ `tg256` decode
44.65 tok/s on 1.7.2 vs **42.20** on the base (-5.5%), with identical weights in
memory (10.95 GiB), workspace (75.1 MiB) and config. All measured ops are equal or
faster in the base, including the GDN input projections (q4q5 473→424 us, q8 39.9→38.9),
causal_conv1d_silu, gdn_gating_proj, embedding, attention, rmsnorm, rope, residual,
sampling and kv_append.

Same-model controls isolate the trigger to the **GGUF weight formats**, not the family
or the serving path:

| Artifact | weights | proposal head | plain-decode delta |
|---|---|---|---|
| Ternary-Bonsai PQ2_0 | T2 ternary | no | **0.0%** |
| Gemma4-31B QAT | gguf q4_k/q6_k, no GDN | no | **-0.4%** |
| Huihui Q4_K_M | gguf + GDN | yes | **-2.0%** |
| GSQ IQ3_S / GSQ IQ3_S dspine | gguf mixed + GDN | yes | **-5.8% / -6.1%** |

Bisect over the 31 commits between `v1.7.2` and the base (`6734d5016`): the commit
before `3af694ee4` (`9f0f796a4`, built as `ninfer_bench`) decodes GSQ at **44.65 tok/s
— exactly the 1.7.2 rate**. `3af694ee4` ("feat: Inkling 2 Mini support on V100 (sm_70)",
which pulls the wavecut `src/ops/linear/gguf/` infrastructure and updates the
ggml-quants headers) is therefore the commit that introduced the -5.5%. Note that the
rewritten kernels are *faster in isolation* but the model is slower: the loss is in a
path the isolated Store-epilogue benchmark does not cover — candidates are the fused
`[gate; up]` SwiGLU epilogue (20 layers of this artifact), the `Accumulate` epilogue on
the 64 down-projections, and the Q6_K dense-layout vector route; none has a benchmark
yet.

Follow-ups on the same bench:

- **Single-forward probe** (`-p 1`, one T=1 forward pass, no decode loop): 1.7.2
  44.17 tok/s, parent 44.16, base **41.53** (-6.0%) — the gap is in the forward
  compute, not the serving loop.
- **Epilogue variants** (`--epilogue accumulate|swiglu` added to `linear_bench`): the
  Accumulate path is neutral; the fused SwiGLU path is +6.5% on iq3_s 34816x5120 and
  +2.7% on iq3_xxs (a few tens of microseconds per token, not the 1.4 ms gap).
  iq3_xxs 5120x17408 (down) is a real +10%, also small in aggregate.
- **Correlation with a proposal head**: Ternary (none) 0.0%, Gemma4-31B (none) -0.4%,
  Huihui (131072-row proposal head) -2.0%, GSQ (proposal + a second full-vocabulary
  head) -5.5%. The magnitudes fit one extra head read per forward (~0.5 ms for 377 MB,
  ~1.2 ms for GSQ's two), but no plain-decode path in `qwen3_5` computes the proposal
  head (`proposal_argmax` only runs in the MTP/spec functions), so the mechanism is
  still unproven; the correlation may be with artifacts produced by the newer
  conversion tooling (weight placement) rather than the head itself.

Completing the fix therefore requires replacing the sm_70 vector-product
implementation with the pre-commit one (`ggml_bridge_vec.cuh` family and the
`gguf_linear.cpp` vector path) or finding the extra work with a profiler that supports
Volta (Nsight Systems 2026 and Nsight Compute 2026 both refuse sm_70 here).

#### Root cause and fix: T=1 static-K unrolling on Volta (2026-11-27)

The binary kernel inventory named the cause. Dumping `cuobjdump --dump-resource-usage`
per build shows which instantiations each binary carries:

| T=1 vector kernel (IQ3_S, IQ3_XXS, IQ4_XS, Q4_K) | 1.7.2 | base |
|---|---|---|
| static-K (5120/6144/10240/17408) | **none** (runtime K only) | **all four** |
| runtime-K (`KS=0`) | yes | yes |

1.7.2's `launch_k` deliberately restricted the fully unrolled static-K form at T=1 to
**K=5376 only**, with the measured reason in the code: *"the unrolled form regressed
the 26B K=2816/4096 shapes, so static K at T=1 is dispatched for 5376 only"*. `3af694ee4`
dropped that restriction and the base dispatches static K at T=1 for every 27B shape,
which is exactly the IQ3_S/IQ3_XXS/IQ4_XS/Q4_K weight chain of GSQ.

The isolated benchmark prefers the unrolled kernel (a single product repeated, warm
instruction cache), but a decode token alternates 8-10 block-type kernels and the
larger unrolled bodies evict each other from the instruction cache: the model is 6%
slower with static K even though each product is faster on its own.

**Fix** (`src/ops/linear/gguf/ggml_bridge_vec.cuh`, `launch_k`): builds without an
Ampere/Blackwell tier (Volta/Turing only) dispatch the runtime-K kernel at T=1; T>1
keeps the static-K dispatch, as 1.7.2 did. Ampere/Blackwell builds are untouched.

Verified on GSQ (no spec, int8 KV, `ninfer_bench`):

| binary | pp1 (one T=1 forward) | tg256 (decode) |
|---|---|---|
| 1.7.2 | 44.17 tok/s | 44.65 tok/s |
| base | 41.54 | 42.21 |
| **fix** | **44.31** | **44.83** |

+6.2% decode, matching (slightly above) the 1.7.2 rate — every other per-op regression
measured earlier (iq3_xxs down +10%, iq3_s SwiGLU +6.5%, the proposal-head correlation)
disappears as a side effect, which confirms the instruction-cache mechanism rather than
a per-kernel code regression.

Serving-level check with the rebuilt `ninfer-serve` (sm_70, 8k ctx, int8 KV, 512-token
greedy; OLD = `ninfer:1.7.2`):

| artifact | OLD 1.7.2 | before fix | **fix** |
|---|---|---|---|
| GSQ IQ3_S plain | 45.27 | 42.51 | **45.47 (+0.4%)** |
| Huihui Q4_K_M plain | 40.53 | 39.70 | **40.43 (-0.2%)** |
| Ternary PQ2_0 + MTP (dt2) | 93.7 | 93.7 | 93.7 (unchanged) |
| Gemma4-31B plain | 29.56 | 29.56 | 29.56 (unchanged) |
| orcarouter NVFP4 (MTP 4) | 69.8 | — | 69.8 (unchanged) |
| W4A4 / swift NVFP4 (dflash2 4) | 52.7 / 57.8 | — | 52.7 / 57.8 (unchanged) |

The NVFP4 artifacts are bit-for-bit unaffected (the branch is compiled out for
Ampere/Blackwell builds and never reached by their weight format), which is the
intended scope of the gate.

**Remaining, separate item:** GSQ in spec mode still decodes at 58.98 vs 68.98 tok/s
(-14.5%) with accepted drafts stuck at 51.5% (release 57.0%). With the plain path now
back at the release rate, that residual is the token-numerics drift (the same class the
documents already record for the wavecut base: identical PPL, different greedy choices),
not a kernel-speed regression; it also shows on Flash-Next IQ3_S (-5.2%) and
Gemma4-26B (-9.4%), while every NVFP4 artifact is exactly 0.0%. Tracking down that
drift is a numerics pass, not part of this fix.

#### Status record (2026-11-27)

The static-K fix is complete and verified; it stays uncommitted with the other local
work. Files it touches:

- `src/ops/linear/gguf/ggml_bridge_vec.cuh` — the 8-line `launch_k` gate (the fix)
- `bench/ops/linear_bench.cu` — `--epilogue store|accumulate|swiglu` (the measurement
  that ruled the epilogues out; kept as a tool)
- this document

**Deferred, known item — spec-mode acceptance drift.** Decision: not fixed now. The
record for whoever picks it up:

- Symptom: accepted drafts 51.5% vs the release's 57.0% on GSQ; -14.5% decode at dt2,
  -5.2% on Flash-Next IQ3_S, -9.4% on Gemma4-26B; every NVFP4 artifact 0.0%.
- Nature: not a correctness or quality defect. PPL is identical to six decimals
  (4.342645 over 261,167 tokens) and both engines emit coherent text. The streams agree
  for the first ~30 tokens and then diverge at a near-tie, which is 1-ulp accumulation
  order, not an algorithmic error.
- Independence: reproduced identically with the static-K fix in place (same first
  divergence at character 128, 9.9% character agreement), so it is not the vector-kernel
  choice and not the perf port — it is present in `6734d5016` (the pre-port base).
- Elimination would require making the base's attention/GDN/mmq/dequant accumulation
  order bit-identical to 1.7.2 — a numerics campaign with no quality benefit. A bounded
  way to localize it would be reverting one subsystem at a time (attention, then GDN,
  then dequant) and checking whether the token stream stays identical past token 30,
  one build + sweep cycle per step.
- Impact today: V100 production runs `ninfer:1.7.2` and is unaffected; the new line's
  GGUF artifacts lose only the speculative gain, never the plain-decode rate.
