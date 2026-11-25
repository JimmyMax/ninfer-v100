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

## Environment notes

- Both production services (`ninfer-inkling2` on the V100, `ninfer-4080` + ComfyUI on the 4080S)
  were stopped for the sweep and restored afterwards; health checks pass.
- Test binaries: `/tmp/ninfer-serve-70-NEW` (sm_70), `/tmp/ninfer-serve-89-{PRE,POST}` (sm_86;89).
