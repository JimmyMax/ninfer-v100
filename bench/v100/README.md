# V100 long-context benchmark clients

Black-box clients for measuring a running `ninfer-serve`: they read the server's own `timings`
(prompt length, prefill time, generated tokens, decode tok/s, MTP acceptance) rather than timing
the HTTP response, so numbers are independent of client-side buffering.

- `ctx_prompt.py` — builds deterministic synthetic prompts with an exact token count. The headline
  shapes are `--kind code --blocks 722` (186,420 tokens) and `--kind zh-doc --blocks 1162`
  (193,104 tokens); smaller block counts give the 8K / 32K levels.
- `decode_bench.py` — sends one prompt per seed and appends one JSON line per request plus a
  summary line to `--out`. The first seed is the cold prefill; later seeds hit the prefix cache
  (decode speed does not depend on that, first-token latency does).

Both are adapted from [huangserva/ninfer-v100-tpx](https://github.com/huangserva/ninfer-v100-tpx)
(`tools/bench/v100/`, Apache License 2.0).

## Example

Serve the artifact on the target GPU with the production shape, then:

```bash
python3 bench/v100/decode_bench.py --url http://127.0.0.1:8000 --model qwen --key <api-key> \
  --kind code --blocks 722 --seeds 1,2 --max-tokens 1024 --tag <label> --out /tmp/ctx.jsonl
```

Server flags used for the published numbers: `--max-context 200000 --prefill-chunk 2048
--kv-capacity auto --kv-dtype int8 --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft
--preserve-thinking --vision`.

## Measurement rules

These rules are what make the numbers repeatable; follow them for any new comparison.

1. **Prefer an in-image switch over two images.** `NINFER_SM70_ATTN_V2=0` selects the original
   INT8 small-T decode attention kernel, so both variants run in one process image with identical
   session state.
2. **Always measure a drift control.** Re-run the baseline after the candidate, or interleave.
   Long-context decode repeats within ~1%, cold prefill within ~5%, and short-request latency
   drifts far more.
3. **One tenant per host.** A concurrent build or image export inflated host-side preparation
   measurements by 2x in one session.
4. **Stop the deployment container** before benchmarking: the model needs the whole GPU, and the
   long-context prefill is memory-bound.
