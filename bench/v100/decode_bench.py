#!/usr/bin/env python3
# Adapted from huangserva/ninfer-v100-tpx tools/bench/v100 (Apache License 2.0),
# itself built on the ninfer-v100 sm_70 port.
"""Decode-speed benchmark against a running ninfer-serve or the TP2 proxy (stdlib only).

Sends the same synthetic prompt once per seed and reports what ninfer measured itself
(`timings` in the response): prompt length, prefill time, generated tokens, decode tok/s,
MTP acceptance and the derived milliseconds per decode round.

    # headline number of the README: 186K code prompt, 4 seeds x 1024 tokens, through the proxy
    python3 decode_bench.py --url http://127.0.0.1:18881 --model qwen38-ninfer \
        --key-file /path/to/api-key --kind code --blocks 722 --seeds 1,2,3,4 --max-tokens 1024

The first request is the cold one (full prefill); later seeds hit the prefix cache unless the
server runs with --no-prefix-reuse. Decode speed does not depend on that; first-token latency does.
One JSON line per request plus one summary line; `--out` appends the same lines to a file.
"""
import argparse, json, statistics, sys, time, urllib.error, urllib.request, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_prompt  # noqa: E402


def request(url, key, body, timeout):
    q = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                               data=json.dumps(body, ensure_ascii=False).encode(),
                               headers={"Content-Type": "application/json",
                                        "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(q, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:18881")
    ap.add_argument("--model", default="qwen38-ninfer")
    ap.add_argument("--key", default=None, help="bearer key (default: read --key-file, else 'k')")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--kind", choices=ctx_prompt.KINDS, required=True)
    ap.add_argument("--blocks", type=int, required=True, help="see ctx_prompt.py for the table")
    ap.add_argument("--offset", type=int, default=0, help="block-number offset (unique prefixes)")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0 = greedy; otherwise top_p 0.8, top_k 20, presence_penalty 1.5")
    ap.add_argument("--draft-tokens", type=int, default=3, help="MTP draft length the server runs with")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    ap.add_argument("--save-text", default=None, help="directory: save each completion as text")
    a = ap.parse_args()
    key = a.key or (open(a.key_file).read().strip() if a.key_file else "k")
    prompt = ctx_prompt.build_prompt(a.kind, a.blocks, a.offset)
    rows = []
    for i, seed in enumerate(int(x) for x in a.seeds.split(",")):
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": a.max_tokens, "seed": seed, "stream": False,
                "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
        if a.temperature == 0:
            body["temperature"] = 0.0
        else:
            body.update({"temperature": a.temperature, "top_p": 0.8, "top_k": 20,
                         "presence_penalty": 1.5})
        try:
            d, wall = request(a.url, key, body, a.timeout)
        except urllib.error.HTTPError as e:
            print(json.dumps({"tag": a.tag, "kind": a.kind, "blocks": a.blocks, "seed": seed,
                              "error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"}),
                  flush=True)
            continue
        tm = d.get("timings") or {}
        dn, da = tm.get("draft_n") or 0, tm.get("draft_n_accepted") or 0
        acc = da / dn if dn else None
        gen = tm.get("predicted_n") or 0
        row = {"tag": a.tag, "kind": a.kind, "blocks": a.blocks, "offset": a.offset, "seed": seed,
               "cold": i == 0, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
               "prefill_s": round((tm.get("prompt_ms") or 0) / 1000, 1),
               "gen": gen, "tok_s": round(tm.get("predicted_per_second") or 0, 2),
               "accept": round(acc, 3) if acc is not None else None,
               "ms_per_round": round((tm.get("predicted_ms") or 0) / gen * (1 + a.draft_tokens * acc), 1)
               if gen and acc is not None else None,
               "finish": (d.get("choices") or [{}])[0].get("finish_reason"),
               "wall_s": round(wall, 1)}
        rows.append(row)
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        if a.out:
            open(a.out, "a").write(line + "\n")
        if a.save_text:
            os.makedirs(a.save_text, exist_ok=True)
            with open(os.path.join(a.save_text, f"{a.tag or 'run'}_{a.kind}_{a.blocks}_s{seed}.txt"), "w") as f:
                f.write(d["choices"][0]["message"]["content"])
    if rows:
        ok = [r for r in rows if r["tok_s"]]
        summ = {"summary": True, "tag": a.tag, "kind": a.kind, "blocks": a.blocks,
                "prompt_n": rows[0]["prompt_n"], "runs": len(ok),
                "avg_tok_s": round(statistics.mean(r["tok_s"] for r in ok), 2) if ok else None,
                "avg_accept": round(statistics.mean(r["accept"] for r in ok if r["accept"] is not None), 3)
                if any(r["accept"] is not None for r in ok) else None,
                "avg_ms_per_round": round(statistics.mean(r["ms_per_round"] for r in ok if r["ms_per_round"]), 1)
                if any(r["ms_per_round"] for r in ok) else None,
                "cold_prefill_s": rows[0]["prefill_s"]}
        line = json.dumps(summ, ensure_ascii=False)
        print(line, flush=True)
        if a.out:
            open(a.out, "a").write(line + "\n")


if __name__ == "__main__":
    main()
