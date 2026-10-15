#!/usr/bin/env python3
"""每轮成本对照：无投机（纯每遍前向）与 MTP4（含接受率换算）。

用法: python3 ab_perround.py TAG=IMAGE [TAG=IMAGE ...]
每臂每场景输出一行 JSON：原始日志行 + 解析字段 + 换算的每轮 ms。
"""
import json
import re
import subprocess
import sys
import time

MODEL = "/workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
COMMON = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-itzy", "--model-id", "qwen",
          "--max-context", "200000", "--prefill-chunk", "2048", "--kv-capacity", "auto",
          "--kv-dtype", "int8", "--max-concurrency", "2", "--preserve-thinking", "--vision"]
MTP = ["--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft"]

FILLER = "".join("第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i)
                 for i in range(1, 1050))
SHORT = "你好"
CTX = "秋天到了。" + FILLER + "\n请用三百字写一段关于秋天的散文。"

CASES = [
    ("short-nospec", SHORT, 96, []),
    ("ctx27k-nospec", CTX, 96, []),
    ("short-mtp4", SHORT, 96, MTP),
    ("ctx27k-mtp4", CTX, 96, MTP),
]

LINE_RE = re.compile(
    r"req#(?P<req>\d+) done.*?\| output (?P<output>[\d,]+)"
    r".*?\| TTFT (?P<ttft>[\d.]+) ms.*?\| total (?P<total>[\d.]+)\s*(?P<unit>ms|s)"
    r".*?\| prefill (?P<prefill>[\d.]+) tok/s.*?\| decode (?P<decode>[\d.]+) tok/s"
    r"(?:.*?accepted (?P<acc_num>[\d,]+)/(?P<acc_den>[\d,]+))?")


def run(tag, image, case, prompt, max_tokens, extra, port):
    name = f"p{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "--gpus", "device=1",
                    "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
                    "--entrypoint", "ninfer-serve", image, MODEL] + COMMON + extra,
                   capture_output=True)
    for _ in range(150):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            break
        time.sleep(4)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return {"tag": tag, "case": case, "error": "startup failed"}
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    body_path = f"/tmp/p{port}-body.json"
    with open(body_path, "w") as fh:
        json.dump({"model": "qwen", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0}, fh, ensure_ascii=False)
    subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "1800", "-o", f"/tmp/p{port}.json",
                    f"http://{ip}:8000/v1/chat/completions",
                    "-H", "Authorization: Bearer sk-itzy",
                    "-H", "Content-Type: application/json", "-d", f"@{body_path}"],
                   capture_output=True)
    log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    raw = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
    rec = {"tag": tag, "case": case, "raw": raw[-1].strip() if raw else ""}
    m = LINE_RE.search(rec["raw"])
    if m:
        out = int(m.group("output").replace(",", ""))
        dec = float(m.group("decode"))
        total = float(m.group("total"))
        if m.group("unit") == "ms":
            total /= 1000.0
        rec.update(output=out, decode=dec, prefill=float(m.group("prefill")),
                   ttft=float(m.group("ttft")), total_s=total)
        if m.group("acc_den"):
            acc_n = int(m.group("acc_num").replace(",", ""))
            acc_d = int(m.group("acc_den").replace(",", ""))
            rounds = acc_d / 4.0
            tok_per_round = out / rounds
            rec.update(acc_pct=round(100.0 * acc_n / acc_d, 1), rounds=round(rounds, 1),
                       tok_per_round=round(tok_per_round, 2),
                       round_ms=round(1000.0 * tok_per_round / dec, 2))
        else:
            rec.update(tok_per_round=1.0, round_ms=round(1000.0 / dec, 2))
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


if __name__ == "__main__":
    arms = [a.split("=", 1) for a in sys.argv[1:]]
    res = []
    for case, prompt, mx, extra in CASES:
        for i, (tag, image) in enumerate(arms):
            res.append(run(tag, image, case, prompt, mx, extra, 19500 + i * 7 + len(res) % 7))
    print("=== 汇总（round_ms = 每轮 GPU/单遍成本，越小越好）===", flush=True)
    for r in res:
        if "error" in r:
            print(f"{r['case']:>14} {r['tag']:>7}: {r['error']}", flush=True)
            continue
        print(f"{r['case']:>14} {r['tag']:>7}: decode={r.get('decode')} tok/s "
              f"acc={r.get('acc_pct')}% tok/轮={r.get('tok_per_round')} "
              f"round={r.get('round_ms')}ms 轮数={r.get('rounds')}", flush=True)
