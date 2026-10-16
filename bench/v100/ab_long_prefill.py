#!/usr/bin/env python3
"""长文生成对照 + prefill 粒度实验。"""
import json
import re
import subprocess
import sys
import time
from wait_ready import wait_ready_ip  # noqa: E402

MODEL = "/workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
BASE = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-itzy", "--model-id", "qwen",
        "--max-context", "200000", "--kv-capacity", "auto", "--kv-dtype", "int8",
        "--max-concurrency", "2", "--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft",
        "--preserve-thinking", "--vision"]
FILLER = "".join("第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i)
                 for i in range(1, 1050))


def run(label, image, prompt, max_tokens, port, chunk=2048, timeout=1800):
    name = f"L{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "--gpus", "device=1",
                    "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
                    "--entrypoint", "ninfer-serve", image, MODEL,
                    "--prefill-chunk", str(chunk)] + BASE, capture_output=True)
    for _ in range(140):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            break
        time.sleep(5)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        print(json.dumps({"label": label, "error": "startup failed"}, ensure_ascii=False), flush=True)
        return
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    wait_ready_ip(ip)  # /health 就绪探针：日志行可能来自上次启动
    bp = f"/tmp/L{port}.json"
    with open(bp, "w") as fh:
        json.dump({"model": "qwen", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0}, fh, ensure_ascii=False)
    t0 = time.time()
    subprocess.run(["curl", "-s", "--noproxy", "*", "-m", str(timeout), "-o", f"/tmp/L{port}-resp.json",
                    f"http://{ip}:8000/v1/chat/completions",
                    "-H", "Authorization: Bearer sk-itzy",
                    "-H", "Content-Type: application/json", "-d", f"@{bp}"], capture_output=True)
    wall = round(time.time() - t0, 1)
    log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    lines = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
    s = lines[-1] if lines else ""
    def g(p, d="?"):
        m = re.search(p, s)
        return m.group(1) if m else d
    text = ""
    try:
        r = json.load(open(f"/tmp/L{port}-resp.json"))
        m = r["choices"][0]["message"]
        text = (m.get("reasoning_content") or "") + (m.get("content") or "")
    except Exception:
        pass
    out = g(r"output ([\d,]+)")
    dec = g(r"decode ([\d.]+)")
    acc = re.search(r"accepted ([\d,]+)/([\d,]+)", s)
    tok_per_round, round_ms, accp = 1.0, None, "?"
    if acc:
        n, den = int(acc.group(1).replace(",", "")), int(acc.group(2).replace(",", ""))
        accp = round(100 * n / den, 1)
        tok_per_round = int(out.replace(",", "")) / (den / 4) if out != "?" else 1.0
    if dec != "?":
        round_ms = round(1000 * tok_per_round / float(dec), 2)
    tm = re.search(r"TTFT ([\d.]+)\s*(ms|s)", raw_line := s)
    ttft_s = round(float(tm.group(1)) / (1000 if tm.group(2) == "ms" else 1), 1) if tm else "?"
    print(json.dumps({"label": label, "image": image, "chunk": chunk, "wall_s": wall,
                      "prefill": g(r"prefill ([\d.]+)k?"), "ttft_s": ttft_s,
                      "decode": dec, "acc": accp, "tok_per_round": round(tok_per_round, 2),
                      "round_ms": round_ms, "text_len": len(text),
                      "text_head": re.sub(r"\s+", "", text)[:40]}, ensure_ascii=False), flush=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


if __name__ == "__main__":
    LONG = "写一篇六百字的散文，主题是秋天的黄昏，要求有具体景物描写。"
    CTX = "秋天到了。" + FILLER + "\n请用三百字写一段关于秋天的散文。"
    run("new 600字", "ninfer:v100-fp16qpn", LONG, 600, 22001)
    run("master 600字", "ninfer:v1.5.0", LONG, 600, 22002)
    run("new 27K chunk4096", "ninfer:v100-fp16qpn", CTX, 32, 22003, chunk=4096)
    run("new 27K chunk8192", "ninfer:v100-fp16qpn", CTX, 32, 22004, chunk=8192)
    run("new 27K chunk1024", "ninfer:v100-fp16qpn", CTX, 32, 22005, chunk=1024)
