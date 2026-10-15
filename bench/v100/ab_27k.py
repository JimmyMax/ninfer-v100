#!/usr/bin/env python3
"""27K 长上下文的 decode/prefill 对照（稳解析：TTFT 可能是秒）。"""
import json
import re
import subprocess
import sys
import time

MODEL = "/workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
BASE = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-itzy", "--model-id", "qwen",
        "--max-context", "200000", "--prefill-chunk", "2048", "--kv-capacity", "auto",
        "--kv-dtype", "int8", "--max-concurrency", "2", "--preserve-thinking", "--vision"]
FILLER = "".join("第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i)
                 for i in range(1, 1050))
PROMPT = "秋天到了。" + FILLER + "\n请用三百字写一段关于秋天的散文。"


def run(label, image, port, mtp, max_tokens=96):
    name = f"x{port}"
    extra = ["--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft"] if mtp else []
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "--gpus", "device=1",
                    "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
                    "--entrypoint", "ninfer-serve", image, MODEL] + BASE + extra,
                   capture_output=True)
    for _ in range(150):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            break
        time.sleep(5)
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    bp = f"/tmp/x{port}.json"
    with open(bp, "w") as fh:
        json.dump({"model": "qwen", "messages": [{"role": "user", "content": PROMPT}],
                   "max_tokens": max_tokens, "temperature": 0}, fh, ensure_ascii=False)
    subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "1800", "-o", f"/tmp/x{port}-resp.json",
                    f"http://{ip}:8000/v1/chat/completions",
                    "-H", "Authorization: Bearer sk-itzy",
                    "-H", "Content-Type: application/json", "-d", f"@{bp}"], capture_output=True)
    log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    lines = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
    s = lines[-1] if lines else ""
    def g(p, d="?"):
        m = re.search(p, s)
        return m.group(1) if m else d
    out = g(r"output ([\d,]+)")
    dec = g(r"decode ([\d.]+)")
    pre = g(r"prefill ([\d.]+)k?")
    ttft = re.search(r"TTFT ([\d.]+)\s*(ms|s)", s)
    acc = re.search(r"accepted ([\d,]+)/([\d,]+)", s)
    tpr, rms, accp = 1.0, None, "?"
    if acc:
        n, den = int(acc.group(1).replace(",", "")), int(acc.group(2).replace(",", ""))
        accp = round(100 * n / den, 1)
        if out != "?":
            tpr = int(out.replace(",", "")) / (den / 4)
    if dec != "?":
        rms = round(1000 * tpr / float(dec), 2)
    rec = {"label": label, "out": out, "prefill": pre + ("k" if "k tok/s" in s else ""),
           "ttft_s": round(float(ttft.group(1)) / (1000 if ttft.group(2) == "ms" else 1), 1) if ttft else "?",
           "decode": dec, "acc": accp, "round_ms": rms}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


if __name__ == "__main__":
    for tag, image in (("new", "ninfer:v100-parity2"), ("master", "ninfer:v1.5.0")):
        run(f"{tag} 27K nospec", image, 23001 + (0 if tag == "new" else 1), mtp=False)
        run(f"{tag} 27K mtp4", image, 23003 + (0 if tag == "new" else 1), mtp=True)
