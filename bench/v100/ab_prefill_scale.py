#!/usr/bin/env python3
"""prefill 长度扫描：同一服务内依次发 4K/8K/16K/27K 请求，比较两线的 prefill 吞吐。

用途：区分「差距随长度增长（注意力）」与「各长度均匀（权重受限的 GEMM 效率）」。
"""
import json
import re
import subprocess
import sys
import time

MODEL = "/workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
BASE = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-itzy", "--model-id", "qwen",
        "--max-context", "200000", "--prefill-chunk", "2048", "--kv-capacity", "auto",
        "--kv-dtype", "int8", "--max-concurrency", "2", "--spec", "mtp", "--draft-tokens", "4",
        "--lm-head-draft", "--preserve-thinking", "--vision"]

UNIT = "第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 "


def prompt_for(chars):
    n = max(2, chars // 40)
    filler = "".join(UNIT % (i, i) for i in range(1, n))
    return "秋天到了。" + filler + "\n请用一句话总结。"


def run(image, tag, lengths, port):
    name = f"y{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "--gpus", "device=1",
                    "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
                    "--entrypoint", "ninfer-serve", image, MODEL] + BASE, capture_output=True)
    for _ in range(150):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            break
        time.sleep(5)
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    print(f"  [{tag}] 服务就绪 ip={ip}", flush=True)
    for chars in lengths:
        bp = f"/tmp/y{port}-{chars}.json"
        with open(bp, "w") as fh:
            json.dump({"model": "qwen", "messages": [{"role": "user", "content": prompt_for(chars)}],
                       "max_tokens": 8, "temperature": 0}, fh, ensure_ascii=False)
        subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "1800", "-o", f"/tmp/y{port}-{chars}.resp",
                        f"http://{ip}:8000/v1/chat/completions",
                        "-H", "Authorization: Bearer sk-itzy",
                        "-H", "Content-Type: application/json", "-d", f"@{bp}"], capture_output=True)
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        lines = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
        s = lines[-1] if lines else ""
        pt = re.search(r"prompt ([\d,]+)", s)
        pre = re.search(r"prefill ([\d.]+)k? tok/s", s)
        ttft = re.search(r"TTFT ([\d.]+)\s*(ms|s)", s)
        dec = re.search(r"decode ([\d.]+) tok/s", s)
        rec = {
            "tag": tag, "chars": chars,
            "prompt_tokens": pt.group(1) if pt else "?",
            "prefill_tok_s": (pre.group(1) + ("k" if "k tok/s" in s else "")) if pre else "?",
            "ttft_s": round(float(ttft.group(1)) / (1000 if ttft.group(2) == "ms" else 1), 2) if ttft else "?",
            "decode": dec.group(1) if dec else "?",
        }
        print("  " + json.dumps(rec, ensure_ascii=False), flush=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


if __name__ == "__main__":
    lengths = [4000, 8000, 16000, 27000]
    arms = [a.split("=", 1) for a in sys.argv[1:]]
    if not arms:
        arms = [["ours", "ninfer:v100-deq"], ["master", "ninfer:v1.5.0"]]
    for i, (tag, image) in enumerate(arms):
        run(image, tag, lengths, 24001 + i * 3)
