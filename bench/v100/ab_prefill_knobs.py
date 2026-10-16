#!/usr/bin/env python3
"""27K prefill 开关扫描：同一镜像、同一 prompt，只改环境变量。"""
import json
import re
import subprocess
import sys
import time
from wait_ready import wait_ready_ip  # noqa: E402

MODEL = "/workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
BASE = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-itzy", "--model-id", "qwen",
        "--max-context", "200000", "--prefill-chunk", "2048", "--kv-capacity", "auto",
        "--kv-dtype", "int8", "--max-concurrency", "2", "--spec", "mtp", "--draft-tokens", "4",
        "--lm-head-draft", "--preserve-thinking", "--vision"]
FILLER = "".join("第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i)
                 for i in range(1, 1050))
PROMPT = "秋天到了。" + FILLER + "\n请用三百字写一段关于秋天的散文。"


def run(label, image, env, port):
    name = f"s{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", "device=1",
           "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
           "--entrypoint", "ninfer-serve"]
    if env:
        env_path = f"/tmp/s{port}.env"
        with open(env_path, "w") as fh:
            for k, v in env.items():
                fh.write(f"{k}={v}\n")
        cmd += ["--env-file", env_path]
    cmd += [image, MODEL] + BASE
    subprocess.run(cmd, capture_output=True)
    for _ in range(120):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            break
        time.sleep(5)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return {"label": label, "error": "startup failed"}
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    wait_ready_ip(ip)  # /health 就绪探针：日志行可能来自上次启动
    body = {"model": "qwen", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 32, "temperature": 0}
    bp = f"/tmp/s{port}-body.json"
    with open(bp, "w") as fh:
        json.dump(body, fh, ensure_ascii=False)
    subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "1800", "-o", f"/tmp/s{port}.json",
                    f"http://{ip}:8000/v1/chat/completions",
                    "-H", "Authorization: Bearer sk-itzy",
                    "-H", "Content-Type: application/json", "-d", f"@{bp}"], capture_output=True)
    log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    lines = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
    s = lines[-1] if lines else ""
    pre = re.search(r"prefill ([\d.]+)k? tok/s", s)
    ttft = re.search(r"TTFT ([\d.]+)\s*(ms|s)", s)
    dec = re.search(r"decode ([\d.]+) tok/s", s)
    ttft_s = None
    if ttft:
        ttft_s = float(ttft.group(1)) / (1000 if ttft.group(2) == "ms" else 1)
    rec = {"label": label, "prefill": pre.group(1) + ("k" if pre and "k tok/s" in s else "") if pre else None,
           "ttft_s": round(ttft_s, 1) if ttft_s else None,
           "decode": dec.group(1) if dec else None}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return rec


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "ninfer:v100-headqpn"
    cases = [
        ("baseline", {}),
        ("pack_gqa", {"NINFER_PROMPT_PACK_GQA": "1"}),
        ("pv_f16", {"NINFER_PROMPT_PV_F16": "1"}),
        ("both", {"NINFER_PROMPT_PACK_GQA": "1", "NINFER_PROMPT_PV_F16": "1"}),
        ("prompt_fast", {"NINFER_PROMPT_FAST": "1"}),
    ]
    for i, (label, env) in enumerate(cases):
        run(label, image, env, 21000 + i)
