#!/usr/bin/env python3
"""三臂端到端对照：新 QPN 路由 / 上一版融合核 / master(v1.5.0)。"""
import json
import re
import subprocess
import time

MODEL = "/workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
BASE = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-itzy", "--model-id", "qwen",
        "--max-context", "200000", "--prefill-chunk", "2048", "--kv-capacity", "auto",
        "--kv-dtype", "int8", "--max-concurrency", "2", "--spec", "mtp", "--draft-tokens", "4",
        "--lm-head-draft", "--preserve-thinking", "--vision"]


def run(image, label, port, prompt="你好", max_tokens=64):
    name = f"ab{port}"
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", "device=1",
           "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
           "--entrypoint", "ninfer-serve", image, MODEL] + BASE
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(cmd, capture_output=True)
    for _ in range(150):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            break
        time.sleep(5)
    else:
        print(f"  [{label}] 启动失败")
        print(subprocess.run(["docker", "logs", name], capture_output=True, text=True).stderr[-500:])
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return None
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    body = {"model": "qwen", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    out_path = f"/tmp/ab-{port}.json"
    r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "2400", "-o", out_path,
                        "-w", "%{http_code}", f"http://{ip}:8000/v1/chat/completions",
                        "-H", "Authorization: Bearer sk-itzy",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps(body)], capture_output=True, text=True)
    log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    lines = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
    s = lines[-1] if lines else ""
    dec = re.search(r"decode ([\d.]+)", s)
    acc = re.search(r"accepted (\d+)/(\d+)", s)
    content = ""
    try:
        content = json.load(open(out_path))["choices"][0]["message"]["content"]
    except Exception:
        pass
    print(f"  [{label}] http={r.stdout} decode={dec.group(1) if dec else '?'} tok/s "
          f"accept={acc.group(0).replace('accepted ', '') if acc else '?'}")
    print(f"      输出: {content[:100]!r}")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return content


if __name__ == "__main__":
    subprocess.run(["docker", "stop", "llama-cpp-text"], capture_output=True)
    time.sleep(5)
    try:
        run("ninfer:v100-fusedgdn", "上一版(融合核) 短", 18801)
        run("ninfer:v100-gdnqpn", "新版(QPN) 短", 18802)
        run("ninfer:v1.5.0", "master 短", 18803)
        run("ninfer:v100-fusedgdn", "上一版 长", 18804, max_tokens=192)
        run("ninfer:v100-gdnqpn", "新版 长", 18805, max_tokens=192)
        run("ninfer:v1.5.0", "master 长", 18806, max_tokens=192)
    finally:
        subprocess.run(["docker", "start", "llama-cpp-text"], capture_output=True)
