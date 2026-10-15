#!/usr/bin/env python3
"""新 QPN 路由的严格验收：长文生成 + 长上下文，三轮镜像对照。"""
import json
import re
import subprocess
import time

MODEL = "/workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
BASE = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-itzy", "--model-id", "qwen",
        "--max-context", "200000", "--prefill-chunk", "2048", "--kv-capacity", "auto",
        "--kv-dtype", "int8", "--max-concurrency", "2", "--spec", "mtp", "--draft-tokens", "4",
        "--lm-head-draft", "--preserve-thinking", "--vision"]

LONG_PROMPT = "写一篇六百字的散文，主题是秋天的黄昏，要求有具体景物描写。"
filler = "".join("第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i)
                 for i in range(1, 1050))
CTX_PROMPT = "秋天到了。" + filler + "\n请用三百字写一段关于秋天的散文。"


def run(image, label, port, prompt, max_tokens):
    name = f"ab{port}"
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", "device=1",
           "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
           "--entrypoint", "ninfer-serve", image, MODEL] + BASE
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(cmd, capture_output=True)
    for _ in range(180):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            break
        time.sleep(5)
    else:
        print(f"  [{label}] 启动失败")
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return None, None
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    body = {"model": "qwen", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    out_path = f"/tmp/ab2-{port}.json"
    r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "2400", "-o", out_path,
                        "-w", "%{http_code}", f"http://{ip}:8000/v1/chat/completions",
                        "-H", "Authorization: Bearer sk-itzy",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps(body)], capture_output=True, text=True)
    log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    lines = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
    s = lines[-1] if lines else ""
    g = lambda p: (re.search(p, s).group(1) if re.search(p, s) else "?")
    usage = {}
    content = ""
    try:
        resp = json.load(open(out_path))
        usage = resp.get("usage", {})
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        pass
    print("  [%s] http=%s prompt_tok=%s decode=%s tok/s prefill=%s TTFT=%s accept=%s" % (
        label, r.stdout, usage.get("prompt_tokens"), g(r"decode ([\d.]+)"),
        g(r"prefill ([\d.]+)"), g(r"TTFT ([\d]+)"),
        g(r"accepted ([\d,]+/[\d,]+ \([\d.]+%\))")))
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return content, s


if __name__ == "__main__":
    subprocess.run(["docker", "stop", "llama-cpp-text"], capture_output=True)
    time.sleep(5)
    outs = {}
    try:
        for tag, image in (("new", "ninfer:v100-gdnqpn"), ("old", "ninfer:v100-fusedgdn"),
                           ("master", "ninfer:v1.5.0")):
            outs[tag] = run(image, f"{tag} 600字", 18901 + hash(tag) % 50, LONG_PROMPT, 600)[0]
        for tag, image in (("new", "ninfer:v100-gdnqpn"), ("master", "ninfer:v1.5.0")):
            outs[tag + "-ctx"] = run(image, f"{tag} 27K上下文", 18951 + hash(tag) % 50,
                                     CTX_PROMPT, 192)[0]
    finally:
        subprocess.run(["docker", "start", "llama-cpp-text"], capture_output=True)
    print("\n=== 输出一致性 ===")
    def norm(x):
        return re.sub(r"\s+", "", x or "")
    print("600字: new==old:", norm(outs.get("new")) == norm(outs.get("old")),
          "| new==master:", norm(outs.get("new")) == norm(outs.get("master")))
    print("27K  : new==master:", norm(outs.get("new-ctx")) == norm(outs.get("master-ctx")))
    for k, v in outs.items():
        print(f"  {k}: {len(v or '')} 字, 头 40: {norm(v)[:40]}")
