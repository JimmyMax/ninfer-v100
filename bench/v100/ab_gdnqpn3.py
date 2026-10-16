#!/usr/bin/env python3
"""新 QPN 路由的严格验收：长文 + 长上下文，多臂对照，逐臂硬超时与清理。

用法: python3 ab_gdnqpn3.py <tag1=image1> [tag2=image2 ...]
输出一行 JSON 结果，便于外部收集。
"""
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

PROMPTS = {
    "long": ("写一篇六百字的散文，主题是秋天的黄昏，要求有具体景物描写。", 600),
    "ctx27k": ("秋天到了。" + "".join(
        "第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i)
        for i in range(1, 1050)) + "\n请用三百字写一段关于秋天的散文。", 192),
}


def run_arm(tag, image, case, port, timeout_s=900):
    prompt, max_tokens = PROMPTS[case]
    name = f"v{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "--gpus", "device=1",
                    "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
                    "--entrypoint", "ninfer-serve", image, MODEL] + BASE, capture_output=True)
    started = False
    for _ in range(120):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            started = True
            break
        time.sleep(5)
    rec = {"tag": tag, "case": case, "started": started}
    if started:
        ip = subprocess.run(["docker", "inspect", "-f",
                             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                            capture_output=True, text=True).stdout.strip()
        wait_ready_ip(ip)  # /health 就绪探针：日志行可能来自上次启动
        body = {"model": "qwen", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0}
        out_path = f"/tmp/v{port}.json"
        body_path = f"/tmp/v{port}-body.json"
        with open(body_path, "w") as fh:
            json.dump(body, fh, ensure_ascii=False)
        t0 = time.time()
        subprocess.run(["curl", "-s", "--noproxy", "*", "-m", str(timeout_s), "-o", out_path,
                        f"http://{ip}:8000/v1/chat/completions",
                        "-H", "Authorization: Bearer sk-itzy",
                        "-H", "Content-Type: application/json",
                        "-d", f"@{body_path}"], capture_output=True)
        rec["wall_s"] = round(time.time() - t0, 1)
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        lines = [x for x in (log.stdout + log.stderr).splitlines() if "done |" in x]
        s = lines[-1] if lines else ""
        for key, pat in (("decode", r"decode ([\d.]+)"), ("prefill", r"prefill ([\d.]+)"),
                         ("ttft_ms", r"TTFT ([\d.]+)"),
                         ("accepted", r"accepted ([^|]+?) \|")):
            m = re.search(pat, s)
            rec[key] = m.group(1).strip() if m else None
        try:
            resp = json.load(open(out_path))
            msg = resp["choices"][0]["message"]
            rec["text"] = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
            rec["usage"] = resp.get("usage", {})
        except Exception as exc:
            rec["text"] = ""
            rec["error"] = str(exc)[:120]
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


if __name__ == "__main__":
    arms = [a.split("=", 1) for a in sys.argv[1:]]
    cases = sys.argv[0] and ["long", "ctx27k"]
    results = []
    for case in cases:
        for i, (tag, image) in enumerate(arms):
            results.append(run_arm(f"{tag}", image, case, 19000 + i * 10 + (0 if case == "long" else 5)))
    print("=== 汇总 ===", flush=True)
    for r in results:
        print(f"{r['case']:>6} {r['tag']:>8}: decode={r['decode']} tok/s prefill={r['prefill']} "
              f"tok/s ttft={r['ttft_ms']}ms accept={r['accepted']} 文本={len(r.get('text', ''))}字",
              flush=True)
    by = {(r["case"], r["tag"]): r for r in results}
    norm = lambda x: re.sub(r"\s+", "", x or "")
    for case in cases:
        tags = [t for c, t in by if c == case]
        if len(tags) >= 2:
            base = by[(case, tags[0])].get("text", "")
            print(f"  [{case}] {tags[0]} vs 其余:", {t: norm(by[(case, t)].get("text")) == norm(base)
                                                    for t in tags[1:]}, flush=True)
