#!/usr/bin/env python3
"""1.5 vs merged-1.6 A/B on the V100: 27K prompt, MTP 4 drafts, no n-gram copies.

Both lines read the v2 nvfp4 artifact; the merged line's n-gram default is now 0
but it is passed explicitly so 1.5 (which has no such flag) is comparable.
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/home/max/code/ninfer-v100/bench/v100")
from wait_ready import wait_ready_ip  # noqa: E402

MODEL = "/workspace/models/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2.ninfer"
EXTRA_ENV = os.environ.get("AB_EXTRA", "").split()
BASE = ["--host", "0.0.0.0", "--port", "8000", "--api-key", "sk-ab", "--model-id", "qwen",
        "--max-context", "131072", "--prefill-chunk", "2048", "--kv-capacity", "auto",
        "--kv-dtype", "int8", "--max-concurrency", "1", "--preserve-thinking"]
MTP = ["--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft"]
FILLER_ITEMS = int(os.environ.get("AB_FILLER", "1050"))
FILLER = "".join("第%d条：关于季节、风、雨、光与影的琐碎记录，编号%d。 " % (i, i)
                 for i in range(1, FILLER_ITEMS))
PROMPT = "秋天到了。" + FILLER + "\n请用三百字写一段关于秋天的散文。"


def run(label, image, port, extra):
    name = "ab%d" % port
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "--gpus", "device=1",
                    "-v", "/mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro",
                    "--entrypoint", "ninfer-serve", image, MODEL] + BASE + extra + EXTRA_ENV,
                   capture_output=True)
    ok = False
    for _ in range(180):
        log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        if "listening on" in log.stdout + log.stderr:
            ok = True
            break
        time.sleep(5)
    if not ok:
        print("%-22s FAILED TO START" % label)
        print((log.stdout + log.stderr)[-1500:])
        return None
    ip = subprocess.run(["docker", "inspect", "-f",
                         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                        capture_output=True, text=True).stdout.strip()
    wait_ready_ip(ip)
    body = "/tmp/ab%d.json" % port
    with open(body, "w") as fh:
        json.dump({"model": "qwen", "messages": [{"role": "user", "content": PROMPT}],
                   "max_tokens": 96, "temperature": 0}, fh, ensure_ascii=False)
    subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "1800",
                    "-o", "/tmp/ab%d-resp.json" % port,
                    "http://%s:8000/v1/chat/completions" % ip,
                    "-H", "Authorization: Bearer sk-ab",
                    "-H", "Content-Type: application/json", "-d", "@" + body],
                   capture_output=True)
    log = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    text = log.stdout + log.stderr
    done = [l for l in text.splitlines() if "done |" in l]
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if not done:
        print("%-22s no request-done line" % label)
        print(text[-1200:])
        return None
    line = done[-1]
    out = {"label": label, "line": line.strip()}
    for key, pat in (("prefill", r"prefill ([\d.]+[kKmM]?) tok/s"),
                     ("decode", r"decode ([\d.]+) tok/s"),
                     ("accept", r"accepted ([\d,]+)/([\d,]+) \(([\d.]+)%\)"),
                     ("ttft", r"TTFT ([\d.]+) ?(m?s|s)")):
        m = re.search(pat, line)
        if m:
            out[key] = m.groups()
    try:
        resp = json.load(open("/tmp/ab%d-resp.json" % port))
        out["text"] = resp["choices"][0]["message"].get("content") or ""
        out["usage"] = resp.get("usage", {})
    except Exception as exc:  # noqa: BLE001
        out["text"] = "<no response: %s>" % exc
    print("%-22s %s" % (label, line.strip()))
    return out


def main():
    if os.environ.get("AB_IMAGES"):
        runs = []
        port = 18081
        for spec in os.environ["AB_IMAGES"].split(";"):
            label, image = spec.split("=", 1)
            runs.append((label, image, port))
            port += 1
    else:
        runs = [
            ("1.5 #1", "ninfer:v1.5.0", 18081),
            ("merged #1", "ninfer:v100-merged", 18082),
            ("1.5 #2", "ninfer:v1.5.0", 18083),
            ("merged #2", "ninfer:v100-merged", 18084),
        ]
    results = []
    spec = os.environ.get("AB_SPEC", "mtp")
    for label, image, port in runs:
        # 1.5 predates copy drafting entirely: only the merged line takes the flag.
        if spec == "plain":
            extra = []
        elif spec == "none":
            extra = ["--spec", "none"]
        else:
            extra = MTP + (["--ngram-draft-tokens", "0"] if image != "ninfer:v1.5.0" else [])
        r = run(label, image, port, extra)
        if r:
            results.append(r)
    json.dump(results, open("/tmp/ab_15_vs_merged.json", "w"), ensure_ascii=False, indent=1)
    print("\n=== summary ===")
    for r in results:
        print(r["label"], "| decode", r.get("decode"), "| accept", r.get("accept"),
              "| prefill", r.get("prefill"), "| chars", len(r.get("text", "")))
    texts = {r["label"].split()[0]: r.get("text") for r in results}
    if len(texts) == 2:
        same = texts.get("1.5") == texts.get("merged")
        print("greedy text identical across lines:", same)


if __name__ == "__main__":
    main()
