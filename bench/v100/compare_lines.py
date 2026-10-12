#!/usr/bin/env python3
"""两条实现的 A/B 对照（V100 / sm_70）。

在**同一张 GPU、同一个 artifact、同一组参数**下依次启动若干镜像，预热后连测，
把 `ninfer-serve` 自己的 `req#N done` 记录解析成一张 markdown 表。

为什么必须预热：V100 上首次请求可能要建立 CUDA graph、预热 KV 页与 L2，
冷启动 decode 可低至稳态的 1/4，直接拿来对比会得出错误结论。

示例：

    # 上游 sm70 线 vs fork 的 V100 线，同一 Qwen artifact
    python3 bench/v100/compare_lines.py \
      --image ninfer:v100-gemma4 --image ninfer:v1.5.0 \
      --model /workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer \
      --extra-args "--spec mtp --draft-tokens 4 --lm-head-draft" \
      --runs 2 --warmup 1

    # 单个镜像调参（比较 --draft-tokens）
    python3 bench/v100/compare_lines.py --image ninfer:v100-gemma4 \
      --model /workspace/models/Gemma4-26B-A4B-...-ninfer-v3.ninfer \
      --extra-args "--draft-tokens 2" --runs 3

输出为 markdown；加 `--json out.json` 同时落原始记录。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

DEFAULT_BASE_ARGS = [
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-context", "200000",
    "--prefill-chunk", "2048",
    "--kv-capacity", "auto",
    "--kv-dtype", "int8",
    "--max-concurrency", "2",
    "--spec", "mtp",
    "--preserve-thinking",
    "--vision",
]

DEFAULT_PROMPT = (
    "详细解释 Transformer 的注意力机制，包括 Q/K/V、多头注意力、位置编码，尽量写长一点。"
)

DONE_RE = re.compile(
    r"req#(?P<req>\d+) done.*?\| output (?P<output>[\d,]+)"
    r".*?\| TTFT (?P<ttft>[\d.]+) ms"
    r".*?\| total (?P<total>[\d.]+)\s*(?P<total_unit>ms|s)"
    r".*?\| prefill (?P<prefill>[\d.]+) tok/s"
    r".*?\| decode (?P<decode>[\d.]+) tok/s"
    r"(?:.*?accepted (?P<acc_num>[\d,]+)/(?P<acc_den>[\d,]+) \((?P<acc_pct>[\d.]+)%\))?"
)
CAPACITY_RE = re.compile(r"capacity \| (?P<body>.*)$")
LOAD_RE = re.compile(r"engine ready \| (?P<name>[^|]+)\| total (?P<total>[\d.]+)s \| weights (?P<weights>[\d.]+) GiB")


def sh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=check)


def logs(name: str) -> str:
    return sh("docker", "logs", name, check=False).stdout + sh(
        "docker", "logs", name, check=False
    ).stderr


def status(name: str) -> str:
    return sh("docker", "inspect", "-f", "{{.State.Status}}", name, check=False).stdout.strip()


@dataclass
class Run:
    image: str
    label: str
    capacity: str = ""
    loaded: str = ""
    records: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.records)

    def avg(self, key: str) -> float | None:
        vals = [r[key] for r in self.records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(self, key: str, suffix: str = "", digits: int = 1) -> str:
        v = self.avg(key)
        return "—" if v is None else f"{v:.{digits}f}{suffix}"


def parse_records(text: str, limit: int) -> list[dict]:
    out: list[dict] = []
    for m in DONE_RE.finditer(text):
        d = m.groupdict()
        total = float(d["total"])
        if d.get("total_unit") == "ms":
            total /= 1000.0
        rec = {
            "req": int(d["req"]),
            "output": int(d["output"].replace(",", "")),
            "ttft_ms": float(d["ttft"]),
            "total_s": total,
            "prefill": float(d["prefill"]),
            "decode": float(d["decode"]),
        }
        if d.get("acc_pct") is not None:
            rec["accept_pct"] = float(d["acc_pct"])
            rec["accept_num"] = int(d["acc_num"].replace(",", ""))
            rec["accept_den"] = int(d["acc_den"].replace(",", ""))
        out.append(rec)
    return out[-limit:] if limit else out


def wait_ready(name: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = logs(name)
        if "listening on" in text:
            return True
        if status(name) in ("exited", "dead"):
            return False
        time.sleep(3)
    return False


def chat(port: int, api_key: str, model_id: str, prompt: str, max_tokens: int, temperature: float) -> None:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )
    subprocess.run(
        [
            "curl", "-sS", "-m", "900",
            f"http://127.0.0.1:{port}/v1/chat/completions",
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-d", body, "-o", "/dev/null",
        ],
        capture_output=True, text=True, check=False,
    )


def run_one(args: argparse.Namespace, image: str, index: int, extra: str) -> Run:
    label = f"{image} [{extra.strip() or '默认参数'}]"
    name = f"ninfer-ab-{index}"
    sh("docker", "rm", "-f", name, check=False)

    cmd = [
        "docker", "run", "-d", "--name", name,
        "--gpus", f"device={args.device}",
        "-v", f"{args.host_mount}:/workspace/models:ro",
        "-p", f"127.0.0.1:{args.port}:8000",
        "--entrypoint", "ninfer-serve",
        image, args.model,
        *args.base_args,
    ]
    if args.api_key:
        cmd += ["--api-key", args.api_key, "--model-id", args.model_id]
    if extra.strip():
        cmd += extra.split()
    if args.draft_tokens is not None:
        cmd += ["--draft-tokens", str(args.draft_tokens)]

    print(f"### {label}", file=sys.stderr)
    proc = sh(*cmd, check=False)
    if proc.returncode != 0:
        return Run(image=image, label=label, error=proc.stderr.strip()[:400])

    if not wait_ready(name, args.startup_timeout):
        err = "\n".join(
            line for line in logs(name).splitlines()
            if re.search(r"ERROR|FATAL|error|failed", line)
        )
        sh("docker", "rm", "-f", name, check=False)
        return Run(image=image, label=label, error=(err or "启动超时")[:600])

    text = logs(name)
    cap = CAPACITY_RE.search(text)
    load = LOAD_RE.search(text)
    result = Run(
        image=image,
        label=label,
        capacity=cap.group("body").strip() if cap else "",
        loaded=load.group(0).strip() if load else "",
    )

    for _ in range(args.warmup):
        chat(args.port, args.api_key, args.model_id, args.warmup_prompt,
             args.warmup_tokens, args.temperature)
    for _ in range(args.runs):
        chat(args.port, args.api_key, args.model_id, args.prompt,
             args.max_tokens, args.temperature)

    result.records = parse_records(logs(name), args.runs)
    if not args.keep:
        sh("docker", "rm", "-f", name, check=False)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", action="append", required=True, help="镜像，可重复")
    p.add_argument("--extra-args", default="", help="追加到所有镜像的参数")
    p.add_argument("--model", required=True, help="容器内 artifact 路径")
    p.add_argument("--model-id", default="qwen")
    p.add_argument("--api-key", default="sk-itzy")
    p.add_argument("--port", type=int, default=18000, help="宿主端口（避开生产 8000）")
    p.add_argument("--device", default="1", help="GPU 序号，V100 一般是 1")
    p.add_argument("--host-mount", default="/mnt/data/dockerdata/llama-cpp/models")
    p.add_argument("--base-args", default=" ".join(DEFAULT_BASE_ARGS))
    p.add_argument("--draft-tokens", type=int, default=None)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--warmup-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--warmup-prompt", default="你好，简单介绍下你自己。")
    p.add_argument("--startup-timeout", type=float, default=420)
    p.add_argument("--keep", action="store_true", help="跑完保留容器")
    p.add_argument("--json", dest="json_out", default=None)
    args = p.parse_args()
    args.base_args = args.base_args.split()

    results = [run_one(args, img, i, args.extra_args) for i, img in enumerate(args.image)]

    print()
    print(f"模型 `{args.model}` ｜ GPU{args.device} ｜ 参数 `{args.base_args + (args.extra_args.split() if args.extra_args else [])}` ｜ 预热 {args.warmup} 次后测 {args.runs} 次")
    print()
    print("| 实现 | decode tok/s | prefill tok/s | TTFT ms | 输出 tok | MTP 接受率 |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        if not r.ok:
            print(f"| `{r.label}` | 失败：{r.error} | | | | |")
            continue
        acc = r.fmt("accept_pct", " %")
        print(f"| `{r.label}` | **{r.fmt('decode')}** | {r.fmt('prefill')} | {r.fmt('ttft_ms', '', 0)} | {r.fmt('output', '', 0)} | {acc} |")
    print()
    for r in results:
        if r.capacity:
            print(f"* `{r.label}` → `capacity | {r.capacity}`")
        if r.loaded:
            print(f"* `{r.label}` → `{r.loaded}`")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([r.__dict__ for r in results], fh, ensure_ascii=False, indent=2)
        print(f"\n原始记录：{args.json_out}")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
