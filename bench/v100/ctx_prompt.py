#!/usr/bin/env python3
# Adapted from huangserva/ninfer-v100-tpx tools/bench/v100 (Apache License 2.0),
# itself built on the ninfer-v100 sm_70 port.
"""Synthetic long-context prompts used for every speed number in this repository.

A prompt is N numbered filler blocks followed by a fixed task. Two kinds:

  code    Python module fragments (high MTP acceptance, ~258 tokens per block at offset 0)
  zh-doc  Chinese technical prose (lower acceptance,   ~166 tokens per block at offset 0)

Block counts we used and the prompt length ninfer reported for them (offset 0):

  code   29 ->   7,353     zh-doc   47 ->   7,792
  code  124 ->  ~32K       zh-doc  193 ->  ~32K
  code  218 ->  55,884     zh-doc  180 ->  29,766
  code  473 -> 121,929     zh-doc  761 -> 126,212
  code  722 -> 186,420     zh-doc 1162 -> 193,104
  code 1007 -> ~260K       zh-doc 1566 -> ~260K

`offset` shifts the block numbers so that prompts of different lengths share no prefix (the
serve prefix cache would otherwise turn a "cold" run into a partially cached one). Larger block
numbers cost a few more tokens per block; the benchmark scripts always print the length ninfer
actually saw (`prompt_n`).

    python3 ctx_prompt.py --kind code --blocks 29            # print the prompt
    python3 ctx_prompt.py --kind zh-doc --blocks 1162 --stats  # characters / estimated tokens
"""
import argparse

ZH_SENT = (
    "第{n}节　显存带宽与自回归解码。在大语言模型的推理过程中，解码阶段每一步只生成一个 token，"
    "却必须把全部权重从显存读入计算单元一次。以 270 亿参数的模型为例，即使采用 4 bit 量化，"
    "权重体积仍有约 15 GB；若显存带宽为每秒 900 GB，则理论上的解码上限只有每秒 60 个 token 左右，"
    "这就是所谓的内存墙。批量推理可以提高算术强度，因为同一份权重可以服务多个请求；"
    "但单流场景下算术强度始终接近于一，瓶颈永远落在带宽上。第{n}节的结论是："
    "提升解码速度只能从降低权重体积、提高带宽利用率、或减少每步读取次数三个方向入手。"
)

CODE_SENT = (
    "# ---- module part {n} ----\n"
    "class TaskQueue{n}:\n"
    "    def __init__(self, max_concurrency: int = {n}) -> None:\n"
    "        self._pq: list[tuple[int, int, Task]] = []\n"
    "        self._running: set[int] = set()\n"
    "        self._max = max_concurrency\n"
    "        self._seq = 0\n\n"
    "    def submit(self, task: 'Task', priority: int = 0) -> int:\n"
    "        self._seq += 1\n"
    "        heapq.heappush(self._pq, (-priority, self._seq, task))\n"
    "        return self._seq\n\n"
    "    async def _drain{n}(self) -> None:\n"
    "        while self._pq and len(self._running) < self._max:\n"
    "            _, _, task = heapq.heappop(self._pq)\n"
    "            self._running.add(task.id)\n"
    "            asyncio.create_task(self._guard{n}(task))\n\n"
    "    async def _guard{n}(self, task: 'Task') -> None:\n"
    "        try:\n"
    "            await asyncio.wait_for(task.run(), timeout=task.timeout)\n"
    "        finally:\n"
    "            self._running.discard(task.id)\n"
)

ZH_ASK = ("\n\n=== 以上是文档全文 ===\n"
          "请阅读上面全部文档，写一篇 600 字左右的中文综合报告，分三部分："
          "（一）这些章节反复强调的核心论点是什么；"
          "（二）它们共同依赖的量化前提有哪些，请给出具体数字；"
          "（三）如果要在这类硬件上把解码速度提高一倍，按收益排序应该先动哪个环节。"
          "不要逐节罗列，要给出你自己的综合判断。")

CODE_ASK = ("\n\n# === 以上是模块全文 ===\n"
            "# 请参照上面的代码风格，补全一个带优先级队列、并发上限、超时重试和优雅关闭的\n"
            "# 异步任务调度器主循环函数（含类型注解和中文注释），约 120 行。\n")

KINDS = ("code", "zh-doc")

# Block counts that give the context levels used in the README / article tables.
LEVELS = {
    "8K":   {"code": 29,   "zh-doc": 47},
    "32K":  {"code": 124,  "zh-doc": 193},
    "128K": {"code": 473,  "zh-doc": 761},
    "186K": {"code": 722,  "zh-doc": 1162},   # the headline 186K code / 193K Chinese prompts
    # 256K: matrix.sh runs this level at block offset 300000, where every block number has six
    # digits (about 277 tokens per code block, 172 per zh-doc block), so 1007 / 1566 blocks would
    # exceed --max-context 262144. 939 / 1516 land at about 260K, like the offset-0 prompts
    # (1007 / 1566 blocks = 260,283 / 260,976 tokens) used for the published 256K numbers.
    "256K": {"code": 939, "zh-doc": 1516},
}

TOK_PER_BLOCK = {"code": 258.0, "zh-doc": 166.2}   # measured at offset 0, for estimates only


def make_filler(kind, n_blocks, offset=0):
    tpl = ZH_SENT if kind == "zh-doc" else CODE_SENT
    return "".join(tpl.format(n=i + offset) for i in range(1, n_blocks + 1))


def build_prompt(kind, n_blocks, offset=0):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    tail = ZH_ASK if kind == "zh-doc" else CODE_ASK
    return make_filler(kind, n_blocks, offset) + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="print size instead of the prompt")
    a = ap.parse_args()
    p = build_prompt(a.kind, a.blocks, a.offset)
    if a.stats:
        print(f"kind={a.kind} blocks={a.blocks} offset={a.offset} chars={len(p)} "
              f"est_tokens~{int(a.blocks * TOK_PER_BLOCK[a.kind])}")
    else:
        print(p, end="")


if __name__ == "__main__":
    main()
