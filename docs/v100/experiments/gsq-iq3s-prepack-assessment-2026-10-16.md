# GSQ（IQ3_S 混合量化）制品的 prepack 可行性 / 显存评估（2026-10-16）

**结论先行：把这份制品的权重量化成 FP16 常驻显存（本仓库给 nvfp4/fp8 用的那种 prepack）在这张 32GB V100 上
不可行** —— 全量 FP16 需要 **81.5 GiB**，而现在常驻只有 **11.9 GiB**。真正可行的路线是**保持块量化位宽、
只把布局在加载期重排成"一次干净反量化"的形式**（本仓库 QPN prepack 的思路），显存代价约 +0.3–0.7 GiB，
但需要新内核（本次不动代码）。

---

## 1. 现象与实测（同一二进制 `ninfer:v100-flashv2`、同一参数、同一 27,217 token 请求、256 输出）

| 制品 | 常驻权重 | KV 容量 | TTFT | prefill | decode | MTP 接受率 |
|---|---|---|---|---|---|---|
| `swift-1.5-…-nvfp4-v3` | 19.1 GiB | 254,208 tok | **27.2s** | **1.00k tok/s** | **82.3** | 62.5% |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3` | 11.9 GiB | 400,000 tok | 30.8s | 885 tok/s | 74.3 | **70.8%** |

**更小的制品反而慢 12%（prefill）/ 10%（decode）**，且不是接受率问题（它的接受率更高）。

## 2. kernel 级归因（nsys，`--cuda-graph-trace=node`，27K prefill）

| 类别 | SWIFT(nvfp4) | GSQ(IQ3_S 混合) | Δ |
|---|---|---|---|
| CUTLASS GEMM（fp16 张量核） | 16.377s ×4142 | 16.619s ×**22714** | +0.24s |
| 反量化 | 2.167s ×4096 | **4.508s ×25197** | **+2.34s** |
| 碎内核（"其他"） | 0.607s ×12990 | **3.197s ×31969** | **+2.59s** |
| 注意力（flash） | 4.015s | 3.855s | −0.16s ✓ |
| GDN | 2.797s | 2.548s | −0.25s ✓ |
| 逐元素 / GGUF 关键字类 | 1.176 / 0.979s | 0.255 / 0.240s | −1.7s ✓ |
| **合计（GPU busy）** | **28.59s** | **31.68s** | **+3.09s** |

具体内核：GSQ 侧是 `gguf::dequantize_rows_kernel`（三个变体共约 20,070 次 = 3.70s）+ `gguf::VecStore`
（6,038 次 = 0.89s）+ 被打碎成 `cutlass::Kernel2<s884…>`（22,714 次，总时间却与 SWIFT 的 4,096 次相同）
+ fork 的 `volta_s884gemm_fp16_128x256_ldg8_tn`（1,934 次）；SWIFT 侧则是规整的
`dequant_nvfp4_qpn_to_fp16` / `dequant_fp8_row_to_fp16`（各 2,048 次）+ 单一 CUTLASS 内核。
两边的注意力、GDN 完全一致 ⇒ **差在"权重怎么被消费"**。

## 3. 制品清单（只读文件头 manifest，未加载模型）

`objects`：1,243 个 tensor + 6 个 resource；平均 **3.95 bit/weight**，张量合计 **13.97 GiB**（常驻 11.9 GiB）。

| 格式 | 字节 | 张量数 | bit/weight |
|---|---|---|---|
| gguf_iq3_s | 3.39 GiB | 131 | 3.44 |
| gguf_iq4_xs | 2.81 GiB | 89 | 4.25 |
| gguf_iq3_xxs | 1.93 GiB | 68 | 3.06 |
| gguf_q4_k | 1.83 GiB | 41 | 4.50 |
| q8_g32_fp16 | 1.76 GiB | 23 | 8.50 |
| gguf_iq2_s | 0.75 GiB | 17 | 2.56 |
| bf16 / q6_k / q2_k / iq2_xs / q4_g64 / q5_g64 / 其余 | 1.50 GiB | 774 | — |

⇒ 它不是"一个 IQ3_S"，而是**18 种格式的混合色拉**（这本身就解释了碎片化：不同块布局 → 不同反量化内核）。

按角色的量（"现"= 制品内字节，"FP16"= 2 B/元素常驻副本）：

| 角色 | 现 | FP16 | 张量数 |
|---|---|---|---|
| `layers/*/mlp/gate` | 3.07 GiB | **14.61 GiB** | 64 |
| `layers/*/mlp/up` | 3.09 GiB | **14.61 GiB** | 64 |
| `layers/*/mlp/down` | 2.42 GiB | **10.62 GiB** | 64 |
| `layers/*/gdn/{query,key,value}` | 各 1.18 GiB | 各 5.45 GiB | 48×3 |
| `layers/*/gdn/z`、`gdn/output` | 0.91 / 0.67 | 4.08 / 2.81 | 48 |
| `token_embedding` / `output_head` | 0.38 / 0.67 | 2.37 / 2.37 | 1 |
| `dflash2/layers/*/mlp/{gate,up}` | 各 0.88 | 各 1.66 | 5 |

**全部绑定权重的 FP16 副本 = 81.52 GiB**（现 13.97 GiB，**+61.1 GiB**）。

## 4. 方案 A：FP16 常驻 prepack —— 不可行

运行期可用显存（GSQ 400K 上下文时）实测 **free 3.08 GiB**。

* 全量 FP16：需 **+61 GiB** ✗✗
* 单个格式（只把 `gguf_iq3_s` 转 FP16）：**+12.4 GiB** ✗（IQ4_XS +7.8、IQ3_XXS +8.2、Q4_K +4.7）
* 单个角色（`mlp/gate`）：**+11.5 GiB** ✗；MLP 三件套（gate+up+down）：**+31.2 GiB** ✗
* 能塞进 3.08 GiB 的只有小角色（如 dflash2 草稿 MLP +0.78 GiB、`q8_g32` 全部 +1.55 GiB），**都不是热点** ✗

即便把上下文从 400K 砍到 254K（≈腾 7 GiB）也远远不够 ✗。KV 单价实测 ≈ **46.6 KiB/token**
（1 GiB ≈ 22K tokens）——这也是"权重省下的显存换成了上下文"的量化解释：
GSQ 比 SWIFT 少 7 GiB 权重，正好换来约 150K 额外上下文（400K vs 254K）。

**⇒ 方案 A 否决。**

## 5. 方案 B：保持块量化位宽，只重排布局（本仓库 QPN 思路）—— 可行，但需要新内核

思路：加载期把块量化权重**重排成反量化友好的常驻布局**（与 nvfp4/fp8 的 QPN 同理），运行期用**一个大内核**
（或每个权重一次）完成"重排布局 → FP16 暂存"，而不是像现在这样每层每 chunk 触发约两万个 gguf 逐行内核。

* 显存代价：**≈ 与现格式同字节**（3–4 bit/weight，加上尺度/对齐开销，估 +0.3–0.7 GiB）⇒
  在 3.08 GiB 空闲内**放得下** ✓
* 需覆盖的格式（按收益排序）：IQ3_S + IQ4_XS + IQ3_XXS + Q4_K = **9.96 GiB = 71%** 的张量字节
  （再加 Q2_K/IQ2_* 可到 ~83%）；其余（q8_g32、q4/q5_g64、bf16）本仓库已有 QPN/薄封装路径
* 预期收益：GSQ 相对 SWIFT 的差距是 **+3.09s GPU busy（prefill −12%）**，其中 gguf 反量化+存储约 4.6s、
  GEMM 碎片化多 5.5 倍；把这两项换成规整路径后**有机会追平 SWIFT（28.6s 级）**，即 prefill/decode 各约 +10%
* 风险与工作量：需要为每种块布局写"重排 + 一次反量化"内核（约 4–5 种），并保证**数值逐位等价**
  （现成的验收链：op 级 oracle + 同请求贪心输出逐字对比，与反量化那次实验一样）

## 6. 方案 C：不改 —— 建议（当前）

* 生产继续用 **nvfp4 制品**（更快，且本仓库对 nvfp4/fp8 的预打包路径最成熟）
* GSQ 制品作为**长上下文/省显存**选项保留：牺牲约 10–12% 速度，换 ~150K 额外上下文与 7 GiB 显存
* 若只是"想跑 GSQ 但嫌慢"，先确认用途：如果上下文需求 ≤ 254K，直接用 nvfp4 制品即可，无需任何改动

## 7. 复现清单分析（只读文件头，不加载模型）

```python
import json, math, collections
p = "/mnt/data/dockerdata/llama-cpp/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"
buf = open(p, "rb").read(192 << 20)                 # 头部 + manifest（约 375 KB）
start = buf.index(b"{", 8)                          # magic "NINFER\x00\x03" + 二进制头
depth = 0; in_str = False; esc = False; end = None  # 花括号配对找 JSON 结尾
for i in range(start, len(buf)):
    c = buf[i]
    if in_str:
        if esc: esc = False
        elif c == 0x5c: esc = True
        elif c == 0x22: in_str = False
        continue
    if c == 0x22: in_str = True
    elif c == 0x7b: depth += 1
    elif c == 0x7d:
        depth -= 1
        if depth == 0: end = i + 1; break
man = json.loads(buf[start:end])
tensors = [o for o in man["objects"] if o["kind"] == "tensor"]
b = sum(o["bytes"] for o in tensors); el = sum(math.prod(o["shape"]) for o in tensors)
print(f"{len(tensors)} 个张量, {b/2**30:.2f} GiB, 平均 {b*8/el:.2f} bit/weight")
print(collections.Counter(o["format"] for o in tensors).most_common(5))
```
