# Volta flash-attention prefill route (2026-10-15)

本仓库 V100 增量。这是 [gdn-recurrent-and-prefill-gap-2026-10-15.md](gdn-recurrent-and-prefill-gap-2026-10-15.md)
里定位到的**最后一个 prefill 差距**的移植：上游参考构建的 prefill 注意力走 vendored llama.cpp 的
tensor-core flash 内核（27K prefill 里 3.93s），本线走自研 int8 `prompt_kernel`（7.28s）。

结论先行：**路线已实现且数值正确（int8/bf16 的 oracle 都通过），但在本引擎当前的分块 prefill 调用形态下
还不划算，因此默认关闭（opt-in）**。原因与数据见下。

## 做了什么

新增 `src/ops/launcher/gqa_attention_volta_flash.cu`，复用本仓库已有的 flash 桥
（`ops/linear/gguf/ggml_fattn.h` 的 `gguf::flash_attention`，vendored `third_party/ggml-quants`）：

1. 调用方先 append K/V（与 sm_70 prompt 路线相同的 `kv_cache_append_batch_launch`）；
2. 一次 gather：paged 缓存 → 连续 FP16（布局 `[head][slot][d]`，桥要求的 stride）；int8 缓存顺便反量化，
   K 走 Hadamard 旋转基（Q 相应旋转）；
3. 逐 128 列 Q-block：用 device 侧 positions 现建因果 mask、跑 flash 内核、把 FP32 结果写回 BF16；
4. 门控：宽度 ≥ 128、单序列、可见 key ≤ 32768（gather 平面每 key 每 kv head 512 B）；
   `NINFER_ATTN_VOLTA_FLASH=1` 打开（bf16），`NINFER_ATTN_VOLTA_FLASH_INT8=1` 另开 int8；
   两个入口（growing 与 cached）都已接入，容量随宽度区间正确预留。

## 用 oracle harness 定位并修掉的三个真 bug

`tests/ops/softmax_attention/causal_cache.cpp` 的 `run_int8_prompt_cases` 本来就有宽度 300–1500 的
int8 用例（≥ 该路线的 128 门槛），harness 按 `causal_softmax_attention_workspace_capacity_bytes` 开
arena、用独立 FP64 oracle 逐元素比对并打印 `actual/reference`，单轮迭代约 40 秒。用它修掉：

1. **容量门控写错**：`min_width <= 128`，而调用方会传 `min_width == max_width == width`（如 300）→
   容量算成 0 → 运行时 `std::bad_alloc`。改为 `max_width >= 128`。
2. **分配未按 256 上取整**：分配器用 `max(256, max_visible_keys)`，而 gather/容量用 `round_up_keys()` →
   可见 key 不是 256 的整数倍时越界写坏相邻缓冲。统一到 `causal_attention_volta_flash_key_extent()`。
3. **每个 Q-block 的 key 上界漏掉前缀**（主因）：写成了 `round_up(begin + tokens)`，而上游是
   `base = 可见key数 − width` 再 `round_up(base + begin + tokens)`。有前缀时（接续/分块 prefill）
   后面的 key 根本没被看到，误差随 token 增大。修好后 int8 用例 0 失败、输出与 FP64 oracle 一致。

定位手段：宿主↔设备逐值对比（把 launcher 的 gathered K/V 在 `cudaStreamSynchronize` 之后 dump，与
harness `HostCache` 的解码值逐元素对齐），先证明 gather 完全正确，从而把范围收敛到"其它环节"；
再加一个 `zero_q` 用例（Q=0 ⇒ 输出 = 可见 key 的 V 均值，与 K 无关）把 K 从怀疑名单里排除。

## 验收

* `./tests/ninfer_tests ninfer_softmax_attention_test`（全存储）**PASS**；
* `--int8-prompt-only` 默认（走 prompt 路线）**PASS**；
* `NINFER_ATTN_VOLTA_FLASH_INT8=1 --int8-prompt-only` **PASS**（证明 int8 flash 分支数值正确）。

## 性能：正确版本还不划算（重要更正）

27,217 token 请求、同参数、GPU 空载（当前生产模型 `...-dflash2-nvfp4-v3.ninfer`）：

| 配置 | TTFT | prefill |
|---|---|---|
| int8 默认（prefill 路线 = 本线 prompt 内核） | **30.1s** | **906 tok/s** |
| int8 + flash（opt-in） | 33.9s | 807 tok/s |
| bf16 + flash（opt-in） | 33.4s | ~815 tok/s |

bench 同一形状也一致（W=2048/ctx=16384：flash 42.2ms vs prompt 41.3ms）。

**原因**：本引擎的 prefill 是分块的（`--prefill-chunk 2048` → 27K 要 14 次调用），而该路线每次调用都要
把**整个可见前缀** staging 成 FP16 并逐 block 建 mask，于是这部分成本按"每块 × 前缀"重复 14 次；上游
引擎把整段 prompt 一次交给注意力算子，只付一次。把 `--prefill-chunk` 放大并不能救（8192/16384 时
TTFT 反而 48–51s，mask 与 per-block 开销随 n_kv 线性增长）。

**注意**：本文件早先记录的"bf16 flash 1150 tok/s（+27%，比 master 快 14%）"是在 bug #3 未修的
*截断 key 范围*版本上测到的——快但不正确，已作废。

## kernel 级证据：瓶颈不是 staging，是 flash 内核本身（更正我先前的判断）

对 int8 的 `--entry append`（W=2048 / ctx=16384）做 nsys（`--cuda-graph-trace=node`），8 次调用：

| 内核 | 合计 | 每次调用 |
|---|---|---|
| **`flash_attn_ext_f16`**（每调用 16 个 128 行块） | **514.9 ms** | **64.4 ms** |
| `flash_attn_stream_k_fixup_general` | 3.5 ms | 0.44 ms |
| `volta_flash_gather` | 1.7 ms | 0.21 ms |
| `volta_flash_build_mask` | 1.1 ms | 0.14 ms |
| convert_q / convert_out | 2.0 ms | 0.25 ms |

即 staging（gather + mask + convert）合计只有 **0.6 ms/调用**，占 1%；**64.4 ms 全在 flash 内核**。
所以我原本计划的"staging 跨 chunk 复用"不可能解决问题。

把 Q-block 从 128 放大到 512（减少 stream-k launch 数）只把每次调用从 64.8 ms 压到 **53.9 ms**，
仍慢于本线 int8 `prompt_kernel` 的 **42.3 ms**。原因是结构性的：flash 内核按 dense tile + mask 计算
（因果下界只到每 block 的 key 上界），而本线的 prompt 内核按因果分块，直接把"看不见的那一半"跳过；
V100 上 int8 DP4A 的因果核在这个形状下比 ggml 的 FP16 tensor-core fattn（含 stream-k/fixup）更快。
上游之所以在这条路上"赢"，是因为它的 prompt 路线是 naive quadratic 内核（bench 1710 ms），不是这条
flash 路线本身有多好。

## 结论与下一步

* 该路线**保持 opt-in、默认关闭**：它数值正确但在 V100 上对本几何（d256-h24-kv4 / 因果 int8 prefill）
  不划算，生产继续用 int8 prompt 路线。
* 若要继续追这 10%：方向应是**让本线 prompt 内核更快**（它已经是两条路线里更快的），例如
  更激进的因果跳块、把 K/V 反量化与 DP4A 流水重叠、或按 visible key 数选择不同的 split；而不是
  搬 flash 内核。
