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

## 性能：换内核后赢下来了（2026-10-15 晚，更正早先的"不划算"结论）

同一型号制品（`...-dflash2-nvfp4-v3.ninfer`）、同一 27,217 token 请求：

| 配置 | TTFT | prefill |
|---|---|---|
| 本线 prompt 路线（旧默认） | 30.1s | 906 tok/s |
| **本线 flash 路线（现在的默认）** | **27.3s** | **~1.00k tok/s** |
| 本线 flash 路线 + bf16 KV | **26.8s** | — |
| 上游参考构建 | 27.0s | 1.01k tok/s |

⇒ **27K prefill 已追平上游参考构建（差 1%），比自己原来的路线快 10%**。bench 同形状：41.1ms → **25.1ms**（−39%）。

### 关键更正：瓶颈是"哪一版 flash 内核"，不是 staging，也不是因果带

早先我归因为"dense tile 多算因果带"（不成立：该调用只做 ~0.42 TFLOP，6.5 TFLOP/s），后来又试过 staging
复用（也不成立：gather+mask+convert 只占 0.6ms/次 = 1%）。真正的差别在于**桥后面的内核版本**：

| 同形状 W=2048 / ctx=16384 / int8 | 每 query-key 对 | 每 op 调用 |
|---|---|---|
| `gguf::flash_attention`（third_party/ggml-quants 的较新 ggml fattn，含 stream-k 变体） | 1.8 ns | 53.9 ms |
| **vendored pinned llama.cpp 内核（本次移植）** | **0.75 ns** | **25.1 ms** |
| 上游参考构建（同一 pinned 内核） | 0.75 ns | 24.7 ms |

## 移植内容

* `third_party/llama_cpp_fattn/`：从上游参考树原样拷入的 vendored 内核（200 KB，pin 62bf73d25）。
* `src/ops/launcher/volta_flash_llama.cu`：`launch_fattn` 的 mma_f16 配置 + 直接 grid（含 stream-K
  fixup），按 d256-h24-kv4 / d256-h16-kv2 两个几何实例化；vendored include 在 CMake 里**只作用于该文件**
  （仓库约定：vendored 头定义 `CUDA_CHECK`/`WARP_SIZE`，与 `src/ops` 冲突）。
* K/V 布局必须是 **[key][head][dim]**（内核硬编码 nb11 = DKQ*kv_heads*2、nb12 = DKQ*2）——早先为 ggml 桥
  改成的 `[head][slot][d]` 已改回。
* 路线**默认开启**（int8 与 bf16），`NINFER_ATTN_VOLTA_FLASH=0` / `NINFER_ATTN_VOLTA_FLASH_INT8=0` 可关。

## 验收

* `--int8-prompt-only`（默认即走 flash）**PASS**（FP64 oracle）；
* 主注意力测试（全存储）**PASS**；
* bf16 + flash 端到端输出正确（中文摘要正常）；
* 关掉路线（`=0`）时行为与旧版本一致（30.1s / 906 tok/s）。

## 仍未追平的部分

* **反量化 +0.49s**：本线刻意选的"精确解码"比上游的移位技巧慢 29%（27K prefill 里占 15% 差距）。
  这是数值契约的选择，不建议为性能改动。
* **27K decode −7%**：来自 MTP 接受率差异（49.7% vs 62.5%），与 prefill 无关。
