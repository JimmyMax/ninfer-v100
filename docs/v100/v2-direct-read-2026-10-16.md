# v2 制品直读（2026-10-16）

## 结论

引擎现在**直接读 v2 制品**（`Swift-1.5-…-DFlash2-V2.ninfer`），不需要离线升级、不需要 24 GB 副本，
性能与"升级成 v3 之后再读"以及 master 完全一致：

| 跑法 | TTFT / prefill | decode | MTP 接受率 |
|---|---|---|---|
| **v2 原件（直读）** | **27.3s / 998.5 tok/s** | 82.6 | 69.2% |
| v2 升级成 v3 后读 | 27.3s / 997.6 tok/s | 82.7 | 69.2% |
| master 读 v2 | 27.1s / 1.00k tok/s | 75.7 | 63.9% |

前两种的输出文本**逐字一致**。v3 路径未受影响的证据：生产制品 SWIFT（v3 nvfp4）用同一二进制
**997.1 tok/s / decode 78.3 / MTP 54.3%**，与改动前逐项相同；`ctest -R artifact` **4/4 通过**。

## 为什么可以原地直读

v2 与 v3 的差别**只在清单**。实测同一个制品（1184 个张量）在升级前后的对象偏移
**1184/1184 完全相同**、对象名字集合相同，升级只是：

* 词汇改名（`NVFP4` → `nvfp4`、`Q4G64_F16S` → `q4_g64_fp16`、`contiguous-le-v1` →
  `contiguous_le_v1` …）；
* 由 v2 的**角色名**合成 1513 条 `bindings`（例如一个 `mlp/gate_up` 拆成 `mlp/gate` + `mlp/up`
  两个 part，`attention/query_key_gate_value` 拆成四个 …）；
* 由 `identity` 合成 `components` 的 config（text/vision/mtp/dflash2）、`uses`、`files`、
  `metadata`、`provenance`。

所以权重可以原地读：v2 头的 JSON 长度与 v3 同位置，只是头是 16 字节、没有 UUID。

## 实现

* `src/artifact/reader.cpp`：接受 v2 magic（`NINFER\0\2`，16 字节头）。v2 没有 UUID，用
  `目录文本 + 文件字节数` 的 128 位 FNV-1a 指纹合成 `ArtifactId`（v2 没有 continuation 文件，
  指纹不会跨文件比较）；`entry_payload_start = align_up(16 + json_len, 4096)`；其余
  （`files[0].payload_bytes` = 头与目录之后剩余字节）沿用读取器原有校验。
* `src/artifact/v2_directory.{h,cpp}`（新增）：`normalize_v2_directory()` 把 `{identity, objects}`
  变成 v3 形态的目录 JSON，再交给**同一个 `parse_directory`**。规则是
  `tools/upgrade_ninfer_v2_to_v3.py` 中 `make_directory` 的逐条移植（格式/布局表、7 组 config
  合成器、`put`/`split`、`encoded_bytes` 长度校验、`_input_names` 与 1513 条 bindings 的完整性
  断言）。那个工具因此是这份实现的**规格与等价性 oracle**。
* 白名单与工具相同（5 个官方 v2 输入：`qwen3.8-27b/…`、`qwen3.6-27b/…`、`qwen3.6-35b-a3b/…`）。
  白名单之外的 v2 制品仍旧报错并指向离线工具——不对未知布局做猜测。

**纯加载期**：热路径（v3 的 `parse_directory`、binder、materializer、所有算子）逐字未动。

## 复现

```bash
# 直接用 v2 原件启动（无需先跑升级工具）
docker run --rm --gpus device=1 \
  -v /mnt/data/ninfer-build-gemma4:/build:ro \
  -v /mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro \
  --entrypoint /build/apps/ninfer-serve ninfer:sm70-buildenv \
  /workspace/models/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2.ninfer \
  --host 0.0.0.0 --port 8000 --api-key sk-itzy --model-id qwen --max-context 200000 \
  --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 --max-concurrency 2 \
  --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0 --preserve-thinking --vision
```

## 对照：离线升级工具仍在

`tools/upgrade_ninfer_v2_to_v3.py`（1 分 21 秒，保真）仍是把一个 v2 制品**永久**升成 v3 的手段
（例如要让旧版本引擎读、或想把前端资源换成维护中的 chat template）。直读只是省掉了这一步。
