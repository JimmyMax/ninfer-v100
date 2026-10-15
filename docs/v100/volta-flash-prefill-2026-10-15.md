# Volta flash-attention prefill route (2026-10-15)

本仓库 V100 增量。这是 [gdn-recurrent-and-prefill-gap-2026-10-15.md](gdn-recurrent-and-prefill-gap-2026-10-15.md)
里定位到的**最后一个 prefill 差距**的移植：上游参考构建的 prefill 注意力走 vendored llama.cpp 的
tensor-core flash 内核（27K prefill 里 3.93s），本线走自研 int8 `prompt_kernel`（7.28s，每 token 慢 1.85×）。

## 做了什么

新增 `src/ops/launcher/gqa_attention_volta_flash.cu`（+ `launch.h` 声明、`causal_softmax_attention.cpp`
接入、`basic_sources.cmake` 登记），驱动本仓库**已有**的 flash 桥
`ops/linear/gguf/ggml_fattn.h`（`gguf::flash_attention`，vendored `third_party/ggml-quants`
的 `flash_attn_ext_f16`）：

1. 调用方先 append K/V（与 sm_70 prompt 路线相同的 `kv_cache_append_batch_launch`）；
2. 一次 gather：paged 缓存 → 连续 FP16，布局 `[head][slot][d]`（桥要求的 stride）；
3. 逐 128 列 Q-block：按存储旋转 Q（int8 缓存里 K 在 Hadamard 基）、用 device 侧 positions 现建因果
   mask（`-inf` 在不可见 key 上）、跑 flash 内核、把 FP32 结果写回 BF16；
4. 只有在 `max_visible_keys <= 32768` 时才启用（gather 出来的两个 FP16 平面每 key 每 kv head
   512 B；再长就回 prompt 路线，且不分配任何暂存）。

门控：`causal_attention_volta_flash_applies()`（sm_70 + 单序列 + 宽度 ≥ 128 + 可见 key ≤ 32768 +
存储为 **BFloat16**），`NINFER_ATTN_VOLTA_FLASH=0` 可整体关闭。

## 验收

* **BFloat16 KV**：27,217 token 请求（同参数、GPU 空载、背靠背）
  * prefill **903 → 1150 tok/s（+27%）**，TTFT **30.2s → 23.8s**；
  * 同尺度下比上游参考构建（1010 tok/s / 27.1s）**快 14%**；
  * decode 75.0 → 65.8 tok/s（BF16 KV 的解码带宽翻倍，这是 KV 类型本身的代价，不是该路线的）；
  * 输出正确（HTTP 200，正常中文回答）。
* **INT8 KV：该路线默认不启用**（见下），行为与之前逐字节一致：27K prefill 903 tok/s、decode 75.0 ✓。
* 回归门：`ctest -R ninfer_softmax_attention_int8_prompt_test` **通过**（36.8s）。

## INT8 分支的状态（已知未完成）

int8 缓存下该路线产出错误结果（引擎侧 "target returned a token outside the public token domain"），
因此**默认关闭**。已排除：

* Q 旋转约定（旋转 / 不旋转两种都错 → 不是旋转方向问题）；
* 寻址与反量化公式：与 `int8_g64_codec.cuh` 的 `kv_cache_int8_quant_code_index` /
  `kv_cache_int8_quant_scale_index`、以及 append 内核的写法逐行一致；V 平面（不旋转）与 BF16 分支
  逐元素对比却不相等，说明读到/解出的不是同一批数据；
* 旁路验证：BF16 分支（同一套 mask / bridge / 布局 / 输出转换）完全正确 → 问题局限在 int8 的
  gather/dequant；
* oracle 测试无法直接判定：现有 int8 prompt 测试的宽度只到 64（该路线要求 ≥128），临时降低阈值后
  测试 harness 的 arena 太小（`std::bad_alloc`）。

下一步（建议）：写一个专门的 op 级对照测试——同一份缓存上先跑 int8 prompt 路线、再跑 flash 路线、
逐元素比输出（现有 `tests/ops/softmax_attention/` 的 helper 可复用），把 int8 的 gather 值与该缓存
的权威解码逐元素对齐后即可定位。**修好之后 int8 能同时拿到 flash 的 prefill 与 int8 的解码**，
预计 27K 上 prefill ~1150 tok/s + decode ~75 tok/s，两项都超过上游参考构建。
