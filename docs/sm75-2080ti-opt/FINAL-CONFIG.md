# 2080 Ti 最终性能配置（2026-11-08 验证）

## compose.yaml（~/docker/ninfer/compose.yaml）

```yaml
command:
  - ninfer-serve
  - /workspace/models/swift15_iq3s_mtp.ninfer   # 11.9 GB GGUF IQ3_S + MTP
  - --host 0.0.0.0 --port 8000 --api-key sk-itzy --model-id qwen
  - --max-context 200000 --prefill-chunk 2048
  - --kv-capacity auto --kv-dtype int8 --max-concurrency 1
  - --spec mtp --draft-tokens 6 --ngram-draft-tokens 16 --ngram-min-match 8
  - --adaptive-mtp --lm-head-draft --gdn-state-fp16
  - --preserve-thinking --vision
```

外部条件：`sudo nvidia-smi -pl 250`（每次重启需重设）。

## 每项配置的依据

|  flag | 依据 |
|---|---|
| `--spec mtp --draft-tokens 6` | EXPERIMENT spec 调优：bench 128.9 tok/s（bare 30.4 的 4.2 倍）|
| `--ngram-draft-tokens 16 --ngram-min-match 8` | ngram+MTP 混合 draft，长重复内容接受率高 |
| `--adaptive-mtp` | 接受率低时自动降 draft 深度，避免负收益 |
| `--lm-head-draft` | +3% decode，零成本（EXPERIMENT-6 验证 byte-identical）|
| `--gdn-state-fp16` | GDN state 半精度，省显存带宽，greedy 验证一致 |
| `--kv-dtype int8` | KV int8-group64，ppl 变化 <0.05%（可接受），省一半 KV 带宽 |
| `--prefill-chunk 2048` | prefill 分块大小；2048 在 22GB 上平衡 workspace 与效率 |
| `--max-concurrency 1` | 单并发；workspace 预留最小化，200k ctx 可放下 |
| power limit 250W | EXPERIMENT：+9.3% decode（200→250W），非重启持久 |

## 实测性能（API 实测，2026-11-08）

| 场景 | 结果 |
|---|---|
| 代码生成 500 tok | **49.4 tok/s** overall（含 prefill）|
| 散文 400 tok | **39.8 tok/s** overall |
| 8k prompt + 200 tok | prefill ~700 tok/s，overall 5.9 tok/s（长 prefill 主导）|
| bench 2K prefill | 708 tok/s |
| bench bare decode | 30.4 tok/s（天花板，LSU-bound）|
| bench spec decode | 128.9 tok/s（2K corpus）|

## 已验证的优化（本分支 16+ commits）

- ✅ GGUF wide products → dequant+FP16 cuBLAS（prefill +7.5%）
- ✅ power limit 250W（+9.3% decode）
- ✅ spec 全套调优（59.9→128.9 bench）
- ✅ GDN recurrent 开关（greedy 一致，与 two_stage 打平，保留为 `NINFER_GDN_RECURRENT=1`）

## 已证伪的方向（勿重试，见 docs/sm75-2080ti-opt/）

- ❌ CUTLASS GEMM / 手动 split-K（cuBLAS 已 79% 效率 + 自带 split-K）
- ❌ MMQ tile 缩小 / FP16-accum / vec 向量化 / smem staging（机制分析见 EXPERIMENT 6-9）
- ❌ store_plane 消除（近 roofline，~1%）
- ❌ attention 8-warp tile（需 int4 V，质量损失）
- ❌ 小 kernel 融合（上限 1-2%）

## 质量约束

所有启用的优化均通过 greedy 逐字一致验证（4+ prompts × 200 tokens）。
未启用任何有损选项（rk8v4 KV、Q6 head、keep-list head）。
