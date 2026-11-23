# S0 结果 — Inkling 2 Mini on V100 (sm_70)

2026-11-23，V100-PCIE-32GB (GPU 1)，CUDA 12.8，`CMAKE_CUDA_ARCHITECTURES=70`。

## 成果

Inkling 2 Mini（looped MoE，26.4B params，3.4B active）在 V100 上完整运行：

- **转换**：`tools/convert/inkling2.py` 从任意 GGUF release 打包 `.ninfer`（text + vision + dspark）
- **加载**：GGUF 块格式全部支持（Q8_0/Q6_K/Q5_K/Q4_K/IQ4_XS/IQ4_NL/NVFP4/MXFP4）
- **推理**：`src/models/inkling2/` executor（48 visits、深度 mix、controller、NVFP4 expert 路由）
- **投机**：DSpark drafter（3 layers over block of 8，~42% 接受率）
- **视觉**：ViT tower（BF16 / Q8_0 mmproj）
- **上游兼容**：WaveCut 的 NInfer-v3 制品直接加载（`Inkling-2-Mini-UD-IQ4_XS-NInfer-v3`）

## V100 实测

| 指标 | 值 |
|---|---|
| decode | **~70 tok/s**（DSpark 3 drafts，42% accepted） |
| prefill | **~12,500 tok/s** |
| max context | **600,000 tokens**（int8 KV，VRAM 32 GB 满） |
| plain decode（--spec none） | ~66 tok/s |

phase timing：plain 15.5 ms/轮，draft 2.1 ms/轮，verify 19.5 ms/轮（4 token forward pass）。

## 改动

1. **GGUF NVFP4/MXFP4 块格式支持**：`GgmlType::NVFP4(64,36)`、`MXFP4(32,17)`，`has_matrix_kernel()` 排除 MXFP4
2. **sm70 桥接**：wavecut 的 `src/ops/linear/gguf/` 目录全量拉入（MoE/NVFP4/MXFP4 kernel、MMQ launch、vec launch）
3. **`third_party/ggml-quants` 头文件更新**：cp-async、mmq-config-ampere、mmq-load-tiles、mmq.cuh
4. **Inkling 2 模型家族**：`src/models/inkling2/`（executor、kernels、vision、dspark）
5. **`--spec none` 支持**：关闭 DSpark 投机
6. **native q8_g32 路径**：`is_linear_format()`、`project_f32()` native 路径、`bf16_cols_f32`（token-major 修复）；V100 上 GGUF MMQ 更快，q8_g32 仅在有 FP4 tensor core 的硬件上有优势
7. **DSpark 自动启用**：`dspark` 组件 + `max_concurrency == 1` 时自动

## 构建

```
docker exec -d ninfer-build-inkling2 bash -c \
  'cmake --build /build -j16 > /tmp/build.log 2>&1'
docker build -f Dockerfile.1.8.0-inkling2.local -t ninfer:1.8.0-inkling2 .
```

## 生产 compose

`~/code/docker/ai/ninfer-v100/docker-compose.inkling2.yml`：端口 8000，GPU 1，600K ctx。
