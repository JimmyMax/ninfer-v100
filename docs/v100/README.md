# `docs/v100/` — 本仓库的 V100 (sm_70) 增量工作

上游 NInfer-all 面向 3090/4090/5090/PRO 6000；这个目录记录本仓库在**单张 Tesla V100** 上的
增量：Gemma 4 与 gguf 家族的可用性，以及把 `JimmyMax/ninfer-v100`（Volta 专精线）的加速
技术正向迁移到这条线上。

| 文档 | 内容 |
|---|---|
| [PLAN.md](PLAN.md) | 路线图 S0–S5：目标、改动点、验收标准、工期、风险与回退 |
| [baseline-2026-10-12.md](baseline-2026-10-12.md) | 第一条 V100-32GB 基线：Gemma 4 可用性与两线 A/B 数据 |
| [S0.5-port-inventory.md](S0.5-port-inventory.md) | S0.5 的移植清单：来源提交、落点结构、分步计划与验收标准 |

相关资产：

| 路径 | 用途 |
|---|---|
| `Dockerfile.sm70.local` | CUDA 12.8 + `-DCMAKE_CUDA_ARCHITECTURES=70` 全量构建 |
| `Dockerfile.sm70-runtime.local` | 复用已编好的 `ninfer-serve` 秒级重打包 |
| `bench/v100/compare_lines.py` | 两条实现的 A/B 对照（预热后连测，输出 markdown 表） |
| `deploy/compose.gemma-v100.example.yml` | Gemma 线的 compose 示例（生产副本在 Docker 仓库里） |

快速开始：

```bash
python3 bench/v100/compare_lines.py \
  --image ninfer:v100-gemma4 --image ninfer:v1.5.0 \
  --model /workspace/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer \
  --extra-args "--spec mtp --draft-tokens 4 --lm-head-draft" --runs 2
```
