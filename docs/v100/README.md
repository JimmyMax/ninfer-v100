# `docs/v100/` — 本仓库的 V100 (sm_70) 增量工作

上游 NInfer-all 面向 3090/4090/5090/PRO 6000；这个目录记录本仓库在**单张 Tesla V100-32GB
（sm_70）**上的增量：把 nvfp4 / fp8 / 混合量化制品的 prefill 与 decode 一路优化到与参考实现
持平（27K prefill 1.00k tok/s、decode 76–78 tok/s），并补齐 v2 制品直读。

| 文档 | 内容 |
|---|---|
| [PLAN.md](PLAN.md) | 路线图 S0–S5：目标、改动点、验收标准、工期、风险与回退 |
| [baseline-2026-10-12.md](baseline-2026-10-12.md) | 第一条 V100-32GB 基线：Gemma 4 可用性与两线 A/B 数据 |
| [S0.5-port-inventory.md](S0.5-port-inventory.md) | S0.5 的移植清单：来源提交、落点结构、分步计划与验收标准 |
| [S0.5-result-2026-10-13.md](S0.5-result-2026-10-13.md) | S0.5 首次 A/B 结果：短上下文 +5.3%，验收标准需改为 oracle |
| [S0.5-oracle-and-longctx-2026-10-13.md](S0.5-oracle-and-longctx-2026-10-13.md) | oracle 单测（v2 数值正确）+ 42K/152K 长上下文 A/B：只有 +3~6% |
| [profiling-nvfp4-2026-10-13.md](profiling-nvfp4-2026-10-13.md) | **定位测量**：4.3× 差距精确落在 NVFP4 投影路线（85 GB/s / DRAM 9.5%，同形状 fp8/q4/q5/q8 已达 480–620 GB/s） |
| [volta-flash-prefill-2026-10-15.md](volta-flash-prefill-2026-10-15.md) | Flash attention prefill 换核（vendored sm_70 mma）：41.1→25.1 ms，27K TTFT 30.1→27.2s |
| [fp8-swiglu-wide-t-2026-10-16.md](fp8-swiglu-wide-t-2026-10-16.md) | 混合量化制品 prefill 慢 2× 的真因：FP8 SwiGLU 宽 T 落在 SIMT 阶梯；改走 FP32 出口的 CUTLASS |
| [v2-direct-read-2026-10-16.md](v2-direct-read-2026-10-16.md) | v2 制品直读：清单在内存里归一化成 v3 形态，权重原地读 |
| [v1.6-vs-v1.5-2026-10-20.md](v1.6-vs-v1.5-2026-10-20.md) | 外部报告的 1.5→1.6 接受率落差：度量口径、验证端 KV 切分、FP16 范围饱和三条原因与收敛计划 |
| [../../RELEASE_NOTES_1.6.2.md](../../RELEASE_NOTES_1.6.2.md) | v1.6.2 发布说明：九次提交的实测矩阵、代价与已知缺口 |

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
