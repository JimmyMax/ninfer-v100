# NInfer v1.7.1

V100 (sm_70) + RTX 4080 SUPER (sm_89) 单镜像双架构发布。一张镜像跑两张卡。

## 镜像

`ninfer:1.7.1` — 内含 `ninfer-serve.sm70` 与 `ninfer-serve.sm89` 两个单架构二进制，
`/usr/local/bin/ninfer-serve` 调度脚本按容器内可见 GPU 的 compute capability 自动选择
（7.0 → sm_70，其余 → sm_89，`NINFER_IMAGE_ARCH` 环境变量可强制覆盖）。

构建方式变更：v1.7.0 及之前按架构分镜像（`1.7.0` / `1.7.0-sm89`）。`70;89` 双架构合编
实测不可行——少数 int8/splitk 内核单元在 sm_89 ptxas 通道病理性放大 30-174 倍
（单文件最慢 345 分钟，全树 ~10 小时）；两个单架构增量构建各几分钟。二进制分发 +
按卡调度，运行时行为与合编完全一致。

## 修复

- **NVFP4 制品 MTP 接受率归零**（`0311d39`）：Q6_K dense prepack 把 proposal 头
  无条件纳入 Volta 布局重排名单，NVFP4 制品的 Q4_G64 proposal 头被 panel 重排后
  草稿全错（decode 28 tok/s，接受率 0），DFlash2 直接启动失败。修复后 NVFP4 27B
  MTP 恢复 65.5 tok/s（bench tg128，接受率 0.44）。
- **多架构编译守卫**（`ec1d53a`）：19 个 Volta `.cu` 的 `NINFER_TIER_VOLTA` 守卫
  在非 Volta 设备编译通道误生效，补 `__CUDA_ARCH__ == 700` 条件。

## 性能（相对 v1.7.0）

- GGUF Q4_K_M 27B：Q6_K dense prepack，decode +8.7%（35.1 → 38.1 tok/s nospec，
  MTP-3 54.1 → 57.2）。
- Gemma 4 26B/31B：constexpr-K 内核分派 + 滑窗 decode attention 重写
  （`573e460`/`c6e1d73`/`10bac0d`）。

## 与 v1.5.0 全面对比（V100，NVFP4 27B 制品，greedy + int8 KV）

| 场景 | v1.5.0 | v1.7.1 | 结果 |
|---|---|---|---|
| nospec decode 短/7K | 33.9 / 32.7 | 34.3 / 33.0 | 胜 |
| MTP d3 散文 | 68.8 | 71.5 | 胜 |
| MTP d4 三 prompt | 70.8 | 78.5 | 胜 +10.9% |
| MTP d3 7K / 27K | 76.0 / 77.4 | 78.8 / 78.5 | 胜 |
| prefill 27K chunk4096 | 965.6 | 982.0 | 胜 +1.7% |

decode 全场景胜出，接受率持平或更高。prefill 在 chunk 2048 下略负（−0.8~−2.7%），
chunk 4096 下反超——v1.5.0 对大 chunk 无收益，v1.7.1 有。

## 升级

- V100 服务：`image: ninfer:1.7.1`，无需其他改动。
- 4080 服务：`image: ninfer:1.7.1`（替代 `ninfer:1.7.0-sm89`），无需其他改动。
- NVFP4 制品 MTP 不再需要 `--full-draft-head` 绕过。
- 回退：V100 用 `ninfer:1.7.0`，4080 用 `ninfer:1.7.0-sm89`。
