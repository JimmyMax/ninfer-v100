# FP8 SwiGLU 宽 T：混合量化制品 prefill 慢 2× 的真因（2026-10-16）

## 症状

用 v2 制品 `Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2.ninfer`（离线升级成 v3 后）在
我们的线上跑 27K 请求：

| | 我们（修复前） | master |
|---|---|---|
| TTFT / prefill | **53.6s / 508 tok/s** ✗ | 27.1s / 1.00k tok/s |
| decode | 84.7 | 75.7 |
| MTP 接受率 | 47/60 | 46/72 |

同一份权重、同一请求、同一参数。master 读原始 v2、读升级后的 v3 副本**结果完全一致**
（27.1s / 1.00k），所以升级本身保真，差的是二进制。我们比 master 慢一倍。

## 归因（nsys 逐内核对比）

GPU busy：我们 55.05s vs master 28.78s。逐类拆开后**只有一处差异**：

| 内核 | 我们 | master |
|---|---|---|
| CUTLASS GEMM 合计 | 14.88s（(15,272)/(15,40)/(15,128)/(15,112)） | 15.49s（(16,…) 同形状） |
| flash attention | 3.95s ×896 | 3.92s ×432 |
| GDN | 2.16s | 2.81s |
| 反量化（fp8/nvfp4/bf16→fp16） | 2.16s | 1.62s |
| **`fp8_simt_kernel<…, Fp8SwiGluOutput, …>`** | **27.67s × 54520，grid=(2176,1,1)** ✗✗ | 无 |

完整内核名解出来是：

```
fp8_simt_kernel<Fp8Geometry<34816,5120>, 4, Fp8SimtSchedule<…>, Fp8SwiGluOutput, …>
grid=(2176,1,1)      # 2176 = 34816/16，一维网格：只在 N 方向并行，T 在核内串行
```

即 **FP8 的 MLP gate_up（34816×5120）落到 SIMT 后备**。注意：**这 27.7s 就是全部差距**
（55.05 − 27.67 ≈ 27.4 ≈ master 的 28.78），而 master 同形状走 CUTLASS（`grid=(16,272)`，T 也并行）。

## 根因

`src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp` 的 sm_70 钩子**只接管 t ≤ 32**：

```cpp
if (weight.n == Fp8N34816K5120::kOutputRows && x.ne[1] <= kFp8VoltaQpnMaxTokens && …) { … QPN 拆分 … }
```

宽 T 交回上游阶梯 → `resolve_route` 给 A16 → `launch_a16` 按**每 4 个 token 一块**循环调用
small-t 内核（`fp8_linear_swiglu_small_t_launch`），而那就是上面那个 SIMT 内核。T=2048 时每层
512 次一维网格的小内核，全模型累起来 27.7s。

为什么别的 FP8 形状没事：`fp8_dispatch` 的宽 T 路线有 **256 MiB 的权重暂存上限**
（`kFp8CutlassWeightLimitBytes`，为了不让 arena 被最宽上下文撑爆）。`34816×5120` 的 FP16 暂存是
**356 MB > 256 MiB** → 既不走 CUTLASS，QPN 又只吃 t ≤ 32 → 掉进 SIMT。其余形状
（14336×5120 = 147MB、16384×5120 = 168MB、5120×17408 = 178MB…）都在限内，所以只有这一个中招。

**因此"混合量化制品慢"与缺 q4/q5/w8 内核无关**：那两个二进制里 q4/q5/w8 相关内核总计只有
0.2–4.5%（master 的 `w8_volta_mma_gemm_kernel` 只有 0.19s）。移植它们对本制品的收益约等于 0。

## 修复

新增一条 **FP32 出口的宽 T CUTLASS 路线**（`fp8_linear_swiglu_cutlass_sm70_launch_f32`）：

1. 复用同一个 sm_70 mma 内核（FP8→FP16 反量化、bf16 激活→FP16、CUTLASS `Sm70` 模板 GEMM），
   只把 CUTLASS 的**输出元素型别换成 `float`**（`GemmF32` / `EpilogueOpF32`）；
2. 行缩放在 FP32 里做（`scale_gate_up_rows_f32_kernel`），**不预先舍入到 BF16**；
3. SwiGLU 直接读这个 FP32 平面并写回 BF16 的 `out`（`swiglu_from_f32_kernel`）。

这样整条路径上只有一次舍入（写回 BF16），**满足上游 A16"激活前保持 FP32"的判据**——这正是原先
"宽 T 交回上游阶梯"的理由（注释原话：单次 CUTLASS 只给出 BF16 的 gate_up 平面，过不了上游判据）。
master 走的是 BF16 平面 + 单独 `silu_mul`（少一次 FP32 平面，数值上与上游契约不同）；我们选了
FP32 平面，实测与 BF16 平面在噪声内，但不改判据、不动现有测试。

接线与记账：

* `fp8_linear_swiglu_dispatch`：sm_70 钩子内 `t ≤ 32` 仍走 QPN 拆分，`t ≥ 33` 走新的 f32 CUTLASS；
* `fp8_linear_swiglu_workspace_capacity_bytes`：修掉原来的 `min_tokens <= 32` 前置判断（单 T 查询
  min==max，T ≥ 33 时整段被跳过 → 返回 0 而实际分配了 FP32 平面 → 测试报"容量/执行不一致"），
  现在窄区算拆分暂存、宽区算 f32 路线、跨区取 max。

文件：`src/ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.{cu,h}`、
`src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp`。改动只影响 FP8 SwiGLU 的分派与容量，
v3/nvfp4 热路径逐字未动。

## 验收

**Oracle**（`ctest -R "linear_swiglu_(fp8|nvfp4|q4_a16|q8_a16)_test|linear_fp8_a16_test|linear_add_fp8_(wide_)?test"`）
**7/7 通过**；其中 `ninfer_linear_swiglu_fp8_test` 覆盖 A16 的 T = 1…1024（含 33/64/65/96/97/128/129/
1024 这些宽 T 用例）、`check_negative_gate` 的非线性精度判据（1/256），以及"单 T 查询容量必须与
执行高水位精确相等"的检查。

**端到端**（27K 请求、int8 KV、MTP draft 4、同参数）：

| 制品 | 修复前 | 修复后 | master |
|---|---|---|---|
| v2（升级成 v3） | 53.6s / 508 tok/s ✗ | **27.3s / 997.6 tok/s** ✓ | 27.1s / 1.00k |
| v2 decode / MTP | 84.7 / 47:60 | **82.7 / 69.2%** | 75.7 / 63.9% |
| SWIFT（生产 v3 nvfp4） | 1.00k / 78.3 | **1.00k / 78.3**（不变 ✓） | — |
| GSQ（IQ3_S） | 884.7 / 66.6 | **884.7 / 66.6**（不变 ✓） | — |

**结论**：混合 FP8 制品上的 2× 差距来自这一处路由，修掉后我们的线与 master 在 v2 制品上
prefill 持平、decode 与 MTP 接受率反超。生产（SWIFT nvfp4）与 GSQ 逐项不变。

## 复现

```bash
# 1) 离线把 v2 升成 v3（约 1m21s，权重字节原样保留）
python3 tools/upgrade_ninfer_v2_to_v3.py \
  /mnt/data/dockerdata/llama-cpp/models/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2.ninfer \
  /mnt/data/dockerdata/llama-cpp/models/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2-upgraded-v3.ninfer

# 2) 用构建产物直接起（生产必须先停，否则显存不足）
docker run --rm --gpus device=1 \
  -v /mnt/data/ninfer-build-gemma4:/build:ro \
  -v /mnt/data/dockerdata/llama-cpp/models:/workspace/models:ro \
  --entrypoint /build/apps/ninfer-serve ninfer:sm70-buildenv \
  /workspace/models/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-V2-upgraded-v3.ninfer \
  --host 0.0.0.0 --port 8000 --api-key sk-itzy --model-id qwen --max-context 200000 \
  --prefill-chunk 2048 --kv-capacity auto --kv-dtype int8 --max-concurrency 2 \
  --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 0 --preserve-thinking --vision
```
