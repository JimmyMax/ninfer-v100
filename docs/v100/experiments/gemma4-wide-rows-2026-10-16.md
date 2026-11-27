# Gemma 4 宽行内核：dense 31B（hidden 5376）(2026-10-16)

## 目标与结论

让**单张 V100-32GB** 能跑 dense 的 Gemma 4 31B（`Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP`）。
现在它**能加载、能正确推理、MTP 也生效** ✓；而且改动只新增代码，对既有制品（26B-A4B、Qwen 的
v2/v3、GGUF 混合制品）**零影响** ✓。

| 测试 | 结果 |
|---|---|
| 短问句（无 MTP） | TTFT 163 ms、decode **29.3 tok/s**，回答正确 ✓ |
| 短问句 + MTP（assistant，4 drafts） | TTFT 195 ms、decode **41.8 tok/s**（+43%）、接受率 24.1% ✓ |
| 26B-A4B（窄行路径，对照） | 输出与改动前逐字一致 ✓ |
| 26B-A4B 强制走宽行（差分测试） | 输出与窄行**逐字一致** ✓ |
| `ctest -R gemma4` | **8/8 通过**（含新增的宽行 oracle）✓ |

## 阻挡点：一行装不进一个 block

`src/models/gemma4/kernels.cu` 的三个行内核（`rms_norm`、`attn_residual`、`ffn_combine`）每线程负责
一个 `float4`，因此一行要装进一个 block：

```cpp
int row_threads(int hidden) {
    if (hidden % 4 != 0 || hidden / 4 > 1024) {        // ← hidden 上限 4096 的来源
        throw std::invalid_argument("gemma4 row kernels: hidden must be whole float4s, at most 4096");
    }
    return (hidden / 4 + 31) / 32 * 32;
}
```

26B-A4B 的 hidden 是 2816（704 个 float4 ✓），dense 31B 是 **5376（1344 个 float4 ✗ > 1024 线程）**。

## 改动

* **新增** `rms_norm_wide_kernel`、`attn_residual_wide_kernel`、`ffn_combine_wide_kernel`：固定
  `kWideRowThreads = 512`（8 的倍数 —— q8_1 的量化组是 8 个连续线程；`hidden % 32 == 0` 保证每个
  32 元素组要么整组在行内、要么整组在外 ✓），按 `blockDim.x` 步进分多轮处理整行；归约仍在循环外
  调用一次，所有车道都调用 `store_q8x4` / `block_sum`（前者用全 warp 的 shuffle，后者带跨 warp
  归约），越界车道传零值与 `valid=false`，与既有内核同构。
* **分派**：三个启动函数里 `wide_for(bit, hidden)` 为真才走宽行，否则调用**原来的内核**（原内核与
  窄行路径逐字未改 ✓）。
* **探针**：`NINFER_PROBE_WIDE=<mask>`（1=rms_norm、2=attn_residual、4=ffn_combine、8=全部）可强制
  某个内核走宽路径，用来分开"宽行内核错"与"别的 5376/dense 假设错"。默认 0 = 只按 `hidden` 判断。

## 踩到的真 bug（值得记下来）

宽行版本里，`h` 的平方和要和 `n` 分两趟算（第二趟才知道整行的和），于是中间值要靠**重算**得到。
第一版把 `a.h` 的写回放在了第二趟 ✗，第三趟重算 `h` 时读到的是**已经写回的 h** ✗ → 残差被加了两次
→ 输出复读乱码。定位手段是差分测试：把 26B 强制走宽行 ✓ 立刻复现乱码 ✓，再按位探针把范围缩到
`attn_residual` 与 `ffn_combine`（两者同病）✓。修法：把 `a.h` 的写回推到**最后一趟** ✓，且每个索引
只被读写一次 ✓。`rms_norm` 不受影响（它写的是另一个输出平面 ✓），探针也证实了这一点。

## 制品正确性（与引擎无关的部分先被排除）

用户提供的三件套（主 GGUF + `mtp-gemma-4-31B-it.gguf` + `mmproj-…-BF16.gguf`）用仓库自带转换器
`--recipe gemma4_gguf --components text,mtp,vision` 转成 v3 制品（18.74 GB），随后**逐元素/逐字节
对照 GGUF**：F32 norms ✓、Q4_K 的 `mlp/gate` 与 `mlp/up`（融合 `gate_up` 的两半 ✓）、Q6_K 的
`mlp/down` ✓、注意力 `q/k/v/o` ✓ —— **9/9 完全一致** ✓。所以乱码的责任在引擎侧，不在转换 ✓。

## 验收

* 新增 oracle `tests/models/gemma4/test_wide_rows.cpp`（`ninfer_gemma4_wide_rows_test`）：同一份随机
  输入同时跑 **hidden 2816（窄行）** 与 **5376（宽行）**，三个内核（含 `mlp_in`/`next_out` 的 BF16
  输出与 `h` 的 FP32 更新）都要与同一个 FP64 参考一致 ✓（bf16 相对误差 ≤ 2e-2，FP32 状态 ≤ 1e-4）。
* `ctest -R gemma4`：8/8 通过（两个需要本地 fixture 的 real 测试照旧 SKIP）。
* 端到端：31B 回答正确 ✓；26B 在探针强制宽行下输出与窄行逐字一致 ✓。
