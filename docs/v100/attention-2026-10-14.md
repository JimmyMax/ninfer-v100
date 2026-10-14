# 解码注意力线侦察（2026-10-14）

> 承接 [`fp8-qpn-2026-10-14.md`](fp8-qpn-2026-10-14.md)。fp8 阶段收尾后转"注意力线"。
> 本文件记录这一段的**测量结论**（其中一条推翻了"剩下的差距在注意力"的假设）、移植的
> 长上下文 split 修复，以及下一步该往哪打。

## 1. 先量：差距到底在不在注意力

同 prompt、同 flags（`--spec mtp --draft-tokens 4 --lm-head-draft --kv-dtype int8`）、
同卡（V100-32GB，GPU 1），本线 `ninfer:v100-fp8` vs fork `ninfer:v1.5.0`：

| 场景 | 本线 | fork | 倍数 |
|---|---:|---:|---:|
| 无投机 decode（t=1 轮次） | 26.9 tok/s（37.2 ms/轮） | 33.8（29.6 ms） | 1.26× |
| MTP4 decode（t=5 轮次） | 28.5（接受 30.0%） | 58.8（33.6%） | **2.06×** |
| 42K prefill | 951–963 tok/s | 622 | 本线更快 |

**只看 t=1，差距只有 1.26×；一到投机验证遍就变 2.06×**——投影已经追平，瓶颈在验证遍。

## 2. draft 数扫描：t=5 是独有异常

| drafts | 验证宽 t | 本线 decode | 本线接受 | 本线轮次 | fork decode | fork 轮次 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 26.9 | — | 37.2 ms | 33.8 | 29.6 ms |
| 1 | 2 | 39.8 | 59.2% | 40.2 ms | 51.3 | 31.4 ms |
| 2 | 3 | 44.9 | 47.9% | 43.6 ms | 57.1 | 33.4 ms |
| 3 | 4 | 43.4 | 33.7% | 52.9 ms | 56.3 | 32.9 ms |
| **4** | **5** | **28.5** | 30.0% | **77.0 ms** | 59.0 | 34.6 ms |
| 5 | 6 | 27.2 | 23.2% | 59.2 ms | 56.3 | 33.8 ms |
| 6 | 7 | 34.7 | 16.6% | 57.5 ms | 54.6 | 35.2 ms |

本线在 t=5 上比 t=4 和 t=6 都慢（77 ms vs 53/59），fork 全程平稳。已排除的因素：

* **不是注意力内核**：`NINFER_SM70_ATTN_V2=1`（S0.5 移植的 fork v2 注意力）与
  `NINFER_SMALLT_PV_F16=1` 对 t=5 轮次**零影响**（28.5 → 28.6）。
* **不是注意力路由**：q_heads=24 + int8 族在 width ≤ 8 时 `prompt_limit = 0`，一律走 SmallT，
  宽度 1..8 没有分界（`causal_softmax_attention.cpp` 的 `causal_attention_resolve_route`）。
* **不是 profile 路由**：`src/calibration/route_catalog.cpp` 明说 sm_70 构建"跑自己的解码注意力
  与单行 prompt 路由（`ops/linear/sm70`、`small_t_sm70.cuh`、`prompt_sm70.cu`），完全不查这些
  投影与注意力键"，只标定 GDN prefill——所以 V100 的 `attn_i8_small/*` 键在
  `device_profiles.json` 里是空的（3090/4090 各有 ~40 条），这是设计如此、不是缺配置。

**结论**：t=5 的异常在验证遍的其余部分（GDN 状态更新 / KV 追加 / MTP 选择器等按宽度切分的算子），
需要一次 kernel 级剖析（`ncu`）才能定位。**这一项是目前最大也最值钱的一块。**

## 3. 立刻可用的绕行：`--adaptive-mtp`

旗舰配置 `--adaptive-mtp`（每轮在 3..`--draft-tokens` 间按存活率取宽度）正好绕开定宽 t=5：

| | 定宽 d=4 | `--adaptive-mtp` d=4 | 提升 |
|---|---:|---:|---:|
| 短上下文 decode | 28.5 | **40.9** | +43% |
| 42K decode | 33.1 | **48.0** | +45% |

接受率同时从 30.0% 升到 34.0%（48.7% @42K）。已落到
`docker-compose.nvfp.yml`；对 fork 的差距从 2.07×/1.87× 收到 1.44×/1.29×。
配置改动，不涉及数值路径。

## 4. 移植：长上下文 split 上限（fork `c58a92ba`）

fork 把 Volta 的 `SmallTMaximumSplits` 从 85 提到 560："85 的封顶在约 41K 开始生效，把越来越多的
KV 工作串行进每个 CTA"。上游这条线仍是 85。移植两部分：

1. `geometry.cuh`：`#ifdef NINFER_SM70` 下 `kSmallTMaximumBaseSplits = 560`。
2. `small_t.cuh`：`causal_merge_split_statistics` 原来"一线程一 split"且断言 `splits <= 256`；
   上限一提，Volta 会在远早于 262K 上下文之前超过 blockDim.x，所以两遍都改成按 256 折叠。

实测（同 prompt、同 flags、**两镜像的 MTP 接受数完全一致**，即数值无变化）：

| 上下文 | 修复前 | 修复后 | 提升 |
|---|---:|---:|---:|
| 62,558 tok | 47.5 | 47.8 | +0.6% |
| 112,558 tok | 35.9 | 36.3 | +1.1% |
| 140,058 tok（280 splits，真正走折叠） | 27.9 | 28.3 | +1.4% |

收益只有 ~1%，因为解码注意力在这些上下文里只占轮次的几毫秒。**保留**是为与 fork 对齐、
并为极端上下文兜底；这条不是终点。注意上游自带 oracle 的最大 window 是 65,536（137 splits），
只覆盖"上限提高"，不覆盖"折叠"；140K 的端到端一致（接受数 14/50 相同）才是折叠路径的验证。

## 5. 下一步（按价值排序）

1. **定位 t=5 验证遍异常**：`ncu` 对验证遍做一次 kernel 级剖析，或按宽度二分（GDN 状态更新 /
   KV 追加 / MTP 选择器）。这一项 1.5–2.0× 的空间。
2. **t=1 的 1.26×**：投影已追平，剩下的是每轮固定开销与小的 kernel 启动/调度。
3. 路线图其余项（Gemma 4 的 S1–S5）本身仍以 Gemma 为对象；对 Swift/Qwen 这条线，上面两项才是
   继续收窄与 fork 差距的地方。

## 6. t=5 异常的排除清单（2026-10-14 续）

按"排除法"逐项实测，全部在 V100 上、同 prompt、定宽 `--draft-tokens`：

| 嫌疑 | 实测 | 结论 |
|---|---|---|
| 注意力算子本身 | `ninfer_causal_softmax_attention_bench --kv-dtype int8 --context 16384 --tokens 1..8`：121.9 / 120.8 / 144.4 / 154.6 / **146.4** / 239.6 / 283.7 / 290.8 µs | **W=5 不是尖峰**（比 W=4 还便宜）✗ |
| 注意力内核实现 | `NINFER_SM70_ATTN_V2=1`、`NINFER_SMALLT_PV_F16=1` 对 t=5 轮次零影响 | 排除 ✗ |
| CUDA Graph 重放 | `--no-cuda-graph`：d=4 28.4 vs 图模式 28.5；d=2 44.8 | 排除 ✗ |
| 投影算子 | `ninfer_fp8_linear_add_bench`（k=6144/17408）T=1..8 **75.8 µs 平坦**；`ninfer_nvfp4_linear_swiglu_bench` 286→308 µs（+7%，无悬崖） | 排除 ✗ |
| GDN 卷积 / KV 追加 | `ninfer_causal_conv1d_silu_bench` 6–16 µs、`ninfer_kv_cache_append_bench` 单节点 | 量级不对 ✗ |
| 注意力路由 | q_heads=24 + int8 在 width ≤ 8 一律 SmallT，无分界 | 排除 ✗ |
| MTP round 内核 | 与 fork 只差参数改名（上游更新），无按宽度分支 | 排除 ✗ |

**正面线索**：`src/models/qwen3_5/program/planning/graph_profiles.cpp` 的 `mtp_graph_profiles`
里对 `verify_window` 专门分支，其中

```cpp
// Target verify and MTP batch both have T=V+1 and W=E+V+1. Preserve one concrete INT8
// implementation per range at the T=4/5/6 launch boundaries.
} else if (verify_window == 5) {
    for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) { ... }
}
```

即上游**知道** T=5 会选到与 T=4/T=6 不同的 INT8 实现（所以要在 128/160/2054/8198 处切边界），
并且 graph 画像的签名里带着 `launch_shape(attention, storage, verify_window + 1U, ...)`。
量级上，t=5 轮次约 77.6 ms ≈ 69 GB 等效流量 ≈ **权重被读了两遍**（t=1 轮次 37.2 ms ≈ 33 GB ≈ 一遍），
所以更像是"某个算子在 5 列时被拆成两次执行/两遍读权重"，而不是某个算子变慢。

**下一步（留给下一次）**：把 `launch_shape` 在 5 列时的取值打出来，定位它对应的具体 INT8 实现，
再对照 4/6 列的实现看差在哪（`mtp_graph_profiles` 的边界正是为它而设）。
kernel 级归因需要 `ncu`，而本机驱动限制了性能计数器权限（`ERR_NVGPUCTRPERM`，
需 `NVreg_RestrictProfilingToAdminUsers=0` 并重载 nvidia 模块——生产在跑，未动）；
替代做法是把 `launch_shape`/profile 选择打成日志，或按宽度二分各算子。
