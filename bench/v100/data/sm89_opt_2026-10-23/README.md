# RTX 4080 SUPER (sm_89) 优化探测数据

2026-10-23,模型 `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`(13.4 GiB,
int8 KV,K=5 dflash2),引擎 `ninfer:1.7.0-sm89`。

## 文件

- `runs.txt` — 配置扫描(5 组:baseline/ch4096/ch8192/K7/bf16KV)。
  生产配置(ch2048 + int8 + K5)已最优;chunk 大小对 4080S 无影响,K7 更差,bf16 KV 更差。
- `gemv_bench_sm89.txt` — 独立 vec GEMV 微基准(复用 `ggml_bridge.cu.o` + 19 个
  `ggml_vec_*.cu.o` 链接的 standalone bench,5 类型 × 4 形状 × t∈{1,5})。
  t=1 数值受 96 MB L2 加成(权重 ≤50 MB 常驻 L2),t=5 更接近真实;
  全部 ≥670 GB/s,接近/达到 DRAM 峰值(~950 GB/s)。
- `node_trace_analysis.txt` — nsys `--cuda-graph-trace=node` 的 decode 相位
  按核族聚合(92 轮,256 tok 请求)。

## 关键数字(decode 相位,每轮 35 ms wall)

| 项 | 时间/轮 | 说明 |
|---|---|---|
| vec::kernel(GEMV) | 21.2 ms(76.6% kernel) | 带宽受限,~630-700 GB/s |
| q8_ksplit_mma(draft Q8 平面) | 3.5 ms(10.2%) | tensor core 路径,450 MB/轮 |
| attention + GDN + norm 等 | ~4 ms | |
| 轮间大停顿(95 个 × ~3.3 ms) | 3.3 ms | host 串行依赖(采样→MTP 记账→下轮 graph) |
| graph 内节点间隙 | <0.1 ms 合计 | 可忽略 |

GPU 利用率 79-89%(含 graph 内 kernel);空闲的 96% 是轮间大停顿,
不是核内低效。
