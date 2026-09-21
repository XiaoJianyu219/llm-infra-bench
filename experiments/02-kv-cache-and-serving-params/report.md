# Day 2 · 服务参数对 KV cache 与吞吐的影响

所有测量基于 `BASELINE.md` 所述配置，每次仅改动一个变量。
负载：512 in / 128 out 合成多模态请求，每档 150 条、3 次重复取中位数。

**读表注意**：`base` 使用 `max-num-seqs=16`（Day 1 配置），其余配置全部使用 64。
因此除 seqs 系列外，各配置的正确对照组是 `seqs64` 而非 `base`。

## 结果

| 配置 | KV cache | KV tokens | conc16 req/s | conc64 req/s | conc64 TTFT_p50 | conc64 TPOT_p50 |
|---|---|---|---|---|---|---|
| base（util .94, seqs16） | 3.05 GiB | 22,208 | 3.34 | 3.32 | 14,560 ms | 33.0 ms |
| seqs64 | 3.05 GiB | 22,208 | 3.33 | 3.60 | 12,259 ms | 37.2 ms |
| seqs256 | 3.05 GiB | 22,208 | 3.33 | 3.59 | 12,406 ms | 36.3 ms |
| util90 | 2.07 GiB | 15,088 | 3.12 | 3.10 | 16,321 ms | 29.7 ms |
| util96 | 3.48 GiB | 25,376 | 3.33 | 3.76 | 11,819 ms | 42.2 ms |
| len4096 | 3.01 GiB | 21,952 | 3.33 | 3.58 | 12,294 ms | 37.5 ms |
| kvmem381 | 3.81 GiB | 27,712 | 3.34 | 4.00 | 9,391 ms | 43.7 ms |

噪声底：conc16 为 4.4–5.8%，conc64 为 0.1–0.7%。
判据：conc16 的差异须超过 6%、conc64 须超过 1%，方视为有效。

## 结论

**1. 每 token 的 KV 开销是模型结构常量，与服务参数无关。**

五种池子配置下均为 144 KiB/token，与理论核算一致
（2 × 36 层 × 8 KV 头 × 128 × 2 字节 = 147,456 B）。
调整 `gpu-memory-utilization` 只改变池子总量，不改变单 token 开销。

**2. 限制并发吞吐的是 KV cache 容量，不是调度器上限。**

```
KV tokens     15,088   22,208   25,376   27,712
conc64 req/s    3.10     3.60     3.76     4.00
                util90   seqs64   util96   kvmem381
```

单调递增，无反常点。池子 +84% 对应吞吐 +29%。
而 `max-num-seqs` 16 → 64 仅带来 +8.4%，16 → 256 无进一步增益，
说明调度器上限在 64 以上已不构成约束。

**3. 收益递减源于 roofline 位置的移动。**

| 配置 | KV tokens | conc64 req/s | conc64 TPOT_p50 |
|---|---|---|---|
| util90 | 15,088 | 3.10 | 29.7 ms |
| seqs64 | 22,208 | 3.60 | 37.2 ms |
| util96 | 25,376 | 3.76 | 42.2 ms |
| kvmem381 | 27,712 | 4.00 | 43.7 ms |

池子放大使有效 batch 增大，吞吐上升，但 TPOT 同步从 29.7 升至 43.7 ms。

单请求 decode 受显存带宽支配（Day 1 实测 TPOT 17.3 ms ≈ 权重 17.9 GB
÷ 1008 GB/s = 17.8 ms），batch 增大后逐步转入算力支配区间，
故池子 +84% 只换来吞吐 +29%。

该结论与本项目 `kvcache_demo` 的独立测量一致：
在 prompt 312 token 下，有无 KV cache 的前向 token 数相差 58.6×，
耗时仅相差 2.10×——GPU 处理 1 个 token 与处理 300 个 token 的耗时接近，
因为权重读取是固定成本。这既解释了批处理为何有效，
也解释了扩大 KV 池子为何收益递减。

**4. 缩小 max-model-len 不提升真实吞吐。**

`len4096` 相对同条件的 `seqs64` 为 3.58 vs 3.60 req/s（在噪声内），
KV 池子亦几乎未变（21,952 vs 22,208 tokens）。

但启动日志的 `Maximum concurrency` 从 2.71x 跳至 5.36x。
该指标以 max-model-len 为分母、按每请求占满上下文的最坏情况计算，
分母减半使其翻倍，属账面变化而非容量增益。

**读取任何"最大并发"类指标前，必须先确认它的分母是什么。**

**5. 池子过小同时恶化尾延迟。**

`util90` 在 conc16 的 TTFT_p99 为 3,896 ms，
约为其他配置（约 1,900 ms）的两倍，而其 TTFT_p50 仅高出 10%。
中位数几乎没变而尾部翻倍，指向偶发的排队或抢占，
而非全局性的速度下降。

## 调优成果

仅通过服务参数调优（零代码改动），conc 64 下相对 baseline：

| 指标 | base | kvmem381 | 变化 |
|---|---|---|---|
| 吞吐 | 3.32 req/s | 4.00 req/s | +20.5% |
| TTFT_p50 | 14,560 ms | 9,391 ms | −35.5% |
| KV 容量 | 22,208 tokens | 27,712 tokens | +24.8% |

主要来自显式指定 `--kv-cache-memory=4087021056`，
榨干 `gpu-memory-utilization` 未覆盖的剩余显存
（该数值由 vLLM 启动日志直接给出，见 `gpu_worker.py:804` 那行提示）；
其次来自放开 `--max-num-seqs` 至 64。

该配置**不替换 baseline**，理由见 `BASELINE.md`。

## 待补

- 未采集压测进行中的 `num_requests_waiting` 与 `gpu_cache_usage_perc`。
  结论 2 目前基于吞吐曲线的间接推断，尚缺直接观测。
  补法：对 util90 与 kvmem381 各压一次 conc64，同时轮询 `/metrics`，
  验证前者的 `gpu_cache_usage_perc` 是否顶到 1.0 而后者未顶到。
- prefix caching 需要共享前缀的负载才能体现，本轮的 random-mm 无法验证。

## 原始数据

```
results/day02/bench_{config}_c{16,64}_r{1,2,3}.json   压测指标
results/day02/serve_{config}.log                      各配置启动日志（KV 数字来源）
results/day02/summary.tsv                             汇总表
results/day02/kvcache_demo.txt                        手写循环对照结果
results/day02/sweep.log                               扫描全过程输出
```