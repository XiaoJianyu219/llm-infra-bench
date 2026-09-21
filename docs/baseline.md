# Baseline — 所有对比的参照点

本文件定义整个项目的实验基准。除非另有说明，所有后续测量均在此配置下进行，
且每次仅改动一个变量。

## 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4090 24G（可用 23.52 GiB） |
| 显存带宽 | 1008 GB/s（规格值） |
| 平台 | AutoDL 单卡实例 |
| CPU / 内存 | 16 核 / 120 GB |

## 软件栈

```
python        3.12.3
torch         2.13.0+cu130
vllm          0.28.0
transformers  5.16.1
镜像          PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 13.0
```

系统未安装 nvcc；torch 与 vllm 使用 pip wheel 自带的 CUDA 运行时。
完整快照见 `results/env/env-day01.txt`。

## 模型

```
Qwen/Qwen3-VL-8B-Instruct，bfloat16

num_hidden_layers    36
num_attention_heads  32
num_key_value_heads  8      （GQA）
head_dim             128
```

## 服务配置

```bash
vllm serve /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.94 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --max-num-seqs 16
```

`--max-model-len 8192` 在项目期内锁定。改动它会改变 KV cache 的每请求占用，
使所有历史数据失去可比性。

## 显存分解

| 项 | 大小 |
|---|---|
| 权重（含 non-torch） | 16.79 GiB |
| 峰值激活 | 2.26 GiB |
| CUDA Graph | 0.12 GiB |
| KV cache | 3.05 GiB |
| 合计 | 22.22 GiB |
| 启动时卡上可用 | 23.13 GiB |

引擎初始化 148.07 s，其中 torch.compile 39.94 s。
后续重启命中编译缓存会更快（缓存位于 `$VLLM_CACHE_ROOT`）。

## KV cache

```
每 token 开销 = 2(K和V) × 36层 × 8个KV头 × 128 × 2字节
             = 147,456 B = 144 KiB

池子容量 = 3.05 GiB ÷ 144 KiB = 22,208 tokens
```

理论核算与实测一致。该常量由模型结构决定，与任何服务参数无关
（已在 5 种池子配置下验证，见 `reports/day02_params.md`）。

## 负载定义

```
合成多模态请求，--dataset-name random-mm
输入 512 token / 输出 128 token
--seed 42
warmup 20 条（并发 4），结果丢弃
每档 3 次重复取中位数，不取最优值
显存峰值由 nvidia-smi 每 0.5 s 采样，取压测进行中的最大值
```

## 性能参照值

采用 **Day 2 的 `base` 行**作为参照数值，因其与 Day 2 其余六个配置同协议测得。

| conc | req/s | out_tok/s | TTFT_p50 | TTFT_p99 | TPOT_p50 |
|---|---|---|---|---|---|
| 16 | 3.34 | 427.1 | 769 ms | 1,928 ms | 29.9 ms |
| 64 | 3.32 | 425.5 | 14,560 ms | 18,949 ms | 33.0 ms |

Day 1 首次测量的完整四档数据见 `reports/day01_baseline.md`。

> **已知差异**：Day 1 与 Day 2 base 在 conc64 分别为 3.52 与 3.32 req/s，
> 相差 5.7%，超出该并发档 0.1% 的噪声底。已知来源是两次使用的
> `--num-prompts` 不同（300 vs 150），请求总数会影响爬坡与排空阶段的占比。
> 因此跨天对比必须固定 num-prompts。本文件采用 Day 2 数值以保持协议一致。

## 噪声底

```
conc 16：4.4 – 5.8%
conc 64：0.1 – 0.7%
```

判据：conc16 的差异须超过 6%、conc64 须超过 1%，方视为有效提升。

噪声底随负载区间变化：系统有余量时 run-to-run 抖动大，
被队列卡死在确定性上限时反而稳定。报告结论时须注明所处区间。

## 关于更优配置

Day 2 找到吞吐更高的配置（`--kv-cache-memory=4087021056` 配合
`--max-num-seqs 64`，conc64 达 4.00 req/s）。**该配置不替换本 baseline。**

保持参照点不变，是为了让「参数调优收益」与后续「量化 / 引擎优化收益」
保持为两笔独立的账，各自可归因。

## 复现

```bash
/root/autodl-tmp/scripts/serve.sh      # 起服务（参数已焊死在脚本内）
/root/autodl-tmp/scripts/check.sh      # 校验参数生效、显存分解与本文件一致
/root/autodl-tmp/scripts/sweep.sh      # 并发扫描
python /root/autodl-tmp/scripts/summarize.py
```

`check.sh` 输出的 KV cache 若不是 3.05 GiB / 22,208 tokens，
说明机器或配置已变，本文件的所有参照值失效，须重建 baseline。