## 通信带宽（第 1 节，所有归因的分母）

| 操作 | 实测峰值 busbw | 取得于 |
|---|---|---|
| all_reduce | **15.30 GB/s** | 1024 MB |
| all_gather | **12.69 GB/s** | 1024 MB |
| reduce_scatter | **12.72 GB/s** | 1024 MB |

| 消息大小 | all-reduce algbw | all-reduce busbw | all-gather busbw | reduce-scatter busbw |
|---|---|---|---|---|
| 1 MB | 10.67 | **10.67** | 8.33 | 9.03 |
| 4 MB | 13.62 | **13.62** | 11.50 | 11.66 |
| 16 MB | 14.61 | **14.61** | 12.45 | 12.53 |
| 64 MB | 14.90 | **14.90** | 12.61 | 12.66 |
| 256 MB | 15.07 | **15.07** | 12.67 | 12.68 |
| 512 MB | 15.08 | **15.08** | 12.67 | 12.71 |
| 1024 MB | 15.30 | **15.30** | 12.69 | 12.72 |

| 参照通道 | GB/s |
|---|---|
| H2D_pinned | 25.1 |
| D2H_pinned | 26.3 |
| H2D_pageable | 19.1 |
| D2H_pageable | 15.2 |
| D2D | 920.0 |

## 并行策略对照表（任务书 3.6）

模型 Qwen3-0.6B，N = 596,049,920 参数；seq_len = 512，每卡 micro batch = 2，梯度累积 = 1（所有配置一致）。

| 策略 | 切分了什么 | 单卡显存 alloc | 理论显存 | 实测/理论 | 总吞吐 | 扩展效率(weak) | 扩展效率(strong) | 每步通信量 |
|---|---|---|---|---|---|---|---|---|
| single (w1) | 不切分（单卡基准） | 12013 MiB | 9095 MiB | 1.32× | 7409 tok/s | — (基准) | — (基准) | 见 report |
| ddp (w2) | 不切分；每步 all-reduce 梯度 | 14286 MiB | 9095 MiB | 1.57× | 7750 tok/s | **52.1%** | **33.0%** | 2.38 GB |
| zero1 (w2) | 优化器状态（fp32 主权重 + Adam m + v = 12N） | 4435 MiB | 5684 MiB | 0.78× | 13194 tok/s | **88.7%** | **56.2%** | 见 report |
| zero2 (w2) | 优化器状态 + 梯度（2N） | 4435 MiB | 5116 MiB | 0.87× | 13459 tok/s | **90.5%** | **57.4%** | 见 report |
| zero3 (w2) | 优化器状态 + 梯度 + 参数（2N） | 5377 MiB | 4548 MiB | 1.18× | 3420 tok/s | **23.0%** | **14.6%** | 见 report |
| fsdp (w2) | 优化器状态 + 梯度 + 参数（FULL_SHARD，等价 ZeRO-3） | 4708 MiB | 4548 MiB | 1.04× | 8180 tok/s | **55.0%** | **34.9%** | 见 report |

> 扩展效率 weak 的分母是单卡 micro=2（7436 tok/s）；strong 的分母是单卡 micro=4（11732 tok/s）。两者分母不同，不可互相比较。

## 逐配置明细（`results/runs/<tag>.json`）

| tag | 策略 | 卡数 | micro | step ms ± std | 总 tok/s | alloc max/min MiB | 不对称 | reserved | nvidia-smi | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| curve_ddp | ddp | 2 | 2 | 264.3 ± 2.6 | 7750 | 14286 / 14286 | 0.00% | 14610 | 15236 | ok |
| curve_single | single | 1 | 2 | 138.2 ± 0.2 | 7409 | 12013 / 12013 | 0.00% | 12662 | 13144 | ok |
| lora_single_mb2 | single | 1 | 2 | 108.7 ± 0.2 | 9421 | 4685 / 4685 | 0.00% | 5028 | 5510 | ok |
| lora_w2_ddp_mb2 | ddp | 2 | 2 | 128.5 ± 1.9 | 15936 | 4702 / 4702 | 0.00% | 5034 | 5660 | ok |
| lora_w2_fsdp_mb2 | fsdp | 2 | 2 | 250.4 ± 1.9 | 8180 | 4708 / 4690 | 0.37% | 6194 | 6820 | ok |
| lora_w2_zero1_mb2 | zero1 | 2 | 2 | 155.2 ± 2.3 | 13194 | 4435 / 4435 | 0.00% | 5342 | 5986 | ok |
| lora_w2_zero2_mb2 | zero2 | 2 | 2 | 152.2 ± 0.4 | 13459 | 4435 / 4435 | 0.00% | 5342 | 5986 | ok |
| lora_w2_zero3_mb2 | zero3 | 2 | 2 | 598.8 ± 10.1 | 3420 | 5377 / 5377 | 0.00% | 6396 | 7040 | ok |
| single_mb2 | single | 1 | 2 | 137.7 ± 0.4 | 7436 | 12013 / 12013 | 0.00% | 12662 | 13144 | ok |
| single_mb4 | single | 1 | 4 | 174.6 ± 0.6 | 11732 | 16050 / 16050 | 0.00% | 16682 | 17164 | ok |
| w2_ddp_mb2 | ddp | 2 | 2 | 267.0 ± 2.6 | 7671 | 14286 / 14286 | 0.00% | 14610 | 15236 | ok |
| w2_fsdp_mb2 | fsdp | 2 | 2 | 226.2 ± 0.3 | 9056 | 7509 / 7509 | 0.00% | 10106 | 10732 | ok |
| w2_zero1_mb2 | zero1 | 2 | 2 | 264.4 ± 6.0 | 7745 | 8318 / 8318 | 0.00% | 11652 | 12296 | ok |
| w2_zero2_mb2 | zero2 | 2 | 2 | 240.0 ± 0.5 | 8535 | 8318 / 8318 | 0.00% | 11652 | 12296 | ok |
| w2_zero3_mb2 | zero3 | 2 | 2 | 516.2 ± 11.3 | 3968 | 9627 / 9627 | 0.00% | 12342 | 12986 | ok |

## 自动核对（任务书 4.3 / 4.4 / 4.5）

- **lora_w2_fsdp_mb2**：FSDP 被 wrap 的模块数 29（期望 ≥ 29）→ OK
- **lora_w2_zero1_mb2**：DS 生效 batch 配置 {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}；engine 反查 {'micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}
- **lora_w2_zero2_mb2**：DS 生效 batch 配置 {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}；engine 反查 {'micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}
- **lora_w2_zero3_mb2**：DS 生效 batch 配置 {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}；engine 反查 {'micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}
- **w2_fsdp_mb2**：FSDP 被 wrap 的模块数 29（期望 ≥ 29）→ OK
- **w2_zero1_mb2**：DS 生效 batch 配置 {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}；engine 反查 {'micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}
- **w2_zero2_mb2**：DS 生效 batch 配置 {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}；engine 反查 {'micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}
- **w2_zero3_mb2**：DS 生效 batch 配置 {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}；engine 反查 {'micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1, 'train_batch_size': 4}
