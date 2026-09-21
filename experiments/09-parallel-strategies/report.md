# Day 09 · 双卡 DDP / ZeRO / FSDP：在没有 NVLink 的机器上，FSDP 比 ZeRO-3 快 2.3 倍

两张 RTX 4090 之间没有 NVLink，驱动也禁掉了 PCIe P2P，NCCL 只能走共享主机内存。
实测 all-reduce 总线带宽 **15.3 GB/s** —— 是同一张卡内部显存带宽（920 GB/s）的 **1/60**。

在这个前提下，"双卡"几乎买不到吞吐：DDP 的 weak scaling 效率只有 **51.6%**，
也就是加了一张卡、吞吐只涨 3%。而原理上等价的 ZeRO-3 与 FSDP，实测差了 **2.28 倍**
——原因不在算法，在于 ZeRO-3 的 prefetch 在这台机器上没有把通信藏进计算里
（重叠系数 0.76，GPU 有 23.8% 的真实空泡），FSDP 藏住了（重叠系数 1.19）。

---

## 1. 硬件事实：先确认，不采信

任务书第 1 节要求自己确认 transport，不要采信"4090 禁用了 P2P"这句话。三条独立证据：

```
# 1) 驱动层面直接问
torch.cuda.can_device_access_peer(0, 1)  ->  False

# 2) NCCL 自己的判断（NCCL_DEBUG=INFO 原文，results/nccl_transport.txt）
NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 0
NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct
NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct
NCCL INFO NVLS multicast support is not available on dev 0
NCCL INFO Pattern 4, crossNic 0, nChannels 1, bw 20.000000/20.000000, type PHB/PIX

# 3) 拓扑
nvidia-smi topo -m  ->  GPU0 <-> GPU1 : NODE
   （NODE = 跨 PCIe Host Bridge，不是 PIX/NV#，两卡之间没有直连）
```

`via SHM/direct` 是关键：**每个字节都要先从 GPU 写进主机内存，再从主机内存读回另一张 GPU**。
NCCL 自己估的拓扑带宽是 20 GB/s，实测 15.3 GB/s，达到它估值的 76%。

### 1.1 集合通信带宽（分母，`results/comm_bench.json`）

algbw 与 busbw 的换算：`algbw = S / t`（S 是用户看到的消息大小），
`busbw = algbw × 2(P−1)/P`（all-reduce）或 `× (P−1)/P`（all-gather / reduce-scatter）。
**P = 2 时 all-reduce 的系数恰好是 1，algbw 与 busbw 数值相等**，这一档没有区分度；
到 P = 4 才拉开（系数 1.5）。all-gather / reduce-scatter 的系数是 0.5，与 all-reduce 不同，
**不能混用**——下表 all-gather 一列的 busbw 是 algbw 的一半。

| 消息大小 | all-reduce busbw | all-gather busbw | reduce-scatter busbw |
|---|---|---|---|
| 1 MB | 10.67 | 8.33 | 9.03 |
| 4 MB | 13.62 | 11.50 | 11.66 |
| 16 MB | 14.61 | 12.45 | 12.53 |
| 64 MB | 14.90 | 12.61 | 12.66 |
| 256 MB | 15.07 | 12.67 | 12.68 |
| 512 MB | 15.08 | 12.67 | 12.71 |
| **1024 MB** | **15.30** | **12.69** | **12.72** |

（单位 GB/s。16 MB 以上基本饱和；1 MB 那档明显偏低，是延迟主导。）

### 1.2 参照系

| 通道 | 实测 | 说明 |
|---|---|---|
| **卡间 all-reduce** | **15.3 GB/s** | 经主机内存中转 |
| H2D pinned | 25.1 GB/s | PCIe 4.0 ×16 单向 |
| D2H pinned | 26.3 GB/s | |
| H2D pageable | 19.1 GB/s | |
| D2H pageable | 15.2 GB/s | |
| **卡内 D2D** | **920.0 GB/s** | 同卡显存拷贝（1 读 1 写） |

**卡间比卡内慢 60 倍。** 后面所有"扩展效率为什么这么低"的归因都回到这一行。

> NCCL 2.30.7。第一次测得 all-reduce @1 GB 为 13.43 GB/s，修掉脚本 bug 后重测为 15.30 GB/s，
> 相差 12%。两次原始数据都保留（`results/logs/comm_bench_attempt1_collective_mismatch.log`
> 与 `results/comm_bench.json`），报告统一用重测值。

---

## 2. 测量协议

沿用 Day 08：同步后计时、按逐步耗时曲线决定丢几步、计时与 profiling 分两次跑、
三档显存全记并注明用哪档、失败如实落盘不重试。分布式侧另加：

- **每步前后都 `dist.barrier()` 再 `torch.cuda.synchronize()`**，否则量到的是本 rank 的
  局部进度而不是全局单步时间。
- **显存按 rank 分别记录**（`all_gather_object`）后取最大值，并报不对称度（任务书 4.3）。
- **只有 rank 0 写结果文件**（任务书 4.2）。
- **每个配置跑完清理孤儿进程并确认两张卡回到基线**（任务书 4.1）。
  本日全部 21 次运行结束后两张卡均为 1 MiB，见 `results/nvidia_smi_after.txt`。
- **DeepSpeed 的三个 batch 字段从 engine 反查生效值**，不相信自己写进 config 的值（任务书 4.5）。
  全部 6 个 DeepSpeed 配置反查结果与设定一致：`micro=2, accum=1, train_batch=4`。
- **FSDP 打印被 wrap 的模块数**（任务书 4.4）：实测 **29**，等于 28 个 `Qwen3DecoderLayer`
  加根模块，`transformer_auto_wrap_policy` 确实生效。

### 2.1 丢几步

`results/runs/curve_single.json` / `curve_ddp.json`，冷启动逐步耗时（ms）：

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 单卡 | **1332.0** | 140.4 | 137.3 | 135.7 | 135.0 | 136.8 | 137.2 | 135.3 | 136.0 | 136.3 |
| DDP 双卡 | **1298.3** | 273.4 | 264.5 | 262.0 | 264.7 | 262.1 | 260.7 | 260.9 | 263.1 | 262.2 |

与 Day 08 一致：**只有第 0 步异常**（9.7× / 4.7×）。
DDP 的第 1 步比稳态高 4.1%（NCCL 建链与首次 bucket 分配），第 2 步起进入稳态。
本日统一丢弃前 10 步，之后 3 个 block × 每 block 10 步取中位数。

### 2.2 显存用哪一档

三档全记在 `results/matrix_summary.tsv` / `comparison_tables.md`。
**策略对照表用 `max_memory_allocated`**，因为要比的是算法本身切分掉了多少张量；
`reserved` 与 `nvidia-smi` 也列出，因为它们决定"还能不能再塞一个配置"。

这个选择在本日会改变结论。以 ZeRO-1 为例：

```
allocated  14,286 -> 8,318 MiB   相对 DDP 降 41.8%
reserved   14,610 -> 11,652      只降 20.2%
nvidia-smi 15,236 -> 12,296      只降 19.3%
```

**用 nvidia-smi 衡量 ZeRO 的收益，会把 42% 看成 19%。**

---

## 3. 显存的理论账与实测对表

`N = 596,049,920`（Qwen3-0.6B 全部参数，Day 08 已逐字节核对）。`1N 字节 = 568.4 MiB`。
混合精度 + AdamW 每参数 16 字节：bf16 参数 2 + bf16 梯度 2 + fp32 主权重 4 + Adam m 4 + v 4。

| 策略 | 切分的张量 | 每卡公式 | 理论 | 实测 alloc | 实测 − 理论 |
|---|---|---|---|---|---|
| 单卡 / DDP | 不切 | 16N | 9,094 MiB | 14,286 | +5,192 |
| ZeRO-1 | 优化器状态 12N | 12N/P + 4N = 10N | 5,684 | 8,318 | +2,634 |
| ZeRO-2 | ＋梯度 2N | 12N/P + 2N/P + 2N = 9N | 5,116 | 8,318 | +3,202 |
| ZeRO-3 | ＋参数 2N | 16N/P = 8N | 4,547 | 9,627 | **+5,080** |
| FSDP FULL_SHARD | 同 ZeRO-3 | 16N/P = 8N | 4,547 | 7,509 | +2,962 |

**差额的来源，逐项解释：**

1. **激活与 logits（所有策略共有，约 2,900–3,200 MiB）。**
   本工作点每卡每步 1,024 token，lm_head 输出 `[1024, 151936]`：bf16 0.30 GB、
   交叉熵内部 upcast 到 fp32 0.59 GB，加上 log_softmax 保存与反传梯度各一份。
   Day 08 已证明这一族张量在这个模型上是显存主导项，**它不被任何 ZeRO stage 切分**。
   FSDP 的 +2,962 MiB 与 ZeRO-1 的 +2,634 MiB 基本就是这一块。

2. **DDP 多出的 ~2,200 MiB：精度组成不同。** 本日 DDP 走的是 Day 08 那套
   fp32 参数 + autocast，每参数同样是 16 字节（4+4+4+4），但 autocast 还要为每个权重
   缓存一份 bf16 副本，额外 2N = 1,137 MiB；DeepSpeed 的 bf16 模式则是
   参数本身就是 bf16。**两者总账都是 16N，但组成不同**，这一点在比较 DDP 与 ZeRO-1
   的显存时必须说明，否则会把"精度组成差异"误读成"分片收益"。

3. **ZeRO-3 的 +5,080 MiB 是本表最反常的一格。** 它的理论占用最低（8N），
   实测却比 ZeRO-1/2 还高 1,309 MiB。原因是 all-gather 的临时缓冲：
   ZeRO-3 每次用到一层参数都要先把它从两张卡上聚合回完整形态，
   `stage3_prefetch_bucket_size` 与 `stage3_max_live_parameters` 控制的这些缓冲区
   是常驻开销。**在 0.6B 这个规模上，缓冲区吃掉的比分片省下的还多。**

4. **ZeRO-1 与 ZeRO-2 实测完全相同（都是 8,318 MiB）。** 理论上 ZeRO-2 还要再切梯度
   （2N/P，应省 568 MiB）。没省下来的原因是 `contiguous_gradients: true` 会预留一块
   完整的连续梯度缓冲。这条是实测推翻配置直觉的一例，已在 journal 记下。

---

## 4. 并行策略对照表（任务书 3.6）

Qwen3-0.6B 全量微调，`seq_len = 512`，每卡 `micro_batch = 2`，梯度累积 = 1（所有配置一致）。
每个配置 warmup 10 步后 3 block × 10 步取中位数。原始数据 `results/runs/*.json`。

| 策略 | 切分了什么 | 单卡显存 alloc | 理论 | 总吞吐 | 扩展效率 weak | 扩展效率 strong | 每步通信量 | 通信占 CUDA 时间 |
|---|---|---|---|---|---|---|---|---|
| 单卡 (micro 2) | — | 12,013 MiB | 9,094 | 7,436 tok/s | 基准 | — | 0 | — |
| 单卡 (micro 4) | — | 16,050 MiB | 9,094 | 11,732 tok/s | — | 基准 | 0 | — |
| **DDP** | 不切；每步 all-reduce 梯度 | 14,286 MiB | 9,094 | 7,671 tok/s | **51.6%** | **32.7%** | 2.38 GB (fp32 梯度) | 51.3% |
| **ZeRO-1** | 优化器状态（fp32 主权重 + m + v） | 8,318 MiB | 5,684 | 7,745 tok/s | **52.1%** | **33.0%** | ≈2.4 GB | — |
| **ZeRO-2** | ＋梯度 | 8,318 MiB | 5,116 | 8,535 tok/s | **57.4%** | **36.4%** | ≈1.8 GB | 64.8% |
| **ZeRO-3** | ＋参数 | 9,627 MiB | 4,547 | 3,968 tok/s | **26.7%** | **16.9%** | ≈3.6 GB | 70.2% |
| **FSDP FULL_SHARD** | ＋参数（等价 ZeRO-3） | **7,509 MiB** | 4,547 | **9,056 tok/s** | **60.9%** | **38.6%** | ≈3.6 GB | 58.8–66.8% |

> **两个分母不同，不可互相比较。** weak 的分母是单卡 micro 2（7,436 tok/s ×2 = 14,872）；
> strong 的分母是单卡 micro 4（11,732 tok/s ×2 = 23,464）。
> strong 一列系统性更低，是因为它的单卡基准本身效率更高——
> 单卡从 micro 2 到 micro 4，吞吐涨了 57.8%（Day 08 已证明这个模型在小 batch 下
> 有大量与 batch 无关的固定开销），而双卡把每卡 batch 压回 2，把这份效率丢掉了。

### 4.1 用实测带宽做归因（交叉验证）

DDP 每步要 all-reduce 一次 fp32 梯度：`4 × 596,049,920 = 2.384 GB`。

```
按实测带宽预测   2.384 GB ÷ 15.30 GB/s = 155.8 ms
profiler 实测    NCCL kernel 169.4 ms/步
偏差             8.7%
```

**两个互相独立的测量吻合在 9% 以内**（一个来自消息大小 ÷ 微基准带宽，
一个来自 profiler 的 kernel 自时间），说明第 1 节的带宽分母是可用的。

同一条账还能解释单步时间：单卡 137.7 ms + 通信 169.4 ms = 307.1 ms（完全暴露），
实测 267.0 ms，说明藏进计算里的约 40 ms。

---

## 5. LoRA 下这张表会整体塌掉

可训练参数 4,587,520 / 600,637,440 = **0.764%**（r=16，target = q/k/v/o_proj），
冻结基座以 bf16 常驻。

| 策略 | 单卡显存 alloc | 相对 DDP | 总吞吐 | 扩展效率 weak |
|---|---|---|---|---|
| 单卡 | 4,685 MiB | — | 9,421 tok/s | 基准 |
| **DDP** | 4,702 MiB | — | 15,936 tok/s | **84.6%** |
| **ZeRO-1** | 4,435 MiB | **−5.7%** | 13,194 tok/s | 70.0% |
| **ZeRO-2** | 4,435 MiB | **−5.7%** | 13,459 tok/s | 71.4% |
| **ZeRO-3** | 5,377 MiB | **＋14.4%** | 3,420 tok/s | 18.2% |
| FSDP | 4,708 MiB | +0.1% | 8,180 tok/s | 43.4% |

三条结论：

**1. ZeRO-1 与 ZeRO-2 在 LoRA 下几乎不省显存，且两者完全相同。**
它们切的是优化器状态与梯度，而这两样在 LoRA 下只跟可训练参数走。
按每参数 16 字节、r = 0.764%、P = 2 手推：ZeRO-1 能省 `6rN = 0.046N ≈ 26 MiB`。
实测省了 267 MiB —— 比手推多，因为 DeepSpeed 还顺带省掉了一些常驻缓冲，
但量级结论成立：**这是个位数百分比，不是 ZeRO-1/2 在全量微调下那 42% 的量级。**

**2. ZeRO-3 在 LoRA 下反而更费显存（+14.4%），而且吞吐只有 DDP 的 21%。**
理论上它切冻结基座的 2N 参数，应该省接近一半。实测相反：
基座本来就只有 bf16 一份（1,137 MiB），切一半省 568 MiB，
而 all-gather 缓冲区的常驻开销超过了这个数。
吞吐上更糟——每步仍要 all-gather 整个基座，而可训练参数少到梯度通信几乎为零，
**通信量没降、计算量降了，比值就更难看**。

**3. LoRA 下 DDP 的扩展效率反而最高（84.6%，对全量微调的 51.6%）。**
因为 DDP 只 all-reduce 梯度，而 LoRA 的梯度只有 4.6M 参数 × 4 字节 = 18 MB，
比全量微调的 2.38 GB 小 130 倍，通信几乎免费。

**所以在这台机器上做 LoRA，正确答案是最朴素的 DDP。**

---

## 6. 通信与计算有没有重叠（任务书 3.5）

指标定义：**重叠系数 = 所有 CUDA kernel 自时间之和 ÷ 干净单步时间**。
NCCL 通信 kernel 跑在独立 stream 上、与计算 kernel 并发，所以各 stream 自时间之和
**可以超过**墙钟时间：

- 重叠系数 > 1 ⟹ 通信确实被藏进了计算，超出部分就是重叠量；
- 重叠系数 < 1 ⟹ 没藏满，差额是 GPU 真实空泡。

干净单步时间取自不挂 profiler 的那次运行（任务书 4.6）。

| 策略 | 通信占 CUDA 时间 | 通信 kernel 时间 | 重叠系数 | 结论 |
|---|---|---|---|---|
| DDP | 51.3% | 169.4 ms/步 | **1.24** | bucketing 生效，藏掉相当于单步 24% |
| ZeRO-2 | 64.8% | 214.8 ms/步 | **1.38** | overlap_comm 生效 |
| **ZeRO-3** | 70.2% | 275.9 ms/步 | **0.76** | **prefetch 没生效，GPU 空泡 23.8%** |
| FSDP | 58.8 / 61.7 / 66.8% | 146 / 166 / 213 ms/步 | 1.10 / 1.19 / **1.41** | 生效，但三次重复抖动大 |

**这一张表解释了本日最大的那个反差。** ZeRO-3 与 FSDP 切分的张量完全相同、
通信量也基本相同（都约 3.6 GB/步），吞吐却差 2.28 倍（3,968 对 9,056 tok/s）。
差别不在算法，在于**通信有没有被藏起来**：FSDP 的 all-gather 与前向计算重叠，
ZeRO-3 的没有——它的 GPU 有近四分之一时间在空等。

ZeRO-3 的 405 次通信 kernel 启动（对 FSDP 的 258 次）也指向同一件事：
粒度更碎、每次更小，在 15 GB/s 的链路上更难摊薄延迟。

> FSDP 的重叠系数四次重复分别为 1.79 / 1.10 / 1.19 / 1.41，中位数 1.19，极差很大，
> 而它的干净单步时间却很稳（226.2 ms，std 0.3 ms）。说明抖动来自 profiler 观测本身
> 而非训练过程。本表取中位数，**不用单次值下结论**。
> 相比之下 ZeRO-3 的重叠系数两次分别是 0.74 与 0.76，稳定，结论可靠。

---

## 7. 在这台机器上该选哪个；换成有 NVLink 的机器会怎么变

### 7.1 这台机器（2 × 4090，无 NVLink，SHM，15.3 GB/s）

**全量微调选 FSDP FULL_SHARD。** 它在本日的对照里同时拿下最高吞吐（9,056 tok/s）
和最低显存（7,509 MiB），没有任何取舍——比 DDP 快 18%、省 47%，比 ZeRO-3 快 2.28 倍、
省 22%。

**LoRA 选 DDP。** 84.6% 的扩展效率，显存与单卡持平。
在 LoRA 场景下引入 ZeRO 只会付出通信代价而换不到显存。

**任何场景都不要在这台机器上用 ZeRO-3。** 它是唯一一个"显存没省到、吞吐还掉一半"的配置。

但更根本的结论是：**双卡在这台机器上对全量微调几乎不值得。**
最好的 FSDP 也只有 60.9% 的 weak scaling 效率——付出两倍的卡，换来 1.22 倍的吞吐。
如果任务能在单卡显存里放下，单卡是更划算的选择；
双卡的真正价值在于**放得下原本放不下的东西**（FSDP 把每卡显存从 12.0 GB 压到 7.5 GB）。

### 7.2 换成有 NVLink 的机器（基于第 1 节实测值外推）

NVLink 4.0 的双向带宽约 900 GB/s，是本机 15.3 GB/s 的 **59 倍**。把通信时间按这个比例缩放：

| 策略 | 本机通信 kernel | 若 ×59 提速 | 本机单步 | 外推单步 | 外推 weak 效率 |
|---|---|---|---|---|---|
| DDP | 169.4 ms | 2.9 ms | 267.0 ms | ≈140 ms | ≈98% |
| ZeRO-2 | 214.8 ms | 3.6 ms | 240.0 ms | ≈145 ms | ≈95% |
| ZeRO-3 | 275.9 ms | 4.7 ms | 516.2 ms | ≈150 ms | ≈92% |
| FSDP | 165.9 ms | 2.8 ms | 226.2 ms | ≈142 ms | ≈97% |

**外推的结论是：在 NVLink 机器上，这四种策略的吞吐会收敛到彼此相差几个百分点之内**，
选择依据从"谁的通信藏得住"变成"谁的显存省得多"——那时 ZeRO-3 / FSDP 的 16N/P
才真正值得，而本日 ZeRO-3 相对 FSDP 那 2.28 倍的劣势会缩小到几乎看不见，
因为它的劣势本来就是"没藏住通信"，而不是"多通信了"。

同时可以预期，**ZeRO-3 在 NVLink 上的相对排名会大幅上升**，
这也解释了为什么文献里 ZeRO-3 的吞吐损失常报个位数百分比——
那些数字是在 NVLink / InfiniBand 上测的，直接搬到消费卡机器上会错得离谱。

> 这一节是**外推，不是实测**。本机没有第二种互联可比，NVLink 的 900 GB/s 是规格值
> 而非本日实测。按线性缩放外推忽略了延迟项与 kernel 启动开销，
> 在通信时间降到几毫秒后这些项会变成主导，所以上表的"外推单步"是乐观下界。

---

## 8. 未完成

- **载体 B（Qwen2.5-VL-3B LoRA）未完成。** modelscope 下载在第二个分片降速到
  约 0.6 MB/s，剩余 3.2 GB 需约 90 分钟，按任务书第 7 节"从后往前砍"放弃。
  说明见 `results/carrier_B_未完成说明.txt`。
  需要澄清的是：3.1 第 4 问那个题眼（LoRA 下 ZeRO-1/2 的显存曲线）**不依赖 VL 模型**，
  已用载体 A 完整测过（第 5 节）；载体 B 缺的是"多模态端到端跑通"这一项凭据。
- **载体 B 的完整配置矩阵**（优先级 7）同上，未做。
- **ZeRO-3 的 prefetch 为什么没生效，只定位到现象没定位到原因。**
  本日只证明了"重叠系数 0.76、空泡 23.8%"，没有去扫
  `stage3_prefetch_bucket_size` / `stage3_max_live_parameters` 看能否调好。

## 9. 局限

- **只有 2 张卡，P = 2。** all-reduce 的 busbw 与 algbw 在 P=2 时恒等，
  这个指标的区分度要 P≥4 才体现；本日的"通信量/step"也都只在 P=2 下成立。
- **只测了一个模型、一个工作点**（0.6B，seq 512，micro 2）。
  第 3 节已经说明 ZeRO-3 的缓冲区开销与模型规模强相关——
  在 7B 以上，16N/P 的收益会压过缓冲区，结论可能反转。
- **DDP 与 DeepSpeed 的精度组成不同**（fp32 参数 + autocast 对 bf16 参数 + fp32 主权重），
  总账都是 16N，但比较显存时这是一个未被消除的变量。
- **FSDP + LoRA 那一格的精度口径与其它格不同**：FSDP1 的 flat param 要求 dtype 统一，
  而 LoRA 的 adapter 默认是 fp32，只能把 adapter 转成 bf16 才能 wrap，
  于是它的主权重是 bf16 而非 fp32。这一格的显存数字不能与同行严格对比。
- **FSDP 的重叠系数重复性差**（1.10–1.79），本文取中位数，n = 4 不足以给置信区间。
- 通信量一列中，ZeRO-1/2/3 的字节数是按算法推算而非逐 kernel 实测，
  只有 DDP 那一行做了带宽交叉验证（4.1）。

## 10. 原始数据

```
results/comm_bench.json            集合通信 + PCIe 带宽扫描（第 1 节，分母）
results/nccl_transport.txt         NCCL_DEBUG=INFO 的 transport 原文摘录
results/nccl_probe.json            can_device_access_peer / NCCL 版本
results/logs/nccl_debug.log        完整 NCCL INFO 日志
results/runs/*.json                21 次运行的原始记录（每 rank 显存、逐步耗时、生效 batch 配置）
results/matrix_summary.tsv         汇总表（三档显存 / 吞吐 / 扩展效率）
results/comparison_tables.md       自动渲染的对照表（防转抄错误）
results/profiles/prof_*.json       4 种策略的算子分解与重叠系数（FSDP 重复 3 次）
results/carrier_B_未完成说明.txt    载体 B 未完成的原因
results/nvidia_smi_{before,after}.txt
results/run_all.log                全流程日志
scripts/comm_bench.py              通信微基准
scripts/nccl_probe.py              transport 探测
scripts/train_dist.py              六种策略同一套训练循环（含 LoRA 开关）
scripts/ds_zero{1,2,3}.json        三个 ZeRO config
scripts/profile_dist.py            通信/计算重叠分析
scripts/analyze09.py               对照表生成
scripts/run_day09.sh               全流程启动
scripts/smoke09.sh                 单卡调通用
```
