# Day 08 · 单卡训练 profiling：MFU 只有 28%，而 GEMM 本身跑到了峰值的 90%

一块 RTX 4090 训练 Qwen3-0.6B，MFU 28.4%。这个数字容易被读成"算力没吃满"，
但算子级数据说的是另一回事：真正在做矩阵乘的那些 kernel 已经跑到实测峰值的
**90.3%**，它们只占了 37.2% 的 GPU 时间。剩下 62.8% 花在优化器和逐元素算子上——
这些活的算术强度远在 ridge point 之下，怎么加算力都不会变快。

本文还给出一个 24 GB 上的硬约束：**能不能跑起来只取决于每步 token 总数，
与 batch 和 seq_len 怎么拆无关**，因为决定性的那个张量是 `[tokens, 151936]` 的 logits。

---

## 1. 四个问题的直接回答

**Q1：compute-bound 还是 memory-bound？—— memory / overhead-bound。三条独立证据。**

| 证据 | 来源 | 数值 | 指向 |
|---|---|---|---|
| E1 算子时间构成 | profiler kernel 侧 | GEMM 家族（GEMM + flash attention）仅占 CUDA 时间 **37.2%**；optimizer 28.8% + elementwise 26.2% + softmax 5.8% + reduction 1.9% = 62.8% | 时间不在算力上 |
| E2 GEMM 内部效率 | profiler `with_flops` | 这些 GEMM 类算子自身达 **140.5 TFLOPS = 实测峰值 155.6 的 90.3%** | 算力侧已接近上限，没有优化空间 |
| E3 batch 缩放 | 不用 profiler，来自 batch 扫描 | `step_ms = 114.3 + 19.7 × batch`（s=512）；b=4 时固定开销占单步 **64.4%**；b1→b2 单步耗时**不升反降**（148.2 → 143.8 ms） | 大半个 step 与计算量无关 |

E1 与 E2 都读 profiler，但问的不是一件事：E1 问"时间花在哪"，E2 问"花在 GEMM 上
的那部分值不值"。E3 完全不依赖 profiler。三条都指向同一结论，且**互不能由对方推出**。

补充排除一种解释：**不是 CPU 下发跟不上。** profiler 记到的 CUDA self time 为
181.9 ms/步，不挂 profiler 的干净单步为 187.8 ms，空泡 ≤ **3.1%**（profiler 只会
放大 wall time，所以这是上界）。每步 5,506 次 kernel launch，GPU 始终是满的——
满的时候在干低算术强度的活。

**Q2：MFU 是多少？—— 28.4%（任务书口径），计入 lm_head 后 34.9%。**

参考工作点 s=2048、b=1、bf16 autocast、flash SDPA、AdamW、无 checkpointing：

```
分母  实测 bf16 GEMM 峰值 155.57 TFLOPS   （规格 165.2，占 94.2%）
分子  每 token 4.052 GFLOPs = 6N(2.643) + 12·L·d_attn·s(1.409)
      N = 440,467,456（非 embedding，手推与实测逐字节一致）
吞吐  10,903 tokens/s  （单步 187.8 ms，4 次独立进程极差 0.42%）
MFU   10903 × 4.052e9 / 155.57e12 = 28.40%
HFU   = MFU（未开 checkpointing 时硬件 FLOPs = 模型 FLOPs）
```

分子分母的手推过程见第 3 节。**计入 lm_head 后 MFU 升到 34.94%** ——
对这种"小隐藏维 + 15 万词表"的模型，标准的"6N 非 embedding"口径系统性低估
真实计算量约 23%，见 3.3。

**Q3：三个开关各自换来多少显存、损失多少速度？**

| 开关 | 工作点 | 吞吐 | 显存 `max_memory_allocated` | 净账 |
|---|---|---|---|---|
| bf16 AMP（对 fp32） | s2048 b1，同用 mem_efficient | 4,875 → 9,909 tok/s，**2.03×** | 18,345 → 16,645 MiB，**−9.3%** | 提速一倍，显存几乎没省 |
| gradient checkpointing | s2048 b1，flash | 10,903 → 8,642 tok/s，**−20.7%** | 16,645 → 11,982 MiB，**−28.0%** | 真正的收益不是省这 28%，而是把 OOM 墙从 4,096 token/步推到 12,288 |
| flash SDPA（对 math） | s512 b4 | 8,966 → 11,539 tok/s，**+28.7%** | 19,328 → 16,644 MiB，**−13.9%** | 同时更快更省；s=2048 时 math 直接 OOM |

**AMP 只省了 9.3% 显存**是本日最反直觉的一项，原因在 3.5：参数、梯度、AdamW 两个
state 全是 fp32，合计 9.09 GB **与精度无关**，autocast 只改变算子精度。

**Q4：权衡曲线的拐点在哪？—— 16.3 GiB 处有一个陡峭的膝点。**

见 `results/tradeoff.png`。Pareto 前沿：

| 峰值显存 (reserved) | 吞吐 | MFU | 配置 |
|---|---|---|---|
| 11.60 GiB | 3,454 tok/s | 6.65% | b1 s512 bf16 flash |
| 12.65 GiB | 7,120 tok/s | 13.71% | b2 s512 bf16 flash |
| 15.20 GiB | 9,079 tok/s | 17.48% | b4 s512 bf16 flash + ckpt |
| **16.29 GiB** | **11,623 tok/s** | 21.06% | b8 s256 bf16 flash |
| 21.28 GiB | 12,591 tok/s | 24.24% | b6 s512 bf16 flash |

从 11.6 → 16.3 GiB，多花 4.7 GiB 换来吞吐 **3.4 倍**；从 16.3 → 21.3 GiB，
再花 5.0 GiB 只换来 **8.3%**。膝点就在 16.3 GiB，即 2,048 token/步那一档。

**24 GB 上应该选哪个：`b6 s512 bf16 flash`（21.3 GiB，12,591 tok/s）如果这台卡
只跑这一件事；否则选 `b8 s256` 或 `b4 s512`（16.3 GiB，11,5xx tok/s）。**
理由：后者让出 5 GiB 而只损失 8%，这 5 GiB 是 OOM 与不 OOM 的距离——
本日 7 个 OOM 配置里有 4 个崩溃前的 reserved 都在 23.1 GiB，
把工作点顶到 21.3 GiB 意味着任何一点额外分配（换个 SDPA 后端、多一个日志张量）
都会翻车。**seq_len 更长不是理由**：在同样 2,048 token/步下，s=2048 与 s=512
的吞吐只差 5.5%（10,903 对 11,539），想要长上下文几乎是免费的。

---

## 2. 测量协议

沿用 Day 01 / Day 03 那套，训练侧的具体化：

- **合成数据**：`torch.randint` 固定 seed 42 生成定长 token，`labels = input_ids`。
  不用真实 dataloader —— profiling 要的是可复现的计算量。
- **同步**：每个计时点前后都 `torch.cuda.synchronize()`，否则量到的是下发时间。
- **丢弃 warmup 20 步**（依据见 2.1），之后 3 个 block × 每 block 10 步，
  报 block 中位数的中位数与 block 间标准差。
- **每个配置一个独立进程**，避免上一次的内存池状态污染下一次。
- **计时与 profiling 分两次跑。** 报告中所有吞吐/显存数字都来自不挂 profiler 的
  `train_loop.py`；profiler 只用于定位算子。
- **OOM 不重试、不降 batch**，原样落盘（哪个配置、第几步、崩溃前三档显存）。
- 每次实验前后 `nvidia-smi` 确认显存归零（`results/nvidia_smi_{before,after}.txt`）。

### 2.1 丢几步：不拍脑袋，画出来

`results/step_curve.json`，s2048 b1 冷启动逐步耗时：

| step | 0 | 1 | 2 | 3 | 4 | … | 19 |
|---|---|---|---|---|---|---|---|
| ms | **1595.2** | 187.4 | 186.8 | 186.6 | 186.4 | … | 186.4 |

**只有第 0 步异常（8.5 倍），第 1 步起就已经稳态**，第 1–19 步全部落在
186.2–187.4 ms，相对中位数 ±0.3%。CUDA 上下文、cuBLAS 句柄、内存池增长
全部发生在第 0 步之内。技术上丢 1 步就够；本日正式测量仍丢 20 步，
因为代价只有 4 秒，而"丢少了"是不报错的静默错误。

### 2.2 噪声底

同一配置在矩阵里被不同 tag 覆盖多次，用作噪声底（`results/tradeoff.log`）：

| 配置 | n | tokens/s 范围 | 极差 |
|---|---|---|---|
| s2048 b1 bf16 flash | 4 | 10,861 – 10,907 | **0.42%** |
| s512 b4 bf16 flash | 4 | 11,486 – 11,560 | **0.64%** |
| s512 b4 bf16 mem_efficient | 2 | 11,231 – 11,245 | 0.12% |
| s512 b4 bf16 flash **+ckpt** | 5 | 8,542 – 9,079 | **5.99%** |
| s2048 b1 bf16 flash **+ckpt** | 4 | 8,103 – 9,035 | **10.79%** |

**不开 checkpointing 时进程间重复性优于 0.7%；开了以后劣化到 6–11%**，
而单进程内 block 间标准差仍只有 0.3–2.4 ms。也就是说抖动发生在进程之间而非步之间，
`use_reentrant=False` 的重算路径对每次运行的显存布局敏感。
本文所有 checkpointing 的数字都取多次运行的中位数（n=4 / n=5），
**单次测量的 checkpointing 结果不可信到 10% 以内**。

### 2.3 三个显存指标用的是哪个

三个数在 `results/matrix_summary.tsv` 里全部保留。s2048 b1 bf16 flash：

```
max_memory_allocated  16,645 MiB   张量实际占用
max_memory_reserved   16,684 MiB   缓存分配器持有（+39 MiB）
nvidia-smi            17,165 MiB   含 CUDA context（+481 MiB）
```

- **权衡曲线的横轴用 `reserved`**，因为它才是决定"还能不能再塞一个配置"的那个数。
- **开关对比的显存栏用 `allocated`**，因为要比的是算法本身的占用。
- 这个选择会改变结论：开 checkpointing 后 `allocated` 降 28.0%，
  而 `reserved` 只降 6.7%、`nvidia-smi` 只降 6.5% —— 分配器把缓存留着没还给驱动。
  **用 nvidia-smi 衡量 checkpointing 的收益，会把 28% 看成 6.5%。**

---

## 3. MFU：分子分母都手推一遍

### 3.1 分母：实测算力与带宽，不抄规格书

`results/gemm_peak.json`，纯 `torch.matmul` 方阵扫描：

| N | 1024 | 2048 | 3072 | 4096 | 6144 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|---|---|---|
| bf16 TFLOPS | 121.4 | **155.6** | 145.3 | 135.8 | 148.9 | 145.7 | 144.3 | 145.9 |
| fp32 TFLOPS | 40.8 | 48.0 | 46.7 | 47.0 | 46.8 | 43.5 | — | — |

| | 规格 | 实测 | 比值 |
|---|---|---|---|
| bf16 稠密算力 | 165.2 TFLOPS | **155.57** | **94.2%** |
| fp32 算力 | 82.6 TFLOPS | 48.07 | 58.2% |
| 显存带宽 | 1008 GB/s | **948.4** | **94.1%** |

带宽三种模式：copy（1 读 1 写）865.8、add（2 读 1 写）917.7、reduce（1 读）948.4 GB/s。

由此 **ridge point = 155.57e12 / 948.4e9 = 164.0 FLOP/byte**。

> **与任务书的预期不符，已核对。** 任务书 4.2 说"实测通常只有规格的 70–85%"，
> 实测是 94.2%。测量本身经得起检查：`out=` 预分配避免了分配开销，
> 计时含 `synchronize`，10 次 warmup 后取 20 次平均，fp16 独立测得 156.45 TFLOPS
> 与 bf16 互证。这不对应任务书第 5 节的任何一条故障，已记入 journal。
>
> 一个必须说明的口径：`torch.backends.cuda.matmul.allow_tf32 = False`（torch 2.14 默认），
> 所以 fp32 那一列是**真 fp32 CUDA core**，不是 TF32 tensor core。
> 若打开 TF32，fp32 的数字会跳到与 bf16 同量级，AMP 的加速比会完全变样。

模型里真实出现的 GEMM 形状（bf16，`results/gemm_peak.json` 的 `model_shapes`）：

| tokens | q_proj | kv | o_proj | gate/up | down | lm_head |
|---|---|---|---|---|---|---|
| 2,048 | 123.2 | 115.5 | 124.7 | 117.8 | 154.7 | 105.6 |
| 8,192 | 151.1 | 158.3 | 164.1 | 151.6 | 138.9 | 150.4 |
| 16,384 | 153.0 | 142.7 | 142.0 | 156.5 | 134.3 | 149.3 |

2,048 token 那一档（也就是本日的参考工作点）**每个形状都明显低于峰值**，
lm_head 只有 105.6 TFLOPS。这解释了 E2 的 90.3% 为什么不是 100%：
不是 kernel 不好，是 m 维只有 2,048 时 tensor core 吃不满。

### 3.2 分子：`6N` 从哪来

一个 `[in, out]` 的线性层，对每个 token：

```
forward        y = xW                  in×out 次乘加 = 2·in·out FLOPs = 2 × 该层参数量
backward dgrad dL/dx = dL/dy · Wᵀ      再 2 × 参数量
backward wgrad dL/dW = xᵀ · dL/dy      再 2 × 参数量
                                        ────────────────────────
                                        fwd + bwd = 6 × 参数量
```

backward 是 forward 的 2 倍，因为它要算两个梯度（对输入、对权重），
每个的计算量与 forward 相同。对所有非 embedding 权重求和即得每 token `6N`。

attention 的两个 batched matmul 不含参数，单列：QK^T 每层每 token `2·s·d_attn`，
A·V 同量，forward 合计 `4·L·s·d_attn`，×3 得 `12·L·s·d_attn`。

代入实际配置（`scripts/model_flops.py`，参数量与 `named_parameters()` 逐字节核对）：

```
每层 = q(1024×2048) + k(1024×1024) + v(1024×1024) + o(2048×1024)
     + gate/up/down(3×1024×3072) + q_norm/k_norm(256) + 2×RMSNorm(2048)
     = 15,730,944                          ← 实测 layer0 参数量 15,730,944 ✓
N_非emb = 28 × 15,730,944 + 1024 = 440,467,456   ← 实测 596,049,920 − 155,582,464 ✓
```

**这里有个会静默出错的地方**：任务书写的是 `12·L·h·s`，`h` 为隐藏维。
Qwen3-0.6B 的 `head_dim = 128`、`num_attention_heads = 16`，
所以 attention 的实际宽度是 `16 × 128 = 2048`，而 `hidden_size` 只有 **1024**。
**两者差 2 倍**，代 `h = 1024` 会把 attention 项低估一半。本文一律用 `d_attn = 2048`。

| 每 token FLOPs | s = 512 | s = 2048 |
|---|---|---|
| `6N`（非 embedding） | 2.643 G | 2.643 G |
| `12·L·d_attn·s` | 0.352 G | 1.409 G |
| 小计（任务书口径） | 2.995 G | 4.052 G |
| **attention 项占比** | **11.8%** | **34.8%** |
| lm_head `6·N_emb` | 0.933 G | 0.933 G |
| 含 lm_head 合计 | 3.929 G | 4.986 G |
| **lm_head 占比** | **23.8%** | **18.7%** |

seq 从 512 涨到 2048，attention 项占比从 11.8% 涨到 34.8%（近 3 倍），
因为它随 `s` 线性增长而 `6N` 不变。

### 3.3 一个被标准口径漏掉的大头：lm_head

`tie_word_embeddings = true`，所以 lm_head 的权重就是 embedding，
按"非 embedding 参数"的定义被排除在 `N` 之外。但它是一个货真价实的
`[tokens, 1024] × [1024, 151936]` 矩阵乘，每 token 0.933 GFLOPs。

对 0.6B 这种隐藏维小、词表大的模型，**它占总计算量的 18.7%（s=2048）到 23.8%（s=512）**。
换句话说，按标准口径报出的 MFU 28.4% 与把 lm_head 算进去的 34.9% 相差 6.5 个百分点，
而两者都"正确"——只是口径不同。本文两个数都给，比较时必须注明用的是哪个。

### 3.4 MFU 与 HFU 的区别

开 gradient checkpointing 后，硬件多跑了一次 forward：

```
不开 ckpt   forward 1 单位 + backward 2 单位 = 3 单位
开   ckpt   forward 1 + 重算 forward 1 + backward 2 = 4 单位
HFU / MFU = 4 / 3 = 1.3333
重算带来的额外 FLOPs 占硬件总 FLOPs = 1/4 = 25%
```

实测（s2048 b1，n=4 取中位数）：**MFU 22.51%，HFU 30.01%，比值 1.333**。
比值是按定义构造的，真正被实测检验的是：吞吐只掉 20.7%，
说明重算那 25% 的额外 FLOPs 跑得比平均更快——因为重算的是 forward，
而 forward 的算子结构比 backward 更规整。

**把这两个数混为一谈会得出相反的结论**：只看 HFU 会觉得"开 checkpointing
让硬件利用率从 28.4% 提到 30.0%，是个改进"，而实际产出的 token 少了 20.7%。
HFU 衡量硬件忙不忙，MFU 衡量有没有产出，**选配置只能看 MFU**。

### 3.5 AMP 为什么只省 9.3% 显存

`torch.autocast` 不改变参数 dtype。s2048 b1 的静态占用：

```
参数   fp32   596,049,920 × 4 B = 2.27 GB
梯度   fp32                       2.27 GB
AdamW  exp_avg + exp_avg_sq       4.55 GB
                                 ─────────
                                  9.09 GB   ← 与是否 AMP 无关
```

AMP 只把激活和中间张量变成 bf16。实测 fp32 18,345 MiB → bf16 16,645 MiB，
省下 1,700 MiB —— 这 1,700 MiB 就是激活减半的那部分。
**"AMP 省一半显存"只在激活占绝对主导的大 batch 场景成立**，
在本日这个静态开销占 55% 的工作点上不成立。

---

## 4. 开关对比矩阵

完整 40 条运行记录见 `results/matrix_summary.tsv`，逐条原始 json 在 `results/runs/`。
下表取同配置多次运行的中位数。

### 4.1 精度（同用 mem_efficient 后端，保证单变量）

| 工作点 | 精度 | step ms | tokens/s | alloc MiB | MFU |
|---|---|---|---|---|---|
| s2048 b1 | fp32 | 420.1 | 4,875 | 18,345 | 12.70% |
| s2048 b1 | bf16 | 206.7 | 9,909 | 16,645 | 25.81% |
| s512 b4 | fp32 | 344.3 | 5,948 | 18,344 | 11.45% |
| s512 b4 | bf16 | 182.2 | 11,238 | 16,644 | 21.64% |

加速 **2.03× / 1.89×**，与实测算力比 155.6 / 48.1 = 3.23× 相比打了折扣——
因为 62.8% 的时间花在不吃算力的地方（Q1），那部分不因换精度而变快。
fp32 下 profiler 显示 GEMM 占 45.7% + attention 28.1%，
计算侧占比反而更高，正说明 fp32 把计算部分拖慢了。

> 为什么精度对照不用 flash 后端：flash SDPA 不支持 fp32，指定后会被拒绝。
> 两边都用 mem_efficient 才是单变量对照。

### 4.2 gradient checkpointing

| 工作点 | ckpt | step ms | tokens/s | alloc | reserved | nvidia-smi | MFU | HFU |
|---|---|---|---|---|---|---|---|---|
| s2048 b1 | off | 187.8 | 10,903 | 16,645 | 16,684 | 17,165 | 28.40% | 28.40% |
| s2048 b1 | on (n=4) | 237.1 | 8,642 | **11,982** | 15,564 | 16,045 | 22.51% | 30.01% |
| s512 b4 | off | 177.5 | 11,539 | 16,644 | 16,682 | 17,163 | 22.22% | 22.22% |
| s512 b4 | on (n=5) | 228.2 | 8,975 | **11,982** | 15,560 | 16,041 | 17.28% | 23.04% |

**它的价值不在这 28%，而在把 OOM 墙往后推：**

| 每步 token 数 | 2,048 | 4,096 | 6,144 | 8,192 |
|---|---|---|---|---|
| 无 ckpt | 11,539 tok/s | **OOM** | **OOM** | **OOM** |
| 开 ckpt | 8,975 tok/s | 11,197 tok/s | 11,022 tok/s | **OOM** |

**但推完之后并没有更快**：能跑的最大配置 b12+ckpt 是 11,022 tok/s，
反而低于不开 ckpt 的 b6 的 12,591 tok/s。在这张卡上、这个模型上，
**checkpointing 换来的容量没有转化成吞吐**。原因见 4.4：
它省的是激活，而挡路的是 logits，而 logits 不被 checkpointing 省掉。

### 4.3 SDPA 后端（后端身份经 kernel 名验证，不靠"我指定了什么"）

| 工作点 | 后端 | step ms | tokens/s | alloc MiB | 实际执行的 kernel |
|---|---|---|---|---|---|
| s2048 b1 | flash | **187.8** | 10,903 | 16,645 | `pytorch_flash::flash_fwd_kernel` / `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` |
| s2048 b1 | cudnn | 191.0 | 10,723 | 16,645 | 同上（回落到 flash 路径） |
| s2048 b1 | mem_efficient | 206.7 | 9,909 | 16,645 | `fmha_cutlassF/B_bf16_aligned_64x64_k128` |
| s2048 b1 | math | **OOM** | — | 22,338（崩溃前） | 无专用 attention kernel |
| s512 b4 | flash | **177.5** | 11,539 | 16,644 | flash |
| s512 b4 | cudnn | 178.8 | 11,457 | 16,644 | flash |
| s512 b4 | mem_efficient | 182.2 | 11,238 | 16,644 | cutlass fmha |
| s512 b4 | math | 228.4 | 8,966 | **19,328** | 拆成 bmm + softmax |

验证方式（任务书 5.5）：读 profiler 里真实出现的 kernel 名。三条结果：

1. **`math` 后端确实没有融合 attention kernel。** softmax 的 launch 次数
   从 flash 的 6 次/3 步 暴涨到 **174 次**，attention 被拆成 `bmm → softmax → bmm`，
   并物化了 `[b, 16, s, s]` 的注意力矩阵。s=512 时多占 2,684 MiB；
   s=2048 时这个矩阵是 `1×16×2048×2048` bf16 = 128 MiB/层 × 28 层，直接 OOM。
2. **指定 `cudnn` 实际跑的是 flash kernel。** 两者的 kernel 名、显存、吞吐
   （191.0 对 187.8 ms，差 1.7%）全部一致。SDPBackend 里有 `CUDNN_ATTENTION` 这个枚举
   不代表这条路径在这个 dtype/head_dim 组合上会被选中——这正是"指定了不等于生效"。
3. **`fmha_cutlassF/B` 是 mem_efficient 而不是 flash。** 第一版脚本把
   `fmha` 当成 flash 的标志，导致 fp32 那次 profiling 把 mem_efficient 的 kernel
   标成了 flash。改成按 `pytorch_flash::` 前缀识别才对。

### 4.4 batch 扫描与 OOM 边界

s=512，从 b=1 倍增：

| batch | tokens/步 | step ms | tokens/s | alloc MiB | 状态 |
|---|---|---|---|---|---|
| 1 | 512 | 148.2 | 3,454 | 11,538 | ok |
| 2 | 1,024 | **143.8** | 7,120 | 12,310 | ok |
| 4 | 2,048 | 177.5 | 11,539 | 16,644 | ok |
| 6 | 3,072 | 244.0 | 12,591 | 20,972 | ok |
| 8 | 4,096 | — | — | 22,946（崩溃前） | **OOM，第 0 步，cross_entropy 里申请 2.32 GiB 失败** |
| 12 | 6,144 | — | — | 22,314（崩溃前） | **OOM，第 0 步** |

**b=1 到 b=2，单步耗时反而下降 4.4 ms，吞吐翻倍。** 这是 E3 的直接观测形式：
b=1 的那 148.2 ms 里绝大部分不是在算这 512 个 token。

**OOM 只由每步 token 总数决定，与怎么拆无关。** 这一点做了专门对照：

| 每步 token | (s, b) 组合 | `max_memory_allocated` | 结果 |
|---|---|---|---|
| 2,048 | (256, 8) / (512, 4) / (1024, 2) / (2048, 1) | 16,643 / 16,644 / 16,644 / **16,645** MiB | 全部 ok |
| 4,096 | (512, 8) / (1024, 4) / (2048, 2) | 22,946 / 22,946 / **22,947** MiB | 全部 OOM |

四种完全不同的 (s, b) 拆法，显存相差 **2 MiB**。原因是主导项为 lm_head 的输出：

```
logits [tokens, 151936]  bf16                    tokens × 0.29 MB
       交叉熵内部 upcast 到 fp32                  tokens × 0.58 MB
       log_softmax 的保存 + 反传梯度               再各一份
4,096 token 时这一族张量合计 ≈ 9.5 GB，叠加 9.09 GB 静态开销即触顶
```

崩溃点也吻合：报错来自 `transformers/loss/loss_utils.py` 的
`nn.functional.cross_entropy`，申请 2.32 GiB 失败。

**这是本日最有部署价值的一条**：想在 24 GB 上训长上下文，
拆 batch 或拆 seq 都没用，要么做 chunked / fused cross-entropy，
要么砍词表。checkpointing 只能把墙从 4,096 推到 12,288 token，推不掉。

---

## 5. CNN 对照（4.6）

Day 05 的检测模型 `/root/autodl-tmp/sod/best.pt`（YOLOv8n 系，1024×1024，batch 4），
同样的计时纪律，同样的 profiler 口径：

| | Qwen3-0.6B (s2048 b1) | YOLOv8n-det (1024², b4) |
|---|---|---|
| 单步 | 187.8 ms | 54.6 ms |
| `max_memory_allocated` | 16,645 MiB | 3,840 MiB |
| 不同 kernel 种类 | 68 | 102 |
| 每步 kernel launch | 5,506 | 1,534 |
| 主算子 | gemm 28.7% | **conv 34.0%** |
| 逐元素 | 26.2% | 25.0% |
| 归一化 | （RMSNorm 计入 elementwise） | **batchnorm 20.7%** |
| optimizer | **28.8%** | **0.26%** |

**预期对了一半。** 预期 CNN 更 memory-bound、kernel 更碎、launch 更多——
kernel 种类确实更多（102 对 68），但**每步 launch 数反而少 3.6 倍**，
因为 transformer 那 28 层的 RMSNorm / rope / 残差每层都要发一串小 kernel。

**没预期到的是最大的差别：optimizer 占比 28.8% 对 0.26%，差了 110 倍。**
这不是结构差异，是参数量差异：0.6B 对约 3M，AdamW 每步要扫过的字节数差两个量级。
**"transformer 训练更 compute-bound"是错的说法**；真正的区分是
"参数量大的模型，优化器本身就是一个 memory-bound 的大头"。
CNN 的 batchnorm 20.7% 则是 transformer 完全没有的一块——
BN 要跨 batch 维做两趟归约，算术强度极低。

> **口径说明**：ultralytics 的 `DetectionModel.loss(batch)` 在本机取不到
> （`AttributeError`，见 `results/cnn_train.json` 的 `loss_mode`），
> 退回用"输出求和"作为反传起点。这**漏掉了 TAL 标签分配器**那部分开销，
> 分配器主要在 CPU 上，会让上表的 GPU 侧占比偏向卷积。
> 因此本节只用于定性对比，不作为 CNN 训练的吞吐基线。

---

## 6. 局限

- **单卡，通信未测。** 本日所有现象都不涉及通信，也没有任何一条被解释成通信问题。
  多卡的 all-reduce 代价留待 Day 09 实测。
- **`nsys` / `ncu` 在本机不存在**，所以没有 SM occupancy 和实测 DRAM 流量。
  bound 判定用的三条证据都建立在 `torch.profiler` 和吞吐缩放上，没有硬件计数器旁证。
- **只测了一个模型、一个优化器。** optimizer 占 28.8% 这个结论依赖
  "AdamW + fp32 master weights"；换成 8-bit optimizer 或 SGD 结论会变。
- **checkpointing 的数字进程间抖动 6–11%**（2.2），本文取中位数，
  但 `n=4/5` 仍不足以给出可靠的置信区间。
- `cudnn` 后端未能真正被激活（4.3 第 2 条），所以"cudnn attention 更快与否"
  本日**没有测到**，表里那一行实际是 flash 的重复测量。
- CNN 对照的 loss 口径不完整（第 5 节末）。

---

## 7. 原始数据

```
results/gemm_peak.json          GEMM 峰值 + 带宽扫描（分母）
results/step_curve.json         冷启动 20 步逐步耗时（决定丢几步）
results/matrix_summary.tsv      40 条运行的汇总表（三档显存 / MFU / HFU 全在）
results/runs/*.json             每条运行的原始记录，含 OOM 现场
results/profiles/*.json         5 个配置的算子分解与 SDPA 后端 kernel 名
results/profiles/*_table.txt    profiler 原始表
results/bound_analysis.json     E1 / E2 / E3 三条证据
results/tradeoff.png            显存-吞吐权衡曲线与 Pareto 前沿
results/cnn_train.json          CNN 对照
results/report_tables.md        本文表格的自动生成版（防转抄错误）
results/v1_s2048b4_oomwall/     第一版矩阵（基准定在 s2048 b4，全线 OOM）的原始证据
results/nvidia_smi_{before,after}.txt
scripts/model_flops.py          FLOPs 手推实现
scripts/train_loop.py           训练循环（计时 / 显存 / OOM 捕获）
scripts/gemm_peak.py            分母微基准
scripts/profile_run.py          profiler
scripts/analyze.py              bound 判定
scripts/plot_tradeoff.py        权衡曲线
scripts/cnn_profile.py          CNN 对照
scripts/run_matrix2.py          开关矩阵驱动
```
