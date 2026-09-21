# Day 08 · journal — 单卡训练 profiling

本文件的第 1 节在任何 GPU 测量之前写完，之后不修改，只在下方追加实测与
「判断错的地方」。写入时刻：2026-09-16 08:26（服务器时间），此时本日尚未
跑过任何 GPU 实验，唯一执行过的命令是环境勘察（`nvidia-smi` / `pip list` /
模型下载）。

---

## 1. 测量前的预测（写于 2026-09-16 08:26，不得修改）

### 1.0 预测所依据的模型常数（来自 config.json，非测量结果）

```
L (层数)            28
h (hidden_size)     1024
n_heads             16      head_dim 128  → d_attn = 16 × 128 = 2048
n_kv_heads          8       (GQA 2:1)
intermediate_size   3072
vocab_size          151936  tie_word_embeddings = true
```

手推参数量（下面所有 FLOPs 预测都基于这组数，若实测点数与此不符，
说明我数错了层内张量，这本身就是一处判断错误）：

```
每层 = q 1024×2048 + k 1024×1024 + v 1024×1024 + o 2048×1024
     + gate/up/down 3×(1024×3072) + q_norm/k_norm 256 + 2×RMSNorm 2048
     = 15,730,944
N_非embedding = 28 × 15,730,944 + 1024(final norm) = 440,467,456 ≈ 0.4405 B
N_embedding   = 151936 × 1024 = 155,582,464（tied，只算一份）
N_总          = 596,049,920 ≈ 0.596 B
```

**我预判这里会踩一个坑**：任务书给的 `12·L·h·s` 里的 `h`，在多数模型里
等于 `n_heads × head_dim`，但 Qwen3-0.6B 的 `head_dim=128` 而
`hidden/n_heads = 64`，两者差 2 倍。我将用 `d_attn = 2048` 代入，
若用 `hidden = 1024` 代入会把 attention 项低估一半。

预测的 FLOPs/token（fwd+bwd）：

| 项 | s=512 | s=2048 |
|---|---|---|
| 6N（非 embedding） | 2.643 G | 2.643 G |
| 12·L·d_attn·s | 0.352 G | 1.409 G |
| 小计（任务书口径） | 2.995 G | 4.052 G |
| attention 项占比 | **11.8%** | **34.8%** |
| lm_head 6×155.6M（任务书口径未计） | 0.933 G | 0.933 G |
| 含 lm_head 合计 | 3.928 G | 4.986 G |

**预测 A**：对 0.6B 这种「小隐藏维 + 15 万词表」的模型，lm_head 会占到
总 FLOPs 的 19%（s=2048）到 24%（s=512）。标准的「6N 非 embedding」口径
会系统性低估真实计算量。我会在 report 里同时给两个口径的 MFU。

### 1.1 GEMM 峰值与带宽（分母）

- **预测 B**：4090 bf16 稠密规格 165.2 TFLOPS，大矩阵 `torch.matmul` 实测
  峰值 **138 TFLOPS**（区间 130–145，即规格的 79–88%）。
- **预测 C**：显存带宽理论 1008 GB/s，实测（大张量 copy/add）
  **830 GB/s**（区间 780–880，77–87%）。
- 由 B/C 推出的 ridge point ≈ 138e12 / 830e9 ≈ **166 FLOP/byte**。
  算术强度高于此值的算子算 compute-bound。

### 1.2 MFU（本日最核心的一猜）

参考配置：seq_len=2048，batch=4，bf16 autocast，SDPA=flash，无 checkpointing，
AdamW，合成数据。

- **预测 D（MFU）**：**30%**，区间 22–38%。
  理由：0.6B 模型的 GEMM 形状偏小（k 维只有 1024/3072），tensor core
  吃不满；lm_head 是唯一的大 GEMM 但它被 6N 口径排除在分子外；
  RMSNorm/SiLU/rope 等 memory-bound 算子按 FLOPs 计几乎为零却要占时间。
  大模型（7B+）在 A100 上常报 40–55%，0.6B 在消费卡上应明显更低。
- **预测 E（tokens/s）**：0.30 × 138e12 / 4.052e9 ≈ **10,200 tok/s**，
  区间 8,000–12,000。对应 batch4×2048=8192 token/step → **0.80 s/step**。
- **预测 F**：把 lm_head 计入分子后，MFU 会从 30% 抬到约 **37%**。

### 1.3 三个开关的方向与幅度（任务书 4.4 要求先写方向）

| 开关 | 方向 | 幅度预测 | 依据 |
|---|---|---|---|
| bf16 AMP（对 fp32） | 提速、省显存 | 速度 **×2.2**（区间 1.8–3.0）；峰值显存 **−28%**（区间 −20…−35%） | 4090 fp32 CUDA core 82.6 TFLOPS 对 bf16 tensor core 165；显存只省激活那一块 |
| gradient checkpointing | 省显存、降速 | 显存 **−45%**（激活部分 −80%）；吞吐 **−28%**（区间 −22…−35%） | 重算多一次 forward：3 单位 → 4 单位，理论 −25%，再加 re-launch 开销 |
| flash SDPA（对 math） | 提速且省显存 | s=2048 下显存 **−35%**，速度 **+25%**（区间 +15…+45%） | math 后端要物化 b×16×2048×2048 的注意力矩阵 |

- **预测 G（HFU）**：开 checkpointing 后 HFU/MFU = **4/3 = 1.333**，
  重算带来的额外 FLOPs 占硬件总 FLOPs 的 **25%**。
- **预测 H**：AMP 省的显存会比直觉少。参数 fp32(2.38 GB) + 梯度 fp32(2.38 GB)
  + AdamW 两个 state(4.77 GB) ≈ **9.53 GB 与精度无关**，AMP 只把激活减半。
  所以如果谁报「AMP 省一半显存」，那是激活占绝对主导的大 batch 场景。

### 1.4 OOM 边界

24 GB 卡，静态开销 9.53 GB，logits 是隐藏的大头：
`vocab 151936 × s × b × 4 B`（fp32 交叉熵）在 s=2048、b=4 时就是 **4.98 GB**，
再加 bf16 的一份 2.49 GB。

- **预测 I**：s=2048、bf16+flash、无 ckpt，**batch 8 可跑，batch 16 OOM**。
- **预测 J**：开 ckpt 后 **batch 32 可跑，batch 64 OOM**（logits 不被 ckpt 省掉，
  所以不会像激活那样线性放宽）。
- **预测 K**：fp32 + math 后端在 s=2048 下 batch 4 就会 OOM。

### 1.5 bound 类型

- **预测 L**：**compute-bound**，但不是干净的 compute-bound。
  GEMM 类 kernel 占 CUDA 时间 **55–70%**；elementwise/norm/softmax 占 20–30%；
  实测吞吐只有 GEMM 峰值的 30%，差额主要被 memory-bound 的尾巴吃掉。
- **预测 M**：GPU 空泡 < 5%。batch4×2048 单步 0.8 s 量级，CPU 下发跟得上。
  若空泡 > 15%，第一怀疑是我丢 warmup 丢少了或 profiler 没关。

### 1.6 warmup

- **预测 N**：前 3 步显著偏慢（CUDA context + cudnn autotune + 内存池增长），
  第 4–8 步仍有 5% 内的抖动，**第 10 步后稳态**。正式测量丢弃前 10 步。

### 1.7 CNN 对照（4.6，若有时间）

- **预测 O**：Day05 的检测模型训练会比 transformer **更 memory-bound**：
  kernel 数量多一个量级、单 kernel 更小、elementwise(BN/激活/上采样) 占比更高，
  MFU 会明显低于 transformer（猜 10–18%），且 CPU 下发更容易成为瓶颈。

---

## 2. 执行前已经发现的、与任务书描述不符的情况

按用户的判据逐条对照任务书第 5 节（会静默出错的坑），以下三条**都不属于**
第 5 节的任何一项，故记录在此并继续执行，不停机。

**2.1 Qwen3-0.6B 不在磁盘上。**
任务书第 3 节说「Day 1–4 期间下载过 Qwen3-0.6B」。实际执行
`find /root/autodl-tmp -maxdepth 5 -name config.json -path "*Qwen*"` 只找到
`Qwen3-VL-8B-Instruct` 与其 FP8 版本，没有 0.6B。
任务书第 3 节自己给了这种情况的处置（「找不到就重新下」），
所以这不是异常，已按其指示用 modelscope 重新下载到
`/root/autodl-tmp/models/Qwen3-0.6B`（1.5 GB，用时 2 分 08 秒）。

**2.2 GPU 上有 Day 07 的实验正在跑。**
连上机器时 `nvidia-smi` 显示 2,867 MiB 被占用、GPU util 15%，
进程是 `experiments/07-end-to-end-serving/scripts/sweep.py` 起的 TensorRT 服务（tmux 会话
`day07sweep`，创建于 08:06）。该扫描共 16 个配置 × 4 并发 × 3 重复，
截至 08:25 完成 60/192 个点。
这不对应第 5 节的任何一条（第 5 节讲的是测量方法上的静默错误，
不是资源争用）。**处置：不 kill 别人的实验**（那是不可逆的破坏），
等它跑完再做任何 GPU 测量；这段时间用来装依赖、下模型、写脚本。
本日所有 GPU 数字都将在 GPU 空闲后采集，并在采集前后各记一次 `nvidia-smi`。

**2.3 装上来的是 transformers 5.17.0，不是 4.x。**
模型 config.json 里写的是 `transformers_version: 4.51.0`。跨大版本，
`gradient_checkpointing_enable` / `attn_implementation` 等接口可能改名或改语义。
按第 5 节第 8 条的纪律，**先 `dir()` 打印真实接口再写代码**，
不按版本号推断。这条算是第 5.8 条的预防性执行，不算偏差。


---

## 3. 预测 vs 实测（第 1 节写于测量之前，未作任何修改）

对照工作点：测量前我把参考配置定为 s2048/**b4**；实测发现 b4 在 24 GB 上必 OOM，
参考点被迫改成 s2048/**b1**。凡是受此影响的预测，下表在"判断错在哪"里注明。

| # | 预测 | 实测 | 判定 |
|---|---|---|---|
| A | lm_head 占总 FLOPs 19%(s2048) / 24%(s512) | 18.7% / 23.8% | ✅（推导，非测量） |
| B | bf16 GEMM 峰值 **138** TFLOPS，区间 130–145 | **155.6**（规格的 94.2%） | ❌ 超出区间上界 |
| C | 带宽 **830** GB/s，区间 780–880 | **948.4**（规格的 94.1%） | ❌ 超出区间上界 |
| D | **MFU 30%**，区间 22–38% | **28.40%** | ✅ 点估计差 1.6 个百分点 |
| E | 吞吐 10,200 tok/s，区间 8,000–12,000 | **10,903** | ✅ |
| F | 计入 lm_head 后 MFU ≈ 37% | 34.94% | ✅ |
| G | HFU/MFU = 1.333，重算额外 FLOPs 占 25% | 30.01 / 22.51 = 1.333 | ✅（按定义构造，不算预测命中） |
| H1 | AMP 提速 ×2.2，区间 1.8–3.0 | ×2.03 / ×1.89 | ✅ |
| H2 | AMP 省显存 −28%，区间 −20…−35% | **−9.3%** | ❌ 方向对，幅度差 3 倍 |
| I | s2048 无 ckpt：**b8 可跑、b16 OOM** | **b1 可跑、b2 就 OOM** | ❌ 差 8 倍 |
| J | 开 ckpt 后 b32 可跑、b64 OOM（= 65,536 token/步） | 12,288 token/步 可跑、16,384 OOM | ❌ 差 5 倍 |
| K | fp32 + math 在 s2048/b4 会 OOM | bf16 + math 在 s2048/**b1** 就 OOM | ❌ 方向对，比预测严重得多 |
| L | **compute-bound**，GEMM 占 CUDA 时间 55–70% | **memory/overhead-bound**，GEMM 家族 **37.2%** | ❌ **本日最重要的判断错误** |
| M | GPU 空泡 < 5% | ≤ 3.1% | ✅ |
| N | 前 3 步明显慢，第 10 步后稳态，丢 10 步 | **只有第 0 步慢（8.5×），第 1 步即稳态** | ❌ |
| O1 | CNN 的 kernel 数量多一个量级 | 种类 102 对 68（多 50%，不是一个量级） | ❌ |
| O2 | CNN 每步 launch 更多 | **反而少 3.6 倍**（1,534 对 5,506） | ❌ |
| O3 | CNN 更 memory-bound | conv 34% + BN 20.7% + elementwise 25%，确实更碎 | ✅ |
| — | ckpt 省显存 −45% | **−28.0%**（allocated 口径） | ❌ |
| — | ckpt 降吞吐 −28%，区间 −22…−35% | **−20.7% / −22.2%** | ⚠️ 擦着区间下界外 |
| — | flash 对 math：s=2048 省显存 −35%、提速 +25%（+15…+45%） | s=2048 下 math 直接 OOM；s=512 下提速 **+28.7%**、省显存 **−13.9%** | 速度 ✅ / 显存 ❌ |

---

## 4. 判断错在哪

### 4.1 最重要的一处：把 compute-bound 和「MFU 不高」当成了同一件事（预测 L）

我预测 compute-bound，理由是"0.6B 的 GEMM 形状偏小、tensor core 吃不满"。
前半句对：2,048 token 那一档的 GEMM 实测 105–155 TFLOPS，确实没吃满峰值。
**但我由此推出了错误的结论。** 实测是：

- 那些 GEMM 自己跑到了实测峰值的 **90.3%**；
- 它们只占 **37.2%** 的 GPU 时间；
- 剩下 62.8% 是 optimizer（28.8%）+ elementwise（26.2%）+ softmax（5.8%）。

我犯的错是**默认"时间花在哪"和"算力吃不吃得满"是同一个问题**。
MFU 28% 的成因不是"GEMM 效率低"，而是"GEMM 只占三分之一时间"。
这两种解释对应完全相反的行动：前者要换 kernel / 调形状，后者要动优化器
（fused / 8-bit）和算子融合，换算力更强的卡几乎没用。

更要命的是，如果我只测了 E2 一条（GEMM 达峰值 90.3%），
很可能会读成"计算侧很健康，所以是 compute-bound"——**这正是任务书 4.3 说的
"三条证据里任意两条一致，可能只是同一个误解的两次表述"**。
E3（batch 缩放，完全不依赖 profiler）才是把结论扳过来的那条：
b1→b2 单步耗时**不升反降**，`step = 114.3 ms + 19.7 ms × batch`，
固定开销占 64.4%。这个现象用 compute-bound 无论如何解释不通。

### 4.2 OOM 边界错了 8 倍：我算漏了 logits（预测 I / J）

预测里我算了参数、梯度、优化器状态、激活，也算了 logits，
但只算了 `[b, s, V]` 的 fp32 一份（4.98 GB @ b4 s2048），
没算 bf16 的一份、log_softmax 保存的一份、以及反传梯度的一份。
实际这一族张量在 4,096 token 时合计约 9.5 GB。

而且我把它当成了"batch 的函数"，实际它是 **token 总数的函数**。
这条被专门对照验证了：2,048 token 的四种 (s,b) 拆法显存相差 **2 MiB**，
4,096 token 的三种拆法全部 OOM 且崩溃前显存相差 **1 MiB**。
**在 24 GB 上，"seq_len 太长"和"batch 太大"根本不是两个问题，是同一个问题。**

顺带推翻了我对 checkpointing 的估计（预测 J 差 5 倍）：
checkpointing 省的是逐层激活，而挡路的是 logits，后者不在 checkpoint 范围内。
所以它把墙从 4,096 推到 12,288 token 就推不动了。

### 4.3 AMP 省显存的幅度（预测 H2）

方向我预判对了（"会比直觉少"，并且事先写出了 9.53 GB 静态开销与精度无关这条推理），
但仍然报了 −28%。实测 −9.3%。错在我按 b4 估激活占比，而实际能跑的是 b1，
激活占比更低，静态开销占到 55%，AMP 能作用的那块更小。
**同一条物理规律，在不同工作点上给出的数字可以差 3 倍。**

### 4.4 warmup 丢 10 步是拍脑袋（预测 N）

我预测"前 3 步慢、第 10 步后稳态"，凭的是经验而不是数据。
实测第 0 步 1595.2 ms，第 1 步 187.4 ms，第 2–19 步全在 186.2–187.4 ms 之间。
**第 1 步就已经稳态了。** 任务书 5.2 要求画出曲线再决定，
这条纪律的价值在这里很具体：如果我按"感觉"丢 3 步，恰好也是对的；
但如果模型更大、初始化更慢，"感觉"就会失效，而曲线不会。

### 4.5 GEMM 峰值和带宽都比预测高（预测 B / C）

两项都超出了我给的区间上界，而且**同时超出**，说明不是偶然：
我低估了这张卡在纯 back-to-back 大矩阵乘下的可达效率。
任务书 4.2 给的经验值"70–85%"在这台机器上也不成立（实测 94.2% / 94.1%）。
这不对应任务书第 5 节的任何一条具体故障，故按约定记录并继续，见 5.6。

### 4.6 CNN 的 launch 次数（预测 O2）

我预测 CNN 每步 launch 更多，实测反而少 3.6 倍。
错在我把"kernel 小而碎"和"launch 次数多"划了等号。
真实情况是 transformer 的 28 层里每层都有 RMSNorm / rope / 残差 / 类型转换
一串小算子，5,506 次 launch 里绝大多数来自这些，而不是来自矩阵乘。

---

## 5. 踩的坑（含我自己写出来的 bug）

### 5.1 `SmiSampler` 用 `self._stop` 盖掉了 `threading.Thread._stop`

第一轮全量矩阵**每一个配置都没有产出 json**，报错是
`TypeError: 'Event' object is not callable`，栈在 `Thread.join()` 内部。
`threading.Thread` 自己有一个私有方法 `_stop()`，我把实例属性起了同名，
`join()` 内部调用 `self._stop()` 时调到了 Event 对象上。
更坏的是它发生在 `finally:` 里，把已经测完的结果一起吞掉。
改名 `_stopev` 解决。**教训：给 Thread 子类加字段前先看基类占了哪些名字。**

### 5.2 带宽口径写错，算出了超过理论值的数字

第一版用 `torch.add(x, x, out=y)` 测带宽，按"2 读 1 写"计流量，
得到 **1329 GB/s**，超过 4090 的理论 1008 GB/s。
硬件只读了一遍 `x`，实际是"1 读 1 写"。改成两个不同张量 `torch.add(x, z, out=y)`
后得 917.7 GB/s。**这个 bug 之所以被抓住，只是因为结果超过了物理上界**——
如果当初写的是 3 读 1 写、算出 890 GB/s，我不会起疑，它会一路进到 report 里
当作 ridge point 的分母。已在脚本里加了 `over_theoretical` 标志，
超过理论带宽的模式不参与取峰值。

### 5.3 profiler 的 `key_averages()` 混着两套记账，我一次踩了三层

按发现顺序：

1. **CPU 侧 aten 行与 GPU 侧 kernel 行的 `self_device_time_total` 是同一段时间的两次记录。**
   最早的输出里 `aten::_efficient_attention_backward` 和 `fmha_cutlassB_...`
   时间完全相同（259.43 ms）——那不是巧合，是同一件事被记了两遍。
   必须按 `device_type == DeviceType.CUDA` 只取 kernel 行。
2. **FLOPs 只挂在 aten 行上**，kernel 行的 `flops` 恒为 0。
   所以"算子达到多少 TFLOPS"必须从 aten 行取，与第 1 条的取法正好相反。
   一份数据要按两个口径各取一次。
3. **`ProfilerStep*` 和 `Optimizer.step#AdamW.step` 也带 `device_type == CUDA`**，
   但它们是 `is_user_annotation == True` 的区间标注，不是 kernel。
   不排掉的话，`ProfilerStep*` 一项就占 21.68%，而它覆盖的是整个 step，
   等于把所有 kernel 又算了一遍。

三层都不报错，都只表现为"占比之和不是 100%"这种很容易被忽略的异常。
按任务书 5.8 的纪律，每一层都是先 `dir()` / 打印真实字段确认后再改的代码，
没有靠"应该是这样"推断。

### 5.4 把 `fmha_cutlass*` 当成了 flash 的标志

第一版的 kernel 分类正则里，`fmha` 被归到 `attention_flash`。
实际 `fmha_cutlassF/B_*` 是 **mem_efficient** 后端（xformers 那套 cutlass 实现），
真正的 flash kernel 叫 `pytorch_flash::flash_fwd_kernel`。
后果是 fp32 那次 profiling 报告"用的是 flash"，而 fp32 下 flash 根本不可用。
**这正是任务书 5.5 说的"不要假设指定了就生效"的一个变种：
不仅指定的后端可能没生效，连"我以为能识别后端的那条规则"本身也可能是错的。**
改成按 `pytorch_flash::` 前缀识别后，反而测出了一条真结论：
指定 `cudnn` 后端时实际跑的也是 flash kernel（见 report 4.3）。

### 5.5 第一版矩阵的基准工作点全线 OOM，白跑一轮

我按预测把基准定在 s2048/b4，结果 24 条配置里只有 3 条能跑。
这是 4.2 那个判断错误的直接代价。原始记录保留在
`results/v1_s2048b4_oomwall/`，没有删——它本身就是 OOM 墙的证据。
第二版把基准改成 s2048/b1，并新增"等 token 数、不同 (s,b) 拆法"的专项对照，
才把"OOM 由 token 总数决定"这条测清楚。

### 5.6 与任务书描述不符、但不对应第 5 节任何一条的现象

按约定记录并继续执行，不停机：

1. **GEMM 实测占规格 94.2%，任务书 4.2 说通常 70–85%。**
   带宽同样是 94.1%。两项同时偏高，测量方法已复核（见 4.5）。
2. **Qwen3-0.6B 不在磁盘上**（任务书 3 节说 Day 1–4 下载过）。
   任务书自己给了处置方式，已按其指示重新下载。
3. **连上机器时 Day 07 的扫描正在跑**，占 2.4 GB 显存。
   不 kill 别人的实验，等它 09:02 跑完 192/192 后才开始任何 GPU 测量。
   等待期间做的是装依赖、下模型、写脚本。
4. **transformers 装上来是 5.17.0 而非模型 config 里的 4.51.0**，
   跨大版本。`from_pretrained` 已经没有 `attn_implementation` 参数，
   改为设 `config._attn_implementation`；`gradient_checkpointing_enable`
   的签名变成 `(gradient_checkpointing_kwargs, every_n_layers, offload)`。
   都是先打印真实签名再写的代码。
5. **本机没有 `nsys` / `ncu`**，所以 bound 判定里拿不到 SM occupancy
   和硬件 DRAM 计数器。已在 report 局限里写明。
6. **服务器中途重启**（SSH 端口与密码更换），`/root/autodl-tmp` 数据完好，
   report 与 journal 是重启后补写的，所有实测数字来自重启前已落盘的
   `results/` 原始文件，未重测。

### 5.7 checkpointing 的进程间抖动

同一配置重复 4–5 次，不开 ckpt 的极差 0.42%–0.64%，
开 ckpt 后跳到 5.99%（s512 b4，n=5）和 10.79%（s2048 b1，n=4），
而**单进程内 block 间标准差仍只有 0.3–2.4 ms**。
抖动在进程之间而不在步之间。第一次测到 ckpt 慢 34.6%，
补测 3 次后中位数是慢 26.2%。
**如果我按第一次的单点数字写 report，checkpointing 的代价会被高估三分之一。**
凡是要写进结论的对照，只测一次不够。

### 5.8 工具链上的三个自伤（与任务无关，但都真实浪费了时间）

- 本机没有 `sshpass`，OpenSSH 不能从 stdin 读密码，最后在 scratchpad 建了
  隔离 venv 装 paramiko 做 SSH 通道，没有动本机全局环境。
- Git Bash 的 MSYS 路径转换把命令里的 `/root/autodl-tmp/...` 改写成
  `D:/Git/root/autodl-tmp/...`，在服务器上凭空建出一个 `~/D:` 目录（已删）。
  需要 `MSYS_NO_PATHCONV=1` 并且本地路径改用 Windows 写法。
- `pkill -f "day08/scripts"` 把执行这条命令的 shell 自己也匹配上杀掉了。
- `run_all.sh` 里等 GPU 空闲的条件写成 `pgrep -f "day07|sweep.py"`，
  而 **tmux 服务进程的 cmdline 会永久保留启动它的那条命令**（含 "day07"），
  这个循环永远不会退出。改成匹配 `experiments/07-end-to-end-serving/scripts/` 并排除 tmux 后才对。
- 服务器上 matplotlib 没有中文字体（`fc-list :lang=zh` 为空），
  图里所有文字改用 ASCII，中文说明放在 report 正文。

### 5.9 `finish_day.sh 08` 本身跑不起来

任务书第 7 节给的收尾命令是 `./finish_day.sh 08 "…"`，实际执行报
`printf: 08: invalid octal number` —— 脚本第 4 行 `printf "%02d" "$1"`，
bash 把带前导零的 `08` 当八进制解析，而 8 不是合法八进制位。
`09` 同样会失败，Day 09 收尾时要注意。
本日改用 `./finish_day.sh 8 "…"` 调用（`printf "%02d" 8` 仍得到 `08`），
没有去改 `finish_day.sh`（它是仓库根目录的共用脚本，不属于本日边界）。
