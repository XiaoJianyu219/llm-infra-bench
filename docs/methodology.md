# 测量纪律

这个仓库里任何一个数字，都经过下面这套流程。**这一页是整个项目的方法论，
也是它和"跑个 benchmark 贴个表"最本质的区别。**

---

## 1. 预测先行：实验开始前把预测写死

每个实验的 `journal.md` 前半部分是在**任何测量之前**写下的预测，
写完就不再修改，只在下方追加实测与"判断错的地方"。

**为什么重要**：事后看数据总能编出一个解释。
只有事前写下的预测才能检验你的心智模型是不是对的。

本仓库有三次预测被自己的实验推翻，原文全部保留：

| 实验 | 预测 | 实测 |
|---|---|---|
| [04](../experiments/04-fp8-quantization/) | 按显存带宽模型，FP8 的 batch=1 TPOT 应降到 10.5 ms | **17.8 ms**，模型缺了 weight-only 量化的反量化项 |
| [05](../experiments/05-onnx-tensorrt-deployment/) → [06](../experiments/06-int8-explicit-quantization/) | 精度收益来自 `rect=True` 的长宽比 letterbox | 核验真实张量后发现**数据全是正方形**，机制是四边各补 16 px 灰边 |
| [07](../experiments/07-end-to-end-serving/) | NMS 是最大单段、是新瓶颈 | 最大单段是 **preprocess 15.441 ms**，NMS 只有 2.279 ms |

---

## 2. 噪声地板先行：先定"多大算变化"

**做法**：配置完全不变，**用独立进程**重复 10 次（不是同一进程内的多个 block——
后面比较各配置时每个配置本来就是一个独立进程，噪声地板必须同口径），取吞吐极差。

**实测值**（[10](../experiments/10-communication-tuning/results/noise_floor.json)）：

```
step time  309.07 ms  std 1.507 ms (0.49%)  极差 1.87%
tokens/s   6,626      std 32.4     (0.49%)  极差 1.87%
```

> **噪声地板 = 1.87%（吞吐极差口径）。所有 |Δ| < 1.87% 的改动一律标为"不构成结论"。**

**它挡住了什么**：扫了 10 个 NCCL 环境变量 + 5 档 `bucket_cap_mb` + 3 个 DDP 开关，
**14 个低于噪声地板**。不先定地板，这 14 个都会被写成"优化成果"。

**其它实验的地板**：
[02](../experiments/02-kv-cache-and-serving-params/) conc16 为 4.4–5.8%、conc64 为 0.1–0.7%；
[08](../experiments/08-single-gpu-profiling/) 不开 checkpointing 时进程间极差 0.42–0.64%，
**开了以后劣化到 5.99–10.79%**（所以那些数字一律取多次中位数）；
[13](../experiments/13-observability/) 面板压测三轮，并发 ≥2 时 8.4%，**并发 1 那档高达 50.3%**。

---

## 3. 分母自己测，而且换机重测

**不抄规格书。** MFU 的分母是自己扫方阵 GEMM 测出来的可达峰值，不是厂商标称值：

| | 规格标称 | 实测 | 比值 |
|---|---|---|---|
| bf16 稠密算力 | 165.2 TFLOPS | **155.57** | 94.2% |
| fp32 算力 | 82.6 TFLOPS | 48.07 | 58.2%（TF32 关闭，是真 fp32 CUDA core） |
| 显存带宽 | 1008 GB/s | **948.4** | 94.1% |

由此 **ridge point = 155.57e12 / 948.4e9 = 164.0 FLOP/byte**。

**换机必须重测。** [10](../experiments/10-communication-tuning/) 换到另一台**同型号** 4090 后：

```
all-reduce busbw   15.30 -> 12.37 GB/s   (-19.2%)
GEMM bf16 峰值     155.57 -> 169.04 TFLOPS (+8.7%)
单卡训练吞吐       7,436 -> 6,029 tok/s   (-18.9%)
```

> **同型号两台机器，算力更高但训练吞吐更低。**
> 不重测分母，[09](../experiments/09-parallel-strategies/) 的全部归因搬过来都是错的。

---

## 4. 每个数字都报口径

同一件事有多个口径时，**选哪个会改变结论**。本仓库强制标注的四组：

**① 显存三档**（allocated / reserved / nvidia-smi）

```
gradient checkpointing:  allocated -28.0%   reserved -6.7%   nvidia-smi -6.5%
ZeRO-1 相对 DDP:         allocated -41.8%   reserved -20.2%  nvidia-smi -19.3%
```

用 nvidia-smi 衡量分片/重算的收益，**会把 42% 看成 19%、把 28% 看成 6.5%**。

**② MFU 含不含 lm_head** —— 模型 `tie_word_embeddings=true`，lm_head 按"非 embedding 参数"
的定义被排除在 N 之外，但它是货真价实的 `[tokens, 1024] × [1024, 151936]` 矩阵乘。
**28.4%（标准口径）与 34.9%（含 lm_head）两个都对，差 6.5 个百分点。**

**③ aggregate vs per-GPU** —— 多卡 MFU 的分母要乘卡数，否则双卡数字凭空翻倍。

**④ busbw 的 P 系数** —— all-reduce 是 `× 2(P−1)/P`，all-gather / reduce-scatter 是 `× (P−1)/P`。
**P=2 时 all-reduce 的系数恰好为 1，这一档没有区分度**，要 P≥4 才拉开。

---

## 5. 用一个独立的物理约束交叉验证

**光看自己的数没用，要找一个不依赖同一条测量路径的约束去撞它。** 本仓库用过四次：

| 约束 | 预测 | 实测 | 偏差 |
|---|---|---|---|
| Little 定律（[01](../experiments/01-inference-baseline/)） | 并发 ÷ 吞吐 | TTFT + 127×TPOT | 四档全部 **<4%** |
| 消息大小 ÷ 实测带宽（[09](../experiments/09-parallel-strategies/)） | 2.384 GB ÷ 15.30 GB/s = 155.8 ms | profiler NCCL kernel 169.4 ms | **8.7%** |
| kernel launch 次数 × 单次开销（[10](../experiments/10-communication-tuning/)） | 46,325 × 10 µs ≈ 463 ms | 实测空档 471 ms | **1.7%** |
| 批内最大值的概率（[03](../experiments/03-continuous-vs-static-batching/)） | 1−(4/5)^16 = 97.2% → 吞吐比 0.25 | 0.94/3.27 = 0.287 | 可解释的差 |
| Little 定律（[13](../experiments/13-observability/)） | QPS × 平均延迟 | 实测在途请求数 | 五档最大 **6.2%** |

---

## 6. 其它几条固定动作

- **同步后计时**：每个计时点前后 `torch.cuda.synchronize()`，否则量到的是 kernel **下发**时间。
- **丢几步不拍脑袋**：画出冷启动逐步耗时曲线再决定。实测**只有第 0 步异常**
  （1595.2 ms vs 稳态 187 ms），第 1 步起就落在 ±0.3%。
- **计时与 profiling 分两次跑**：报告里的吞吐/显存数字一律来自**不挂 profiler** 的那次。
- **每个配置一个独立进程**，避免上一次的内存池状态污染下一次。
- **失败原样落盘，不重试到好看为止**：OOM 记录哪个配置、第几步、崩溃前三档显存；
  作废的实验保留在 `invalid_*` / `attempt1_*` 目录里。
- **"指定了"不等于"生效了"**：SDPA 后端身份按 profiler 里真实出现的 kernel 名判定
  （实测**指定 cudnn 实际跑的是 flash**）；FSDP 打印被 wrap 的模块数；
  DeepSpeed 的 batch 字段从 engine 反查而不是相信自己写进 config 的值。

---

## 7. 这套纪律的代价

诚实地说，它很慢。

- 噪声地板要 10 次独立重复，[10](../experiments/10-communication-tuning/) 光这一步就跑掉一小时
- 换机重测分母意味着 [09](../experiments/09-parallel-strategies/) 和
  [10](../experiments/10-communication-tuning/) 的数字不能直接比较，两边都要留
- 严格确定性模式让 [11](../experiments/11-fault-tolerance/) 的每步慢 **2.4%**
- 三次预测被推翻，意味着三次要回头重写结论

**但它换来的是：这个仓库里每一个数字，被追问两层都不会塌。**
