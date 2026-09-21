# 边界与未完成

**这一页和结论同等重要。** 把没做过的、只有外推的、被自己推翻的分开列清楚，
是为了让读者知道哪些数字可以直接用、哪些只能当参考。

---

## 1. 完全没做过，只有原理理解

以下内容本仓库**一次都没实测过**，任何相关的表述都只是原理层面的理解：

| 内容 | 为什么没做 | 相邻的实测（可以当作它的前提或类比） |
|---|---|---|
| **投机解码** | 没实现 | [01](../experiments/01-inference-baseline/) 实测 batch=1 的 TPOT 17.3 ms ≈ 权重读取下限 17.8 ms，**算力确实大量闲置**——这是它成立的前提，但方法本身没测 |
| **MoE 并行** | 没有 MoE 模型与多机环境 | [11](../experiments/11-fault-tolerance/) 实测过"掉队者"这个机制的一般形式：同步操作的耗时是所有 rank 里最慢的那个 |
| **张量并行 / 流水线并行** | **单机双卡做不了有意义的对照** | [09](../experiments/09-parallel-strategies/) 实测本机 all-reduce 只有 15.3 GB/s、是卡内的 1/60，可以说清"为什么不能做"，但说不清"做了会怎样" |
| **真实千卡集群** | 最大只到 2 张卡 | [11](../experiments/11-fault-tolerance/) 的千卡数字全是外推，见下节 |
| **MIG / MPS** | 消费级 4090 不支持 MIG | [12](../experiments/12-containers-and-kubernetes/) 实测了 Kubernetes 这一侧的后果（扩展资源必须 requests==limits、只能整数） |
| **AWQ / GPTQ 等 4bit** | [04](../experiments/04-fp8-quantization/) 明确记录"AWQ-4bit arm 未跑" | 同一天测过 FP8 的 weight-only，**反量化开销在 batch=1 时抵消了访存收益**；4bit 的反量化开销只会更大——但这是推论不是实测 |
| **PagedAttention 的开关对照** | vLLM 里分页是默认且不可关 | [02](../experiments/02-kv-cache-and-serving-params/) 量化了它针对的那个浪费：`max-model-len` 减半使 `Maximum concurrency` 从 2.71× 跳到 5.36×，**而实测吞吐在噪声内** |
| **GPU 节点上的 K8s 调度** | 没有 GPU 节点、没有 device plugin | [12](../experiments/12-containers-and-kubernetes/) 写了 manifest 与论述，**明确标注"未在真实 GPU 节点上验证"，未伪造任何 kubectl 输出** |

---

## 2. 只有外推，不是实测

这三处的结论**依赖外部假设或线性缩放**，原文里都已逐条标注：

### 2.1 NVLink 上的并行策略排名（[09](../experiments/09-parallel-strategies/) §7.2、[10](../experiments/10-communication-tuning/) §7）

用本机实测的 15.30 / 12.37 GB/s 与 **NVLink 4.0 的规格值 900 GB/s** 做线性缩放，
推出"四种策略的吞吐会收敛到彼此相差几个百分点"。

**未经实测**：本机没有第二种互联可比；线性缩放忽略了延迟项与 kernel 启动开销，
通信降到几毫秒后这些项会变成主导，**所以外推值是乐观下界**。

### 2.2 千卡规模的 checkpoint 开销（[11](../experiments/11-fault-tolerance/) §7）

**本仓库实测的只有**：δ = 5.923 s、单写者 1.21 GB/s、δ 的抖动、D2H 26.3 GB/s、
小消息 all-reduce 0.110 ms。

**外部假设**：单卡 MTBF 取 Llama 3 技术报告的量级、共享存储聚合写带宽 100 GB/s、
模型 70B 按 16 字节/参数。**这三条都不是本仓库测的。**

其中**最有把握的一条只依赖除法**：共享存储 100 GB/s 在 **N ≈ 83 个并发写者**时饱和，
此后再加卡不会让保存变快，只会让故障更频繁。

### 2.3 异步快照的收益（[11](../experiments/11-fault-tolerance/) §7.4）

"阻塞从 5.92 s 降到约 0.27 s（22 倍）"是按实测 D2H 带宽**推算**的，
**本仓库没有实现过异步写**。

---

## 3. 定位到现象但没定位到原因

| 项 | 定位到了什么 | 没定位到什么 |
|---|---|---|
| ZeRO-3 的 prefetch 为什么没生效（[09](../experiments/09-parallel-strategies/) §8） | 重叠系数 0.76、GPU 空泡 23.8%、通信 kernel 启动 405 次 vs FSDP 258 次 | **没扫过 `stage3_prefetch_bucket_size` / `stage3_max_live_parameters`**，不知道能不能调好 |
| cudnn attention 后端（[08](../experiments/08-single-gpu-profiling/) §4.3） | 指定 cudnn 后**实际执行的是 flash kernel** | "cudnn attention 更快与否"**本身没测到** |
| FP8 的反量化开销（[04](../experiments/04-fp8-quantization/)） | 6–7 ms，**由抵消关系反推** | 不是直接测量；直接验证需要 profiler 逐 kernel 计时 |
| SIGSTOP 之后那段 92 秒尾巴（[11](../experiments/11-fault-tolerance/) §6.2） | 由 dump + watchdog 退出 + agent 发现 abort 三段构成 | **没有逐段定位到各自由哪个变量控制** |

---

## 4. 硬件与规模上的天花板

- **最多 2 张卡**，且**无 NVLink、驱动禁用 P2P、NCCL 走共享主机内存**。
  所有多卡结论都强依赖这条链路，换到 NVLink 机器上**结论会变**（[09](../experiments/09-parallel-strategies/) §7.2 已外推）。
- **P = 2**，all-reduce 的 busbw 与 algbw 恒等，这个指标要 P≥4 才有区分度。
- **本机没有 nsys / ncu**，所以没有 SM occupancy 与实测 DRAM 流量；
  [08](../experiments/08-single-gpu-profiling/) 的 bound 判定靠三条互相独立的证据，
  **没有硬件计数器旁证**。
- **宿主是共享的**（128 核，测量期间 loadavg ≈ 21），噪声地板已把这部分包含在内，
  但极端长尾未被 10 次重复覆盖。
- **AutoDL 容器内起不了 dockerd**（实测缺 `cap_sys_admin`、cgroup2 只读、user namespace 被禁），
  所以 [12](../experiments/12-containers-and-kubernetes/) 是在本机 WSL2 上做的。
- **Grafana 在实验机上装不上**（三个下载源实测 12 秒 0 字节），
  [13](../experiments/13-observability/) 的 dashboard JSON 交付了但**未在 Grafana 中渲染验证**，
  面板图是用同一批 PromQL 从 Prometheus 取数渲染的。

---

## 5. 被自己推翻的三个判断（原文保留，未删改）

**这一节是主动列出来的，因为它比任何一个正面结论都更能说明这套方法在工作。**

1. **[04](../experiments/04-fp8-quantization/)**：用显存带宽模型预测 FP8 的 batch=1 TPOT 会从
   17.2 降到 10.5 ms，**实测 17.8 ms，预测失败**。
   修正后的模型是 `TPOT ≈ 权重字节/带宽 + 反量化开销 + batch 相关算力项`。
   原模型在"改变 batch"和"改变硬件"两个维度上都外推成功过，**在"改变权重 dtype"这一维上失败**。

2. **[05](../experiments/05-onnx-tensorrt-deployment/) → [06](../experiments/06-int8-explicit-quantization/)**：
   把一处 +0.0379 mAP 的收益归因于"按原图长宽比 letterbox、padding 少"，
   [06](../experiments/06-int8-explicit-quantization/) 核验真实张量才发现
   **这批图全是正方形**，真实机制是四边各补 16 px 灰边（1056×1056）。
   **现象和数字是真的，原因归错了**；而且进一步发现固定 1056 也能复现同样收益，
   所以"动态形状"根本不是必要条件。

3. **[07](../experiments/07-end-to-end-serving/)**：预测 NMS 是最大单段、是新瓶颈，
   实测最大单段是 **preprocess（15.441 ms）**，NMS 只有 2.279 ms、占端到端 6.6%。
   **不能把上一个实验计数器里的分段结论直接搬到新链路上。**

---

## 6. 数据自身的三处不一致（审计发现，如实记录）

- **[09](../experiments/09-parallel-strategies/) 的 `results/comparison_tables.md` 扩展效率列分母有误**：
  自动生成的表在 LoRA 段套用了全量微调的单卡基准（13194/(2×7436)=88.7%），
  而 `report.md` 用的是 LoRA 自己的单卡基准（13194/(2×9421)=70.0%）。
  **引用扩展效率一律以 `report.md` 为准。**
- **[11](../experiments/11-fault-tolerance/) 的"0.47 s"与"0.44 s"是两次不同运行**：
  前者出自 §6.3 的弹性重启，后者出自 §6.1 与 `kill_test.log`。两个都真实。
- **[01](../experiments/01-inference-baseline/) 报告标注的原始启动日志已不存在**
  （`work/serve_startup_run1.log`）。KV cache 的 144 KiB/token 可由
  [02](../experiments/02-kv-cache-and-serving-params/) 的启动日志与理论算式两头对上，结论不受影响。
