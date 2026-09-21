# 结论索引

每条一行，带出处。**完整版（Day01–12 全部可用结论，约 200 条）在
[experiments/13-observability/results/facts.md](../experiments/13-observability/results/facts.md)**，
这里是其中最值得先看的二十条。

读之前先看 [methodology.md](methodology.md) 的四条口径，尤其是"三台机器不可混比"。

---

## 推理服务

**1. KV cache 的单价是模型结构常量，与服务参数无关。**
实测 144 KiB/token，与理论 `2 × 36层 × 8 KV头 × 128 × 2B = 147,456 B` **逐字节吻合**；
五种不同的池子配置下这个数一点不变。
→ [01](../experiments/01-inference-baseline/report.md)、[02](../experiments/02-kv-cache-and-serving-params/report.md)

**2. GQA 的收益可以直接量化。** 该模型 32 个注意力头只有 8 组 KV。
若为标准 MHA，每 token 要 576 KiB，同样 3.05 GiB 只能装 **5,552** 个 token 而不是 22,208——
**GQA 把可服务上下文拉了 4 倍**。
→ [01](../experiments/01-inference-baseline/report.md)

**3. 限制并发吞吐的是 KV cache 容量，不是调度器上限。**
KV tokens 15,088→27,712（+84%）对应 conc64 吞吐 3.10→4.00 req/s（+29%），单调无反常点；
而 `max-num-seqs` 16→64 只带来 +8.4%、16→256 无进一步增益。
→ [02](../experiments/02-kv-cache-and-serving-params/report.md)

**4. 读任何"最大并发"类指标前，必须先确认它的分母。**
把 `max-model-len` 从 8192 砍到 4096，启动日志里的 `Maximum concurrency` 从 2.71× 跳到 **5.36×**，
**而实测吞吐 3.58 vs 3.60 req/s 在噪声内、KV 池子也几乎没变**——分母减半它就翻倍，纯账面。
→ [02](../experiments/02-kv-cache-and-serving-params/report.md)

**5. 吞吐饱和之后，多出来的并发全变成排队。**
并发 16→64 吞吐只涨 2%，而 TTFT_p50 涨 **16.4 倍**（844 → 13,814 ms）。
同一形状在另外两个服务上重复出现（[07](../experiments/07-end-to-end-serving/report.md) 的 p99 377.70→1492.50 ms、
[13](../experiments/13-observability/report.md) 的 49.8→524.7 ms）。
→ [01](../experiments/01-inference-baseline/report.md)

**6. 连续批处理的收益主要来自处理长度不均的能力，不是批处理本身。**
定长负载加速比只有 **1.17×**（3.82 vs 3.27 req/s），变长负载 **3.69×**，相差 3.2 倍。
最刺眼的一个数：一个只要 32 个输出 token 的请求，静态批处理下等了 **92.6 秒**，
而它自己只需要 **0.6 秒**——被放大 154 倍。
→ [03](../experiments/03-continuous-vs-static-batching/report.md)

**7. 定长负载下代价体现在延迟而不是吞吐——只报吞吐的加速比会严重低估差距。**
吞吐只差 17%，而 p50 差 3.9 倍、p99 差 5.7 倍。
→ [03](../experiments/03-continuous-vs-static-batching/report.md)

**8. 量化不是"省显存"，是在固定预算下重新分配显存。**
FP8 权重 16.65→10.18 GiB（−38.9%），KV cache 3.05→9.52 GiB（**+212%**），
**总占用几乎不变**（21,985 vs 22,041 MiB）。权重占比 75%→46%，KV 占比 14%→43%。
→ [04](../experiments/04-fp8-quantization/report.md)

**9. FP8 的 +25% 吞吐里，带宽侧贡献是零。**
按带宽模型预测 batch=1 的 TPOT 会降到 10.5 ms，**实测 17.8 ms，预测失败**——
weight-only 量化的反量化开销是每步固定的纯计算项，在 batch=1 时抵消了访存收益。
**若负载以单请求低并发为主，FP8 的 decode 一点都不会变快。**
→ [04](../experiments/04-fp8-quantization/report.md)

---

## 模型部署

**10. 一个自研注意力模块从未执行过。**
由 ONNX 导出失败暴露的 unbacked SymInt 反查，运行时探针实测执行 **0/4** 次。
逐位证明等价（最大绝对误差 **0.000e+00**）后移除，参数量减少 **34.86%** 而输出完全不变。
**这个缺陷在 eager 模式下没有任何征兆——是导出器对静态形状的要求把它暴露出来的。**
→ [05](../experiments/05-onnx-tensorrt-deployment/report.md)

**11. FP16 的精度损失全部落在框定位上，不是漏检或误分类。**
mAP50 只降 **0.0005**，mAP50-95 降 **0.0049**，相差十倍；类别翻转 **0** 次，
box 最大误差 0.556 px。半像素在 IoU 0.5 上无影响，在高 IoU 档位直接掉分。
→ [05](../experiments/05-onnx-tensorrt-deployment/report.md)

**12. 输入形状策略的代价远大于数值精度。**
输入形状带来 **0.0379** mAP50-95 的差，是 FP16 量化损失（0.0049）的 **7.7 倍**。
**先把形状策略弄对，比在精度档位上纠结值钱得多。**
（Day 06 核验后修正了这项差异的原因归因：原"长宽比 letterbox"的解释不成立，
真实机制是另一种输入策略在有效图像四边补了少量灰边，且固定形状同样能复现收益。）
→ [05](../experiments/05-onnx-tensorrt-deployment/report.md)、[06](../experiments/06-int8-explicit-quantization/report.md)

**13. INT8 精度归零的根因是输出表示，不是量化本身。**
模型最终把像素坐标与类别概率拼进同一张量后统一量化，scale 为 **8.534**，
**0–1 的概率全部落入零的舍入区间**，因此类别分数整体失效。
排除输出头后可用，但仍有超过六分之一的高分候选跌破阈值——
**argmax 未翻转不等于检测敏感度未变。**
→ [06](../experiments/06-int8-explicit-quantization/report.md)

**14. 引擎层的加速几乎没有传导到端到端。**
七段配对计时：引擎段加速 **3.81×**（5.601→1.471 ms），端到端只剩 **1.12×**（38.650→34.538 ms）。
真正的大头是 CPU 解码 13.284 ms + 预处理 15.441 ms，**合计占端到端 83%**，而 NMS 只有 2.279 ms。
服务吞吐最高到 43.9 QPS 时，**GPU 平均利用率只有 7–9%**。
→ [07](../experiments/07-end-to-end-serving/report.md)

---

## 分布式训练

**15. "MFU 只有 28.4%"不等于"算力没吃满"。**
GEMM 家族只占 CUDA 时间 **37.2%**，但这些 GEMM 自身已经跑到实测峰值的 **90.3%**；
剩下 62.8% 是优化器（28.8%）、逐元素（26.2%）等算术强度远低于 ridge point 的活，
**怎么加算力都不会变快**。三条互不依赖的证据（算子构成 / GEMM 内部效率 / batch 缩放）指向同一结论。
→ [08](../experiments/08-single-gpu-profiling/report.md)

**16. AMP 只省 9.3% 显存。**
参数、梯度、AdamW 的两个 state 全是 fp32，合计 **9.09 GB 与精度无关**；
`autocast` 只把激活和中间张量变成 bf16。**"AMP 省一半显存"只在激活占绝对主导时成立。**
→ [08](../experiments/08-single-gpu-profiling/report.md)

**17. 显存墙只由每步 token 总数决定，与 batch / seq 怎么拆无关。**
2,048 token 的四种 (seq, batch) 拆法显存只差 **2 MiB**，4,096 token 的三种拆法**全部 OOM**。
主导项是 lm_head 输出的 logits（`[tokens, 151936]`，交叉熵内部还要 upcast 到 fp32），
**它不被 gradient checkpointing 省掉**——所以正解是 chunked/fused cross-entropy 而不是调 batch。
→ [08](../experiments/08-single-gpu-profiling/report.md)

**18. FSDP 比 ZeRO-3 快 2.28 倍，而两者切的张量与通信量基本相同。**
差别不在算法，在**通信有没有被藏进计算**：重叠系数 FSDP **1.19** vs ZeRO-3 **0.76**
（后者 GPU 有 23.8% 的时间在空等），通信 kernel 启动 258 次 vs 405 次。
→ [09](../experiments/09-parallel-strategies/report.md)

**19. 百分比不能相加。**
梯度累积 k=8（+86.8%）与 micro batch 4（+88.0%）叠加实测 **+265.1%**，
而可加假设只有 +174.8%（**超可加 +90.3 pp**）；同在通信侧的两项则是**次可加**（−6.7 pp）。
**方向相反，正是拆解必须做实验而不能算术相加的理由。**
→ [10](../experiments/10-communication-tuning/report.md)

**20. "掉一张卡"是两种相差三个数量级的故障。**
进程被 `kill -9` 时 torchrun 在 **0.44 秒**内把整组拉掉（NCCL 超时根本没被触发）；
进程假死（`SIGSTOP`）时另一 rank 卡在 all-reduce，要**等满集合超时 600 秒再加约 92 秒**。
**危险的是慢的那种：进程还在、监控正常、作业已经死了十分钟。**
→ [11](../experiments/11-fault-tolerance/report.md)

---

## 服务基础设施

**21. CPU 利用率是错误的扩容指标（有直接实验证据）。**
单副本阶梯加压，负载涨 16 倍时：在途请求 1→**21**、p50 1.94→**22.36 s**、吞吐 0.48→**0.25**（反而掉一半），
而 **CPU 利用率只从 390% 走到 398%**。两个机制：
① 利用率在 `limits/requests` 处封顶（本例 **400%**，不是 100%）；
② **这个封顶值是人为设定的**，改 `requests` 就变。
→ [12](../experiments/12-containers-and-kubernetes/report.md)

**22. 多阶段构建本身只省 2.9%。**
镜像 447→434 MB，因为纯 wheel 安装没有编译工具链可丢。
而**把"瘦身技巧"叠满反而更大**（452 MB）：`apt install curl` 是纯负担
（httpGet 探针由 kubelet 发起，根本不进容器），`RUN chown -R` 会复制一整层。
→ [12](../experiments/12-containers-and-kubernetes/report.md)

**23. 容量可以标定成一个公式。**
用 cgroup v2 的 `memory.peak` 实测：峰值内存 ≈ **233 + 262 × 并发** MiB
（并发 8 预测 2329、实测 2355，误差 1.1%），据此反推 1 GiB 上限只能承受**并发 3**。
→ [12](../experiments/12-containers-and-kubernetes/report.md)

**24. 面板要能自洽。**
`QPS × 平均延迟 ≈ 平均在途请求数`（Little 定律），五个并发档最大偏差 **6.2%**。
三个指标走的是三条独立采集路径，约束成立说明没有系统性错误。
→ [13](../experiments/13-observability/report.md)
