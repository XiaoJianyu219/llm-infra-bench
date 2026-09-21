# llm-infra-bench

在一到两张 RTX 4090 上做的推理和训练性能实验，2026 年 9 月，共 13 组。
内容包括 vLLM 推理服务、ONNX / TensorRT 引擎构建、单卡和双卡训练的 profiling、断点续训，
以及容器化和监控。每个实验都记录了预期与实测结果的差异。

## 实验

| # | 内容 | 主要结果 |
|---|------|----------|
| 01 | vLLM 推理基线 | KV cache 实测 144 KiB/token，与按模型结构算出的值一致；吞吐在并发 16 附近饱和 |
| 02 | 服务参数扫描 | 并发吞吐受 KV cache 容量限制；只调参数，conc64 吞吐 +20.5%，TTFT_p50 −35.5% |
| 03 | 连续批处理 vs 静态批处理 | 定长输出下吞吐 1.17×，变长输出下 3.69× |
| 04 | FP8 权重量化 | conc64 吞吐 +25%；batch=1 的 TPOT 没有下降（预测 10.5 ms，实测 17.8 ms） |
| 05 | ONNX / TensorRT 多精度引擎 | 由导出失败反查出恒不执行的分支，逐位验证后移除，参数量减少 34.86% 而输出不变；自建引擎后 FP16 相对加速 6.41×，mAP50 仅降 0.0005 |
| 06 | INT8 显式量化 | 全量 QDQ 后精度归零，原因是坐标和类别概率共用一个量化 scale；排除输出头后可用，但比 FP16 慢 |
| 07 | 端到端服务 | 引擎段加速 3.81×，端到端只有 1.12×，主要耗时在 CPU 解码和预处理 |
| 08 | 单卡训练 profiling | MFU 28.4%（计入 lm_head 为 34.9%），GEMM 只占 37.2% 的 GPU 时间 |
| 09 | DDP / ZeRO / FSDP 对照 | 两卡间没有 P2P，all-reduce 15.3 GB/s；FSDP 吞吐是 ZeRO-3 的 2.28 倍 |
| 10 | 通信调优 | 双卡 MFU 从 5.85% 到 31.82%；试过的 18 项 NCCL / DDP 设置里 14 项的影响小于噪声 |
| 11 | 容错与断点续训 | 保存完整状态后恢复，loss 与不中断运行逐位一致；单次保存 5.92 s，按 MTBF 1 天算最优间隔约 1000 s |
| 12 | Docker / Kubernetes | 复现了探针配置错误导致的 CrashLoopBackOff；并发 1→16 时 CPU 利用率几乎不变，在途请求从 1 到 21 |
| 13 | Prometheus 监控 | 服务指标与面板；QPS × 平均延迟与在途请求数的偏差在 6.2% 以内 |

按主题整理的说明在 `docs/`：推理服务、分布式训练、服务基础设施、方法与未覆盖内容。
所有结论和出处的汇总见 `docs/findings.md`。

## 测量方法

- 硬件峰值自己测，不用标称值：bf16 GEMM 155.57 TFLOPS（标称 165.2），显存带宽 948.4 GB/s。
  中途换过一台同型号机器，重测后 GEMM 为 169.04 TFLOPS、all-reduce 带宽低了 19.2%，
  所以 09 和 10 的数字不直接比较。
- 调参前先用同一配置跑 10 个独立进程，吞吐极差 1.87%，小于这个幅度的差异不算有效。
- 显存按 allocated / reserved / nvidia-smi 三种口径分别记录。同一个开关在不同口径下差别可能很大，
  比如 gradient checkpointing 在 allocated 上省 28.0%，在 nvidia-smi 上只看得到 6.5%。
- 能估算的地方先估再测。例如 DDP 每步 all-reduce 2.384 GB，按实测带宽算应为 155.8 ms，
  profiler 里是 169.4 ms。

细节见 `docs/methodology.md`。

## 目录
experiments/NN-topic/
  report.md    结论和依据
  results/     原始数据（json / tsv / log）
  scripts/     运行脚本
docs/          按主题整理的说明、方法与未覆盖内容
tools/         辅助脚本

模型权重和数据集没有放进仓库。

## 环境

- GPU：RTX 4090 24GB，一到两张。两卡之间没有 NVLink，驱动禁用了 P2P，NCCL 走共享内存
- 模型：Qwen3-VL-8B（推理）、Qwen3-0.6B（训练）、一个自研的小目标检测模型（引擎构建与服务化）
- 软件：vLLM、PyTorch 2.14 / CUDA 13.0、TensorRT 11.3、DeepSpeed、ONNX Runtime、Docker、kind、Prometheus

每个实验的 `results/` 里有当次的环境记录，基线配置见 `docs/baseline.md`。

## 没有覆盖的内容

以下内容没有做实验：投机解码、MoE、张量并行和流水线并行、多机多卡、MIG / MPS、
4bit 量化（AWQ / GPTQ）、GPU 节点上的 Kubernetes 调度。有三处结论是外推的，不是实测：
NVLink 下各并行策略的表现、千卡规模的 checkpoint 开销、异步保存的收益，原文里都写了所用的假设。
另外有几处实验前的判断被实测推翻了（04 的 TPOT 预测、05 对精度差异原因的解释、
07 对瓶颈位置的判断），原文保留。完整列表见 `docs/limitations.md`。

## License

MIT