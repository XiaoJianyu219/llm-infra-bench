# 服务基础设施

覆盖 [12](../experiments/12-containers-and-kubernetes/) – [13](../experiments/13-observability/)。
载体：ONNX Runtime CPU 检测服务（12）与 TensorRT FP16 检测服务（13）。

---

## 先说环境判定：为什么这两个实验换了机器

**AutoDL 的容器实例内起不了 dockerd，而且不是"没装"**：

```
capsh --print 的 bounding set   无 cap_sys_admin
mount | grep cgroup             cgroup2 (ro,...)   只读挂载
ls /dev/fuse                    No such file or directory
unshare -Ur true                Operation not permitted   ← rootless 退路同样不通
/etc/subuid                     空
```

> **不是缺软件，是缺 `cap_sys_admin` 与可写的 cgroup。**
> 所以 [12](../experiments/12-containers-and-kubernetes/) 改在本机 WSL2 + Docker + kind 上做。
> **先判定再动手，比装半天发现装不上强。**

---

## 12 · 容器化与 Kubernetes 调度

### 镜像瘦身：逐项归因，而不是堆技巧

| 策略 | 体积（`docker images` 未压缩口径） |
|---|---|
| `python:3.12-slim` 基座 | 190 MB |
| 单阶段 + 保留 pip 缓存 | 555 MB |
| 单阶段 + `--no-cache-dir` | 447 MB |
| **多阶段** | **434 MB** |
| 多阶段 + `apt install curl` + `RUN chown -R` | **452 MB（反而更大）** |

> **多阶段构建本身只省 13 MB（2.9%）**——本日依赖全是预编译 wheel，没有编译工具链可丢。
> **多阶段的收益不取决于用不用多阶段，取决于镜像里有没有"只在构建期有用"的东西。**

**两个"看起来很专业"的动作净增 18 MB**：
`apt install curl` 是纯负担（**`httpGet` 探针由 kubelet 从节点发起，根本不进容器**）；
`RUN chown -R` 会复制一整层（overlayfs 里改属主 = 写新文件），正确写法是 `COPY --chown`。

**体积有三个互不相等的口径**（`docker images` / `docker history` 各层求和 / registry 压缩），
实测三者不一致且差值也对不上——**报体积必须报口径**。
`.dockerignore` 补全后 build context 从 **72.98 MB → 26.11 kB**。

### 探针：两个失败场景，其中一个是自己冒出来的

**场景一（人为制造）**：liveness 打到 `/readyz` 且不配 startupProbe，模型加载 90 s。
结果 `Liveness probe failed: HTTP probe failed with statuscode: 503` ×21 → 被杀 ×6 →
**CrashLoopBackOff**（重启 7 次）。
> 一个容易看错的细节：`lastState` 是 **exitCode 0 / reason Completed**——
> 进程没崩溃，是 kubelet 发的 SIGTERM。**查日志找崩溃原因会一无所获，只能从 Events 看出来。**

**场景二（不需要人为制造，它在正确配置下自己出现）**：
三探针分工清楚、liveness 只打极轻量的 `/healthz`，并发 8 压测下 **4 个 Pod 各重启 3–4 次**。
直接证据：**同一个 `/healthz` 空载 2–12 ms，压测中最高 5378 ms（约 2700 倍）**。

根因在应用代码不在 YAML：`/predict` 写成了 `async def`，
FastAPI 对 `async def` 处理函数**直接在事件循环线程里执行**，
而 ONNX Runtime 的 `run()` 是同步阻塞的 C++ 调用（1.5–4 s），**整个事件循环停转**。

> **探针的轻重由整个进程的调度决定，不由这个 handler 自己决定。**

**修好之后故障没消失，只是换了出口**：改成同步 `def` 后 liveness 失败事件归 **0**，
但四个 Pod 全部 **`exitCode 137 / OOMKilled`**，成功 27 / 失败 5405，
**比修复前（210 / 45）差一个数量级**——`async def` 版本事实上充当了限流器。

### 容量可以标定成公式

用 cgroup v2 的 `memory.peak` 实测（单副本、limit 放到 3Gi、每档换新 Pod）：

| 并发 | QPS | p50 | 峰值内存 |
|---|---|---|---|
| 1 | 0.32 | 2.84 s | 495 MiB |
| 2 | 0.24 | 7.21 s | 785 MiB |
| 4 | 0.17 | 21.91 s | 1305 MiB |
| 8 | 0.33 | 19.93 s | 2355 MiB |

```
峰值内存(MiB) ≈ 233 + 262 × 并发       并发 8 预测 2329，实测 2355，误差 1.1%
=> limits.memory: 1Gi 反推可承受并发 = (1024 − 233) / 262 ≈ 3.0
```

**空载（模型已加载、无请求）只有 90 MiB**——
**第一次推理才会把 ORT arena 撑起来，只看"服务刚起来的内存"会严重低估。**

> 所以并发 8 下 OOM **是算得出来的必然，不是意外**。
> `requests`/`limits` 从此可以是算出来的，不是拍出来的。

### CPU 利用率是错误的扩容指标（有直接实验证据）

单副本、无 HPA、并发 1→2→4→8→16 阶梯加压，同时采在途请求与 CPU：

| 并发 | 在途请求 | CPU 利用率 | QPS | p50 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 390% | 0.48 | 1.94 s |
| 4 | 4 | 400% | 0.27 | 10.31 s |
| 8 | 8 | 400% | 0.25 | 22.36 s |
| 16 | **21** | **398%** | 0.00 | 60 s 内无一完成 |

> **负载涨 16 倍：在途涨 21 倍、p50 涨 11 倍、吞吐掉一半，而 CPU 利用率 390%→398% 在噪声里。
> HPA 全程看到的是一个常数。**

两个机制：
① **利用率在 `limits/requests` 处封顶**（本例 `2 / 0.5 = 400%`，**不是 100%**），封顶后过载全藏在队列里；
② **这个封顶值是人为设定的**——`requests.cpu` 从 500m 改成 2，同样的负载会读成 100%，
**同一份 HPA YAML 挂到不同 `requests` 的 Deployment 上行为完全不同。**

### 扩缩容时间线

```
T+0     加压开始，1 副本，cpu 0%/60%
T+26s   HPA 决策  SuccessfulRescale New size: 3        <- 滞后的 63% 在这一段
T+28s   Pod 创建
T+35s   原健康副本掉出 Ready，就绪端点数 = 0           <- 约 3 秒服务完全不可用
T+38s   新副本 Ready，开始接流量
T+87s   4/4 Ready
停压后  +306s 4->3    +366s 3->2    +426s 2->1
```

**缩容与 `behavior` 配置逐条吻合**（`stabilizationWindowSeconds: 300` + `1 Pod/60s`），
**而且比扩容慢一个数量级是设计意图**（防抖），不是 bug。

**扩容过程中那 3 秒完全不可用**对应压测第一个 60 s 窗口的 3856 个失败
（其余四个窗口合计只有 1 个）——**"副本数从 1 涨到 4"这张图上完全看不出中间断过。**

### 三个静默失败

1. `kind load` 的报错被 `tail -1` 吞掉 → metrics-server `ImagePullBackOff` →
   **HPA 显示 `<unknown>` 而不报错**
2. Pod 不 Ready 时 HPA 拿不到指标 → 只发 `FailedGetResourceMetric` Warning，**副本数不动**
3. （**未实测，论述**）GPU 节点没有空闲卡时 HPA 照样扩，新 Pod 停在 `Pending`，HPA 不报错

> **共同点：控制器只负责把期望值写下去，写不下去不是它的错误路径。**

---

## 13 · 可观测性

**技术栈**：服务暴露 Prometheus 文本格式 `/metrics` → Prometheus（单二进制，
**scrape_interval 压到 1 s**）→ 面板。

**histogram 的 bucket 按实测范围设定而不是用默认值**
（默认 15 s 的 scrape 在 5 分钟压测里只采 20 个点，波形无法判读）：
端到端 bucket 取 5 ms – 5 s，依据是上一个实验实测的 p50 26 ms – 1460 ms。
**实测验证够用**：最高并发档 p99 524.7 ms 落在 500/750 ms 之间没顶到 `+Inf`。

**阶梯加压（并发 1→16，各 60 s，10,232 请求 0 失败）**：

| 并发 | QPS | p50 | p99 | 在途 | GPU% | 平均批大小 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.1 | 40.0 ms | 49.8 ms | 0.97 | 5.6 | 1.00 |
| 4 | 37.0 | 124.0 | 149.5 | 3.75 | 6.5 | 2.00 |
| 16 | 36.1 | 395.4 | **524.7** | 15.30 | 6.5 | 7.75 |

**面板自洽性校验（Little 定律）**：`QPS × 平均延迟 ≈ 平均在途请求数`，
五个并发档偏差 4.8 / 6.2 / 0.7 / 1.2 / 1.2%，**最大 6.2%**。
三个指标走的是三条独立采集路径（counter 的 rate、histogram 的 sum/count、gauge），
**约束成立说明没有系统性错误**。

**补做的噪声地板（同配置三轮独立重复）改变了一个结论**：

| 指标 | 并发 1 | 并发 2 | 并发 4 | 并发 8 | 并发 16 |
|---|---:|---:|---:|---:|---:|
| QPS 三轮极差 | **50.3%** | 8.2% | 0.7% | 3.0% | 3.5% |
| p99 三轮极差 | 25.1% | 0.1% | 0.0% | 0.1% | **24.0%** |

> 初稿写的是"吞吐在并发 4 就饱和，**后两档已在下降**"——**这句话不成立**：
> 后两档差异 3.0% / 3.5% 落在 8.4% 的噪声地板以内，**能站住的只有"不再上升"**。
> 另外**并发 1 那一档完全不可信**（三轮 27.08 / 26.92 / 40.54），
> 而 **p99 在最高并发档的极差是 24%**——**尾延迟本身就是高方差量，报 p99 必须带重复次数。**

**该用什么指标扩容**：队列深度扩容、p99 告警、利用率留给缩容
（缩容时"利用率掉下来"是可靠信号，因为那时没有封顶问题）。
两者都不是 kubelet 原生指标，要经 Prometheus Adapter 注册成 custom metrics。

**排障链固定四层**（面板空白时按顺序查，不要从最后一层开始）：
服务 `/metrics` 能不能 curl 到 → Prometheus 的 Targets 是否 UP →
PromQL 在 Prometheus 自带界面能否查出结果 → 最后才怀疑面板。
本实验第一次启动就卡在第 1 层，两分钟定位到根因是**同名模块被 cwd 抢先解析**。

---

## 边界

- **Grafana 在实验机上装不上**（三个下载源实测 12 秒 0 字节）：
  dashboard JSON 交付了但**未在 Grafana 中渲染验证**，
  面板图是用同一批 PromQL 从 Prometheus 取数渲染的，**不是 Grafana 截图**。
- **GPU 调度未在真实节点验证**：单节点 kind 无 GPU、无 device plugin，
  [12](../experiments/12-containers-and-kubernetes/) 只给了 manifest 与论述，**未伪造 kubectl 输出**。
- [13](../experiments/13-observability/) 的服务用 `MAX_BATCH=8`，与
  [07](../experiments/07-end-to-end-serving/) 推荐的 `batch=1` 不同（为了让批大小 panel 有东西可看），
  **两者跨机器跨引擎，不做严格对比**。
