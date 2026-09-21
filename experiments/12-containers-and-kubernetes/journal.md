# Day 12 · journal — 容器化与 Kubernetes 调度

第 1–2 节写于**任何构建与部署之前**，之后不修改，只在下方追加实测与「判断错的地方」。
写入时刻：2026-09-18 16:40（服务器时间）。此时只做过第 1 节的环境判定（只读探测），
没有构建过任何镜像、没有起过任何集群。

---

## 1. 环境判定结果（任务书第 1 节，先做，因为它决定后面能做什么）

任务书说「AutoDL 的容器实例内大概率无法运行 dockerd，**必须实测确认，不要采信上面这段话**」。
实测结论：**确实不行，而且不止是「没装」，是权限上根本起不来。**

```
which docker dockerd containerd podman nerdctl kind minikube kubectl   → 全部无输出
docker info                    → bash: docker: command not found
systemctl status docker        → System has not been booted with systemd as init system (PID 1)
cat /proc/1/comm               → bash          （1 号进程是 bash，不是 systemd）
cat /proc/1/cgroup             → 0::/
capsh --print 的 Bounding set  → 无 cap_sys_admin（只有 chown/dac_override/…/mknod/setfcap）
mount | grep cgroup            → cgroup2 (ro,...)   ← 只读挂载
ls /dev/fuse                   → No such file or directory
```

原始输出在 `results/env_probe.txt`。

**又测了一条退路**：rootless 构建（podman / buildah）依赖 user namespace，

```
unshare -Ur true               → unshare failed: Operation not permitted
newuidmap / newgidmap          → 未安装
/etc/subuid                    → 空
```

**user namespace 也被禁**，所以 rootless 那条路同样不通。
apt 源里有 podman 3.4.4 可装，但装了也跑不起来——**不是缺软件，是缺 `cap_sys_admin` 与可写的 cgroup**。

**本机 Windows 侧同样不具备条件**（只读探测）：

```
wsl --version          → 未安装用于 Linux 的 Windows 子系统
docker / kind / kubectl / minikube / helm → 全部未安装
Docker Desktop         → 未安装
HypervisorPresent      → False
内存 7.8 GB，C 盘剩 15.5 GB
```

> **这一条与任务书第 4 节的坑不冲突**（第 4 节第 1 条说的正是这件事），
> 但它把本日的落地环境变成一个需要用户决定的问题：
> 要么在本机装 WSL2 + Docker Desktop（管理员安装、可能要在 BIOS 里开虚拟化、重启、C 盘吃紧），
> 要么另找一台能跑 Docker 的 Linux。在用户决定之前，本日先做**不依赖该决定**的部分：
> 3.1 预测、服务代码与本地压测、Dockerfile、manifests、3.6 与 3.7 的论述。
> 镜像体积对比表（3.2）、扩容时间线（3.5）、两个失败场景的复现（3.3）
> 必须等真实 Docker / k8s 环境，拿不到就按第 5 节如实标注未完成，**不编造 kubectl 输出**。

---

## 2. 四条预测（任务书 3.1，写于构建之前，不得修改）

### 预测 1 — 三种 base 的镜像体积

指 `docker images` 报的**未压缩**大小（不是 registry 上的压缩体积，两者能差 2–3 倍，
report 里也会写明用的是哪个口径）。

| 构建策略 | 预测体积 |
|---|---|
| `nvidia/cuda:12.x-devel-ubuntu22.04` + 全量依赖（含 onnxruntime-gpu、torch 之类） | **8.5 GB**（区间 6–11 GB） |
| `nvidia/cuda:12.x-runtime-ubuntu22.04` + 同样依赖 | **3.6 GB**（区间 2.5–5 GB） |
| `python:3.12-slim` + onnxruntime(CPU) | **420 MB**（区间 300–600 MB） |

依据：`devel` 比 `runtime` 多一整套 CUDA 编译工具链（nvcc、各种静态库、头文件），
经验上是 3–4 GB 的差；`python:3.12-slim` 基础层约 130 MB，
onnxruntime CPU wheel + numpy + fastapi/uvicorn + Pillow 合计约 250–300 MB。

**图像解码用 Pillow 而不是 opencv-python**：后者会带进整套 OpenCV（约 90 MB）
以及一串 libGL/libglib 的 apt 依赖，对只做解码的服务是纯浪费。

### 预测 2 — 多阶段构建还能再省多少

**预测：对 `python:3.12-slim` + 纯 wheel 安装这条路线，多阶段构建的收益 < 10%（50 MB 以内），
远小于大多数人的预期。**

理由：多阶段真正省掉的是**构建期才需要、运行期不需要**的东西——编译工具链、源码、构建缓存。
本日的依赖全是预编译 wheel，`pip install` 不需要 gcc，也就没有工具链可丢；
`pip --no-cache-dir` 已经把 pip 缓存挡住了，`--no-install-recommends` + 清 apt 列表
再省掉几十 MB。**在没有编译步骤的镜像里，多阶段主要是心理安慰。**

反过来预测：如果 base 换成 `nvidia/cuda:*-devel`，多阶段（devel 编译 → runtime 运行）
能省掉 4 GB 以上，因为那时确实有一整套工具链要丢。
**多阶段的收益不取决于用不用多阶段，取决于这个镜像里有没有"只在构建期有用"的东西。**

### 预测 3 — HPA 按 CPU 利用率对这个推理服务合适吗（题眼）

**预测：本日这个 ONNX Runtime **CPU** 版服务上它能用、且 3.5 的扩容能跑通；
但它是个巧合，换成 GPU 推理就立刻失效。**

- 本日推理在 CPU 上算，CPU 利用率与实际负载**强相关**，所以 HPA 能正常触发。
- 换成 GPU 推理，主要时间花在 GPU kernel 上，CPU 只做 HTTP 解析与调度，
  **满负载时 CPU 利用率可能只有百分之十几**，HPA 永远不触发。
- 即使在 CPU 推理下，CPU 利用率仍是**滞后且失真**的：它是一段窗口内的平均，
  队列已经堆积几秒了利用率才爬上来；而且利用率封顶在 100%，
  **过载 2 倍和过载 10 倍在这个指标上看起来一样**，它无法表达"排了多少队"。
- 正确指标是**队列深度**或 **p99 延迟**，经 Prometheus Adapter 暴露成 custom metrics。

**可证伪的具体预测**：本日服务会暴露 `inflight` 这个 gauge。
压测时我预计能看到 —— CPU 利用率到 100% 封顶之后，
**继续加压时 `inflight` 仍在线性上升而 CPU 利用率不再变化**。
这就是"CPU 利用率与排队脱节"的直接证据。

### 预测 4 — 一个 Pod 声明 `nvidia.com/gpu: 1` 后，同一张卡还能被别的 Pod 用吗（题眼）

**预测：不能。** 并且：

- `nvidia.com/gpu` 是 **device plugin 提供的扩展资源**，不是 kubelet 原生统计的 CPU/内存。
  节点上有几个可分配单位完全由 plugin 上报，调度器只做整数记账。
- 它**只能整数分配、不能超卖**：不像 CPU 可以 `requests=0.5` 且 limits 可以高于 requests，
  扩展资源**必须写在 `limits` 里，且 `requests` 自动等于 `limits`**，也不允许小数。
- 不开 MPS / MIG 时，一张卡同一时刻只属于一个容器。
- **对 HPA 的含义：副本数上限被物理卡数硬顶死。** HPA 会照样按指标算出期望副本数并去扩，
  但扩出来的 Pod 因为没有空闲 GPU 而停在 `Pending`，
  HPA 的 `desiredReplicas` 与实际 `readyReplicas` 长期不一致。
  **HPA 不会因为"扩不出来"而报错，这是个静默失败。**
- 所以 GPU 服务的弹性不该靠 HPA 扩副本，而该靠单副本内的动态批处理提高单卡吞吐
  （Day 7 已实测），或靠 Cluster Autoscaler 加**节点**。

### 预测 5（扩容时间线，任务书 3.5 的核心产出，一并先猜）

从开始加压到新副本真正承接流量，**预测总滞后 60–90 秒**，分段：

| 段 | 预测 | 依据 |
|---|---|---|
| 指标采集延迟 | 15–30 s | metrics-server 默认 15 s 抓一次，HPA 控制器 15 s 轮询一次 |
| HPA 决策 | 0–15 s | 同上，取决于落在哪个周期 |
| 调度 + 创建 Pod | < 5 s | 单节点 kind，无资源竞争 |
| 拉镜像 | **≈ 0 s** | 镜像已 `kind load`，`imagePullPolicy: IfNotPresent` |
| 容器启动 + 模型加载 | 3–10 s | best.onnx 仅 26 MB，ONNX Runtime CPU 初始化为主 |
| 通过 readiness 开始接流量 | +0–10 s | readinessProbe 的 periodSeconds 决定 |

**预测滞后的大头在第一段（指标采集）而不是启动**，因为模型很小。
这与"模型大的服务滞后主要在加载"正好相反，report 里会写清这个前提。

---

## 3. 本日实测与判断错的地方

写于全部实验结束之后（2026-09-18 22:30 左右，本机时间）。
第 1、2 节一字未改。完整的数据与论证在 `report.md`，这里只记**预测的账**和**我错在哪**。

### 3.1 预测的账

| # | 预测 | 实测 | 判定 |
|---|---|---|---|
| 1 | `python:3.12-slim` + onnxruntime CPU = **420 MB**（区间 300–600） | **434 MB** | ✅ 误差 +3.3% |
| 1 | CUDA runtime 3.6 GB / devel 8.5 GB（未压缩口径） | 只拿到压缩口径 1398 MB / 3922 MB | ⚠️ **未验证** |
| 2 | 多阶段收益 **< 10%（50 MB 以内）** | 447 → 434 MB = **13 MB / 2.9%** | ✅ |
| 3 | HPA 按 CPU 在本日 CPU 推理服务上能用、3.5 能跑通 | 能用：41 s 扩出副本并承接流量，稳态 QPS 0.78 | ✅ |
| 3 | 利用率**封顶在 100%**，"过载 2 倍与 10 倍看起来一样" | **封顶在 400%**，不是 100% | ❌ 数值错，机理对 |
| 3 | 可证伪断言：封顶后 **`inflight` 继续上升而利用率不变** | 并发 1→16：`inflight` 1→21，利用率 390%→398% | ✅ **坐实** |
| 4 | 一张卡不能被两个 Pod 共用；HPA 会静默扩出 `Pending` Pod | 单节点 kind 无 GPU、无 device plugin | ⚠️ **未验证** |
| 5 | 扩容总滞后 **60–90 s**，大头在指标采集段 | 总滞后 **41 s**；指标段 26 s，占 **63%** | ⚠️ 分段对，总量高估约 2 倍 |

预测 1 的 CUDA 两行和预测 4 都拿不到同口径实测，按任务书第 5 节标注未完成，
**没有做换算去冒充实测，也没有伪造任何 `kubectl` 输出。**

### 3.2 判断错的地方

**(1) 我把 `averageUtilization` 的分母搞错了。**
预测里写"利用率封顶在 100%"，这是把它默认成了"占满整机的比例"。
实际定义是 `实际用量 / requests`。本日 `requests.cpu=500m`、`limits.cpu=2`，
所以封顶值是 `2 / 0.5 = 400%`，实测长期停在 390–400%。

这不是一个无害的口误：**HPA 的阈值是按这个量程设的。**
按 60% 设阈值，在 400% 量程里等于"CPU 刚用到 0.3 核就扩容"。
本日实测 CPU 从 1% 跳到 253% 只用了一个采样周期，就是这个原因。
换句话说，**这个阈值的含义完全取决于 `requests` 这个人挑的数字**，
同一份 HPA YAML 挂到不同 `requests` 的 Deployment 上，行为完全不同。

好的一面是：预测里那条**可证伪的具体断言**——"封顶之后 `inflight` 继续线性上升
而 CPU 利用率不再变化"——被实验直接证实了（run F）。
封顶值猜错了，封顶这件事本身和它的后果都猜对了。

**(2) 分段估计对，但每段都取保守值再相加，总量系统性偏大。**
预测 5 我分了五段，每一段实测都落在预测区间里，
但总量预测 60–90 s、实测 41 s。原因是我在每段都取了偏上的值。
模型只有 26 MB，`load_sec` 实测 0.225 s，启动段几乎可以忽略，
而我给了 3–10 s 并按上限计。
**教训：分段估计的误差不是独立的，不能简单按保守值相加。**

**(3) 最值得记的一条：我把"探针配置"当成了独立于应用代码的配置问题。**

任务书 3.3 要复现两个失败场景。场景一（liveness 打 `/readyz` + 不配 startupProbe）
按计划人为制造，完美复现：503 × 21 → 被杀 6 次 → `CrashLoopBackOff`。

场景二我原本准备了 `deployment-bad2.yaml` 去人为制造。
结果**它在完全正确的配置下自己出现了**：三探针分工清楚、liveness 只打极轻量的
`/healthz`，压测时四个 Pod 照样全部被 liveness 杀掉。

根因在应用代码：`/predict` 写成了 `async def`，
FastAPI 对 `async def` 处理函数**直接在事件循环线程里跑**，
而 `onnxruntime.run()` 是同步阻塞的 C++ 调用，一跑 1.5–4 秒，
这段时间整个 uvicorn 事件循环停转。实测同一个 `/healthz`：
**空载 2–12 ms，压测中最高 5378 ms**，慢了约 2700 倍。

> **探针的轻重由整个进程的调度决定，不由这个 handler 自己决定。**
> "liveness 必须极轻量"这条建议我遵守了，Pod 照样被杀。

**(4) 由 (3) 引出的更深一层：我修好了阻塞，故障没消失，只是换了出口。**

单变量改成同步 `def`（FastAPI 丢进线程池）之后：
liveness 失败事件 **0**，`/healthz` 压测中回到 3–90 ms。
但四个 Pod 全部 **`exitCode 137 / OOMKilled`**，成功 27 / 失败 5405，
比没修之前（成功 210 / 失败 45）**差一个数量级**。

原因：`async def` 版本事实上充当了限流器——事件循环被独占，一个 Pod 同时只跑一次推理。
改成线程池后一个 Pod 真的同时跑 8 路，内存从 ~290 MiB 涨到 ~2.3 GiB，
撞穿 `limits.memory: 1Gi`。

事后做的容量标定（run C，读 cgroup v2 的 `memory.peak`）给出：

```
峰值内存(MiB) ≈ 233 + 262 × 并发        （并发 8 预测 2329，实测 2355，误差 1.1%）
1Gi 上限 → 可承受并发 = (1024 − 233) / 262 ≈ 3.0
```

**并发 8 下 OOM 是算得出来的必然，不是意外。**

> **顺序错了：我先配探针和 HPA，最后才做容量标定。**
> 正确顺序是先标定单副本的并发-内存-吞吐曲线，
> 由它定 `limits`、定应用层的并发上限，再由这个上限去配探针超时和 HPA 阈值。
> **容量是因，探针参数是果。** 这一条比本日任何一个数字都值钱。

### 3.3 一个没有预料到的观察

扩容过程中出现了**服务完全不可用的窗口**：T+35 s 时原来那个健康副本
因 readiness 超时被摘出 endpoints，而新副本要到 T+38 s 才进来，
中间约 3 秒 Service 的就绪端点数为 **0**。
压测数据上对应第一个 60 s 窗口的 3856 个失败（其余四个窗口合计只有 1 个）。

**"副本数从 1 涨到 4"这张图上完全看不出中间服务断过。**
同一时刻 HPA 还报了 `FailedGetResourceMetric`（Pod 不 Ready 时没有指标），
但那只是 Warning，不改副本数、不报错、不回滚。

本日一共撞见三个**静默失败**，值得并排记住：

1. `kind load` 的报错被 `tail -1` 吞掉 → metrics-server `ImagePullBackOff` → HPA 显示 `<unknown>` 而不报错；
2. Pod 不 Ready 时 HPA 拿不到指标 → 只发 Warning，副本数不动；
3. （未实测，论述）GPU 节点没有空闲卡时，HPA 照样扩，新 Pod 停在 `Pending`，HPA 不报错。

**共同点：控制器只负责"把期望值写下去"，写不下去不是它的错误路径。**

### 3.4 环境上踩的坑

按耗时排序，第一条花的时间比本日任何一个实验都多：

1. **WSL2 的 VM 空闲自动关机，把 kind 集群一起带走**，表现为 `kubectl` 间歇性
   `connection refused`，看起来像端口映射坏了。修法：`.wslconfig` 里 `vmIdleTimeout=-1`
   加一个常驻 `setsid sleep infinity` 保活；`networkingMode=mirrored` 在本机会引起
   localhost 解析告警，退回 NAT 更稳。
2. **`kind load docker-image` 对本机 buildkit 产物报 `content digest not found`**
   （attestation manifest 缺内容）。绕开 kind 的包装：
   `docker save <img> | docker exec -i <node> ctr --namespace=k8s.io images import --snapshotter=overlayfs -`。
3. **PowerShell → wsl → bash 的嵌套引号**：后台任务打印了 `started` 但根本没起来，
   日志文件不存在。今天犯了三次。修法：把 nohup 与重定向写进 `.sh` 文件，外面只调这个文件。
   同一个 wsl 调用链还会**吃掉 shell 变量**（`for f in ...; echo "== $f =="` 打出 `== ==`），
   修法相同。
4. **阿里云 pip 镜像对个别包回源 pythonhosted 随机超时**，A/B 对照的重建镜像就栽在这里。
   修法：做 A/B 不要重装依赖，用 `FROM <已有镜像>` 只叠一层改动的代码。
5. **第一条扩容时间线作废**，因为建 HPA 之前做的空载延迟测量把 CPU 顶上去了，
   HPA 一创建就先扩到 3，T0 时刻不是 1 副本。修法：建 HPA 前等 CPU 连续 3 次采样
   回落到基线以下，并在 T0 前打印一次 `kubectl get hpa` 存证。

### 3.5 一个没能测干净的量

**扩容的加速比我给不出可信数字。** 需要一个"同镜像、同并发、单副本、无重启"的基线，
而三个候选都不合格：run E（async 单副本）该档有 3219 个失败，Pod 正被 liveness 反复杀；
run C 与 run F 同为 sync 单副本并发 2，却给出 0.24 与 0.53，差 2.2 倍——
差别在每档是否换新 Pod，即 ONNX Runtime 的 arena 是冷是热。

在这个分散度下拿 0.78 去除以其中任何一个都是自欺，所以 report 第 5.4 节不给加速比，
只给**不依赖这个比值**的结论：单 Pod 的 CPU 实测稳定顶在 1950–2000 m（= `limits.cpu: 2`），
节点 8 核，所以扩容的天花板是 4 副本——由**节点核数 ÷ 单 Pod CPU limit** 决定，
而不是由 `maxReplicas` 决定。

**教训：要比较的两个数，得在同一次实验设计里一起测出来**，
事后从不同轮次里各挑一个凑成比值，分散度会大到让比值本身失去意义。
Day 10 的噪声地板做对了这件事，本日在吞吐上没做。
