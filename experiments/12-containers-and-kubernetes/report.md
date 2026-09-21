# Day 12 · 容器化与 Kubernetes 调度

> 预测写于任何构建之前，见 `journal.md` 第 2 节，本报告不回头修改它们。
> 所有数字都有 `results/` 里的原始命令输出作为依据。
> 本日有一节（3.6 GPU 调度）**没有真实 GPU 节点可验证**，按任务书第 5 节如实标注，只给 manifest 与论述。

---

## 0. 本日最值得记住的四件事

1. **两个探针故障不是两个问题，是同一个容量问题的两个出口。**
   把阻塞事件循环的 `async def` 改成同步函数之后，liveness 误杀归零——
   然后四个 Pod 全部 `OOMKilled`。修好"探针被饿死"只是把死法从 CPU 轴换到内存轴。

2. **HPA 的 `averageUtilization` 分母是 `requests`，不是核数，也不是 100% 封顶。**
   实测利用率长期停在 **400%**，因为 `limits.cpu / requests.cpu = 2 / 0.5`。
   我在预测里写的"封顶 100%"是错的，但"封顶后指标失去表达力"是对的——
   并发从 1 涨到 16（16 倍负载），CPU 利用率从 390% 变到 398%，**几乎不动**；
   同一段时间里在途请求数 1 → 21，p50 延迟 1.94 s → 22.36 s，吞吐反而从 0.53 掉到 0.25。

3. **扩容滞后的大头确实在"指标被看见"这一段**（26 s / 41 s = 63%），
   但绝对值比预测快一倍（实测 41 s，预测 60–90 s）。

4. **缩容与 `behavior` 配置精确吻合**：停止加压后 306 s 走第一步，此后每 60 s 一步，
   426 s 回到 1 副本 —— 对应 `stabilizationWindowSeconds: 300` 与 `policies: 1 Pod / 60s`。

---

## 1. 环境判定（任务书第 1 节）

结论与全部探测输出在 `journal.md` 第 1 节与 `results/env_probe.txt`，此处只复述落点：

- **AutoDL 容器内无法运行 dockerd**，且不是"没装"：`capsh --print` 的 bounding set 里没有
  `cap_sys_admin`，cgroup2 是只读挂载，`/dev/fuse` 不存在。
- **rootless 退路同样不通**：`unshare -Ur true` → `Operation not permitted`，
  `/etc/subuid` 为空，user namespace 被禁。**不是缺软件，是缺能力与可写 cgroup。**
- 因此本日的构建与集群全部落在**本机 Windows + WSL2(Ubuntu) + Docker + kind** 上，
  工作目录 `D:\llm-infra-day12`（模型与测试图从 AutoDL 取下）。

WSL2 侧踩到的两件事写在第 8 节，其中一件（VM 空闲自动关机把 kind 集群一起带走）
花掉的时间比本日任何一个实验都多。

---

## 2. 镜像体积对比表（任务书 3.2）

### 2.1 口径声明（这一段必须先写）

任务书要的是"体积"，但 Docker 自己就有**三个互不相等的口径**，本日全都撞上了：

| 口径 | 来源 | 含义 |
|---|---|---|
| `docker images` 的 SIZE | 本地镜像 | 解压后、按层去重前的本地占用 |
| `docker history` 各层求和 | 镜像 manifest | 各层 diff 大小之和 |
| registry manifest 求和 | `docker manifest inspect` | **压缩后**的下载量 |

实测三者不一致，且不是常数偏移：

```
python:3.12-slim      images=190MB    layers_sum=142.1 MB
sod:slim              images=555MB    layers_sum=392.2 MB
sod:B                 images=447MB    layers_sum=337.2 MB
sod:C                 images=434MB    layers_sum=327.1 MB
sod-onnx-cpu:latest   images=434MB    layers_sum=327.2 MB
```

连"省掉多少"这个差值都对不上：`docker images` 说 pip 缓存值 108 MB（555→447），
`docker history` 说那一层只差 55 MB（250 MB→195 MB）。
**报体积必须报口径**，否则两个人拿着同一个镜像会吵起来。
下面主表统一用 `docker images` 口径（也是预测里声明的口径）。

### 2.2 主表（`docker images`，未压缩，原始输出 `results/image_variants.txt`）

| # | 构建策略 | 体积 | 相对上一行省掉的是什么 |
|---|---|---|---|
| 0 | `python:3.12-slim` 基座 | **190 MB** | — |
| A | 单阶段 + `pip install`（保留缓存） | **555 MB** | 最朴素的写法 |
| B | 单阶段 + `pip --no-cache-dir` | **447 MB** | pip 下载的 wheel 缓存（`~/.cache/pip`） |
| C | 多阶段：builder `--prefix=/install` → runtime `COPY` | **434 MB** | builder 层里的 pip 元数据与中间产物 |
| D | 多阶段 + `apt-get install curl` + `RUN chown -R` | **452 MB** | ——**反而多了 18 MB**，见 2.4 |
| ★ | 最终交付：C 的结构 + `COPY --chown` 建非 root 用户，不装 apt | **434 MB** | 与 C 同体积，但跑非 root |

CUDA 两行**拿不到同口径数字**：本机 WSL2 磁盘不足以拉下 `nvidia/cuda:*-devel`，
只取到 registry 压缩体积，**与上表不是一个单位，不能并排比较**
（`results/cuda_manifest_sizes.txt`）：

```
nvidia/cuda:12.4.1-runtime-ubuntu22.04    压缩 1398 MB
nvidia/cuda:12.4.1-devel-ubuntu22.04      压缩 3922 MB
python:3.12-slim                          压缩   44 MB
```

同一个 `python:3.12-slim` 压缩 44 MB / 未压缩 190 MB = **4.3 倍**，
所以把 1398 MB 直接当作"runtime 基座 1.4 GB"是错的，真实未压缩量在 3 GB 量级。
**devel 比 runtime 多 2.5 GB（压缩口径）**，这部分就是 nvcc、静态库、头文件——
运行期一个都用不到，这正是多阶段构建在 CUDA 场景下的价值所在。

### 2.3 层级归因（`docker history`，`results/docker_history.txt`）

```
sod:slim   250MB  pip install --index-url ...        ← 含 wheel 缓存
sod:B      195MB  pip install --no-cache-dir ...     ← 缓存被挡掉
sod:C      185MB  COPY dir:... (从 builder 拷 /install)
```

三个镜像的基座层完全相同（87.5 MB debian rootfs + 13.2 MB apt 依赖 + 41.4 MB python 构建），
**全部差异集中在一层**。找最大的那几层比盲目试有效，这一条任务书说对了。

### 2.4 一个反直觉的结果：把"瘦身技巧"叠满，镜像反而更大

D 变体（多阶段 + 清缓存 + 非 root + apt 瘦身）452 MB > C 的 434 MB。原因两条，都值得记：

1. **`apt-get install curl` 是纯负担。** 装它的理由是"探针要用 curl"——
   这是个误解：`httpGet` 探针由 **kubelet 从节点发起**，根本不进容器，
   容器里不需要任何 HTTP 客户端。这一条装进去约 12 MB（含依赖）。
2. **`RUN chown -R svc:svc /app` 会复制一整层。** overlayfs 里改文件属主 = 写新文件，
   被 chown 过的内容在新层里整份再存一遍。正确写法是 `COPY --chown=svc:svc`，
   在拷贝时就定好属主，**不产生额外层**。

> 教训：瘦身技巧不是越多越好，每一条都要问"它在这个镜像里到底省掉了什么"。
> 在没有编译步骤的纯 wheel 镜像里，多阶段的收益只有 **13 MB / 2.9%**（447→434），
> 这与预测 2 一致；而两个"看起来很专业"的动作净增了 18 MB。

### 2.5 `.dockerignore`

第一次构建的 build context 是 **72.98 MB**——因为 `bin/` 里放着下载来的 `kind` 与 `kubectl`
（合计 69 MB）。补上 `bin/`、`*.sh`、`*.yaml` 之后：

```
Sending build context to Docker daemon  26.11kB
```

**72.98 MB → 26.11 kB，缩小约 2800 倍。** context 不进镜像，但每次构建都要打包发给 daemon，
在迭代 Dockerfile 时这是实打实的等待。模型权重（26 MB）也在排除之列——
模型走**挂载**而不是烤进镜像，否则换模型就得重新构建、重新分发。

---

## 3. 三种探针的分工，与两个失败场景的复现（任务书 3.3）

### 3.1 服务侧的三个端点

| 端点 | 内容 | 给谁用 |
|---|---|---|
| `/healthz` | 只返回常量，不碰模型、不加锁 | startupProbe + livenessProbe |
| `/readyz` | 模型未加载完返回 **503**，加载完返回 200 + 加载耗时 | readinessProbe |
| `/predict` | 真正的推理 | 业务流量 |

模型在**后台线程**里加载，因此 `/healthz` 在加载期间就能应答——这正是 startupProbe 能工作的前提。

### 3.2 失败场景一：liveness 打到 `/readyz`，且不配 startupProbe

`manifests/deployment-bad1.yaml`：`STARTUP_DELAY_SEC=90`，liveness → `/readyz`，
`periodSeconds: 5`、`failureThreshold: 3`（约 20 s 容忍窗口），无 startupProbe。

实测（`results/bad1_events.txt`、`results/bad1_states.txt`）：

```
Warning  Unhealthy  2m30s (x21 over 5m5s)  Liveness probe failed: HTTP probe failed with statuscode: 503
Normal   Killing    2m50s (x6 over 4m55s)  Container sod failed liveness probe, will be restarted
Warning  BackOff    2m29s (x4 over 3m32s)  Back-off restarting failed container sod
```

5 分钟内被杀 **6 次**，重启计数到 **7**，最终 `CrashLoopBackOff`。
容器每次只活约 20 s，而模型需要 90 s——**永远到不了能服务的那一刻**。

一个容易看错的细节：`lastState` 是

```json
{"terminated":{"exitCode":0,"reason":"Completed", ...}}
```

**exitCode 0、reason Completed。** 进程没有崩溃，是 kubelet 发 SIGTERM、uvicorn 优雅退出。
所以"查日志找崩溃原因"会一无所获——**这类问题只能从 Events 看出来**，
这也是任务书第 8 节把 `describe pod` 排在 `logs` 前面的原因。

**修法**：加 startupProbe 打 `/healthz`，`periodSeconds: 5 × failureThreshold: 60` = 300 s 窗口。
startupProbe 未成功之前 liveness 与 readiness **都不生效**，模型加载再慢也不会被误杀。

### 3.3 失败场景二：liveness 在重负载下被饿死 —— 它是自己冒出来的

这一场景**不需要故意做坏配置**。用**正确**的 deployment（三探针分工清楚、liveness 只打 `/healthz`）
做并发 8 的压测（run A），结果：

```
sod-79475c7bc9-6skcj   1/1  Running  3 (3m14s ago)
sod-79475c7bc9-7qbcc   1/1  Running  3 (2m10s ago)
sod-79475c7bc9-mp5q6   1/1  Running  4 (108s ago)
sod-79475c7bc9-shqkt   1/1  Running  1 (2m44s ago)
```

四个 Pod 全部反复重启，事件清一色：

```
Normal  Killing  Container sod failed liveness probe, will be restarted
Warning Unhealthy Liveness probe failed: Get "http://10.244.0.5:8000/healthz": context deadline exceeded
```

**"liveness 必须极轻量"这条建议本身没被违反，Pod 照样被杀。** 直接证据（run D 采样，
`results/runD_healthz_under_load.txt`）——同一个 `/healthz`，同一个容器：

| 状态 | `/healthz` 响应时间 |
|---|---|
| 空载 | 2 – 12 ms |
| 压测中 | 最高 **5378 ms** |

慢了约 **2700 倍**，远超 `timeoutSeconds: 2`，连续三次即被杀。

**根因不在探针配置，在应用代码：**

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ...
    out = M.sess.run(None, {M.input_name: x})[0]     # 同步阻塞调用
```

FastAPI 对 `async def` 处理函数**直接在事件循环线程里执行**。
`onnxruntime.InferenceSession.run()` 是同步阻塞的 C++ 调用，一跑就是 1.5–4 秒，
这段时间**整个 uvicorn 事件循环停转**，任何端点都排不上——包括那个"极轻量"的 `/healthz`。

> 换句话说：**探针的轻重由整个进程的调度决定，不由这个 handler 自己决定。**
> 只要同进程里有任何东西能长时间独占事件循环，"轻量端点"就是纸面上的。

### 3.4 修好之后：故障没有消失，只是换了一个出口

单变量改动（`Dockerfile.patch` 只叠一层 `app.py`，依赖层原封不动）：

```python
@app.post("/predict")
def predict(file: UploadFile = File(...)):      # 去掉 async，FastAPI 丢进线程池
    raw = file.file.read()
```

run B（同样并发 8、同样探针配置、同一批依赖）：

| | run A（`async def`） | run B（同步 `def`） |
|---|---|---|
| `/healthz` 压测中 | 最高 5378 ms | 多数 3–90 ms，偶发 2.1 s |
| liveness 失败事件 | 每个 Pod 3–4 次，全部被杀 | **0** |
| 容器终止原因 | exitCode 0 / Completed（被 SIGTERM） | **exitCode 137 / OOMKilled** |
| 压测结果 | 成功 210 / 失败 45 | **成功 27 / 失败 5405** |

**修好阻塞，服务反而更不可用。** 原因是 `async def` 版本在事实上起了**限流器**的作用：
事件循环被独占意味着一个 Pod 同时只跑一次推理。改成线程池后，一个 Pod 真的同时跑 8 路推理，
内存需求从 ~290 MiB 跳到 ~2.3 GiB，撞上 `limits.memory: 1Gi`，被内核 OOM Killer 杀掉，
重启、再被压垮、再被杀。

> **结论：这两个"失败场景"是同一个容量问题的两个出口。**
> 探针参数与 `async`/`def` 都只决定它从哪个出口出来。
> 正解是第 4 节：**按实测给单副本定一个并发上限，再按这个上限定 requests/limits。**

---

## 4. 集群与工作负载：requests / limits 的实测依据（任务书 3.4）

### 4.1 标定实验（run C，`results/runC_concurrency.txt`）

单副本、无 HPA、`limits.memory` 临时放到 3Gi 以免 OOM 打断测量，
每档换一个新 Pod 使 `memory.peak` 从零开始，读 cgroup v2 的
`/sys/fs/cgroup/memory.peak`：

| 并发 | 成功 | QPS | p50 | **峰值内存** | 常驻内存 |
|---|---|---|---|---|---|
| 1 | 19 | 0.32 | 2.84 s | **495 MiB** | 471 MiB |
| 2 | 13 | 0.24 | 7.21 s | **785 MiB** | 743 MiB |
| 4 | 10 | 0.17 | 21.91 s | **1305 MiB** | 1237 MiB |
| 8 | 20 | 0.33 | 19.93 s | **2355 MiB** | 2319 MiB |

空载（模型已加载、无请求）：**90 MiB**（`results/runC_idle_mem.txt`）。

峰值内存对并发几乎是完美线性：

```
峰值(MiB) ≈ 233 + 262 × 并发
预测 并发=8 → 233 + 262×8 = 2329 MiB，实测 2355 MiB，误差 1.1%
```

- **截距 233 MiB**：进程基线 + ONNX Runtime 的常驻 arena。
  注意它远大于空载的 90 MiB——**第一次推理才会把 arena 撑起来**，
  只看"服务刚起来的内存"会严重低估。
- **斜率 262 MiB / 并发**：每一路在途请求的 1024×1024×3 float32 张量与中间激活。

**直接推出 1Gi 的承受能力**：`(1024 − 233) / 262 ≈ 3.0`。
所以 run B 在并发 8 下 OOM 不是意外，是**算得出来的必然**。

### 4.2 最终取值与依据

```yaml
resources:
  requests: { cpu: "500m", memory: "512Mi" }
  limits:   { cpu: "2",    memory: "1Gi"  }
```

- `requests.memory: 512Mi` —— 覆盖 233 MiB 截距 + 1 路在途（495 MiB 实测），留一点余量。
  这是**调度器用来放 Pod 的数字**，应当对应"稳态正常工作"而不是"最坏峰值"。
- `limits.memory: 1Gi` —— 对应**并发 3** 的上限。它是一道保险，不是目标。
  **如果应用不自己限并发，这个 limit 就会变成一台 OOM 制造机**（run B 实测）。
- `requests.cpu: 500m` —— 见下一条，它的主要作用其实是 HPA 的分母。
- `limits.cpu: "2"` —— 单 Pod 最多吃 2 核，节点 8 核正好放 4 个满载副本，
  与 `maxReplicas: 4` 对齐。

**两个都必须显式写**：不写 `requests` 的 Pod 在调度器眼里是零开销，会被超卖到崩；
不写 `limits` 的 Pod 会把节点上别的 Pod 一起拖死。

**必须补的一条**：按本日实测，正确配置还应包含"应用层限并发"——
一个信号量把同时进入 `sess.run()` 的请求限制在 3 以内，超出的直接返回 429。
没有这一条，`limits.memory` 只能在事后杀进程，杀掉的是**已经算了一半的在途请求**。
本日的 manifest 没有加这个改动（它属于应用逻辑变更，超出任务书范围），
在此**如实标注为已知缺陷**。

### 4.3 `terminationGracePeriodSeconds: 60`

默认 30 s 对这个服务不够：单张图 CPU 推理 p50 就有 2–22 s，
并发高时在途请求可能还要更久。缩容或滚动更新时 30 s 会把在途请求直接切断。

---

## 5. HPA 与扩缩容时间线（任务书 3.5 —— 本日核心产出）

### 5.1 实验设置（run D）

前两次尝试的时间线都不干净，原因记在第 8 节。最终这一轮做了两处控制：

- **并发降到 2**：单副本不会被压到探针饿死，时间线不被重启打断；
- **HPA 在 CPU 回落到基线之后才创建**（连续 3 次 `kubectl top` < 100m），
  保证 T0 时刻确实是 1 副本、`cpu: 0%/60%`。

镜像与 run A 相同（`sod-onnx-cpu:latest`）。
HPA：`minReplicas 1 / maxReplicas 4 / averageUtilization 60`，
`scaleUp.stabilizationWindowSeconds: 0` + `2 Pods / 15s`，
`scaleDown.stabilizationWindowSeconds: 300` + `1 Pod / 60s`。

### 5.2 扩容时间线（T0 = 21:48:49 = 加压开始）

| 时刻 | T+ | 事件 | 来源 |
|---|---|---|---|
| 21:48:49 | 0 s | 开始加压，1 副本，`cpu: 0%/60%` | `runD_hpa_before.txt` |
| 21:49:15 | **+26 s** | **HPA 决策**：`SuccessfulRescale New size: 3; reason: cpu resource utilization above target` | HPA Events |
| 21:49:17 | +28 s | 新 Pod 出现在 API（age=2s），CPU 利用率首次被报为 **253%** | 状态采样 |
| 21:49:24 | +35 s | **原副本掉出 Ready**，就绪端点数 = 0 | 状态采样 |
| 21:49:27 | +38 s | 2/3 Ready，开始接流量 | 状态采样 |
| 21:49:30 | +41 s | 3/3 Ready | 状态采样 |
| 21:50:06 | +77 s | HPA 第二次决策：`New size: 4` | HPA Events |
| 21:50:16 | +87 s | 4/4 Ready | 状态采样 |

**滞后分解（到第一批新副本承接流量为止，共 41 s）：**

| 段 | 实测 | 占比 | 预测 |
|---|---|---|---|
| 指标采集 + HPA 决策 | **26 s** | 63% | 15–45 s ✓ |
| 调度 + 创建 Pod | **2 s** | 5% | < 5 s ✓ |
| 拉镜像 | **0 s** | 0% | ≈ 0 s ✓（已 `kind load`） |
| 容器启动 + 模型加载 | **~8 s** | 20% | 3–10 s ✓（`load_sec: 0.225`，其余是解释器与 import） |
| 通过 readiness | **~3 s** | 7% | 0–10 s ✓ |

**分段判断全部命中，但总量预测偏保守**：预测 60–90 s，实测 41 s。

### 5.3 一个不在预期里的东西：扩容过程中出现了完全不可用窗口

T+35 s 时**原来那个健康的副本掉出了 Ready**，而新副本要到 T+38 s 才进来，
中间有约 3 秒钟 Service 的就绪端点数为 **0**。压测数据上对应第一个 60 s 窗口的
3856 个失败（后面四个窗口合计只有 1 个）。

机理与 3.3 同源：老副本被 CPU 饱和 + 事件循环阻塞拖慢，`readinessProbe`
（`timeoutSeconds: 3`）连续失败被摘出 endpoints。
**"扩容"的第一个可见效果是服务短暂彻底中断**，这在只看"副本数从 1 涨到 4"的图上完全看不见。

同一时间 HPA 自己也报了两条 Warning：

```
FailedGetResourceMetric  failed to get cpu utilization: did not receive metrics
                         for targeted pods (pods might be unready)
```

Pod 不 Ready 时 metrics-server 没有它的数据，HPA 算不出利用率。
**这只是 Warning，HPA 不改副本数、不报错、不回滚**——又一个静默行为。

### 5.4 扩容对吞吐的影响：能看到提升，但单副本基线本身不够稳

压测期间的分窗口吞吐（run D，并发 2，`results/runD_loadtest.jsonl`）：

| 窗口 | 副本数 | QPS |
|---|---|---|
| T+0–60 s | 1 → 3（含 5.3 的中断） | 0.28 |
| T+60–120 s | 4 | 0.53 |
| T+120–180 s | 4 | 0.56 |
| T+180–240 s | 4 | 0.75 |
| T+240–300 s | 4 | **0.78** |

**这里必须承认一个测量上的不足**：我手上没有与 run D**同镜像、同并发、无重启**的单副本基线。
可用的单副本参考点是：

| 来源 | 镜像 | 并发 | QPS | 说明 |
|---|---|---|---|---|
| run C | sync | 2 | 0.24 | 每档换新 Pod，ORT arena 是冷的 |
| run F | sync | 2 | **0.53** | 连续阶梯，Pod 是热的 |
| run E | async | 2 | 0.12 | **不可用**：该档有 3219 个失败，Pod 正被 liveness 反复杀 |

同为 sync、单副本、并发 2 的两次测量相差 **2.2 倍**（0.24 vs 0.53），
差别在 ONNX Runtime 的 arena 是冷是热。**在这个分散度下，
拿 0.78 去除以其中任何一个都得不出可信的加速比**，所以本报告不给"扩容提升了 N 倍"这个数字。

**能确定的是天花板，而且它不依赖上面的比值：**

- 单 Pod 被 `limits.cpu: 2` 限制在 2 核。run E / run F 实测：
  **不论并发 1 还是 16，单 Pod 的 CPU 都稳定在 1950–2000 m**，即牢牢顶在自己的 limit 上。
- 节点 8 核，因此最多 4 个副本能同时满载 → `maxReplicas: 4` 正是 8 核 / 2 核。
- 再往上扩只会在同一批核上互抢。run A（并发 8、4 副本）QPS 只有 0.78，
  与 run D（并发 2、4 副本）相同，就是这个天花板的表现：
  **负载翻了 4 倍，吞吐一点没动。**

> 换句话说：本日扩容的收益上限由**节点核数除以单 Pod 的 CPU limit**决定，
> 而不是由 HPA 的 `maxReplicas` 决定。把 `maxReplicas` 调到 8 不会有任何好处。

### 5.5 缩容（停止加压 = 21:53:48）

| 时刻 | 停压后 | 事件 |
|---|---|---|
| 21:58:54 | **+306 s** | `New size: 3; reason: All metrics below target` |
| 21:59:54 | **+366 s** | `New size: 2` |
| 22:00:54 | **+426 s** | `New size: 1` |

与配置**逐条吻合**：
`scaleDown.stabilizationWindowSeconds: 300` → 第一步在 300 s 之后（实测 306 s）；
`policies: [{type: Pods, value: 1, periodSeconds: 60}]` → 此后每 60 s 一步（实测 60 s、60 s）。

**缩容比扩容慢一个数量级（426 s vs 41 s），而且这是设计意图**：
稳定窗口取的是窗口内**最高**的推荐值，用来防止流量抖动引发反复扩缩。
代价是流量退潮后要多付 7 分钟的副本钱。
`scaleUp` 那边我特意设了 `stabilizationWindowSeconds: 0`——
**扩容宁可快，缩容宁可慢**，这个不对称是故意的。

---

## 6. GPU 资源声明与调度（任务书 3.6）

> **未在真实 GPU 节点上验证。** 本日集群是单节点 kind，节点上没有 GPU，
> 也没有 nvidia device plugin。以下是 manifest（`manifests/deployment-gpu.yaml`）
> 与论述，**没有任何 `kubectl` 输出被伪造**。

```yaml
resources:
  limits:
    nvidia.com/gpu: 1        # 只能写在 limits，requests 自动等于 limits
```

**`nvidia.com/gpu` 与 CPU/内存在调度上的四点不同：**

1. **来源不同。** CPU/内存由 kubelet 自己统计上报；`nvidia.com/gpu` 是
   **device plugin 通过 kubelet 的 device plugin API 上报的扩展资源**，
   节点上"有几个可分配单位"完全由 plugin 说了算。没装 plugin 的节点上，
   这个资源就是 0，Pod 永远 `Pending`。
2. **不能超卖，只能整数分配。** CPU 可以 `requests: 500m` 而 `limits: 2`（超卖 4 倍），
   扩展资源**必须 `requests == limits`，且必须是整数**，
   `nvidia.com/gpu: 0.5` 直接被 API server 拒绝。
3. **不启用 MPS / MIG 时，一张卡同一时刻只属于一个容器。** 设备被整块以
   `/dev/nvidiaN` 的形式注入容器，没有时间片共享的调度层。
4. **调度器只做整数记账，不理解"显存够不够"。** 声明 1 张卡就独占整卡，
   哪怕模型只用 2 GB 显存，剩下 22 GB 也不会分给别人。

**对 HPA 的含义（题眼）：**

- **副本数上限被物理卡数硬顶死。** HPA 照样会按指标算出 `desiredReplicas` 并去扩，
  但新 Pod 因为没有空闲 GPU 而停在 `Pending`。
- **HPA 不会因为"扩不出来"而报错。** 它的职责到"改 Deployment 的 replicas"为止，
  调度失败是调度器的事。表现是 `desiredReplicas` 与 `readyReplicas` 长期不一致，
  没有任何红色告警——**又一个静默失败**，和 5.3 里的 `FailedGetResourceMetric` 同类。
- 因此 **GPU 服务的弹性不该靠 HPA 扩副本**，而该靠：
  (a) 单副本内的**动态批处理**提高单卡吞吐（Day 7 已实测），
  (b) **Cluster Autoscaler 加节点**（加的是卡，不是副本）。

---

## 7. 论述：CPU 利用率是推理服务的错误指标（任务书 3.7）

第 5 节用 CPU 利用率把扩容跑通了。这一节论证它为什么仍然是**错的指标**——
并且本日拿到了直接的实验证据，不是纯论述。

### 7.1 实验：单副本、无 HPA、阶梯加压，同时采排队深度与 CPU

run F（`results/runF_inflight_vs_cpu.txt`、`results/runF_throughput.txt`）：
单副本、`requests.cpu 500m` / `limits.cpu 2`，并发 1→2→4→8→16 每档 60 s，
每 5 s 同时采 `/metrics` 的 `sod_inflight_requests` 与 `kubectl top` 的 CPU。

| 并发 | 在途请求 `inflight` | CPU | **利用率（相对 requests）** | QPS | p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | ~1950 m | **390 %** | 0.48 | 1.94 s |
| 2 | 2 | 1590–1940 m | **318–388 %** | 0.53 | 3.33 s |
| 4 | 4 | 2000 m | **400 %** | 0.27 | 10.31 s |
| 8 | 8 | 2000 m | **400 %** | 0.25 | 22.36 s |
| 16 | **21** | ~1990 m | **398 %** | 0.00 | 60 s 内无一完成 |

**从并发 1 到并发 16，负载涨 16 倍：**

- CPU 利用率：390% → 398%，**变化 2%，在噪声里**
- 在途请求数：1 → 21，**涨 21 倍**
- p50 延迟：1.94 s → 22.36 s（并发 8），**涨 11 倍**
- 吞吐：0.48 → 0.25，**反而掉了一半**

**HPA 在整个过程中看到的是一个常数。** 服务从"健康"退化到"完全不可用"，
它的输入信号没有任何变化。

### 7.2 三条具体理由

**(1) 指标在 `limits/requests` 处封顶，而不是在 100% 处。**
这是我预测错的地方（见第 9 节）。真正的封顶值是
`limits.cpu / requests.cpu = 2 / 0.5 = 400%`。
一旦 Pod 打满自己的 CPU limit，利用率就钉死，**此后所有的过载全部体现在队列里，指标上看不见**。
更糟的是这个封顶值是**人为设定的**：把 `requests.cpu` 从 500m 改成 2，
同样的负载会读成 100% 而不是 400%，同一份 HPA 配置会做出完全不同的决策。
**`averageUtilization` 的分母是一个人挑的数字，不是物理量。**

**(2) GPU 推理时 CPU 根本不忙。**
本日是 ONNX Runtime **CPU** 版，CPU 利用率与负载强相关，纯属巧合。
GPU 推理时主要时间在 GPU kernel 上，CPU 只做 HTTP 解析、预处理与调度，
满负载时 CPU 利用率可能只有百分之十几——**HPA 永远不触发**。
本日无 GPU 节点，这一条未实测，只作论述。

**(3) 它是滞后的平均值。**
第 5 节实测：队列已经堆了 26 秒，指标才第一次报出越界。
对 p50 就有 2–22 秒的服务来说，26 秒意味着几十个请求已经在排队了。

### 7.3 该用什么

- **队列深度**（本日的 `sod_inflight_requests`）：7.1 里它是唯一一个**全程单调跟随负载**的量。
- **p99 延迟**：直接对应 SLO，而且在 CPU 封顶之后仍然继续增长。
- 两者都不是 kubelet 原生指标，要经 **Prometheus Adapter** 注册成 custom / external metrics，
  HPA 用 `type: Pods` 或 `type: External` 引用。

一个务实的组合：**用队列深度扩容、用 p99 做告警、把 CPU 利用率留给缩容**
（缩容时"CPU 掉下来"是可靠信号，因为那时没有封顶问题）。

---

## 8. 踩到的坑（都会静默出错）

**1. `kind load docker-image` 报错被 `tail -1` 吞掉，metrics-server 静默 ImagePullBackOff。**
kind 内部调用 `ctr images import --all-platforms --digests`，对本机 buildkit/docker 产物
会报 `content digest sha256:... not found`（attestation manifest 缺内容）。
绕开 kind 的包装直接导入就好：

```bash
docker save <image> | docker exec --privileged -i <cluster>-control-plane \
  ctr --namespace=k8s.io images import --snapshotter=overlayfs -
```

`kubectl top` 因此一直不可用，HPA 显示 `<unknown>` 且**不报错**——正是任务书第 4 节第 3 条。
排查顺序有用：`describe pod` → `ImagePullBackOff` → `crictl images` 确认节点上没有这个镜像。

**2. WSL2 的 VM 空闲自动关机，把 kind 集群一起带走。**
表现是 `kubectl` 间歇性 `connection refused to 127.0.0.1:<port>`，像是端口映射坏了。
真实原因：每个一次性的 `wsl -- bash -lc` 退出后 VM 判定空闲并关机。
修法：`%USERPROFILE%\.wslconfig` 里 `vmIdleTimeout=-1`，外加一个常驻的
`setsid sleep infinity` 保活进程。**这一条花的时间比本日任何一个实验都多。**
另外 `networkingMode=mirrored` 在本机会导致 localhost 解析警告，退回 NAT 更稳。

**3. 后台任务的嵌套引号。**
`powershell -Command "... wsl ... bash -lc \"... nohup ... &\""` 打印了 `started`
但脚本根本没起来，日志文件不存在。**修法：把 nohup/重定向写进一个 `.sh` 文件，
外面只 `wsl -- bash /path/launch.sh`。** 同类问题今天出现过三次。

**4. `wsl -- bash -lc '...$var...'` 里的 shell 变量会被吃掉。**
`for f in a b; do echo "== $f ==" ; done` 打出的是 `== ==`。
**修法同上：写成脚本文件。** 排查时容易误判成"文件是空的"。

**5. 阿里云 pip 镜像对个别包回源 pythonhosted，构建随机超时。**
A/B 对照的第一次重建镜像就栽在这里，`sod-onnx-cpu:sync` 根本没生成，
`kubectl set image` 指向不存在的镜像 → 新 Pod `ImagePullBackOff`。
**修法**：做 A/B 时不要重装依赖，用 `FROM <已有镜像>` 只叠一层改动的代码。

> 这次失败顺带演示了一件好事：滚动更新期间新副本始终没通过 readiness，
> **Service 就从未把流量转给它们**，压测全程没有因此失败。
> `maxUnavailable` 的默认值在这里救了场——这正是 readiness 探针存在的意义。

**6. 第一次跑出来的扩容时间线不可用，因为起点不是 1 副本。**
创建 HPA 之前做的空载延迟测量把 CPU 顶上去了，HPA 一创建就先扩到 3。
**修法**：建 HPA 前等 CPU 连续 3 次采样回落到基线以下，并在 T0 前打印一次 `kubectl get hpa` 存证。

---

## 9. 预测 vs 实测（对照 `journal.md` 第 2 节，预测不回改）

| # | 预测 | 实测 | 判定 |
|---|---|---|---|
| 1 | `python:3.12-slim` + onnxruntime CPU = **420 MB**（区间 300–600） | **434 MB** | ✅ 误差 +3.3% |
| 1 | CUDA runtime 3.6 GB / devel 8.5 GB（未压缩） | 只拿到压缩口径 1398 / 3922 MB | ⚠️ **未验证**，口径不同不能比 |
| 2 | 多阶段收益 **< 10%（50 MB 以内）** | 447 → 434 MB = **13 MB / 2.9%** | ✅ |
| 3 | HPA 按 CPU 能用、扩容能跑通 | 能用：41 s 扩出副本并承接流量，稳态 QPS 0.78（加速比因单副本基线不稳未给出，见 5.4） | ✅ |
| 3 | "利用率**封顶在 100%**，过载 2 倍与 10 倍看起来一样" | **封顶在 400%**（= limits/requests），不是 100% | ❌ **数值错**，机理对 |
| 3 | 可证伪断言：封顶后 **inflight 继续上升而利用率不变** | 并发 1→16：inflight 1→21，利用率 390%→398% | ✅ **坐实** |
| 4 | 一张卡不能被两个 Pod 共用；HPA 会静默扩出 `Pending` Pod | 无 GPU 节点 | ⚠️ **未验证**，只有论述 |
| 5 | 扩容总滞后 **60–90 s**，大头在指标采集 | 总滞后 **41 s**，指标段 26 s 占 **63%** | ⚠️ 分段对，**总量高估 ~2 倍** |

### 判断错在哪里（三条）

**第一条：把"CPU 利用率封顶 100%"当成常识写进了预测。**
`averageUtilization` 的定义是 `实际用量 / requests`，不是 `实际用量 / 可用核数`。
只要 `limits > requests`（本日 4 倍），利用率就能超过 100%，封顶值是 `limits/requests`。
**我把"利用率"默认成了"占满整机的比例"**，这个错误会直接导致 HPA 阈值设错：
按 60% 设阈值，在 400% 封顶的量程里意味着"CPU 刚用到 0.3 核就扩容"，实际上过于敏感。
本日 CPU 从 1% 跳到 253% 只用了一个采样周期，就是这个原因。

**第二条：预测扩容滞后时高估了指标采集段。**
我按"metrics-server 15 s + HPA 15 s + 窗口"估了 15–45 s，实测 26 s 落在区间内，
但我又在总量上加了启动与 readiness 的保守值，得出 60–90 s。
实际上模型只有 26 MB、`load_sec: 0.225`，启动段几乎可以忽略。
**教训：分段估计正确，但把每段都取保守值再相加，总量会系统性偏大。**

**第三条（方法层面，最值得记）：我把"探针配置"当成了一个独立于应用代码的配置问题。**
3.3 的第二个场景我原本打算靠 `deployment-bad2.yaml` 人为制造，
结果它在**正确配置**下自己出现了，根因在 `async def`。
**探针的行为由整个进程的调度决定，不由 YAML 决定。**
更进一步，修好它之后故障换成了 OOMKilled——
说明我一开始就该先做第 4 节的容量标定，再去配探针和 HPA，
而不是反过来。**容量是因，探针参数是果。**

---

## 10. 未完成 / 如实标注

1. **CUDA 两行镜像体积只有压缩口径**，未拉取到本地（WSL2 磁盘不足）。
   与主表不是同一单位，已在 2.2 明确标注，未做换算冒充实测。
2. **3.6 GPU 调度未在真实 GPU 节点上验证**：单节点 kind 无 GPU、无 device plugin。
   manifest 与论述已给出，**没有伪造任何 `kubectl` 输出**。
3. **应用层限并发未实现**（见 4.2）。这是本日实测暴露出来的正确修法，
   但属于应用逻辑变更，本日只写进报告，没有改进 manifest 与代码。
4. **run A / run B 的扩容时间线不作为 3.5 的产出**，因为起点不是 1 副本、
   且被探针重启打断。它们作为 3.3 的证据保留在 `results/runA_*`、`results/runB_*`。

---

## 11. 交付物索引

```
experiments/12-containers-and-kubernetes/
├── report.md                     本文
├── journal.md                    预测（第 2 节，未改）+ 实测与判断错的地方（第 3 节）
├── manifests/  configmap.yaml deployment.yaml service.yaml hpa.yaml
│              deployment-gpu.yaml deployment-bad1.yaml deployment-bad2.yaml
├── scripts/    app.py Dockerfile Dockerfile.slim Dockerfile.patch
│              Dockerfile.cuda-runtime Dockerfile.cuda-devel .dockerignore
│              requirements.txt requirements.lock.txt kind-config.yaml
│              loadtest.py watch_k8s.sh analyze12.py
│              full_run.sh runB2.sh runC.sh runD.sh runE.sh runF.sh
└── results/
    ├── env_probe.txt                     第 1 节环境判定原始输出
    ├── image_variants.txt image_sizes.txt docker_history.txt
    │   cuda_manifest_sizes.txt build_context.txt   第 2 节
    ├── bad1_events.txt bad1_states.txt bad1_laststate.txt   3.2 失败场景一
    ├── runA_* runB_*                     3.3 失败场景二与修复后的对照
    ├── runC_concurrency.txt runC_idle_mem.txt   第 4 节容量标定
    ├── runD_* scale_events_abs.txt scaledown.txt   第 5 节时间线
    ├── runE_inflight_vs_cpu.txt runF_inflight_vs_cpu.txt
    │   runF_throughput.txt               第 7 节指标论证
    └── timeline_analysis.txt             analyze12.py 的汇总输出
```
