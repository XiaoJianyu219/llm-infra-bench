# Day 11 · 容错与断点续训：完整 checkpoint 能逐位续上，而"卡挂了"有两种相差三个数量级的故障

三条结论，每条都有实测：

1. **断点续训可以做到逐位一致。** 完整 checkpoint 恢复后的 loss 序列与不中断运行的差是 **0.0**；
   只存"权重 + 优化器"时，恢复后第一步就差 **2.91**。但前提是先把训练本身变成可复现的——
   默认设置下连跑两次都不一致（Flash Attention 的 backward 非确定），开严格确定模式后才一致，代价 **+2.4%** 步时。
2. **"掉一张卡"有两种完全不同的故障。** 进程直接死掉（kill -9）时，torchrun 在 **0.47 秒**内把整组拉掉；
   进程还活着但不响应（SIGSTOP）时，另一个 rank 会一直卡在 all-reduce 里，等满集合超时（默认 **600 秒**）
   再加约 **92 秒**才退出。
3. **最优保存间隔在单机量级是一千步上下，千卡量级缩到几分钟。**
   实测 δ = 5.92 s，MTBF 一天时 τ_opt = 1,012 s、开销 1.17%；外推到 16,384 卡（70B 模型）τ_opt = 8.3 分钟、开销 4.49%。

---

## 1. 起点

在 Day 10 的最优单卡配置上做（`micro_bsz=4, grad_accum=8`），但**去掉了 `torch.compile`**：
本日的核心判据是逐位对齐，compile 会引入另一层非确定来源，与"checkpoint 是否完整"混在一起就分不清了。
本日配置单步 1.274 s（严格确定模式），Day 10 带 compile 的最优配置是 0.788 s，第 5 节两者都代入。

数据改成**有结构的合成语料**（64 个 16-token 的 motif 拼接，固定 seed），不用 Day 08–10 的纯随机 token：
纯随机 token 的 loss 恒在 ln(151936)=11.93 附近不动，"数据顺序变了"不会在 loss 上留下痕迹，判据就失效了。
sampler 每个 epoch 的排列只由 (seed, epoch) 决定，与全局 RNG 无关；模型打开 `attention_dropout=0.1`，
让全局 RNG 真正进入 forward。这样"数据位置"与"RNG"可以分别失效、分别观察。`num_workers=0`。

## 2. 状态清单与体积

| 状态 | 是否保存（c2） | 实测体积 | 说明 |
|---|---|---|---|
| 模型权重（fp32） | ✓ | **2.38 GB（33.3%）** | 596,049,920 × 4 B；tied embedding 在 state_dict 里出现两次，按 data_ptr 去重 |
| 优化器状态（AdamW m、v、step） | ✓ | **4.77 GB（66.7%）** | m、v 各 4N |
| 数据加载位置（epoch、已消费样本数） | ✓ | c1 − c0 = **128 B** | |
| global step / 已消费 token 数 | ✓ | 同上 | |
| RNG：python / numpy / torch / cuda | ✓ | c2 − c1 ≈ **14 KB** | 含 scheduler 与配置快照 |
| LR scheduler | ✓ | 同上 | |
| 配置快照 | ✓ | 同上 | |
| GradScaler | — | — | bf16 训练不需要 |
| **每个 rank 各自的 RNG** | ✓（修复后） | 每 rank 约 10 KB 旁路文件 | 2.4 实测：只存 rank 0 的 RNG，恢复后 rank 1 对不上 |
| 决定性开关（`use_deterministic_algorithms`、`CUBLAS_WORKSPACE_CONFIG`） | 随配置快照 | — | 不是"状态"，但决定第 3 节的判据能不能用 |

完整 checkpoint **7.15 GB**，其中 **99.9998%** 是权重和优化器状态，其余所有"小状态"合计约 14 KB。
**"checkpoint 太大"的答案是优化器，不是模型**：它是模型本身的两倍。

约束：只在梯度累积边界保存。`global_step` 只记优化器步，记不出累积到第几个 micro-step，
中途保存会丢已累积但未 step 的梯度。

## 3. 前置：不中断连跑两次是否一致（任务书 3.5）

| 模式 | 两次运行的 loss | 最大绝对差 | 步时 |
|---|---|---|---|
| 默认 | 第 0 步相同，第 1 步起分叉 | **2.962e-3** | 1256.3 ms |
| `use_deterministic_algorithms(True, warn_only=True)` | 仍不一致 | 1.02e-2 | +0.4% |
| **`use_deterministic_algorithms(True, warn_only=False)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`** | **逐位一致** | **0.0** | 1286.1 ms（**+2.4%**） |

非确定来源由 PyTorch 自己的告警指出：
`Flash Attention defaults to a non-deterministic algorithm. To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False)`。
`warn_only=True` 只会打告警、不会切换实现，所以第二行仍然不一致。

**所以 2.2 的判据可以用严格的逐位一致，而不必退到容许误差**；
若不开确定模式，判据必须放宽到 3e-3 量级，而那会掩盖掉第 4 节里 RNG 缺失造成的 4e-3 量级差异。

## 4. 逐位对齐：先做残缺版本（任务书 2.2）

A 组：固定 seed 连跑 16 步。B 组：跑 8 步 → 存 → **退出进程** → 新进程加载 → 再跑 8 步。
比较 A[8:16] 与 B 的 loss，三档 checkpoint 完整度，全部在严格确定模式下：

| 版本 | 存了什么 | 首步差 | 最大差 | 平均差 | 逐位一致 | 对应症状 |
|---|---|---|---|---|---|---|
| **c0** | 权重 + 优化器 | **2.911** | 3.149 | 1.810 | 否 | 第一步就差很多 + 整体偏移 |
| **c1** | ＋数据位置 + global step | **4.12e-2** | 1.165 | 0.397 | 否 | 首步小，之后越漂越大 |
| **c2** | ＋RNG + scheduler + 配置 | **0.0** | **0.0** | **0.0** | **是** | 无 |

逐步差（前 6 步）：

```
c0  2.911  3.149  2.461  1.438  1.397  1.200
c1  0.041  0.036  0.001  0.179  0.426  0.533
c2  0      0      0      0      0      0
```

读法：

- **c0 首步就差 2.9**：sampler 从 epoch 0 位置 0 重来，恢复后喂的是前 8 步已经训过的 batch。
- **c1 首步只差 0.04（只剩 dropout 不同），但后面越来越大**：c1 没有恢复 scheduler，
  LR 预热从头开始，恢复后的更新幅度比 A 组小，偏差随步数累积。这对应症状表里没有的一行——
  "首步很小、之后单调变大"是 **LR 没恢复**，而不是"只有前几步差"。
- **c2 全部为 0。**

一个没拆开的地方：c1 → c2 同时补了 RNG 和 scheduler 两项，本日无法把 c1 的漂移在两者之间分账。
首步 0.04 那一份只可能来自 RNG（首步 forward 还没受 LR 影响），后面的增长主要来自 LR，但没有单独实验证实。

## 5. 保存开销与最优间隔（任务书 2.3）

### 5.1 单次写入

同步写（`torch.cuda.synchronize` 后 `torch.save`），5 次重复：
**δ = 5.923 s 中位，范围 5.776–7.217 s，7.15 GB，折合单写者 1.21 GB/s。**

**异步写本日没有实测。** 我原想用"组装 state_dict 的耗时"代表异步写的阻塞时间，测出来是 0.004 s，
但那个数没有意义：`state_dict()` 只返回张量的引用，GPU→CPU 的拷贝发生在 `torch.save` 内部。
真正的异步快照需要先显式拷到 pinned 内存，按 Day 09 实测的 D2H 26.3 GB/s 推算约 **0.27 s**（推算，非实测）。

### 5.2 保存越频繁，单次保存越慢

| 间隔 k 步 | 实测单次 δ（均值） | 若 δ 为常数，预测开销 | 实测开销 |
|---|---|---|---|
| 2 | **14.52 s** | 231% | **539%** |
| 4 | **17.88 s** | 116% | **333%** |
| 8 | 7.67 s | 58% | 72% |
| 16 | 6.34 s | 29.5% | 30.1% |

每 2–4 步存一次时，每 3–5 秒要写 7 GB，磁盘的回写跟不上，单次 δ 被拉长到 **2.5–3 倍**（最长一次 20.49 s）。
**Young/Daly 假设 δ 为常数，这个假设在保存间隔短于"写完一次所需时间"的几倍时不成立。**
间隔 16 步（约 20 s）时 δ 已回到孤立测量值，与公式吻合到 0.6 个百分点。
最优间隔都在几百秒以上，远离这个区间，所以下面的代入仍然有效。

> 任务书原定的扫描是 10 / 50 / 200 / 1000 步。第一次按这个跑时每个配置只跑了 ≤ 40 步，
> 50 步以上一次都没存，结果作废（保留在 `results/runs/invalid_first_sweep/`）。
> 按 1.27 s/步，跑满 1000 步间隔要 21 分钟一档，改为实测 2/4/8/16 步、大间隔用公式外推。

### 5.3 Young/Daly 代入

推导见 `experiments/10-communication-tuning/journal.md` 第 3 节：写出完成 τ 单位有用计算的期望墙钟时间
`T(τ) = τ + δ + (τ+δ)/M · (τ/2 + R)`，一阶近似下开销率 `H(τ) ≈ δ/τ + τ/(2M)`，
对 τ 求导得 **τ_opt = √(2δM)**，代回得**最小开销 √(2δ/M)**，此时保存开销与重算损失各占一半。
Daly 高阶修正 `τ = √(2δM)[1 + √(δ/2M)/3 + (δ/2M)/9] − δ`。

代入实测 δ = 5.923 s：

| MTBF M | τ_opt（一阶） | τ_opt（Daly） | 本日配置 | Day10 最优配置 | 最小开销 |
|---|---|---|---|---|---|
| 1 小时 | 207 s | 203 s | 162 步 | 262 步 | 5.74% |
| 1 天 | 1,012 s | 1,008 s | **794 步** | **1,284 步** | **1.17%** |
| 1 周 | 2,677 s | 2,673 s | 2,101 步 | 3,398 步 | 0.44% |

δ/M 很小，Daly 修正只差 4 秒，一阶公式够用。
**单机量级（MTBF 以天计）最优间隔约一千步、开销约 1%**；
任务书扫描表里 10 / 50 / 200 步那三档都是过度保存，每 10 步存一次（本日配置）会让吞吐降到约 **68%**。

## 6. 掉卡模拟（任务书 2.4，双卡）

先读真实接口：`_get_default_timeout("nccl")` = **0:10:00**；`TORCH_NCCL_ASYNC_ERROR_HANDLING` 默认为 1。
**`TORCH_NCCL_TIMEOUT_MS` 这个环境变量不存在**——`libtorch_cuda.so` 里读取的 24 个 `TORCH_NCCL_*`
变量中没有它（清单见 `results/logs/`），集合超时只能通过 `init_process_group(timeout=...)` 设置。
我在预测和第一版测试里都用了这个不存在的变量，第一版超时测试因此作废（见 journal）。

### 6.1 kill -9：进程消失 → 0.47 秒内整组退出

```
09:57:39.306  rank1 第 4 步 kill -9
09:57:39.746  torchrun 检测到 local_rank 1 exitcode -9                    +0.44 s
              对 rank0 发 SIGTERM，rank0 exitcode -15
```

**剩下的 rank 没有卡在 all-reduce 上。** torchrun 的 elastic agent 监控子进程，
本地进程一死就整组拉掉，NCCL 的超时机制根本没被触发。
`TORCH_NCCL_ASYNC_ERROR_HANDLING=0` 下重复一次，时间线完全相同——这个开关在这种故障下不起作用。

### 6.2 SIGSTOP：进程还活着但不响应 → 等满超时

把 rank1 冻结（进程仍存在，agent 看不出异常），rank0 在下一次 all-reduce 上阻塞：

| 事件 | timeout = 30 s | timeout = 60 s |
|---|---|---|
| rank1 冻结 | 10:13:24.692 | 10:15:42.289 |
| rank0 watchdog 报超时（`ALLREDUCE ran for …`） | **+30.2 s**（30,074 ms） | **+60.2 s**（60,040 ms） |
| 开始 dump 调试信息 | +31.1 s | +60.8 s |
| watchdog 线程抛异常终止 | +92.3 s | +121.2 s |
| agent 检测到 rank0 SIGABRT（-6），SIGKILL rank1 | **+121.8 s** | **+151.9 s** |

超时由 `timeout` 参数精确控制（误差 < 0.1 s），但之后有一段约 **92 s 的固定尾巴**：
约 61 s 用于调试信息 dump 与 watchdog 退出，约 30 s 是 agent 发现进程已 abort。本日没有逐一验证尾巴里各段分别由哪个变量控制。

**默认配置下（600 s），一次"对端假死"约 11.5 分钟才会被发现（推算：600 + 92）。**
第一版测试用了不存在的环境变量、实际超时仍是 600 s，rank0 在 243 s 内没有任何报错，直到被外层 `timeout` 杀掉，与此一致。

### 6.3 弹性重启：torchrun --max-restarts=2 + 自动从最近 checkpoint 恢复

每 2 步存一次 c2 checkpoint，第 5 步 kill rank1。恢复逻辑是脚本自己的（`--auto-resume` 挑目录里最新的 ckpt），
torchrun 只负责把进程重新拉起来，它不知道训练状态。

**第一次（只由 rank0 存 RNG）**：

```
10:10:10.778  step 4 完成；此前 step 2、4 各存过一次
10:10:10.779  rank1 kill -9
10:10:11.249  agent 检测到                                                +0.47 s
10:10:17.638  重启，auto-resume 选中 elastic/ck.pt                       +6.9 s
10:10:21.298  恢复 level=c2 step=4
10:10:23.159  重跑 step 4：rank0 loss 10.933997（与掉卡前一致）          +12.4 s
                           rank1 loss 11.271402（掉卡前 11.316912，不一致）
```

checkpoint 只由 rank0 写，**只装了 rank0 的 RNG**；rank1 恢复时拿到的是 rank0 的 dropout 序列。

**第二次（每个 rank 另存一份 RNG 旁路文件，严格确定模式）**：

```
10:20:02.824  rank1 kill -9
10:20:03.208  agent 检测到                                                +0.38 s
10:21:10.265  第一次重启失败：rank1 exitcode 1，挂在 DDP 构造时的
              _verify_params_across_processes（一个集合操作）            +67.4 s
10:21:16.725  第二次重启，auto-resume 选中 elastic_v2/ck.pt
10:21:20.495  恢复 c2 + 本 rank RNG 旁路
10:21:22.145  重跑 step 4：rank0 10.926353 = 10.926353，rank1 11.312866 = 11.312866   +79.3 s
```

**两个 rank 都逐位对齐，2.2 的判据在弹性恢复后仍然成立。**
但这次 `--max-restarts=2` 的余量用掉了一半：第一次重启失败在 DDP 构造的参数校验上，约 67 s 后才退出。
**`--max-restarts` 至少要留 2 次。**

## 7. 千卡规模下这套机制在哪会失效（任务书 2.5）

以下外推中，**本仓库实测**的只有：δ = 5.923 s、单写者 1.21 GB/s、δ 的抖动、D2H 26.3 GB/s（Day 09）、
小消息 all-reduce 延迟 0.110 ms（Day 10）。**外部假设**：
单卡 MTBF 取 Llama 3 技术报告的量级（54 天 419 次非计划中断 @16,384 卡 → 单卡约 50,600 h）；
共享存储聚合写带宽 100 GB/s；模型 70B，按 16 字节/参数，一次 checkpoint 1.12 TB。

### 7.1 故障率随规模放大

| 卡数 N | 集群 MTBF | 单次 δ（70B） | τ_opt | 最小开销 |
|---|---|---|---|---|
| 8 | 264 天 | 115.9 s | 20.2 h | 0.32% |
| 64 | 33 天 | 14.5 s | 2.5 h | 0.32% |
| 1,024 | 2.1 天 | 11.2 s | 33 min | 1.12% |
| 16,384 | 3.1 h | 11.2 s | **8.3 min** | **4.49%** |
| 100,000 | 30 min | 11.2 s | **3.4 min** | **11.08%** |

开销按 √(1/M) 增长，即 √N。**最小开销到 10% 的临界规模约 8.1 万卡**；
那时 τ_opt 只有 3 分钟多，而 5.2 节已经实测到：保存间隔短到写一次所需时间的几倍时，δ 本身会被拉长 2.5–3 倍，
实际开销会比公式更差。

### 7.2 存储带宽饱和

每个 rank 并行写自己的分片时，δ = 总量 / (N × 1.21 GB/s)，随 N 下降；
共享存储 100 GB/s 在 **N ≈ 83 个并发写者**时饱和，此后 δ 固定在 1.12 TB / 100 GB/s = 11.2 s，
**再加卡也不会让保存变快，只会让故障更频繁**。上表 1,024 卡以后 δ 不再变化就是这个原因。

### 7.3 全局同步 checkpoint 的代价

所有 rank 必须停在同一步才能保存。同步本身不贵：Day 10 实测 1 MB all-reduce 0.110 ms，相对秒级的 δ 可以忽略。
贵的是**掉队者**：本日 5 次孤立保存的 δ 极差 1.44 s（中位的 24%），频繁保存时最慢一次是中位的 **3.5 倍**。
同步保存的耗时是所有 rank 里最慢的那个，N 越大越接近尾部，而不是中位数。

### 7.4 改法：异步快照

训练线程只做 GPU → pinned 内存的拷贝，写盘交给后台线程。按 D2H 26.3 GB/s，
7.15 GB 的阻塞从 5.92 s 降到约 **0.27 s（22 倍）**，代入 √(2δ/M)，同一 MTBF 下最小开销降为原来的 **21%**
（16,384 卡那一行从 4.49% 降到约 0.96%）。

- **解决的是 7.1**：δ 在公式里指训练被阻塞的时间，它直接变小。
- **解决不了 7.2**：字节总量不变，存储该饱和还是饱和；若下一次快照开始时上一次还没写完，要么阻塞、要么丢弃。
  所以约束从 "τ ≥ 公式值" 变成 "τ ≥ 一次写盘所需时间"，5.2 节的回写拥堵依然存在。
- **代价**：每个 rank 常驻一份与分片等大的 pinned 内存（本日配置 7.15 GB），且快照与训练争用 PCIe。
- 以上 0.27 s 与 21% 是**推算**，本日没有实现异步写。

## 8. 未完成与局限

- **异步写未实测**（5.1、7.4 的数字是按 D2H 带宽推算的）。
- **任务书原定的 10 / 50 / 200 / 1000 步扫描未按原样完成**，改为实测 2/4/8/16 步加公式外推（5.2）。
- **c1 → c2 同时补了 RNG 和 scheduler**，两者对 c1 漂移的贡献没有拆开（第 4 节）。
- **`TORCH_NCCL_ASYNC_ERROR_HANDLING=0` 只在 kill -9 下测过**，没有在 SIGSTOP 下测，那才是它可能起作用的场景。
- **6.2 那段约 92 s 尾巴的各组成部分没有逐一定位到控制变量。**
- **`num_workers=0`**，DataLoader worker 各自的 RNG 没有被这次实验覆盖。
- **单机双卡**，"掉卡"用 kill -9 / SIGSTOP 模拟，没有覆盖跨节点故障与 GPU 掉线（Xid）。
- 第 7 节的 MTBF 与存储带宽是外部假设，不是本仓库测量。
- `/root/autodl-tmp/train/` 里留有约 40 GB 的实验 checkpoint，未删除。

## 9. 原始数据

```
results/runs/determinism_{1,2}.json          默认模式两次运行
results/runs/determinism_det_{1,2}.json      warn_only=True
results/runs/determinism_strict_{1,2}.json   严格确定模式
results/runs/A_continuous.json               A 组（严格确定，16 步）
results/runs/B1_c{0,1,2}.json / B2_*.json    B 组保存 / 恢复
results/alignment.md / .json                 对齐判据分析
results/runs/bench_save.json, bench_c*.json  checkpoint 耗时与体积
results/runs/every_{2,4,8,16}.json           保存间隔扫描
results/runs/invalid_first_sweep/            作废的第一版扫描
results/youngdaly.md / .json                 Young/Daly 代入与千卡外推
results/kill_test.log, logs/kill_*.log       kill -9 与第一次弹性重启
results/hang_test2.log, logs/hang_*.log      SIGSTOP 超时
results/elastic_v2.log, logs/kill_C3.log     逐 rank RNG 的弹性重启
scripts/train_ft.py                          训练 + checkpoint + 故障注入
scripts/run11.sh fix11.sh kill_test.sh hang_test2.sh elastic_v2.sh
scripts/align.py youngdaly.py
```
