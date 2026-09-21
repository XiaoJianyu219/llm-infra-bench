# Day 12 已完成

本文件原为跨环境续作的临时笔记，现已无效。成果见：

- `report.md` — 镜像体积对比表、两个探针失败场景、容量标定、扩缩容时间线、GPU 调度论述、指标选择论证
- `journal.md` — 第 1–2 节为构建前写下的环境判定与五条预测（未改动），第 3 节为实测与判断错的地方

落地环境：本机 Windows + WSL2(Ubuntu) + Docker + kind 单节点集群。
AutoDL 容器内无法运行 dockerd（缺 cap_sys_admin、cgroup2 只读、user namespace 被禁），
判定过程见 `results/env_probe.txt` 与 journal 第 1 节。

app.py 有两个版本，A/B 对照用：
- `scripts/app_async_original.py` — /predict 为 async def，推理阻塞事件循环（故障侧）
- `scripts/app.py` — /predict 改为同步 def，交给线程池（修复侧）
