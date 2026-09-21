#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Day13 3.1：把压测期间的 Prometheus 时序渲染成面板图存档。

**口径声明（写进 report，不能含糊）**：
Grafana 在本机装不上（dl.grafana.com / GitHub / 两个国内镜像实测 12 秒 0 字节，
见 results/grafana_unreachable.txt），所以这几张 PNG 不是 Grafana 截图，
而是**用 Grafana dashboard JSON 里同一批 PromQL 从 Prometheus 取回的数据渲染的**。
dashboards/sod_dashboard.json 里的表达式与本文件 EXPR 一一对应。
Prometheus 自带界面上的实时波形已当场核对过（report 第 2.4 节）。
"""
import json, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

S = json.load(open("results/panel_series.json"))
meta = json.loads(open("results/loadgen.jsonl").readline())
marks = [m for m in meta["marks"]]
t0 = meta["wall_start"]


def rel(series):
    return [(t - t0) for t, _ in series], [v for _, v in series]


def mark_stages(ax):
    for m in marks:
        ax.axvline(m["t"], color="#999", lw=0.8, ls="--", alpha=0.7)
        ax.text(m["t"] + 2, ax.get_ylim()[1] * 0.93,
                ("并发 %d" % m["concurrency"]) if m["concurrency"] else "停压",
                fontsize=7, color="#555")


def panel(ax, keys, title, ylabel, colors=None, scale=1.0):
    colors = colors or ["#2e7d32", "#1565c0", "#c62828", "#6a1b9a"]
    for i, k in enumerate(keys):
        if not S.get(k):
            continue
        x, y = rel(S[k])
        ax.plot(x, [v * scale for v in y], lw=1.2, color=colors[i % len(colors)], label=k)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, meta["wall_end"] - t0)
    if len(keys) > 1:
        ax.legend(fontsize=7, loc="upper left")
    mark_stages(ax)


plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
panel(axes[0], ["qps"], "QPS  —  sum(rate(sod_requests_total{status=\"200\"}[10s]))", "req/s")
panel(axes[1], ["p50", "p95", "p99"],
      "latency p50/p95/p99  —  histogram_quantile(...)  [NOT the mean]", "seconds")
panel(axes[2], ["inflight", "queue"], "in-flight requests / queue depth  (gauge)", "count")
panel(axes[3], ["gpu_util"], "GPU utilization  —  nvidia-smi utilization.gpu", "%")
panel(axes[4], ["avg_batch"], "avg batch size  —  dynamic batching", "images/batch")
panel(axes[5], ["gpu_mem"], "GPU memory used  —  nvidia-smi scope (whole card, incl. CUDA context)", "MiB")
axes[-1].set_xlabel("seconds since load start (T+0 = %s)" % time.strftime("%H:%M:%S", time.localtime(t0)),
                    fontsize=8)
fig.suptitle("Day13 monitoring panel — TensorRT FP16 detection service on RTX 4090\n"
             "rendered from Prometheus (scrape_interval=1s); Grafana could not be installed on this host",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig("results/dashboard_overview.png", dpi=110)
print("wrote results/dashboard_overview.png")

# 单独两张：面板最该讲的两件事
fig2, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
panel(ax[0], ["qps"], "QPS saturates at ~37 while concurrency goes 1 -> 16", "req/s")
panel(ax[1], ["p99"], "p99 keeps climbing: 0.05 -> 0.52 s  (throughput flat, latency 10x)", "seconds")
ax[-1].set_xlabel("seconds since load start", fontsize=8)
fig2.suptitle("Saturation: adding concurrency buys latency, not throughput", fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig("results/dashboard_saturation.png", dpi=110)
print("wrote results/dashboard_saturation.png")

fig3, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
panel(ax[0], ["inflight"], "in-flight requests tracks load (1 -> 16)", "count")
panel(ax[1], ["gpu_util"], "GPU utilization stays at 5-7% the whole time", "%")
ax[-1].set_xlabel("seconds since load start", fontsize=8)
fig3.suptitle("Why queue depth, not utilization: the pressure signal is in-flight, not GPU%",
              fontsize=11)
fig3.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig("results/dashboard_metric_choice.png", dpi=110)
print("wrote results/dashboard_metric_choice.png")
