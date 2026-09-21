#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Day13 3.1：从 Prometheus 取回压测期间的时序，做两件事——
  1) 按并发档汇总 QPS / p99 / 在途请求 / GPU 利用率，证明面板上的波形是真的
  2) Little 定律自洽校验：QPS × 平均延迟 ≈ 平均在途请求数
     对不上说明某个指标算错了。Day01 用过这条交叉验证，这里再用一次。
"""
import json, sys, time, urllib.parse, urllib.request

PROM = "http://127.0.0.1:19090"


def q_range(expr, start, end, step=1):
    url = PROM + "/api/v1/query_range?" + urllib.parse.urlencode(
        dict(query=expr, start=start, end=end, step=step))
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    assert d["status"] == "success", d
    res = d["data"]["result"]
    if not res:
        return []
    return [(float(t), float(v)) for t, v in res[0]["values"] if v not in ("NaN",)]


def seg(series, a, b):
    return [v for t, v in series if a <= t < b]


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    meta = json.loads(open(sys.argv[1] if len(sys.argv) > 1
                           else "results/loadgen.jsonl").readline())
    start, end = meta["wall_start"] - 20, meta["wall_end"] + 10
    marks = meta["marks"]

    E = {
        "qps":       'sum(rate(sod_requests_total{status="200"}[10s]))',
        "err":       'sum(rate(sod_requests_total{status!="200"}[10s]))',
        "p50":       'histogram_quantile(0.50, sum by (le) (rate(sod_request_duration_seconds_bucket[30s])))',
        "p95":       'histogram_quantile(0.95, sum by (le) (rate(sod_request_duration_seconds_bucket[30s])))',
        "p99":       'histogram_quantile(0.99, sum by (le) (rate(sod_request_duration_seconds_bucket[30s])))',
        "avg_lat":   'sum(rate(sod_request_duration_seconds_sum[30s])) / sum(rate(sod_request_duration_seconds_count[30s]))',
        "inflight":  'sod_inflight_requests',
        "queue":     'sod_queue_depth',
        "gpu_util":  'sod_gpu_utilization_percent',
        "gpu_mem":   'sod_gpu_memory_used_bytes / 1048576',
        "avg_batch": 'sum(rate(sod_batch_size_sum[30s])) / sum(rate(sod_batch_size_count[30s]))',
    }
    S = {k: q_range(v, start, end) for k, v in E.items()}
    for k, v in S.items():
        print("%-10s 采到 %d 个点" % (k, len(v)))
    out = {"expr": E, "stages": []}

    print()
    print("=" * 96)
    print("压测各阶段（每档取该档后 40 秒的均值，避开切换瞬态）")
    print("=" * 96)
    print("%-6s %-10s %8s %8s %9s %9s %9s %8s %9s %8s" %
          ("并发", "墙钟", "QPS", "p50(ms)", "p95(ms)", "p99(ms)", "在途", "队列", "GPU%", "批大小"))
    rows = []
    for i, m in enumerate(marks):
        if m["concurrency"] == 0:
            continue
        a = meta["wall_start"] + m["t"] + 20          # 跳过前 20 秒爬坡
        b = meta["wall_start"] + (marks[i + 1]["t"] if i + 1 < len(marks) else m["t"] + 60)
        r = dict(concurrency=m["concurrency"], wall=m["wall"],
                 qps=mean(seg(S["qps"], a, b)),
                 p50=mean(seg(S["p50"], a, b)) * 1000,
                 p95=mean(seg(S["p95"], a, b)) * 1000,
                 p99=mean(seg(S["p99"], a, b)) * 1000,
                 avg_lat=mean(seg(S["avg_lat"], a, b)),
                 inflight=mean(seg(S["inflight"], a, b)),
                 queue=mean(seg(S["queue"], a, b)),
                 gpu=mean(seg(S["gpu_util"], a, b)),
                 gpu_mem=mean(seg(S["gpu_mem"], a, b)),
                 batch=mean(seg(S["avg_batch"], a, b)))
        rows.append(r)
        print("%-6d %-10s %8.1f %8.1f %9.1f %9.1f %9.2f %8.2f %9.1f %8.2f" %
              (r["concurrency"], r["wall"], r["qps"], r["p50"], r["p95"],
               r["p99"], r["inflight"], r["queue"], r["gpu"], r["batch"]))
    out["stages"] = rows

    print()
    print("=" * 96)
    print("Little 定律自洽校验：QPS × 平均延迟 ≈ 平均在途请求数")
    print("=" * 96)
    print("%-6s %12s %14s %16s %14s %10s" %
          ("并发", "QPS", "平均延迟(s)", "QPS×延迟(预测)", "实测在途", "偏差"))
    checks = []
    for r in rows:
        pred = r["qps"] * r["avg_lat"]
        obs = r["inflight"]
        dev = abs(pred - obs) / obs * 100 if obs else float("nan")
        checks.append(dict(concurrency=r["concurrency"], predicted=pred, observed=obs, dev_pct=dev))
        print("%-6d %12.2f %14.4f %16.2f %14.2f %9.1f%%" %
              (r["concurrency"], r["qps"], r["avg_lat"], pred, obs, dev))
    out["little"] = checks
    worst = max((c["dev_pct"] for c in checks if c["dev_pct"] == c["dev_pct"]), default=float("nan"))
    print("\n最大偏差 %.1f%%  ->  %s" % (worst, "自洽" if worst < 15 else "不自洽，需要查指标"))
    out["little_worst_dev_pct"] = worst

    # GPU 空载/满载对照
    idle = [v for t, v in S["gpu_util"] if t > meta["wall_end"] - 30]
    out["gpu_idle_after_load"] = mean(idle)
    print("\n停压后 30 秒 GPU 利用率均值: %.1f%%" % mean(idle))

    json.dump(out, open("results/panel_numbers.json", "w"), ensure_ascii=False, indent=1)
    json.dump({k: v for k, v in S.items()}, open("results/panel_series.json", "w"))
    print("\n已写 results/panel_numbers.json 与 results/panel_series.json")


if __name__ == "__main__":
    main()
