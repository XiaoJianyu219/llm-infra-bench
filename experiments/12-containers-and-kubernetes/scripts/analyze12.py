#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day12 3.5：把 k8s 状态采样与压测逐秒指标对齐到同一条墙钟时间轴。

产出两张表：
  1) 扩容/缩容时间线：每个状态变化点（HPA 利用率、期望副本、就绪副本）的墙钟时刻
  2) QPS / 延迟随时间的变化，以及"新副本就绪"与"QPS 抬升"之间的滞后
"""
import json, re, sys, os
from datetime import datetime

R = "results"


def hms(s):
    return datetime.strptime(s, "%H:%M:%S")


def parse_timeline(path):
    """-> [(wall_str, util_pct|None, desired, ready_cnt, total_cnt)]"""
    rows = []
    cur = None
    hpa = None
    pods = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.rstrip("\n")
        m = re.match(r"^=== (\d\d:\d\d:\d\d) ===$", ln)
        if m:
            if cur:
                rows.append((cur, hpa, pods))
            cur, hpa, pods = m.group(1), None, []
            continue
        if cur is None:
            continue
        if ln.startswith("sod   Deployment/sod") or re.match(r"^sod\s+Deployment/sod", ln):
            f = ln.split()
            # sod Deployment/sod cpu: 1%/60% 1 4 1 30s
            u = re.search(r"cpu:\s*(<unknown>|\d+)%?/", ln)
            util = None if (not u or u.group(1) == "<unknown>") else int(u.group(1))
            hpa = (util, int(f[-2]))          # (利用率, REPLICAS 列)
        elif re.match(r"^sod-\S+\s+\d/\d\s+", ln):
            f = ln.split()
            pods.append((f[0], f[1], f[2]))   # name, READY(1/1), STATUS
    if cur:
        rows.append((cur, hpa, pods))
    out = []
    for w, h, ps in rows:
        util, desired = (h if h else (None, None))
        ready = sum(1 for _, r, st in ps if r.split("/")[0] == r.split("/")[1] and st == "Running")
        out.append((w, util, desired, ready, len(ps)))
    return out


def changes(tl):
    """只保留状态发生变化的采样点"""
    out, prev = [], None
    for w, util, desired, ready, total in tl:
        key = (desired, ready, total)
        if key != prev:
            out.append((w, util, desired, ready, total))
            prev = key
    return out


def load_lt(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def report(tag):
    tlp = os.path.join(R, "run%s_k8s_timeline.txt" % tag)
    ltp = os.path.join(R, "run%s_loadtest.jsonl" % tag)
    if not (os.path.exists(tlp) and os.path.exists(ltp)):
        print("[%s] 缺数据: %s / %s" % (tag, tlp, ltp))
        return
    tl = parse_timeline(tlp)
    lt = load_lt(ltp)
    t0 = hms(lt[0]["wall"])

    print("\n" + "=" * 78)
    print("Run %s  扩缩容时间线（T0 = 加压开始 %s）" % (tag, lt[0]["wall"]))
    print("=" * 78)
    print("%-10s %7s %6s %7s %7s  %s" % ("墙钟", "T+秒", "CPU%", "期望副本", "就绪副本", "事件"))
    prev_ready = None
    for w, util, desired, ready, total in changes(tl):
        dt = (hms(w) - t0).total_seconds()
        ev = ""
        if prev_ready is not None:
            if total > (prev_total if prev_ready is not None else 0):
                ev = "新 Pod 被创建/调度"
            if ready > prev_ready:
                ev = "新副本进入 Ready（开始接流量）"
            elif ready < prev_ready:
                ev = "副本掉出 Ready（重启或被杀）"
        else:
            ev = "起点"
        print("%-10s %7.0f %6s %7s %7s  %s" %
              (w, dt, ("-" if util is None else util), desired, "%d/%d" % (ready, total), ev))
        prev_ready, prev_total = ready, total

    # CPU 首次越线
    first_over = next(((w, u) for w, u, d, r, t in tl if u is not None and u >= 60), None)
    if first_over:
        print("\nCPU 利用率首次 >= 60%% 目标值: %s (T+%.0fs), 值=%d%%"
              % (first_over[0], (hms(first_over[0]) - t0).total_seconds(), first_over[1]))

    ok = sum(r["ok"] for r in lt)
    err = sum(r["err"] for r in lt)
    p50s = [r["p50"] for r in lt if r["p50"]]
    print("\nRun %s 压测汇总: 成功 %d, 失败 %d, 失败率 %.1f%%, 平均 QPS %.2f"
          % (tag, ok, err, 100.0 * err / max(ok + err, 1), ok / max(len(lt), 1)))
    if p50s:
        p50s.sort()
        print("           逐秒 p50 的中位数 %.2fs, 最大 %.2fs" % (p50s[len(p50s) // 2], p50s[-1]))

    # 分段 QPS：按 60s 窗口
    print("\n%-12s %8s %8s %10s" % ("窗口(T+)", "成功数", "失败数", "窗口QPS"))
    for s in range(0, 301, 60):
        seg = [r for r in lt if s <= r["t"] < s + 60]
        if not seg:
            continue
        o = sum(r["ok"] for r in seg)
        e = sum(r["err"] for r in seg)
        print("%-12s %8d %8d %10.2f" % ("%d-%ds" % (s, s + 60), o, e, o / float(len(seg))))


if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["A", "B"]):
        report(tag)
