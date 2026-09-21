#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Day13 收尾：三轮同配置重复的噪声地板（沿用 Day10 的口径：独立进程、极差）。"""
import json, os
R = "results"
files = [("r1", f"{R}/panel_numbers_r1.json"), ("r2", f"{R}/panel_numbers_r2.json"),
         ("r3", f"{R}/panel_numbers_r3.json")]
runs = {}
for tag, p in files:
    if os.path.exists(p):
        runs[tag] = {s["concurrency"]: s for s in json.load(open(p))["stages"]}
print("参与统计的轮次:", list(runs))
print()
hdr = "%-6s" % "并发" + "".join("%12s" % t for t in runs) + "%12s%10s%10s" % ("中位", "极差%", "2σ口径%")
for metric, name in (("qps", "QPS"), ("p99", "p99(ms)"), ("inflight", "在途")):
    print("=" * 76); print("指标:", name); print(hdr)
    for c in (1, 2, 4, 8, 16):
        vals = [runs[t][c][metric] for t in runs if c in runs[t]]
        if len(vals) < 2: continue
        vs = sorted(vals); med = vs[len(vs) // 2]
        rng = (max(vals) - min(vals)) / med * 100
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        print("%-6d" % c + "".join("%12.2f" % runs[t][c][metric] for t in runs if c in runs[t])
              + "%12.2f%10.1f%10.1f" % (med, rng, 2 * sd / mean * 100))
    print()
qs = []
for c in (2, 4, 8, 16):
    vals = [runs[t][c]["qps"] for t in runs if c in runs[t]]
    if len(vals) >= 2:
        qs.append((max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100)
print("=" * 76)
print("并发>=2 各档 QPS 极差: " + ", ".join("%.1f%%" % q for q in qs))
print("=> 噪声地板（QPS 极差口径，取各档最大）: %.1f%%" % max(qs))
print("注：并发 1 那一档单独看，见下方")
v1 = [runs[t][1]["qps"] for t in runs if 1 in runs[t]]
print("并发 1 的三轮 QPS: %s，极差 %.1f%%  <-- 明显高于其它档" %
      (["%.1f" % v for v in v1], (max(v1) - min(v1)) / (sum(v1) / len(v1)) * 100))
