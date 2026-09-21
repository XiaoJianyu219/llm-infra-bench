import json, statistics as st
from collections import defaultdict

for mode in ("fixed", "varlen"):
    for s in ("continuous", "static"):
        d = json.load(open(f"/root/autodl-tmp/bench/day03/{s}_{mode}.json"))
        g = defaultdict(list)
        for r in d["records"]:
            g[r["out_len"]].append(r["latency"])
        print(f"\n=== {s} / {mode} ===")
        print(f"{'out_len':>8} {'n':>4} {'延迟中位数':>12}")
        for k in sorted(g):
            print(f"{k:>8} {len(g[k]):>4} {st.median(g[k]):>11.1f}s")
