import json, statistics as st

def load(s, m):
    return json.load(open(f"/root/autodl-tmp/bench/day03/{s}_{m}.json"))

print(f"{'负载':>8} {'策略':>12} {'吞吐 req/s':>12} {'延迟 p50':>10} {'延迟 p99':>10}")
for mode in ("fixed", "varlen"):
    row = {}
    for s in ("continuous", "static"):
        d = load(s, mode)
        lat = sorted(r["latency"] for r in d["records"])
        thr = len(d["records"]) / d["makespan"]
        row[s] = thr
        print(f"{mode:>8} {s:>12} {thr:>12.2f} "
              f"{st.median(lat):>9.1f}s {lat[int(len(lat)*0.99)]:>9.1f}s")
    print(f"{'':>8} {'加速比':>12} {row['continuous']/row['static']:>12.2f}×\n")