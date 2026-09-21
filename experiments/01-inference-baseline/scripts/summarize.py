import json, glob, os, re, statistics
OUT = "/root/autodl-tmp/bench/day01_4090D_4090D"
rows = {}
for f in sorted(glob.glob(f"{OUT}/bench_c*_r*.json")):
    c, r = map(int, re.search(r"c(\d+)_r(\d+)", os.path.basename(f)).groups())
    d = json.load(open(f))
    mem = f"{OUT}/mem_c{c}_r{r}.log"
    if os.path.exists(mem):
        v = [int(x) for x in open(mem).read().split() if x.strip().isdigit()]
        d["_mem"] = max(v) if v else None
    rows.setdefault(c, []).append(d)

def med(lst, k):
    v = [x.get(k) for x in lst if x.get(k) is not None]
    return statistics.median(v) if v else float("nan")

print("conc\treq/s\tout_tok/s\tTTFT_p50\tTTFT_p99\tTPOT_p50\tTPOT_p99\tmem_MiB")
for c in sorted(rows):
    r = rows[c]
    print(f"{c}\t{med(r,'request_throughput'):.2f}\t{med(r,'output_throughput'):.1f}\t"
          f"{med(r,'median_ttft_ms'):.0f}\t{med(r,'p99_ttft_ms'):.0f}\t"
          f"{med(r,'median_tpot_ms'):.1f}\t{med(r,'p99_tpot_ms'):.1f}\t"
          f"{med(r,'_mem'):.0f}")
