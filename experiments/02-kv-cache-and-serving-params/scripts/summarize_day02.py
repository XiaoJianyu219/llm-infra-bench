import json, glob, os, re, statistics
OUT = "/root/autodl-tmp/bench/day02"
rows = {}
for f in sorted(glob.glob(f"{OUT}/bench_*_c*_r*.json")):
    m = re.search(r"bench_(.+)_c(\d+)_r(\d+)\.json$", os.path.basename(f))
    if not m: continue
    rows.setdefault((m.group(1), int(m.group(2))), []).append(json.load(open(f)))

def med(l, k):
    v = [x[k] for x in l if x.get(k) is not None]
    return statistics.median(v) if v else float("nan")
def spread(l, k):
    v = [x[k] for x in l if x.get(k) is not None]
    if len(v) < 2: return float("nan")
    m = statistics.median(v)
    return (max(v) - min(v)) / m * 100 if m else float("nan")

print("config\tconc\treq/s\t波动%\tout_tok/s\tTTFT_p50\tTTFT_p99\tTPOT_p50")
for name in ["base","seqs64","seqs256","util90","util96","len4096","kvmem381"]:
    for c in (16, 64):
        l = rows.get((name, c))
        if not l: continue
        print(f"{name}\t{c}\t{med(l,'request_throughput'):.2f}\t{spread(l,'request_throughput'):.1f}\t"
              f"{med(l,'output_throughput'):.1f}\t{med(l,'median_ttft_ms'):.0f}\t"
              f"{med(l,'p99_ttft_ms'):.0f}\t{med(l,'median_tpot_ms'):.1f}")
