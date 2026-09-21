"""Day08 4.5 显存 <-> 吞吐权衡曲线 + Pareto 前沿。

横轴用 max_memory_reserved（缓存分配器真正向驱动要走的量），
因为它才是决定"还能不能再塞一个配置"的那个数；allocated 会低估，
nvidia-smi 会把 CUDA context 也算进来。三个数都在 matrix_summary.tsv 里。

两处排版处理：
  * 同一配置在矩阵里被覆盖多次（例如 s512/b4/flash 测了 4 遍），按配置签名归组
    取中位数那一次画点，同时把重复测量的极差打出来当作噪声底。
  * 2048 token/step 的配置显存几乎完全相同，标签会挤成一团，
    所以按 0.5 GiB 分箱后在箱内逐个下移，不靠 matplotlib 自动排版。
"""
import json, os, glob, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")


def load():
    ok, oom = [], []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(p))
        (ok if r.get("status") == "ok" else oom).append(r)
    return ok, oom


def label(r):
    a = r["args"]
    return "b%d s%d %s %s%s" % (a["batch_size"], a["seq_len"], a["precision"],
                                {"mem_efficient": "memeff"}.get(a["sdpa"], a["sdpa"]),
                                "+ckpt" if a["ckpt"] else "")


def pareto(points):
    front = []
    for i, (m, t, r) in enumerate(points):
        dominated = any((m2 <= m and t2 >= t and (m2 < m or t2 > t))
                        for j, (m2, t2, _) in enumerate(points) if j != i)
        if not dominated:
            front.append((m, t, r))
    return sorted(front)


def main():
    ok, oom = load()
    groups = {}
    for r in ok:
        a = r["args"]
        sig = (a["seq_len"], a["batch_size"], a["precision"], a["sdpa"], a["ckpt"])
        groups.setdefault(sig, []).append(r)
    pts, repeat_info = [], []
    for sig, rs in groups.items():
        rs.sort(key=lambda x: x["tokens_per_sec"])
        r = rs[len(rs) // 2]
        pts.append((r["mem_reserved_mib"] / 1024, r["tokens_per_sec"], r))
        if len(rs) > 1:
            tps = [x["tokens_per_sec"] for x in rs]
            repeat_info.append((sig, len(rs), min(tps), max(tps),
                                (max(tps) - min(tps)) / statistics.median(tps)))
    front = pareto(pts)
    front_ids = {id(r) for _, _, r in front}

    fig, ax = plt.subplots(figsize=(13.5, 8))
    style = {("bf16", 0): ("o", "#1f77b4", "bf16"),
             ("bf16", 1): ("s", "#2ca02c", "bf16 + checkpointing"),
             ("fp32", 0): ("^", "#d62728", "fp32"),
             ("fp32", 1): ("v", "#9467bd", "fp32 + checkpointing")}
    shown = set()
    for m, t, r in pts:
        a = r["args"]
        mk, c, lbl = style.get((a["precision"], a["ckpt"]), ("D", "#7f7f7f", "other"))
        on_front = id(r) in front_ids
        ax.scatter(m, t, marker=mk, s=150 if on_front else 62, c=c,
                   edgecolors="black" if on_front else "none", linewidths=1.6,
                   zorder=4, label=lbl if lbl not in shown else None)
        shown.add(lbl)

    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], "--", c="black",
                lw=1.5, zorder=3, label="Pareto frontier")

    lo = min(t for _, t, _ in pts)
    hi = max(t for _, t, _ in pts)
    ax.set_xlim(10.6, 25.4)
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.42 * (hi - lo))

    bins = {}
    for m, t, r in pts:
        bins.setdefault(round(m * 2) / 2, []).append((m, t, r))
    for group in bins.values():
        group.sort(key=lambda x: -x[1])
        for i, (m, t, r) in enumerate(group):
            bold = id(r) in front_ids
            ax.annotate(label(r), (m, t), fontsize=7.5,
                        xytext=(9, 2 - 12 * i), textcoords="offset points",
                        color="black" if bold else "#444444",
                        fontweight="bold" if bold else "normal", zorder=6)

    y1 = ax.get_ylim()[1]
    for i, r in enumerate(sorted(oom, key=lambda x: x.get("mem_reserved_mib", 0))):
        m = r.get("mem_reserved_mib", 0) / 1024
        ax.axvline(m, color="red", ls=":", alpha=.30, zorder=1)
        ax.scatter(m, y1 - 0.004 * (y1 - lo), marker="x", s=80, c="red", zorder=5,
                   label="OOM (x = peak reserved before crash)" if i == 0 else None)
        ax.annotate("OOM " + label(r),
                    (m, y1 - (0.015 + 0.105 * (i % 3)) * (y1 - lo)),
                    rotation=90, fontsize=7, color="red", ha="center", va="top", zorder=6)

    ax.axvline(24564 / 1024, color="red", lw=2.2, alpha=.75, zorder=2)
    ax.annotate("RTX 4090 24 GB", (24564 / 1024 - 0.13, lo), rotation=90,
                color="red", fontsize=9, ha="right", va="bottom")
    ax.set_xlabel("Peak memory, torch.cuda.max_memory_reserved (GiB)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Day08  RTX 4090 24GB  Qwen3-0.6B training: memory vs throughput "
                 "(dashed = Pareto frontier)")
    ax.grid(alpha=.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=.96)
    out = os.path.join(ROOT, "results", "tradeoff.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)

    print("\n=== Pareto 前沿 ===")
    for m, t, r in front:
        print("%6.2f GiB  %8.0f tok/s  MFU %5.2f%%  %s" % (m, t, r.get("MFU", 0) * 100, label(r)))

    if repeat_info:
        print("\n=== 同配置重复测量的离散度（噪声底）===")
        for sig, n, mn, mx, spread in sorted(repeat_info, key=lambda x: -x[4]):
            print("  s%-5d b%-3d %-5s %-14s ckpt%d  n=%d  %.0f~%.0f tok/s  极差 %.2f%%"
                  % (sig[0], sig[1], sig[2], sig[3], sig[4], n, mn, mx, spread * 100))

    print("\n=== OOM 配置（显存为崩溃前峰值）===")
    for r in sorted(oom, key=lambda x: x.get("mem_reserved_mib", 0)):
        a = r["args"]
        print("%-30s tokens/step=%-6d alloc %.0f MiB  resv %.0f MiB  阶段=%s 第%s步"
              % (r.get("tag"), a["seq_len"] * a["batch_size"],
                 r.get("mem_allocated_mib", 0), r.get("mem_reserved_mib", 0),
                 r.get("oom_at_phase"), r.get("oom_step_index")))


if __name__ == "__main__":
    main()
