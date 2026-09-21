"""Day10 3.3 噪声地板：同一配置、独立进程重复 N 次，定出"多大的变化才算变化"。

用独立进程而不是同一进程内的多个 block，因为后面比较各个配置时
每个配置本来就是一个独立进程——噪声地板必须与被比较的对象同口径。
"""
import json, glob, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)


def main():
    rows = []
    for p in sorted(glob.glob(R("runs", "noise_*.json"))):
        d = json.load(open(p))
        if d.get("status") == "ok":
            rows.append((os.path.basename(p)[:-5], d["step_sec_median"] * 1000,
                         d["tokens_per_sec"], d.get("MFU", 0) * 100,
                         d["step_rel_range_blocks"] * 100))
    if len(rows) < 3:
        print("噪声地板样本不足：只有 %d 个成功的 run" % len(rows))
        return

    steps = [r[1] for r in rows]
    tps = [r[2] for r in rows]
    mfus = [r[3] for r in rows]

    out = dict(n=len(rows), runs=[dict(tag=r[0], step_ms=r[1], tokens_per_sec=r[2],
                                       MFU_pct=r[3], within_run_range_pct=r[4])
                                  for r in rows])
    for name, v in (("step_ms", steps), ("tokens_per_sec", tps), ("MFU_pct", mfus)):
        m = st.median(v)
        out[name] = dict(median=m, mean=st.mean(v), std=st.pstdev(v),
                         rel_std_pct=st.pstdev(v) / m * 100,
                         min=min(v), max=max(v),
                         rel_range_pct=(max(v) - min(v)) / m * 100)

    # 噪声地板取"极差"口径：任何小于它的差异一律不作为结论
    floor = out["tokens_per_sec"]["rel_range_pct"]
    out["noise_floor_pct_range"] = floor
    out["noise_floor_pct_2sigma"] = 2 * out["tokens_per_sec"]["rel_std_pct"]
    out["decision_rule"] = ("相对基线的吞吐变化绝对值 < %.2f%%（极差口径）"
                            "一律记为「低于噪声地板，不构成结论」" % floor)

    json.dump(out, open(R("noise_floor.json"), "w"), indent=1)

    print("=== 噪声地板：%d 次独立进程重复，配置完全相同 ===" % len(rows))
    print("%-10s %10s %12s %8s %14s" % ("tag", "step_ms", "tokens/s", "MFU%", "进程内极差%"))
    for r in rows:
        print("%-10s %10.1f %12.0f %8.2f %14.2f" % r)
    print()
    for name in ("step_ms", "tokens_per_sec", "MFU_pct"):
        d = out[name]
        print("%-16s 中位 %10.2f  std %8.3f (%.2f%%)  极差 %.2f%%  [%.2f, %.2f]"
              % (name, d["median"], d["std"], d["rel_std_pct"], d["rel_range_pct"],
                 d["min"], d["max"]))
    print()
    print("**噪声地板 = %.2f%%（吞吐极差口径），2σ 口径 %.2f%%**"
          % (floor, out["noise_floor_pct_2sigma"]))
    print(out["decision_rule"])
    print("saved", R("noise_floor.json"))


if __name__ == "__main__":
    main()
