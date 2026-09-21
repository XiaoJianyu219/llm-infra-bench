"""Day11 2.3 / 2.5：把实测 δ 代入 Young/Daly，并做千卡外推。

推导本身在 days/day10/journal.md 第 3 节（总开销期望 → 求极值 → τ_opt = √(2δM)，
最小开销 √(2δ/M)，Daly 高阶修正）。这里只代数。

标注规则：来自本仓库实测的数字写明出处文件；外部假设单独列出并标「假设」。
"""
import json, os, math, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)
L = lambda t: json.load(open(R("runs", t + ".json")))


def young(delta, M):
    return math.sqrt(2 * delta * M)


def daly(delta, M):
    if delta >= 2 * M:
        return M
    x = delta / (2 * M)
    return math.sqrt(2 * delta * M) * (1 + math.sqrt(x) / 3 + x / 9) - delta


def overhead(delta, M, tau):
    """H(τ) = δ/τ + τ/(2M)，一阶口径（与推导一致）。"""
    return delta / tau + tau / (2 * M)


def main():
    bench = L("bench_save")
    delta = bench["bench_save_median_total"]                 # 实测 δ
    deltas = [r["total_sec"] for r in bench["bench_save"]]
    ckpt_bytes = bench["bench_save_bytes"]
    write_bw = ckpt_bytes / delta                             # 单写者实测写入带宽
    step_det = st.median(L("A_continuous")["step_times"])     # 本日配置（严格确定、无 compile）
    step_d10 = 0.7877                                         # Day10 最优单卡（compile），day10 results/runs/single_mb4_k8_compile.json
    d2h = 26.3e9                                              # Day09 实测 D2H pinned，day09 results/comm_bench.json

    out = dict(delta_sec=delta, delta_samples=deltas, ckpt_bytes=ckpt_bytes,
               write_bw_Bps=write_bw, step_sec_this_day=step_det, step_sec_day10_best=step_d10)
    lines = []
    w = lines.append

    w("### Young/Daly 代入（δ = %.3f s，实测 5 次中位，范围 %.3f–%.3f s）"
      % (delta, min(deltas), max(deltas)))
    w("")
    w("| M（MTBF） | τ_opt 一阶 | τ_opt Daly | 本日配置步数（%.3f s/步） | Day10 最优配置步数（%.4f s/步） | 最小开销 √(2δ/M) |"
      % (step_det, step_d10))
    w("|---|---|---|---|---|---|")
    yd = []
    for name, M in (("1 小时", 3600), ("1 天", 86400), ("1 周", 604800)):
        t1, t2 = young(delta, M), daly(delta, M)
        h = math.sqrt(2 * delta / M)
        yd.append(dict(M_name=name, M=M, tau_young=t1, tau_daly=t2,
                       steps_this=t1 / step_det, steps_d10=t1 / step_d10, min_overhead=h))
        w("| %s | %.0f s | %.0f s | %.0f 步 | %.0f 步 | %.2f%% |"
          % (name, t1, t2, t1 / step_det, t1 / step_d10, h * 100))
    out["young_daly"] = yd
    w("")

    # 与实测扫描对照：常数 δ 的预测 vs 实测
    A = step_det
    w("### 实测扫描 vs 常数 δ 假设（每 k 步存一次，保存开销摊到每步）")
    w("")
    w("| 间隔 k | 实测单次 δ（均值） | 常数 δ 预测的开销 δ/(k·t) | 实测开销 | 偏差来源 |")
    w("|---|---|---|---|---|")
    sweep = []
    for k in (2, 4, 8, 16):
        r = L("every_%d" % k)
        ts, se = r["step_times"], r["save_events"]
        dsave = sum(e["total_sec"] for e in se)
        meas = (sum(ts) + dsave) / sum(ts) - 1
        pred = delta / (k * st.median(ts))
        dm = dsave / len(se)
        sweep.append(dict(k=k, n_save=len(se), mean_delta=dm, pred_overhead=pred,
                          meas_overhead=meas, deltas=[e["total_sec"] for e in se]))
        w("| %d | %.2f s | %.1f%% | %.1f%% | %s |"
          % (k, dm, pred * 100, meas * 100,
             "δ 被连续写盘拉长 %.1f×" % (dm / delta) if dm > 1.2 * delta else "δ 基本不变"))
    out["sweep"] = sweep
    w("")

    # ---------------- 2.5 千卡外推 ----------------
    # 外部假设（非本仓库实测）：
    #   M1：Llama 3 技术报告 54 天内 419 次非计划中断、16,384 卡 → 集群 MTBF ≈ 3.09 h，
    #       折合单卡 M1 ≈ 3.09 h × 16384 ≈ 50,600 h。只作量级锚点。
    #   B_s：共享并行存储聚合写带宽，假设 100 GB/s。
    #   模型：70B，按任务书 16 字节/参数 → 1.12 TB/次。
    M1_h = 54 * 24 / 419 * 16384
    M1 = M1_h * 3600
    Bs = 100e9
    total_70b = 16 * 70e9
    out["assumptions"] = dict(M1_hours=M1_h, shared_storage_Bps=Bs, model_bytes_70B=total_70b,
                              source_M1="Llama 3 技术报告：54 天 419 次非计划中断 @16,384 卡")
    w("### 2.5 千卡外推（外部假设见表下）")
    w("")
    n_sat = Bs / write_bw
    w("存储饱和点：每写者实测 %.2f GB/s（%.2f GB / %.2f s），"
      "共享存储 %.0f GB/s 在 **N ≈ %.0f 个并发写者**时饱和。"
      % (write_bw / 1e9, ckpt_bytes / 1e9, delta, Bs / 1e9, n_sat))
    w("")
    w("| 卡数 N | 集群 MTBF M₁/N | 70B 单次 δ = 1.12TB / min(N·%.2f GB/s, 100 GB/s) | τ_opt | 最小开销 |"
      % (write_bw / 1e9))
    w("|---|---|---|---|---|")
    scale = []
    for N in (8, 64, 1024, 16384, 100000):
        MN = M1 / N
        dN = total_70b / min(N * write_bw, Bs)
        tau = young(dN, MN)
        h = math.sqrt(2 * dN / MN)
        scale.append(dict(N=N, M_N_sec=MN, delta_sec=dN, tau_sec=tau, overhead=h))
        w("| %s | %s | %.1f s | %s | **%.2f%%** |"
          % ("{:,}".format(N), fmt(MN), dN, fmt(tau), h * 100))
    out["scale"] = scale
    # 开销到 10% 时的规模
    d_sat = total_70b / Bs
    M_10 = 2 * d_sat / 0.10 ** 2
    N_10 = M1 / M_10
    out["N_at_10pct"] = N_10
    w("")
    w("饱和后 δ 恒为 %.1f s；最小开销达到 10%% 需 M_N ≤ 2δ/0.01 = %.0f s，即 **N ≈ %s 卡**。"
      % (d_sat, M_10, "{:,}".format(int(N_10))))
    w("")

    # 同步 checkpoint 的掉队者
    w("掉队者：本日 5 次孤立保存 δ 的极差 %.2f s（中位的 %.0f%%）；"
      "频繁保存时单次 δ 最长 %.2f s（中位的 %.1f 倍）。"
      % (max(deltas) - min(deltas), (max(deltas) - min(deltas)) / delta * 100,
         max(max(s["deltas"]) for s in sweep), max(max(s["deltas"]) for s in sweep) / delta))
    w("")
    w("异步快照：D2H 实测 26.3 GB/s（Day09），7.15 GB 快照的训练阻塞 ≈ %.2f s，"
      "比同步 %.2f s 短 %.0f 倍；代入 √(2δ/M)，同一 M 下最小开销降为原来的 %.0f%%。"
      % (ckpt_bytes / d2h, delta, delta / (ckpt_bytes / d2h),
         math.sqrt((ckpt_bytes / d2h) / delta) * 100))
    out["async_stall_sec_est"] = ckpt_bytes / d2h
    json.dump(out, open(R("youngdaly.json"), "w"), indent=1)
    txt = "\n".join(lines)
    open(R("youngdaly.md"), "w").write(txt)
    print(txt)


def fmt(s):
    if s >= 86400:
        return "%.1f 天" % (s / 86400)
    if s >= 3600:
        return "%.1f h" % (s / 3600)
    if s >= 60:
        return "%.1f min" % (s / 60)
    return "%.0f s" % s


if __name__ == "__main__":
    main()
