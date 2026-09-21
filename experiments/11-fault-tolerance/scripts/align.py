"""Day11 2.2 逐位对齐判据分析。"""
import json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)
L = lambda t: (json.load(open(R("runs", t + ".json")))
               if os.path.exists(R("runs", t + ".json")) else None)

SYMPTOM = [
    (lambda d0, mx, drift: d0 > 0.05, "第一步就差很多 → 优化器状态或 LR 没恢复"),
    (lambda d0, mx, drift: drift > 0.01, "loss 整体偏移 → 数据顺序变了（sampler 没恢复）"),
    (lambda d0, mx, drift: 0 < mx <= 0.05, "差异很小但持续存在 → RNG 没恢复（dropout 序列不同）"),
]
LEVELS = {"c0": "权重 + 优化器",
          "c1": "＋数据位置 + global step",
          "c2": "＋RNG + scheduler + 配置（完整）"}


def main():
    out, lines = {}, []
    w = lines.append

    # 噪声带：两次不中断运行之间的差（默认模式 vs 严格确定模式）
    for tag, name in (("determinism", "默认（Flash Attention backward 非确定）"),
                      ("determinism_strict", "严格确定（use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG）")):
        a, b = L(tag + "_1"), L(tag + "_2")
        if a and b:
            d = [abs(x - y) for x, y in zip(a["losses"], b["losses"])]
            out[tag] = dict(bitwise=all(x == 0 for x in d), max_abs=max(d),
                            mean_abs=sum(d) / len(d), n=len(d))
            w("- **%s**：逐位一致 = %s，最大绝对差 **%.3e**"
              % (name, "是" if out[tag]["bitwise"] else "否", max(d)))
    w("")

    A = L("A_continuous")
    if not A:
        w("缺少 A 组，无法对齐")
    else:
        la = A["losses"]
        w("| checkpoint 完整度 | 存了什么 | 恢复了哪些字段 | 首步差 | 最大差 | 平均差 | 逐位一致 | 症状判定 |")
        w("|---|---|---|---|---|---|---|---|")
        for lv in ("c0", "c1", "c2"):
            B = L("B2_" + lv)
            if not B:
                w("| %s | %s | — | — | — | — | — | **缺失** |" % (lv, LEVELS[lv]))
                continue
            n0 = B.get("resumed_at_step", 8)
            lb = B["losses"]
            ref = la[n0:n0 + len(lb)]
            d = [abs(x - y) for x, y in zip(ref, lb)]
            d0, mx = d[0], max(d)
            drift = sum(d[len(d) // 2:]) / max(1, len(d) - len(d) // 2)
            hits = [m for c, m in SYMPTOM if c(d0, mx, drift)] or ["无差异"]
            out[lv] = dict(first=d0, max=mx, mean=sum(d) / len(d), drift_tail=drift,
                           bitwise=all(x == 0 for x in d),
                           restored=B.get("restored_fields"), symptoms=hits,
                           L_A=ref, L_B=lb)
            w("| **%s** | %s | %s | %.3e | %.3e | %.3e | %s | %s |"
              % (lv, LEVELS[lv], ", ".join(B.get("restored_fields", [])),
                 d0, mx, sum(d) / len(d),
                 "**是**" if out[lv]["bitwise"] else "否", "；".join(hits)))
        w("")
        for lv in ("c0", "c1", "c2"):
            if lv in out:
                w("**%s 逐步差（前 6 步）**：%s" % (lv, ["%.3e" % abs(x - y) for x, y in
                    zip(out[lv]["L_A"][:6], out[lv]["L_B"][:6])]))
        w("")

    # 保存耗时
    b = L("bench_save")
    if b and b.get("bench_save"):
        sz = b["bench_save_bytes"]
        w("### checkpoint 写入耗时与体积")
        w("")
        w("| | 值 |")
        w("|---|---|")
        w("| 体积 | **%.2f GB** |" % (sz / 1e9))
        w("| 阻塞耗时（组装 + D2H）中位 | **%.3f s** |" % b["bench_save_median_blocking"])
        w("| 总耗时（含写盘）中位 | **%.3f s** |" % b["bench_save_median_total"])
        sb = b.get("size_breakdown", {})
        if sb:
            w("| 其中 模型权重 | %.2f GB（%.1f%%） |"
              % (sb["model_bytes"] / 1e9, sb["model_bytes"] / sz * 100))
            w("| 其中 优化器状态 | %.2f GB（%.1f%%） |"
              % (sb["optimizer_state_bytes"] / 1e9, sb["optimizer_state_bytes"] / sz * 100))
        w("")
        out["bench"] = b["bench_save"]
        out["delta_sec"] = b["bench_save_median_total"]

    # 间隔扫描
    rows = []
    for ev in (2, 4, 8, 16):
        r = L("every_%d" % ev)
        if r and r.get("step_times"):
            rows.append((ev, len(r.get("save_events", [])),
                         sum(r["step_times"]) + sum(e["total_sec"] for e in r.get("save_events", [])),
                         len(r["step_times"])))
    base = L("A_continuous")
    if rows and base:
        import statistics as st
        b0 = st.median(base["step_times"])
        w("### 保存间隔扫描（含保存开销的等效吞吐）")
        w("")
        w("| 间隔（步） | 实测保存次数 | 平均每步墙钟 | 相对无保存 |")
        w("|---|---|---|---|")
        for ev, nsave, tot, n in rows:
            per = tot / n
            w("| %d | %d | %.3f s | %+.1f%% |" % (ev, nsave, per, (per / b0 - 1) * 100))
        w("")
    json.dump(out, open(R("alignment.json"), "w"), indent=1, default=str)
    txt = "\n".join(lines)
    open(R("alignment.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
