"""Day09 3.6：把 results/runs/*.json 汇成并行策略对照表。

两种扩展效率的定义（任务书 3.4 要求分别标注，混为一谈是最常见的误用）：

  weak scaling   每卡 batch 固定，global batch 随卡数翻倍
                 eff_weak = 双卡总吞吐 / (P × 单卡吞吐@同样的每卡 batch)
                 反映的主要是通信开销

  strong scaling global batch 固定，每卡 batch 减半
                 eff_strong = 双卡总吞吐 / (P × 单卡吞吐@整个 global batch)
                 同时改变了每卡的计算效率，所以它比 weak 更低是正常的

理论显存（混合精度 + AdamW，16 字节/参数）：
  DDP 16N | ZeRO-1 12N/P+4N | ZeRO-2 12N/P+2N/P+2N | ZeRO-3 & FSDP 16N/P
"""
import json, os, glob, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)

SHARDS = {
    "single": "不切分（单卡基准）",
    "ddp": "不切分；每步 all-reduce 梯度",
    "zero1": "优化器状态（fp32 主权重 + Adam m + v = 12N）",
    "zero2": "优化器状态 + 梯度（2N）",
    "zero3": "优化器状态 + 梯度 + 参数（2N）",
    "fsdp": "优化器状态 + 梯度 + 参数（FULL_SHARD，等价 ZeRO-3）",
}


def theory_bytes_per_param(strategy, P):
    if strategy in ("single",) or P == 1:
        return 16.0
    if strategy == "ddp":
        return 16.0
    if strategy == "zero1":
        return 12.0 / P + 4.0
    if strategy == "zero2":
        return 12.0 / P + 2.0 / P + 2.0
    if strategy in ("zero3", "fsdp"):
        return 16.0 / P
    return None


def load():
    d = {}
    for p in sorted(glob.glob(R("runs", "*.json"))):
        r = json.load(open(p))
        d[os.path.basename(p)[:-5]] = r
    return d


def comm_summary():
    p = R("comm_bench.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    if not isinstance(c.get("collectives"), list):
        return c
    best = {}
    for row in c["collectives"]:
        for op in ("all_reduce", "all_gather", "reduce_scatter"):
            if op in row:
                b = row[op]["busbw_gbps"]
                if b > best.get(op, (0, 0))[0]:
                    best[op] = (b, row["size_mb"])
    c["_best_busbw"] = {k: dict(busbw_gbps=v[0], at_size_mb=v[1]) for k, v in best.items()}
    return c


def main():
    runs = load()
    ok = {k: r for k, r in runs.items() if r.get("status") == "ok"}
    if not ok:
        print("没有成功的 run，无法出表")
        return

    N = next(iter(ok.values())).get("params_total")
    lines = []
    w = lines.append

    # ---- 分母：通信带宽 ----
    c = comm_summary()
    w("## 通信带宽（第 1 节，所有归因的分母）")
    w("")
    if c is None:
        w("`results/comm_bench.json` 不存在，未完成。")
    elif not isinstance(c.get("collectives"), list):
        w("集合通信部分：**%s**" % c["collectives"])
        if "pcie" in c:
            w("")
            w("| 通道 | GB/s |")
            w("|---|---|")
            for k, v in c["pcie"].items():
                w("| %s | %.1f |" % (k, v["gbps"]))
    else:
        w("| 操作 | 实测峰值 busbw | 取得于 |")
        w("|---|---|---|")
        for k, v in c.get("_best_busbw", {}).items():
            w("| %s | **%.2f GB/s** | %d MB |" % (k, v["busbw_gbps"], v["at_size_mb"]))
        w("")
        w("| 消息大小 | all-reduce algbw | all-reduce busbw | all-gather busbw | reduce-scatter busbw |")
        w("|---|---|---|---|---|")
        for row in c["collectives"]:
            if "all_reduce" not in row:
                w("| %d MB | %s | | | |" % (row["size_mb"], row.get("error", "—")))
                continue
            g = lambda op: ("%.2f" % row[op]["busbw_gbps"]) if op in row else "—"
            w("| %d MB | %.2f | **%.2f** | %s | %s |"
              % (row["size_mb"], row["all_reduce"]["algbw_gbps"],
                 row["all_reduce"]["busbw_gbps"], g("all_gather"), g("reduce_scatter")))
        if "pcie" in c:
            w("")
            w("| 参照通道 | GB/s |")
            w("|---|---|")
            for k, v in c["pcie"].items():
                w("| %s | %.1f |" % (k, v["gbps"]))
    w("")

    # ---- 扩展效率的两个分母 ----
    mb = None
    for k, r in ok.items():
        if r["args"]["strategy"] == "single" and r["args"]["scaling"] == "weak":
            mb = r["args"]["micro_bsz"]
            base_weak = r
    base_strong = None
    for k, r in ok.items():
        if r["args"]["strategy"] == "single" and r["args"]["scaling"] == "strong":
            base_strong = r
    tps_weak_base = base_weak.get("total_tokens_per_sec") if mb else None
    tps_strong_base = base_strong.get("total_tokens_per_sec") if base_strong else None

    w("## 并行策略对照表（任务书 3.6）")
    w("")
    w("模型 Qwen3-0.6B，N = %s 参数；seq_len = %d，每卡 micro batch = %s，"
      "梯度累积 = %s（所有配置一致）。"
      % ("{:,}".format(N) if N else "?", base_weak["args"]["seq_len"], mb,
         base_weak["args"]["grad_accum"]))
    w("")
    w("| 策略 | 切分了什么 | 单卡显存 alloc | 理论显存 | 实测/理论 | 总吞吐 | 扩展效率(weak) | 扩展效率(strong) | 每步通信量 |")
    w("|---|---|---|---|---|---|---|---|---|")

    order = ["single", "ddp", "zero1", "zero2", "zero3", "fsdp"]
    for st in order:
        cands = [r for r in ok.values() if r["args"]["strategy"] == st
                 and r["args"]["micro_bsz"] == mb]
        if not cands:
            continue
        r = cands[0]
        P = r["world_size"]
        alloc = r.get("mem_allocated_max_mib", r.get("mem_allocated_mib"))
        bpp = theory_bytes_per_param(st, P)
        theo = (N * bpp / 2 ** 20) if (N and bpp) else None
        tot = r.get("total_tokens_per_sec")
        ew = (tot / (P * tps_weak_base) * 100) if (tot and tps_weak_base and P > 1) else None
        es = (tot / (P * tps_strong_base) * 100) if (tot and tps_strong_base and P > 1) else None
        comm = r.get("grad_comm_bytes_per_step")
        w("| %s (w%d) | %s | %.0f MiB | %s | %s | %.0f tok/s | %s | %s | %s |"
          % (st, P, SHARDS.get(st, "?"), alloc,
             ("%.0f MiB" % theo) if theo else "—",
             ("%.2f×" % (alloc / theo)) if theo else "—",
             tot or 0,
             ("**%.1f%%**" % ew) if ew else "— (基准)",
             ("**%.1f%%**" % es) if es else "— (基准)",
             ("%.2f GB" % (comm / 1e9)) if comm else "见 report"))
    w("")
    w("> 扩展效率 weak 的分母是单卡 micro=%s（%s tok/s）；"
      "strong 的分母是单卡 micro=%s（%s tok/s）。两者分母不同，不可互相比较。"
      % (mb, ("%.0f" % tps_weak_base) if tps_weak_base else "?",
         base_strong["args"]["micro_bsz"] if base_strong else "?",
         ("%.0f" % tps_strong_base) if tps_strong_base else "未测"))
    w("")

    # ---- 逐 run 明细 ----
    w("## 逐配置明细（`results/runs/<tag>.json`）")
    w("")
    w("| tag | 策略 | 卡数 | micro | step ms ± std | 总 tok/s | alloc max/min MiB | 不对称 | reserved | nvidia-smi | 状态 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(runs):
        r = runs[k]
        a = r["args"]
        if r.get("status") != "ok":
            w("| %s | %s | %d | %d | — | — | %.0f | — | %.0f | — | **%s** |"
              % (k, a["strategy"], r.get("world_size", 1), a["micro_bsz"],
                 r.get("mem_allocated_mib", 0), r.get("mem_reserved_mib", 0),
                 r.get("status")))
            continue
        w("| %s | %s | %d | %d | %.1f ± %.1f | %.0f | %.0f / %.0f | %.2f%% | %.0f | %s | ok |"
          % (k, a["strategy"], r["world_size"], a["micro_bsz"],
             r["step_sec_median"] * 1e3, r["step_sec_std_across_blocks"] * 1e3,
             r.get("total_tokens_per_sec", 0),
             r.get("mem_allocated_max_mib", r["mem_allocated_mib"]),
             r.get("mem_allocated_min_mib", r["mem_allocated_mib"]),
             r.get("mem_asymmetry_pct", 0.0), r["mem_reserved_mib"],
             r.get("smi_peak_mib")))
    w("")

    # ---- 需要人肉核对的几项 ----
    w("## 自动核对（任务书 4.3 / 4.4 / 4.5）")
    w("")
    for k in sorted(ok):
        r = ok[k]
        notes = []
        if r.get("mem_asymmetry_pct", 0) > 5:
            notes.append("rank 间显存不对称 %.1f%%" % r["mem_asymmetry_pct"])
        if "fsdp_wrapped_modules" in r:
            notes.append("FSDP 被 wrap 的模块数 %d（期望 ≥ %d）→ %s"
                         % (r["fsdp_wrapped_modules"], r["fsdp_expected_at_least"],
                            "OK" if r.get("fsdp_wrap_policy_ok") else "**wrap policy 可疑**"))
        if "ds_config_effective" in r:
            eff = {kk[len("ds_engine_"):]: vv for kk, vv in r.items()
                   if kk.startswith("ds_engine_")}
            notes.append("DS 生效 batch 配置 %s；engine 反查 %s"
                         % (r["ds_config_effective"], eff or "未取到"))
        if notes:
            w("- **%s**：%s" % (k, "；".join(notes)))
    w("")

    txt = "\n".join(lines)
    open(R("comparison_tables.md"), "w").write(txt)
    print(txt)
    print("\nsaved", R("comparison_tables.md"))


if __name__ == "__main__":
    main()
