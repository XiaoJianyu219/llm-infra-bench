"""Day08 把 results/ 下的原始 json 直接渲染成 report 用的 markdown 表。

存在的意义是 R05/第 6 节：report 里每个数字都能追到 results/ 的某个文件，
不经过人工转抄。
"""
import json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)
OUT = []


def w(s=""):
    OUT.append(s)


def load_runs():
    d = {}
    for p in glob.glob(R("runs", "*.json")):
        r = json.load(open(p))
        d[r.get("tag") or os.path.basename(p)[:-5]] = r
    return d


def row(r):
    a = r["args"]
    if r["status"] != "ok":
        return ("| %s | **%s** | — | %.0f | %.0f | %s | — | — |"
                % (desc(a), r["status"], r.get("mem_allocated_mib", 0),
                   r.get("mem_reserved_mib", 0), r.get("smi", {}).get("max_mib", "—")))
    return ("| %s | %.0f ± %.0f | %.0f | %.0f | %.0f | %s | %.2f%% | %.2f%% |"
            % (desc(a), r["step_sec_median"] * 1000, r["step_sec_std_across_blocks"] * 1000,
               r["tokens_per_sec"], r["mem_allocated_mib"], r["mem_reserved_mib"],
               r["smi"]["max_mib"], r.get("MFU", 0) * 100, r.get("HFU", 0) * 100))


def desc(a):
    return "s%d b%d (%d tok) %s %s%s" % (a["seq_len"], a["batch_size"],
                                        a["seq_len"] * a["batch_size"], a["precision"],
                                        a["sdpa"], " +ckpt" if a["ckpt"] else "")


def main():
    runs = load_runs()

    w("### 全配置汇总（原始文件：`results/runs/<tag>.json`）")
    w()
    w("| 配置 | step ms ± std | tokens/s | alloc MiB | reserved MiB | nvidia-smi MiB | MFU | HFU |")
    w("|---|---|---|---|---|---|---|---|")
    for tag in sorted(runs):
        w(row(runs[tag]))
    w()

    gp = R("gemm_peak.json")
    if os.path.exists(gp):
        g = json.load(open(gp))
        w("### GEMM 峰值扫描（`results/gemm_peak.json`）")
        w()
        w("| N (N×N×N) | bf16 TFLOPS | fp16 TFLOPS | fp32 TFLOPS |")
        w("|---|---|---|---|")
        for s in g["square"]:
            f = lambda v: ("%.1f" % v) if v else "OOM/跳过"
            w("| %d | %s | %s | %s |" % (s["m"], f(s.get("bf16")), f(s.get("fp16")), f(s.get("fp32"))))
        w()
        w("| 模型真实 GEMM 形状 | bf16 TFLOPS |")
        w("|---|---|")
        for s in g["model_shapes"]:
            w("| tok=%d %s [%d×%d]×[%d×%d] | %.1f |"
              % (s["tokens"], s["label"], s["m"], s["k"], s["k"], s["n"], s["tflops"]))
        w()
        w("| 带宽模式 | GB/s | 占 1008 GB/s |")
        w("|---|---|---|")
        for k, v in g["bandwidth"].items():
            w("| %s | %.0f | %.1f%% |" % (k, v["gbps"], v["gbps"] / 1008 * 100))
        w()

    sc = R("step_curve.json")
    if os.path.exists(sc):
        c = json.load(open(sc))["cold_steps_sec"]
        w("### 冷启动逐步耗时（`results/step_curve.json`，前 %d 步）" % len(c))
        w()
        w("| step | ms | 相对稳态 |")
        w("|---|---|---|")
        steady = sorted(c[len(c) // 2:])[len(c[len(c) // 2:]) // 2]
        for i, t in enumerate(c):
            w("| %d | %.1f | %+.1f%% |" % (i, t * 1000, (t / steady - 1) * 100))
        w()

    for name in ("ref_s2048_b1", "ref_s512_b4", "prof_math_s2048_b1",
                 "prof_ckpt_s2048_b1", "prof_fp32_s2048_b1"):
        p = R("profiles", name + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        w("### profiler · %s（`results/profiles/%s.json`）" % (name, name))
        w()
        w("SDPA 实测后端 kernel：`%s`" % ", ".join(d["sdpa_backend_seen"]))
        w()
        w("| 算子类别 | 占 CUDA 时间 |")
        w("|---|---|")
        for k, v in sorted(d["by_category"].items(), key=lambda kv: -kv[1]["us"]):
            w("| %s | %.2f%% |" % (k, v["share"] * 100))
        w()
        w("| # | 算子 | 类别 | self CUDA ms | 占比 | 调用数 |")
        w("|---|---|---|---|---|---|")
        for i, r in enumerate(d["top"][:10], 1):
            w("| %d | `%s` | %s | %.2f | %.2f%% | %d |"
              % (i, r["name"][:58], r["cat"], r["self_cuda_us"] / 1000,
                 r["self_cuda_us"] / d["total_self_cuda_us"] * 100, r["count"]))
        w()

    ba = R("bound_analysis.json")
    if os.path.exists(ba):
        d = json.load(open(ba))
        w("### bound 判定（`results/bound_analysis.json`）")
        w()
        for v in d["verdict_lines"]:
            w("- " + v)
        w()

    txt = "\n".join(OUT)
    open(R("report_tables.md"), "w").write(txt)
    print(txt)
    print("\nsaved", R("report_tables.md"))


if __name__ == "__main__":
    main()
