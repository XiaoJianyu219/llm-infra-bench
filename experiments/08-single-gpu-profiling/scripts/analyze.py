"""Day08 4.3 bound 类型判定：三条相互独立的证据。

E1 算子时间构成   —— GEMM 类 kernel 占 CUDA 时间多少（来自 profiler）
E2 GEMM 内部效率  —— 这些 kernel 自己跑到了实测峰值的百分之几（FLOPs/时间，来自 profiler with_flops）
E3 batch 缩放     —— step_time = a + b·batch 的固定开销 a 占比（来自 batch 扫描，不用 profiler）

E1 和 E2 都来自 profiler 但问的不是同一件事：E1 问"时间花在哪"，
E2 问"花在 GEMM 上的时间用得值不值"。E3 完全不依赖 profiler。
任务书 4.3 要求两条独立证据，这里给三条。
"""
import json, os, glob, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)

GEMM_CATS = {"gemm", "attention_flash", "attention_mem_efficient",
             "attention_cudnn", "conv"}


def main():
    peak = json.load(open(R("gemm_peak.json")))
    peak_tf = peak["peak_bf16_tflops"]
    peak_bw = peak["peak_bandwidth_gbps"]
    ridge = peak_tf * 1e12 / (peak_bw * 1e9)
    out = dict(peak_bf16_tflops=peak_tf, peak_bandwidth_gbps=peak_bw,
               ridge_point_flops_per_byte=ridge)

    # ---------- E1 / E2 ----------
    prof_path = R("profiles", "ref_s2048_b1.json")
    if os.path.exists(prof_path):
        p = json.load(open(prof_path))
        cats = p["by_category"]
        gemm_us = sum(v["us"] for k, v in cats.items() if k in GEMM_CATS)
        total_us = p["total_self_cuda_us"]
        out["E1"] = dict(
            gemm_family_share=gemm_us / total_us,
            by_category={k: round(v["share"], 4) for k, v in
                         sorted(cats.items(), key=lambda kv: -kv[1]["us"])},
            note="GEMM 家族 = gemm + 各 attention 后端 kernel")

        # E2：FLOPs 只挂在 CPU 侧 aten 行上，profile_run 已按此口径算好
        ach = p.get("gemm_achieved_tflops")
        out["E2"] = dict(
            flops_reported=p.get("gemm_flops_total"), us=p.get("gemm_flops_us"),
            achieved_tflops=ach,
            frac_of_measured_peak=(ach / peak_tf) if ach else None,
            note="GEMM/attention 类算子的 FLOPs 与其 device 时间之比；"
                 "FLOPs 取自 key_averages 的 aten 行（kernel 行不带 flops）")
        out["E1"]["kernel_launches_per_step"] = p.get("n_kernel_launches_per_step")
        out["E1"]["distinct_kernels"] = p.get("n_distinct_kernels")
    else:
        out["E1"] = out["E2"] = {"error": "profile json missing: " + prof_path}

    # ---------- E3 ----------
    pts = []
    for p in sorted(glob.glob(R("runs", "bsw_s512_b*_ckpt0.json"))):
        r = json.load(open(p))
        if r["status"] == "ok":
            pts.append((r["args"]["batch_size"], r["step_sec_median"], r["tokens_per_sec"]))
    pts.sort()
    if len(pts) >= 3:
        n = len(pts)
        xs = [x for x, _, _ in pts]
        ys = [y for _, y, _ in pts]
        mx, my = sum(xs) / n, sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        ref_b = 4   # s512 下的参考 batch
        ref_t = dict((x, y) for x, y, _ in pts).get(ref_b, ys[0])
        out["E3"] = dict(points=[dict(batch=x, step_sec=y, tokens_per_s=t) for x, y, t in pts],
                         fixed_overhead_sec=a, slope_sec_per_batch=b,
                         fixed_share_at_b4=(a / ref_t) if ref_t else None,
                         tps_ratio_b1_to_max=pts[0][2] / max(t for _, _, t in pts),
                         note="step_time = a + b*batch。a 大 => 每步有与计算量无关的固定开销（下发/优化器/lm_head 之外的常数项）")
    else:
        out["E3"] = {"error": "need >=3 batch points, got %d" % len(pts)}

    # ---------- 判定 ----------
    verdict = []
    if "gemm_family_share" in out.get("E1", {}):
        s = out["E1"]["gemm_family_share"]
        verdict.append("E1: GEMM 家族占 CUDA 时间 %.1f%% -> %s"
                       % (s * 100, "compute 侧主导" if s > .5 else "非 GEMM 占多数，memory 侧主导"))
    if out.get("E2", {}).get("frac_of_measured_peak"):
        s = out["E2"]["frac_of_measured_peak"]
        verdict.append("E2: GEMM 类算子内部达到实测峰值的 %.1f%% -> %s"
                       % (s * 100, "这些 kernel 自身接近算力上限" if s > .5
                          else "连 GEMM 自己都没吃满，形状太小/被访存拖住"))
    if out.get("E3", {}).get("fixed_share_at_b4") is not None:
        s = out["E3"]["fixed_share_at_b4"]
        verdict.append("E3: s512/b=4 时固定开销占单步 %.1f%% -> %s"
                       % (s * 100, "缩放接近线性，计算量主导" if s < .25
                          else "固定开销显著，小 batch 下是 launch/访存 主导"))
    out["verdict_lines"] = verdict

    json.dump(out, open(R("bound_analysis.json"), "w"), indent=1)
    print("peak bf16 %.1f TFLOPS | peak BW %.0f GB/s | ridge %.1f FLOP/byte"
          % (peak_tf, peak_bw, ridge))
    for v in verdict:
        print(" ", v)
    if "by_category" in out.get("E1", {}):
        print("\n算子分类占比：")
        for k, v in out["E1"]["by_category"].items():
            print("   %-18s %5.2f%%" % (k, v * 100))
    if "points" in out.get("E3", {}):
        print("\nbatch 缩放：")
        for d in out["E3"]["points"]:
            print("   b=%-3d step %7.1f ms  %8.0f tok/s" % (d["batch"], d["step_sec"] * 1000,
                                                            d["tokens_per_s"]))
        print("   拟合 a=%.1f ms  b=%.1f ms/batch" % (out["E3"]["fixed_overhead_sec"] * 1000,
                                                     out["E3"]["slope_sec_per_batch"] * 1000))
    print("\nsaved", R("bound_analysis.json"))


if __name__ == "__main__":
    main()
