"""Day08 4.4 开关对比矩阵。每个配置一个独立进程，避免上一次的内存池污染下一次。

单变量纪律：每组只改一个开关，基准配置 s2048/b4/bf16/flash/ckpt0 固定不动。
OOM 不重试、不降 batch，如实落盘（任务书 4.4）。
"""
import json, os, subprocess, sys, time, itertools

PY = "/root/autodl-tmp/venvs/det/bin/python"
D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)


def peak_tflops():
    p = os.path.join(ROOT, "results", "gemm_peak.json")
    return json.load(open(p))["peak_bf16_tflops"]


def one(tag, **kw):
    out = os.path.join(ROOT, "results", "runs", tag + ".json")
    if os.path.exists(out):
        print("skip (exists)", tag)
        return json.load(open(out))
    cmd = [PY, os.path.join(D, "train_loop.py"), "--tag", tag, "--out", out,
           "--peak-tflops", str(peak_tflops())]
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("[%6.1fs] %s" % (time.time() - t0, r.stdout.strip().splitlines()[-2:] or r.stderr[-300:]))
    sys.stdout.flush()
    if not os.path.exists(out):
        print("  !! no output; stderr tail:", r.stderr[-500:])
        return None
    return json.load(open(out))


def main():
    os.makedirs(os.path.join(ROOT, "results", "runs"), exist_ok=True)
    jobs = []

    # --- 基准 ---
    jobs.append(("base_s2048_b4_bf16_flash", dict(seq_len=2048, batch_size=4,
                                                  precision="bf16", sdpa="flash", ckpt=0)))
    # --- A 精度：同一 sdpa 后端(mem_efficient，fp32 下唯一可用的非 math 后端) ---
    for b in (1, 2):
        for prec in ("fp32", "bf16"):
            jobs.append(("prec_%s_b%d_memeff" % (prec, b),
                         dict(seq_len=2048, batch_size=b, precision=prec,
                              sdpa="mem_efficient", ckpt=0)))
    # --- B gradient checkpointing ---
    for c in (0, 1):
        jobs.append(("ckpt%d_s2048_b4_bf16_flash" % c,
                     dict(seq_len=2048, batch_size=4, precision="bf16", sdpa="flash", ckpt=c)))
    # --- C SDPA 后端 ---
    for bk in ("math", "mem_efficient", "flash", "cudnn"):
        jobs.append(("sdpa_%s_s2048_b4_bf16" % bk,
                     dict(seq_len=2048, batch_size=4, precision="bf16", sdpa=bk, ckpt=0)))
    # --- D batch 倍增直到 OOM（无 ckpt） ---
    for b in (1, 2, 4, 8, 16, 32):
        jobs.append(("bs%d_s2048_bf16_flash_ckpt0" % b,
                     dict(seq_len=2048, batch_size=b, precision="bf16", sdpa="flash", ckpt=0)))
    # --- E batch 倍增（开 ckpt） ---
    for b in (4, 8, 16, 32, 64):
        jobs.append(("bs%d_s2048_bf16_flash_ckpt1" % b,
                     dict(seq_len=2048, batch_size=b, precision="bf16", sdpa="flash", ckpt=1)))
    # --- F 序列长度（4.2 要求 512 / 2048 两档） ---
    for s, b in ((512, 16), (2048, 4)):
        jobs.append(("seq%d_bf16_flash_iso" % s,
                     dict(seq_len=s, batch_size=b, precision="bf16", sdpa="flash", ckpt=0)))

    seen, results = set(), {}
    for tag, kw in jobs:
        if tag in seen:
            continue
        seen.add(tag)
        print("=== %s ===" % tag, flush=True)
        r = one(tag, **kw)
        if r:
            results[tag] = r

    # 汇总表
    cols = ["tag", "status", "seq_len", "batch_size", "precision", "sdpa", "ckpt",
            "step_ms", "step_std_ms", "tokens_per_s", "alloc_MiB", "reserved_MiB",
            "smi_MiB", "MFU_%", "MFU_head_%", "HFU_%"]
    lines = ["\t".join(cols)]
    for tag, r in results.items():
        a = r["args"]
        g = lambda k, d="": r.get(k, d)
        row = [tag, r["status"], a["seq_len"], a["batch_size"], a["precision"], a["sdpa"], a["ckpt"]]
        if r["status"] == "ok":
            row += ["%.1f" % (r["step_sec_median"] * 1000),
                    "%.1f" % (r["step_sec_std_across_blocks"] * 1000),
                    "%.0f" % r["tokens_per_sec"],
                    "%.0f" % r["mem_allocated_mib"], "%.0f" % r["mem_reserved_mib"],
                    r["smi"]["max_mib"],
                    "%.2f" % (r.get("MFU", 0) * 100), "%.2f" % (r.get("MFU_with_head", 0) * 100),
                    "%.2f" % (r.get("HFU", 0) * 100)]
        else:
            row += ["", "", "", "%.0f" % r.get("mem_allocated_mib", 0),
                    "%.0f" % r.get("mem_reserved_mib", 0), r.get("smi", {}).get("max_mib", ""),
                    "", "", ""]
        lines.append("\t".join(str(x) for x in row))
    open(os.path.join(ROOT, "results", "matrix_summary.tsv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved results/matrix_summary.tsv")


if __name__ == "__main__":
    main()
