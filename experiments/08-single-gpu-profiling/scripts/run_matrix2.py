"""Day08 4.4 开关对比矩阵 v2。

v1 把基准定在 s2048/b4，结果 24 GB 上除了 b1 全部 OOM —— 原因是 lm_head 输出
[b, s, 151936] 这个张量，显存需求只跟"每步 token 总数"走，与 seq_len 怎么拆无关。
v2 因此改成：
  * 两个等 token 的参考点：s2048×b1 与 s512×b4（都是 2048 token/step），
    两者 logits 开销相同，差别只在 attention 的 O(s) 项 —— 正好隔离出 attention 项。
  * 所有开关对照都放在 s2048×b1 上做（单变量）。
  * batch 扫描改在 s512 上做，才有多于一个点的曲线。
  * 另加一条"每步 token 数"扫描，用来验证 OOM 墙确实由 token 数决定。
OOM 不重试、不降 batch，如实落盘。
"""
import json, os, subprocess, sys, time

PY = "/root/autodl-tmp/venvs/det/bin/python"
D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)


def peak_tflops():
    return json.load(open(os.path.join(ROOT, "results", "gemm_peak.json")))["peak_bf16_tflops"]


def one(tag, **kw):
    out = os.path.join(ROOT, "results", "runs", tag + ".json")
    if os.path.exists(out):
        print("skip (exists)", tag, flush=True)
        return json.load(open(out))
    cmd = [PY, os.path.join(D, "train_loop.py"), "--tag", tag, "--out", out,
           "--peak-tflops", str(peak_tflops())]
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.startswith("[") or l.startswith("OOM")]
    print("[%6.1fs] %s" % (time.time() - t0, tail or r.stderr[-300:]), flush=True)
    return json.load(open(out)) if os.path.exists(out) else None


def main():
    os.makedirs(os.path.join(ROOT, "results", "runs"), exist_ok=True)
    jobs = []
    A = dict(seq_len=2048, batch_size=1, precision="bf16", sdpa="flash", ckpt=0)
    B = dict(seq_len=512, batch_size=4, precision="bf16", sdpa="flash", ckpt=0)

    # 两个等 token 参考点
    jobs.append(("ref_s2048_b1", dict(A)))
    jobs.append(("ref_s512_b4", dict(B)))

    # 开关 1：gradient checkpointing（两个参考点各做一次）
    jobs.append(("ckpt1_s2048_b1", dict(A, ckpt=1)))
    jobs.append(("ckpt1_s512_b4", dict(B, ckpt=1)))

    # 开关 2：SDPA 后端（在 s2048 上做，attention 项占比最大）
    for bk in ("math", "mem_efficient", "flash", "cudnn"):
        jobs.append(("sdpa_%s_s2048_b1" % bk, dict(A, sdpa=bk)))
        jobs.append(("sdpa_%s_s512_b4" % bk, dict(B, sdpa=bk)))

    # 开关 3：精度。fp32 下 flash 不可用，故两边都用 mem_efficient 以保证单变量
    for prec in ("fp32", "bf16"):
        jobs.append(("prec_%s_s2048_b1" % prec, dict(A, precision=prec, sdpa="mem_efficient")))
        jobs.append(("prec_%s_s512_b4" % prec, dict(B, precision=prec, sdpa="mem_efficient")))

    # batch 扫描（s512，倍增到 OOM）
    for b in (1, 2, 4, 6, 8, 12):
        jobs.append(("bsw_s512_b%d_ckpt0" % b, dict(B, batch_size=b)))
    for b in (4, 8, 12, 16):
        jobs.append(("bsw_s512_b%d_ckpt1" % b, dict(B, batch_size=b, ckpt=1)))

    # 每步 token 数扫描：不同 (b,s) 组合、相同 token 数，验证 OOM 墙由 token 数决定
    for s, b in ((256, 8), (1024, 2), (2048, 1), (512, 4)):
        jobs.append(("tok2048_s%d_b%d" % (s, b),
                     dict(seq_len=s, batch_size=b, precision="bf16", sdpa="flash", ckpt=0)))
    for s, b in ((512, 8), (1024, 4), (2048, 2)):
        jobs.append(("tok4096_s%d_b%d" % (s, b),
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

    cols = ["tag", "status", "seq_len", "batch_size", "tokens_per_step", "precision", "sdpa",
            "ckpt", "step_ms", "step_std_ms", "tokens_per_s", "alloc_MiB", "reserved_MiB",
            "smi_MiB", "MFU_%", "MFU_head_%", "HFU_%"]
    lines = ["\t".join(cols)]
    for tag in sorted(results):
        r = results[tag]
        a = r["args"]
        row = [tag, r["status"], a["seq_len"], a["batch_size"],
               a["seq_len"] * a["batch_size"], a["precision"], a["sdpa"], a["ckpt"]]
        if r["status"] == "ok":
            row += ["%.1f" % (r["step_sec_median"] * 1000),
                    "%.2f" % (r["step_sec_std_across_blocks"] * 1000),
                    "%.0f" % r["tokens_per_sec"], "%.0f" % r["mem_allocated_mib"],
                    "%.0f" % r["mem_reserved_mib"], r["smi"]["max_mib"],
                    "%.2f" % (r.get("MFU", 0) * 100), "%.2f" % (r.get("MFU_with_head", 0) * 100),
                    "%.2f" % (r.get("HFU", 0) * 100)]
        else:
            row += ["", "", "", "%.0f" % r.get("mem_allocated_mib", 0),
                    "%.0f" % r.get("mem_reserved_mib", 0),
                    (r.get("smi") or {}).get("max_mib", ""), "", "", ""]
        lines.append("\t".join(str(x) for x in row))
    p = os.path.join(ROOT, "results", "matrix_summary.tsv")
    open(p, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved", p)


if __name__ == "__main__":
    main()
