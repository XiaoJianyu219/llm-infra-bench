"""Day08 4.2 分母：实测这张卡的 GEMM 峰值与显存带宽。不抄规格书。"""
import torch, time, json, argparse, sys, subprocess

def sync(): torch.cuda.synchronize()

def time_op(fn, iters, warmup=10):
    for _ in range(warmup): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync()
    return (time.perf_counter() - t0) / iters

def gemm(m, k, n, dtype, iters=30):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    c = torch.empty(m, n, device="cuda", dtype=dtype)
    dt = time_op(lambda: torch.matmul(a, b, out=c), iters)
    del a, b, c; torch.cuda.empty_cache()
    return dt, 2.0 * m * k * n / dt / 1e12          # TFLOPS

def bandwidth(nbytes_per_elem=2, n=1 << 28, iters=30):
    """三种纯访存模式，取最好的作为可达带宽。"""
    dt_ = torch.bfloat16 if nbytes_per_elem == 2 else torch.float32
    x = torch.randn(n, device="cuda", dtype=dt_)
    z = torch.randn(n, device="cuda", dtype=dt_)
    y = torch.empty_like(x)
    nb = x.numel() * nbytes_per_elem
    res = {}
    t = time_op(lambda: y.copy_(x), iters)
    res["copy"] = dict(sec=t, gbps=2 * nb / t / 1e9, streams="1读1写")
    # 两个操作数必须是不同张量。写成 add(x, x) 的话硬件只读一遍，
    # 却按"2读1写"计流量，会算出超过理论带宽的假数字（第一版就踩了这个）。
    t = time_op(lambda: torch.add(x, z, out=y), iters)
    res["add"] = dict(sec=t, gbps=3 * nb / t / 1e9, streams="2读1写")
    t = time_op(lambda: x.sum(), iters)
    res["reduce"] = dict(sec=t, gbps=1 * nb / t / 1e9, streams="1读")
    for k, v in res.items():
        v["over_theoretical"] = v["gbps"] > 1008.0
    del x, y, z; torch.cuda.empty_cache()
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gemm_peak.json")
    a = ap.parse_args()
    torch.manual_seed(42)
    prec = {}
    for name, path in [("matmul_fp32_precision", "torch.backends.cuda.matmul.fp32_precision"),
                       ("cudnn_conv_fp32_precision", "torch.backends.cudnn.conv.fp32_precision")]:
        try: prec[name] = str(eval(path))
        except Exception as e: prec[name] = f"N/A ({type(e).__name__})"
    try: prec["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    except Exception: pass
    try: prec["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    except Exception: pass

    out = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
               cuda=torch.version.cuda, precision_flags=prec,
               spec_bf16_dense_tflops=165.2, spec_bandwidth_gbps=1008.0,
               square=[], model_shapes=[], bandwidth={})

    print("=== 方阵 GEMM 扫描 ===")
    print(f"{'N':>7} {'bf16 TF':>9} {'fp16 TF':>9} {'fp32 TF':>9}")
    for n in [1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]:
        row = dict(m=n, k=n, n=n)
        for dt, tag in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
            if dt is torch.float32 and n > 8192: row[tag] = None; continue
            try:
                sec, tf = gemm(n, n, n, dt, iters=20 if n <= 8192 else 8)
                row[tag] = round(tf, 2); row[tag + "_sec"] = sec
            except torch.cuda.OutOfMemoryError:
                row[tag] = None; torch.cuda.empty_cache()
        out["square"].append(row)
        f = lambda v: f"{v:9.2f}" if v else "        -"
        print(f"{n:7d} {f(row.get('bf16'))} {f(row.get('fp16'))} {f(row.get('fp32'))}")

    print("\n=== 模型里真实出现的 GEMM 形状（bf16, tokens=b*s） ===")
    H, D_ATTN, FFN, V = 1024, 2048, 3072, 151936
    for toks in [2048, 8192, 16384]:
        for label, (m, k, n) in [("qkv_q", (toks, H, D_ATTN)), ("kv", (toks, H, 1024)),
                                 ("o_proj", (toks, D_ATTN, H)), ("gate_up", (toks, H, FFN)),
                                 ("down", (toks, FFN, H)), ("lm_head", (toks, H, V))]:
            try:
                sec, tf = gemm(m, k, n, torch.bfloat16, iters=20)
                out["model_shapes"].append(dict(tokens=toks, label=label, m=m, k=k, n=n,
                                                tflops=round(tf, 2), sec=sec))
                print(f"tok={toks:6d} {label:9s} [{m}x{k}]x[{k}x{n}]  {tf:7.2f} TFLOPS")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); print(f"tok={toks} {label} OOM")

    print("\n=== 显存带宽 ===")
    out["bandwidth"] = bandwidth()
    for k, v in out["bandwidth"].items():
        flag = "  <-- 超过理论带宽，口径有问题" if v["over_theoretical"] else ""
        print(f"{k:8s} {v['gbps']:8.1f} GB/s  ({v['gbps']/1008*100:.1f}% of 1008) {v['streams']}{flag}")

    best_bf16 = max(r["bf16"] for r in out["square"] if r.get("bf16"))
    sane = [v["gbps"] for v in out["bandwidth"].values() if not v["over_theoretical"]]
    best_bw = max(sane) if sane else max(v["gbps"] for v in out["bandwidth"].values())
    out["peak_bf16_tflops"] = best_bf16
    out["peak_bandwidth_gbps"] = best_bw
    out["ridge_point_flops_per_byte"] = best_bf16 * 1e12 / (best_bw * 1e9)
    print(f"\n实测 bf16 峰值 {best_bf16:.1f} TFLOPS = 规格 165.2 的 {best_bf16/165.2*100:.1f}%")
    print(f"实测带宽峰值 {best_bw:.1f} GB/s = 规格 1008 的 {best_bw/1008*100:.1f}%")
    print(f"ridge point = {out['ridge_point_flops_per_byte']:.1f} FLOP/byte")
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)

if __name__ == "__main__":
    main()
