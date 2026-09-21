"""Day10 3.4 overlap 观测。只定位不计时（任务书 4.4）。

回答三个问题：
  1) all-reduce kernel 与 backward 的计算 kernel 有没有重叠
  2) GPU 等通信的空泡占多少
  3) DDP 的 bucket 是在 backward 过程中分批触发，还是全部堆在最后

第 3 问靠导出 chrome trace 后看 NCCL kernel 的时间分布：
  把一个 step 的 GPU 时间轴归一化到 [0,1]，看 NCCL kernel 的启动时刻落在哪。
  若 bucket 按预期在 backward 中分批触发，NCCL 的启动时刻应当散布在后半段；
  若全堆在最后，它们会挤在接近 1.0 的地方。

重叠系数的定义同 Day 09：Σ(所有 CUDA kernel 自时间) / 干净单步时间。
NCCL kernel 在独立 stream 上与计算并发，所以该值可以 > 1，超出部分即重叠量。
"""
import argparse, json, os, re, sys, glob
import torch
import torch.distributed as dist
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity, schedule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_opt as TO

COMM = r"nccl|ncclDevKernel|ncclKernel|AllReduce|ReduceScatter|AllGather|Broadcast"


def cat(n):
    if re.search(COMM, n, re.IGNORECASE):
        return "communication"
    low = n.lower()
    for c, pat in (("gemm", r"gemm|cutlass|ampere_|simt|s16816"),
                   ("attention", r"flash_fwd|flash_bwd|pytorch_flash|fmha"),
                   ("optimizer", r"adam|multi_tensor|foreach"),
                   ("softmax", r"softmax"),
                   ("elementwise", r"elementwise|copy|fill|cast|memcpy|memset"),
                   ("reduction", r"reduce_kernel|norm_kernel")):
        if re.search(pat, low):
            return c
    return "other"


def trace_bucket_analysis(path):
    """从 chrome trace 里看 NCCL kernel 在一个 step 内的时间分布。"""
    try:
        ev = json.load(open(path)).get("traceEvents", [])
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}
    ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")
           and "ts" in e and "dur" in e]
    if not ker:
        return {"error": "trace 里没有 kernel 事件"}
    comm = [e for e in ker if re.search(COMM, e.get("name", ""), re.IGNORECASE)]
    if not comm:
        return {"n_comm_kernels": 0, "note": "没有通信 kernel"}
    t0 = min(e["ts"] for e in ker)
    t1 = max(e["ts"] + e["dur"] for e in ker)
    span = t1 - t0 or 1.0
    pos = sorted(((e["ts"] - t0) / span) for e in comm)   # 每个通信 kernel 的相对启动时刻
    n = len(pos)
    q = lambda f: pos[min(n - 1, int(f * n))]
    return dict(n_comm_kernels=n,
                gpu_span_us=span,
                comm_total_us=sum(e["dur"] for e in comm),
                first_comm_at=pos[0], last_comm_at=pos[-1],
                p25=q(0.25), p50=q(0.50), p75=q(0.75),
                spread=pos[-1] - pos[0],
                note=("通信 kernel 的相对启动时刻；散布在 [first,last] 区间说明 bucket "
                      "在 backward 过程中分批触发，挤在接近 1.0 说明堆在最后"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bsz", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--bucket-cap-mb", type=int, default=0)
    p.add_argument("--grad-as-bucket-view", type=int, default=1)
    p.add_argument("--static-graph", type=int, default=0)
    p.add_argument("--find-unused", type=int, default=0)
    p.add_argument("--fused-adam", type=int, default=0)
    p.add_argument("--set-to-none", type=int, default=1)
    p.add_argument("--tf32", type=int, default=-1)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--active", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategy", default="ddp")
    p.add_argument("--clean-step-json", default="")
    p.add_argument("--tag", default="prof")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    torch.cuda.set_device(TO.LOCAL_RANK)
    if a.strategy != "single" or TO.WORLD > 1:
        dist.init_process_group("nccl")

    info = {}
    model = TO.build(a, info)
    opt = TO.make_opt(model, a, info)
    g = torch.Generator(device="cpu").manual_seed(a.seed + TO.RANK)
    ids = torch.randint(0, TO.VOCAB, (a.micro_bsz, a.seq_len), generator=g).cuda()
    labels = ids.clone()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    import contextlib
    has_ns = hasattr(model, "no_sync")

    def one_step():
        opt.zero_grad(set_to_none=bool(a.set_to_none))
        for i in range(a.grad_accum):
            last = (i == a.grad_accum - 1)
            with (contextlib.nullcontext() if (last or not has_ns) else model.no_sync()):
                with autocast:
                    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
                (loss / a.grad_accum).backward()
        opt.step()

    for _ in range(a.warmup):
        one_step()
    torch.cuda.synchronize()

    trace_path = os.path.join(os.path.dirname(a.out) or ".", a.tag + "_trace.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=1, active=a.active, repeat=1),
                 record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for _ in range(1 + a.active):
            one_step()
            torch.cuda.synchronize()
            prof.step()
    if TO.RANK == 0:
        prof.export_chrome_trace(trace_path)

    rows = []
    for e in prof.key_averages():
        us = float(getattr(e, "self_device_time_total", 0) or 0)
        if us <= 0 or getattr(e, "is_user_annotation", False):
            continue
        if getattr(e, "device_type", None) != DeviceType.CUDA:
            continue
        rows.append(dict(name=e.key, us=us, count=int(e.count), cat=cat(e.key)))
    rows.sort(key=lambda r: -r["us"])
    total = sum(r["us"] for r in rows)
    bycat = {}
    for r in rows:
        d = bycat.setdefault(r["cat"], dict(us=0.0, n=0))
        d["us"] += r["us"]
        d["n"] += r["count"]
    for d in bycat.values():
        d["share"] = d["us"] / total

    clean = None
    if a.clean_step_json and os.path.exists(a.clean_step_json):
        clean = json.load(open(a.clean_step_json)).get("step_sec_median")

    out = dict(config=vars(a), info={k: v for k, v in info.items() if k != "args"},
               total_self_cuda_us=total, active_steps=a.active,
               cuda_busy_ms_per_step=total / a.active / 1000, clean_step_sec=clean,
               overlap_factor=((total / a.active / 1e6) / clean) if clean else None,
               bubble_frac=(max(0.0, 1 - (total / a.active / 1e6) / clean) if clean else None),
               overlap_gain=(max(0.0, (total / a.active / 1e6) / clean - 1) if clean else None),
               by_category={k: v for k, v in sorted(bycat.items(), key=lambda kv: -kv[1]["us"])},
               top=rows[:20])
    if TO.RANK == 0:
        out["bucket_timeline"] = trace_bucket_analysis(trace_path)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("=== %s (world=%d, k=%d) ===" % (a.tag, TO.WORLD, a.grad_accum))
        for k, v in out["by_category"].items():
            print("  %-16s %6.2f%%  %8.2f ms  launches=%d"
                  % (k, v["share"] * 100, v["us"] / 1000, v["n"]))
        if clean:
            print("  kernel 自时间合计 %.1f ms/步；干净单步 %.1f ms；重叠系数 %.2f×"
                  % (out["cuda_busy_ms_per_step"], clean * 1000, out["overlap_factor"]))
            print("  → %s %.1f%%" % (("通信与计算重叠，重叠量相当于单步的"
                                      if out["overlap_factor"] >= 1 else "GPU 真实空泡"),
                                     (out["overlap_gain"] if out["overlap_factor"] >= 1
                                      else out["bubble_frac"]) * 100))
        bt = out["bucket_timeline"]
        if "n_comm_kernels" in bt and bt["n_comm_kernels"]:
            print("  bucket 时间分布：%d 个通信 kernel，相对启动时刻 "
                  "first=%.2f p25=%.2f p50=%.2f p75=%.2f last=%.2f（0=步首 1=步尾）"
                  % (bt["n_comm_kernels"], bt["first_comm_at"], bt["p25"],
                     bt["p50"], bt["p75"], bt["last_comm_at"]))
        print("saved", a.out)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
