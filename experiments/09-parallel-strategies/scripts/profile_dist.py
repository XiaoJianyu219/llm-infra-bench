"""Day09 3.5：通信与计算有没有重叠。

只定位不计时（Day 08 的教训，任务书 4.6）：报告里的吞吐一律来自不挂 profiler 的
train_dist.py，本脚本产出的只有两类结论：
  1) NCCL 通信 kernel 占 CUDA 时间多少
  2) 重叠系数 = (所有 CUDA kernel self time 之和) / (干净单步时间)
     NCCL 通信 kernel 跑在独立 stream 上，与计算 kernel 并发执行，
     所以各 stream 自时间之和**可以超过**墙钟时间：
        > 1  ⇒ 通信与计算确实重叠了，超出部分就是重叠量
        < 1  ⇒ 没重叠满，差额是 GPU 真实空泡
     干净单步时间从 results/runs/<对应 tag>.json 里取，不用本脚本自己的墙钟
     （挂了 profiler 的墙钟含 profiler 开销，Day 08 的教训）。

profiler 的 key_averages() 有三层坑（Day 08 踩过，这里直接按正确口径取）：
  a) CPU 侧 aten 行与 GPU 侧 kernel 行是同一段时间的两次记账 → 只取 device_type==CUDA
  b) FLOPs 只挂在 aten 行 → 本脚本不用 FLOPs，忽略
  c) ProfilerStep* / Optimizer.step#... 是 is_user_annotation 的区间标注 → 必须排掉
"""
import argparse, json, os, re, sys, time
import torch
import torch.distributed as dist
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity, schedule
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_dist as TD

COMM = r"nccl|ncclDevKernel|ncclKernel|AllReduce|ReduceScatter|AllGather|Broadcast"


def cat(name):
    if re.search(COMM, name, re.IGNORECASE):
        return "communication"
    low = name.lower()
    if re.search(r"gemm|cutlass|ampere_|simt|s16816", low):
        return "gemm"
    if re.search(r"flash_fwd|flash_bwd|pytorch_flash|fmha", low):
        return "attention"
    if re.search(r"adam|multi_tensor|foreach", low):
        return "optimizer"
    if re.search(r"softmax", low):
        return "softmax"
    if re.search(r"elementwise|copy|fill|cast|memcpy|memset", low):
        return "elementwise"
    if re.search(r"reduce_kernel|norm_kernel", low):
        return "reduction"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bsz", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--active", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--clean-step-json", default="",
                   help="不挂 profiler 那次的结果 json，用它的 step_sec_median 算空泡")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    torch.cuda.set_device(TD.LOCAL_RANK)
    if a.strategy != "single" or TD.WORLD > 1:
        dist.init_process_group("nccl")

    info = {}
    model = TD.build_model(torch.bfloat16 if a.lora else torch.float32)
    if a.lora:
        model = TD.apply_lora(model, a, info)
    if a.strategy == "single":
        net, opt, mode = TD.setup_single(model, a, info)
    elif a.strategy == "ddp":
        net, opt, mode = TD.setup_ddp(model, a, info)
    elif a.strategy.startswith("zero"):
        net, opt, mode = TD.setup_deepspeed(model, a, info, int(a.strategy[-1]))
    elif a.strategy == "fsdp":
        net, opt, mode = TD.setup_fsdp(model, a, info)
    else:
        raise ValueError(a.strategy)
    is_ds = (mode == "deepspeed")

    g = torch.Generator(device="cpu").manual_seed(a.seed + TD.RANK)
    ids = torch.randint(0, TD.VOCAB, (a.micro_bsz, a.seq_len), generator=g).cuda()
    labels = ids.clone()
    import contextlib
    ctx = (contextlib.nullcontext() if (is_ds or a.strategy == "fsdp" or a.lora)
           else torch.autocast("cuda", dtype=torch.bfloat16))

    def one_step():
        if is_ds:
            loss = net(input_ids=ids, labels=labels, use_cache=False).loss
            net.backward(loss)
            net.step()
            return
        opt.zero_grad(set_to_none=True)
        with ctx:
            loss = net(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        opt.step()

    for _ in range(a.warmup):
        one_step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=1, active=a.active, repeat=1),
                 record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for _ in range(1 + a.active):
            one_step()
            torch.cuda.synchronize()
            prof.step()

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

    out = dict(strategy=a.strategy, lora=a.lora, world=TD.WORLD, rank=TD.RANK,
               total_self_cuda_us=total, active_steps=a.active,
               cuda_busy_ms_per_step=total / a.active / 1000,
               clean_step_sec=clean,
               overlap_factor=((total / a.active / 1e6) / clean) if clean else None,
               bubble_frac=(max(0.0, 1 - (total / a.active / 1e6) / clean)
                            if clean else None),
               overlap_gain=(max(0.0, (total / a.active / 1e6) / clean - 1)
                             if clean else None),
               by_category={k: v for k, v in sorted(bycat.items(), key=lambda kv: -kv[1]["us"])},
               top=rows[:20], info={k: v for k, v in info.items() if k != "args"})

    if TD.RANK == 0:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("=== %s%s (world=%d) ===" % (a.strategy, " +LoRA" if a.lora else "", TD.WORLD))
        for k, v in out["by_category"].items():
            print("  %-16s %6.2f%%  %8.2f ms  launches=%d"
                  % (k, v["share"] * 100, v["us"] / 1000, v["n"]))
        if clean:
            print("  kernel 自时间合计 %.1f ms/步；干净单步 %.1f ms；重叠系数 %.2f×"
                  % (out["cuda_busy_ms_per_step"], clean * 1000, out["overlap_factor"]))
            if out["overlap_factor"] >= 1:
                print("  → 通信与计算重叠，重叠量相当于单步的 %.1f%%" % (out["overlap_gain"] * 100))
            else:
                print("  → 未重叠满，GPU 真实空泡 %.1f%%" % (out["bubble_frac"] * 100))
        else:
            print("  kernel 自时间合计 %.1f ms/步（未提供干净单步，无法算重叠）"
                  % out["cuda_busy_ms_per_step"])
        print("saved", a.out)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
