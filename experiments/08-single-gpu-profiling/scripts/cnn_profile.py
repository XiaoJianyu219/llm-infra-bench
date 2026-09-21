"""Day08 4.6 可选对照：CNN 检测模型的训练瓶颈画像，与 transformer 对比。

用 Day05 那个模型 /root/autodl-tmp/sod/best.pt（ultralytics YOLOv8n 系）。
优先走 ultralytics 自己的训练 loss（含 TAL 分配器）；拿不到就退回
"输出求和"作为反传起点，并在结果里标明这一点 —— 这会漏掉分配器那部分
CPU 开销，对比时必须说清楚。
"""
import argparse, json, os, sys, time, statistics, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_run import categorize, split_rows, by_category

WEIGHTS = "/root/autodl-tmp/sod/best.pt"


def build(imgsz, batch, device="cuda"):
    from ultralytics import YOLO
    y = YOLO(WEIGHTS)
    m = y.model.to(device).float().train()
    for p in m.parameters():
        p.requires_grad_(True)
    g = torch.Generator().manual_seed(42)
    img = torch.rand(batch, 3, imgsz, imgsz, generator=g).to(device)

    # 试真 loss
    mode = "ultralytics_loss"
    nb = 4 * batch
    batch_dict = dict(
        img=img,
        cls=torch.randint(0, max(1, int(getattr(m, "nc", 4))), (nb, 1)).float().to(device),
        bboxes=(torch.rand(nb, 4, generator=g) * 0.4 + 0.3).to(device),
        batch_idx=torch.arange(nb).to(device) % batch,
    )
    try:
        # DetectionModel.loss(batch) 才是训练路径（含 TAL 分配器）；
        # 直接 m(batch_dict) 在 ultralytics 8.3 上走不通，先 dir() 确认过再写
        assert hasattr(m, "loss"), "no .loss on model"
        out = m.loss(batch_dict)
        loss = out[0].sum() if isinstance(out, (tuple, list)) else out.sum()
        float(loss.detach())

        def step_fn():
            o = m.loss(batch_dict)
            return (o[0].sum() if isinstance(o, (tuple, list)) else o.sum())
    except Exception as e:
        mode = "sum_of_outputs (fallback: %s)" % type(e).__name__

        def step_fn():
            o = m(img)
            if isinstance(o, (list, tuple)):
                flat = []
                for x in o:
                    if isinstance(x, (list, tuple)):
                        flat += [t for t in x if torch.is_tensor(t)]
                    elif torch.is_tensor(x):
                        flat.append(x)
                return sum(t.float().sum() for t in flat)
            return o.float().sum()
    return m, step_fn, mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--active", type=int, default=3)
    p.add_argument("--tag", default="cnn_train")
    a = p.parse_args()

    torch.manual_seed(42)
    rec = dict(args=vars(a), weights=WEIGHTS)
    try:
        m, step_fn, mode = build(a.imgsz, a.batch)
        rec["loss_mode"] = mode
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def one_step():
            opt.zero_grad(set_to_none=True)
            loss = step_fn()
            loss.backward()
            opt.step()

        # --- 计时（不挂 profiler，纪律同 transformer 侧） ---
        for _ in range(a.warmup):
            one_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        blocks = []
        for _ in range(a.repeats):
            ts = []
            for _ in range(a.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                one_step()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            blocks.append(statistics.median(ts))
        rec["step_sec_median"] = statistics.median(blocks)
        rec["step_sec_std_across_blocks"] = statistics.pstdev(blocks)
        rec["images_per_sec"] = a.batch / rec["step_sec_median"]
        rec["mem_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rec["mem_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20

        # --- profiler（另起一次，只用于定位） ---
        from torch.profiler import profile, ProfilerActivity, schedule
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=schedule(wait=0, warmup=1, active=a.active, repeat=1),
                     record_shapes=True, with_flops=True) as prof:
            for _ in range(1 + a.active):
                one_step()
                torch.cuda.synchronize()
                prof.step()
        rows, ops = split_rows(prof.key_averages())   # 只取 device_type==CUDA 的 kernel 行
        tot = sum(r["self_cuda_us"] for r in rows)
        bycat = by_category(rows)
        gl = {"gemm", "conv", "attention_flash", "attention_mem_efficient"}
        f_us = sum(r["self_cuda_us"] for r in ops if r["flops"] and r["cat"] in gl)
        f_fl = sum(r["flops"] for r in ops if r["flops"] and r["cat"] in gl)
        rec["gemm_achieved_tflops"] = (f_fl / (f_us * 1e-6) / 1e12) if f_us else None
        rec["total_self_cuda_us"] = tot
        rec["n_distinct_kernels"] = len(rows)
        rec["n_kernel_launches_per_step"] = sum(r["count"] for r in rows) / a.active
        rec["top"] = rows[:20]
        rec["by_category"] = bycat
        rec["status"] = "ok"

        print("模式:", mode)
        print("step %.1f ms  img/s %.1f  alloc %.0f MiB  resv %.0f MiB"
              % (rec["step_sec_median"] * 1000, rec["images_per_sec"],
                 rec["mem_allocated_mib"], rec["mem_reserved_mib"]))
        print("不同 kernel 数 %d，每步 launch 约 %.0f 次；conv/GEMM 类实测 %s TFLOPS"
              % (rec["n_distinct_kernels"], rec["n_kernel_launches_per_step"],
                 ("%.1f" % rec["gemm_achieved_tflops"]) if rec["gemm_achieved_tflops"] else "n/a"))
        print("\n分类占比：")
        for c, d in sorted(bycat.items(), key=lambda kv: -kv[1]["us"]):
            print("  %-18s %6.2f%%  %8.2f ms  kernels=%d" % (c, d["share"] * 100, d["us"] / 1000, d["n"]))
        print("\nTop 10:")
        for r in rows[:10]:
            print("  %6.2f%%  %-16s %s" % (r["self_cuda_us"] / tot * 100, r["cat"], r["name"][:64]))
    except Exception as e:
        rec["status"] = "ERROR"
        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()
        print(rec["traceback"])

    os.makedirs("results", exist_ok=True)
    json.dump(rec, open("results/%s.json" % a.tag, "w"), indent=1)
    print("saved results/%s.json" % a.tag)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
