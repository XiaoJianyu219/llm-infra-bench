"""Day08 4.3 profiler：定位瓶颈，不用来计时。

任务书 5.1：profiler 有开销，报告里的吞吐/延迟必须来自不挂 profiler 的 train_loop.py。
本脚本只回答两件事：
  1) CUDA 时间花在哪些算子上（Top N + 分类占比）
  2) 实际用的是哪个 SDPA 后端（看 kernel 名字，不看我指定了什么）

两个必须踩对的坑（第一版都踩错了）：
  a) key_averages() 里同时有 CPU 侧的 aten 算子行和 GPU 侧的 kernel 行，
     两者的 self_device_time_total 是同一段时间的两次记账。做时间构成只能取
     device_type == CUDA 的行，否则每项都被算两遍。
  b) FLOPs 只挂在 CPU 侧的 aten 行上（kernel 行的 flops 恒为 0）。
     所以"算子达到多少 TFLOPS"要从 aten 行取，不能从 kernel 行取。
  c) ProfilerStep* 和 Optimizer.step#AdamW.step 这类 user annotation 也标成
     device_type==CUDA，但它们是区间标注不是 kernel，必须按 is_user_annotation 排掉。
"""
import argparse, json, os, sys, time, re
import torch
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity, schedule
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_flops import model_flops, QWEN3_06B
import train_loop as TL

# 顺序敏感：先匹配更具体的。注意 fmha_cutlassF/B 是 mem_efficient 后端
# （xformers 那套 cutlass 实现），不是 flash —— 第一版把它标成 flash 了。
CATEGORIES = [
    ("attention_mem_efficient", r"fmha_cutlass|cutlassF|cutlassB|efficient_attention|mem_eff"),
    ("attention_flash", r"flash_fwd|flash_bwd|pytorch_flash|flash::|flash_attn"),
    ("attention_cudnn", r"cudnn.*(attn|mha)|(attn|mha).*cudnn"),
    ("conv", r"conv|winograd|implicit_gemm|nchwToNhwc|nhwcToNchw|scudnn|xmma_"),
    ("batchnorm", r"bn_bw|bn_fw|batch_norm|batchnorm"),
    ("gemm", r"gemm|cutlass|simt|ampere_|turing_|volta_|s16816|nn_align|tn_align|\bmm\b|addmm|bmm|dot_kernel"),
    ("softmax", r"softmax"),
    ("optimizer", r"adamw|multi_tensor|foreach|fused_adam"),
    ("reduction", r"reduce_kernel|norm_kernel|\bsum\b|Reduce"),
    ("elementwise", r"elementwise|fill|copy|cast|silu|\bmul\b|\badd\b|Memset|memcpy"),
    ("index_embed", r"index|embedding|gather|scatter|nll|cross_entropy"),
]


def categorize(name):
    for cat, pat in CATEGORIES:
        if re.search(pat, name, re.IGNORECASE):
            return cat
    return "other"


def split_rows(key_averages):
    """返回 (kernel_rows, op_rows)。kernel_rows 用于时间构成，op_rows 用于 FLOPs。"""
    kernels, ops = [], []
    for e in key_averages:
        us = float(getattr(e, "self_device_time_total", 0) or 0)
        if us <= 0:
            continue
        # user annotation（ProfilerStep*、Optimizer.step#AdamW.step 等）也带
        # device_type==CUDA，但它是覆盖一段区间的标注，不是 kernel；
        # 不排掉的话它下面的 kernel 会被再记一遍。
        if getattr(e, "is_user_annotation", False):
            continue
        rec = dict(name=e.key, self_cuda_us=us, count=int(e.count),
                   flops=float(getattr(e, "flops", 0) or 0), cat=categorize(e.key))
        if getattr(e, "device_type", None) == DeviceType.CUDA:
            kernels.append(rec)
        else:
            ops.append(rec)
    kernels.sort(key=lambda r: -r["self_cuda_us"])
    ops.sort(key=lambda r: -r["self_cuda_us"])
    return kernels, ops


def by_category(rows):
    d = {}
    tot = sum(r["self_cuda_us"] for r in rows) or 1.0
    for r in rows:
        x = d.setdefault(r["cat"], dict(us=0.0, n=0))
        x["us"] += r["self_cuda_us"]
        x["n"] += r["count"]
    for x in d.values():
        x["share"] = x["us"] / tot
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--sdpa", default="flash")
    p.add_argument("--ckpt", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--active", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="prof")
    p.add_argument("--trace", type=int, default=0)
    a = p.parse_args()

    class A:
        pass
    args = A()
    for k, v in vars(a).items():
        setattr(args, k, v)

    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = TL.build(args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    ids = torch.randint(0, QWEN3_06B["vocab_size"], (a.batch_size, a.seq_len),
                        generator=g).to("cuda")
    labels = ids.clone()
    amp = a.precision == "bf16"
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if amp
                else torch.autocast("cuda", enabled=False))

    def one_step():
        opt.zero_grad(set_to_none=True)
        with autocast, TL.sdpa_ctx(a.sdpa):
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        opt.step()

    for _ in range(a.warmup):
        one_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=1, active=a.active, repeat=1),
                 record_shapes=True, profile_memory=True,
                 with_flops=True, with_stack=False) as prof:
        for _ in range(1 + a.active):
            one_step()
            torch.cuda.synchronize()
            prof.step()
    t_prof = time.perf_counter() - t0

    ka = prof.key_averages()
    kernels, ops = split_rows(ka)
    total_cuda = sum(r["self_cuda_us"] for r in kernels)
    kcat = by_category(kernels)

    # FLOPs 侧：从 aten 行取，算"GEMM 类算子自己跑到多少 TFLOPS"
    gemm_like = {"gemm", "attention_flash", "attention_mem_efficient", "attention_cudnn", "conv"}
    f_us = sum(r["self_cuda_us"] for r in ops if r["flops"] and r["cat"] in gemm_like)
    f_fl = sum(r["flops"] for r in ops if r["flops"] and r["cat"] in gemm_like)

    attn = [r for r in kernels if r["cat"].startswith("attention")]
    backend_seen = sorted({r["cat"] for r in attn}) or ["none (math 路径无专用 attention kernel)"]

    out = dict(config=vars(a), total_self_cuda_us=total_cuda,
               profiled_wall_sec=t_prof, active_steps=a.active,
               cuda_busy_us_per_step=total_cuda / a.active,
               n_kernel_launches_per_step=sum(r["count"] for r in kernels) / a.active,
               n_distinct_kernels=len(kernels),
               top=kernels[:25], by_category=kcat,
               flops_rows=[r for r in ops if r["flops"]][:25],
               gemm_flops_total=f_fl, gemm_flops_us=f_us,
               gemm_achieved_tflops=(f_fl / (f_us * 1e-6) / 1e12) if f_us else None,
               sdpa_backend_kernels=[r["name"] for r in attn][:8],
               sdpa_backend_seen=backend_seen)

    os.makedirs("results/profiles", exist_ok=True)
    json.dump(out, open("results/profiles/%s.json" % a.tag, "w"), indent=1)
    with open("results/profiles/%s_table.txt" % a.tag, "w") as f:
        f.write(ka.table(sort_by="self_device_time_total", row_limit=30))
    if a.trace:
        prof.export_chrome_trace("results/profiles/%s_trace.json" % a.tag)

    print("=== %s : Top 12 CUDA kernel（只取 device_type==CUDA 的行）===" % a.tag)
    for r in kernels[:12]:
        print("%7.2f%%  %8.2f ms  n=%-5d %-24s %s"
              % (r["self_cuda_us"] / total_cuda * 100, r["self_cuda_us"] / 1000,
                 r["count"], r["cat"], r["name"][:62]))
    print("\n=== 分类占比（kernel 侧，合计 100%%）===")
    for c, d in sorted(kcat.items(), key=lambda kv: -kv[1]["us"]):
        print("%-24s %6.2f%%  %9.2f ms  launches=%d" % (c, d["share"] * 100, d["us"] / 1000, d["n"]))
    print("\nSDPA 实测后端(kernel 名):", backend_seen, "->", out["sdpa_backend_kernels"][:2])
    if f_us:
        print("GEMM 类算子实测 %.1f TFLOPS（FLOPs 取自 aten 行）" % out["gemm_achieved_tflops"])
    print("每步 kernel launch 约 %.0f 次，不同 kernel %d 种"
          % (out["n_kernel_launches_per_step"], out["n_distinct_kernels"]))
    print("总 CUDA self time %.1f ms / %d 步 = %.1f ms/步（含 profiler 开销，不用于报告计时）"
          % (total_cuda / 1000, a.active, total_cuda / a.active / 1000))

    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
