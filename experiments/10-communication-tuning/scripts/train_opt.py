"""Day10 载体 A（Qwen3-0.6B 全量微调）+ 所有待扫的开关，每次只改一个。

沿用 Day 08/09 的测量纪律：
  - 每个计时点前后 torch.cuda.synchronize()，分布式下外加 dist.barrier()
  - 冷启动曲线单独记，按曲线定 warmup（任务书 4.5）
  - repeats 个 block 取中位数，报 block 间标准差与极差（噪声地板要用）
  - 显存三档全记，按 rank 分别记录后取最大
  - 只有 rank 0 写文件
  - torch.compile 的首次编译计入 warmup 而非 step time（任务书 4.6）
  - 吞吐一律按 tokens/s 报，梯度累积改变有效 batch 时不会给出相反结论（任务书 4.2）

MFU 口径（任务书 3.2，aggregate）：
    MFU = 每 step 总 model FLOPs / (step 时间 × 卡数 × 单卡实测 GEMM 峰值)
"""
import argparse, json, os, sys, time, socket, statistics, traceback, subprocess
import contextlib, functools, threading
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_flops import model_flops, QWEN3_06B

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
VOCAB = QWEN3_06B["vocab_size"]


def envi(k, d=0):
    return int(os.environ.get(k, d))


RANK, LOCAL_RANK, WORLD = envi("RANK"), envi("LOCAL_RANK"), envi("WORLD_SIZE", 1)


def smi_used(idx):
    try:
        for line in subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"], text=True).strip().splitlines():
            i, v = line.split(",")
            if int(i) == idx:
                return int(v)
    except Exception:
        pass
    return None


class SmiSampler(threading.Thread):
    """字段不能叫 _stop：Thread 有同名私有方法（Day 08 踩过）。"""

    def __init__(self, idx, interval=0.2):
        super().__init__(daemon=True)
        self.idx, self.interval, self.samples = idx, interval, []
        self._stopev = threading.Event()

    def run(self):
        while not self._stopev.is_set():
            v = smi_used(self.idx)
            if v:
                self.samples.append(v)
            self._stopev.wait(self.interval)

    def stop(self):
        self._stopev.set()
        self.join(timeout=10)
        return max(self.samples) if self.samples else None


def barrier():
    if dist.is_initialized():
        dist.barrier()


def gather_obj(o):
    if not dist.is_initialized():
        return [o]
    buf = [None] * WORLD
    dist.all_gather_object(buf, o)
    return buf


def build(args, info):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    m = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.float32)
    m.to("cuda").train()

    if args.compile:
        t0 = time.perf_counter()
        m = torch.compile(m)
        info["compile_wrap_sec"] = time.perf_counter() - t0
        info["compile_mode"] = "default"

    if args.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        kw = dict(device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
                  gradient_as_bucket_view=bool(args.grad_as_bucket_view),
                  find_unused_parameters=bool(args.find_unused))
        if args.bucket_cap_mb > 0:
            kw["bucket_cap_mb"] = args.bucket_cap_mb
        if args.static_graph:
            kw["static_graph"] = True
        m = DDP(m, **kw)
        info["ddp_kwargs"] = {k: v for k, v in kw.items() if k != "device_ids"}
        # 反查真实生效的 bucket 上限，不相信我传进去的值
        for attr in ("bucket_bytes_cap", "bucket_bytes_cap_default"):
            if hasattr(m, attr):
                info["ddp_" + attr + "_MB"] = getattr(m, attr) / 2 ** 20
    return m


def make_opt(model, args, info):
    kw = dict(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    if args.fused_adam:
        kw["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), **kw)
    info["optimizer_fused"] = bool(args.fused_adam)
    info["optimizer_kwargs"] = {k: str(v) for k, v in kw.items()}
    return opt


def run(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(LOCAL_RANK)

    # TF32 开关必须在建图前设（任务书 3.5 单卡侧）
    if args.tf32 >= 0:
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    if args.strategy != "single" or WORLD > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")

    info = dict(strategy=args.strategy, world_size=WORLD, rank=RANK,
                host=socket.gethostname(), torch=torch.__version__,
                device=torch.cuda.get_device_name(LOCAL_RANK), args=vars(args),
                smi_baseline_mib=smi_used(LOCAL_RANK))
    try:
        info["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    except Exception:
        pass
    # 记录本进程实际看到的 NCCL 环境变量（任务书 4.1：改了没生效和改了没用是两回事）
    info["nccl_env"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}

    model = build(args, info)
    opt = make_opt(model, args, info)
    n_total = sum(p.numel() for p in model.parameters())
    info["params_total"] = n_total
    info["grad_comm_bytes_per_optimizer_step"] = n_total * 4  # fp32 梯度

    g = torch.Generator(device="cpu").manual_seed(args.seed + RANK)
    ids = torch.randint(0, VOCAB, (args.micro_bsz, args.seq_len), generator=g).cuda()
    labels = ids.clone()
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16"
                else torch.autocast("cuda", enabled=False))

    has_no_sync = hasattr(model, "no_sync")

    def maybe_no_sync(last):
        # DDP + 梯度累积必须 no_sync，否则每个 micro-step 都通信，
        # 通信量被乘以 grad_accum，整个实验就白做了
        return contextlib.nullcontext() if (last or not has_no_sync) else model.no_sync()

    def one_step():
        opt.zero_grad(set_to_none=bool(args.set_to_none))
        loss = None
        for i in range(args.grad_accum):
            with maybe_no_sync(i == args.grad_accum - 1):
                with autocast:
                    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
                (loss / args.grad_accum).backward()
        opt.step()
        return loss

    rec = dict(info)
    try:
        # ---- 冷启动。compile 的首次编译在这里发生，绝不能算进 step time ----
        t_cold0 = time.perf_counter()
        curve, loss = [], None
        for _ in range(max(args.curve, args.warmup)):
            barrier(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = one_step()
            torch.cuda.synchronize(); barrier()
            curve.append(time.perf_counter() - t0)
        rec["cold_steps_sec"] = curve
        rec["cold_total_sec"] = time.perf_counter() - t_cold0
        rec["loss_after_warmup"] = float(loss.detach()) if loss is not None else None

        # ---- 正式测量 ----
        torch.cuda.reset_peak_memory_stats()
        sampler = SmiSampler(LOCAL_RANK); sampler.start()
        blocks, all_steps = [], []
        for _ in range(args.repeats):
            ts = []
            for _ in range(args.steps):
                barrier(); torch.cuda.synchronize()
                t0 = time.perf_counter()
                one_step()
                torch.cuda.synchronize(); barrier()
                ts.append(time.perf_counter() - t0)
            blocks.append(statistics.median(ts))
            all_steps += ts
        smi_peak = sampler.stop() or info["smi_baseline_mib"]

        step = statistics.median(blocks)
        rec.update(
            block_medians_sec=blocks, all_steps_sec=all_steps,
            step_sec_median=step,
            step_sec_std_across_blocks=statistics.pstdev(blocks) if len(blocks) > 1 else 0.0,
            step_sec_std_across_steps=statistics.pstdev(all_steps),
            step_rel_std_blocks=(statistics.pstdev(blocks) / step) if len(blocks) > 1 else 0.0,
            step_rel_range_blocks=((max(blocks) - min(blocks)) / step) if len(blocks) > 1 else 0.0,
            mem_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
            mem_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
            smi_peak_mib=smi_peak,
            tokens_per_optimizer_step_per_rank=args.micro_bsz * args.seq_len * args.grad_accum,
            status="ok")
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="OOM", oom_message=str(e).split("\n")[0],
                   mem_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                   mem_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
    except Exception as e:
        rec.update(status="ERROR", error=repr(e), traceback=traceback.format_exc())
        print("[rank %d]\n%s" % (RANK, rec["traceback"]), flush=True)

    per_rank = gather_obj({k: rec.get(k) for k in
                           ("rank", "status", "step_sec_median", "mem_allocated_mib",
                            "mem_reserved_mib", "smi_peak_mib", "loss_after_warmup")})

    if RANK == 0:
        out = dict(rec, per_rank=per_rank)
        if rec["status"] == "ok":
            oks = [r for r in per_rank if r.get("status") == "ok"]
            allocs = [r["mem_allocated_mib"] for r in oks] or [rec["mem_allocated_mib"]]
            out["mem_allocated_max_mib"] = max(allocs)
            out["mem_asymmetry_pct"] = (max(allocs) - min(allocs)) / max(allocs) * 100
            toks = WORLD * rec["tokens_per_optimizer_step_per_rank"]
            out["tokens_per_optimizer_step_total"] = toks
            out["tokens_per_sec"] = toks / rec["step_sec_median"]
            f1 = model_flops(QWEN3_06B, args.seq_len, include_lm_head=False)
            f2 = model_flops(QWEN3_06B, args.seq_len, include_lm_head=True)
            out["flops_per_token"] = f1["total_per_token"]
            out["achieved_tflops_aggregate"] = (out["tokens_per_sec"]
                                                * f1["total_per_token"] / 1e12)
            if args.peak_tflops:
                den = args.peak_tflops * WORLD
                out["peak_tflops_per_gpu"] = args.peak_tflops
                out["MFU"] = out["achieved_tflops_aggregate"] / den
                out["MFU_with_head"] = (out["tokens_per_sec"] * f2["total_per_token"]
                                        / 1e12 / den)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        if out["status"] == "ok":
            print("[%s] step %.1f ms (std %.2f, 极差 %.2f%%) %.0f tok/s  MFU %s  "
                  "alloc %.0f resv %.0f smi %s"
                  % (args.tag, out["step_sec_median"] * 1e3,
                     out["step_sec_std_across_blocks"] * 1e3,
                     out["step_rel_range_blocks"] * 100, out["tokens_per_sec"],
                     ("%.2f%%" % (out["MFU"] * 100)) if args.peak_tflops else "n/a",
                     out["mem_allocated_max_mib"], out["mem_reserved_mib"],
                     out["smi_peak_mib"]), flush=True)
        else:
            print("[%s] %s" % (args.tag, out["status"]), flush=True)
        print("saved", args.out, flush=True)

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="ddp", choices=["single", "ddp"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bsz", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--curve", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peak-tflops", type=float, default=0.0)
    # 通信侧
    p.add_argument("--bucket-cap-mb", type=int, default=0, help="0 = 用 DDP 默认 25")
    p.add_argument("--grad-as-bucket-view", type=int, default=1)
    p.add_argument("--static-graph", type=int, default=0)
    p.add_argument("--find-unused", type=int, default=0)
    # 单卡侧
    p.add_argument("--fused-adam", type=int, default=0)
    p.add_argument("--set-to-none", type=int, default=1)
    p.add_argument("--tf32", type=int, default=-1, help="-1 保持 torch 默认")
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
