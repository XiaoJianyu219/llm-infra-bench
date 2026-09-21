"""Day09 载体 A：Qwen3-0.6B 全量微调，六种并行策略同一套循环。

strategy ∈ {single, ddp, zero1, zero2, zero3, fsdp}

测量纪律沿用 Day 08：
  - 每个计时点前后 torch.cuda.synchronize()，分布式下再加 dist.barrier()
  - 冷启动逐步耗时曲线单独记录，按曲线决定丢几步（任务书 4.7）
  - repeats 个 block 取中位数，报 block 间标准差
  - 显存三档全记，且**按 rank 分别记录**后取最大值，并标注是否不对称（任务书 4.3）
  - 只有 rank 0 写文件（任务书 4.2）
  - OOM 捕获后如实落盘，不重试不降 batch
  - DeepSpeed 的三个 batch 字段按实际 world_size 覆盖，并把生效值写进结果（任务书 4.5）

合成数据：torch.randint 固定 seed，labels = input_ids。不用真实 dataloader。
"""
import argparse, json, os, sys, time, socket, statistics, traceback, subprocess, functools
import threading, contextlib
import torch
import torch.distributed as dist

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
VOCAB = 151936


# ---------------------------------------------------------------- 基础设施

def envi(k, d=0):
    return int(os.environ.get(k, d))


RANK, LOCAL_RANK, WORLD = envi("RANK"), envi("LOCAL_RANK"), envi("WORLD_SIZE", 1)


def p0(*a, **kw):
    if RANK == 0:
        print(*a, flush=True, **kw)


def smi_used(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        for line in out:
            i, v = line.split(",")
            if int(i) == idx:
                return int(v)
    except Exception:
        pass
    return None


class SmiSampler(threading.Thread):
    """后台采本卡的 nvidia-smi 显存。字段名不能叫 _stop —— Thread 自己有个同名
    私有方法，覆盖后 join() 内部会报 'Event' object is not callable（Day 08 踩过）。"""

    def __init__(self, idx, interval=0.2):
        super().__init__(daemon=True)
        self.idx, self.interval = idx, interval
        self.samples = []
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


def gather_obj(obj):
    if not dist.is_initialized():
        return [obj]
    buf = [None] * WORLD
    dist.all_gather_object(buf, obj)
    return buf


# ---------------------------------------------------------------- 模型

def build_model(dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    return AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=dtype)


def decoder_layer_cls():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    return Qwen3DecoderLayer


def apply_lora(model, args, info):
    """任务书 3.2 的 LoRA 重算 / 3.1 第 4 问要验证的那一支。

    LoRA 下冻结基座只有常驻参数、无梯度无优化器状态，ZeRO-1/2 切分的对象
    缩水到可训练参数那一小块，所以理论上它们几乎省不到显存 —— 这条要实测。
    """
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=[s for s in args.lora_target.split(",") if s])
    model = get_peft_model(model, cfg)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    info["lora"] = dict(r=args.lora_r, alpha=args.lora_alpha,
                        target_modules=args.lora_target,
                        trainable=tr, total=tot, trainable_ratio=tr / tot)
    return model


# ---------------------------------------------------------------- 各策略的搭建

def setup_single(model, args, info):
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    info["param_dtype"] = "fp32 (autocast bf16)"
    return model, opt, None


def setup_ddp(model, args, info):
    from torch.nn.parallel import DistributedDataParallel as DDP
    model.cuda().train()
    ddp = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
              gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    info["param_dtype"] = "fp32 (autocast bf16)"
    info["ddp_bucket_cap_mb"] = getattr(ddp, "bucket_bytes_cap", None)
    if info["ddp_bucket_cap_mb"]:
        info["ddp_bucket_cap_mb"] /= 2 ** 20
    # DDP 通信的是梯度，dtype 随参数 = fp32。体积记下来用于归因。
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["grad_comm_bytes_per_step"] = n * 4
    return ddp, opt, None


def setup_deepspeed(model, args, info, stage):
    import deepspeed
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "ds_zero%d.json" % stage)))
    # 任务书 4.5：三个字段必须自洽，按实际 world_size 覆盖后落盘生效值
    cfg["train_micro_batch_size_per_gpu"] = args.micro_bsz
    cfg["gradient_accumulation_steps"] = args.grad_accum
    cfg["train_batch_size"] = args.micro_bsz * args.grad_accum * WORLD
    info["ds_config_effective"] = {k: cfg[k] for k in
                                   ("train_micro_batch_size_per_gpu",
                                    "gradient_accumulation_steps",
                                    "train_batch_size")}
    info["ds_zero_stage"] = cfg["zero_optimization"]["stage"]
    engine, opt, _, _ = deepspeed.initialize(model=model, config=cfg,
                                             model_parameters=model.parameters())
    info["ds_version"] = deepspeed.__version__
    # 从 engine 反查生效值，而不是相信我自己写进去的（任务书 4.5）
    for k, fn in (("micro_batch_size_per_gpu", "train_micro_batch_size_per_gpu"),
                  ("gradient_accumulation_steps", "gradient_accumulation_steps"),
                  ("train_batch_size", "train_batch_size")):
        try:
            info["ds_engine_" + k] = getattr(engine, fn)()
        except Exception:
            pass
    info["param_dtype"] = "bf16 params + fp32 master (DeepSpeed)"
    return engine, opt, "deepspeed"


def setup_fsdp(model, args, info):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    # FSDP1 的 flat param 要求同一个 wrap 单元内 dtype 统一。LoRA 下基座是 bf16、
    # peft 新建的 adapter 是 fp32，直接 wrap 会报
    # "Must flatten tensors with uniform dtype but got torch.bfloat16 and torch.float32"。
    # 把 adapter 也转成 bf16 使 dtype 统一；代价是 torch AdamW 的主权重也变成 bf16，
    # 与 DeepSpeed（bf16 参数 + fp32 主权重）不是同一套精度口径，report 里要注明。
    if args.lora:
        n_cast = 0
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
                n_cast += 1
        info["fsdp_lora_cast_to_bf16"] = n_cast
    layer = decoder_layer_cls()
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={layer})
    mp = MixedPrecision(param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)
    wrapped = FSDP(model,
                   auto_wrap_policy=policy,
                   sharding_strategy=ShardingStrategy.FULL_SHARD,
                   mixed_precision=mp,
                   device_id=torch.device("cuda", LOCAL_RANK),
                   use_orig_params=True)
    # 任务书 4.4：必须验证包装粒度，不能假设默认合理
    n_fsdp = sum(1 for m in wrapped.modules() if isinstance(m, FSDP))
    info["fsdp_wrapped_modules"] = n_fsdp
    info["fsdp_expected_at_least"] = 1 + 28   # 根模块 + 28 个 Qwen3DecoderLayer
    info["fsdp_wrap_policy_ok"] = n_fsdp >= 29
    info["fsdp_layer_cls"] = layer.__name__
    info["param_dtype"] = "fp32 shard + bf16 compute (FSDP MixedPrecision)"
    wrapped.train()
    opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    return wrapped, opt, None


# ---------------------------------------------------------------- 主流程

def run(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(LOCAL_RANK)

    # 除 single 外都需要进程组。注意不能只按 WORLD>1 判断：
    # 单卡调通时用的是 torchrun --nproc_per_node=1，WORLD=1 但 DDP/ZeRO/FSDP
    # 仍然要求 init_process_group，否则报
    # "Default process group has not been initialized"。
    if (args.strategy != "single" or WORLD > 1) and not dist.is_initialized():
        dist.init_process_group("nccl")

    info = dict(strategy=args.strategy, world_size=WORLD, rank=RANK,
                host=socket.gethostname(), torch=torch.__version__,
                device=torch.cuda.get_device_name(LOCAL_RANK),
                args=vars(args))
    smi_base = smi_used(LOCAL_RANK)
    info["smi_baseline_mib"] = smi_base

    # LoRA 下基座用 bf16 常驻（否则基座 fp32 会掩盖 adapter 的那点差别）
    model = build_model(torch.bfloat16 if args.lora else torch.float32)
    if args.lora:
        model = apply_lora(model, args, info)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["params_total"] = n_total
    info["params_trainable"] = n_train
    info["trainable_ratio"] = n_train / n_total

    S = args.strategy
    if S == "single":
        net, opt, mode = setup_single(model, args, info)
    elif S == "ddp":
        net, opt, mode = setup_ddp(model, args, info)
    elif S in ("zero1", "zero2", "zero3"):
        net, opt, mode = setup_deepspeed(model, args, info, int(S[-1]))
    elif S == "fsdp":
        net, opt, mode = setup_fsdp(model, args, info)
    else:
        raise ValueError(S)
    is_ds = (mode == "deepspeed")

    g = torch.Generator(device="cpu").manual_seed(args.seed + RANK)
    ids = torch.randint(0, VOCAB, (args.micro_bsz, args.seq_len),
                        generator=g).cuda()
    labels = ids.clone()

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    nullamp = torch.autocast("cuda", enabled=False)
    # DeepSpeed / FSDP 自己管混合精度，外面再套 autocast 会重复转换
    ctx = nullamp if (is_ds or S == "fsdp") else amp

    # DDP 下梯度累积必须用 no_sync() 把中间 micro-step 的 all-reduce 关掉，
    # 否则每个 micro-step 都会通信一次，通信量被乘以 grad_accum，
    # 与 DeepSpeed（自己只在累积边界通信）就不是同一个口径了。
    has_no_sync = hasattr(net, "no_sync")

    def maybe_no_sync(last):
        if last or not has_no_sync:
            return contextlib.nullcontext()
        return net.no_sync()

    def one_step():
        if is_ds:
            loss = None
            for _ in range(args.grad_accum):
                with ctx:
                    loss = net(input_ids=ids, labels=labels, use_cache=False).loss
                net.backward(loss)
                net.step()          # DS 自己判断是否到累积边界
            return loss
        opt.zero_grad(set_to_none=True)
        loss = None
        for i in range(args.grad_accum):
            with maybe_no_sync(i == args.grad_accum - 1):
                with ctx:
                    loss = net(input_ids=ids, labels=labels, use_cache=False).loss
                (loss / args.grad_accum).backward()
        opt.step()
        return loss

    rec = dict(info)
    try:
        # ---- 冷启动曲线 ----
        curve = []
        loss = None
        for _ in range(max(args.curve, args.warmup)):
            barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = one_step()
            torch.cuda.synchronize()
            barrier()
            curve.append(time.perf_counter() - t0)
        rec["cold_steps_sec"] = curve
        rec["loss_after_warmup"] = float(loss.detach()) if loss is not None else None

        # ---- 正式测量 ----
        torch.cuda.reset_peak_memory_stats()
        sampler = SmiSampler(LOCAL_RANK)   # 后台采样，不挤进计时循环
        sampler.start()
        blocks, all_steps = [], []
        for _ in range(args.repeats):
            ts = []
            for _ in range(args.steps):
                barrier()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                one_step()
                torch.cuda.synchronize()
                barrier()
                ts.append(time.perf_counter() - t0)
            blocks.append(statistics.median(ts))
            all_steps += ts
        smi_peak = sampler.stop() or smi_base
        step = statistics.median(blocks)
        rec.update(
            block_medians_sec=blocks, all_steps_sec=all_steps,
            step_sec_median=step,
            step_sec_std_across_blocks=statistics.pstdev(blocks) if len(blocks) > 1 else 0.0,
            mem_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
            mem_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
            smi_peak_mib=smi_peak,
            tokens_per_step_per_rank=args.micro_bsz * args.seq_len * args.grad_accum,
            status="ok")
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="OOM", oom_message=str(e).split("\n")[0],
                   oom_step_index=len(rec.get("cold_steps_sec", [])),
                   mem_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                   mem_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
    except Exception as e:
        rec.update(status="ERROR", error=repr(e), traceback=traceback.format_exc())
        print("[rank %d] %s" % (RANK, rec["traceback"]), flush=True)

    # ---- 按 rank 汇总（任务书 4.3）----
    per_rank = gather_obj({k: rec.get(k) for k in
                           ("rank", "status", "step_sec_median", "mem_allocated_mib",
                            "mem_reserved_mib", "smi_peak_mib", "loss_after_warmup")})
    if RANK == 0:
        out = dict(rec)
        out["per_rank"] = per_rank
        oks = [r for r in per_rank if r.get("status") == "ok"]
        if oks and rec["status"] == "ok":
            allocs = [r["mem_allocated_mib"] for r in oks]
            out["mem_allocated_max_mib"] = max(allocs)
            out["mem_allocated_min_mib"] = min(allocs)
            out["mem_asymmetry_pct"] = (max(allocs) - min(allocs)) / max(allocs) * 100
            out["total_tokens_per_sec"] = (WORLD * rec["tokens_per_step_per_rank"]
                                           / rec["step_sec_median"])
            out["tokens_per_sec_per_rank"] = (rec["tokens_per_step_per_rank"]
                                              / rec["step_sec_median"])
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        if out["status"] == "ok":
            print("[%s w%d] step %.1f ms (std %.1f) total %.0f tok/s | "
                  "alloc max %.0f min %.0f MiB (不对称 %.2f%%) resv %.0f smi %s"
                  % (args.strategy, WORLD, out["step_sec_median"] * 1e3,
                     out["step_sec_std_across_blocks"] * 1e3,
                     out["total_tokens_per_sec"], out["mem_allocated_max_mib"],
                     out["mem_allocated_min_mib"], out["mem_asymmetry_pct"],
                     out["mem_reserved_mib"], out["smi_peak_mib"]), flush=True)
        else:
            print("[%s w%d] %s" % (args.strategy, WORLD, out["status"]), flush=True)
        print("saved", args.out, flush=True)

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True,
                   choices=["single", "ddp", "zero1", "zero2", "zero3", "fsdp"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bsz", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--curve", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scaling", default="weak", choices=["weak", "strong", "na"],
                   help="只记录，不改变行为；由启动脚本决定 micro_bsz 怎么取")
    p.add_argument("--lora", action="store_true",
                   help="LoRA 模式：基座 bf16 冻结，只训 adapter（验证预测 4）")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
