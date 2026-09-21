"""Day08 4.1 可复现的最小训练循环。不用 HF Trainer，每一步都可控。

计时纪律（沿用 Day01/Day03）：
  - 每个计时点前 torch.cuda.synchronize()，否则量到的是下发时间
  - 丢弃前 --warmup 步（默认 10，依据见 results/step_curve.json）
  - --repeats 个 block，每 block --steps 步；报中位数与 block 间标准差
  - 显存三档全记：max_memory_allocated / max_memory_reserved / nvidia-smi
  - OOM 捕获并如实记录，不静默降 batch
"""
import argparse, json, os, sys, time, threading, subprocess, statistics, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_flops import model_flops, QWEN3_06B

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"


def smi_used():
    v = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()[0]
    return int(v)


class SmiSampler(threading.Thread):
    """后台按 100 ms 采 nvidia-smi 的 memory.used（含 CUDA context，与 torch 口径不同）。

    注意：字段名不能叫 `_stop`。threading.Thread 自己有一个内部方法 `_stop()`，
    实例属性会把它盖掉，join() 内部调用 self._stop() 时就报
    "'Event' object is not callable"，而且是在 finally 里报，正好吞掉整轮结果。
    """

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stopev = threading.Event()

    def run(self):
        while not self._stopev.is_set():
            try:
                self.samples.append(smi_used())
            except Exception:
                pass
            self._stopev.wait(self.interval)

    def stop(self):
        self._stopev.set()
        self.join(timeout=10)
        return dict(max_mib=max(self.samples) if self.samples else None,
                    min_mib=min(self.samples) if self.samples else None,
                    n=len(self.samples))


def build(args):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    # 训练用 fp32 权重。AMP 只改算子精度，不改参数精度（任务书 5.6）
    model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.float32)
    model.to("cuda").train()
    if args.ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def sdpa_ctx(name):
    import contextlib
    from torch.nn.attention import sdpa_kernel, SDPBackend
    table = {"math": SDPBackend.MATH,
             "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
             "flash": SDPBackend.FLASH_ATTENTION,
             "cudnn": SDPBackend.CUDNN_ATTENTION}
    if name == "default":
        return contextlib.nullcontext()
    return sdpa_kernel(table[name])


def run(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    smi_base = smi_used()

    model = build(args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    ids = torch.randint(0, QWEN3_06B["vocab_size"],
                        (args.batch_size, args.seq_len), generator=g).to("cuda")
    labels = ids.clone()

    amp = args.precision == "bf16"
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if amp
                else torch.autocast("cuda", enabled=False))

    def one_step():
        opt.zero_grad(set_to_none=True)
        with autocast, sdpa_ctx(args.sdpa):
            out = model(input_ids=ids, labels=labels, use_cache=False)
            loss = out.loss
        loss.backward()
        opt.step()
        return loss

    rec = dict(args=vars(args), smi_baseline_mib=smi_base,
               torch=torch.__version__, device=torch.cuda.get_device_name(0))
    sampler = SmiSampler()
    sampler.start()
    try:
        # ---- 冷启动曲线：前 n_cold 步逐步耗时，用来决定丢几步 ----
        curve = []
        n_cold = max(args.curve, args.warmup)
        loss = None
        for _ in range(n_cold):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = one_step()
            torch.cuda.synchronize()
            curve.append(time.perf_counter() - t0)
        rec["cold_steps_sec"] = curve
        rec["warmup_discarded"] = args.warmup
        rec["loss_after_warmup"] = float(loss.detach()) if loss is not None else None

        # ---- 正式测量：重置峰值统计，只统计稳态 ----
        torch.cuda.reset_peak_memory_stats()
        blocks, all_steps = [], []
        for _ in range(args.repeats):
            ts = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                one_step()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            blocks.append(statistics.median(ts))
            all_steps += ts
        rec["block_medians_sec"] = blocks
        rec["all_steps_sec"] = all_steps
        step = statistics.median(blocks)
        rec["step_sec_median"] = step
        rec["step_sec_std_across_blocks"] = statistics.pstdev(blocks) if len(blocks) > 1 else 0.0
        rec["step_sec_std_across_steps"] = statistics.pstdev(all_steps)
        toks = args.batch_size * args.seq_len
        rec["tokens_per_step"] = toks
        rec["tokens_per_sec"] = toks / step
        rec["mem_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rec["mem_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
        rec["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "OOM"
        rec["oom_message"] = str(e).split("\n")[0]
        rec["oom_at_phase"] = "cold" if "cold_steps_sec" not in rec else "measure"
        rec["oom_step_index"] = len(rec.get("cold_steps_sec", []))
        rec["mem_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rec["mem_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
        print("OOM:", rec["oom_message"])
    except Exception as e:
        rec["status"] = "ERROR"
        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()
        print(rec["traceback"])
    finally:
        rec["smi"] = sampler.stop()

    if rec["status"] == "ok":
        f1 = model_flops(QWEN3_06B, args.seq_len, include_lm_head=False)
        f2 = model_flops(QWEN3_06B, args.seq_len, include_lm_head=True)
        step = rec["step_sec_median"]
        toks = rec["tokens_per_step"]
        mflops = f1["total_per_token"] * toks
        mflops2 = f2["total_per_token"] * toks
        # HFU：checkpointing 下硬件多跑一次 forward。fwd=1 单位、bwd=2 单位 → 3 变 4
        recompute = 4.0 / 3.0 if args.ckpt else 1.0
        rec["flops_per_token_brief"] = f1["total_per_token"]
        rec["flops_per_token_with_head"] = f2["total_per_token"]
        rec["model_tflops_achieved"] = mflops / step / 1e12
        rec["model_tflops_achieved_with_head"] = mflops2 / step / 1e12
        rec["hardware_tflops_achieved"] = mflops * recompute / step / 1e12
        rec["recompute_factor"] = recompute
        if args.peak_tflops:
            rec["peak_tflops_used"] = args.peak_tflops
            rec["MFU"] = rec["model_tflops_achieved"] / args.peak_tflops
            rec["MFU_with_head"] = rec["model_tflops_achieved_with_head"] / args.peak_tflops
            rec["HFU"] = rec["hardware_tflops_achieved"] / args.peak_tflops

    del model, opt
    torch.cuda.empty_cache()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--sdpa", choices=["default", "math", "mem_efficient", "flash", "cudnn"],
                   default="flash")
    p.add_argument("--ckpt", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--curve", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peak-tflops", type=float, default=0.0)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="")
    a = p.parse_args()
    rec = run(a)
    rec["tag"] = a.tag
    out = a.out or ("results/runs/%s.json" % a.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1)
    if rec["status"] == "ok":
        msg = ("[%s] step %.1f ms (std_blocks %.1f ms) tok/s %.0f alloc %.0f resv %.0f smi %s MiB"
               % (a.tag, rec["step_sec_median"] * 1000,
                  rec["step_sec_std_across_blocks"] * 1000, rec["tokens_per_sec"],
                  rec["mem_allocated_mib"], rec["mem_reserved_mib"], rec["smi"]["max_mib"]))
        if a.peak_tflops:
            msg += " MFU %.1f%% HFU %.1f%%" % (rec["MFU"] * 100, rec["HFU"] * 100)
        print(msg)
    else:
        print("[%s] %s" % (a.tag, rec["status"]))
    print("saved", out)


if __name__ == "__main__":
    main()
