"""Day11 容错与断点续训。载体沿用 Day 10 的最优单卡配置（micro4 / grad_accum8）。

设计要点（journal 第 0 节已说明理由）：
  * 有结构的合成语料：固定 seed 造一个 [n_samples, seq_len] 数组当"数据集"，
    loss 会真的下降，"数据顺序变了"才会在 loss 上留下痕迹。
  * sampler 的每 epoch 排列由 (base_seed, epoch) 决定，与全局 RNG 无关，
    这样"数据位置"与"RNG"两种状态可以分别失效、分别观察。
  * attention_dropout = 0.1，让全局 RNG 真正进入 forward。
  * checkpoint 三档完整度 c0 / c1 / c2，见 --ckpt-level。
  * 只在梯度累积边界保存（journal 预测 1 的第 2 条：global step 记不出 micro-step 位置）。
"""
import argparse, json, os, random, sys, time, socket, statistics, subprocess
import numpy as np
import torch
import torch.distributed as dist

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
VOCAB = 151936


def envi(k, d=0):
    return int(os.environ.get(k, d))


RANK, LOCAL_RANK, WORLD = envi("RANK"), envi("LOCAL_RANK"), envi("WORLD_SIZE", 1)


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def ts():
    return time.strftime("%H:%M:%S") + ".%03d" % int((time.time() % 1) * 1000)


# ---------------------------------------------------------------- 数据

class SyntheticCorpus:
    """固定的 token 数组当数据集；排列只由 (seed, epoch) 决定，与全局 RNG 无关。"""

    def __init__(self, n_samples, seq_len, seed=1234):
        g = torch.Generator().manual_seed(seed)
        # 有结构：每条样本是若干个短 motif 的重复，模型能学到东西，loss 会下降
        n_motif, motif_len = 64, 16
        motifs = torch.randint(0, VOCAB, (n_motif, motif_len), generator=g)
        idx = torch.randint(0, n_motif, (n_samples, seq_len // motif_len), generator=g)
        self.data = motifs[idx].reshape(n_samples, -1)[:, :seq_len].contiguous()
        self.n = n_samples
        self.seed = seed

    def permutation(self, epoch):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
        return torch.randperm(self.n, generator=g)


class ResumableSampler:
    """可恢复的顺序：状态 = (epoch, 已消费样本数)。不依赖全局 RNG。"""

    def __init__(self, corpus, batch_size, rank=0, world=1):
        self.c, self.bs, self.rank, self.world = corpus, batch_size, rank, world
        self.epoch, self.consumed = 0, 0
        self.perm = self.c.permutation(0)

    def next_batch(self):
        need = self.bs * self.world
        if self.consumed + need > self.c.n:
            self.epoch += 1
            self.consumed = 0
            self.perm = self.c.permutation(self.epoch)
        sl = self.perm[self.consumed + self.rank * self.bs:
                       self.consumed + (self.rank + 1) * self.bs]
        self.consumed += need
        return self.c.data[sl]

    def state_dict(self):
        return dict(epoch=self.epoch, consumed=self.consumed)

    def load_state_dict(self, s):
        self.epoch, self.consumed = s["epoch"], s["consumed"]
        self.perm = self.c.permutation(self.epoch)


# ---------------------------------------------------------------- RNG

def rng_state():
    return dict(python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=[torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())])


def rng_load(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu() if torch.is_tensor(s["torch"]) else s["torch"])
    for i, st in enumerate(s["cuda"]):
        if i < torch.cuda.device_count():
            torch.cuda.set_rng_state(st.cpu() if torch.is_tensor(st) else st, i)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 模型

def build(args):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    cfg.attention_dropout = args.dropout
    m = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.float32)
    m.to("cuda").train()
    if args.compile:
        m = torch.compile(m)
    if args.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        m = DDP(m, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
                gradient_as_bucket_view=True)
    return m


def unwrap(m):
    m = getattr(m, "module", m)
    return getattr(m, "_orig_mod", m)


# ---------------------------------------------------------------- checkpoint

CKPT_FIELDS = {
    "c0": "模型权重 + 优化器状态",
    "c1": "c0 + 数据加载位置（epoch / 已消费索引）+ global step",
    "c2": "c1 + 四套 RNG + LR scheduler + 配置快照（完整）",
}


def save_ckpt(path, model, opt, sched, sampler, step, tokens, args, level, async_write=False):
    """返回 (阻塞耗时, 总耗时, 体积字节)。任务书 3.2：计时前必须 synchronize。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    blob = dict(level=level,
                model=unwrap(model).state_dict(),
                optimizer=opt.state_dict())
    if level in ("c1", "c2"):
        blob.update(sampler=sampler.state_dict(), global_step=step, tokens_seen=tokens)
    if level == "c2":
        blob.update(rng=rng_state(),
                    scheduler=sched.state_dict() if sched else None,
                    config=vars(args))
    t_blocking = time.perf_counter() - t0          # 组装（含 D2H）的阻塞时间
    torch.save(blob, path)
    t_total = time.perf_counter() - t0
    return t_blocking, t_total, os.path.getsize(path)


def load_ckpt(path, model, opt, sched, sampler):
    blob = torch.load(path, map_location="cuda", weights_only=False)
    unwrap(model).load_state_dict(blob["model"])
    opt.load_state_dict(blob["optimizer"])
    lv = blob.get("level", "c0")
    restored = ["model", "optimizer"]
    if "sampler" in blob:
        sampler.load_state_dict(blob["sampler"])
        restored.append("sampler")
    if "rng" in blob:
        rng_load(blob["rng"])
        restored.append("rng")
    if blob.get("scheduler") is not None and sched is not None:
        sched.load_state_dict(blob["scheduler"])
        restored.append("scheduler")
    return blob.get("global_step", 0), blob.get("tokens_seen", 0), lv, restored


# ---------------------------------------------------------------- 主循环

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="single", choices=["single", "ddp"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bsz", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--steps", type=int, default=20, help="本次要跑的优化器步数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ckpt-level", default="c2", choices=["c0", "c1", "c2"])
    p.add_argument("--save-at", type=int, default=-1, help="在第几步之后保存（-1 不保存）")
    p.add_argument("--save-path", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--save-every", type=int, default=0, help=">0 则周期性保存，用于 2.3 扫描")
    p.add_argument("--bench-save", type=int, default=0, help=">0 则只测保存耗时，重复 N 次")
    p.add_argument("--kill-at", type=int, default=-1, help="2.4：rank1 在第几步 kill -9 自己")
    p.add_argument("--per-rank-rng", type=int, default=0,
                   help="逐 rank 保存/恢复 RNG（修 2.4-C2 发现的 rank1 不对齐）")
    p.add_argument("--pg-timeout-sec", type=int, default=0,
                   help="集合操作超时。注意：不存在 TORCH_NCCL_TIMEOUT_MS 这个环境变量"
                        "（已在 libtorch_cuda.so 里核实），超时只能经 init_process_group(timeout=) 传")
    p.add_argument("--hang-at", type=int, default=-1,
                   help="2.4：rank1 在第几步 SIGSTOP 冻结自己（模拟'进程还在但无响应'，"
                        "这才会走到 NCCL watchdog 超时那条路）")
    p.add_argument("--kill-marker", default="",
                   help="给定时只 kill 一次：文件已存在就不再 kill（弹性重启用）")
    p.add_argument("--auto-resume", default="", help="目录：自动挑最新的 ckpt 恢复（弹性重启用）")
    p.add_argument("--deterministic", type=int, default=0,
                   help="开确定性算法（需 CUBLAS_WORKSPACE_CONFIG=:4096:8）")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.deterministic:
        # journal 预测 1 的第 3 项：这些开关不是"状态"，但决定 2.2 的判据能不能用
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Flash Attention 的 backward 默认非确定（PyTorch 的告警原文指出了这一点），
        # warn_only=False 才会强制它走确定性实现
    torch.cuda.set_device(LOCAL_RANK)
    if a.strategy == "ddp" or WORLD > 1:
        import datetime as _dt
        kw = {}
        if a.pg_timeout_sec > 0:
            kw["timeout"] = _dt.timedelta(seconds=a.pg_timeout_sec)
        dist.init_process_group("nccl", **kw)
        try:
            from torch.distributed.distributed_c10d import _get_default_timeout
            rec_timeout = str(_get_default_timeout(dist.get_backend()))
        except Exception as e:
            rec_timeout = "无法读取: %s" % type(e).__name__
        log("[%s] 进程组已建立 world=%d，默认集合超时 = %s，本次实际传入 timeout = %s；"
            "TORCH_NCCL_ASYNC_ERROR_HANDLING=%s TORCH_NCCL_BLOCKING_WAIT=%s"
            % (ts(), WORLD, rec_timeout,
               ("%ds" % a.pg_timeout_sec) if a.pg_timeout_sec > 0 else "未传（=默认）",
               os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "<未设>"),
               os.environ.get("TORCH_NCCL_BLOCKING_WAIT", "<未设>")))
    seed_all(a.seed + RANK)

    rec = dict(args=vars(a), rank=RANK, world=WORLD, host=socket.gethostname(),
               cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               deterministic_algorithms=bool(a.deterministic),
               pid=os.getpid(), start_ts=ts(), torch=torch.__version__,
               ckpt_semantics=CKPT_FIELDS[a.ckpt_level])

    corpus = SyntheticCorpus(a.n_samples, a.seq_len, seed=1234)
    sampler = ResumableSampler(corpus, a.micro_bsz, RANK, WORLD)
    model = build(a)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20))

    step0, tokens, resumed_level, restored = 0, 0, None, []
    if a.auto_resume and not a.resume:
        import glob as _g
        cands = sorted(_g.glob(os.path.join(a.auto_resume, "*.pt")), key=os.path.getmtime)
        if cands:
            a.resume = cands[-1]
            log("[%s] auto-resume 选中 %s" % (ts(), a.resume))
        else:
            log("[%s] auto-resume：目录里没有 ckpt，从头开始" % ts())
    if a.resume:
        step0, tokens, resumed_level, restored = load_ckpt(a.resume, model, opt, sched, sampler)
        side = "%s.rng%d" % (a.resume, RANK)
        if a.per_rank_rng and os.path.exists(side):
            rng_load(torch.load(side, weights_only=False))
            restored = restored + ["rng(本 rank 旁路)"]
        rec.update(resumed_from=a.resume, resumed_at_step=step0,
                   resumed_level=resumed_level, restored_fields=restored)
        log("[%s] 从 %s 恢复：level=%s step=%d 恢复了 %s"
            % (ts(), a.resume, resumed_level, step0, restored))

    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    losses, step_times, save_events, step_stamps = [], [], [], []

    # --- 训练 ---
    for s in range(step0, step0 + a.steps):
        if (a.kill_at >= 0 and RANK == 1 and s == a.kill_at
                and not (a.kill_marker and os.path.exists(a.kill_marker))):
            if a.kill_marker:
                open(a.kill_marker, "w").write(str(time.time()))
            print("[%s][rank1 pid=%d] 第 %d 步，kill -9 自己" % (ts(), os.getpid(), s),
                  flush=True)
            os.kill(os.getpid(), 9)

        if a.hang_at >= 0 and RANK == 1 and s == a.hang_at:
            import signal as _sig
            print("[%s][rank1 pid=%d] 第 %d 步，SIGSTOP 冻结自己（进程仍存活）"
                  % (ts(), os.getpid(), s), flush=True)
            os.kill(os.getpid(), _sig.SIGSTOP)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for i in range(a.grad_accum):
            batch = sampler.next_batch().cuda()
            last = (i == a.grad_accum - 1)
            import contextlib
            ctx = (contextlib.nullcontext() if (last or not hasattr(model, "no_sync"))
                   else model.no_sync())
            with ctx:
                with autocast:
                    loss = model(input_ids=batch, labels=batch, use_cache=False).loss
                (loss / a.grad_accum).backward()
            acc += float(loss.detach())
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tokens += a.micro_bsz * a.seq_len * a.grad_accum * WORLD
        losses.append(acc / a.grad_accum)
        step_times.append(dt)
        step_stamps.append(ts())
        if a.kill_at >= 0:
            print("[%s][rank%d] step %d done loss %.6f" % (ts(), RANK, s, losses[-1]),
                  flush=True)
        if RANK == 0 and (s - step0) < 5:
            log("[%s] step %d loss %.8f  %.1f ms" % (ts(), s, losses[-1], dt * 1000))

        # 逐 rank 的 RNG 旁路文件：主 ckpt 只由 rank0 写，只装得下 rank0 的 RNG。
        # 2.4-C2 实测：不这样做，恢复后 rank0 逐位对齐而 rank1 的 dropout 序列变了。
        if a.save_every and (s + 1) % a.save_every == 0 and a.per_rank_rng and a.save_path:
            torch.save(rng_state(), "%s.rng%d" % (a.save_path, RANK))
        # 周期性保存（只在累积边界，此处天然满足）
        if a.save_every and (s + 1) % a.save_every == 0 and RANK == 0:
            path = a.save_path or "/root/autodl-tmp/train/periodic.pt"
            tb, tt, sz = save_ckpt(path, model, opt, sched, sampler, s + 1, tokens, a,
                                   a.ckpt_level)
            save_events.append(dict(step=s + 1, blocking_sec=tb, total_sec=tt, bytes=sz))

        if a.save_at >= 0 and s == step0 + a.save_at - 1 and RANK == 0:
            tb, tt, sz = save_ckpt(a.save_path, model, opt, sched, sampler, s + 1, tokens, a,
                                   a.ckpt_level)
            rec.update(saved_at_step=s + 1, save_blocking_sec=tb, save_total_sec=tt,
                       save_bytes=sz)
            log("[%s] 在 step %d 保存 %s（level=%s，%.2f GB，阻塞 %.2f s，总 %.2f s）"
                % (ts(), s + 1, a.save_path, a.ckpt_level, sz / 1e9, tb, tt))

    # --- 保存耗时（任务书 2.3）。必须放在训练之后：
    #     AdamW 的 m/v 要到第一次 step() 才分配，之前存出来只有权重。---
    if a.bench_save:
        rows = []
        for i in range(a.bench_save):
            path = "%s.bench%d" % (a.save_path, i)
            tb, tt, sz = save_ckpt(path, model, opt, sched, sampler, 0, 0, a, a.ckpt_level)
            rows.append(dict(i=i, blocking_sec=tb, total_sec=tt, bytes=sz))
            os.remove(path)
            log("  保存 #%d：阻塞 %.3f s，总计 %.3f s，%.2f GB"
                % (i, tb, tt, sz / 1e9))
        rec["bench_save"] = rows
        rec["bench_save_median_total"] = statistics.median(r["total_sec"] for r in rows)
        rec["bench_save_median_blocking"] = statistics.median(r["blocking_sec"] for r in rows)
        rec["bench_save_bytes"] = rows[0]["bytes"]
        # 分项体积
        sd = unwrap(model).state_dict()
        osd = opt.state_dict()
        def nb(d):
            seen, tot = set(), 0
            for v in d.values():
                if torch.is_tensor(v) and v.data_ptr() not in seen:
                    seen.add(v.data_ptr()); tot += v.numel() * v.element_size()
            return tot   # tie_word_embeddings 会让同一块权重在 state_dict 里出现两次
        exp = sum(sum(t.numel() * t.element_size() for t in s.values() if torch.is_tensor(t))
                  for s in osd["state"].values()) if osd.get("state") else 0
        rec["size_breakdown"] = dict(model_bytes=nb(sd), optimizer_state_bytes=exp)
        if RANK == 0:
            json.dump(rec, open(a.out, "w"), indent=1, default=str)
            log("saved", a.out)


    rec.update(losses=losses, step_times=step_times, save_events=save_events,
               step_stamps=step_stamps,
               first_step=step0, last_step=step0 + a.steps - 1, tokens_seen=tokens,
               end_ts=ts(),
               step_ms_median=statistics.median(step_times) * 1000 if step_times else None,
               mem_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
    if RANK == 0:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1, default=str)
        log("saved", a.out)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
