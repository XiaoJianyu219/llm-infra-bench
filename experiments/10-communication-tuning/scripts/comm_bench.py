"""Day09 第 1 节：通信带宽实测。这是后面所有扩展效率归因的分母。

沿用 Day 08「自测 GEMM 峰值而不抄规格书」的同一条原则。

算法带宽 algbw 与总线带宽 busbw 的区别（report 里要写清）：
    algbw = S / t                      S 是用户看到的消息大小
    busbw = algbw × 2(P-1)/P           ring all-reduce 每个字节实际过总线的次数
P=2 时 2(P-1)/P = 1，两者数值相等 —— 这一档没有区分度，
到 P≥4 才拉开（P=4 时 busbw = 1.5×algbw）。all-gather / reduce-scatter
的系数是 (P-1)/P，与 all-reduce 不同，不能混用。

用法：
    torchrun --nproc_per_node=2 comm_bench.py --out results/comm_bench.json
单卡跑也能出 H2D/D2H/D2D 那部分，集合通信部分会标 skipped。
"""
import argparse, json, os, time, socket
import torch
import torch.distributed as dist


def env_int(k, d=0):
    return int(os.environ.get(k, d))


def sync():
    torch.cuda.synchronize()


def timeit(fn, iters, warmup=5, sync_ranks=True):
    """sync_ranks=False 用于只在 rank 0 上跑的本地测量（H2D/D2H/D2D）。

    这里踩过一次：pcie_bench() 只在 rank 0 调用，而 timeit 里无条件 dist.barrier()，
    于是 rank 0 进了一个 rank 1 永远不会进的 barrier，集合操作不配对，
    NCCL watchdog 等满 600 秒后把进程 SIGABRT 掉，整个 comm_bench 的 json 没写成。
    NCCL 的报错原文就是 "the order of collectives is not same for all ranks"。
    """
    for _ in range(warmup):
        fn()
    sync()
    if sync_ranks and dist.is_initialized():
        dist.barrier()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def coll_bench(rank, world, dtype=torch.bfloat16):
    """扫消息大小，测 all-reduce / all-gather / reduce-scatter。"""
    out = []
    esz = torch.tensor([], dtype=dtype).element_size()
    for mb in (1, 4, 16, 64, 256, 512, 1024):
        nbytes = mb * 1024 * 1024
        n = nbytes // esz
        iters = 20 if mb <= 64 else (10 if mb <= 256 else 5)
        row = dict(size_mb=mb, bytes=nbytes, dtype=str(dtype))
        try:
            x = torch.empty(n, dtype=dtype, device="cuda")
            x.uniform_()

            t = timeit(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), iters)
            alg = nbytes / t / 1e9
            row["all_reduce"] = dict(sec=t, algbw_gbps=alg,
                                     busbw_gbps=alg * 2 * (world - 1) / world)

            # all-gather：每 rank 出 n/world，凑成 n
            shard = torch.empty(n // world, dtype=dtype, device="cuda")
            shard.uniform_()
            t = timeit(lambda: dist.all_gather_into_tensor(x, shard), iters)
            alg = nbytes / t / 1e9
            row["all_gather"] = dict(sec=t, algbw_gbps=alg,
                                     busbw_gbps=alg * (world - 1) / world)

            t = timeit(lambda: dist.reduce_scatter_tensor(shard, x,
                                                          op=dist.ReduceOp.SUM), iters)
            alg = nbytes / t / 1e9
            row["reduce_scatter"] = dict(sec=t, algbw_gbps=alg,
                                         busbw_gbps=alg * (world - 1) / world)
            del x, shard
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            row["error"] = "OOM: " + str(e).split("\n")[0]
            torch.cuda.empty_cache()
        except Exception as e:
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
            torch.cuda.empty_cache()
        out.append(row)
        if rank == 0 and "all_reduce" in row:
            print("all-reduce %5d MB  %8.3f ms  algbw %7.2f GB/s  busbw %7.2f GB/s"
                  % (mb, row["all_reduce"]["sec"] * 1e3,
                     row["all_reduce"]["algbw_gbps"], row["all_reduce"]["busbw_gbps"]),
                  flush=True)
    return out


def pcie_bench():
    """H2D / D2H（pinned 与 pageable）与 D2D，作为集合通信的参照系。"""
    res = {}
    n = 256 * 1024 * 1024  # 256 MB
    d = torch.empty(n, dtype=torch.uint8, device="cuda")
    for tag, pinned in (("pinned", True), ("pageable", False)):
        h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
        t = timeit(lambda: d.copy_(h, non_blocking=pinned), 10, sync_ranks=False)
        res["H2D_" + tag] = dict(sec=t, gbps=n / t / 1e9)
        t = timeit(lambda: h.copy_(d, non_blocking=pinned), 10, sync_ranks=False)
        res["D2H_" + tag] = dict(sec=t, gbps=n / t / 1e9)
        del h
    d2 = torch.empty_like(d)
    t = timeit(lambda: d2.copy_(d), 20, sync_ranks=False)
    res["D2D"] = dict(sec=t, gbps=2 * n / t / 1e9)   # 1 读 1 写
    del d, d2
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/comm_bench.json")
    a = ap.parse_args()

    rank = env_int("RANK", 0)
    local = env_int("LOCAL_RANK", 0)
    world = env_int("WORLD_SIZE", 1)
    torch.cuda.set_device(local)

    rec = dict(world_size=world, host=socket.gethostname(),
               device=torch.cuda.get_device_name(local),
               torch=torch.__version__, nccl=None,
               visible=torch.cuda.device_count())
    try:
        rec["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass

    if world > 1:
        dist.init_process_group("nccl")
        rec["backend"] = dist.get_backend()
        rec["collectives"] = coll_bench(rank, world)
    else:
        rec["collectives"] = "skipped: world_size=1，单卡上没有卡间通信可测"
        if rank == 0:
            print("world_size=1，跳过集合通信部分（任务书第 1 节需要 2 卡）")

    if rank == 0:
        rec["pcie"] = pcie_bench()
        for k, v in rec["pcie"].items():
            print("%-16s %8.1f GB/s" % (k, v["gbps"]))
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
        print("saved", a.out)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
