"""Day09 第 1 节第 1 步：跑一次最小 all-reduce，把 NCCL 选用的 transport 逼出来。

配合 NCCL_DEBUG=INFO 使用，日志原文摘录进 report：
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV torchrun --nproc_per_node=2 nccl_probe.py

要在日志里找的是 "Channel .. via P2P" / "via SHM" / "via NET" 这类字样。
任务书第 1 节要求自己确认，不要采信"4090 禁用了 P2P"这句话。
"""
import os, json, torch
import torch.distributed as dist


def main():
    local = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")

    info = dict(rank=rank, world=world,
                device=torch.cuda.get_device_name(local),
                capability=torch.cuda.get_device_capability(local))
    try:
        info["nccl_version"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass

    # 直接问驱动：本卡能不能对另一张卡做 P2P
    if world > 1:
        peers = {}
        for other in range(torch.cuda.device_count()):
            if other != local:
                try:
                    peers[str(other)] = torch.cuda.can_device_access_peer(local, other)
                except Exception as e:
                    peers[str(other)] = "err: %s" % type(e).__name__
        info["can_device_access_peer"] = peers

    x = torch.ones(1024 * 1024, device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    info["all_reduce_ok"] = bool(x[0].item() == world)

    print("RANK %d %s" % (rank, json.dumps(info, ensure_ascii=False)), flush=True)
    if rank == 0:
        os.makedirs("results", exist_ok=True)
        json.dump(info, open("results/nccl_probe.json", "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
