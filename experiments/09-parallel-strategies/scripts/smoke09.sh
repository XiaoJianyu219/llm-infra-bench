#!/bin/bash
# 任务书 §0：先把所有脚本在单卡上调通，再开双卡跑矩阵。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day09
PY=/root/autodl-tmp/venvs/det/bin/python
TR=/root/autodl-tmp/venvs/det/bin/torchrun
mkdir -p /tmp/smoke9
A="--seq-len 512 --micro-bsz 1 --warmup 2 --steps 2 --repeats 1"
for ST in single ddp zero1 zero2 zero3 fsdp; do
  echo "=========== $ST ==========="
  if [ "$ST" = "single" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY scripts/train_dist.py --strategy $ST $A \
      --out /tmp/smoke9/$ST.json 2>&1 | grep -vE "Loading weights|^\[20|it/s\]" | tail -6
  else
    CUDA_VISIBLE_DEVICES=0 $TR --nproc_per_node=1 --master_port=29601 scripts/train_dist.py \
      --strategy $ST $A --out /tmp/smoke9/$ST.json 2>&1 | grep -vE "Loading weights|it/s\]" | tail -8
  fi
  pkill -f torchrun 2>/dev/null; sleep 2
done
echo "=========== lora single ==========="
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_dist.py --strategy single $A --lora \
  --out /tmp/smoke9/lora.json 2>&1 | grep -vE "Loading weights|it/s\]" | tail -6
echo "=========== comm_bench 单卡（只有 PCIe 部分） ==========="
CUDA_VISIBLE_DEVICES=0 $PY scripts/comm_bench.py --out /tmp/smoke9/comm.json 2>&1 | tail -10
echo "=========== 结果 ==========="
$PY - <<'PY'
import json,glob,os
for p in sorted(glob.glob("/tmp/smoke9/*.json")):
    d=json.load(open(p)); n=os.path.basename(p)
    if "status" in d:
        extra=""
        if d.get("fsdp_wrapped_modules"): extra=" fsdp_wrapped=%d ok=%s"%(d["fsdp_wrapped_modules"],d.get("fsdp_wrap_policy_ok"))
        if d.get("ds_zero_stage"): extra=" ds_stage=%s eff=%s"%(d["ds_zero_stage"],d.get("ds_config_effective"))
        if d.get("lora"): extra=" lora_trainable=%d (%.3f%%)"%(d["lora"]["trainable"],d["lora"]["trainable_ratio"]*100)
        print("%-12s %-6s step=%s alloc=%s%s"%(n,d["status"],
            round(d.get("step_sec_median",0)*1000,1), round(d.get("mem_allocated_mib",0)),extra))
        if d["status"]!="ok": print("   ", (d.get("error") or d.get("oom_message"))[:200])
    else:
        print(n, "pcie:", {k:round(v["gbps"],1) for k,v in d.get("pcie",{}).items()})
PY
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
