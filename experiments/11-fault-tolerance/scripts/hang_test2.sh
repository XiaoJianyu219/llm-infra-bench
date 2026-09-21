#!/bin/bash
# 2.4-D/E 重做：超时经 init_process_group(timeout=) 传入（TORCH_NCCL_TIMEOUT_MS 不存在）
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day11
TR=/root/autodl-tmp/venvs/det/bin/torchrun
A="--strategy ddp --micro-bsz 2 --grad-accum 2 --steps 12 --hang-at 4"
step(){ echo; echo "########## $* ##########"; date '+%F %T'; }
for T in 30 60; do
  step "2.4-D SIGSTOP 冻结 rank1，集合超时 ${T}s"
  ( timeout $((T+150)) $TR --nproc_per_node=2 --master_port=2982$((T/30))     scripts/train_ft.py $A --pg-timeout-sec $T --tag hang$T     --out results/runs/hang_$T.json ) > results/logs/hang_$T.log 2>&1
  echo "--- timeline ---"
  grep -E "进程组已建立|SIGSTOP 冻结|step 3 done|Watchdog caught|ran for|exitcode|SIGABRT|Received a dump"     results/logs/hang_$T.log | sed "s/Exception raised.*//" | head -10
  pkill -f train_ft.py 2>/dev/null; sleep 4
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
done
echo "HANG2DONE $(date '+%F %T')"
