#!/bin/bash
# Day11 2.4 掉卡模拟（需双卡）。每一步都打时间戳，report 里要写 timeline。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day11
PY=/root/autodl-tmp/venvs/det/bin/python
TR=/root/autodl-tmp/venvs/det/bin/torchrun
CK=/root/autodl-tmp/train
mkdir -p results/runs results/logs $CK
step(){ echo; echo "########## $* ##########"; date '+%F %T'; }
clean(){ sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; }

A="--strategy ddp --micro-bsz 2 --grad-accum 2 --steps 12 --deterministic 0"

step "2.4-0 默认集合超时是多少（先读真实值，不靠记忆）"
$TR --nproc_per_node=2 --master_port=29801 scripts/train_ft.py $A --steps 1 \
  --tag probe --out results/runs/kill_probe.json 2>&1 | grep -E "进程组已建立" | head -1
clean

step "2.4-A 默认设置下 kill rank1：剩下的 rank 是立刻报错还是卡住"
# 用 60 秒超时把 timeline 压到可观测范围；默认 600 秒的值由 2.4-0 给出
( TORCH_NCCL_TIMEOUT_MS=60000 timeout 200 $TR --nproc_per_node=2 --master_port=29802 \
    scripts/train_ft.py $A --kill-at 4 --tag killA \
    --out results/runs/kill_A.json ) > results/logs/kill_A.log 2>&1
echo "--- timeline ---"
grep -E "rank1 pid|step [0-9]+ done|Watchdog|timeout|Timeout|exitcode|SIGABRT|进程组已建立" \
  results/logs/kill_A.log | head -18
clean

step "2.4-B TORCH_NCCL_ASYNC_ERROR_HANDLING=0（关掉异步错误处理）"
( TORCH_NCCL_TIMEOUT_MS=60000 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 timeout 200 \
  $TR --nproc_per_node=2 --master_port=29803 scripts/train_ft.py $A --kill-at 4 \
  --tag killB --out results/runs/kill_B.json ) > results/logs/kill_B.log 2>&1
grep -E "rank1 pid|step [0-9]+ done|Watchdog|timeout|Timeout|exitcode|SIGABRT" \
  results/logs/kill_B.log | head -14
clean

step "2.4-C 弹性重启：torchrun --max-restarts=2 + 自动从最近 ckpt 恢复"
rm -f $CK/elastic_*.pt $CK/killed.marker
# 先跑一段并周期性存 ckpt，再在第 4 步掉卡；重启后 --auto-resume 接上
( TORCH_NCCL_TIMEOUT_MS=60000 timeout 400 $TR --nproc_per_node=2 --max-restarts=2 \
  --master_port=29804 scripts/train_ft.py $A --steps 12 --save-every 2 \
  --ckpt-level c2 --save-path $CK/elastic_ck.pt --auto-resume $CK \
  --kill-at 4 --kill-marker $CK/killed.marker --tag killC \
  --out results/runs/kill_C.json ) > results/logs/kill_C.log 2>&1
echo "--- timeline ---"
grep -E "rank1 pid|auto-resume|从 .* 恢复|step [0-9]+ done|Watchdog|restart|Restart|exitcode" \
  results/logs/kill_C.log | head -24
clean
echo "DONE $(date '+%F %T')"
