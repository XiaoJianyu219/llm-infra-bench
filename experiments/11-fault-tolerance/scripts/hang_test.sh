#!/bin/bash
# Day11 2.4 补充：SIGSTOP 冻结一个 rank —— 进程还活着，elastic agent 看不出问题，
# 于是剩下的 rank 会真的卡在 all-reduce 上直到 NCCL watchdog 超时。
# kill -9 那一组走的是完全不同的路径（agent 快速失败），两者必须分开写进 report。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day11
TR=/root/autodl-tmp/venvs/det/bin/torchrun
A="--strategy ddp --micro-bsz 2 --grad-accum 2 --steps 12"
step(){ echo; echo "########## $* ##########"; date '+%F %T'; }

step "2.4-D SIGSTOP 冻结 rank1，超时设为 45 秒"
( TORCH_NCCL_TIMEOUT_MS=45000 timeout 260 $TR --nproc_per_node=2 --master_port=29811   scripts/train_ft.py $A --hang-at 4 --tag hangD --out results/runs/hang_D.json   ) > results/logs/hang_D.log 2>&1
echo "--- timeline ---"
grep -E "SIGSTOP 冻结|step [0-9]+ done|Watchdog|ran for|Timeout\(ms\)|exitcode|SIGABRT|进程组已建立"   results/logs/hang_D.log | head -16
pkill -f train_ft.py 2>/dev/null; sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

step "2.4-E 同样冻结，但超时设为 20 秒（验证确实由 TORCH_NCCL_TIMEOUT_MS 控制）"
( TORCH_NCCL_TIMEOUT_MS=20000 timeout 200 $TR --nproc_per_node=2 --master_port=29812   scripts/train_ft.py $A --hang-at 4 --tag hangE --out results/runs/hang_E.json   ) > results/logs/hang_E.log 2>&1
grep -E "SIGSTOP 冻结|Watchdog|ran for|Timeout\(ms\)|exitcode" results/logs/hang_E.log | head -10
pkill -f train_ft.py 2>/dev/null; sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "HANGDONE $(date '+%F %T')"

step "2.4-C2 弹性重启（重做：用干净的独立目录，确保 kill 真的触发）"
ED=/root/autodl-tmp/train/elastic
rm -rf $ED; mkdir -p $ED
( TORCH_NCCL_TIMEOUT_MS=60000 timeout 400 $TR --nproc_per_node=2 --max-restarts=2   --master_port=29813 scripts/train_ft.py $A --steps 12 --save-every 2   --ckpt-level c2 --save-path $ED/ck.pt --auto-resume $ED   --kill-at 5 --kill-marker $ED/killed.marker --tag killC2   --out results/runs/kill_C2.json ) > results/logs/kill_C2.log 2>&1
echo "--- timeline ---"
grep -E "auto-resume|从 .* 恢复|rank1 pid=|step [0-9]+ done|exitcode|Restarting|restart"   results/logs/kill_C2.log | head -30
pkill -f train_ft.py 2>/dev/null; sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "ALLDONE $(date '+%F %T')"
