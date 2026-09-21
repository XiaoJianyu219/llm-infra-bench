#!/bin/bash
# Day11 主流程。tmux 里跑。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day11
PY=/root/autodl-tmp/venvs/det/bin/python
TR=/root/autodl-tmp/venvs/det/bin/torchrun
CK=/root/autodl-tmp/train
mkdir -p results/runs results/logs $CK
step(){ echo; echo "########## $* ##########"; date '+%F %T'; }
A="--strategy single --steps 8 --micro-bsz 4 --grad-accum 8 --deterministic 1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # 严格确定性需要，见 journal

step "1. A 组：不中断连跑 16 步（基准序列 L_A）"
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py --strategy single --steps 16 \
  --micro-bsz 4 --grad-accum 8 --tag A --out results/runs/A_continuous.json \
  > results/logs/A_continuous.log 2>&1
grep -E "^\[" results/logs/A_continuous.log | tail -2

step "2. B 组：跑 8 步 → 存 → 退出进程 → 新进程恢复 → 再跑 8 步（三档完整度）"
for LV in c0 c1 c2; do
  echo "--- $LV : $(grep -o "\"$LV\": \"[^\"]*\"" scripts/train_ft.py | head -1) ---"
  rm -f $CK/ck_$LV.pt
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py $A --ckpt-level $LV --save-at 8 \
    --save-path $CK/ck_$LV.pt --tag B1_$LV --out results/runs/B1_$LV.json \
    > results/logs/B1_$LV.log 2>&1
  grep -E "保存" results/logs/B1_$LV.log | tail -1
  # 真的退出了进程：上面这条命令已经结束，下面是全新的 python 进程
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py $A --ckpt-level $LV \
    --resume $CK/ck_$LV.pt --tag B2_$LV --out results/runs/B2_$LV.json \
    > results/logs/B2_$LV.log 2>&1
  grep -E "恢复" results/logs/B2_$LV.log | tail -1
done

step "3. checkpoint 单次写入耗时与体积（2.3）"
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py --strategy single --steps 1 \
  --micro-bsz 4 --grad-accum 8 --bench-save 5 --ckpt-level c2 \
  --save-path $CK/bench.pt --tag bench --out results/runs/bench_save.json \
  > results/logs/bench_save.log 2>&1
grep -E "保存 #" results/logs/bench_save.log

step "4. 保存间隔扫描（2.3）"
for EV in 10 50 200 1000; do
  N=$EV; [ "$N" -gt 40 ] && N=40
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py --strategy single --steps $N \
    --micro-bsz 4 --grad-accum 8 --save-every $EV --ckpt-level c2 \
    --save-path $CK/periodic.pt --tag every$EV --out results/runs/every_$EV.json \
    > results/logs/every_$EV.log 2>&1
  echo "  间隔 $EV 步：$(grep -c '保存' results/logs/every_$EV.log) 次保存"
done
rm -f $CK/periodic.pt

step "5. 对齐判据分析"
$PY scripts/align.py 2>&1 | tee results/alignment.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "DONE $(date '+%F %T')"
