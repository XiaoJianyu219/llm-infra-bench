#!/bin/bash
# Day11 修正批次：A 组补 deterministic、bench 在训练后测、间隔扫描换成可行区间
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day11
PY=/root/autodl-tmp/venvs/det/bin/python
CK=/root/autodl-tmp/train
export CUBLAS_WORKSPACE_CONFIG=:4096:8
step(){ echo; echo "########## $* ##########"; date '+%F %T'; }

step "R1. A 组重跑（补 --deterministic 1，与 B 组同口径）"
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py --strategy single --steps 16   --micro-bsz 4 --grad-accum 8 --deterministic 1 --tag A   --out results/runs/A_continuous.json > results/logs/A_continuous.log 2>&1
tail -2 results/logs/A_continuous.log | head -1

step "R2. checkpoint 体积/耗时（先训 2 步让 AdamW 的 m/v 建起来）"
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py --strategy single --steps 2   --micro-bsz 4 --grad-accum 8 --deterministic 1 --bench-save 5 --ckpt-level c2   --save-path $CK/bench.pt --tag bench --out results/runs/bench_save.json   > results/logs/bench_save.log 2>&1
grep -E "保存 #" results/logs/bench_save.log

step "R3. 各档完整度的体积对照"
for LV in c0 c1 c2; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py --strategy single --steps 2     --micro-bsz 4 --grad-accum 8 --deterministic 1 --bench-save 2 --ckpt-level $LV     --save-path $CK/bench_$LV.pt --tag bench_$LV --out results/runs/bench_$LV.json     > results/logs/bench_$LV.log 2>&1
  echo "  $LV: $(grep -E '保存 #0' results/logs/bench_$LV.log)"
done

step "R4. 保存间隔扫描（实测小间隔，大间隔用 delta 外推）"
for EV in 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_ft.py --strategy single --steps 16     --micro-bsz 4 --grad-accum 8 --deterministic 1 --save-every $EV --ckpt-level c2     --save-path $CK/periodic.pt --tag every$EV --out results/runs/every_$EV.json     > results/logs/every_$EV.log 2>&1
  echo "  间隔 $EV 步 完成"
done
rm -f $CK/periodic.pt $CK/bench_*.pt

step "R5. 重新分析"
$PY scripts/align.py 2>&1 | tee results/alignment.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "FIXDONE $(date '+%F %T')"
