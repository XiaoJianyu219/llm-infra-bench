#!/bin/bash
# Day08 全流程 v2。基准工作点从 s2048/b4 改为 s2048/b1（v1 的基准在 24 GB 上必 OOM）。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day08
PY=/root/autodl-tmp/venvs/det/bin/python
mkdir -p results/runs results/profiles

step() { echo; echo "############ $* ############"; date '+%F %T'; }

step "0. GPU 状态"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv > results/nvidia_smi_before.txt
nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> results/nvidia_smi_before.txt
cat results/nvidia_smi_before.txt

step "1. 冷启动逐步耗时曲线 (4.1 / 坑 5.2)"
$PY scripts/train_loop.py --tag step_curve --curve 20 --warmup 20 --steps 10 --repeats 3 \
   --seq-len 2048 --batch-size 1 --precision bf16 --sdpa flash --ckpt 0 \
   --out results/step_curve.json 2>&1 | grep -v "Loading weights" | tee results/step_curve.log

step "2. 开关对比矩阵 v2 (4.4)"
$PY scripts/run_matrix2.py 2>&1 | grep -v "Loading weights" | tee results/matrix.log

step "3. profiler (4.3) —— 只定位，不计时"
while read -r TAG SDPA CKPT PREC SEQ BS; do
  echo "--- profiler: $TAG (sdpa=$SDPA ckpt=$CKPT prec=$PREC s=$SEQ b=$BS) ---"
  $PY scripts/profile_run.py --tag "$TAG" --sdpa "$SDPA" --ckpt "$CKPT" --precision "$PREC" \
      --seq-len "$SEQ" --batch-size "$BS" --warmup 10 --active 3 2>&1 \
      | grep -v "Loading weights" | tee -a results/profile.log
done <<'CFG'
ref_s2048_b1 flash 0 bf16 2048 1
ref_s512_b4 flash 0 bf16 512 4
prof_math_s2048_b1 math 0 bf16 2048 1
prof_ckpt_s2048_b1 flash 1 bf16 2048 1
prof_fp32_s2048_b1 mem_efficient 0 fp32 2048 1
CFG

step "4. bound 类型判定 (4.3)"
$PY scripts/analyze.py 2>&1 | tee results/bound_analysis.log

step "5. 显存-吞吐权衡曲线 (4.5)"
$PY scripts/plot_tradeoff.py 2>&1 | tee results/tradeoff.log

step "6. CNN 对照 (4.6，可选)"
$PY scripts/cnn_profile.py --imgsz 1024 --batch 4 --warmup 10 --steps 10 --repeats 3 \
   2>&1 | grep -v "it/s\]" | tee results/cnn_profile.log

step "7. 渲染 report 用表"
$PY scripts/make_tables.py > /dev/null 2>&1 && echo "report_tables.md 已生成"

step "8. 收尾：确认显存已释放"
sleep 3
nvidia-smi > results/nvidia_smi_after.txt 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> results/nvidia_smi_after.txt
tail -6 results/nvidia_smi_after.txt
echo
echo "ALL DONE $(date '+%F %T')"
