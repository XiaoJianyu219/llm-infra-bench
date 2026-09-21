#!/bin/bash
# Day08 v3：只重跑 profiler 及其下游。矩阵与 step_curve 的数据 v2 已经是对的，不重测。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day08
PY=/root/autodl-tmp/venvs/det/bin/python
step() { echo; echo "############ $* ############"; date '+%F %T'; }

step "3. profiler 重跑（修正 kernel/aten 重复计数与后端分类）"
rm -f results/profile.log
while read -r TAG SDPA CKPT PREC SEQ BS; do
  echo "--- profiler: $TAG (sdpa=$SDPA ckpt=$CKPT prec=$PREC s=$SEQ b=$BS) ---"
  $PY scripts/profile_run.py --tag "$TAG" --sdpa "$SDPA" --ckpt "$CKPT" --precision "$PREC" \
      --seq-len "$SEQ" --batch-size "$BS" --warmup 10 --active 3 2>&1 \
      | grep -v "Loading weights" | tee -a results/profile.log
done <<'CFG'
ref_s2048_b1 flash 0 bf16 2048 1
ref_s512_b4 flash 0 bf16 512 4
prof_math_s2048_b1 math 0 bf16 2048 1
prof_memeff_s2048_b1 mem_efficient 0 bf16 2048 1
prof_ckpt_s2048_b1 flash 1 bf16 2048 1
prof_fp32_s2048_b1 mem_efficient 0 fp32 2048 1
CFG

step "4. bound 判定"
$PY scripts/analyze.py 2>&1 | tee results/bound_analysis.log

step "5. 权衡曲线"
$PY scripts/plot_tradeoff.py 2>&1 | tee results/tradeoff.log

step "6. CNN 对照"
$PY scripts/cnn_profile.py --imgsz 1024 --batch 4 --warmup 10 --steps 10 --repeats 3 2>&1 \
  | grep -vE "it/s\]" | tee results/cnn_profile.log

step "7. report 表"
$PY scripts/make_tables.py > /dev/null 2>&1 && echo "report_tables.md 已生成"

step "8. 收尾"
sleep 3
nvidia-smi > results/nvidia_smi_after.txt 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> results/nvidia_smi_after.txt
tail -3 results/nvidia_smi_after.txt
echo "ALL DONE v3 $(date '+%F %T')"
