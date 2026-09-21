#!/bin/bash
# Day08 全流程。放 tmux 里跑，输出 tee 到 results/run_all.log。
# 不用 set -e：单个配置 OOM / 失败是结论的一部分，不能让整条链停。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day08
PY=/root/autodl-tmp/venvs/det/bin/python
mkdir -p results/runs results/profiles

step() { echo; echo "############ $* ############"; date '+%F %T'; }

# ---------- 0. 等 GPU 空出来（Day07 的扫描还在跑，不去 kill 它） ----------
step "0. 等待 GPU 空闲"
# 注意：tmux 服务进程的 cmdline 会永久保留启动它的那条命令（含 "day07"），
# 所以必须排除 tmux 本身，否则这个循环永远不会退出。
WAITED=0
while true; do
  BUSY=$(pgrep -fa "days/day07/scripts/" | grep -v ' tmux ' | grep -cv pgrep || true)
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$BUSY" -eq 0 ] && [ "$APPS" -eq 0 ] && [ "$USED" -lt 600 ]; then
    echo "GPU 空闲：used=${USED} MiB，其他实验进程 ${BUSY} 个。开始。"; break
  fi
  if [ "$WAITED" -ge 150 ]; then
    echo "!! 等了 75 分钟 GPU 仍被占用（used=${USED} MiB，day07 进程 ${BUSY} 个）。"
    echo "!! 不在有争用的 GPU 上测量，本次退出，等人决定。"
    exit 3
  fi
  if [ $((WAITED % 4)) -eq 0 ]; then
    echo "$(date '+%T') 等待中：used=${USED} MiB，GPU 进程 ${APPS} 个，day07 脚本进程 ${BUSY} 个"
  fi
  WAITED=$((WAITED + 1))
  sleep 30
done
nvidia-smi > results/nvidia_smi_before.txt 2>&1
$PY -c "import torch;print('torch',torch.__version__,torch.version.cuda)" > results/env.txt 2>&1
$PY -m pip list 2>/dev/null >> results/env.txt

# ---------- 1. 分母：GEMM 峰值与带宽 ----------
step "1. GEMM 峰值 / 显存带宽 (4.2 分母)"
$PY scripts/gemm_peak.py --out results/gemm_peak.json 2>&1 | tee results/gemm_peak.log

# ---------- 2. 冷启动逐步耗时曲线：决定丢几步 ----------
step "2. 前 20 步逐步耗时曲线 (4.1 / 坑 5.2)"
$PY scripts/train_loop.py --tag step_curve --curve 20 --warmup 20 \
   --steps 10 --repeats 3 --seq-len 2048 --batch-size 4 --precision bf16 \
   --sdpa flash --ckpt 0 --out results/step_curve.json 2>&1 | tee results/step_curve.log

# ---------- 3. 开关对比矩阵 ----------
step "3. 开关对比矩阵 (4.4)"
$PY scripts/run_matrix.py 2>&1 | tee results/matrix.log

# ---------- 4. profiler：只用来定位，不用来计时 ----------
step "4. profiler (4.3)"
for CFG in "base_s2048_b4_bf16_flash flash 0 bf16" \
           "prof_math math 0 bf16" \
           "prof_ckpt flash 1 bf16" \
           "prof_fp32 mem_efficient 0 fp32"; do
  set -- $CFG
  echo "--- profiler: $1 (sdpa=$2 ckpt=$3 prec=$4) ---"
  $PY scripts/profile_run.py --tag "$1" --sdpa "$2" --ckpt "$3" --precision "$4" \
      --seq-len 2048 --batch-size 4 --warmup 10 --active 3 2>&1 | tee -a results/profile.log
done

# ---------- 5. bound 判定 ----------
step "5. bound 类型判定 (4.3)"
$PY scripts/analyze.py 2>&1 | tee results/bound_analysis.log

# ---------- 6. 权衡曲线 ----------
step "6. 显存-吞吐权衡曲线 (4.5)"
$PY scripts/plot_tradeoff.py 2>&1 | tee results/tradeoff.log

# ---------- 7. 收尾 ----------
step "7. 收尾：确认显存已释放"
sleep 3
nvidia-smi > results/nvidia_smi_after.txt 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> results/nvidia_smi_after.txt 2>&1
cat results/nvidia_smi_after.txt | tail -12
echo
echo "ALL DONE $(date '+%F %T')"
