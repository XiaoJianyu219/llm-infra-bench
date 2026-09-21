#!/bin/bash
# Day10 全流程。tmux 里跑，tee 到 results/run_all.log。
# 不用 set -e：单个配置失败是结论的一部分。
# 用法: bash run_day10.sh [stage]   stage ∈ all|peak|noise|base|accum|bucket|ddpflag|nccl|single|inter|prof
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day10
PY=/root/autodl-tmp/venvs/det/bin/python
TR=/root/autodl-tmp/venvs/det/bin/torchrun
STAGE="${1:-all}"
mkdir -p results/runs results/logs results/profiles

SEQ=${SEQ:-512}
MB=${MB:-2}

step() { echo; echo "############ $* ############"; date '+%F %T'; }

cleanup() {
  pkill -f train_opt.py 2>/dev/null
  sleep 3
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
  for u in $used; do
    if [ "$u" -gt 600 ]; then echo "  !! 卡未回到基线: ${used}"; sleep 8; fi
  done
}

peak() { $PY -c "import json;print(json.load(open('results/gemm_peak.json'))['peak_bf16_tflops'])" 2>/dev/null || echo 0; }

# run <tag> <nproc> [额外参数...]
run() {
  local tag=$1 np=$2; shift 2
  if [ -f "results/runs/${tag}.json" ]; then echo "skip(exists) $tag"; return; fi
  local PK; PK=$(peak)
  local t0=$SECONDS
  if [ "$np" = "1" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY scripts/train_opt.py --strategy single --seq-len $SEQ \
      --peak-tflops "$PK" --tag "$tag" --out "results/runs/${tag}.json" "$@" \
      > "results/logs/${tag}.log" 2>&1
  else
    $TR --nproc_per_node=$np --master_port=29701 scripts/train_opt.py --strategy ddp \
      --seq-len $SEQ --peak-tflops "$PK" --tag "$tag" --out "results/runs/${tag}.json" "$@" \
      > "results/logs/${tag}.log" 2>&1
  fi
  echo "  [$((SECONDS-t0))s] $(grep -E '^\[' "results/logs/${tag}.log" | tail -1)"
  cleanup
}

# runenv <tag> <nproc> <ENV=V ...> -- <额外参数...>
runenv() {
  local tag=$1 np=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  if [ -f "results/runs/${tag}.json" ]; then echo "skip(exists) $tag"; return; fi
  local PK; PK=$(peak); local t0=$SECONDS
  env "${envs[@]}" $TR --nproc_per_node=$np --master_port=29702 scripts/train_opt.py \
    --strategy ddp --seq-len $SEQ --peak-tflops "$PK" --tag "$tag" \
    --out "results/runs/${tag}.json" "$@" > "results/logs/${tag}.log" 2>&1
  echo "  [$((SECONDS-t0))s] ${envs[*]} -> $(grep -E '^\[' "results/logs/${tag}.log" | tail -1)"
  cleanup
}

# ============================== 0 / 分母 ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = peak ]; then
step "0. 环境与 GPU 基线"
nvidia-smi > results/nvidia_smi_before.txt 2>&1
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
$PY -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count(),'p2p',torch.cuda.can_device_access_peer(0,1))" | tee results/env.txt
$PY -m pip list 2>/dev/null >> results/env.txt

step "0a. 本机 GEMM 峰值（MFU 的分母，不沿用 Day08 在另一台机器上的值）"
$PY scripts/gemm_peak.py --out results/gemm_peak.json 2>&1 | tail -12 | tee results/gemm_peak.log
cleanup

step "0b. 本机通信带宽（Day09 是另一台机器，必须重测）"
$TR --nproc_per_node=2 --master_port=29703 scripts/comm_bench.py \
  --out results/comm_bench.json 2>&1 | grep -vE "return func|Warning|^W0918|socket.cpp" \
  | tee results/comm_bench.log | tail -14
cleanup
fi

# ============================== 3.3 噪声地板 ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = noise ]; then
step "1. 噪声地板：同一配置、独立进程重复 10 次（任务书 3.3，最高优先级）"
for i in $(seq 1 10); do
  run "noise_$i" 2 --micro-bsz $MB --grad-accum 1 --steps 10 --warmup 10 --repeats 3
done
$PY scripts/noise_floor.py 2>&1 | tee results/noise_floor.log
fi

# ============================== 3.2 基线 ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = base ]; then
step "2. MFU 基线（单卡 / 双卡 DDP）+ 冷启动曲线"
run base_single_mb2 1 --micro-bsz $MB --grad-accum 1 --steps 10 --warmup 10 --repeats 3 --curve 20
run base_ddp_mb2    2 --micro-bsz $MB --grad-accum 1 --steps 10 --warmup 10 --repeats 3 --curve 20
fi

# ============================== 3.5 通信侧 ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = accum ]; then
step "3. 梯度累积（预期收益最大的一项）"
run accum_k1 2 --micro-bsz $MB --grad-accum 1 --steps 10 --warmup 10 --repeats 3
run accum_k2 2 --micro-bsz $MB --grad-accum 2 --steps 8  --warmup 6  --repeats 3
run accum_k4 2 --micro-bsz $MB --grad-accum 4 --steps 6  --warmup 4  --repeats 3
run accum_k8 2 --micro-bsz $MB --grad-accum 8 --steps 4  --warmup 3  --repeats 3
run accum_k16 2 --micro-bsz $MB --grad-accum 16 --steps 3 --warmup 2 --repeats 3
fi

if [ "$STAGE" = all ] || [ "$STAGE" = bucket ]; then
step "4. bucket_cap_mb 扫描"
for B in 1 5 25 100 200; do
  run "bucket_${B}mb" 2 --micro-bsz $MB --grad-accum 1 --bucket-cap-mb $B \
      --steps 10 --warmup 10 --repeats 3
done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = ddpflag ]; then
step "5. DDP 其它开关"
run ddpflag_nobucketview 2 --micro-bsz $MB --grad-accum 1 --grad-as-bucket-view 0 --steps 10 --warmup 10 --repeats 3
run ddpflag_static       2 --micro-bsz $MB --grad-accum 1 --static-graph 1 --steps 10 --warmup 10 --repeats 3
run ddpflag_findunused   2 --micro-bsz $MB --grad-accum 1 --find-unused 1 --steps 10 --warmup 10 --repeats 3
fi

# ============================== NCCL 变量 ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = nccl ]; then
step "6. NCCL 环境变量（每个单独改，事后与噪声地板比）"
A="--micro-bsz $MB --grad-accum 1 --steps 10 --warmup 10 --repeats 3"
runenv nccl_buffsize8m   2 NCCL_BUFFSIZE=8388608          -- $A
runenv nccl_buffsize1m   2 NCCL_BUFFSIZE=1048576          -- $A
runenv nccl_maxch8       2 NCCL_MAX_NCHANNELS=8           -- $A
runenv nccl_minch8       2 NCCL_MIN_NCHANNELS=8           -- $A
runenv nccl_nthreads512  2 NCCL_NTHREADS=512              -- $A
runenv nccl_algo_ring    2 NCCL_ALGO=Ring                 -- $A
runenv nccl_proto_simple 2 NCCL_PROTO=Simple              -- $A
runenv nccl_proto_ll128  2 NCCL_PROTO=LL128               -- $A
runenv nccl_socknthr4    2 NCCL_SOCKET_NTHREADS=4         -- $A
runenv nccl_shm_disable  2 NCCL_SHM_DISABLE=1             -- $A
fi

# ============================== 单卡侧 ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = single ]; then
step "7. 单卡侧改动（与通信侧分开计量）"
A="--grad-accum 1 --steps 10 --warmup 10 --repeats 3"
run sc_mb4        2 --micro-bsz 4 $A
run sc_mb6        2 --micro-bsz 6 $A
run sc_fusedadam  2 --micro-bsz $MB --fused-adam 1 $A
run sc_tf32on     2 --micro-bsz $MB --tf32 1 $A
run sc_tf32off    2 --micro-bsz $MB --tf32 0 $A
run sc_setgradzero 2 --micro-bsz $MB --set-to-none 0 $A
# compile 首次编译很慢，warmup 给足（任务书 4.6）
run sc_compile    2 --micro-bsz $MB --grad-accum 1 --compile 1 --steps 10 --warmup 15 --repeats 3
fi

# ============================== 3.6 交互作用 ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = inter ]; then
step "8. 交互作用：只加 A / 只加 B / A+B"
run inter_A_accum8      2 --micro-bsz $MB --grad-accum 8 --steps 4 --warmup 3 --repeats 3
run inter_B_mb6         2 --micro-bsz 6 --grad-accum 1 --steps 10 --warmup 10 --repeats 3
run inter_AB_mb6_accum8 2 --micro-bsz 6 --grad-accum 8 --steps 3 --warmup 2 --repeats 3
step "8b. 累加到最优配置"
run best_stack 2 --micro-bsz 6 --grad-accum 8 --fused-adam 1 --steps 3 --warmup 2 --repeats 3
fi

# ============================== 3.4 overlap ==============================
if [ "$STAGE" = all ] || [ "$STAGE" = prof ]; then
step "9. overlap 观测（只定位不计时）"
for T in base_ddp_mb2 accum_k8; do
  [ -f "results/runs/${T}.json" ] || continue
  GA=1; [ "$T" = accum_k8 ] && GA=8
  $TR --nproc_per_node=2 --master_port=29704 scripts/profile_opt.py \
    --micro-bsz $MB --grad-accum $GA --seq-len $SEQ --warmup 8 --active 3 \
    --clean-step-json "results/runs/${T}.json" --tag "prof_${T}" \
    --out "results/profiles/prof_${T}.json" 2>&1 \
    | grep -E "^===|^  [a-z]|重叠|空泡|kernel 自|bucket" | tail -14
  cleanup
done
fi

step "10. 汇总"
$PY scripts/analyze10.py 2>&1 | tee results/summary.log

step "11. 收尾"
cleanup
nvidia-smi > results/nvidia_smi_after.txt 2>&1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "ALL DONE $(date '+%F %T')"
