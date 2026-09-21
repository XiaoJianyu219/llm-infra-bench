#!/bin/bash
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day10
PY=/root/autodl-tmp/venvs/det/bin/python
TR=/root/autodl-tmp/venvs/det/bin/torchrun
PK=$($PY -c "import json;print(json.load(open('results/gemm_peak.json'))['peak_bf16_tflops'])")
cleanup(){ pkill -f train_opt.py 2>/dev/null; sleep 3; }
# go <tag> <ENV...> -- <args...>
go(){ local tag=$1; shift; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ -f "results/runs/${tag}.json" ] && { echo "skip $tag"; return; }
  local t0=$SECONDS
  env "${envs[@]}" $TR --nproc_per_node=2 --master_port=29721 scripts/train_opt.py \
    --strategy ddp --seq-len 512 --peak-tflops "$PK" --tag "$tag" \
    --out "results/runs/${tag}.json" "$@" > "results/logs/${tag}.log" 2>&1
  echo "[$((SECONDS-t0))s] ${envs[*]:-无env} -> $(grep -E '^\[' results/logs/${tag}.log|tail -1)"
  cleanup; }

echo "### 交互作用 v2（B 换成 mb4，mb6 在本机 OOM）"
go inter2_A_accum8       -- --micro-bsz 2 --grad-accum 8 --steps 4 --warmup 3 --repeats 3
go inter2_B_mb4          -- --micro-bsz 4 --grad-accum 1 --steps 10 --warmup 10 --repeats 3
go inter2_AB_mb4_accum8  -- --micro-bsz 4 --grad-accum 8 --steps 3 --warmup 2 --repeats 3

echo "### 第二组交互：通信侧两项叠加（k=8 之后 MIN_NCHANNELS 还有用吗）"
go inter3_minch8_only NCCL_MIN_NCHANNELS=8 -- --micro-bsz 2 --grad-accum 1 --steps 10 --warmup 10 --repeats 3
go inter3_accum8_minch8 NCCL_MIN_NCHANNELS=8 -- --micro-bsz 2 --grad-accum 8 --steps 4 --warmup 3 --repeats 3

echo "### 逐项累加到最优（每加一项记一次）"
go stack1_mb4                 -- --micro-bsz 4 --grad-accum 1 --steps 10 --warmup 10 --repeats 3
go stack2_mb4_k8              -- --micro-bsz 4 --grad-accum 8 --steps 3 --warmup 2 --repeats 3
go stack3_mb4_k8_minch NCCL_MIN_NCHANNELS=8 -- --micro-bsz 4 --grad-accum 8 --steps 3 --warmup 2 --repeats 3
go stack4_mb4_k8_minch_fused NCCL_MIN_NCHANNELS=8 -- --micro-bsz 4 --grad-accum 8 --fused-adam 1 --steps 3 --warmup 2 --repeats 3
go stack5_full NCCL_MIN_NCHANNELS=8 NCCL_NTHREADS=512 -- --micro-bsz 4 --grad-accum 8 --fused-adam 1 --compile 1 --steps 3 --warmup 4 --repeats 3

echo "### NCCL channel 数细扫（MIN 的机制确认）"
for C in 2 4 16; do
  go "minch_${C}" NCCL_MIN_NCHANNELS=$C -- --micro-bsz 2 --grad-accum 1 --steps 10 --warmup 10 --repeats 3
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
