#!/bin/bash
set -u
MODEL=/root/autodl-tmp/models/Qwen3-VL-8B-Instruct
OUT=/root/autodl-tmp/bench/day02/metrics
mkdir -p "$OUT"
COMMON="--backend openai-chat --endpoint /v1/chat/completions --model $MODEL \
        --dataset-name random-mm --random-input-len 512 --random-output-len 128 \
        --num-prompts 150 --max-concurrency 64 --seed 42"

run () {
  local name=$1; shift
  echo "########## $name ##########"
  pkill -f "vllm serve" 2>/dev/null || true; sleep 6
  vllm serve "$MODEL" --port 8000 "$@" > "$OUT/serve_${name}.log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s -o /dev/null http://localhost:8000/health && break || sleep 5
  done
  curl -s -o /dev/null http://localhost:8000/health || { echo "!! $name 启动失败"; return; }
  grep -E "GPU KV cache size" "$OUT/serve_${name}.log"

  ( while true; do
      ts=$(date +%s)
      curl -s http://localhost:8000/metrics \
        | grep -E "^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc)" \
        | awk -v t="$ts" '{print t, $1, $2}'
      sleep 1
    done ) > "$OUT/metrics_${name}.log" &
  local M=$!

  vllm bench serve $COMMON > "$OUT/bench_${name}.txt" 2>&1
  kill $M 2>/dev/null || true

  echo "--- $name 峰值 ---"
  awk '/gpu_cache_usage_perc/{if($3+0>c)c=$3+0}
       /num_requests_running/{if($3+0>r)r=$3+0}
       /num_requests_waiting/{if($3+0>w)w=$3+0}
       END{printf "cache_usage_max=%.3f  running_max=%.0f  waiting_max=%.0f\n",c,r,w}' \
      "$OUT/metrics_${name}.log"
}

run util90   --max-model-len 8192 --gpu-memory-utilization 0.90 \
             --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 64
run kvmem381 --max-model-len 8192 --kv-cache-memory=4087021056 \
             --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 64

pkill -f "vllm serve" 2>/dev/null || true
echo ">>> 完成"
