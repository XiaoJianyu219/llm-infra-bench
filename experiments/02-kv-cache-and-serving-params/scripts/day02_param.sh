#!/bin/bash
set -u
MODEL=/root/autodl-tmp/models/Qwen3-VL-8B-Instruct
OUT=/root/autodl-tmp/bench/day02
mkdir -p "$OUT"
COMMON="--backend openai-chat --endpoint /v1/chat/completions --model $MODEL \
        --dataset-name random-mm --random-input-len 512 --random-output-len 128 --seed 42"

run_cfg () {
  local name=$1; shift
  echo "########## $name : $* ##########"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 6
  vllm serve "$MODEL" --port 8000 "$@" > "$OUT/serve_${name}.log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s -o /dev/null http://localhost:8000/health && break || sleep 5
  done
  if ! curl -s -o /dev/null http://localhost:8000/health; then
    echo "!! $name 启动失败，跳过"; return
  fi
  grep -E "Available KV cache|GPU KV cache size|Maximum concurrency" "$OUT/serve_${name}.log"

  vllm bench serve $COMMON --num-prompts 20 --max-concurrency 4 >/dev/null 2>&1
  for conc in 16 64; do
    for rep in 1 2 3; do
      vllm bench serve $COMMON --num-prompts 150 --max-concurrency $conc \
        --save-result --result-dir "$OUT" \
        --result-filename "bench_${name}_c${conc}_r${rep}.json" \
        > "$OUT/bench_${name}_c${conc}_r${rep}.txt" 2>&1
    done
  done
}

BASE="--max-model-len 8192 --gpu-memory-utilization 0.94 --limit-mm-per-prompt {\"image\":1,\"video\":0}"

run_cfg base      --max-model-len 8192 --gpu-memory-utilization 0.94 --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 16
run_cfg seqs64    --max-model-len 8192 --gpu-memory-utilization 0.94 --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 64
run_cfg seqs256   --max-model-len 8192 --gpu-memory-utilization 0.94 --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 256
run_cfg util90    --max-model-len 8192 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 64
run_cfg util96    --max-model-len 8192 --gpu-memory-utilization 0.96 --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 64
run_cfg len4096   --max-model-len 4096 --gpu-memory-utilization 0.94 --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 64
run_cfg kvmem381  --max-model-len 8192 --kv-cache-memory=4087021056 --limit-mm-per-prompt '{"image":1,"video":0}' --max-num-seqs 64

pkill -f "vllm serve" 2>/dev/null || true
echo ">>> 全部完成"
