#!/bin/bash
set -eu
MODEL=/root/autodl-tmp/models/Qwen3-VL-8B-Instruct
OUT=/root/autodl-tmp/bench/day01_4090D_4090D
[ -d "$MODEL" ] || { echo "模型目录不存在: $MODEL"; exit 1; }
mkdir -p "$OUT"
COMMON="--backend openai-chat --endpoint /v1/chat/completions --model $MODEL \
        --dataset-name random-mm --random-input-len 512 --random-output-len 128"

echo ">>> warmup (结果丢弃)"
vllm bench serve $COMMON --num-prompts 20 --max-concurrency 4 --seed 0 >/dev/null 2>&1

for conc in 1 4 16 64; do
  case $conc in
    1) N=30 ;; 4) N=60 ;; 16) N=150 ;; 64) N=300 ;;
  esac
  for rep in 1 2 3; do
    tag="c${conc}_r${rep}"
    echo "=== $tag (num-prompts=$N) ==="
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 \
      > "$OUT/mem_${tag}.log" & SMI=$!
    vllm bench serve $COMMON \
      --num-prompts $N --max-concurrency $conc --seed 42 \
      --save-result --result-dir "$OUT" --result-filename "bench_${tag}.json" \
      > "$OUT/bench_${tag}.txt" 2>&1
    kill $SMI 2>/dev/null || true
    echo "峰值显存 MiB: $(sort -n "$OUT/mem_${tag}.log" | tail -1)"
  done
done
echo ">>> done"
