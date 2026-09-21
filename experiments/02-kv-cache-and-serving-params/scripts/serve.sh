#!/bin/bash
set -u
MODEL=/root/autodl-tmp/models/Qwen3-VL-8B-Instruct
LOG=/root/autodl-tmp/work/serve_$(date +%m%d_%H%M).log
[ -d "$MODEL" ] || { echo "模型目录不存在: $MODEL"; exit 1; }

source /root/autodl-tmp/venvs/infer/bin/activate
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
mkdir -p "$(dirname "$LOG")"
ln -sfn "$LOG" /root/autodl-tmp/work/serve_latest.log

vllm serve "$MODEL" \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.94 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --max-num-seqs 16 \
  2>&1 | tee "$LOG"
