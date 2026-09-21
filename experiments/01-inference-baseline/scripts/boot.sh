#!/bin/bash
echo "=== 机器身份 ==="
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
echo "CPU $(nproc) 核"; free -g | head -2
echo
echo "=== 文件检查 ==="
for p in /root/autodl-tmp/venvs/infer \
         /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
         /root/autodl-tmp/scripts \
         /root/autodl-tmp/work/llm-infra-bench; do
  [ -e "$p" ] && echo "OK    $p" || echo "缺失  $p"
done
echo
df -h /root /root/autodl-tmp | tail -3
