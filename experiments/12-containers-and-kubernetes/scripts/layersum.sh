#!/bin/bash
cd /mnt/d/llm-infra-day12
for i in python:3.12-slim sod:slim sod:B sod:C sod-onnx-cpu:latest; do
  printf "%-24s images=%-8s layers_sum=" "$i" "$(docker images --format '{{.Size}}' "$i" | head -1)"
  docker history --format '{{.Size}}' "$i" | awk '
    /GB/{gsub("GB","");t+=$1*1000}
    /MB/{gsub("MB","");t+=$1}
    /kB/{gsub("kB","");t+=$1/1000}
    END{printf "%.1f MB\n", t}'
done
