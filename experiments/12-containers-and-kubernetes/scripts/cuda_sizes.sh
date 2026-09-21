#!/bin/bash
# Day12 3.2 的 CUDA 两行：先用 registry manifest 拿**压缩**体积（秒级、不落盘），
# 再在后台尝试真拉 runtime 版拿**未压缩**体积。两个口径差 2~3 倍，report 里必须分开写。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
step(){ echo; echo "########## $* ##########"; date "+%F %T"; }

sum_layers () {   # $1 = 镜像引用
  docker manifest inspect "$1" 2>/dev/null \
    | python3 -c "
import json,sys
m=json.load(sys.stdin)
if 'manifests' in m:            # 多架构索引，挑 linux/amd64
    for e in m['manifests']:
        p=e.get('platform',{})
        if p.get('architecture')=='amd64' and p.get('os')=='linux':
            print('DIGEST', e['digest']); break
else:
    tot=sum(l['size'] for l in m.get('layers',[]))
    print('COMPRESSED_BYTES', tot)
"
}

step "1. CUDA 基座的压缩体积（registry 口径）"
for tag in nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia/cuda:12.4.1-devel-ubuntu22.04 python:3.12-slim; do
  out=$(sum_layers "$tag")
  if [[ "$out" == DIGEST* ]]; then
    d=${out#DIGEST }
    out=$(sum_layers "${tag%%:*}@$d")
  fi
  b=${out#COMPRESSED_BYTES }
  if [[ "$b" =~ ^[0-9]+$ ]]; then
    printf "%-46s 压缩 %.0f MB\n" "$tag" "$(echo "$b/1048576" | bc -l)"
  else
    printf "%-46s 查询失败: %s\n" "$tag" "$out"
  fi
done | tee $R/cuda_manifest_sizes.txt

step "2. 后台真拉 runtime 版（拿未压缩体积）"
setsid nohup docker pull nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  > $R/pull_runtime.log 2>&1 < /dev/null &
disown
sleep 5
tail -2 $R/pull_runtime.log | cut -c1-120
echo CUDA_SIZES_DONE
