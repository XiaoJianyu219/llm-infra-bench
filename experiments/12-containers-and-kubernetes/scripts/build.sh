#!/bin/bash
# Day12 3.2：镜像体积对比。每个策略单独 build，记录 docker images 与 docker history。
# 注意：docker images 报的是**未压缩**体积，registry 上的压缩体积约为其 1/2~1/3，
# report 里会写明用的是哪个口径。
set -u
cd /mnt/d/llm-infra-day12
R=results
mkdir -p $R
step(){ echo; echo "########## $* ##########"; date "+%F %T"; }

step "0. 给 docker daemon 配代理（daemon 不读 shell 的 http_proxy）"
mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=http://localhost:15236"
Environment="HTTPS_PROXY=http://localhost:15236"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
EOF
systemctl daemon-reload
systemctl restart docker
sleep 4
systemctl show docker --property=Environment | tr ' ' '\n' | grep -i proxy || echo "(未读到)"

step "1. 试拉一个小镜像，确认 daemon 能出网"
if docker pull --quiet python:3.12-slim; then
  echo "PULL_OK"
  docker images python:3.12-slim --format "{{.Repository}}:{{.Tag}} {{.Size}}"
else
  echo "PULL_FAIL —— 后面的 build 都会失败，先解决网络"
  exit 1
fi

step "2. 构建：策略 3（python:3.12-slim 单阶段，无瘦身技巧）"
docker build -q -f Dockerfile.slim -t sod:slim . && echo built

step "3. 构建：策略 4/5/6（多阶段 + --no-cache-dir + --no-install-recommends + 清 apt + .dockerignore + 非 root）"
docker build -q -f Dockerfile -t sod-onnx-cpu:latest . && echo built

step "4. 体积对比（未压缩口径）"
{
  echo "# docker images（未压缩体积）"
  docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" \
    | grep -E "^(sod|sod-onnx-cpu|python|nvidia/cuda)"
} | tee $R/image_sizes.txt

step "5. docker history：看最大的几层在哪"
for img in sod:slim sod-onnx-cpu:latest; do
  echo "===== $img ====="
  docker history --no-trunc --format "{{.Size}}\t{{.CreatedBy}}" "$img" \
    | sed 's/\t\(.\{0,110\}\).*/\t\1/' | head -15
  echo
done | tee $R/docker_history.txt

step "6. 构建上下文有多大（.dockerignore 的作用）"
{
  echo "--- 目录总大小（含模型与测试图）---"
  du -sh /mnt/d/llm-infra-day12 2>/dev/null
  echo "--- .dockerignore 排除后，真正送进 daemon 的上下文 ---"
  docker build -f Dockerfile -t sod-ctx-probe . 2>&1 | grep -iE "transferring context|load build context" | head -3
} | tee $R/build_context.txt

echo BUILD_DONE $(date "+%F %T")
