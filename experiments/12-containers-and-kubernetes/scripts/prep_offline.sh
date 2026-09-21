#!/bin/bash
# Day12：趁镜像网络还能用代理，先把后面要用的镜像全部拉到本地，
# 然后就可以切回 NAT 网络（更稳），全程离线 kind load。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }

step "1. dockerd 稳定性证据"
systemctl show docker --property=ActiveEnterTimestamp,NRestarts 2>&1
echo "uptime: $(uptime -p)"

step "2. 拉 metrics-server 镜像（HPA 的前提）"
MS_VER=$(curl -fsSL --max-time 30 https://api.github.com/repos/kubernetes-sigs/metrics-server/releases/latest 2>/dev/null | grep -m1 tag_name | sed 's/.*"v\?\([0-9.]*\)".*/\1/')
MS_VER=${MS_VER:-0.8.0}
echo "metrics-server 版本: v$MS_VER"
docker pull registry.k8s.io/metrics-server/metrics-server:v$MS_VER 2>&1 | tail -2
docker images | grep metrics-server

step "3. 存一份 components.yaml 到本地（切 NAT 后就下不动了）"
curl -fsSL --max-time 60 -o metrics-server.yaml \
  https://github.com/kubernetes-sigs/metrics-server/releases/download/v$MS_VER/components.yaml \
  && wc -l metrics-server.yaml \
  || echo "下载失败"

step "4. 确认 kind 节点镜像也在本地"
docker images | grep kindest

step "5. 当前本地镜像清单"
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | tee $R/local_images.txt
echo PREP_DONE $(date "+%F %T")
