#!/bin/bash
# Day12 一键：建集群 -> load 镜像 -> 部署。在 WSL2 里跑。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
CLUSTER=${CLUSTER:-day12}
IMG=${IMG:-sod-onnx-cpu:latest}
M=https://mirrors.aliyun.com/pypi/simple/
step(){ echo; echo "########## $* ##########"; date "+%F %T"; }

step "0. 停掉抢带宽的后台拉取"
pkill -f "docker pull nvidia" 2>/dev/null; echo ok

step "1. 版本"
docker version --format "docker {{.Server.Version}}" 2>&1 | tail -1
kind version; kubectl version --client 2>&1 | head -1

step "2. 镜像（已构建则跳过）"
if docker image inspect "$IMG" >/dev/null 2>&1; then
  docker images "$IMG" --format "已存在 {{.Repository}}:{{.Tag}} {{.Size}}"
else
  docker build -f Dockerfile -t "$IMG" --build-arg PIP_INDEX_URL=$M .
fi

step "3. 建 kind 集群（extraMounts 把 models 映射成节点的 /models）"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "集群 $CLUSTER 已存在，跳过"
else
  kind create cluster --name "$CLUSTER" --config kind-config.yaml 2>&1 | tail -12
fi
kubectl cluster-info --context "kind-$CLUSTER" 2>&1 | head -2
echo "--- 节点上 /models 挂进来了吗（任务书 4.4 的坑）---"
docker exec "${CLUSTER}-control-plane" ls -la /models 2>&1 | head -5

step "4. load 镜像进 kind（任务书 4.2：不做会一直 ImagePullBackOff）"
kind load docker-image "$IMG" --name "$CLUSTER" 2>&1 | tail -3

step "5. 部署"
kubectl apply -f manifests/configmap.yaml
kubectl apply -f manifests/deployment.yaml
kubectl apply -f manifests/service.yaml
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -3

step "6. 自检"
kubectl get pods -o wide 2>&1
kubectl get svc 2>&1
echo "--- NodePort 直连 ---"
curl -s --max-time 10 http://localhost:30080/readyz; echo
echo "--- 三种探针分别打到哪 ---"
kubectl get deploy sod -o jsonpath='startup={.spec.template.spec.containers[0].startupProbe.httpGet.path} readiness={.spec.template.spec.containers[0].readinessProbe.httpGet.path} liveness={.spec.template.spec.containers[0].livenessProbe.httpGet.path}{"\n"}'
echo "UP_DONE $(date "+%F %T")"
