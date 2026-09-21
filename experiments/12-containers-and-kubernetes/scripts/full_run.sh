#!/bin/bash
# Day12 3.4 + 3.5：重建集群 -> 部署 -> metrics-server -> HPA -> 压测并记录扩容时间线
# 全程离线（镜像已在本地），一次跑完，避免中途 WSL/集群状态丢失。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
CL=day12
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }
apiok(){ kubectl get --raw /readyz >/dev/null 2>&1; }

step "1. 重建集群"
kind delete cluster --name $CL >/dev/null 2>&1
kind create cluster --name $CL --config kind-config.yaml 2>&1 | tail -4
for i in $(seq 1 30); do apiok && break; sleep 5; done
apiok || { echo "API server 起不来"; exit 1; }
kubectl get nodes 2>&1 | head -2
docker exec ${CL}-control-plane ls /models 2>&1 | head -3

step "2. load 镜像（离线）"
kind load docker-image sod-onnx-cpu:latest --name $CL 2>&1 | tail -1
kind load docker-image registry.k8s.io/metrics-server/metrics-server:v0.9.0 --name $CL 2>&1 | tail -1

step "3. 部署业务"
kubectl apply -f manifests/configmap.yaml >/dev/null
kubectl apply -f manifests/deployment.yaml >/dev/null
kubectl apply -f manifests/service.yaml >/dev/null
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -1
kubectl get pods -o wide 2>&1 | head -3

step "4. 服务自检"
for i in $(seq 1 12); do
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://localhost:30080/readyz)
  echo "readyz -> $c"; [ "$c" = 200 ] && break; sleep 5
done
curl -s --max-time 10 http://localhost:30080/readyz | tee $R/readyz.json; echo
echo "--- 打一次 /predict 确认端到端 ---"
curl -s --max-time 120 -X POST -F "file=@testdata/LD_000016.png" http://localhost:30080/predict \
  | head -c 400 | tee $R/predict_sample.json; echo

step "5. metrics-server（本地 yaml + 本地镜像 + kubelet-insecure-tls）"
kubectl apply -f metrics-server.yaml >/dev/null 2>&1
kubectl -n kube-system patch deployment metrics-server --type=json -p='[
 {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"},
 {"op":"replace","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"IfNotPresent"}]' 2>&1 | tail -1
kubectl -n kube-system rollout status deployment/metrics-server --timeout=300s 2>&1 | tail -1
for i in $(seq 1 30); do kubectl top pods >/dev/null 2>&1 && break; sleep 10; done
kubectl top pods 2>&1 | head -3

step "6. HPA"
kubectl apply -f manifests/hpa.yaml >/dev/null
sleep 30
kubectl get hpa 2>&1 | tee $R/hpa_before.txt

step "7. 压测 + 时间线（后台记 k8s 状态，前台加压 300s）"
nohup bash watch_k8s.sh $R/k8s_timeline.txt 480 > /dev/null 2>&1 &
WPID=$!
sleep 2
python3 loadtest.py --url http://localhost:30080/predict --img testdata/LD_000016.png \
  --concurrency 8 --duration 300 --timeout 180 --out $R/loadtest.jsonl 2>&1 | tail -40

step "8. 停止加压后继续观察缩容 120s"
sleep 120
kubectl get hpa 2>&1 | tee $R/hpa_after.txt
kubectl get pods -l app=sod 2>&1 | head -8
wait $WPID 2>/dev/null
echo FULL_RUN_DONE $(date "+%F %T")
