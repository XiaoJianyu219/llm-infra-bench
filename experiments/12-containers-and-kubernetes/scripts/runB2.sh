#!/bin/bash
# Day12 对照组 B（重做）：唯一变量 = /predict 从 async def 改为同步 def。
# 依赖层原封不动（FROM sod-onnx-cpu:latest 只叠一层 app.py），不触网。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
CL=day12
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }
loadimg(){ docker save "$1" | docker exec --privileged -i ${CL}-control-plane \
             ctr --namespace=k8s.io images import --snapshotter=overlayfs - 2>&1 | tail -1; }

step "1. 叠一层 app.py"
docker build -t sod-onnx-cpu:sync -f Dockerfile.patch . 2>&1 | tail -3
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep sod-onnx-cpu

step "2. 推进节点并确认"
loadimg sod-onnx-cpu:sync
docker exec ${CL}-control-plane crictl images 2>/dev/null | grep sod

step "3. 清干净：删 HPA、删 Deployment，重新以 sync 镜像部署 1 副本"
kubectl delete hpa sod >/dev/null 2>&1
kubectl delete deployment sod --wait=true >/dev/null 2>&1
sleep 5
sed 's#image: sod-onnx-cpu:latest#image: sod-onnx-cpu:sync#' manifests/deployment.yaml > /tmp/dep-sync.yaml
grep -n "image:" /tmp/dep-sync.yaml
kubectl apply -f /tmp/dep-sync.yaml >/dev/null
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -1
kubectl get pods -l app=sod --no-headers 2>&1
for i in $(seq 1 12); do
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://localhost:30080/readyz)
  echo "readyz -> $c"; [ "$c" = 200 ] && break; sleep 5
done
echo "--- 确认跑的是 sync 镜像 ---"
kubectl get pods -l app=sod -o jsonpath='{.items[0].spec.containers[0].image}{"\n"}'

step "4. 空载基线（无并发，3 次）"
for i in 1 2 3; do
  curl -s --max-time 120 -X POST -F "file=@testdata/LD_000016.png" \
    http://localhost:30080/predict | python3 -c "import sys,json; print(json.load(sys.stdin)['timing_ms'])"
done | tee $R/runB_idle_latency.txt

step "5. 空载时 /healthz 的响应时间（对照：此时事件循环空闲）"
for i in 1 2 3; do curl -s -o /dev/null -w "healthz %{time_total}s\n" http://localhost:30080/healthz; done

step "6. 建 HPA"
kubectl apply -f manifests/hpa.yaml >/dev/null
sleep 30
kubectl get hpa 2>&1 | tee $R/runB_hpa_before.txt

step "7. 同参压测（concurrency 8 / 300s）；期间每 20s 探一次 /healthz 看事件循环是否被堵"
nohup bash watch_k8s.sh $R/runB_k8s_timeline.txt 480 > /dev/null 2>&1 &
WPID=$!
( for i in $(seq 1 15); do
    sleep 20
    echo "$(date +%H:%M:%S) healthz $(curl -s -o /dev/null -w '%{http_code} %{time_total}s' --max-time 10 http://localhost:30080/healthz)"
  done > $R/runB_healthz_under_load.txt 2>&1 ) &
HPID=$!
sleep 2
python3 loadtest.py --url http://localhost:30080/predict --img testdata/LD_000016.png \
  --concurrency 8 --duration 300 --timeout 180 --out $R/runB_loadtest.jsonl 2>&1 | tail -8

step "8. 停压后观察 120s"
sleep 120
kubectl get hpa 2>&1 | tee $R/runB_hpa_after.txt
echo "B 组重启次数："
kubectl get pods -l app=sod --no-headers 2>&1 | awk '{print $1, $2, $3, "restarts=" $4}'
kubectl get events --sort-by=.lastTimestamp > $R/runB_events.txt 2>&1
echo "B 组 liveness 失败事件数: $(grep -c 'failed liveness probe' $R/runB_events.txt)"
echo "--- 压测期间 /healthz ---"; cat $R/runB_healthz_under_load.txt
kubectl describe pod -l app=sod > $R/runB_describe_pods.txt 2>&1
wait $WPID $HPID 2>/dev/null
echo RUNB2_DONE $(date "+%F %T")
