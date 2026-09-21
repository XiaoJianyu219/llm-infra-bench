#!/bin/bash
# Day12 第四轮：取一条干净的 3.5 扩容/缩容时间线。
# 与 A 组的区别只有两点，都是为了让"扩容"这件事本身可观测：
#   1) 并发降到 2 —— 单副本不会被压到探针饿死，时间线不被重启打断
#   2) HPA 在 CPU 回落到基线之后才创建 —— 保证 T0 时刻确实是 1 副本
# 镜像用 latest（async 版），与 A 组同一个二进制。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }
waitready(){ for i in $(seq 1 40); do
    [ "$(curl -s -o /dev/null -w %{http_code} --max-time 5 http://localhost:30080/readyz)" = 200 ] && return 0
    sleep 3; done; return 1; }

step "1. 回到 1 副本、latest 镜像、无 HPA"
kubectl delete hpa sod >/dev/null 2>&1
kubectl apply -f manifests/deployment.yaml >/dev/null
kubectl scale deployment/sod --replicas=1 >/dev/null
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -1
waitready && echo "readyz OK"
kubectl get pods -l app=sod --no-headers 2>&1

step "2. 等 CPU 回落到基线（要求连续 3 次 top 都 < 100m）"
n=0
for i in $(seq 1 30); do
  M=$(kubectl top pod -l app=sod --no-headers 2>/dev/null | awk '{gsub("m","",$2); s+=$2} END{print s+0}')
  echo "$(date +%H:%M:%S) cpu=${M}m"
  if [ "${M:-999}" -lt 100 ]; then n=$((n+1)); else n=0; fi
  [ $n -ge 3 ] && break
  sleep 10
done

step "3. 建 HPA，确认 T0 前是 1 副本且利用率低"
kubectl apply -f manifests/hpa.yaml >/dev/null
sleep 45
kubectl get hpa 2>&1 | tee $R/runD_hpa_before.txt
kubectl get pods -l app=sod --no-headers 2>&1

step "4. 加压：并发 2 / 300s，同时采样 k8s 状态"
nohup bash watch_k8s.sh $R/runD_k8s_timeline.txt 540 > /dev/null 2>&1 &
WPID=$!
( for i in $(seq 1 20); do
    sleep 20
    echo "$(date +%H:%M:%S) healthz $(curl -s -o /dev/null -w '%{http_code} %{time_total}s' --max-time 10 http://localhost:30080/healthz)"
  done > $R/runD_healthz_under_load.txt 2>&1 ) &
HPID=$!
sleep 2
python3 loadtest.py --url http://localhost:30080/predict --img testdata/LD_000016.png \
  --concurrency 2 --duration 300 --timeout 180 --out $R/runD_loadtest.jsonl 2>&1 | tail -6

step "5. 停压后观察缩容 240s（scaleDown 稳定窗口 300s，预期观察不到缩容）"
sleep 240
kubectl get hpa 2>&1 | tee $R/runD_hpa_after.txt
kubectl get pods -l app=sod --no-headers 2>&1 | awk '{print $1, $2, $3, "restarts=" $4}'
kubectl get events --sort-by=.lastTimestamp > $R/runD_events.txt 2>&1
echo "liveness 失败事件数: $(grep -c 'failed liveness probe' $R/runD_events.txt)"
echo "--- 压测期间 /healthz ---"; cat $R/runD_healthz_under_load.txt
wait $WPID $HPID 2>/dev/null
echo RUND_DONE $(date "+%F %T")
