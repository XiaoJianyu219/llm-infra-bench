#!/bin/bash
# Day12 第三轮：
#   Part 1 — 3.3 失败场景一：liveness 打到 /readyz 且不配 startupProbe -> CrashLoopBackOff
#   Part 2 — 单副本内存/吞吐随并发的标定，给 requests/limits 一个实测依据
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }
podname(){ kubectl get pod -l app=sod -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }
waitready(){ for i in $(seq 1 40); do
    [ "$(curl -s -o /dev/null -w %{http_code} --max-time 5 http://localhost:30080/readyz)" = 200 ] && return 0
    sleep 3; done; return 1; }

############################ Part 1 ############################
step "P1. 失败场景一：liveness -> /readyz，无 startupProbe，模型加载 90s"
kubectl delete deployment sod-bad1 >/dev/null 2>&1; sleep 3
kubectl apply -f manifests/deployment-bad1.yaml >/dev/null
echo "时刻            状态                 重启  存活"
for i in $(seq 1 30); do
  printf "%s  " "$(date +%H:%M:%S)"
  kubectl get pod -l app=sod-bad1 --no-headers 2>/dev/null | awk '{printf "%-20s %-5s %s\n", $3, $4, $5}'
  sleep 10
done | tee $R/bad1_states.txt
echo "--- 事件 ---"
kubectl describe pod -l app=sod-bad1 2>&1 | sed -n '/Events:/,$p' | tee $R/bad1_events.txt
kubectl get pod -l app=sod-bad1 -o jsonpath='{.items[0].status.containerStatuses[0].lastState}{"\n"}' \
  | tee $R/bad1_laststate.txt
kubectl delete deployment sod-bad1 --grace-period=5 >/dev/null 2>&1

############################ Part 2 ############################
step "P2. 标定：单副本，内存上限放到 3Gi，关掉 HPA，只变并发"
kubectl delete hpa sod >/dev/null 2>&1
sed -e 's#image: sod-onnx-cpu:latest#image: sod-onnx-cpu:sync#' \
    -e 's#memory: "1Gi"#memory: "3Gi"#' manifests/deployment.yaml > /tmp/dep-cal.yaml
grep -nE "image:|memory:" /tmp/dep-cal.yaml
kubectl apply -f /tmp/dep-cal.yaml >/dev/null
kubectl scale deployment/sod --replicas=1 >/dev/null
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -1

echo
echo "并发  成功  失败   QPS    p50(s)   峰值内存(MiB)  常驻内存(MiB)"
for C in 1 2 4 8; do
  # 每轮换一个新 Pod，memory.peak 从零开始
  kubectl delete pod -l app=sod --grace-period=5 --wait=true >/dev/null 2>&1
  waitready || { echo "并发 $C: Pod 起不来"; continue; }
  P=$(podname)
  BASE=$(kubectl exec "$P" -- cat /sys/fs/cgroup/memory.current 2>/dev/null)
  python3 loadtest.py --url http://localhost:30080/predict --img testdata/LD_000016.png \
    --concurrency $C --duration 60 --timeout 180 --out $R/runC_c$C.jsonl > /dev/null 2>&1
  PEAK=$(kubectl exec "$P" -- cat /sys/fs/cgroup/memory.peak 2>/dev/null)
  CUR=$(kubectl exec "$P" -- cat /sys/fs/cgroup/memory.current 2>/dev/null)
  python3 - "$C" "$R/runC_c$C.jsonl" "${PEAK:-0}" "${CUR:-0}" "${BASE:-0}" <<'PY'
import json,sys
c,path,peak,cur,base=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
rs=[json.loads(l) for l in open(path) if l.strip()]
ok=sum(r["ok"] for r in rs); err=sum(r["err"] for r in rs)
ps=sorted(r["p50"] for r in rs if r["p50"])
print("%4s %5d %5d %6.2f %8.2f %14.0f %14.0f" % (
    c, ok, err, ok/float(max(len(rs),1)), (ps[len(ps)//2] if ps else float('nan')),
    peak/1048576.0, cur/1048576.0))
PY
done | tee $R/runC_concurrency.txt

step "P2b. 空载常驻内存（模型加载完、无请求）"
kubectl delete pod -l app=sod --grace-period=5 --wait=true >/dev/null 2>&1
waitready && kubectl exec "$(podname)" -- sh -c \
  'echo idle_current_MiB=$(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))' | tee $R/runC_idle_mem.txt

step "P3. 恢复正常部署（latest 镜像 + 原 limits）"
kubectl apply -f manifests/deployment.yaml >/dev/null
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -1
echo RUNC_DONE $(date "+%F %T")
