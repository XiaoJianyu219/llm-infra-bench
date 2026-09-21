#!/bin/bash
# 每 2 秒记一次 HPA / Pod / 事件，带墙钟时间戳。3.5 的时间线就靠它对齐。
set -u
OUT=${1:-results/k8s_timeline.txt}
DUR=${2:-420}
mkdir -p "$(dirname "$OUT")"
END=$((SECONDS + DUR))
: > "$OUT"
while [ $SECONDS -lt $END ]; do
  {
    echo "=== $(date '+%H:%M:%S') ==="
    kubectl get hpa sod --no-headers 2>/dev/null
    kubectl get pods -l app=sod -o wide --no-headers 2>/dev/null
  } >> "$OUT"
  sleep 2
done
echo "--- 事件（按时间排序）---" >> "$OUT"
kubectl get events --sort-by=.lastTimestamp >> "$OUT" 2>&1
kubectl describe hpa sod >> "$OUT" 2>&1
